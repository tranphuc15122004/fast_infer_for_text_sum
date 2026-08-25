# Baseline Repository Guide
## Efficient Inference for Long-Context Text Summarization

> File này dùng để quản lý các repository baseline trong `externals/`, phân loại chúng theo hướng nghiên cứu, ghi chú vai trò của từng baseline và hỗ trợ thiết kế thực nghiệm công bằng cho đề tài **quality-preserving + production-efficient long-context text summarization**.

---

## 1. Research scope

Đề tài không bị giới hạn vào speculative decoding. Các baseline được tổ chức quanh toàn bộ inference pipeline:

```text
Document
   │
   ├── Semantic / Context Reduction
   │      ├── LLMLingua
   │      ├── GemFilter
   │      ├── HiGOE
   │      └── Speculative Prefill
   │
   ├── Prefill / Attention Acceleration
   │      └── MInference
   │
   ├── Internal Token / KV Optimization
   │      ├── FastKV
   │      └── RocketKV
   │
   ├── Generic Decode Acceleration
   │      ├── EAGLE
   │      └── DFlash
   │
   └── Long-Context Speculative Decoding
          ├── MagicDec
          ├── LongSpec
          └── SpecExtend
```

Ngoài ra:

```text
SpecForge = infrastructure / training framework
Dense AR + FlashAttention/vLLM/SGLang = production control baseline
```

**SpecForge không nên được tính như một algorithmic baseline độc lập.**

---

# 2. Baseline taxonomy

## A. Semantic / Content-Aware Input Reduction

Các phương pháp trong nhóm này giảm lượng thông tin mà target LLM cần xử lý. Đây là nhóm đặc biệt quan trọng đối với summarization vì thường `L_input >> L_output`, nên prefill và source KV có thể chiếm tỷ trọng lớn trong chi phí inference.

### A1. `LLMLingua`

**Category:** Prompt / context compression  
**Granularity:** token / phrase / context  
**Core idea:** nén input trước khi target model chạy, nhằm giữ lại phần thông tin quan trọng với budget nhỏ hơn.

**Role trong benchmark**
- Đại diện cho **external semantic prompt compression**.
- Model/engine-agnostic hơn các phương pháp sửa bên trong attention/KV.
- Kiểm tra câu hỏi: *Có cần thay đổi inference engine không, hay semantic preprocessing đã đủ?*

**Quality semantics:** approximate. Target model không còn thấy full source.  
**Production strengths:** preprocessing middleware, dễ ghép với vLLM/SGLang/API, giảm prefill length và KV footprint.  
**Risks:** rare-but-critical facts, entity/number/date, minority-topic coverage.  
**Priority:** `P0`.

---

### A2. `GemFilter`

**Category:** Early-layer semantic token filtering  
**Granularity:** token  
**Core idea:** dùng early/middle layers của chính target LLM để nhận diện token quan trọng, sau đó chỉ giữ subset token cho các layer còn lại.

**Role**
- Đại diện cho **internal self-filtering**.
- Direct comparison với `speculative_prefill`.

```text
Speculative Prefill : external small model -> token importance
GemFilter           : target early layers  -> token importance
```

**Quality semantics:** approximate.  
**Training:** training-free / no full retraining.  
**Main research question:** *Target model có thể tự nhận diện source evidence tốt hơn một external selector hay không?*  
**Priority:** `P0`.

---

### A3. `HiGOE`

**Category:** Evidence / proposition-aware semantic selection  
**Granularity:** proposition / evidence / graph  
**Core idea:** xây cấu trúc evidence thay vì chỉ chọn token theo attention hoặc top-k salience.

**Role**
- Đại diện cho **summarization-aware semantic selection** ở mức cao hơn token.
- Hữu ích nếu phương pháp mới claim factuality-aware, evidence-preserving hoặc coverage-aware selection.

**Quality semantics:** approximate.  
**Production concern:** preprocessing phức tạp hơn token filtering.  
**Main question:** *Chọn semantic evidence có giữ quality/factuality tốt hơn token-level compression tại cùng input budget hay không?*  
**Priority:** `P1`.

---

### A4. `speculative_prefill`

**Category:** Token-selective prefill  
**Granularity:** token / chunk  
**Core idea:** lightweight model dự đoán important prompt tokens; target model chỉ prefill selected tokens.

**Primary phase:** Prefill / TTFT  
**Training:** training-free  
**Quality semantics:** approximate  
**Production relevance:** rất cao

**Role**
- Đại diện cho **aggressive token-level input reduction**.
- Strong production-oriented baseline cho TTFT, QPS và high-load serving.

**Priority:** `P0`.

---

## B. Prefill / Attention Acceleration

### B1. `MInference`

**Category:** Dynamic sparse attention  
**Phase:** Prefill  
**Granularity:** attention entries / blocks

**Core idea:** không xóa semantic input; chỉ giảm số attention interactions phải tính.

Patterns:
- A-shape
- Vertical-Slash
- Block-Sparse

**Role**
- Đại diện cho **sparse-attention acceleration**.
- Đặc biệt phù hợp extreme long context.

**Quality semantics:** near-dense / approximate sparse computation  
**Training:** no additional fine-tuning  
**Kernel dependence:** cao

**Best regime:** `64K -> 128K -> 1M`.  
**Không nên kỳ vọng:** luôn tốt nhất ở 4K–16K hoặc giải quyết MLP bottleneck ở batch lớn.  
**Priority:** `P1`.

---

## C. Internal Token Propagation / KV Optimization

### C1. `FastKV`

**Category:** Joint prefill + decode optimization

**Core idea:** tách token budget dùng cho later-layer prefill propagation và KV budget dùng cho decoding.

```text
tokens needed for prompt understanding
!=
tokens needed repeatedly during decoding
```

**Role**
- Đại diện cho **joint prefill + KV optimization**.
- Rất quan trọng nếu mục tiêu là E2E inference.

**Quality semantics:** approximate / near-dense  
**Production relevance:** cao

**Metrics cần chú ý:** TTFT, TPOT, E2E, KV memory, max batch/concurrency.  
**Priority:** `P0`.

---

### C2. `RocketKV`

**Category:** Standalone KV-cache compression / sparse decode attention  
**Phase:** Decode  
**Core idea:** giảm KV workload bằng coarse + fine-grained selection.

**Role**
- Đại diện cho nhánh **KV-cache reduction độc lập với SD**.
- Cần thiết vì `MagicDec` có compressed KV nhưng KV compression chỉ là một phần của speculative architecture.

**Quality semantics:** approximate  
**Production relevance:** rất cao vì giảm memory/request và tăng max concurrency.

**Critical warning:** với summarization phải kiểm tra factuality riêng; ROUGE không đủ.  
**Priority:** `P1`.

---

## D. Generic Speculative / Parallel Decoding

### D1. `EAGLE`

**Category:** Learned speculative decoding  
**Phase:** Decode  
**Core idea:** trained auxiliary drafter predicts future tokens/features; target verifies candidates.

**Role**
- Strong generic learned SD baseline.
- Nếu repo hỗ trợ EAGLE-3 thì ưu tiên EAGLE-3 cho final comparison.

**Quality semantics:** lossless under standard speculative verification  
**Training:** required  
**Production concern:** extra model/head artifact, target compatibility, deployment/maintenance overhead.

**Main question:** *Một strong generic learned drafter có đủ tốt cho summarization hay task-/context-aware methods có lợi rõ ràng hơn?*  
**Priority:** `P0` cho SD branch.

---

### D2. `dflash`

**Category:** Parallel / diffusion-style drafting  
**Phase:** Decode  
**Role:** đại diện cho một architecture-level alternative so với conventional autoregressive drafter.

**Production concerns:** draft overhead, batch behavior, verification cost.  
**Priority:** `P1`.

---

## E. Long-Context Speculative Decoding

### E1. `MagicDec`

**Category:** Long-context self/speculative decoding  
**Core idea:** tận dụng sparse/compressed KV và long-context regime để cải thiện latency–throughput trade-off.

**Strength:** đặc biệt quan tâm large batch / serving.  
**Quality semantics:** lossless speculative framework.  
**Role:** đại diện cho long-context + high-batch + self/target-based drafting.  
**Priority:** `P0`.

---

### E2. `LongSpec`

**Category:** Purpose-built long-context learned speculative decoding

**Core components**
- constant-sized draft KV,
- Anchor-Offset positional training,
- Hybrid Tree Attention.

**Role**
- Strong long-context learned SD baseline.
- Đại diện cho co-design **draft architecture + training + verification kernel**.

**Quality semantics:** lossless SD  
**Training:** dedicated drafter required  
**Production concern:** model-specific artifact / maintenance complexity  
**Priority:** `P0` nếu final direction nằm ở SD.

---

### E3. `SpecExtend`

**Category:** Drop-in long-context speculative decoding enhancement

**Core idea**

```text
Target verification attention
        ↓
Cross-model Retrieval
        ↓
Draft KV keeps relevant source context
```

**Why important**
- Trực tiếp evaluate long-document summarization.
- Training-free enhancement.
- Sát với source-aware long-context speculative summarization.

**Quality semantics:** lossless target verification  
**Production concern:** cần stress-test batch/concurrency ngoài paper setup.  
**Priority:** `P0`.

---

# 3. Infrastructure repository

## `SpecForge`

**Type:** Infrastructure / training framework  
**Không tính như algorithmic baseline.**

Có thể dùng cho:
- training speculative drafters,
- EAGLE-style experiments,
- DFlash-style experiments,
- unified training / evaluation.

Trong paper có thể ghi ở implementation details thay vì main baseline table.

---

# 4. Missing control baseline that does not require a repo

## Dense AR + Production Serving Engine

Đây là baseline bắt buộc nhưng không cần folder riêng trong `externals/`.

Recommended:

```text
Target model
+ vLLM or SGLang
+ FlashAttention / optimized attention backend
+ full KV
+ no input pruning
+ no speculation
```

Tên nên dùng trong bảng: `Dense AR` hoặc `Full-Context AR`.

**Không nên dùng naïve HuggingFace `generate()` làm main denominator.**

---

# 5. Current repository inventory

| Repo | Main branch | Phase | Quality | Training | Production relevance | Priority |
|---|---|---|---|---|---|---|
| `LLMLingua` | Semantic prompt compression | Preprocess / Prefill | Approx. | No/light | High | P0 |
| `GemFilter` | Internal semantic token filtering | Prefill | Approx. | No | High | P0 |
| `HiGOE` | Evidence/proposition selection | Preprocess | Approx. | Method-specific | Medium–High | P1 |
| `speculative_prefill` | Token-selective prefill | Prefill | Approx. | No | Very High | P0 |
| `MInference` | Dynamic sparse attention | Prefill | Near-dense | No | High for extreme context | P1 |
| `FastKV` | Joint token/KV optimization | Prefill + Decode | Approx. | Light/none | Very High | P0 |
| `RocketKV` | KV cache compression | Decode | Approx. | No | Very High | P1 |
| `EAGLE` | Learned speculative decoding | Decode | Lossless SD | Yes | High | P0 |
| `dflash` | Parallel/diffusion drafting | Decode | Depends on verification | Yes | Medium–High | P1 |
| `MagicDec` | Long-context SD / sparse KV | Decode | Lossless SD | No/light | Very High | P0 |
| `LongSpec` | Long-context trained SD | Decode | Lossless SD | Yes | Medium–High | P0 |
| `SpecExtend` | Target-guided long-context SD | Decode | Lossless SD | No | High | P0 |
| `SpecForge` | Infrastructure | — | — | — | Infrastructure | — |

---

# 6. Main evaluation tracks

Không nên đặt toàn bộ repo vào một bảng speedup duy nhất vì semantics khác nhau.

## Track A — Quality-preserving / lossless decoding acceleration

```text
Dense AR
EAGLE-3
MagicDec
LongSpec
SpecExtend
DFlash           # nếu exact target verification được giữ
```

Question:
> How much faster can we generate while preserving the target distribution / output behavior?

## Track B — Approximate input/prefill/KV acceleration

```text
Dense AR
LLMLingua
GemFilter
Speculative Prefill
MInference
FastKV
RocketKV
HiGOE
```

Question:
> What quality–latency–memory trade-off is achievable?

Main evaluation phải Pareto-oriented, không chỉ raw speedup.

---

# 7. Semantic-selection sub-benchmark

Đây nên là một experiment riêng vì sát với summarization.

## Methods

```text
Full Context
Lead-K
TextRank / LexRank
Embedding + MMR
LLMLingua
GemFilter
Speculative Prefill
HiGOE
```

Các baseline `Lead-K`, `TextRank/LexRank`, `Embedding+MMR` không cần repo riêng.

## Retention budgets

```text
100%
75%
50%
30%
20%
10%
```

## Efficiency metrics
- selector overhead
- retained tokens
- TTFT
- E2E latency
- peak KV memory
- throughput / QPS
- P50 / P95 / P99

## Summary quality
- ROUGE-1/2/L
- semantic metric
- factuality metric
- entity recall
- number/date preservation
- source claim/evidence coverage

**Important**

```text
T_E2E = T_selector + T_target
```

Không report target latency mà bỏ selector overhead.

---

# 8. Production-oriented benchmark protocol

## Input length buckets

```text
4K
8K
16K
32K
64K
128K
```

Chỉ dùng bucket mà dataset/model thực sự hỗ trợ.

## Offline batch

```text
B = 1, 4, 8, 16, 32, 64
```

## Online serving

Nên dùng continuous batching và tăng arrival rate.

Measure:
- QPS
- P50 / P95 / P99 latency
- TTFT
- TPOT
- E2E latency
- GPU utilization
- peak GPU memory
- KV memory/request

Production metric hữu ích:

```text
max QPS subject to P95(E2E) < SLO
```

---

# 9. Recommended datasets

## Core
- **CNN/DailyMail** — moderate extractiveness, source overlap cao.
- **XSum** — highly abstractive, stress-test aggressive selection/SD.
- **GovReport** — long reports, core long-document benchmark.
- **Multi-News** — multi-document, coverage + redundancy.
- **QMSum** — long meeting summarization, sparse salient content.

## Extended
- BookSum
- arXiv
- PubMed
- PG-19 khi cần long-context system stress.

---

# 10. Suggested reproduction order

## Stage 0 — Dense control

```text
Dense AR + vLLM/SGLang
```

## Stage 1 — Input/prefill map

```text
LLMLingua
GemFilter
Speculative Prefill
MInference
FastKV
```

Goal: xác định acceleration opportunity chính nằm ở đâu.

## Stage 2 — KV map

```text
RocketKV
FastKV
```

Goal: đánh giá decode/memory benefit và factuality risk.

## Stage 3 — Speculative-decoding map

```text
EAGLE-3
MagicDec
SpecExtend
LongSpec
DFlash
```

Goal: xác định drafting/verification paradigm nào hữu ích cho long-input summarization và large batch.

## Stage 4 — Semantic-selection depth study

```text
Lead-K
TextRank / LexRank
Embedding + MMR
LLMLingua
GemFilter
Speculative Prefill
HiGOE
```

Goal: kiểm tra semantic preprocessing có tạo quality–latency frontier tốt hơn low-level optimizations hay không.

---

# 11. Research questions supported by this baseline suite

### RQ1 — Which phase dominates long-document summarization inference?
Prefill, decode, KV bandwidth hay scheduling?

### RQ2 — Is semantic source reduction more effective than low-level sparse computation?
Compare `LLMLingua/HiGOE` vs `GemFilter/SpecPrefill` vs `MInference`.

### RQ3 — How much source can be removed while preserving summary quality?
Retention -> Quality / Factuality / Latency.

### RQ4 — Do long-context speculative methods survive large batch?
Compare Dense AR, EAGLE, MagicDec, LongSpec, SpecExtend, DFlash.

### RQ5 — Is lossless acceleration worth lower peak speed for production?
Compare lossless SD vs approximate semantic/context reduction.

### RQ6 — Can orthogonal methods compose?
Potential combinations:
- Semantic selection + EAGLE
- Semantic selection + SpecExtend
- MInference + SD
- FastKV + SD
- LLMLingua + Dense AR
- LLMLingua + SD

Không giả định gains cộng tuyến tính; phải đo actual E2E interaction.

---

# 12. Checklist

## Repositories

- [x] DFlash
- [x] EAGLE
- [x] FastKV
- [x] GemFilter
- [x] HiGOE
- [x] LLMLingua
- [x] LongSpec
- [x] MagicDec
- [x] MInference
- [x] RocketKV
- [x] SpecExtend
- [x] SpecForge
- [x] Speculative Prefill

## Optional additions only if needed

```text
FlexPrefill    # second adaptive sparse-attention baseline
KVTuner        # KV quantization branch
SSSD/UniSpec   # training-free n-gram SD branch
500xCompressor # learned soft/latent prompt compression
```

## Still required in own codebase

- [ ] Dense AR production baseline
- [ ] Lead-K semantic baseline
- [ ] TextRank/LexRank
- [ ] Embedding + MMR
- [ ] unified dataset loader
- [ ] unified metric implementation
- [ ] unified latency profiler
- [ ] selector-overhead measurement
- [ ] continuous-batching benchmark
- [ ] quality/factuality evaluation
- [ ] common result schema

---

# 13. Recommended unified result schema

Mỗi run nên lưu ít nhất:

```json
{
  "method": "",
  "dataset": "",
  "model": "",
  "input_tokens": 0,
  "retained_tokens": 0,
  "output_tokens": 0,
  "batch_size": 0,
  "selector_latency_ms": 0.0,
  "prefill_ms": 0.0,
  "decode_ms": 0.0,
  "ttft_ms": 0.0,
  "tpot_ms": 0.0,
  "e2e_ms": 0.0,
  "pipeline_e2e_ms": 0.0,
  "throughput_tok_s": 0.0,
  "qps": 0.0,
  "peak_memory_gb": 0.0,
  "rouge1": 0.0,
  "rouge2": 0.0,
  "rougeL": 0.0,
  "semantic_score": 0.0,
  "factuality_score": 0.0,
  "entity_recall": 0.0,
  "number_recall": 0.0
}
```

Khi chạy cùng một request với dense/reference target, có thể ghi thêm các
timing cặp sau để tính speedup:

```json
{
  "dense_e2e_ms": 0.0,
  "dense_ttft_ms": 0.0,
  "dense_prefill_ms": 0.0,
  "dense_decode_ms": 0.0
}
```

Collector tính ratio của mean: `ESR = dense_e2e / method_pipeline_e2e`,
`DSR = dense_decode / method_decode`, cùng `prefill_speedup` và
`ttft_speedup`. Timing thiếu làm metric tương ứng unavailable, không bị thay
bằng zero.

Speculative methods add:

```json
{
  "avg_accept_length": 0.0,
  "acceptance_rate": 0.0,
  "draft_latency_ms": 0.0,
  "verification_latency_ms": 0.0,
  "rejected_draft_ratio": 0.0
}
```

---

# 14. Current interpretation of the repository suite

```text
Semantic preprocessing        : LLMLingua, HiGOE
Token-selective prefill       : GemFilter, Speculative Prefill
Sparse attention              : MInference
Joint prefill / KV            : FastKV
KV compression                : RocketKV
Generic learned speculation   : EAGLE
Alternative parallel drafting : DFlash
Long-context speculation      : MagicDec, LongSpec, SpecExtend
Infrastructure                : SpecForge
Production control            : Dense AR (to implement)
```

Từ đây, ưu tiên không còn là clone thêm thật nhiều repo mà là xây một **unified benchmark harness** để tất cả phương pháp được đánh giá trên cùng model, dataset, prompt template, hardware, batch/concurrency và quality metrics.

---

# 15. Suggested final-paper comparison sets

## Cross-paradigm main set

```text
Dense AR
LLMLingua
GemFilter
Speculative Prefill
MInference
FastKV
RocketKV
EAGLE-3
MagicDec
SpecExtend
LongSpec
Ours
```

## Semantic-selection focused set

```text
Full Context
Lead-K
TextRank
Embedding + MMR
LLMLingua
GemFilter
Speculative Prefill
HiGOE
Ours
```

## Speculative-decoding focused set

```text
Dense AR
EAGLE-3
MagicDec
LongSpec
SpecExtend
DFlash
Ours
```

---

_Last updated for the current literature/reproduction stage._
