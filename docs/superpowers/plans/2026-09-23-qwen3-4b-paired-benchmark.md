# Qwen3-4B Paired Online Benchmark Implementation Plan

> **For agentic workers:** Thực hiện từng task theo TDD; không dùng artifact
> Llama 3.1-8B cũ để điền vào kết quả Qwen3-4B.

**Mục tiêu:** Xây dựng một benchmark LongBench mới trên cùng target Qwen3-4B,
đo timing ở batch size 1 và ghi ESR/DSR theo từng sample ngay trong quá trình
chạy các baseline Vanilla HF, Vanilla FA, DFlash, Domino và EAGLE3.

**Thư mục thực nghiệm:**
`outputs/Benchmark_results/qwen3_4b_paired_online/<run-id>/`

**Giả thuyết:** Khi target, prompt, output budget, batch size, reference và
measurement boundary được khóa chung, chênh lệch decode throughput giữa các
speculative method sẽ phản ánh đúng chi phí runtime của method thay vì chênh
lệch output length hoặc khác reference.

**Phạm vi validation:** LongBench gồm `gov_report`, `qmsum`, `multi_news`,
`lcc`, `repobench-p`; 100 sample/dataset ở full run. Baseline chính gồm
`vanilla_hf`, `vanilla_fa`, `dflash`, `domino`, `eagle3`. DSpark, FAFO,
MagicDec và SpecExtend nằm ngoài phạm vi matrix Qwen3-4B này; không xóa code
hoặc artifact cũ của chúng.

**Thiết kế đánh giá:** Chỉ có một lượt generation fixed-K cho mỗi method:
`K=1024` ở full run (`K=32` ở smoke), batch size 1 và tắt EOS stopping. Lưu
nguyên continuation token IDs; từ đó xác định EOS đầu tiên và tạo
`quality_text` bằng cách giải mã phần trước EOS. Như vậy speed và quality
được thu trong cùng một lượt, không cần chạy natural-EOS profile thứ hai.

Reference Vanilla HF chạy trước theo từng sample và ghi sidecar. Baseline
đọc sidecar đó, join bằng `dataset + sample_id + prompt_hash + run_config_hash`
và ghi ESR/DSR ngay trong record. ESR/DSR ghép theo input/config và fixed-K,
không yêu cầu các method sinh cùng chuỗi token; exact/token agreement được ghi
riêng làm correctness audit và không làm điều kiện speedup. Raw token IDs và
timing được lưu để audit/offline recompute mà không chạy model lại.

**Giới hạn input/output:** Phân tích `data/longbench_100_14k/` hiện có 500
sample cho thấy input token đã được chuẩn hóa nằm trong khoảng 165–13.904
token; p50 toàn bộ là 5.497,5, p90 là 12.351,3, p95 là 13.025,4 và p99 là
13.489,1. Vì vậy full run khóa `MAX_INPUT_TOKENS=14000` sau khi áp dụng
tokenize prompt canonical đang có; không loại bỏ sample hiện tại và không
truncate âm thầm. Smoke dùng cùng input cap. Mỗi adapter tokenizes cùng prompt
bằng tokenizer snapshot đã khóa, áp dụng chung truncation deterministic và
ghi `input_truncated=true`, `original_input_tokens`, `prompt_hash`; runner
strict-join các hash trước khi công nhận speedup.

Phân bố theo dataset hiện tại:

| Dataset | Min | P50 | P90 | Max |
|---|---:|---:|---:|---:|
| `gov_report` | 2.080 | 7.891,5 | 11.965,5 | 13.904 |
| `qmsum` | 2.632 | 10.088,5 | 13.044,1 | 13.355 |
| `multi_news` | 165 | 2.075,5 | 4.607,3 | 11.922 |
| `lcc` | 1.004 | 2.261,0 | 5.539,3 | 12.229 |
| `repobench-p` | 2.735 | 6.898,5 | 12.732,3 | 13.774 |

Fixed-K run khóa `SPEED_OUTPUT_TOKENS=1024` ở full và 32 ở smoke. Mọi method
phải sinh đủ đúng số token; record đồng thời giữ `generated_token_ids`,
`first_eos_index`, `quality_output_tokens`, `quality_text` và
`full_output_text`. Chất lượng được đo trên phần text trước EOS; tốc độ dùng
đúng toàn bộ fixed-K continuation.

**Kiến trúc:** Dùng `benchmark_runtime.py` làm schema/timing core, thêm một
reference sidecar contract dùng chung, rồi nối Domino vào registry/LongBench
adapter. Vanilla HF/FA, DFlash và EAGLE3 tái sử dụng loader hiện có nhưng phải
tuân thủ Qwen3-4B manifest và paired-reference contract. Collector chỉ kiểm
tra coverage, parity và tổng hợp; không được thay thế hoặc sửa raw timing.

---

## Metric contract

### Raw fields bắt buộc trên mỗi sample

```text
dataset, sample_id, prompt_hash, run_config_hash
model, target_model_revision, batch_size, device, dtype, attention_backend
input_tokens, output_tokens, max_new_tokens
original_input_tokens, input_truncated, speed_output_tokens
generated_token_ids, generated_token_count, first_eos_index
quality_output_tokens, quality_text, full_output_text, fixed_budget_reached
fixed_continuation_exact_match, quality_prefix_exact_match
fixed_continuation_token_match_ratio, fixed_continuation_first_divergence_index
prefill_ms, decode_ms, e2e_ms
throughput_tok_s, decode_throughput_tok_s, tpot_ms
temperature, warmup_runs, measurement_scope
```

Speculative methods phải ghi thêm:

```text
acceptance_lengths, avg_accept_length, acceptance_rate
draft_latency_ms, verification_latency_ms
```

### Paired reference fields

Mỗi method record trong speed profile phải ghi:

```text
reference_baseline: vanilla_hf
reference_run_id
dense_prefill_ms, dense_decode_ms, dense_e2e_ms
dense_output_tokens
speedup_valid
speedup_scope: qwen3_4b_batch1_fixed_budget
```

### Derived metrics

```text
decode_tok_s = output_tokens / (decode_ms / 1000)
ESR = dense_e2e_ms / method_e2e_ms
DSR = method_decode_tok_s / dense_decode_tok_s
```

Với fixed budget và cùng số output token, DSR cũng bằng
`dense_decode_ms / method_decode_ms`. Aggregate phải dùng tổng token chia
tổng thời gian, không lấy mean không trọng số của tok/s từng sample.

`tau` là mean accepted tokens mỗi speculative step/sample; Vanilla HF và
Vanilla FA ghi `null`, không ghi `0`.

### Fairness rules

- Target duy nhất: local snapshot `Qwen3-4B`; không bật thinking mode.
- Cùng tokenizer, prompt file, sample IDs, seed, temperature,
  dtype, `MAX_INPUT_TOKENS=14000`, output budget và warmup.
- Reference và baseline dùng cùng prompt source/tokenizer snapshot; prompt hash
  sau truncation phải khớp chính xác trong paired join.
- Fixed-K run phải có `output_tokens == SPEED_OUTPUT_TOKENS`; EOS sớm không
  dừng generation mà chỉ đánh dấu prefix chất lượng.
- Batch size luôn bằng 1; process startup và model loading không nằm trong
  sample timing.
- Có `torch.cuda.synchronize()` trước và sau vùng đo.
- Không chạy reference và method đồng thời trên cùng GPU.
- `vanilla_hf` là reference chính; `vanilla_fa` là dense control riêng.
- Không fallback âm thầm từ FlashAttention sang backend khác.
- Nếu token parity hoặc paired join fail, record bị loại khỏi speedup chính và
  phải xuất hiện trong audit.

## Shared scaffold

### Existing infrastructure — giữ và tái sử dụng

- `scripts/common/benchmark_runtime.py`: timing, CUDA synchronization, record
  schema và throughput.
- `scripts/common/metrics.py`: aggregate timing/speedup hiện có; cần mở rộng
  contract, không dùng ratio aggregate cũ làm thay thế paired speedup.
- `scripts/common/metric_audit.py`: kiểm tra coverage và missing metrics.
- `scripts/common/longbench_adapter.py`: registry và command builder hiện tại.
- `scripts/run_longbench_200.py`: orchestration/sharding/manifest.
- `scripts/infer_vanilla_hf.py`, `scripts/infer_vanilla_fa.py`,
  `scripts/infer_dflash.py`, `scripts/eagle3_infer_qwen3.py`.
- `externals/Domino/`: implementation Domino Qwen3 và benchmark API upstream.

### Artifact layout mới

```text
outputs/Benchmark_results/qwen3_4b_paired_online/<run-id>/
├── run_manifest.json
├── inputs/source/<dataset>.jsonl
├── inputs/<dataset>.eagle.jsonl
├── vanilla_hf/<dataset>.jsonl
├── vanilla_fa/<dataset>.jsonl
├── {dflash,domino,eagle3}/<dataset>.jsonl
├── references/<dataset>.jsonl
└── metrics_summary.{json,csv,md}
```

Không ghi đè `outputs/Benchmark_results/ketqua_benchmark/` hoặc report Llama
hiện tại cho đến khi run mới đạt toàn bộ acceptance criteria.

## Task 1: Freeze Qwen3-4B experiment manifest và config contract

**Files:**

- Create: `docs/experiments/2026-09-23-qwen3-4b-paired-benchmark.md`
- Modify: `docs/fast_infer_master.example.env`
- Modify: `scripts/common/config.sh`
- Modify: `scripts/run_longbench_200.py` nếu cần thêm profile/sidecar flags
- Test: `tests/test_qwen3_4b_paired_contract.py`

**Implementation:**

- Thêm profile rõ ràng cho target Qwen3-4B, local-files-only, BF16, batch 1,
  `temperature=0`, `enable_thinking=False`.
- Tách `REFERENCE_BASELINE=vanilla_hf` khỏi `vanilla_fa`.
- Thêm `SPEED_OUTPUT_TOKENS=1024` cho full và `32` cho smoke; giá trị phải
  được ghi vào manifest, không lấy ngầm từ config cũ 2048.
- Thêm `MAX_INPUT_TOKENS=14000` cho cả smoke/full và lưu phân tích input
  distribution/hash của từng JSONL vào manifest; không dùng `14k` như tên
  thư mục mà bỏ qua số token thực tế sau tokenize/truncation.
- Khóa danh sách baseline mới; DSpark không được tự động xuất hiện trong run.
- Sinh `run_config_hash` từ target path, dataset SHA-256, budget, seed, dtype,
  backend và workload; strict pairing đồng thời đối chiếu target/tokenizer
  revision thực tế được ghi trong output rows.

**Tests:**

- Manifest có đúng 5 dataset và 5 baseline.
- `batch_size == 1`, `reference_baseline == vanilla_hf` và
  `enable_thinking == False`.
- `MAX_INPUT_TOKENS == 14000`, `SPEED_OUTPUT_TOKENS == 1024` ở full profile;
  smoke dùng 32 output token.
- Data preflight xác nhận sample count theo từng JSONL; nếu có truncation,
  mọi baseline phải có cùng prompt hash sau cap.
- Không chấp nhận trộn target Llama hoặc output directory cũ.

## Task 2: Implement paired reference sidecar và online ESR/DSR

**Files:**

- Create: `scripts/common/paired_reference.py`
- Modify: `scripts/common/benchmark_runtime.py`
- Modify: `scripts/common/metrics.py`
- Modify: `scripts/common/metric_audit.py`
- Test: `tests/test_paired_reference_metrics.py`

**Implementation:**

- Implement sidecar writer/reader keyed by dataset, sample ID, prompt hash và
  run config hash.
- Implement strict join: khác prompt, budget, target/tokenizer revision, batch size hoặc
  config hash phải trả `speedup_valid=False` với reason cụ thể.
- Derive per-record `decode_throughput_tok_s`, `tpot_ms`, `esr`, `dsr` sau khi
  method timing hoàn tất nhưng trước khi append JSONL; đây là “online” metric.
- Giữ `dense_*` và raw method timing trong record.
- Thêm aggregate weighted token throughput và paired coverage.
- Không cho collector dùng `vanilla_fa` làm reference nếu manifest khóa
  `vanilla_hf`.

**Tests:**

- Reference join đúng sample tạo ESR/DSR đúng công thức.
- Mismatch sample/config/output budget bị invalid.
- DSR dùng token-normalized throughput khi output token khác nhau.
- Fixed equal-budget case chứng minh DSR bằng inverse decode-latency ratio.
- Missing decode timing không được tạo DSR giả.

## Task 3: Chuẩn hóa Vanilla HF/FA Qwen3-4B reference và dense control

**Files:**

- Modify: `scripts/common/vanilla_inference.py` nếu thiếu fields/config
- Modify: `scripts/infer_vanilla_hf.py`
- Modify: `scripts/infer_vanilla_fa.py`
- Modify: `scripts/common/longbench_adapter.py`
- Test: `tests/test_qwen3_vanilla_paired_contract.py`

**Implementation:**

- Vanilla HF chạy reference pass trước, một sample/request, ghi prompt hash,
  token IDs hoặc token count, input cap/truncation metadata, timing đầy đủ và
  sidecar.
- Vanilla FA chạy cùng input/config như control; nếu FlashAttention không
  tương thích thì fail rõ ràng, không fallback.
- Speed reference tắt EOS và sinh đủ `SPEED_OUTPUT_TOKENS`; quality text được
  tạo offline từ token IDs trước EOS, không thực hiện generation thứ hai.
- Tách model-load timing khỏi sample E2E.
- Kiểm tra target token parity giữa Vanilla HF và Vanilla FA ở deterministic
  speed smoke; mismatch phải được ghi audit trước khi dùng Vanilla FA làm
  control.

**Tests:**

- CPU/schema test kiểm tra record có toàn bộ raw timing field và `batch_size=1`.
- Smoke contract xác nhận `dense_*` chỉ được gắn từ reference cùng sample.
- Parser không tự bật thinking mode hoặc đổi max output budget.

## Task 4: Tích hợp Domino Qwen3-4B vào canonical adapter

**Files:**

- Create: `scripts/infer_domino.py`
- Create: `scripts/runners/run_domino.sh`
- Modify: `scripts/run.sh`
- Modify: `scripts/common/longbench_adapter.py`
- Modify: `scripts/common/metric_audit.py`
- Modify: `scripts/common/config.sh`
- Test: `tests/test_domino_longbench_adapter.py`

**Implementation:**

- Dùng Domino checkpoint `Qwen3-4B-Domino-b16` tương thích target
  `Qwen3-4B`; không dùng default Qwen3-8B của upstream.
- Chuẩn hóa LongBench JSONL sang input API Domino, giữ nguyên `sample_id` và
  reference output.
- Ghi prefill, decode, E2E, output tokens, acceptance trace và reference
  timing vào schema chung.
- Bắt buộc `block_size` và attention backend xuất hiện trong manifest.
- Đảm bảo không chạy benchmark upstream GSM8K thay cho LongBench adapter.

**Tests:**

- Adapter command có target/draft Qwen3-4B và local-files-only.
- Output normalization giữ đúng sample ID/order.
- Unit test reject checkpoint target/draft không cùng vocab/hidden size.
- Import/preflight không load model hoặc khởi tạo CUDA.

## Task 5: EAGLE3 Qwen3 parity gate

**Files:**

- Modify: `scripts/eagle3_infer_qwen3.py`
- Modify: `scripts/common/longbench_adapter.py`
- Modify: `scripts/common/metric_audit.py`
- Test: `tests/test_eagle3_qwen3_parity_contract.py`

**Implementation:**

- Giữ EAGLE3 là baseline experimental; chỉ cho phép full speed record nếu
  target parity smoke pass.
- Dùng đúng Qwen3 tokenizer, cùng prompt canonical và `enable_thinking=False`.
- Nhận cùng tokenized input và fixed output budget từ paired runner; không tự
  tokenize lại bằng prompt contract khác.
- Ghi token IDs hoặc hash của generated continuation để so với Vanilla HF;
  không so sánh trực tiếp text của EAGLE3 với DFlash.
- Ghi `tau`, draft/verification timing và paired reference fields.
- Nếu Qwen3 EAGLE checkpoint hoặc loader không đạt parity, trả trạng thái
  `correctness_failed` và loại khỏi speedup chính, không tự fallback Llama.

**Tests:**

- Target parity pass/fail được phản ánh vào status và audit.
- `tau` không bị ghi thành zero khi thiếu acceptance trace.
- EAGLE3 command không trỏ nhầm checkpoint Llama 3.1.

## Task 6: DFlash và toàn bộ experiment runner `[INTEGRATION]`

**Files:**

- Modify: `scripts/infer_dflash.py`
- Modify: `scripts/run_longbench_200.py`
- Modify: `scripts/common/longbench_adapter.py`
- Modify: `scripts/common/config.sh`
- Create: `scripts/run_qwen3_4b_paired.sh` nếu runner hiện tại không đủ để
  tách reference pass và method pass
- Test: `tests/test_qwen3_4b_paired_runner.py`

**Implementation:**

- DFlash dùng target `Qwen3-4B` và draft `Qwen3-4B-DFlash-b16`.
- Chạy theo thứ tự: manifest/data validation → Vanilla HF reference → Vanilla
  FA control → DFlash/Domino/EAGLE3.
- Reference và method đều nhận `MAX_INPUT_TOKENS=14000` và fixed budget; cùng
  record token IDs dùng cho cả tốc độ và chất lượng.
- Mỗi child process chỉ dùng batch 1; không chạy reference và method đồng thời
  trên cùng GPU.
- Truyền `reference_sidecar`, `reference_run_id`, `run_config_hash` xuống mọi
  method.
- Ghi progress/log riêng từng dataset/baseline; retry chỉ sample lỗi, không
  thay timing thành zero.
- Không đăng ký DSpark vào default matrix.

**Smoke integration:**

- 1 dataset, 2 sample, 1 GPU, fixed output budget nhỏ.
- Kiểm tra đủ record Vanilla HF/FA, DFlash, Domino và EAGLE3 hoặc status
  correctness failure rõ ràng.
- Kiểm tra ít nhất một record có `dense_*`, `esr`, `dsr` và `speedup_valid`.

**Full run:**

- 5 dataset × 100 sample.
- 8 B200 data-parallel nếu manifest server xác nhận cùng runtime; từng request
  vẫn batch 1.
- Chạy một fixed-K smoke cho toàn matrix; cùng record raw token IDs phục vụ cả
  speed và quality, không chạy thêm quality natural-EOS pass.

## Task 7: Audit, aggregate và cập nhật report

**Files:**

- Modify: `scripts/collect_metrics.py` hoặc collector mới nếu cần profile
- Modify: `scripts/make_table1_benchmark.py`
- Create/Modify: `tests/test_qwen3_4b_report_contract.py`
- Modify: `outputs/Benchmark_results/longbench_full_benchmark_analysis.md`
  chỉ sau khi full run complete
- Create: `docs/experiments/2026-09-23-qwen3-4b-paired-benchmark.md`

**Implementation:**

- Audit mỗi cell có đúng 100 unique sample IDs, 100% paired reference hợp lệ,
  đúng measurement scope và không có missing prefill/decode/E2E.
- Audit input có cùng `prompt_hash` giữa reference và method, input cap đúng
  14000, không có sample bị drop; audit output speed có đúng 1024 token cho
  mọi record hợp lệ.
- Report bảng chính theo từng dataset:

```text
Method | tau | Decode tok/s | Prefill ms | Decode ms | ESR vs Vanilla HF | DSR vs Vanilla HF
```

- Ghi thêm `n`, paired coverage, output budget và runtime/backend trong footnote
  hoặc appendix.
- `vanilla_fa` xuất hiện như dense control, không dùng để thay thế Vanilla HF
  trong ESR/DSR chính.
- Không gọi kết quả là “reproduce paper speedup” nếu model, backend hoặc
  measurement boundary khác paper; chỉ ghi observed Qwen3-4B speedup.
- DSpark/FAFO/MagicDec/SpecExtend ghi ngoài phạm vi của matrix mới.

**Acceptance criteria:**

- Manifest target duy nhất là Qwen3-4B.
- Tất cả successful speed records có `batch_size=1` và cùng fixed budget.
- Tất cả successful speed records có cùng `input_tokens`/`prompt_hash` với
  reference và `output_tokens == 1024`.
- Reference join coverage đạt 100% cho mọi baseline được xếp hạng.
- `decode_throughput_tok_s`, ESR và DSR đều tính được từ raw fields.
- Không có `speedup_valid=True` khi parity, sample join hoặc config hash fail.
- Report không trộn artifact Llama cũ và Qwen3-4B mới.

## Execution order and stop gates

1. Chạy unit/static tests cho Tasks 1–5.
2. Chạy Qwen3 Vanilla HF/FA smoke; nếu Vanilla FA không pass backend gate,
   giữ Vanilla HF làm dense reference và ghi FA control unavailable.
3. Chạy DFlash smoke.
4. Chạy Domino smoke.
5. Chạy EAGLE3 parity smoke; fail thì loại EAGLE3 khỏi full ranking.
6. Chạy integration smoke đủ matrix.
7. Chỉ khi smoke pass mới chạy full 5×100 speed profile.
8. Audit full artifact; nếu acceptance criteria đạt thì cập nhật report chính
   trực tiếp từ artifact này, không chạy lại model để lấy quality output.

Không chạy full GPU hoặc tuyên bố speedup trong paper trước khi stop gates trên
đạt. Kế hoạch này không thay thế kết quả cũ; nó tạo một benchmark Qwen3-4B
độc lập, có provenance và reference pairing rõ ràng.
