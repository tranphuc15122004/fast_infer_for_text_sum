# RECAP-KV V2 — thiết kế Source-State Lease

> Revision được phê duyệt sau pilot V0. Nội dung V0 bên dưới được giữ làm
> historical record; V2 thay thế core hypothesis và validation gate.

## V2 decision

V0 đã thất bại trên pilot: hidden-state semantic index kém current attention
29–32 điểm phần trăm, còn `current attention + residual` cũng không cải thiện
baseline. Vì vậy V2 bỏ future-importance prediction và residual score khỏi
core method.

V2 kiểm chứng câu hỏi mới:

> Với query hiện tại, trong bao lâu ta có thể chứng nhận rằng cold source
> attention vẫn dưới một error budget mà không rescore toàn bộ source K?

V2 phase 1 chỉ là certificate/lease-headroom experiment. Không physical KV
eviction, quantization, offload hoặc custom kernel trước khi certificate gate
đạt.

## V2 method contract

Tại anchor `t0`, với query `q0`, rotated source key `k_j` và source block `B_i`:

```text
z_j(q) = q^T k_j / sqrt(d)
Z_i^0 = sum_{j in B_i} exp(z_j(q0))
kappa_i = max_{j in B_i} ||k_j||_2 / sqrt(d)
D_t = ||q_t - q0||_2
```

Với hot set `H` và cold set `C`, upper/lower contributions là:

```text
U_C(D) = sum_{i in C} exp(D * kappa_i) * Z_i^0
L_H(D) = sum_{i in H} exp(-D * kappa_i) * Z_i^0
```

Live non-source keys được tính exact khi có thể. Certificate cho cold mass là:

```text
bound_C(D) = U_C(D) / (U_C(D) + L_H(D) + Z_live(t))
```

Nếu `bound_C(D_t) <= delta_cert`, anchor lease còn valid. Khi bound vượt
ngưỡng, lease expire và hệ thống phải refresh full source score. Đây là upper
bound theo geometry, không phải future-importance prediction.

### Registered V2 defaults

- target: existing `/home/tuantb/models/Qwen3-0.6B`;
- source blocks: 128 tokens;
- `delta_anchor = 0.05`: hot set giữ ít nhất 95% anchor source mass;
- `delta_cert = 0.10`: upper bound cold mass cho phép;
- all full-attention Qwen3 layers and query/KV head mapping;
- greedy decode, seed 42, tối đa 128 generated tokens;
- pilot: 20 GovReport + 20 Multi-News.

## V2 metrics and gates

Mỗi sample/report phải ghi:

- certificate violation rate: `actual_cold_mass > bound + 1e-5`;
- lease length và median lease length;
- certified cold block/token fraction;
- refresh rate;
- bound tightness `bound - actual_cold_mass`;
- ranking Jaccard giữa anchor hot set và actual current hot set;
- source-score events/FLOPs avoided proxy;
- per-dataset and per-layer/head summaries.

Kill gate cho bước physical tiếp theo:

- violation rate `<= 1e-3`;
- certified cold fraction `>= 0.20`;
- median lease length `>= 4` decode tokens;
- refresh rate cho thấy có source rescoring saving thực tế.

Nếu lease chủ yếu dài 1–2 token hoặc certified cold fraction dưới 20%, dừng
nhánh physical tiering. Đây là headroom gate, không phải quality/speedup claim.

## Validation scope

- L0: pure-Python certificate math, log-domain stability, GQA mapping và brute-
  force soundness trên random fixtures.
- L1: Modal smoke với trace query/K thật, checkpoint/venv/GPU validation.
- R5 pilot: hai dataset, 40 document, không future oracle trong certificate.
- Physical KV tiering và quantization: chưa mở trong V2 phase 1.

## Impact on Plan

- V0 policy/evaluator/collector giữ nguyên để làm negative diagnostic và tái lập
  kết quả cũ.
- Integration runner được mở rộng thêm `--mode lease`/V2 output nhưng không
  làm thay đổi output schema V0.
- Thêm `lease.py` cho certificate math; thêm `lease_collector.py` cho query,
  source K geometry và event-driven lease trace.
- Modal runner giữ checkpoint, venv, GPU và Volume; pilot command thêm V2 mode.
- Scientific gate chuyển từ `RECAP_SUPPORTED` sang certificate headroom gate.

## Mục tiêu

Kiểm tra trên target frozen liệu source evidence trong long-document
summarization có lifecycle `unseen -> active -> consumed -> reactivated`, và
liệu policy dùng residual evidence utility có xếp hạng future source use tốt
hơn historical/current-attention baselines hay không.

V0 là experiment/policy simulator, không phải production KV eviction. Nó
không thay đổi model weights, không training, không xóa KV thật và không claim
latency gain. Physical active/backing store chỉ được mở sau khi gate khoa học
đạt.

## Đánh giá ý tưởng

Ý tưởng có scientific hook rõ: các method phổ biến giữ importance tích lũy,
current query salience hoặc rate/precision theo token; RECAP đặt giả thuyết
rằng utility còn lại của source giảm sau khi evidence đã được verbalize. Đây là
một hướng đủ đặc thù cho summarization để đáng kiểm chứng.

Rủi ro chính là `cos(summary_hidden, source_hidden)` chỉ là proxy cho causal
consumption. Vì vậy V0 bắt buộc tách ba tín hiệu:

1. `future_use_oracle`: source-block attention trên các query tương lai, chỉ
   dùng để đánh giá hindsight ceiling;
2. `current_attention`: relevance quan sát ở prefix hiện tại;
3. `recap`: current relevance điều chỉnh bởi residual state cập nhật sau từng
   semantic segment.

Nếu RECAP không cải thiện ranking future use hoặc retention oracle không an
toàn, report phải kết luận `gate_fail/inconclusive`, không chuyển sang kernel.

## Hypothesis và biến

**Hypothesis:** sau khi một source block được dùng để sinh một summary
segment, future utility của block thường giảm; do đó `pi_i * residual_i` dự
đoán future-use tốt hơn historical cumulative attention hoặc current attention.

**Independent variables:** policy (`current`, `historical`, `recap`), source
chunk size, `beta`, `eta`, residual floor, active-budget rule và dataset.

**Dependent variables:** future-use Recall@K, NDCG@K, Spearman rank
correlation, stale-salience gap, oracle active-source fraction, retention
top-1 agreement/KL/NLL khi có ablation.

**Controls:** cùng model checkpoint, tokenizer, prompt template, seed,
generation config, source partition và generated continuation; full source là
reference bắt buộc. Future attention không được dùng làm online feature.

## Kiến trúc V0

```text
HF target checkpoint (frozen)
        |
        +-- incremental greedy trace: source attention + hidden vectors
        |
        +-- block index: pooled source hidden/prototype
        |
        +-- segment boundary: punctuation or fixed fallback window
        |
        +-- offline simulator
              +-- current attention baseline
              +-- historical attention baseline
              +-- RECAP relevance/consumption/residual policy
              +-- future-use oracle + ranking/gate report
```

Core code nằm trong `src/TrainingFree` và không import test/validation code.
Collector có thể tái sử dụng prompt rendering và model-call conventions của
`src/analyze/groundsync/trace_target.py`, nhưng output có schema riêng
`recap.trace.v1`. E29-A vẫn là reference cho hindsight retirement/retention;
V0 không copy lại E29 math.

### Policy contract

Với `M` blocks, query vectors `Q_s`, source prototypes `P_i,r` và source
embeddings `e_i`:

```text
g_i = max_q,r cosine(q, P_i,r) / temperature
pi  = softmax(g)
C_i = pi_i * max(0, cosine(segment_hidden, e_i))
R_i' = max(residual_floor, R_i * exp(-eta * C_i))
U_i = pi_i * (1 - beta * (1 - R_i))
```

`TopK(U, K_s)` là active set cho segment kế tiếp; entropy của `pi` quyết định
`K_s` trong `[k_min, k_max]`. Residual chỉ bị demote, không bị xóa, nên block
có thể re-activate khi current relevance tăng.

Collector V0 dùng last-layer hidden representations làm post-hoc semantic
index để tránh phụ thuộc vào private KV-cache layout. Đây là một biến thể
được ghi rõ trong manifest; target-key index sẽ là ablation sau nếu gate pass.

## Dataset/checkpoint/runtime

- Default target: `/home/tuantb/models/Qwen3-0.6B`, được mount vào Modal từ
  checkpoint hiện có.
- Các checkpoint `Qwen3-4B-Domino-b16`, `Qwen3-4B_eagle3` và
  `dspark_qwen3_4b_block7` là draft/speculator architectures, không dùng làm
  RECAP target; CLI cho phép override khi target CausalLM tương ứng có trên
  Volume.
- Initial smoke: `data/representative_100/govreport_representative.jsonl`, 1
  mẫu, tối đa 32 output tokens.
- Pilot: GovReport và Multi-News, tối đa 20 mẫu/dataset, tối đa 128 output
  tokens, source chunk 128, các budget `K={2,4,8}` nếu số block cho phép.
- Modal dùng image `requirements.modal.txt`, Python 3.12 venv
  `/mnt/fast-in/venv` với `--system-site-packages`, GPU mặc định A100-80GB,
  Volume `fast-infer-text-sum-cache` cho cache/output.

## Validation scope và gates

### L0 — bắt buộc, local CPU

- policy math deterministic, finite-value validation, tie-breaking;
- trace schema round-trip và invalid-shape rejection;
- Recall/NDCG/retirement metrics không leak future oracle vào policy;
- `py_compile` và unit tests package.

### L1 — Modal GPU

- preflight import/runtime/device/checkpoint;
- one-sample smoke sinh trace, summary và report;
- pilot chỉ chạy nếu smoke có `status=ok`, output trace hữu hạn và artifact
  manifest đọc được.

### Scientific gate

Gate `RECAP_SUPPORTED` chỉ khi đồng thời:

- ít nhất 2 dataset có tối thiểu 10 document hợp lệ;
- RECAP NDCG@K và Recall@K trung bình không thấp hơn current-attention
  baseline, và có improvement trên ít nhất một budget trung bình;
- stale-salience: có nhóm block historical attention cao nhưng future-use thấp
  với bootstrap CI không suy biến;
- nếu có retention ablation, top-1 agreement >= 95% ở cấu hình đăng ký.

Nếu thiếu dữ liệu, Modal lỗi, hoặc gate chưa đủ, report ghi `INCONCLUSIVE`;
không gọi đó là evidence consumption đã được chứng minh.

## Review criteria

```yaml
review_criteria:
  metrics:
    - name: recap_ndcg_at_k_vs_current
      direction: ">="
      threshold: 0.0
    - name: recap_recall_at_k_vs_current
      direction: ">="
      threshold: 0.0
    - name: retention_top1_agreement
      direction: ">="
      threshold: 0.95
  performance:
    - "CPU unit suite completes without model load"
    - "Modal smoke writes manifest and trace artifact"
  observability:
    - "record model/checkpoint, GPU, venv, seed, data, policy config"
    - "report per-dataset and per-budget metrics"
  stability:
    - "no NaN/Inf in trace or policy scores"
    - "per-sample failures are explicit and do not become PASS"
  custom:
    - "future-use oracle is labeled hindsight and never treated as online policy"
    - "physical KV eviction remains out of scope until scientific gate passes"
```

## Literature note

The proposal is positioned against dynamic/query-aware/semantic KV compression,
not claimed absolutely novel before a complete literature review. DYLE supports
the premise that snippet relevance can vary during decoding; newer work such as
SemantiCache and VarRate covers semantic grouping and nonzero variable-rate
allocation. V0 therefore tests the narrower causal/temporal principle instead
of building another mixed-precision system.
