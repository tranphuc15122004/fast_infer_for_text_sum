# Báo cáo kết quả chi tiết — RECAP-KV V0

**Ngày:** 2026-09-18  
**Mục tiêu:** kiểm chứng principle `residual evidence utility` cho training-free
KV retention trong long-context summarization.  
**Quyết định:** chưa đủ bằng chứng để triển khai physical KV eviction/offload.

## CONFIRMED

### 1. Pipeline đã được triển khai và chạy end-to-end

V0 đã có đầy đủ các lớp sau:

- `policy.py`: tính current relevance, consumption, residual và utility;
- `collector.py`: chạy frozen target, thu source-block hidden states,
  attention và semantic segments;
- `evaluation.py`: future-use hindsight oracle, current-attention baseline,
  historical baseline, RECAP và hai ablation;
- `run.py`: sinh `trace.jsonl`, `metrics.jsonl`, `metrics.json`,
  `manifest.json` và `report.md`;
- `scripts/modal_trainingfree.py`: chạy bằng checkpoint local đã mount, venv
  persistent và GPU Modal.

Validation local cuối cùng:

```text
18 passed
compileall: pass
git diff --check: pass
```

### 2. Modal runtime đã hoạt động đúng

Pilot dùng:

| Thành phần | Giá trị |
|---|---|
| Target checkpoint | `/opt/models/Qwen3-0.6B` |
| GPU | Modal `A100-80GB` |
| Python | `/mnt/fast-in/venv/bin/python` |
| CUDA | `cuda_available=true` |
| Block size | 128 token |
| Segment cap | 16 token |
| Max generated tokens | 128 |
| Seed | 42 |
| Budgets | `K={1,2,4,8}` |

Smoke run `recap-v0-smoke-qwen06b` sinh trace hợp lệ trên Modal. Pilot
`recap-v0-pilot-qwen06b-ablation` hoàn tất `40/40` mẫu, gồm 20 GovReport và
20 Multi-News, không có error hoặc inconclusive sample.

Artifact chính trong Modal Volume:

```text
outputs/recap_kv_v0/recap-v0-pilot-qwen06b-ablation/
├── trace.jsonl
├── metrics.jsonl
├── metrics.json
├── manifest.json
└── report.md
```

Pilot có tổng cộng 265 transition có strict future segment để đánh giá.
GovReport có trung bình 8.85 transition/document; Multi-News có 4.40.

### 3. Scientific gate đã được áp dụng và trả về `GATE_FAIL`

Gate V0 yêu cầu tối thiểu hai dataset, mỗi dataset ít nhất 10 document; đồng
thời RECAP phải không thấp hơn current attention trên cả Recall@K và NDCG@K,
và phải có improvement ở ít nhất một budget. Pilot đạt điều kiện dữ liệu nhưng
không đạt điều kiện metric.

Đây là kết quả đã xác nhận của experiment: với implementation hiện tại,
RECAP hidden-index không vượt current attention ở bất kỳ budget nào.

## EXPLORATORY

Các số dưới đây là phân tích pilot và ablation. Chúng hữu ích để chẩn đoán
nhưng không nên diễn giải thành bằng chứng rằng mọi biến thể của principle đều
thất bại.

### 1. Kết quả aggregate trên 40 document

`Delta = policy - current attention`; số âm nghĩa là thấp hơn baseline.

| K | Policy | Recall@K | Δ Recall | NDCG@K | Δ NDCG |
|---:|---|---:|---:|---:|---:|
| 1 | Current attention | 0.702077 | +0.000000 | 0.702077 | +0.000000 |
| 1 | RECAP hidden-index | 0.405200 | -0.296878 | 0.405200 | -0.296878 |
| 1 | Attention-residual ablation | 0.689996 | -0.012081 | 0.689996 | -0.012081 |
| 1 | Historical attention | 0.611924 | -0.090153 | 0.611924 | -0.090153 |
| 2 | Current attention | 0.745277 | +0.000000 | 0.732914 | +0.000000 |
| 2 | RECAP hidden-index | 0.437947 | -0.307330 | 0.428815 | -0.304099 |
| 2 | Attention-residual ablation | 0.731391 | -0.013885 | 0.719606 | -0.013308 |
| 2 | Historical attention | 0.675815 | -0.069462 | 0.656770 | -0.076144 |
| 4 | Current attention | 0.814056 | +0.000000 | 0.780046 | +0.000000 |
| 4 | RECAP hidden-index | 0.498017 | -0.316039 | 0.469881 | -0.310166 |
| 4 | Attention-residual ablation | 0.799950 | -0.014106 | 0.767167 | -0.012879 |
| 4 | Historical attention | 0.791679 | -0.022378 | 0.735800 | -0.044245 |
| 8 | Current attention | 0.886485 | +0.000000 | 0.830441 | +0.000000 |
| 8 | RECAP hidden-index | 0.595427 | -0.291058 | 0.532282 | -0.298159 |
| 8 | Attention-residual ablation | 0.881485 | -0.005001 | 0.822526 | -0.007915 |
| 8 | Historical attention | 0.890464 | +0.003978 | 0.806353 | -0.024088 |

### 2. Phân tích theo dataset

#### GovReport — 20 document

| K | RECAP Δ Recall | RECAP Δ NDCG | Attention-residual Δ Recall | Attention-residual Δ NDCG |
|---:|---:|---:|---:|---:|
| 1 | -0.376871 | -0.376871 | -0.017964 | -0.017964 |
| 2 | -0.379914 | -0.379045 | -0.007737 | -0.010200 |
| 4 | -0.405137 | -0.396006 | -0.008886 | -0.010092 |
| 8 | -0.393393 | -0.392573 | -0.006747 | -0.008557 |

#### Multi-News — 20 document

| K | RECAP Δ Recall | RECAP Δ NDCG | Attention-residual Δ Recall | Attention-residual Δ NDCG |
|---:|---:|---:|---:|---:|
| 1 | -0.216884 | -0.216884 | -0.006198 | -0.006198 |
| 2 | -0.234746 | -0.229153 | -0.020034 | -0.016416 |
| 4 | -0.226941 | -0.224325 | -0.019326 | -0.015666 |
| 8 | -0.188723 | -0.203745 | -0.003255 | -0.007273 |

RECAP hidden-index thất bại trên cả hai dataset và cả bốn budget. Vì vậy kết
quả không giống một lỗi chỉ xảy ra trên GovReport hoặc Multi-News.

### 3. Chẩn đoán từ ablation

Hai policy dùng cùng residual update và khác chủ yếu ở tín hiệu relevance:

- `RECAP hidden-index`: relevance từ cosine giữa query hidden và source
  prototypes;
- `Attention-residual`: relevance lấy trực tiếp từ current source attention,
  sau đó mới áp dụng residual update.

Khoảng cách lớn giữa hai biến thể cho thấy semantic hidden-state index/proxy là
điểm hỏng chính của V0. Attention-residual gần current baseline hơn nhiều,
nhưng vẫn âm trên mọi Recall/NDCG budget. Historical chỉ có một điểm dương là
Recall@8 aggregate.

Đây là chẩn đoán tương quan từ ablation, chưa phải chứng minh nhân quả. Nó
không chứng minh residual principle sẽ tốt khi thay bằng target key/query,
head aggregation hoặc consumption signal khác.

### 4. Ý nghĩa của historical baseline

Historical attention cải thiện Recall@8 aggregate `+0.003978`, nhưng NDCG@8
vẫn giảm `-0.024088`. Do gate yêu cầu đồng thời Recall và NDCG, observation này
không đủ để gọi historical policy là phương pháp tốt hơn.

## FAILED / INCOMPLETE

### Scientific failure

1. RECAP hidden-index không đạt locked success criterion ở `K=1,2,4,8`.
2. RECAP hidden-index thấp hơn current attention khoảng 29–32 điểm phần trăm
   trên aggregate Recall/NDCG.
3. Không có evidence rằng residual utility giúp ranking future use tốt hơn
   baseline trong cấu hình đã đăng ký.
4. Không được triển khai physical KV eviction, KV offload hoặc kernel tối ưu;
   làm vậy sau kết quả này sẽ tạo rủi ro đánh đổi quality chưa được kiểm chứng.

### Incomplete scope

Các phần sau chưa được kiểm chứng:

- full LongBench study ngoài pilot 40 document;
- nhiều seed và confidence interval/bootstrap;
- ROUGE hoặc quality của summary sau retention thật;
- NLL/logit drift sau khi bỏ hoặc offload KV;
- latency, throughput và memory saving;
- retention top-1 agreement;
- stale-salience gap;
- target-key/query index;
- physical cache ablation.

Các checkpoint `Qwen3-4B-Domino-b16`, `Qwen3-4B_eagle3` và
`dspark_qwen3_4b_block7` không được dùng làm target trong experiment này vì
config của chúng là draft/speculator architecture, không phải complete causal
LM target tương thích với collector V0. Runner vẫn cho phép override khi có
checkpoint target đầy đủ trên Modal Volume.

## HIGHEST VERIFIED RUNG

**Highest verified empirical rung: R5 — short pilot trên real data, real
Modal launcher và locked evaluation metric.** Artifact chứng minh là:

```text
Modal Volume:
outputs/recap_kv_v0/recap-v0-pilot-qwen06b-ablation/manifest.json
outputs/recap_kv_v0/recap-v0-pilot-qwen06b-ablation/metrics.json
outputs/recap_kv_v0/recap-v0-pilot-qwen06b-ablation/metrics.jsonl
```

R4 smoke đã xanh và R1 static/unit validation đã xanh. R6 full study chưa
được chạy; do đó kết quả này không phải bằng chứng full-scale hoặc multi-seed.
Báo cáo hiện tại là decision memo ở mức review, nhưng không nâng pilot thành
R6.

**Supported:** implementation chạy ổn định end-to-end trên Modal và pilot
được đánh giá đúng schema, đúng baseline, đúng future-use oracle không leakage.

**Not supported yet:** RECAP hidden-index vượt current attention; residual
principle cải thiện retention; quality/latency/memory benefit của physical KV
eviction.

## EVIDENCE GAPS

1. **Tín hiệu semantic:** hidden-state cosine có thể không phản ánh source
   token thực sự được dùng để tạo segment.
2. **Granularity:** block 128 token và segment 16 token có thể làm mất lifecycle
   evidence ở mức câu hoặc token.
3. **Attention measurement:** collector đang collapse attention theo layer/head
   bằng mean; chưa có head selection hoặc target key/query matching.
4. **Oracle limitation:** future-use là hindsight oracle để chấm điểm, không
   phải online signal và không được dùng trong policy.
5. **Generalization:** mới có một target model, một seed và hai dataset pilot.
6. **System value:** chưa có thí nghiệm quality-preserving KV deletion nên
   chưa thể kết luận về speedup hoặc memory reduction.

## RECOMMENDED NEXT

Thí nghiệm tiếp theo duy nhất nên ưu tiên là **thay hidden-state semantic proxy
bằng target key/query-based consumption**, giữ nguyên tất cả điều kiện còn lại:
checkpoint, dataset, block size, segment size, seed, budgets và evaluator.

Thiết kế so sánh:

```text
current attention
historical attention
RECAP(hidden-state)          # giữ làm negative diagnostic
RECAP(target-key/query)      # biến thể cần kiểm chứng
RECAP(attention + residual)  # ablation control
```

Promotion criterion cho pilot sửa lỗi:

- Recall@K và NDCG@K không thấp hơn current attention ở mọi budget;
- có improvement ở ít nhất một budget trung gian;
- không có NaN/Inf và tất cả sample hoàn tất;
- sau đó mới cân nhắc R6: nhiều seed, dataset rộng hơn và retention ablation.

Không nên bật physical KV eviction trước khi biến thể target-key/query vượt qua
gate này.

## Phụ lục A — công thức V0

Với source block `i`, query `q`, prototype `p`, source embedding `e` và
segment hidden `h`:

```text
g_i = max cosine(q, p_i,r) / temperature
pi  = softmax(g)
C_i = pi_i * max(0, cosine(h, e_i))
R_i' = max(residual_floor, R_i * exp(-eta * C_i))
U_i = pi_i * (1 - beta * (1 - R_i))
```

Policy chỉ dùng thông tin hiện tại và state residual trước đó. Future attention
chỉ được tính sau khi trace hoàn tất để làm oracle đánh giá.

## Phụ lục B — lệnh tái lập

```bash
# Smoke
MODAL_GPU=A100-80GB modal run scripts/modal_trainingfree.py \
  --mode smoke --run-id recap-v0-smoke-qwen06b

# Pilot + ablation
MODAL_GPU=A100-80GB modal run scripts/modal_trainingfree.py \
  --mode pilot --run-id recap-v0-pilot-qwen06b-ablation
```

Artifact có thể lấy từ Volume bằng:

```bash
modal volume get fast-infer-text-sum-cache \
  outputs/recap_kv_v0/recap-v0-pilot-qwen06b-ablation/metrics.json \
  /tmp/recap-v0-metrics.json
```

**Kết luận chuẩn:** Verified through R5 (short pilot). Not yet verified by a
full study or by a quality-preserving physical KV implementation.
