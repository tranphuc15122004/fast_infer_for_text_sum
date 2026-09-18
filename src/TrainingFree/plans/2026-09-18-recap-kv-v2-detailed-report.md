# Báo cáo kỹ thuật chi tiết — RECAP-KV V2 Source-State Lease

**Ngày thực hiện:** 2026-09-18
**Mục tiêu:** triển khai và kiểm chứng ý tưởng RECAP-KV V2 trên checkpoint
Qwen3-0.6B bằng GPU Modal, venv dùng chung và dữ liệu long-context hiện có.
**Mức đánh giá:** R5 — short pilot trên dữ liệu thật và GPU cloud.
**Quyết định cuối:** STOP_BEFORE_PHYSICAL; chưa triển khai physical KV
eviction, quantization hoặc offload.

## 1. Tóm tắt điều hành

RECAP-KV V2 thay đổi principle so với V0. V0 cố gắng dự đoán future-use của
source block từ hidden-state relevance. Kết quả V0 cho thấy tín hiệu semantic
proxy không ổn định và RECAP hidden-index thấp hơn current-attention baseline.
V2 vì vậy không dự đoán future importance. Thay vào đó, V2 cấp một lease cho
source block khi có certificate toán học cho biết query đã drift chưa đủ xa để
cold block có thể vượt ngưỡng attention đã đăng ký.

Implementation V2 đã chạy end-to-end trên Modal:

- smoke: 1/1 sample thành công;
- pilot chính: 40/40 sample thành công, gồm 20 GovReport và 20 Multi-News;
- target: checkpoint local /opt/models/Qwen3-0.6B;
- runtime: /mnt/fast-in/venv/bin/python;
- GPU thực tế theo manifest: NVIDIA A10;
- schema trace: recap.lease.trace.v1.

Certificate không có violation sau tolerance số học, nhưng quá conservative:
mean bound là 0.918639 trong khi actual cold mass chỉ 0.011961. Lease trung
bình chỉ đạt 1.525 token và refresh rate là 95.16%. Do đó method chưa tạo
được headroom đủ lớn để có giá trị systems; physical tiering không được mở.

## 2. Bối cảnh và câu hỏi nghiên cứu

### 2.1. Bài học từ V0

V0 được xây quanh principle residual evidence utility: dùng hidden-state
relevance, consumption và residual evidence để xếp hạng source block. Pilot V0
trên 40 document cho thấy:

- RECAP hidden-index thấp hơn current attention ở mọi budget K={1,2,4,8};
- attention-residual gần baseline hơn nhưng vẫn không vượt baseline;
- hidden-state cosine không phải evidence đủ tin cậy cho future source use;
- chưa có cơ sở để xoá, nén hoặc offload KV vật lý.

Vì vậy V2 không tiếp tục tối ưu cùng hidden-state proxy. Câu hỏi mới là:

> Nếu query chỉ drift một lượng nhỏ so với query tại anchor, có thể chứng nhận
> rằng cold source block vẫn nằm dưới một upper bound attention hay không?

### 2.2. Giả thuyết V2

Tại một anchor query, source attention được chia thành hot và cold block. Nếu
certificate cho phép lease dài hơn một vài decode step, có thể tránh full
source-K rescoring ở phần lớn bước decode.

Điều kiện để chuyển sang physical KV tiering là đồng thời đạt:

1. certificate violation rate <= 0.001;
2. certified cold fraction >= 0.20;
3. median lease length >= 4 token;
4. source-score avoidance dương.

Các ngưỡng trên là kill gate. Nếu không đạt, chỉ giữ instrumentation và không
động vào KV cache thật.

## 3. Thiết kế phương pháp

### 3.1. Anchor và hot/cold split

Ở anchor, hệ thống tính source attention theo block. Hot set là tập nhỏ nhất
giữ ít nhất 95% source mass:

~~~text
delta_anchor = 0.05
hot = smallest set retaining (1 - delta_anchor) source mass
cold = all remaining source blocks
~~~

Việc split được thực hiện từ attention quan sát tại anchor, không dùng
future-use oracle.

### 3.2. Certificate query-drift

Với source block i, hệ thống lưu:

- log_Z0[i]: log partition contribution tại anchor;
- kappa[i]: upper geometric scale, lấy từ max key norm trong block;
- D: khoảng cách giữa query hiện tại và query anchor;
- live_log_Z: exact contribution của các key ngoài source span.

Với cold và hot block, bound được tính trong log domain:

~~~text
U_C(D) = sum_i in cold exp(log_Z0[i] + D * kappa[i])
L_H(D) = sum_i in hot  exp(log_Z0[i] - D * kappa[i])
bound_cold = U_C(D) / (U_C(D) + L_H(D) + live_Z)
~~~

Lease còn hiệu lực khi bound_cold <= delta_cert, với delta_cert = 0.10.
Khi bound vượt ngưỡng, lease expire và bước đó được xem là cần full source
score refresh. Trong phase này, hệ thống chỉ ghi trace; không xoá hoặc thay đổi
KV cache.

### 3.3. Xử lý Qwen3 và GQA

Collector thực hiện:

1. bắt query sau q_proj, q_norm và RoPE bằng temporary forward hooks;
2. lấy key từ DynamicCache theo từng layer;
3. map query heads sang KV heads theo GQA;
4. tính source-block geometry vectorized;
5. tính live contribution ngoài source span;
6. ghi actual cold mass từ attention thật để audit certificate;
7. refresh trạng thái lease khi certificate expire.

Collector mặc định instrument toàn bộ self-attention layers của Qwen3.

## 4. Thành phần đã triển khai

| Thành phần | Vai trò |
|---|---|
| src/TrainingFree/lease.py | Log-domain certificate, hot/cold selection, LeaseState |
| src/TrainingFree/lease_collector.py | Q/K capture, DynamicCache, GQA, geometry và trace |
| src/TrainingFree/lease_evaluation.py | Per-sample metrics, aggregate metrics và kill gate |
| src/TrainingFree/schema.py | Schema recap.lease.trace.v1 và validation |
| src/TrainingFree/run.py | CLI --experiment lease, output và markdown report |
| scripts/modal_trainingfree.py | Modal image, checkpoint mount, venv và remote run |
| src/TrainingFree/tests/test_lease*.py | Unit/regression tests cho V2 |
| tests/test_trainingfree_lease_contract.py | Contract test cho runner |

Artifact V2 được ghi tại:

~~~text
outputs/recap_kv_v2/recap-v2-lease-pilot-qwen06b-a10g-rerun/
├── lease_trace.jsonl
├── lease_metrics.jsonl
├── lease_metrics.json
├── manifest.json
└── lease_report.md
~~~

## 5. Quy trình triển khai và kiểm chứng

### 5.1. TDD và kiểm tra cục bộ

Triển khai được thực hiện theo vòng RED-GREEN:

1. viết test cho certificate math trước khi có implementation;
2. viết test cho GQA mapping, cache extraction và block geometry;
3. thêm regression test cho partial final block;
4. thêm test evaluator, schema và runner contract;
5. sửa implementation đến khi test xanh;
6. compile toàn bộ module trước khi chạy Modal.

Kết quả verification cuối:

~~~text
38 passed in 2.91s
compileall: exit 0
trailing whitespace: none
~~~

Toàn bộ 40 trace pilot sau khi tải về cũng được validate bằng
validate_lease_trace_record: 40 records, 0 schema errors.

### 5.2. Các lỗi gặp phải và cách xử lý

#### Lỗi partial final block

Bản vectorized geometry đầu tiên xử lý sai indexing khi source length không
chia hết cho block size. Pilot đầu tiên sau tối ưu vectorization bị:

~~~text
recap-v2-lease-pilot-qwen06b-32: 40/40 IndexError
~~~

Regression test đã tái hiện lỗi với final block ngắn. Lỗi được sửa bằng cách
mask đúng các vị trí padding trên tensor key norm. Pilot chạy lại sau patch đạt
40/40 sample thành công. Lần chạy lỗi không được dùng trong số liệu khoa học.

#### Lỗi event accounting

Trace collector đời đầu lưu expired=True nhưng chưa đặt full_source_score=True
trên cùng expiry row. Evaluator được sửa để coi expired=True là refresh
fallback, nhờ đó refresh rate được recompute đúng từ lease_trace.jsonl. Báo cáo
dùng số recompute sau patch, không dùng refresh rate raw của lease_metrics.json
được tạo trước patch này.

#### Sai số số học bfloat16

Một bước có actual cold mass lớn hơn bound khoảng 2.2e-5, phù hợp với sai số
khi đọc Q/K ở bfloat16. Evaluator dùng tolerance 1e-4 cho phép so sánh
certificate; tolerance này được ghi rõ trong manifest/report, không che giấu
violation lớn.

### 5.3. Các lần chạy Modal

| Run | Mục đích | Kết quả | Diễn giải |
|---|---|---|---|
| recap-v2-lease-smoke-qwen06b | Smoke trên 1 sample | 1/1 OK | Runtime/schema xanh; scientific status inconclusive |
| recap-v2-lease-pilot-qwen06b | Pilot A100, 128-token plan | Bị dừng khi chờ GPU | Operational interruption, không phải scientific result |
| recap-v2-lease-pilot-qwen06b-32 | Pilot sau vectorization | 40/40 lỗi IndexError | Runtime failure; bị loại khỏi metric |
| recap-v2-lease-pilot-qwen06b-a10g-rerun | Pilot sau regression fix | 40/40 OK | Scientific pilot được dùng trong báo cáo |

### 5.4. Cấu hình pilot chính

Lệnh thực thi:

~~~bash
MODAL_GPU=A10G modal run scripts/modal_trainingfree.py \
  --experiment lease \
  --mode pilot \
  --max-new-tokens 32 \
  --run-id recap-v2-lease-pilot-qwen06b-a10g-rerun
~~~

Modal app: https://modal.com/apps/tdphuc-work/main/ap-H2cQoFCfppXivbP2djSdRg

| Thuộc tính | Giá trị |
|---|---|
| Target checkpoint | /opt/models/Qwen3-0.6B |
| Python | /mnt/fast-in/venv/bin/python |
| GPU thực tế | NVIDIA A10 |
| Dataset | GovReport, Multi-News |
| Documents | 20 + 20 |
| Block size | 128 token |
| delta_anchor | 0.05 |
| delta_cert | 0.10 |
| Max new tokens | 32 |
| Runtime | 1463.396 s |
| Decode steps trung bình | 31.1/document |
| Kết quả | 40/40 OK |

Giới hạn 32 token được chọn để kiểm tra headroom với full source-score
instrumentation. Đây chưa phải full 128-token study.

## 6. Kết quả thực nghiệm

### 6.1. Smoke

Smoke run trên Modal A100 request đạt 1/1 sample hợp lệ. Sample có:

- lease length: 1;
- certified cold fraction: 0.195876;
- violation rate: 0.0;
- mean bound: 0.944811;
- actual cold mass: 0.005449.

Smoke chỉ xác nhận runtime và schema. Vì không đạt tối thiểu hai dataset và 10
document/dataset, nó được gắn INCONCLUSIVE về mặt khoa học.

### 6.2. Aggregate pilot — 40 document

| Metric | Giá trị |
|---|---:|
| Decode steps trung bình | 31.100 |
| Valid steps trung bình | 2.475 |
| Lease count trung bình | 28.675 |
| Median lease length | 1.525 |
| Mean lease length | 1.558 |
| Certified cold fraction | 0.164768 |
| Certificate violation rate | 0.000000 |
| Mean bound | 0.918639 |
| Mean actual cold mass | 0.011961 |
| Mean bound tightness | 0.906678 |
| Refresh rate | 0.951643 |
| Source-score avoidance proxy | 0.048357 |

source-score avoidance proxy = 1 - refresh_rate. Đây là proxy của headroom,
chưa phải latency speedup đo bằng wall-clock hay kernel benchmark.

### 6.3. Kết quả theo dataset

| Dataset | N | Median lease | Cold fraction | Violation | Refresh rate | Score avoidance |
|---|---:|---:|---:|---:|---:|---:|
| GovReport | 20 | 0.000000 | 0.180093 | 0.000000 | 1.000000 | 0.000000 |
| Multi-News | 20 | 3.050000 | 0.149443 | 0.000000 | 0.903287 | 0.096713 |

GovReport gần như expire ngay sau anchor. Multi-News có lease dài hơn nhưng vẫn
chưa đạt ngưỡng lease 4 token.

### 6.4. Kill gate

| Điều kiện | Ngưỡng | Quan sát | Pass |
|---|---:|---:|:---:|
| Certificate violation | <= 0.001 | 0.000000 | Có |
| Certified cold fraction | >= 0.20 | 0.164768 | Không |
| Median lease length | >= 4 | 1.525000 | Không |
| Refresh saving | > 0 | 0.048357 | Có |

Chỉ 2/4 điều kiện đạt. Quyết định hợp lệ là:

~~~text
STOP_BEFORE_PHYSICAL
~~~

## 7. Đánh giá kết quả

### 7.1. CONFIRMED

1. V2 đã được triển khai end-to-end với Qwen3-0.6B, DynamicCache, GQA và
   tracing query/key trên Modal GPU.
2. Pilot chính hoàn tất 40/40 document không có runtime error hoặc
   inconclusive sample.
3. Trace schema hợp lệ cho toàn bộ 40 document.
4. Certificate không có violation vượt tolerance trong pilot cuối.
5. Có cold block khác rỗng và có source-score avoidance dương trên aggregate.

### 7.2. EXPLORATORY

1. Multi-News có lease ổn định hơn GovReport trong pilot này.
2. Actual cold mass thấp hơn bound rất nhiều, cho thấy certificate có thể còn
   cải thiện nếu denominator/lower bound được làm chặt.
3. Tolerance 1e-4 đủ loại sai số số học nhỏ trong trace bfloat16, nhưng cần
   kiểm tra lại bằng fp32 hoặc margin audit trước full study.

### 7.3. FAILED / INCOMPLETE

1. Kill gate không đạt ở cold fraction và lease length.
2. Refresh rate 95.16% khiến source-score avoidance chỉ 4.84%; chưa có bằng
   chứng systems value.
3. Chưa chạy full 128-token study, nhiều seed, checkpoint thứ hai hoặc
   confidence interval.
4. Chưa đo chất lượng summary, NLL/logit drift, latency, throughput hay VRAM
   sau physical KV modification.
5. Chưa có bằng chứng rằng V2 vượt baseline inference hoặc tạo speedup thật.

### 7.4. Highest verified rung

Highest verified empirical rung là R5 — short pilot trên real data và Modal
GPU. Artifact chứng minh gồm manifest.json, lease_trace.jsonl,
lease_metrics.jsonl và lease_report.md trong Modal Volume của run chính. Đây
chưa phải R6 full study.

## 8. Phân tích nguyên nhân certificate chưa đạt

Mean actual cold mass chỉ 0.011961, nhưng mean bound lên tới 0.918639. Điều
này cho thấy certificate an toàn nhưng ít hữu ích: nó expire trước khi cold
block thực sự có cơ hội trở nên quan trọng.

Các nguyên nhân kỹ thuật có khả năng cao:

1. kappa lấy max key norm theo block là upper bound tiện dụng nhưng lỏng;
2. query drift đang gộp semantic movement và positional/RoPE movement;
3. L_H chưa khai thác đủ cấu trúc phân bố của hot contribution;
4. geometry được gom qua nhiều layer/head nên mất cấu trúc head-specific;
5. block size 128 có thể quá lớn cho lifecycle của source evidence.

Đây là chẩn đoán từ trace, chưa phải chứng minh nhân quả.

## 9. Khuyến nghị thực nghiệm tiếp theo

Thực nghiệm tiếp theo nên tập trung vào một thay đổi duy nhất: certificate
tighter theo từng layer/head, giữ nguyên dataset, checkpoint và kill gate để
so sánh công bằng. Cụ thể:

1. không lấy mean geometry giữa các head khi cấp lease;
2. tách positional/RoPE drift khỏi semantic query drift;
3. đánh giá block size 32/64/128;
4. ghi scatter bound-vs-actual theo từng layer/head;
5. chỉ khi median lease và cold fraction cùng vượt gate mới mở physical
   retention/quantization experiment.

Mục tiêu gần nhất là một R5 follow-up có certificate tight hơn. Full R6 study
chỉ nên chạy sau khi follow-up đó đạt cả hai điều kiện:
median lease >= 4 và certified cold fraction >= 0.20.

## 10. Tài liệu và artifact liên quan

- Báo cáo tóm tắt: 2026-09-18-recap-kv-v2-results.md
- Certificate math: src/TrainingFree/lease.py
- Collector: src/TrainingFree/lease_collector.py
- Evaluator: src/TrainingFree/lease_evaluation.py
- Runner: src/TrainingFree/run.py
- Modal launcher: scripts/modal_trainingfree.py

**Kết luận chuẩn:** Supported: implementation và certificate instrumentation
chạy ổn định qua R5 pilot. Not supported yet: V2 tạo ra lợi ích inference
thực tế hoặc đủ an toàn/hiệu quả để mở physical KV tiering. Verified through
R5 (short pilot); not yet verified by full study.
