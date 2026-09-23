# E43 — Temporal Support Reuse: triển khai và kết quả

Ngày: 2026-09-23
Model: Qwen3-0.6B
Layer: 27, layer cuối
Dữ liệu: 20 gov_report + 20 multi_news
Decode: greedy, seed 42, tối đa 32 token
GPU: NVIDIA A10 theo manifest Modal

## 1. Kết luận ngắn

Đã triển khai và chạy các kiểm tra E43-0 đến E43-5:

- GQA-aware oracle union;
- temporal support transfer với lag 1/2/4/8/16;
- fixed source budgets 10/20/30/40%;
- adaptive per-head K95 với alpha 1.0/1.25/1.5;
- periodic refresh R=2/4/8/16;
- blockification block 16/32/64;
- recurrent simulation không dùng current-token oracle ngoài refresh.

Kết quả **không tìm thấy deployable candidate** theo gate đã khóa:

source cost < 50%, mean missed mass ≤ 1%, P99 missed mass ≤ 5%.

Điểm gần nhất là fixed budget 10%, refresh R=8, block 16:

| Candidate | Source cost | Mean missed | P99 missed | Kết quả |
|---|---:|---:|---:|---|
| R8 + C16 + F0.1 | 52.55% | 0.448% | 12.89% | FAIL cost và tail |

Adaptive K95 an toàn hơn về mean mass nhưng working cache quá lớn sau GQA/blockification; candidate rẻ nhất trong full adaptive pilot có source cost khoảng 77.39%.

Vì vậy chưa chạy E43-6 masked generation, chưa physicalize KV, và nên đóng source-selection branch như kế hoạch đã đề xuất.

## 2. Thiết kế đã triển khai

### 2.1 Temporal analyzer

File mới:

- src/TrainingFree/temporal.py
- src/TrainingFree/temporal_collector.py
- scripts/analyze_trainingfree_e43.py

Analyzer giữ state theo document:

1. Ở mỗi refresh, đọc exact current source attention.
2. Chọn support theo fixed fraction hoặc alpha × K95.
3. Map 16 query heads xuống 8 KV heads.
4. Union support ở KV-head granularity.
5. Round support lên contiguous blocks.
6. Ở các bước non-refresh, chỉ dùng support của refresh trước.
7. Dùng exact current attention chỉ để audit missed mass, không dùng để cập nhật policy.

Source cost của recurrent simulation được tính:

- refresh step: cost = 1.0, đại diện cho dense source attention;
- non-refresh step: cost = GQA working-cache fraction sau blockification.

### 2.2 Runtime và schema

Đã thêm:

- experiment mode temporal vào src/TrainingFree/run.py;
- các CLI options cho lag, budget, alpha, refresh interval, block size;
- schema recap.e43.temporal.trace.v1;
- Modal output root outputs/recap_kv_e43;
- report analyzer JSON/CSV/Markdown;
- fixed-budget recurrent branch bên cạnh adaptive branch.

Không thay đổi model KV cache và không thêm physical executor.

## 3. Kiểm thử trước GPU

TDD cycle:

- RED: test fail vì temporal module chưa tồn tại.
- GREEN: implement support selection, GQA union, K95, lag transfer và recurrent block cache.
- Kết quả temporal/schema/analyzer: 13 test pass sau các vòng sửa.
- Smoke/integration CLI và compileall: đạt.

Smoke Modal đầu tiên phát hiện schema mismatch do temporal-only trace bị validate bằng V3 hierarchy schema. Đã tách validator riêng; smoke v2 hoàn tất 1/1 mẫu, 0 lỗi, thời gian 64.4 giây.

## 4. Các lần chạy Modal

### 4.1 Smoke

| Run | Kết quả |
|---|---|
| e43-temporal-smoke-qwen06b-a10g | 1 sample lỗi schema, đã sửa; không dùng làm scientific result |
| e43-temporal-smoke-qwen06b-a10g-v2 | 1/1, 0 lỗi, trace hợp lệ, 64.4 s |

### 4.2 Full adaptive pilot

Run: e43-temporal-pilot-qwen06b-a10g-20260923
App: <https://modal.com/apps/tdphuc-work/main/ap-yMvf3zX8TZ34EN7PSQMoEH>

- 40/40 sample;
- 20 gov_report, 20 multi_news;
- 0 error;
- 2,380.54 giây;
- alpha = 1.0/1.25/1.5;
- refresh = 2/4/8/16;
- block = 16/32/64.

### 4.3 Fixed-budget targeted run

Run: e43-temporal-fixed10x2-qwen06b-a10g-20260923
App: <https://modal.com/apps/tdphuc-work/main/ap-9r1JRtUT7Q3eYLX1CEVXzz>

- 20/20 sample;
- 10 gov_report, 10 multi_news;
- 0 error;
- 696.18 giây;
- fixed budgets = 10/20/30/40%;
- refresh = 4/8;
- block = 16/32/64;
- adaptive branch tắt để tránh tính trùng với full adaptive pilot.

Một run fixed sweep rộng hơn đã bị dừng sau sample đầu vì cấu hình tính đồng thời 48 fixed + adaptive candidates làm runtime tăng mạnh. Đây là operational interruption, không dùng partial result trong aggregate.

## 5. E43-0 — GQA-aware oracle union

Ở full adaptive pilot, fixed query-head source budgets được union theo KV head:

| Query budget | GQA union expansion | Mean missed mass | P99 missed mass |
|---:|---:|---:|---:|
| 10% | 14.08% | 1.033% | 18.78% |
| 20% | 26.33% | 0.522% | 10.96% |
| 30% | 37.69% | 0.290% | 6.58% |
| 40% | 48.45% | 0.165% | 3.86% |

GQA union không làm 20% query budget tăng lên 60–70%; nó tăng lên khoảng 26.3%. Đây là tín hiệu tốt cho physicalizability ở mức oracle. Tuy nhiên tail risk vẫn cao: tại 20%, P99 missed mass là 10.96%.

## 6. E43-1/E43-2 — Temporal transfer và refresh

Với fixed budget 20%:

| Lag | GQA support fraction | Mean missed mass | P99 missed mass |
|---:|---:|---:|---:|
| 1 | 26.32% | 0.710% | 23.02% |
| 2 | 26.32% | 0.824% | 21.87% |
| 4 | 26.32% | 0.959% | 24.19% |
| 8 | 26.32% | 0.969% | 20.29% |
| 16 | 26.32% | 1.109% | 23.90% |

Kết luận: support có temporal locality ở mean, nhưng tail không ổn định. Ngay cả lag 1 đã vượt P99 gate rõ rệt; refresh thường xuyên hơn không giải quyết được tail risk.

## 7. E43-3 — Adaptive K95

Adaptive K95 không giữ working support nhỏ:

| Recurrent candidate | Source cost | Working cost | Mean missed | P99 missed |
|---|---:|---:|---:|---:|
| R16 + C16 + alpha 1.0 | 77.39% | 75.85% | 0.041% | 7.33% |
| R8 + C16 + alpha 1.0 | 78.01% | 74.84% | 0.040% | 5.31% |
| R4 + C16 + alpha 1.0 | 80.69% | 74.21% | 0.038% | 6.66% |
| R8 + C16 + alpha 1.5 | 85.13% | 82.98% | 0.013% | 2.84% |

Alpha lớn hơn giảm tail risk nhưng làm source cost tăng. Alpha 1.0 có cost thấp nhất nhưng vẫn không đạt cost gate và còn fail P99 ở aggregate.

Task asymmetry rõ:

| Candidate | GovReport cost / P99 | Multi-News cost / P99 |
|---|---:|---:|
| R8 + C16 + alpha 1.0 | 66.02% / 5.31% | 90.16% / 3.94% |
| R8 + C16 + alpha 1.5 | 74.93% / 2.84% | 95.46% / 1.83% |

Multi-News có working cache lớn hơn nhiều; đây là nguyên nhân chính khiến aggregate source cost không đạt.

## 8. E43-4/E43-5 — Fixed-budget block cache và recurrent simulation

Targeted 10+10 run:

| Candidate | Source cost | Working cost | Mean missed | P99 missed | Gate |
|---|---:|---:|---:|---:|---|
| R8 + C16 + F0.1 | 52.55% | 45.65% | 0.448% | 12.89% | FAIL |
| R4 + C16 + F0.1 | 59.09% | 45.32% | 0.423% | 12.89% | FAIL |
| R8 + C16 + F0.2 | 68.71% | 64.16% | 0.154% | 6.28% | FAIL |
| R8 + C16 + F0.3 | 79.42% | 76.42% | 0.059% | 3.63% | FAIL cost |

Block size effect tại R8/F0.1:

| Block size | Source cost | Mean missed | P99 missed |
|---:|---:|---:|---:|
| 16 | 52.55% | 0.448% | 12.89% |
| 32 | 61.89% | 0.275% | 10.03% |
| 64 | 70.61% | 0.154% | 7.69% |

Block 16 là operating point tốt nhất, nhưng vẫn không đạt. Tăng block size giảm risk nhẹ nhờ giữ thêm token, đổi lại làm cost tăng rõ rệt.

Theo dataset, candidate tốt nhất R8/C16/F0.1 có:

- GovReport: cost 45.97%, mean missed 0.404%, P99 12.89%;
- Multi-News: cost 59.24%, mean missed 0.492%, P99 12.07%.

GovReport riêng đạt cost gate nhưng fail tail. Multi-News fail cả cost và tail. Vì vậy không được promote candidate dựa trên GovReport riêng.

## 9. Quyết định gate

Gate khóa trước:

1. average source cost ratio < 50%;
2. mean missed mass ≤ 1%;
3. P99 missed mass ≤ 5%.

Không có candidate nào đạt cả ba:

- Adaptive K95: mean risk đạt, nhưng cost 77–85% và/hoặc P99 fail.
- Fixed F0.1: cost gần nhất 52.55%, nhưng P99 12.89%.
- Fixed F0.2/F0.3: tail cải thiện dần nhưng cost tăng lên 68.71–79.42%.
- Larger block: không tạo gain, chỉ làm cost tăng.

Do đó:

STOP_SOURCE_SELECTION

Không chạy masked generation E43-6 vì không có candidate qua recurrent gate. Không chạy physical KV mutation.

## 10. Artifact

Metrics đã lưu trong:

- outputs/recap_kv_e43/e43-temporal-pilot-qwen06b-a10g-20260923/analysis/e43_metrics.json
- outputs/recap_kv_e43/e43-temporal-pilot-qwen06b-a10g-20260923/analysis/e43_recurrent.csv
- outputs/recap_kv_e43/e43-temporal-pilot-qwen06b-a10g-20260923/analysis/e43_report.md
- outputs/recap_kv_e43/e43-temporal-fixed10x2-qwen06b-a10g-20260923/analysis/e43_metrics.json
- outputs/recap_kv_e43/e43-temporal-fixed10x2-qwen06b-a10g-20260923/analysis/e43_recurrent.csv
- outputs/recap_kv_e43/e43-temporal-fixed10x2-qwen06b-a10g-20260923/analysis/e43_report.md

Raw trace:

- Modal Volume: outputs/recap_kv_e43/e43-temporal-pilot-qwen06b-a10g-20260923/temporal_trace.jsonl
- Modal Volume: outputs/recap_kv_e43/e43-temporal-fixed10x2-qwen06b-a10g-20260923/temporal_trace.jsonl
- Local temporary copies: /tmp/e43-temporal-pilot-final/ và /tmp/e43-temporal-fixed-targeted-final/

## 11. Review kết luận

### CONFIRMED

- Temporal analyzer và GQA-aware recurrent audit đã được triển khai, test và chạy trên GPU.
- Full adaptive pilot đạt 40/40 mẫu, 0 lỗi; fixed targeted run đạt 20/20 mẫu, 0 lỗi.
- GQA union của 20% query support là khoảng 26.3%, không phải 60–70%.
- Temporal support reuse có mean missed mass thấp, nhưng P99 tail cao ở fixed low-budget support.
- Không có candidate đạt đồng thời source cost <50%, mean missed ≤1% và P99 missed ≤5%.

### EXPLORATORY

- GovReport có locality/cost tốt hơn Multi-News.
- Block 16 là lựa chọn tốt nhất trong block sweep.
- R8/F0.1 là operating point gần cost gate nhất, nhưng tail risk còn quá cao.
- Adaptive K95 là safety-oriented nhưng không tạo working cache nhỏ.

Các kết quả trên là dev/search cohort, layer 27, tối đa 32 token; chưa phải holdout hay E2E speedup.

### FAILED / INCOMPLETE

- Đây là scientific gate failure của source-selection policy, không phải GPU/runtime failure.
- Một targeted run rộng bị dừng sau 1 sample vì runtime tăng do tính dư nhiều cấu hình; partial trace không đưa vào aggregate.
- E43-6 masked generation chưa chạy vì không có candidate pass gate.
- Chưa physicalize KV, chưa đo TPOT, throughput, VRAM, gather/kernel overhead hoặc ROUGE/NLL.

### HIGHEST VERIFIED RUNG

Verified through R5 short pilot plus trace-level decision memo: full adaptive 40-document Modal pilot và fixed 20-document targeted pilot đều hoàn tất với schema-valid traces. Chưa verified ở masked generation, physical executor, E2E systems benchmark hoặc untouched holdout.

### EVIDENCE GAPS

1. Chưa nối missed attention mass với logit agreement, NLL và summary quality.
2. Source cost hiện là analytical recurrent cost; chưa đo kernel/gather/compaction overhead.
3. Chỉ đo layer 27 và tối đa 32 decode tokens.
4. Fixed recurrent run chỉ có 10+10 docs; đủ targeted coverage nhưng nhỏ hơn adaptive 40-doc pilot.
5. Chưa có untouched holdout hoặc model/GPU transfer.

### RECOMMENDED NEXT

Không tiếp tục mở rộng source-selection branch, không xây score sketch/certificate mới và không chạy physical V3. Theo kế hoạch đã đề xuất, pivot sang **KV representation/compression** (mixed precision hoặc head/layer-specific KV quantization) là bước tiếp theo có xác suất tạo gain systems cao hơn selection khi model cần truy cập source rộng.

Supported: temporal support có locality trung bình và GQA union không phá hỏng oracle headroom hoàn toàn.

Not supported: một policy Refresh → Compact → Reuse hiện tại tạo được gain deployable dưới gate cost/risk.
