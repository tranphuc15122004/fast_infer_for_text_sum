# Tổng hợp metric benchmark B200: DSpark, DFlash, Eagle3, Domino

> Báo cáo tổng hợp từ hai thư mục run đã cung cấp. Số liệu gốc được đọc từ `run_report.json`, `events.jsonl`, `samples.jsonl`, `excluded_samples.jsonl`, `report_vi.md` và `console.log`; bảng latency theo dataset và percentile được tính lại từ event log. Acceptance length theo dataset được ước lượng từ cửa sổ metric trong console log và được đánh dấu riêng vì không có số đo theo từng request.

## 1. Kết luận nhanh

- Cả hai run có `status=completed`, `execution_complete=true`, `execution_pass=true`; mỗi run hoàn tất 446 mẫu đủ điều kiện × 3 method = 1.338 request, toàn bộ request có `status=success`.
- `quality_pass=true` ở đây có nghĩa 446/446 output mỗi method qua guard và không có cờ lặp; đây không phải kết luận về chất lượng task-aware.
- `correctness_pass=false` ở cả hai run. Parity greedy của speculative methods chỉ 304–318/446 mẫu; vì vậy các số DSR/ESR là quan sát tốc độ khi output khác greedy vanilla, chưa chứng minh speedup giữ nguyên correctness.
- Mỗi dataset có 100 mẫu nguồn, tổng 500; 54 mẫu (10,8%) bị loại theo `input_limit=12224`, còn 446 mẫu được chạy. Hai run dùng đúng cùng sample manifest và cùng tập bị loại.
- Một cờ cần kiểm tra: parity theo từng sample của DFlash (GPU0), Eagle3 và Domino (GPU1) giống hệt nhau trong `run_report.json` trên cả 446 ID, gồm trạng thái exact-match và LCS. Artifact này chưa đủ để xác định nguyên nhân.

## 2. Phạm vi, môi trường và cấu hình

| Thuộc tính | GPU0 — DSpark/DFlash | GPU1 — Eagle3/Domino |
|---|---|---|
| Run ID | b200-gpu0-dspark-dflash | b200-gpu1-eagle3-domino |
| Model đích | /workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B | /workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B |
| GPU / thiết bị được chọn | NVIDIA B200; CUDA_VISIBLE_DEVICES=0 | NVIDIA B200; CUDA_VISIBLE_DEVICES=1 |
| Chế độ / batch | full / 1 | full / 1 |
| Dtype / seed / temperature | bfloat16 / 42 / 0.0 | bfloat16 / 42 / 0.0 |
| max_model_len / max_new_tokens | 12.288 / 64 token | 12.288 / 64 token |
| gpu_memory_utilization | 0,88 | 0,88 |
| Prefix caching / enforce eager | tắt / tắt | tắt / tắt |
| Warmup | warmup_per_sample=true | warmup_per_sample=true |
| Python / PyTorch / Transformers / vLLM | 3.12.3 / 2.13.0+cu130 / 5.10.4 / 0.30.0 | 3.12.3 / 2.13.0+cu130 / 5.10.4 / 0.30.0 |
| CUDA runtime / driver | 13.0 / 580.105.08 | 13.0 / 580.105.08 |
| Bắt đầu UTC | 2026-10-01 19:35:37.501 | 2026-10-01 19:35:50.840 |
| Kết thúc evaluation UTC | 2026-10-01 19:53:37.734 | 2026-10-01 19:57:29.852 |
| Thời lượng evaluation report | 1080.2 giây | 1299.0 giây |
| Mẫu được infer / bị loại | 446 / 54 | 446 / 54 |
| Cờ tổng thể | execution PASS; quality guard PASS; correctness FAIL | execution PASS; quality guard PASS; correctness FAIL |

Model đích là Qwen3-4B, kiến trúc Qwen3, BF16, 36 layer, hidden size 2.560, 32 attention heads và 8 KV heads; `max_position_embeddings` trong config model là 40.960 nhưng engine giới hạn `max_model_len=12.288`. Các draft checkpoint và số speculative token cấu hình: DSpark (`Dspark-Qwen3-4B-b7`, 7); DFlash (`Qwen3-4B-DFlash-b16`, 16); Eagle3 (`Qwen3-4B_eagle3`, 16); Domino (`Qwen3-4B-Domino-b16`, 16).

`max_samples_requested=0`, `max_samples_effective=0`, `max_input_tokens=0` trong run config biểu thị không đặt cap theo số mẫu / cap riêng từ config. Tuy nhiên, 54 mẫu vẫn bị loại khi input vượt phần ngữ cảnh khả dụng của engine (`max_model_len - max_new_tokens = 12.224`).

SHA-256 data file chung: `6a03a49f4a92aa72e8ddf41727e2ab356758019b7a9aac4135c62d9827a0cda5` (`longbench_500_combined.jsonl`).

## 3. Độ phủ năm dataset và độ dài input

| Dataset | Task | Nguồn | Đã chạy | Bị loại | Lý do loại | Input token min / median / p95 / max (đã chạy) | Input token min–max (bị loại) |
|---|---|---|---|---|---|---|---|
| gov_report | Tóm tắt | 100 | 91 | 9 | input_limit: 9 | 2092 / 7381 / 11762 / 12009 | 12259–13916 |
| multi_news | Tóm tắt | 100 | 100 | 0 | — | 177 / 2088 / 5530 / 11934 | — |
| qmsum | Tóm tắt | 100 | 70 | 30 | input_limit: 30 | 2644 / 9311 / 12125 / 12130 | 12530–13367 |
| lcc | Hoàn thành code | 100 | 99 | 1 | input_limit: 1 | 1016 / 2245 / 6338 / 8815 | 12241–12241 |
| repobench-p | Hoàn thành code | 100 | 86 | 14 | input_limit: 14 | 2747 / 6121 / 11125 / 11959 | 12279–13786 |

Trong tập đã chạy, mean input tokens lần lượt được tính từ request event và giống nhau giữa các method:

| Dataset | n chạy | Mean input tokens | Median | p95 | Max |
|---|---|---|---|---|---|
| gov_report | 91 | 7433.8 | 7381 | 11762 | 12009 |
| multi_news | 100 | 2518.8 | 2088 | 5530 | 11934 |
| qmsum | 70 | 8586.7 | 9311 | 12125 | 12130 |
| lcc | 99 | 2831.6 | 2245 | 6338 | 8815 |
| repobench-p | 86 | 6691.5 | 6121 | 11125 | 11959 |

`gov_report`, `multi_news`, `qmsum` có task type `summarization`; `lcc`, `repobench-p` là `code_completion`. Vì vậy cần đánh giá theo hai họ metric khác nhau.

## 4. Metric tổng hợp theo method

Giá trị trong bảng này được chép từ `method_metrics` / `parity` của mỗi `run_report.json`. ROUGE là giá trị gộp toàn bộ 446 mẫu của từng run, gồm cả hai dataset code completion; xem hạn chế ở mục 10.

| Run | Method | Success / n | Guard hợp lệ | Lặp | ROUGE-1 | ROUGE-2 | ROUGE-L | Mean prefill ms | Mean TPOT ms | Mean throughput tok/s* | DSR | ESR | Exact token match vs vanilla | Mean token LCS vs vanilla |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPU0 | vanilla_vllm | 446/446 | 446/446 (100.0%) | 0 | 0.1143 | 0.0302 | 0.0744 | 44.817 | 3.113 | 321.2 | 1.000 | 1.000 | 446/446 (100.0%) | 100.0% |
| GPU0 | dspark | 446/446 | 446/446 (100.0%) | 0 | 0.1143 | 0.0303 | 0.0751 | 44.948 | 1.413 | 707.5 | 2.203 | 1.803 | 304/446 (68.2%) | 92.8% |
| GPU0 | dflash | 446/446 | 446/446 (100.0%) | 0 | 0.1155 | 0.0310 | 0.0759 | 45.139 | 1.843 | 542.5 | 1.689 | 1.499 | 318/446 (71.3%) | 93.3% |
| GPU1 | vanilla_vllm | 446/446 | 446/446 (100.0%) | 0 | 0.1143 | 0.0302 | 0.0744 | 45.714 | 3.122 | 320.4 | 1.000 | 1.000 | 446/446 (100.0%) | 100.0% |
| GPU1 | eagle3 | 446/446 | 446/446 (100.0%) | 0 | 0.1155 | 0.0310 | 0.0759 | 47.088 | 2.935 | 340.8 | 1.064 | 1.051 | 318/446 (71.3%) | 93.3% |
| GPU1 | domino | 446/446 | 446/446 (100.0%) | 0 | 0.1155 | 0.0310 | 0.0759 | 46.270 | 4.959 | 201.6 | 0.629 | 0.676 | 318/446 (71.3%) | 93.3% |

`* Mean throughput tok/s` là trường được report cung cấp; report không ghi công thức của trường này và event log giữ trong thư mục không có `decode_ms` để tái dựng chính xác. Không đồng nhất trường này với output tokens / end-to-end latency tính từ event log.

### Metric theo dataset: exact match và LCS với vanilla

| Dataset | n | GPU0 DSpark exact | GPU0 DSpark LCS | GPU0 DFlash exact | GPU0 DFlash LCS | GPU1 Eagle3 exact | GPU1 Eagle3 LCS | GPU1 Domino exact | GPU1 Domino LCS |
|---|---|---|---|---|---|---|---|---|---|
| gov_report | 91 | 67/91 (73.6%) | 94.9% | 66/91 (72.5%) | 95.1% | 66/91 (72.5%) | 95.1% | 66/91 (72.5%) | 95.1% |
| multi_news | 100 | 64/100 (64.0%) | 94.5% | 70/100 (70.0%) | 95.5% | 70/100 (70.0%) | 95.5% | 70/100 (70.0%) | 95.5% |
| qmsum | 70 | 51/70 (72.9%) | 93.2% | 55/70 (78.6%) | 93.8% | 55/70 (78.6%) | 93.8% | 55/70 (78.6%) | 93.8% |
| lcc | 99 | 61/99 (61.6%) | 89.7% | 67/99 (67.7%) | 90.8% | 67/99 (67.7%) | 90.8% | 67/99 (67.7%) | 90.8% |
| repobench-p | 86 | 61/86 (70.9%) | 91.6% | 60/86 (69.8%) | 91.2% | 60/86 (69.8%) | 91.2% | 60/86 (69.8%) | 91.2% |

Exact match ở đây là so sánh output token IDs của method với greedy `vanilla_vllm`, không phải exact match với reference. Vanilla tự so với chính nó đạt 446/446 và LCS 100%.

### DSR / ESR theo dataset

Các hệ số dưới đây tính lại từ `request_finished` event, ghép cùng sample ID. DSR là tỷ số mean TPOT; ESR dùng mean prefill và mean số output token ghép cặp nhỏ hơn theo công thức trong report. Vanilla làm mốc 1,0.

#### GPU0 — b200-gpu0-dspark-dflash

| Dataset | n | Vanilla mean TPOT ms | dspark DSR | dspark ESR | dflash DSR | dflash ESR |
|---|---|---|---|---|---|---|
| gov_report | 91 | 3.222 | 2.216 | 1.727 | 1.594 | 1.400 |
| multi_news | 100 | 2.958 | 2.300 | 2.031 | 1.830 | 1.687 |
| qmsum | 70 | 3.291 | 1.868 | 1.523 | 1.340 | 1.231 |
| lcc | 99 | 2.973 | 2.396 | 2.081 | 2.023 | 1.820 |
| repobench-p | 86 | 3.195 | 2.228 | 1.767 | 1.728 | 1.497 |

#### GPU1 — b200-gpu1-eagle3-domino

| Dataset | n | Vanilla mean TPOT ms | eagle3 DSR | eagle3 ESR | domino DSR | domino ESR |
|---|---|---|---|---|---|---|
| gov_report | 91 | 3.230 | 1.139 | 1.103 | 0.641 | 0.700 |
| multi_news | 100 | 2.967 | 1.060 | 1.054 | 0.633 | 0.659 |
| qmsum | 70 | 3.293 | 1.024 | 1.018 | 0.637 | 0.705 |
| lcc | 99 | 2.983 | 1.062 | 1.055 | 0.613 | 0.640 |
| repobench-p | 86 | 3.207 | 1.030 | 1.023 | 0.625 | 0.679 |

### Latency trung bình theo dataset và method

Đơn vị: milliseconds cho prefill, TPOT và end-to-end; input/output là token. Các số này tính lại từ event log, không phải ROUGE/output quality. P50/p95 dùng nearest-rank (`ceil(p × n)`).

#### GPU0 — mean theo dataset

| Dataset | Method | n | Mean input tok | Mean output tok | Prefill mean ms | TPOT mean ms | E2E mean ms |
|---|---|---|---|---|---|---|---|
| gov_report | vanilla_vllm | 91 | 7433.8 | 64.00 | 62.55 | 3.222 | 265.56 |
| gov_report | dspark | 91 | 7433.8 | 64.00 | 62.44 | 1.454 | 154.07 |
| gov_report | dflash | 91 | 7433.8 | 64.00 | 62.58 | 2.022 | 189.97 |
| multi_news | vanilla_vllm | 100 | 2518.8 | 64.00 | 21.45 | 2.958 | 207.78 |
| multi_news | dspark | 100 | 2518.8 | 64.00 | 22.23 | 1.286 | 103.27 |
| multi_news | dflash | 100 | 2518.8 | 64.00 | 22.37 | 1.617 | 124.22 |
| qmsum | vanilla_vllm | 70 | 8586.7 | 62.87 | 72.94 | 3.291 | 276.47 |
| qmsum | dspark | 70 | 8586.7 | 62.73 | 72.75 | 1.762 | 181.35 |
| qmsum | dflash | 70 | 8586.7 | 62.74 | 72.90 | 2.456 | 224.55 |
| lcc | vanilla_vllm | 99 | 2831.6 | 64.00 | 23.21 | 2.973 | 210.51 |
| lcc | dspark | 99 | 2831.6 | 64.00 | 23.36 | 1.241 | 101.52 |
| lcc | dflash | 99 | 2831.6 | 64.00 | 23.61 | 1.470 | 116.21 |
| repobench-p | vanilla_vllm | 86 | 6691.5 | 64.00 | 55.21 | 3.195 | 256.52 |
| repobench-p | dspark | 86 | 6691.5 | 64.00 | 55.09 | 1.434 | 145.44 |
| repobench-p | dflash | 86 | 6691.5 | 64.00 | 55.34 | 1.849 | 171.83 |

#### GPU0 — p50 / p95 theo dataset

| Dataset | Method | n | Prefill p50/p95 ms | TPOT p50/p95 ms | E2E p50/p95 ms |
|---|---|---|---|---|---|
| gov_report | vanilla_vllm | 91 | 59.13/105.08 | 3.232/3.388 | 263.09/318.14 |
| gov_report | dspark | 91 | 58.81/106.15 | 1.444/1.874 | 150.96/211.09 |
| gov_report | dflash | 91 | 58.72/106.97 | 2.075/2.549 | 190.61/258.52 |
| multi_news | vanilla_vllm | 100 | 17.52/42.90 | 2.923/3.185 | 201.74/243.02 |
| multi_news | dspark | 100 | 17.74/42.40 | 1.247/1.724 | 96.70/150.13 |
| multi_news | dflash | 100 | 17.38/43.02 | 1.593/2.208 | 119.10/182.54 |
| qmsum | vanilla_vllm | 70 | 78.53/111.21 | 3.293/3.467 | 278.37/315.21 |
| qmsum | dspark | 70 | 77.32/109.30 | 1.752/2.099 | 187.70/230.97 |
| qmsum | dflash | 70 | 78.78/110.80 | 2.470/2.836 | 229.28/266.03 |
| lcc | vanilla_vllm | 99 | 19.58/49.65 | 2.931/3.209 | 203.41/251.00 |
| lcc | dspark | 99 | 18.82/48.85 | 1.222/1.678 | 99.22/151.62 |
| lcc | dflash | 99 | 18.21/49.85 | 1.414/2.171 | 108.10/179.59 |
| repobench-p | vanilla_vllm | 86 | 46.86/97.79 | 3.201/3.365 | 247.98/306.70 |
| repobench-p | dspark | 86 | 47.08/99.48 | 1.411/1.817 | 137.99/193.67 |
| repobench-p | dflash | 86 | 47.20/99.99 | 1.856/2.352 | 170.62/233.19 |

#### GPU1 — mean theo dataset

| Dataset | Method | n | Mean input tok | Mean output tok | Prefill mean ms | TPOT mean ms | E2E mean ms |
|---|---|---|---|---|---|---|---|
| gov_report | vanilla_vllm | 91 | 7433.8 | 64.00 | 63.52 | 3.230 | 267.01 |
| gov_report | eagle3 | 91 | 7433.8 | 64.00 | 63.61 | 2.836 | 242.31 |
| gov_report | domino | 91 | 7433.8 | 64.00 | 63.62 | 5.037 | 380.94 |
| multi_news | vanilla_vllm | 100 | 2518.8 | 64.00 | 22.41 | 2.967 | 209.32 |
| multi_news | eagle3 | 100 | 2518.8 | 64.00 | 25.50 | 2.797 | 201.75 |
| multi_news | domino | 100 | 2518.8 | 64.00 | 23.87 | 4.685 | 319.06 |
| qmsum | vanilla_vllm | 70 | 8586.7 | 62.87 | 74.16 | 3.293 | 277.86 |
| qmsum | eagle3 | 70 | 8586.7 | 62.74 | 74.37 | 3.216 | 272.97 |
| qmsum | domino | 70 | 8586.7 | 62.74 | 74.16 | 5.168 | 393.46 |
| lcc | vanilla_vllm | 99 | 2831.6 | 64.00 | 23.73 | 2.983 | 211.68 |
| lcc | eagle3 | 99 | 2831.6 | 64.00 | 26.64 | 2.808 | 203.57 |
| lcc | domino | 99 | 2831.6 | 64.00 | 24.67 | 4.866 | 331.24 |
| repobench-p | vanilla_vllm | 86 | 6691.5 | 64.00 | 56.12 | 3.207 | 258.15 |
| repobench-p | eagle3 | 86 | 6691.5 | 64.00 | 56.05 | 3.114 | 252.23 |
| repobench-p | domino | 86 | 6691.5 | 64.00 | 56.12 | 5.134 | 379.56 |

#### GPU1 — p50 / p95 theo dataset

| Dataset | Method | n | Prefill p50/p95 ms | TPOT p50/p95 ms | E2E p50/p95 ms |
|---|---|---|---|---|---|
| gov_report | vanilla_vllm | 91 | 60.68/108.32 | 3.263/3.389 | 269.27/319.64 |
| gov_report | eagle3 | 91 | 59.79/108.42 | 2.815/3.301 | 243.35/302.69 |
| gov_report | domino | 91 | 60.85/107.97 | 5.069/5.487 | 382.11/443.77 |
| multi_news | vanilla_vllm | 100 | 18.14/42.87 | 2.925/3.236 | 202.63/245.97 |
| multi_news | eagle3 | 100 | 21.18/43.83 | 2.780/3.433 | 198.94/252.41 |
| multi_news | domino | 100 | 18.39/44.07 | 4.737/5.125 | 320.17/366.22 |
| qmsum | vanilla_vllm | 70 | 80.11/112.12 | 3.303/3.410 | 280.69/315.53 |
| qmsum | eagle3 | 70 | 80.29/110.23 | 3.179/3.611 | 273.57/317.45 |
| qmsum | domino | 70 | 80.37/109.95 | 5.224/5.461 | 395.41/444.03 |
| lcc | vanilla_vllm | 99 | 18.71/51.04 | 2.936/3.260 | 203.99/255.61 |
| lcc | eagle3 | 99 | 21.72/51.10 | 2.743/3.666 | 198.97/264.63 |
| lcc | domino | 99 | 18.84/50.32 | 4.915/5.412 | 330.05/392.13 |
| repobench-p | vanilla_vllm | 86 | 48.08/100.23 | 3.225/3.379 | 250.14/311.51 |
| repobench-p | eagle3 | 86 | 47.68/102.35 | 3.146/3.514 | 250.28/303.97 |
| repobench-p | domino | 86 | 47.81/101.81 | 5.158/5.503 | 377.10/438.80 |

### Draft proposal / acceptance metrics

| Run | Method | Đơn vị proposal theo report | Spec metrics available n | Tổng proposal units | Mean proposal / request | Acceptance valid n | Mean acceptance rate | Mean acceptance % | Mean accept length | Accepted units total |
|---|---|---|---|---|---|---|---|---|---|---|
| GPU0 | vanilla_vllm | — | 0 | — | — | 0 | — | — | — | — |
| GPU0 | dspark | draft_tokens | 446 | 66010 | 148.00 | 0 | — | — | — | — |
| GPU0 | dflash | draft_tokens | 446 | 194928 | 437.06 | 0 | — | — | — | — |
| GPU1 | vanilla_vllm | — | 0 | — | — | 0 | — | — | — | — |
| GPU1 | eagle3 | draft_tree_nodes | 446 | 218016 | 488.83 | 0 | — | — | — | — |
| GPU1 | domino | draft_tokens | 446 | 421584 | 945.26 | 0 | — | — | — | — |

Trong metric có cấu trúc của `run_report.json` và các event `request_finished`, acceptance rate / acceptance percent / average accept length / accepted draft tokens đều null (valid count = 0). `console.log` lại có các dòng `SpecDecoding metrics` với acceptance length, draft acceptance rate và token counters được gom theo cửa sổ runtime; bảng kế tiếp khai thác acceptance length từ các dòng này. Chỉ có tổng proposal per-request trong report là trực tiếp quan sát được. `draft_proposal_units` của Eagle3 là `draft_tree_nodes`; các method khác ghi `draft_tokens`, nên tổng proposal units không so sánh trực tiếp giữa họ method.

### Acceptance length theo dataset (ước lượng từ console log)

`Mean acceptance length` bên dưới là giá trị rolling do vLLM in ra, không phải mean tính trên từng request. Để gán log vào dataset, mỗi dòng được ghép với các `request_finished` của method trong khoảng 10 giây trước thời điểm log; nếu khoảng này chứa nhiều dataset, dòng được gán cho dataset có nhiều request hoàn tất nhất trong khoảng đó. Giá trị trong ô là trung bình số học các cửa sổ được gán như vậy, làm tròn 2 chữ số. Ký hiệu `thuần/tổng` cho biết số cửa sổ chỉ có request của dataset đó trên tổng số cửa sổ được gán vào dataset; các cửa sổ hỗn hợp vẫn góp vào giá trị, nên đây là ước lượng để tham khảo, không phải thống kê dataset chính xác.

| Dataset | GPU0 DSpark | GPU0 DFlash | GPU1 Eagle3 | GPU1 Domino |
|---|---:|---:|---:|---:|
| gov_report | 3.06 (1/1) | 2.12 (1/2) | 2.25 (2/2) | 1.08 (3/3) |
| qmsum | 2.56 (1/1) | 1.82 (1/1) | 2.02 (1/2) | 1.06 (1/3) |
| multi_news | 3.07 (0/2) | 2.52 (0/2) | 2.05 (1/2) | 1.09 (3/3) |
| lcc | 3.29 (0/1) | 2.82 (0/1) | 2.08 (1/2) | 1.05 (2/4) |
| repobench-p | 3.11 (1/1) | 2.42 (1/1) | 2.04 (1/2) | 1.05 (2/2) |

Các cửa sổ dùng ở đây nằm trong thời gian infer đo chính thức; đã loại log trước request đầu tiên và cửa sổ đầu nếu kết thúc trong 10 giây đầu kể từ request đầu của method để giảm khả năng trộn warmup vào kết quả. Mỗi method chạy 446 request trên cùng 5 dataset. Với `vanilla_vllm`, acceptance length không áp dụng vì không có draft model. Nguồn để audit từng cửa sổ là `console.log`; event log cung cấp timestamp/sample để xác định dataset, nhưng không lưu lại acceptance length theo request.

## 5. Thời gian khởi tạo và thực thi

| Run | Method | Engine status | Load ms | Method wall duration s* | Warmup finished events | Request finished |
|---|---|---|---|---|---|---|
| GPU0 | vanilla_vllm | success | 149655.9 | 376.5 | 446 | 446 |
| GPU0 | dspark | success | 74260.0 | 205.8 | 446 | 446 |
| GPU0 | dflash | success | 50068.4 | 205.0 | 446 | 446 |
| GPU1 | vanilla_vllm | success | 148466.6 | 376.9 | 446 | 446 |
| GPU1 | eagle3 | success | 46983.0 | 265.1 | 446 | 446 |
| GPU1 | domino | success | 47745.8 | 378.5 | 446 | 446 |

`*` Method wall duration tính từ `method_started` đến `method_finished` trong event log, gồm cả engine load/warmup/inference cho method đó. Hai GPU run bắt đầu cách nhau khoảng một giây và có thời gian chạy chồng lấp; đây là wall duration từng method, không phải tổng thời gian GPU độc quyền.

## 6. GPU và memory telemetry

GPU được report nhận diện compute capability 10.0. Mỗi request lưu `gpu_memory` gồm: `cuda_available`, `device_index`, `device_name`, `total_memory_bytes`, `allocated_bytes`, `reserved_bytes`, `peak_allocated_bytes`, `peak_reserved_bytes`, `compute_capability`. Tuy nhiên, trong tất cả 1.338 request của mỗi run, `allocated_bytes`, `reserved_bytes`, `peak_allocated_bytes`, `peak_reserved_bytes` đều bằng 0; các `gpu_process_memory` ở run report cũng bằng 0. Đây không phải số VRAM hợp lệ để kết luận mức dùng bộ nhớ của vLLM.

Report đồng thời lưu snapshot hệ thống GPU từ `nvidia-smi`. Tóm tắt min–max trên các snapshot ban đầu / sau load / sau method trong hai run:

| GPU physical index | Tên | Total MiB | Memory used MiB (min–max snapshot) | Utilization % (min–max snapshot) | Nhiệt độ °C (min–max snapshot) |
|---|---|---|---|---|---|
| 0 | NVIDIA B200 | 183359 | 5–165219 | 0–100 | 33–53 |
| 1 | NVIDIA B200 | 183359 | 5–165401 | 0–99 | 38–65 |

Hai run ghi `CUDA_VISIBLE_DEVICES` riêng (`0` và `1`), còn `nvidia-smi` snapshot chụp cả hai card trong lúc hai run chạy chồng lấp. Do đó telemetry system-level ở trên phản ánh toàn máy tại thời điểm chụp, không quy kết memory/utilization cho riêng một method.

## 7. Danh mục metric đã ghi và định nghĩa

| Nhóm | Metric / field | Ý nghĩa hoặc cách dùng | Tình trạng artifact |
|---|---|---|---|
| Coverage | sample_count, excluded_sample_count, dataset, sample_id | Số mẫu được infer, bị loại và khóa ghép mẫu. | Đủ; 446 chạy + 54 loại mỗi run. |
| Input | input_tokens | Độ dài prompt sau render/tokenize, từ `request_started` và sample manifest. | Đủ cho 446 mẫu; loại mẫu lưu input_tokens và limit. |
| Generation | output_tokens | Số token sinh ra cho request. | Đủ trong mọi request_finished. |
| Latency | prefill_ms | Thời gian prefill của request. | Đủ; báo cáo mean/p50/p95 theo dataset. |
| Latency | tpot_ms | `(last_token_ts - first_token_ts) / (output_tokens - 1)` theo report. | Đủ; báo cáo mean/p50/p95 theo dataset. |
| Latency | e2e_ms | Thời gian end-to-end mỗi request theo event. | Đủ; báo cáo mean/p50/p95 theo dataset. |
| Throughput | mean_throughput_tok_s | Throughput summary do run report cung cấp. | Có giá trị; report không lưu định nghĩa/công thức đủ để tái dựng. |
| Speedup | DSR | `mean TPOT vanilla / mean TPOT method`. | Có global; per-dataset tính lại từ event log. |
| Speedup | ESR | `(vanilla prefill + vanilla TPOT × paired min output length) / (vanilla prefill + method TPOT × paired min output length)`. | Có global; per-dataset tính lại theo cặp sample. |
| Quality | ROUGE-1 / ROUGE-2 / ROUGE-L | Overlap n-gram / LCS với reference, do run report tổng hợp. | Chỉ có số gộp năm dataset; không có per-dataset trong artifact được cung cấp. |
| Correctness | exact_match_rate | Tỷ lệ output token IDs giống hoàn toàn greedy vanilla. | Có global và từng sample trong parity report. |
| Correctness | token_lcs_overlap | `LCS(output token IDs, vanilla token IDs) / vanilla output token count`. | Có global và từng sample trong parity report. |
| Output guard | quality_valid_outputs / quality_valid_rate | Tỷ lệ output qua guard định dạng/nội dung của harness. | 446/446 ở tất cả method; không đồng nghĩa đúng task. |
| Output guard | repetition_flags | Số output bị gắn cờ lặp. | 0 cho mọi method. |
| Speculative | speculative_metrics_available_samples | Số request có trace/metadata speculative. | 446 cho mỗi speculative method. |
| Speculative | proposed_draft_tokens_observed | Tổng đơn vị proposal draft theo report. | Có tổng; đơn vị Eagle3 khác các method còn lại. |
| Speculative | acceptance_rate, acceptance_rate_percent | Tỷ lệ proposal draft được chấp nhận. | Null theo request trong JSON; `console.log` có `Avg Draft acceptance rate` theo cửa sổ runtime, không tách chính xác theo dataset. |
| Speculative | avg_accept_length | Độ dài trung bình accepted draft sequence. | Null theo request trong JSON; console log có `Mean acceptance length` rolling. Mục 4 có ước lượng theo dataset, không phải mean chính xác trên request. |
| Speculative | accepted_draft_tokens_observed | Tổng token draft được chấp nhận. | Null theo request trong JSON; `console.log` in `Accepted` token counter theo cửa sổ, không tái dựng được tổng chính xác cho từng dataset. |
| Runtime | load_ms | Thời gian load engine/model từng method. | Có trong model config; liệt kê mục 5. |
| Runtime | evaluation_runtime_ms | Thời gian evaluation tổng thể theo run report. | Có trong cả hai run. |
| GPU memory | allocated/reserved/peak bytes | Counters lấy từ CUDA/PyTorch context trong log. | Toàn bộ bằng 0; không dùng làm VRAM peak. |
| GPU system | memory used/free, utilization, temperature, driver | Snapshot toàn máy từ `nvidia-smi`. | Có snapshot rời rạc; bị ảnh hưởng bởi hai run chạy đồng thời. |

## 8. Chất lượng và correctness: cách đọc kết quả

- ROUGE-L được report: GPU0 vanilla 0,0744; DSpark 0,0751; DFlash 0,0759. GPU1 vanilla 0,0744; Eagle3 0,0759; Domino 0,0759. Đây là số mô tả từ report, không dùng làm so sánh task-aware vì gộp cả `lcc` và `repobench-p` code completion.
- Parity toàn bộ: vanilla 446/446; DSpark 304/446 (68,16%); DFlash 318/446 (71,30%); Eagle3 318/446 (71,30%); Domino 318/446 (71,30%). Mean LCS với vanilla: DSpark 92,76%; DFlash/Eagle3/Domino 93,27%.
- Tỷ lệ mismatch với greedy vanilla lần lượt là 142/446 cho DSpark và 128/446 cho từng DFlash/Eagle3/Domino. Do `correctness_pass=false`, không kết luận method đạt speedup tương đương chất lượng greedy.
- Mọi method có output 64 token ở gần như toàn bộ mẫu; `max_new_tokens=64` là trần chung. Vì vậy số ROUGE và latency phản ánh cấu hình generation cap này.

## 9. Kiểm tra tính toàn vẹn artifact

- Signature parity per-sample DFlash ↔ Eagle3 ↔ Domino giống hoàn toàn: **có**. So sánh gồm sample ID, exact-match bool, LCS overlap và số token method/reference.
- `samples.jsonl` hai run có cùng SHA-256; `excluded_samples.jsonl` cũng giống nội dung theo report. Điều này xác nhận cùng đầu vào, không xác nhận cùng generation output.
- Hai thư mục cung cấp không có `results.jsonl`, `results.partial.jsonl` hoặc `warmup.jsonl`. `artifact_paths` trong report trỏ tới vị trí trên server `/workspace/storage-shared/...`; những file đó không tồn tại dưới đường dẫn hiện có của workspace. Vì vậy không thể audit text/token IDs sinh ra hay tái tính ROUGE độc lập.
- `quality_pass=true` và guard 100% không sửa được parity mismatch và không thay thế kiểm tra metric đúng loại task.

## 10. Hạn chế và metric chưa thể kết luận

1. **Parity fail:** chưa có speculative method nào trong hai run đạt output equality 100% với greedy vanilla. Các DSR/ESR là exploratory khi correctness chưa đạt.
2. **Thiếu 54/500 input:** phần lớn loại ở `qmsum` (30/100), sau đó `repobench-p` (14), `gov_report` (9), `lcc` (1). Tập infer không còn đại diện đủ 100 mẫu/dataset.
3. **ROUGE gộp sai họ task:** `lcc` và `repobench-p` là code completion; cần metric exact/edit similarity riêng. Artifact không lưu metric task-aware theo dataset.
4. **Generation cap 64 token:** hạn chế khả năng đánh giá summary dài và có thể ảnh hưởng latency/proposal rate.
5. **Parity fingerprint trùng bất thường:** DFlash, Eagle3, Domino có cùng trạng thái match và LCS cho từng sample. Cần kiểm tra kết quả gốc và liên kết method trước khi tin metric chất lượng/parity.
6. **Acceptance chỉ có ở độ hạt cửa sổ:** event/JSON không có acceptance per-request. Console log có rolling acceptance length, acceptance rate và accepted-token counters; bảng theo dataset ở mục 4 là ước lượng gán cửa sổ, không đủ để kết luận chính xác hiệu quả draft trên từng dataset.
7. **VRAM counter không hợp lệ:** counters theo request ghi 0 bytes dù snapshot `nvidia-smi` cho thấy khoảng 161–165 GiB đang dùng trong các thời điểm chạy. Không so sánh peak memory theo method từ artifact này.
8. **Không có nhiều seed:** cấu hình ghi seed=42; chỉ có một run cho mỗi nhóm method. Báo cáo không cung cấp sai số/CI giữa nhiều seed.

## 11. Khuyến nghị bước tiếp theo

Trước khi dùng kết quả để xếp hạng baseline, khôi phục `results.jsonl` gốc từ hai đường dẫn artifact trong report và chạy integrity audit theo `(run_id, method, sample_id)`: đối chiếu generation token IDs, parity từng mẫu và task type; tính lại ROUGE riêng trên `gov_report/multi_news/qmsum`, exact match/edit similarity riêng trên `lcc/repobench-p`. Sau đó quyết định rõ xử lý 54 mẫu quá dài (mở rộng ngữ cảnh hay quy tắc truncation chuẩn) rồi chạy lại full matrix có lưu acceptance per-request và VRAM counters đã được kiểm tra.

## 12. Artifact nguồn

### GPU0 — `b200-gpu0-dspark-dflash`

| Artifact | Đường dẫn trong workspace | Vai trò |
|---|---|---|
| Báo cáo run | `outputs/Benchmark_VLLM_full/b200-gpu0-dspark-dflash/report_vi.md` | Bảng tổng hợp ngắn gốc. |
| Run report JSON | `outputs/Benchmark_VLLM_full/b200-gpu0-dspark-dflash/run_report.json` | Config, method metrics, parity từng sample, model/runtime metadata. |
| Event log | `outputs/Benchmark_VLLM_full/b200-gpu0-dspark-dflash/events.jsonl` | Request timings, guard flags, speculative counters, runtime events. |
| Sample manifest | `outputs/Benchmark_VLLM_full/b200-gpu0-dspark-dflash/samples.jsonl` | Prompt/input metadata, references, IDs; không có generated output. |
| Excluded samples | `outputs/Benchmark_VLLM_full/b200-gpu0-dspark-dflash/excluded_samples.jsonl` | 54 mẫu bị loại và lý do/input length. |
| Console log | `outputs/Benchmark_VLLM_full/b200-gpu0-dspark-dflash/console.log` | Stdout và rolling speculative metrics, gồm acceptance length theo cửa sổ. |
| Progress | `outputs/Benchmark_VLLM_full/b200-gpu0-dspark-dflash/progress.json` | Trạng thái/record count của run. |

Link nhanh: [report_vi.md](../../outputs/Benchmark_VLLM_full/b200-gpu0-dspark-dflash/report_vi.md), [run_report.json](../../outputs/Benchmark_VLLM_full/b200-gpu0-dspark-dflash/run_report.json), [events.jsonl](../../outputs/Benchmark_VLLM_full/b200-gpu0-dspark-dflash/events.jsonl), [console.log](../../outputs/Benchmark_VLLM_full/b200-gpu0-dspark-dflash/console.log)

### GPU1 — `b200-gpu1-eagle3-domino`

| Artifact | Đường dẫn trong workspace | Vai trò |
|---|---|---|
| Báo cáo run | `outputs/Benchmark_VLLM_full/b200-gpu1-eagle3-domino/report_vi.md` | Bảng tổng hợp ngắn gốc. |
| Run report JSON | `outputs/Benchmark_VLLM_full/b200-gpu1-eagle3-domino/run_report.json` | Config, method metrics, parity từng sample, model/runtime metadata. |
| Event log | `outputs/Benchmark_VLLM_full/b200-gpu1-eagle3-domino/events.jsonl` | Request timings, guard flags, speculative counters, runtime events. |
| Sample manifest | `outputs/Benchmark_VLLM_full/b200-gpu1-eagle3-domino/samples.jsonl` | Prompt/input metadata, references, IDs; không có generated output. |
| Excluded samples | `outputs/Benchmark_VLLM_full/b200-gpu1-eagle3-domino/excluded_samples.jsonl` | 54 mẫu bị loại và lý do/input length. |
| Console log | `outputs/Benchmark_VLLM_full/b200-gpu1-eagle3-domino/console.log` | Stdout và rolling speculative metrics, gồm acceptance length theo cửa sổ. |
| Progress | `outputs/Benchmark_VLLM_full/b200-gpu1-eagle3-domino/progress.json` | Trạng thái/record count của run. |

Link nhanh: [report_vi.md](../../outputs/Benchmark_VLLM_full/b200-gpu1-eagle3-domino/report_vi.md), [run_report.json](../../outputs/Benchmark_VLLM_full/b200-gpu1-eagle3-domino/run_report.json), [events.jsonl](../../outputs/Benchmark_VLLM_full/b200-gpu1-eagle3-domino/events.jsonl), [console.log](../../outputs/Benchmark_VLLM_full/b200-gpu1-eagle3-domino/console.log)

---
Báo cáo này phân biệt số liệu được ghi trong run report, metric tính lại từ event log và metric hiện không thể audit do thiếu raw results. Không xem `execution_pass=true` là bằng chứng baseline thắng về quality hoặc correctness.
