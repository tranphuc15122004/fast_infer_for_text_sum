# Phân tích semantic-selection cho tóm tắt dài

## Phạm vi và cách đọc kết quả

Báo cáo này phân tích lần chạy
[`semantic_qwen3_4b_debug_all.jsonl`](../outputs/semantic_qwen3_4b_debug_all.jsonl)
và file tổng hợp tương ứng
[`semantic_qwen3_4b_debug_all.jsonl.summary.json`](../outputs/semantic_qwen3_4b_debug_all.jsonl.summary.json).
Nó không gộp file `rebuild_smoke` vì file đó chỉ có hai request, không đủ để
so sánh thống kê giữa các scheme.

| Hạng mục | Giá trị |
|---|---|
| Target model | `Qwen/Qwen3-4B` |
| Thiết bị / dtype / attention | CUDA / FP16 / SDPA |
| Sinh text | greedy; tắt Qwen thinking; tối đa 128 token |
| Số document | 16: 4 GovReport, 4 Multi-News, 4 CNN/DailyMail, 4 XSum |
| Scheme | `full`, `random`, `lead`, `tfidf`, `textrank`, `mmr` |
| Budget selection | 512, 1,024, 2,048 source tokens |
| Chất lượng | ROUGE-1/2/L F1 so với `reference` |
| Số request đo | 256 = 16 document × (1 `full` + 5 scheme × 3 budget) |

Mỗi chỉ số trong các bảng là trung bình trên cùng 16 document, trừ P95. Cột
speedup là trung bình của tỷ lệ speedup theo từng document do runner ghi lại;
vì thế nó không nhất thiết đúng bằng tỷ số giữa hai giá trị E2E trung bình.

## Kết luận tóm tắt

1. `lead` với budget 512 là điểm cân bằng tốt nhất trong lần chạy này: ROUGE
   tổng thể cao nhất, selection nhanh nhất, TTFT thấp nhất và E2E thấp nhất
   trong các phương án 512 token.
2. Nén xuống khoảng 512 token giảm TTFT rất mạnh. `lead-512` giảm TTFT trung
   bình từ 4.532 s xuống 0.669 s và đưa E2E từ 17.010 s xuống 8.860 s, với
   speedup trung bình 1.891×.
3. Tuy nhiên, lợi thế ROUGE trung bình của `lead-512` không ổn định theo từng
   document: ROUGE-1 thắng 4, hòa 4 và thua 8 document so với `full`. Trung
   bình tăng là do một vài cải thiện lớn; không nên diễn giải là selection luôn
   nâng chất lượng.
4. `mmr` không có lợi thế chất lượng trong cấu hình hiện tại. Dù E2E ở budget
   lớn có thể thấp hơn một ít, selector tốn CPU hơn nhiều và riêng bước khởi
   tạo embedding model mất 29.3 s, chưa tính vào per-request E2E.
5. Hành vi phụ thuộc dataset: `lead` tốt trên GovReport và CNN/DailyMail,
   TextRank nổi bật hơn ở Multi-News, còn XSum mất ROUGE khi nén xuống 512.

## Baseline full-context

`full` giữ toàn bộ source context, trung bình 1,791.9 source token. Đây là
đường chuẩn để so sánh chất lượng và latency.

| Selector | Token giữ | TTFT mean / P95 | E2E mean / P95 | Output tok/s | Peak GPU | ROUGE-1 / 2 / L |
|---|---:|---:|---:|---:|---:|---|
| `full` | 1,791.9 (100%) | 4.532 / 12.439 s | 17.010 / 32.590 s | 9.73 | 9,744 MB | .2589 / .0781 / .1624 |

Tất cả group selection có cùng mean output length là 125.31 token, bằng
`full`. Vì vậy khác biệt latency chủ yếu đến từ context/prefill và chi phí
selector, không phải do selector sinh summary ngắn hơn.

## So sánh đầy đủ theo budget

`Giữ lại` là tỷ lệ context thực tế sau selection, không đơn thuần là
`budget / mean source tokens`: vài document vốn ngắn hơn budget nên không thể
luôn lấp đầy budget.

| Budget | Selector | Token chọn | Giữ lại | Selection | TTFT | E2E | Speedup E2E | ROUGE-1 / 2 / L |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 512 | Random | 454.7 | 52.6% | 128 ms | .813 s | 8.920 s | 1.868× | .2303 / .0563 / .1443 |
| 512 | **Lead** | 454.4 | 52.5% | **116 ms** | **.669 s** | **8.860 s** | **1.891×** | **.2619 / .0874 / .1638** |
| 512 | TF-IDF | 455.9 | 52.8% | 127 ms | .730 s | 9.009 s | 1.857× | .2253 / .0597 / .1364 |
| 512 | TextRank | 455.9 | 52.8% | 125 ms | .710 s | 9.186 s | 1.811× | .2278 / .0610 / .1383 |
| 512 | MMR | 445.4 | 51.9% | 781 ms | 1.257 s | 9.723 s | 1.689× | .2108 / .0480 / .1272 |
| 1,024 | Random | 766.4 | 69.4% | 182 ms | 1.300 s | 11.333 s | 1.584× | .2215 / .0507 / .1401 |
| 1,024 | **Lead** | 767.2 | 69.5% | **175 ms** | **1.285 s** | 10.640 s | 1.581× | **.2533 / .0751 / .1570** |
| 1,024 | TF-IDF | 767.2 | 69.5% | 187 ms | 1.343 s | 10.577 s | 1.564× | .2439 / .0695 / .1542 |
| 1,024 | TextRank | 767.6 | 69.5% | 197 ms | 1.363 s | 10.541 s | 1.581× | .2397 / .0668 / .1533 |
| 1,024 | MMR | 754.1 | 68.9% | 590 ms | 1.636 s | **10.407 s** | 1.568× | .2231 / .0533 / .1402 |
| 2,048 | Random | 1,224.2 | 84.4% | **248 ms** | 2.570 s | 12.986 s | 1.236× | .2405 / .0629 / .1556 |
| 2,048 | **Lead** | 1,223.6 | 84.4% | 290 ms | **2.413 s** | 12.850 s | 1.247× | .2582 / **.0839 / .1632** |
| 2,048 | TF-IDF | 1,224.3 | 84.5% | 313 ms | 2.468 s | 12.806 s | 1.252× | **.2584** / .0705 / .1552 |
| 2,048 | TextRank | 1,224.5 | 84.5% | 303 ms | 2.456 s | 13.068 s | 1.226× | .2522 / .0719 / .1560 |
| 2,048 | MMR | 1,204.2 | 83.9% | 625 ms | 2.661 s | **12.663 s** | **1.265×** | .2369 / .0569 / .1439 |

## Latency đuôi và GPU memory

Budget 512 là điểm nén có lợi ích rõ nhất. Tất cả scheme đều giảm memory và
P95 E2E so với full-context; `lead` có TTFT P95 thấp nhất, còn Random có E2E
P95 thấp nhất một chút. Với chỉ 16 document, chênh lệch nhỏ ở P95 chưa đủ để
xếp hạng ổn định.

| Scheme | P95 selection | P95 TTFT | P95 E2E | Peak GPU mean | Nhận xét |
|---|---:|---:|---:|---:|---|
| `full` | 0 ms | 12.439 s | 32.590 s | 9,744 MB | Không nén context |
| Random-512 | .404 s | 1.351 s | **10.106 s** | 7,972 MB | Lower-bound ngẫu nhiên |
| Lead-512 | **.303 s** | **1.003 s** | 10.575 s | 7,972 MB | TTFT tốt nhất |
| TF-IDF-512 | .334 s | 1.428 s | 10.606 s | 7,973 MB | Không đổi được quality thành lợi thế |
| TextRank-512 | .338 s | 1.021 s | 10.446 s | 7,973 MB | Tail E2E thấp trong nhóm relevance-aware |
| MMR-512 | 1.694 s | 2.202 s | 11.663 s | 7,968 MB | CPU selection tạo tail lớn |

So với `full`, mean peak GPU allocation giảm khoảng 18.2% ở 512 token,
16.2% ở 1,024 token và 11.0% ở 2,048 token. Lợi ích memory không khác đáng
kể giữa các selector cùng budget; nó chủ yếu do số token context được giữ lại.

## Chất lượng và độ ổn định của Lead-512

`Lead-512` có ROUGE trung bình cao hơn `full`, nhưng paired comparison cho
thấy kết quả không đồng đều theo document.

| Metric | Mean delta vs full | Median delta | Thắng / hòa / thua trên 16 document |
|---|---:|---:|---:|
| ROUGE-1 | +.0031 | -.0011 | 4 / 4 / 8 |
| ROUGE-2 | +.0093 | .0000 | 6 / 4 / 6 |
| ROUGE-L | +.0014 | -.0008 | 4 / 4 / 8 |

Insight quan trọng là: `lead-512` là phương án latency–quality tốt nhất *trên
trung bình*, không phải phương án luôn bảo toàn quality cho mọi input. Khả năng
cao một phần lợi thế đến từ tính lead-biased của tin tức, đồng thời việc loại
bớt phần context gây nhiễu có thể giúp model tập trung khi output bị giới hạn
128 token.

## Phân rã theo dataset tại budget 512

| Dataset | Mean source token | Full: E2E / R1-R2-RL | Lead-512: E2E / R1-R2-RL | Insight |
|---|---:|---|---|---|
| GovReport | 2,783.8 | 22.82 s / .239-.082-.157 | 9.03 s / **.258-.104-.174** | Lead vừa nhanh vừa tăng cả ba ROUGE trong 4 mẫu. |
| Multi-News | 1,892.0 | 17.63 s / .278-.069-.162 | 8.88 s / .246-.050-.147 | TextRank-512 đạt R1 .278 và R2 .073, nhưng RL .155 thấp hơn full; đây là dataset nên thử TextRank thêm. |
| CNN/DailyMail | 1,244.5 | 14.35 s / .368-.140-.231 | 9.20 s / **.401-.179-.243** | Lead là winner rất rõ; phù hợp đặc tính tin tức đặt thông tin quan trọng ở đầu. |
| XSum | 1,247.2 | 13.25 s / **.150-.022-.099** | 8.33 s / .142-.017-.091 | Nén 512 token làm giảm quality; nên giữ full hoặc thử budget cao hơn. |

Các số theo dataset chỉ dựa trên bốn document/dataset. Vì vậy chúng hữu ích để
đặt giả thuyết cho lần chạy lớn hơn, chưa đủ để đưa ra policy cứng theo dataset.

## So sánh từng semantic-selection scheme

### Lead

Đây là phương án thực dụng nhất hiện tại. Nó không cần model phụ, selection
latency thấp và có tổ hợp ROUGE-2/L tốt nhất ở mọi budget trong aggregate
(TF-IDF nhỉnh hơn Lead .0002 ROUGE-1 tại 2,048 token). Đặc biệt tốt với
CNN/DailyMail và GovReport trong tập debug. Hạn chế là bias về phần đầu
document: Multi-News và XSum cho thấy thông tin quan trọng có thể nằm ngoài
prefix.

### Random

Random là lower bound hợp lệ để chứng minh lợi ích không chỉ đến từ giảm token.
Nó nhanh gần Lead nhưng giảm ROUGE ở mọi budget; ví dụ tại 512 token, ROUGE-1
thấp hơn full .0285. Không nên dùng làm scheme production, nhưng nên giữ trong
benchmark để xác định selection có thực sự học/relevance-aware hay không.

### TF-IDF centroid

TF-IDF có chi phí selection vừa phải và tại 2,048 token gần bằng full ở
ROUGE-1 (.2584 so với .2589). Tuy vậy ROUGE-2/L vẫn thấp hơn Lead và nó không
thắng Lead ở điểm latency–quality. Đây là baseline lexical tốt, phù hợp làm
đối chứng nhẹ CPU hơn MMR, nhưng không phải lựa chọn mặc định theo kết quả này.

### TextRank

TextRank không thắng aggregate, nhưng là tín hiệu đáng chú ý cho Multi-News:
ở 512 token nó đạt ROUGE-1 ngang full và ROUGE-2 cao hơn full trong bốn mẫu
Multi-News. Vì Multi-News ghép nhiều bài báo, graph-based centrality có thể
giúp phủ nhiều nội dung hơn prefix-only Lead. Cần xác nhận bằng 100 mẫu trước
khi chọn riêng một policy cho Multi-News.

### MMR

MMR hướng tới đa dạng semantic, nhưng chi phí không tương xứng ở đây:
selection mất 0.59–0.78 s trung bình, P95 1.48–2.13 s, và startup embedding
model 29.3 s chưa được cộng vào E2E. Dù MMR có E2E mean nhỏ nhất tại 1,024 và
2,048 token, ROUGE thấp nhất hoặc gần thấp nhất. Với request đơn lẻ hoặc tải
thấp, startup cost càng làm MMR bất lợi; chỉ nên đánh giá lại nếu có batch lớn,
selector model luôn resident và mục tiêu chính là coverage/diversity thay vì
ROUGE.

## Khuyến nghị vận hành tạm thời

| Mục tiêu | Khuyến nghị | Lý do |
|---|---|---|
| Cân bằng latency và quality | `lead --token-budgets 512` | Pareto tốt nhất trên aggregate hiện có. |
| Ưu tiên quality, vẫn cần nén | `lead --token-budgets 2048` | Gần bằng full ở ROUGE; TTFT/E2E vẫn thấp hơn full. |
| Multi-News | So sánh `lead-512` và `textrank-512` trên 100 mẫu | TextRank có tín hiệu tốt về R1/R2 trên dữ liệu đa nguồn. |
| XSum | Giữ `full` hoặc thử 1,024/2,048 trước | 512 token làm mất ROUGE trong tập debug. |
| Production đơn giản | Không chọn MMR ở vòng đầu | Chi phí selection và startup cao, quality chưa chứng minh được lợi ích. |

## Giới hạn của kết luận và bước tiếp theo

- Cỡ mẫu chỉ có 16 document; không có confidence interval hoặc kiểm định ý
  nghĩa. Cần chạy bốn file representative 100 mẫu/dataset để có 400 document.
- ROUGE là overlap từ vựng, không đo trực tiếp tính factual, coverage hay mức
  độ hallucination. Nên bổ sung kiểm tra factuality hoặc human evaluation cho
  các scheme được chọn.
- Hardware model không được ghi trong artifact, nên các latency tuyệt đối chỉ
  hợp lệ cho môi trường chạy này. So sánh tương đối giữa selector mới là điểm
  đáng dùng lại.
- `max_new_tokens=128` tạo cùng độ dài output trung bình ở tất cả group. Nếu
  deployment dùng output dài hơn, cần chạy lại vì decode có thể chiếm tỷ trọng
  lớn hơn và làm thay đổi trade-off E2E.
- Lần chạy lớn nên giữ cố định model, attention backend, seed, max output,
  dataset split; báo cáo mean, median, P95 và bootstrap CI của chênh lệch
  ROUGE/E2E theo từng document.
