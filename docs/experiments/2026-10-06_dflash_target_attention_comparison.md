# So sánh phân phối attention giữa DFlash và target

Ngày: 2026-10-06. Trạng thái: hoàn tất phép đo attention ghép cặp trên Modal
L40S. Đây là một khảo sát phân phối; nó chưa kiểm tra chất lượng khi nén hoặc
lược bỏ KV.

## Tóm tắt

**Có phần tương đồng ở mức phân bổ theo nhóm key, nhưng thứ hạng key cụ thể
không trùng tốt và khác biệt tăng khi context dài.** Trên toàn bộ key scope,
JS divergence trung bình tăng từ 0,404 ở prompt 3K lên 0,459 ở 16K. Giao của
Top-1K vị trí cache giảm từ 49,0% xuống 19,8%. Ở 16K, target đặt trung bình
46,1% tổng attention mass lên tập Top-1K do DFlash chọn; chiều ngược lại,
DFlash đặt 30,8% tổng mass lên tập Top-1K do target chọn.

Phân bổ theo vùng rộng giống nhau hơn thứ hạng token: ở 16K, target trung
bình dành 56,6% mass cho prompt, 15,9% cho output đã commit và 27,5% cho
block hiện tại. DFlash tương ứng là 58,8%, 19,9% và 21,4%. Vì vậy một
selector chỉ nhìn tổng mass của prompt/output/block sẽ bỏ lỡ khác biệt trong
cách hai mô hình chọn vị trí cụ thể bên trong cache.

Các số trên **không chứng minh** vị trí được chú ý nhiều là cần thiết cho
acceptance hoặc chất lượng sinh. Chúng chỉ cho biết attention target và DFlash
không tạo cùng một bản xếp hạng token. Cần can thiệp cache để kiểm tra giá trị
nhân quả trước khi chốt tín hiệu huấn luyện module chọn ngữ cảnh.

## Thiết lập phép đo

- 10 instance GovReport, cùng prompt và token đầu ra với run gốc; bốn độ dài
  prompt ban đầu: 3.072, 5.120, 8.192 và 16.384 token.
- Target Qwen3-4B có 36 layer; đo layer 0 (thấp), 18 (giữa), 35 (cao). DFlash
  Qwen3-4B-DFlash-b16 có 5 layer; đo cả năm layer 0–4.
- Tổng cộng 40 trajectory và 4.366 lượt draft. Mọi token đầu ra và số lượt
  draft đều khớp run tham chiếu: 40/40 cho cả hai kiểm tra.
- Mỗi query được ghép theo proposal token mà nó dự đoán: DFlash mask query
  1–15 với target verifier query 0–14.
- So sánh chính dùng phân phối xác suất trên **100% key scope**:
  prompt + output đã commit + 16 vị trí block hiện tại. Block được giữ nguyên
  trong phân tích vì nó sẽ luôn có mặt trong phương pháp dự kiến. Top-K và
  overlap bên dưới xếp hạng riêng các key cache trước block; cache gồm prompt
  cùng output đã commit.
- Target attention lấy từ Q/K của cùng forward target, áp causal/sliding mask
  đang dùng và cộng xác suất sau softmax trong float32. Target giữ nguyên hàm
  SDPA để tính output. Mass của target lên key tương lai trong verifier bằng
  0 trong toàn bộ phép đo.
- Với mỗi layer, attention được mean qua heads và 15 query đã ghép. Tính mean
  lượt trong từng document, sau đó mean 10 document với trọng số bằng nhau.

JS là Jensen–Shannon divergence theo log₂; 0 biểu thị hai vector giống nhau,
1 là tối đa khác nhau. JS cache-conditional là chỉ số phụ: bỏ 16 key block,
rồi chuẩn hóa phần cache riêng về tổng 1. Nó trả lời câu hỏi thứ hạng/cache
phân phối ra sao sau khi loại block, không thay thế phép so sánh chính trên
100% mass.

## Kết quả chính trên toàn bộ 100% mass

| Prompt | JS toàn scope ↓ | JS cache-conditional ↓ | Giao vị trí Top-1K cache | Target mass trên Top-1K DFlash | DFlash mass trên Top-1K target |
|---|---:|---:|---:|---:|---:|
| 3K | 0,404 | 0,478 | 49,0% | 56,8% | 54,8% |
| 5K | 0,427 | 0,503 | 33,6% | 50,1% | 44,9% |
| 8K | 0,441 | 0,520 | 26,0% | 48,8% | 37,9% |
| 16K | 0,459 | 0,539 | 19,8% | 46,1% | 30,8% |

Các cột “mass trên Top-1K của mô hình kia” dùng tổng mass gốc làm mẫu số và
chỉ cộng weight của các vị trí cache được mô hình kia xếp hạng cao. Đây là
độ phủ chéo có hướng, không phải phần trăm vị trí giao nhau. Cột giao vị trí
là số index chung chia cho 1.024.

![JS giữa từng layer target và DFlash trên toàn key scope và cache có điều kiện](../../outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/analysis/figures/attention_js_matrices.png)

![Giao Top-K vị trí cache theo từng cặp layer](../../outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/analysis/figures/attention_topk_overlap.png)

## Nhận xét

### 1. Mass theo nhóm key khá giống; chọn token cụ thể khác nhau

Ở bốn độ dài, mass trung bình vào từng vùng ổn định tương đối:

| Prompt | Mô hình | Prompt | Output đã commit | Block 16 vị trí |
|---|---|---:|---:|---:|
| 3K | Target | 55,3% | 16,0% | 28,7% |
| 3K | DFlash | 50,4% | 26,8% | 22,8% |
| 5K | Target | 56,5% | 15,5% | 28,1% |
| 5K | DFlash | 53,9% | 24,2% | 21,9% |
| 8K | Target | 56,1% | 15,7% | 28,2% |
| 8K | DFlash | 55,0% | 22,6% | 22,3% |
| 16K | Target | 56,6% | 15,9% | 27,5% |
| 16K | DFlash | 58,8% | 19,9% | 21,4% |

Target dành thêm khoảng 5–7 điểm phần trăm vào block hiện tại; DFlash dành
nhiều hơn cho output đã commit, chênh khoảng 4–11 điểm theo cap. Dù tổng
prompt mass gần nhau, giao Top-1K giảm mạnh. Do đó tương đồng ở các vùng lớn
không đồng nghĩa có chung một danh sách KV quan trọng.

![Mass được chia cho prompt, output đã commit và block hiện tại](../../outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/analysis/figures/attention_mass_by_scope.png)

### 2. Khó dùng một tập vị trí chung cố định ở context dài

Ở prompt 16K, trung bình chỉ khoảng **203/1.024 vị trí** trong Top-1K của
target và DFlash trùng nhau. Target đặt trung bình 46,1% tổng mass lên
Top-1K của DFlash, nhưng DFlash chỉ đặt 30,8% lên Top-1K của target.
Độ phủ chéo bất đối xứng cho thấy hai mô hình có thể cho các vị trí khác
nhau trọng số và độ sắc nét khác nhau. Vì vậy không nên dùng Top-K của
attention một mô hình như ground truth duy nhất cho selector của mô hình kia.

Top-4K overlap cần đọc theo kích thước cache: ở 3K cache ngắn hơn 4K nên
mọi cách chọn lấy toàn bộ cache và overlap bằng 100%; ở 5K tập Top-4K buộc
phải giao nhau đáng kể do chỉ có ít hơn 5K key. Các mốc này không phải bằng
chứng độc lập cho selector tốt.

### 3. Layer target sâu nhất lệch rõ hơn

Ở 16K, mean JS toàn scope qua năm layer DFlash là 0,396 cho target layer 0,
0,428 cho layer 18 và 0,553 cho layer 35. Cache-conditional JS tương ứng
là 0,461 / 0,483 / 0,672. Layer 35 lệch hơn trên cả phân phối toàn scope
và cache đã chuẩn hóa. Đây là dấu hiệu rằng “target attention” không phải
một phân phối duy nhất: layer sâu cho thứ hạng cache khác hơn so với các
layer thấp/giữa trong phép ghép này.

Trung bình qua ba target layer ở cap 16K, DFlash layer 1 có JS toàn scope
thấp nhất (0,367), kế tiếp là layer 2 (0,418); layer 0 cao nhất (0,548).
Đây là mean mô tả theo tập và cách ghép đã chọn, không đủ để bỏ layer khác
khỏi module. Hình ma trận thể hiện các cặp cụ thể; các lượt/query/head riêng
có thể biến thiên lớn hơn phép mean này.

### 4. Profile vị trí cuối chưa chỉ ra “cuối prompt” là quan trọng

Hình profile gom **toàn bộ key scope** thành 10 bin theo index tương đối.
Các bin cuối chứa lẫn phần cuối prompt, output đã sinh và block hiện tại.
Vì vậy spike ở bin cuối không thể được quy riêng cho các token cuối prompt.
Đây là lý do báo cáo dùng mass theo từng nhóm key để diễn giải, và không dùng
profile này làm bằng chứng cho một ưu tiên vị trí đơn giản.

![Profile attention trên 10 bin của toàn key scope](../../outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/analysis/figures/attention_position_profiles.png)

## Đọc kết quả theo mức bằng chứng

### CONFIRMED

- Trong phạm vi 10 GovReport × 4 độ dài đã khai báo, phép thu thập hoàn tất
  4.366 lượt draft; target và DFlash được ghép trên cùng proposal, cùng
  trajectory tham chiếu. Replay khớp chính xác output và số vòng ở 40/40
  trajectory.
- Các phân phối đo được có tổng mass bằng 1 trong sai số số học tối đa
  1,07×10⁻⁴; không phát hiện target attention vào key verifier tương lai.
- Theo metric đã định nghĩa, phân phối layer-pair có JS trung bình 0,404–0,459
  theo cap; Top-1K cache overlap trung bình giảm 0,490 xuống 0,198.

### EXPLORATORY

- Mass prompt/output/block gần nhau hơn so với thứ hạng từng key. Target layer
  35 và DFlash layer 0/4 cho một số divergence cao; DFlash layer 1 có mean
  divergence thấp nhất ở 16K.
- Các quan sát có thể gợi ý selector cần tín hiệu theo layer và không gian
  token phân tán. Chúng chưa chứng minh tín hiệu đó cải thiện suy đoán.

### FAILED / INCOMPLETE

- Một lượt chạy thử trước đó bị loại vì thay nhãn backend attention khiến
  trajectory lệch run gốc. Bản chính giữ target trên SDPA, chỉ bọc hàm SDPA
  để đo Q/K; bản này khớp 40/40 và là nguồn duy nhất của các số/hình trong
  báo cáo.
- Chưa có phép xóa/chọn KV và chưa đo acceptance, latency, ROUGE hay chất
  lượng đầu ra sau nén.

### HIGHEST VERIFIED RUNG

**R7 cho khảo sát attention có phạm vi giới hạn này:** toàn bộ ma trận 10
instance × 4 cap đã chạy, được kiểm tra replay và có decision memo này.
Artifact số liệu: `outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/analysis/summary.json`.
R7 ở đây chỉ kết luận về phép đo phân phối, không phải hiệu quả của phương
pháp nén context.

### EVIDENCE GAPS

- Mười document chỉ thuộc GovReport, một sample seed và một target/drafter
  pair; chưa biết xu hướng có chuyển sang task/domain/model khác không.
- Attention được mean qua heads và 15 query. Mean có thể che mất head chuyên
  biệt, query đầu/cuối hoặc các lượt sinh có hành vi khác.
- Target causal queries và DFlash mask queries có ngữ nghĩa/mask khác nhau.
  Phần block của DFlash có mask positions, còn target verifier có proposal
  token thật và không thể nhìn key tương lai; phần chênh lệch JS có thể phản
  ánh khác biệt cấu trúc này.
- Attention mass là thống kê nội bộ, không phải phép đo nhân quả về token
  cần thiết cho acceptance hoặc chất lượng tóm tắt.

### RECOMMENDED NEXT

Chạy một can thiệp cache nhỏ trên cùng 10×4 trajectory: giữ nguyên block 16
vị trí, cho target verifier giữ full context, và chỉ rút gọn context KV mà
DFlash nhận. So sánh full cache với selector Top-K từ attention DFlash, từ
target (tách layer 0/18/35), random và recency tại budget 1K/4K. Primary
metric nên là DFlash acceptance/accepted tokens mỗi round; đồng thời kiểm tra
đầu ra greedy cuối so target full-context. Bước này sẽ cho biết ranking nào
thực sự giữ lại thông tin hữu ích cho DFlash, thay vì chỉ có phân phối gần
nhau.

**Verified through R7 for the scoped attention-distribution study. The proposed
cache-compression method has not yet been verified by a causal quality or
acceptance evaluation.**

## Artifact và tái lập

- Run Modal: `outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/`
- Phân tích: `outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/analysis/summary.json`, `summary.csv`, `round_metrics.jsonl`.
- Audit: `outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b/analysis/analysis_audit.json`.
- Collector, analyzer, plotter và Modal entrypoint được lưu bản sao dưới thư mục `analysis/code/`.
- Lệnh phân tích offline:

```bash
python3 scripts/analyze_dflash_target_comparison.py \
  --input outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b
python3 scripts/plot_dflash_target_comparison.py \
  --input outputs/dflash_attention_probe/dflash-target-attention-compare-fullmass-govreport10-l40s-20261006b
```

Kết quả trong `outputs/` là artifact local bị gitignore; báo cáo này được
commit-able cùng các script để giữ mô tả phương pháp.
