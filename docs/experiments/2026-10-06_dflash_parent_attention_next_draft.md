# Parent attention của target có dự đoán được cache DFlash lượt kế tiếp?

Ngày: 2026-10-06. Trạng thái: hoàn tất phép đo parent-to-next-draft trên Modal
L40S. Câu hỏi cụ thể: attention của target query đã tạo anchor cho block kế
tiếp có giống attention mà DFlash dùng ở lượt kế tiếp không?

## Kết quả ngắn

**Độ giống giảm theo context length.** Ở prompt 16K, Top-1K key cache được
query target của lượt trước chú ý nhiều nhất chỉ nhận **29,8% tổng mass**
của DFlash ở lượt sau. Nếu chuẩn hóa riêng phần cache và bỏ mass của block
16 vị trí mới, Top-1K đó nhận **38,0% cache mass**. Hai mô hình chỉ trùng
**19,5%** vị trí trong Top-1K.

Top-1K này giữ **87,3% parent-query attention mass**, nên chênh lệch không
phải do target query có một phân bố hoàn toàn dàn đều. Nó cho thấy các key
target query trước xếp hạng cao không đủ để dự đoán phần lớn cache mass mà
DFlash cần ở context 16K. Đây là một tín hiệu phản đối selector thuần túy
dựa trên attention của target query trước; chưa phải phép thử chất lượng khi
lược bỏ KV.

## Ghép đúng lượt và đúng query

Ở lượt verify `r`, gọi `k` là số proposal DFlash được target chấp nhận liên
tiếp. Query target `q=k` tạo token được đưa làm anchor cho lượt DFlash `r+1`.
Trong code hiện tại, khi `k=15`, `q=15` là literal **bonus query** sau khi
chấp nhận hết 15 proposal. Khi `k<15`, `q=k` tạo token correction sau điểm
không khớp; token đó vẫn là anchor ở lượt draft sau.

Vì vậy thí nghiệm ghép attention của target query `q=k` trong verify `r` với
attention DFlash ở draft `r+1`. Prefix target query nhìn thấy phải có đúng số
key như cache DFlash lượt mới. Run kiểm tra điều này ở mọi chuyển tiếp.

Phép đo chính giữ mẫu số **100% key scope**: prompt + output đã commit + 16
key của block DFlash hiện tại. Parent target query được đặt mass 0 lên anchor
mới và 15 mask mới vì các key này chưa tồn tại khi parent query chạy. Cache
conditional JS và coverage chuẩn hóa trên cache chỉ là chỉ số chẩn đoán bổ
sung. Phương pháp nén dự kiến vẫn giữ nguyên block hiện tại.

## Thiết lập

- 10 instance GovReport, seed 42; prompt ban đầu 3.072 / 5.120 / 8.192 /
  16.384 token.
- Target Qwen3-4B: layer 0, 18, 35; DFlash Qwen3-4B-DFlash-b16: cả 5 layer.
- 4.366 lượt draft và 4.326 chuyển tiếp parent-to-next-draft. Lượt DFlash
  đầu không có parent từ lượt verify trước nên không tính vào chuyển tiếp.
- Mỗi parent query vector mean qua attention heads tại đúng query `q=k`.
  DFlash vector mean qua heads và 15 proposal queries trong từng layer.
  JS / overlap được tính cho từng cặp target-parent layer × DFlash layer.
- Tính mean chuyển tiếp trong từng document rồi mean đều 10 document. Hình
  histogram dùng các mốc chuyển tiếp đầu, giữa, cuối; mốc đầu là DFlash lượt
  2 vì lượt đầu chưa có parent query.
- Tất cả 40 trajectory khớp chính xác token output và số draft round với run
  nguồn. Tổng mass của attention vector sai tối đa 1,07×10⁻⁴; mass target lên
  key verifier tương lai bằng 0.

Top-1K overlap chia cho 1.024 vị trí chung. “DFlash mass trên parent Top-1K”
dùng tổng attention mass DFlash làm mẫu số; cột cache-conditional chuẩn hóa
riêng các key prompt + output đã commit sau khi bỏ block 16.

## Số đo trên mọi chuyển tiếp

| Prompt | Chuyển tiếp | JS toàn scope | JS cache-conditional | Giao Top-1K | Parent mass trên DFlash Top-1K | DFlash tổng mass trên Parent Top-1K | DFlash cache mass trên Parent Top-1K |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3K | 808 | 0,544 | 0,475 | 47,3% | 83,9% | 52,8% | 68,8% |
| 5K | 904 | 0,562 | 0,499 | 32,6% | 76,0% | 43,3% | 55,7% |
| 8K | 1.196 | 0,582 | 0,520 | 25,3% | 73,3% | 36,7% | 47,6% |
| 16K | 1.418 | 0,604 | 0,548 | 19,5% | 70,3% | 29,8% | 38,0% |

Ở 16K có ba chi tiết đáng chú ý:

1. Parent Top-1K giữ 87,3% mass của chính parent query. Nhưng DFlash chỉ
   dành 29,8% **tổng mass** cho tập đó, hay 38,0% sau khi chuẩn hóa riêng
   cache. Top-1K của DFlash ngược lại nhận 70,3% mass của parent query.
   Hai chiều coverage bất đối xứng.
2. Chênh lệch vẫn còn khi bỏ block rồi chuẩn hóa cache: JS là 0,548 và
   coverage DFlash-cache của Parent Top-1K là 38,0%. Vì vậy khác biệt không
   chỉ do DFlash dành khoảng 21,3% mass cho block mới.
3. Ở target layer 35, mean JS toàn scope / cache là **0,704 / 0,666**, cao
   hơn layer 0 (**0,557 / 0,495**) và layer 18 (**0,550 / 0,485**). Parent
   signal phụ thuộc rõ vào layer target được dùng.

![Histogram parent target attention và DFlash ở lượt kế tiếp](../../outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b/analysis/figures/parent_attention_histograms_fullmass.png)

Mỗi nhóm cột dùng cùng denominator 100%: prompt chia chunk 1K, output đã
commit chia chunk 128, anchor mới và 15 vị trí block. DFlash là mean 5 layer;
ba series target là parent query tại layer 0/18/35. Parent query không thể
attend anchor/block mới, nên mass ở hai nhóm này bằng 0 theo định nghĩa.

![Ma trận parent query ở lượt r so với DFlash ở lượt r+1](../../outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b/analysis/figures/parent_attention_comparison_matrices.png)

## Đọc theo vai trò bonus/correction

Trong 4.326 chuyển tiếp, query `q=accepted_proposals` có mean **1,12**, max
**10**; không có query nào là `q=15`. Có 7 round chấp nhận đủ 15 proposal,
nhưng cả 7 đều là round cuối của trajectory nên không có lượt draft kế tiếp
để ghép attention. Do đó:

- Kết quả chính kiểm tra query target thật sự tạo anchor cho lượt DFlash kế
  tiếp ở các trajectory này. Các query đều là correction query, không phải
  literal bonus query `q=15`.
- Nhánh bonus-only có **0 transition**, nên run này không thể kết luận riêng
  về bonus query sau khi chấp nhận trọn block. Cần báo cáo riêng nếu một
  tập/model sinh ra các transition full-accept.

## Kết luận theo mức bằng chứng

### CONFIRMED

- Trong 10 GovReport × 4 context length đã định, có 4.326 cặp chuyển tiếp
  được ghép bằng query `q=k` và cache prefix khớp chính xác. Output greedy
  và số vòng draft khớp run nguồn ở 40/40 trajectory.
- Theo metric đã định, JS toàn scope tăng **0,544 → 0,604**, Top-1K overlap
  giảm **47,3% → 19,5%**, và DFlash tổng mass trên Parent Top-1K giảm
  **52,8% → 29,8%** từ 3K đến 16K.
- Trong sample này không có literal bonus query transition; giới hạn này
  được kiểm tra trực tiếp qua `accepted_proposals`.

### EXPLORATORY

- Attention của parent query có thể bao phủ nhiều mass của DFlash Top-1K,
  nhưng chiều ngược lại yếu hơn và giảm khi context dài. Layer 35 khác rõ
  với layer 0/18.
- Kết quả gợi ý attention target trước đó **không đủ làm selector duy nhất**
  cho DFlash ở 16K. Có thể dùng nó như một feature cho module học selector,
  kết hợp với DFlash/layer/depth/context features.

### FAILED / INCOMPLETE

- Run thử đầu dừng ở sanity check trước khi tạo dữ liệu vì reduction nhầm
  trục query/head. Run đó không được dùng. Rerun đã sửa trục, hoàn tất và
  khớp 40/40 trajectory.
- Literal bonus-only comparison chưa có sample; phép thử causal cache
  eviction và chất lượng/acceptance sau nén cũng chưa chạy.

### HIGHEST VERIFIED RUNG

**R7 cho phép đo parent-to-next-draft trong phạm vi đã nêu:** toàn bộ 10 × 4
conditions hoàn tất, replay đã khớp và có phân tích/audit. Artifact kết quả:
`outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b/analysis/summary.json`.
Rung này chỉ kết luận về phân phối attention, không xác nhận module chọn
context cải thiện suy đoán.

### EVIDENCE GAPS

- Chỉ có GovReport, một seed và một cặp target/DFlash; chưa biết mức khái
  quát sang task hoặc model khác.
- Parent vector mean qua heads ở một query; chưa phân tích selector theo
  từng head.
- Không có literal `q=15` full-accept transition trong sample.
- Attention coverage không đo nhân quả. Sau khi xóa KV, attention distribution
  và draft hidden states sẽ đổi.

### RECOMMENDED NEXT

Chạy ablation cache nhỏ trên cùng 10 × 4 trajectory. Dùng query `q=k` của
target lượt trước để chọn 1K/4K cache key; giữ nguyên block 16 và để target
verifier dùng full context. So sánh với full-cache, DFlash-attention Top-K,
random và recency. Primary metric: accepted proposals mỗi draft round; kiểm
tra thêm exact output/ROUGE. Đây là phép thử đầu tiên cho biết parent target
attention có ích cho selector khi KV thật sự bị lược bỏ hay không.

**Verified through R7 for this parent-attention distribution study. Cache
selection quality and bonus-only alignment remain unverified.**

## Artifact và tái lập

- Run Modal: `outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b/`
- Tóm tắt: `analysis/summary.json`, `analysis/parent_summary.csv`.
- Audit SHA-256 và kiểm tra ghép chuyển tiếp: `analysis/analysis_audit.json`.
- Mã collector/analyzer/plotter/Modal launcher: `analysis/code/`.
- Lệnh phân tích/plot offline:

```bash
python3 scripts/analyze_dflash_target_comparison.py \
  --input outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b
python3 scripts/plot_dflash_parent_attention.py \
  --input outputs/dflash_attention_probe/dflash-parent-attention-next-draft-govreport10-l40s-20261006b
```

Run và hình nằm trong `outputs/` (gitignored); report và script nằm trong repo.
