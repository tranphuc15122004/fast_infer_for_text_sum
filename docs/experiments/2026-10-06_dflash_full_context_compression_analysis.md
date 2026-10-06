# Phân tích DFlash cho nén toàn bộ context

Ngày: 2026-10-06. Trạng thái: hoàn tất phân tích CPU trên artifact của run
Modal L40S; không chạy thêm inference hoặc training.

**Chỉ số chính dùng tổng mass trên toàn bộ key làm mẫu số 100%:** prompt,
output đã commit và draft block hiện tại. Mỗi insight có thêm một phép
phân tích cache, loại đúng 16 key của block rồi chuẩn hóa mass còn lại.
Các bảng này ghi rõ mẫu số là cache. Cache vẫn gồm cả prompt và output đã
commit. Phương pháp đề xuất **luôn giữ block 16 token và nén context KV
phục vụ drafter**, được tạo từ target features.

## 1. Kết quả chính

Ở context 16K, Top-1K mọi key bao phủ **67,29%** tổng mass; Top-4K bao phủ
**82,61%**. Cửa sổ liên tiếp tốt nhất cùng budget chỉ giữ **49,61%** và
**63,36%**. Vì vậy nén dựa trên một vùng liên tiếp hoặc một tập key nhỏ cố
định có thể bỏ sót một phần đáng kể của attention quan sát được.

Phân bố có một nhóm key rất tập trung và một phần mass trải rộng: **64 key
cao nhất nhận 46,54% mass**, nhưng cần trung bình **7.035 key để đạt 90%**.
Context dài làm giảm mức tập trung tương đối; toàn bộ phân bố vẫn khác xa
phân bố đều. Mười phần trăm key cuối sequence nhận **50,93%** tổng mass ở
16K, trong khi phân bố đều sẽ dành khoảng 10% cho vùng này.

Tập key được ưu tiên thay đổi theo quá trình sinh. Ở 16K, Top-4K tại lượt
giữa chỉ bao phủ **54,72%** mass của lượt cuối, so với **82,01%** nếu chọn
lại tại lượt cuối. Quan sát này tạo động cơ cho memory thích ứng với trạng
thái sinh và cho phép tổng hợp thông tin từ nhiều vị trí trên toàn bộ context.

Khi loại block khỏi phép đo tại 16K, Top-1K cache bao phủ **58,58% cache
mass**, Top-4K bao phủ **77,96%**. Nếu giữ nguyên block cùng các cache key
được chọn, tổng mass gốc được giữ lần lượt là **67,43% / 82,66%**. Nhóm
cache key tập trung, phần mass trải rộng, thiên lệch về cuối và thay đổi
tập chọn theo decoding đều còn hiện diện sau khi loại block. Vì vậy các
insight này còn liên quan trực tiếp tới phần cache mà module sẽ nén.

## 2. Dữ liệu, phép đo và giới hạn phạm vi

Nguồn: run `dflash-attention-fullmass-govreport10-l40s-20261006`:

- 10 document GovReport, seed chọn mẫu 42, mỗi document đủ ít nhất 16K source token.
- Bốn prompt ban đầu: 3.072 / 5.120 / 8.192 / 16.384 token; tổng 40 lần sinh.
- Target Qwen3-4B, drafter DFlash-b16 năm layer; một GPU Modal L40S.
- 4.366 lượt draft thật; cả 40 lần sinh dừng ở EOS, output dài 111–435 token.
- Attention của drafter được thu bằng eager attention, sau softmax, ở từng layer.
- Vector hiện có đã mean qua heads, 15 proposal queries và năm draft layers.
  Anchor query được loại khỏi phép mean; anchor vẫn nằm trong key scope.

**100% ở đây là 100% key mass của các proposal queries được đo.** Nó không
mở rộng phạm vi sang attention của target hoặc các query không được thu.
NPZ còn giữ riêng từng draft layer, nhưng không giữ tensor riêng cho từng
head/query.

Với vector attention trung bình `a_t(i)` của lượt draft `t`, đặt:

```text
p_t(i) = a_t(i) / sum_i a_t(i)
coverage_t(S) = sum_{i thuộc S} p_t(i)
N_t = prompt_tokens + committed_output_tokens + 16
```

Coverage toàn bộ không chia cho mass của một đoạn context. Mỗi lượt được
chuẩn hóa trên mọi key để loại sai số số học nhỏ; tổng `p_t` bằng 1. Số key trung bình
của các trajectory 3K / 5K / 8K / 16K lần lượt khoảng **3.207 / 5.246 /
8.329 / 16.522**. Nhãn 3K–16K chỉ độ dài prompt ban đầu, không phải tổng key
count cố định trong suốt quá trình sinh.

Với cùng budget K, so sánh Top-K rải rác, cửa sổ liên tiếp tốt nhất, K key
đầu/cuối/giữa và K/2 key đầu cộng K/2 key cuối. Mọi chiến lược dùng
`min(K, N_t)` key. Top-K được chọn từ attention của chính lượt đang đo;
đây là lựa chọn oracle theo score quan sát được, chưa là selector triển khai
trước attention.

Tính mean qua lượt của từng document, sau đó mean 10 document với trọng số
bằng nhau. Mốc đầu/giữa/cuối là record đầu, record `len(records)//2` và
record cuối của từng trajectory; mốc giữa không nhất thiết là 50% số output
token. Các range trong báo cáo là min–max của mean từng document, không phải
confidence interval.

### Phép phân tích bổ sung: loại block, nén cache, luôn giữ block

Gọi `B_t` là 16 vị trí cuối: **một anchor và 15 mask/proposal positions**.
Gọi `C_t` là prefix phía trước: prompt cộng output đã commit. Anchor của
lượt hiện tại thuộc `B_t`, kể cả ở lượt đầu chưa có committed output.

```text
b_t = sum_{i thuộc B_t} p_t(i)                  # block mass / tổng mass
m_t = sum_{i thuộc C_t} p_t(i) = 1 - b_t        # cache mass / tổng mass
q_t(i) = p_t(i) / m_t, i thuộc C_t             # tổng q_t = 1 trên cache
C_t(S) = sum_{i thuộc S} q_t(i)                # coverage / cache mass
R_t(S) = b_t + m_t * C_t(S)                    # mass giữ / tổng mass gốc
```

`R_t` là weight gốc đặt lên tập `B_t ∪ S`, được tính **từng lượt trước khi
lấy mean**. Không nhân trực tiếp các mean `b`, `m`, `C`. Đây là ước lượng
coverage của tập key dự kiến giữ, chưa là attention sau một forward nén.
Budget K ở nhánh này là `min(K, |C_t|)` cache key **cộng 16 block key**;
tổng budget khác bảng Top-K mọi key trước đó. 1K = 1.024 và 4K = 4.096.

Cache có trung bình khoảng 3.191 / 5.230 / 8.313 / 16.506 key. Block nhận
22,82% / 21,85% / 22,32% / 21,35% tổng mass ở bốn cap tương ứng. Không
đặt block mass cố định bằng 20% trong phép tính.

Việc loại 16 key là xử lý lại vector attention dense đã lưu, **chưa chạy
inference bỏ block**. Vector được mean qua heads/queries/layers trước rồi
chuẩn hóa trên cache; cách này không tương đương chuẩn hóa từng head/query
trước khi mean. Khi xét từng layer, dùng mẫu số cache riêng của layer đó.

Trong DFlash của run này, target hidden features được hợp nhất rồi chiếu
thành context K/V bởi từng draft layer. Báo cáo dùng “context KV từ target”
để chỉ cache này; cache của target phục vụ verification là một cache riêng.
Chi tiết trong [mã DFlash](../../externals/dflash/dflash/model.py) và
[collector](../../scripts/probe_dflash_attention.py).

## 3. Insight: key có score cao phân bố ở nhiều vị trí

| Prompt ban đầu | Top-1K | Cửa sổ 1K tốt nhất | Top-4K | Cửa sổ 4K tốt nhất |
|---|---:|---:|---:|---:|
| 3K | 92,38% | 66,27% | 100,00% | 100,00% |
| 5K | 83,92% | 59,61% | 98,89% | 82,89% |
| 8K | 76,96% | 55,62% | 93,59% | 73,70% |
| 16K | 67,29% | 49,61% | 82,61% | 63,36% |

**Mọi số trong bảng là phần trăm của tổng attention mass.** Ở 3K, budget 4K
lớn hơn mọi key count thực tế, nên 100% ở hàng này không phản ánh khả năng
nén xuống một tập key nhỏ.

Ở 16K, chọn rải rác tăng coverage **17,68 điểm phần trăm** so với cửa sổ 1K
tốt nhất, và **19,25 điểm** với budget 4K. Đây là bằng chứng cho việc thu
thập từ nhiều vị trí có lợi về coverage; chưa đo lợi ích về acceptance.

Trong cả 4.366 lượt, cửa sổ liên tiếp tốt nhất bằng cửa sổ K key cuối với
các budget 1K và 4K được kiểm tra. Vị trí cuối bao gồm output và draft block,
do đó ưu thế này không thể diễn giải thành ưu thế riêng của phần cuối report.

Một vài cách chọn vị trí tại 16K, cùng budget 1K:

| Cách chọn | Coverage toàn bộ |
|---|---:|
| 1K key đầu | 5,83% |
| 1K key ở giữa | 1,99% |
| 1K key cuối | 49,61% |
| 512 key đầu + 512 key cuối | 50,91% |
| 1K key cao nhất, rải rác | 67,29% |

Với budget 4K, chia đều đầu/cuối giữ 59,72%, thấp hơn cửa sổ 4K cuối
(63,36%) và Top-4K rải rác (82,61%). Phân bổ budget cố định theo đầu/cuối
chưa bao phủ tốt bằng chọn theo score trong study này.

![So sánh chọn key trên toàn bộ context](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/full_context_analysis/figures/selection_strategies_full_context.png)

### Khi loại 16 block key khỏi phép đo

| Prompt ban đầu | Top-1K cache / cache mass | Cửa sổ 1K / cache mass | Top-4K cache / cache mass | Cửa sổ 4K / cache mass |
|---|---:|---:|---:|---:|
| 3K | 90,31% | 57,83% | 100,00% | 100,00% |
| 5K | 79,68% | 48,60% | 98,61% | 78,35% |
| 8K | 70,56% | 42,87% | 91,81% | 66,12% |
| 16K | 58,58% | 35,95% | 77,96% | 53,42% |

Ở 16K, chênh lệch rải rác–liên tiếp vẫn là **22,63 / 24,54 điểm cache
coverage** với 1K/4K. Khi luôn giữ block, tổng mass giữ của Top-K là
**67,43% / 82,66%**, còn cửa sổ tốt nhất chỉ **49,66% / 63,40%**. Chênh
lệch trên tổng mass là **17,78 / 19,27 điểm**. Ưu thế chọn rải rác vẫn rõ
trên phần cache sẽ nén.

Với budget 1K tại 16K, đầu/giữa/cuối cache giữ 7,43% / 2,55% / 35,95%
cache mass; chia 512 đầu + 512 cuối giữ 37,66%. Sau khi loại block, cửa sổ
tốt nhất bằng cửa sổ cuối trong 4.140/4.366 lượt ở 1K và 4.302/4.366 lượt
ở 4K. Vì thế nhận xét “cửa sổ cuối luôn tốt nhất” chỉ đúng với phạm vi mọi
key của study gốc; trên cache có các lượt cửa sổ khác tốt hơn.

![Chọn cache key và tổng mass khi luôn giữ block](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/figures/cache_selection_coverage.png)

Cột trái của hình chuẩn hóa trên cache; cột phải đánh giá trên tổng mass
gốc khi giữ block. Mỗi hàng dùng cùng budget cache. Ở 3K, 4K cache lớn hơn
cache hiện có nên các cách chọn đều giữ tất cả.

## 4. Insight: Top-K tối ưu vẫn có phần mass nằm ngoài tập chọn

| Prompt ban đầu | Top-1K | Top-2K | Top-4K | Top-8K | Key cho 90% mass | Key cho 95% mass |
|---|---:|---:|---:|---:|---:|---:|
| 3K | 92,38% | 97,75% | 100,00% | 100,00% | 772 | 1.387 |
| 5K | 83,92% | 92,10% | 98,89% | 100,00% | 1.706 | 2.638 |
| 8K | 76,96% | 84,92% | 93,59% | 99,97% | 3.066 | 4.593 |
| 16K | 67,29% | 74,25% | 82,61% | 92,10% | 7.035 | 10.129 |

Ở 16K, ngoài Top-1K còn **32,71%** mass; ngoài Top-4K còn **17,39%**.
Tăng số key bốn lần, từ 1K lên 4K, chỉ tăng coverage thêm **15,32 điểm**.
Budget 4K đạt trên 90% ở 8K nhưng chưa đạt 90% ở 16K; một budget tuyệt đối
cố định khó giữ cùng mức coverage khi context dài hơn.

Top-K tối đa hóa tổng weight của K key đang tồn tại trong vector đã đo.
Vì thế bảng là giới hạn coverage của **chọn K key từ phân bố gốc**, không
phải giới hạn của module tạo memory mới. Một memory vector có thể tổng hợp
thông tin của nhiều key; attention tới memory sau nén cũng sẽ thay đổi.
Các số 90%/95% là mốc coverage để mô tả phân bố, chưa là ngưỡng acceptance.

![Đường coverage và chuẩn hóa budget theo số key](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/full_context_analysis/figures/concentration_full_context.png)

### Khi loại 16 block key khỏi phép đo

| Prompt ban đầu | Top-1K / cache mass | Top-4K / cache mass | Cache key cho 90% cache mass | Cache key cho 90% tổng mass, giữ block |
|---|---:|---:|---:|---:|
| 3K | 90,31% | 100,00% | 981 | 756 |
| 5K | 79,68% | 98,61% | 2.039 | 1.690 |
| 8K | 70,56% | 91,81% | 3.645 | 3.050 |
| 16K | 58,58% | 77,96% | 8.187 | 7.019 |

Hai cột cuối trả lời hai câu hỏi khác nhau. Cột 90% cache yêu cầu bao phủ
90% phần sẽ nén; cột 90% tổng mass tính cả block được giữ sẵn. Các count là
mean của số key tối thiểu từng lượt, không phải ngưỡng đảm bảo cho mọi lượt.

Ở 16K, ngoài Top-4K còn **22,04% cache mass**, tương ứng **17,34% tổng mass
gốc** dù giữ toàn bộ block. Top-8K cache giữ 89,99% cache mass và **92,12%
tổng mass** khi cộng block. Để đạt 95% cache mass cần khoảng **10.983 cache
key**; để đạt 95% tổng mass với block giữ sẵn cần **10.113 cache key + 16**.
Nén cache xuống 1K/4K bằng cách chỉ giữ token vẫn có phần mass bị bỏ ngoài
đáng kể; đây là động cơ khảo sát memory tổng hợp cho phần còn lại.

![Độ tập trung trên cache sau khi loại block](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/figures/cache_concentration.png)

## 5. Insight: context dài có phần mass trải rộng hơn, vẫn có nhóm key rất tập trung

Chỉ so Top-1K giữa các độ dài chưa đủ để kết luận phân bố phẳng hơn, vì 1K
chiếm tỷ lệ key ngày càng nhỏ. So cùng tỷ lệ số key cho thấy xu hướng vẫn
tồn tại:

| Prompt ban đầu | Top 10% số key | Top 25% số key | Tỷ lệ key cho 90% mass | Entropy / log(N) |
|---|---:|---:|---:|---:|
| 3K | 81,16% | 90,24% | 23,99% | 0,722 |
| 5K | 76,43% | 86,84% | 32,51% | 0,737 |
| 8K | 74,81% | 85,13% | 36,80% | 0,737 |
| 16K | 71,95% | 82,73% | 42,58% | 0,745 |

Phân bố đều có Top 10% key chứa khoảng 10% mass và entropy chuẩn hóa bằng
1. Ở 16K, Top 10% key vẫn chứa 71,95%; mức tập trung còn cao. Entropy tăng
từ 0,722 lên 0,745, không tăng tới gần 1.

Đặc biệt, ở 16K:

- Top-16 key nhận **26,23%** mass.
- Top-64 key nhận **46,54%** mass, dù chỉ chiếm khoảng 0,39% key count.
- Top-1K nhận **67,29%**; phần ngoài Top-1K vẫn nhận **32,71%**.

Diễn đạt phù hợp với dữ liệu: **nhóm key có weight cao vẫn rất tập trung,
trong khi phần mass còn lại đòi hỏi nhiều key hơn khi context dài**. Không
nên diễn đạt rằng toàn bộ DFlash attention trở nên đều hoặc mọi token có
độ quan trọng gần nhau.

### Khi loại 16 block key khỏi phép đo

| Prompt ban đầu | Top 10% cache key / cache mass | Top 25% / cache mass | Tỷ lệ cache key cho 90% cache mass | Entropy cache / log(N_cache) |
|---|---:|---:|---:|---:|
| 3K | 76,25% | 87,55% | 30,66% | 0,749 |
| 5K | 70,25% | 83,31% | 38,97% | 0,775 |
| 8K | 67,80% | 80,95% | 43,85% | 0,785 |
| 16K | 64,45% | 78,08% | 49,60% | 0,803 |

Xu hướng phần mass trải rộng còn rõ trên cache: tỷ lệ key cần cho 90%
cache mass tăng từ 30,66% lên 49,60%. Đồng thời ở 16K, **Top-16 cache key
vẫn nhận 22,49% cache mass**, Top-64 nhận **34,31%**. Nhóm tập trung không
chỉ là các key trong draft block.

Do đó giả thuyết giữ một nhóm cache key chi tiết và học cách tổng hợp phần
trải rộng vẫn có cơ sở quan sát. Không diễn giải 34,31% cache mass thành
34,31% tổng mass: nhóm Top-64 cache nằm trong phần cache có mass trung bình
78,65% của tổng, và quy đổi chính xác cần tính từng lượt theo công thức `R`.

## 6. Insight: thiên lệch đầu/cuối giảm nhưng phần cuối sequence vẫn mạnh

Chia toàn bộ sequence thành 10 bin, mỗi bin chứa khoảng 10% số key. Đây là
phép chia theo vị trí tương đối của **mọi key**, không chỉ prompt.

| Prompt ban đầu | 10% key đầu | 10% key cuối | Hai vùng gộp, 20% số key |
|---|---:|---:|---:|
| 3K | 12,63% | 58,91% | 71,54% |
| 5K | 11,13% | 55,46% | 66,60% |
| 8K | 6,35% | 54,79% | 61,14% |
| 16K | 7,50% | 50,93% | 58,43% |

Hai vùng đầu/cuối chiếm ít mass hơn khi cap tăng, từ 71,54% xuống 58,43%,
nhưng vẫn cao hơn nhiều mức 20% của phân bố đều. Ở 16K, đây chủ yếu là
thiên lệch về **cuối sequence**; đầu sequence không tăng đồng đều theo cap.

Ở lượt draft đầu, 10% key đầu nhận 32,52% mass tại 3K và 8,49% tại 16K.
Mười phần trăm key cuối tương ứng nhận 48,92% và 47,25%. Như vậy ưu thế
ở đầu giảm mạnh hơn ưu thế ở cuối tại mốc này. Những nhận xét dựa trên
histogram cần ghi rõ đang xét lượt đầu hay mean cả trajectory.

Vùng cuối chứa các key của trạng thái đang sinh. Các vị trí mask tiếp tục
được biến đổi qua layer, nên mass tại chúng không đồng nghĩa mass dành cho
một embedding mask chưa mang thông tin. Vị trí KV có weight cao cũng không
đo trực tiếp lượng evidence gốc mà hidden feature đó chứa.

![Phân bố vị trí của toàn bộ context](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/full_context_analysis/figures/spatial_profiles_full_context.png)

### Khi loại 16 block key khỏi phép đo

Chia lại cache thành 10 bin có số key gần bằng nhau; toàn bộ bin cộng thành
100% **cache mass**, gồm cả output prefix.

| Prompt ban đầu | 10% cache key đầu / cache mass | 10% cache key cuối / cache mass | Hai vùng gộp, 20% cache key |
|---|---:|---:|---:|
| 3K | 16,49% | 47,01% | 63,50% |
| 5K | 14,36% | 43,12% | 57,48% |
| 8K | 8,24% | 41,83% | 50,07% |
| 16K | 9,55% | 37,61% | 47,16% |

Thiên lệch về cuối vẫn mạnh: tại 16K, bin cuối nhận gần 3,8 lần mức 10%
của cache đều. Hai vùng đầu/cuối giữ 47,16% cache mass; **52,84% còn lại
ở 80% cache giữa**. Vì vậy ưu tiên recency có ích nhưng chưa bao phủ cache
đầy đủ. Phần cuối cache có cả prompt gần cuối và output đã commit, nên
bảng chưa tách được đóng góp của recency và nội dung ở từng phần.

![Phân bố vị trí trên cache không chứa block](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/figures/cache_spatial_profiles.png)

Hình cho thấy mass ở cuối cache tăng từ lượt đầu tới lượt cuối ở các cap.
Bin được chia lại khi cache dài ra; xu hướng này mô tả vị trí tương đối,
không khẳng định cùng một nhóm token tăng weight. Sự xuất hiện của output
prefix cũng góp phần vào thay đổi này.

## 7. Insight: mass tổng ổn định có thể đi cùng tập key thay đổi

Đối chiếu persistent prefix key bằng absolute sequence position và current
block bằng slot 0–15. Matching block slot không có nghĩa hidden states tại
slot đó giống nhau qua các lượt. Overlap là số key ID chung chia số key đã
chọn ở lượt trước; không phải Jaccard.

| Prompt ban đầu | Top-1K trùng giữa hai lượt liền kề | Top-1K trùng lượt giữa–cuối | Top-4K trùng lượt giữa–cuối |
|---|---:|---:|---:|
| 3K | 88,38% | 54,29% | 100,00%* |
| 5K | 86,07% | 49,20% | 88,02% |
| 8K | 83,03% | 37,18% | 68,94% |
| 16K | 82,88% | 31,92% | 49,81% |

*Ở 3K, Top-4K bao gồm mọi key hiện có. Các key đã chọn ở lượt giữa vẫn tồn
tại ở cuối, nhưng output mới xuất hiện sau đó chưa thuộc tập chọn cũ; vì
thế overlap 100% không có nghĩa tập cũ vẫn giữ 100% mass.

Coverage tại **lượt cuối của 16K**, cùng budget và cùng mẫu số toàn bộ:

| Tập key dùng tại lượt cuối | Budget 1K | Budget 4K |
|---|---:|---:|
| Giữ tập đã chọn tại lượt giữa | 40,37% | 54,72% |
| Chọn lại Top-K tại lượt cuối | 66,20% | 82,01% |
| Chênh lệch | 25,83 điểm | 27,29 điểm |

Chênh lệch bao gồm cả thay đổi weight trên prefix cũ và sự xuất hiện của
output key mới. Với phạm vi nén toàn bộ context, module cần xử lý cả hai.

Ở hai lượt liền kề tại 16K, Top-1K trùng 82,88% và Top-4K trùng 88,89%.
Dùng tập chọn từ lượt trước thay vì chọn lại làm coverage thấp hơn trung
bình 4,72 điểm với 1K và 4,46 điểm với 4K. Đây là mức ổn định ngắn hạn để
khảo sát cách cập nhật memory; chưa xác định tần suất refresh có lợi về
latency/acceptance. Snapshot ở các mốc xa nhau chưa cho thấy mọi lượt đều
cần dựng lại memory hoàn toàn.

![Tập chọn thay đổi theo thời gian](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/full_context_analysis/figures/temporal_selection_full_context.png)

### Khi loại 16 block key khỏi phép đo

Chỉ matching cache key bằng absolute prefix position; loại hoàn toàn slot
của block khỏi phép overlap.

| Prompt ban đầu | Top-1K cache trùng hai lượt liền kề | Top-1K cache trùng giữa–cuối | Top-4K cache trùng giữa–cuối |
|---|---:|---:|---:|
| 3K | 88,31% | 53,89% | 100,00%* |
| 5K | 85,93% | 48,73% | 88,10% |
| 8K | 82,85% | 36,42% | 68,96% |
| 16K | 82,65% | 31,11% | 49,67% |

*Ở 3K, tập Top-4K cache ở lượt giữa chứa mọi cache key lúc đó. Output mới
chưa có trong tập cũ vẫn có thể nhận mass lớn ở lượt cuối.

Tại **lượt cuối của 16K**, khi cả hai chiến lược đều giữ 16 key của block
hiện tại:

| Budget cache | Tập cache từ lượt giữa / cache mass | Chọn lại / cache mass | Tập cũ + block / tổng mass | Chọn lại + block / tổng mass |
|---|---:|---:|---:|---:|
| 1K | 25,36% | 57,81% | 40,47% | 66,35% |
| 4K | 43,30% | 77,52% | 54,78% | 82,07% |

Chênh lệch tổng mass vẫn là **25,88 / 27,29 điểm**. Thay đổi tập chọn vì
vậy còn hiện diện trong cache sẽ nén, ngay cả khi luôn giữ block mới.
Các con số ở lượt cuối khác mean cả trajectory trong bảng trước.

Giữa hai lượt liền kề ở 16K, tái sử dụng tập cache làm mất 5,99 / 5,67
điểm cache coverage với 1K/4K, tương ứng **4,72 / 4,46 điểm tổng mass**
khi giữ block. Module nên khảo sát cơ chế tiếp nhận output mới và cập nhật
ưu tiên trên prefix cũ; số liệu chưa xác định lịch refresh tối ưu.

![Tập chọn cache thay đổi và tổng mass khi giữ block](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/figures/cache_temporal_selection.png)

## 8. Insight: một tập key chung có coverage khác nhau giữa các layer

Chọn Top-K bằng vector mean năm layer, sau đó đo tập key đó trên từng
layer, vẫn chia cho **tổng mass của layer trên mọi key**. Ở 16K:

| Draft layer, index từ 0 | Coverage của tập Top-1K chung | Coverage của tập Top-4K chung |
|---|---:|---:|
| 0 | 53,19% | 73,83% |
| 1 | 69,48% | 81,63% |
| 2 | 80,58% | 91,45% |
| 3 | 62,55% | 79,05% |
| 4 | 70,65% | 87,11% |
| Mean năm layer | 67,29% | 82,61% |

Mean vẫn phù hợp để mô tả một memory dùng chung. Tuy nhiên coverage trung
bình cao chưa đảm bảo cùng mức coverage ở mọi layer. Nếu huấn luyện module
chung, nên xem thêm mức tái tạo attention/value output từng layer để nhận
biết layer bị mất nhiều thông tin hơn. Chưa suy ra layer nào quyết định
acceptance chỉ từ bảng này.

### Khi loại 16 block key khỏi phép đo

Chọn Top-K cache chung từ vector mean năm layer, rồi đánh giá trên từng
layer tại 16K. Hai cột đầu dùng cache mass của chính layer làm mẫu số; hai
cột sau dùng tổng mass của layer và luôn cộng block.

| Draft layer | Top-1K chung / cache mass | Top-4K chung / cache mass | Top-1K + block / tổng mass | Top-4K + block / tổng mass |
|---|---:|---:|---:|---:|
| 0 | 49,76% | 71,88% | 53,36% | 73,90% |
| 1 | 49,38% | 69,53% | 69,56% | 81,68% |
| 2 | 69,64% | 86,58% | 80,70% | 91,48% |
| 3 | 56,32% | 75,54% | 62,70% | 79,11% |
| 4 | 68,15% | 85,97% | 70,83% | 87,16% |
| Vector mean, chuẩn hóa trên cache | 58,58% | 77,96% | 67,43% | 82,66% |

Coverage cache của layer 1 thấp dù coverage tổng của nó cao hơn layer 0,
vì block chiếm nhiều mass hơn ở layer 1. Khi chỉ nén cache, coverage tổng
có thể che việc phần cache của một layer được bao phủ chưa tốt. Cần xem
cả hai mẫu số khi thiết kế loss hoặc đánh giá module.

Hai cột cache ở hàng cuối **không phải mean số học năm hàng layer**:
chuẩn hóa cache sau khi mean layer làm các layer có cache mass lớn đóng
góp nhiều hơn. Hai cột tổng + block khớp mean layer trong sai số số học.

## 9. Mức lặp lại giữa các instance và giới hạn suy luận

Ở 16K, mean Top-1K theo document nằm trong **66,50–68,52%**; Top-4K trong
**82,24–83,34%**. Các giá trị coverage khá gần nhau trong 10 document này,
nhưng vị trí token được chọn có thể khác nhau. Chưa có một format chọn
context cố định dùng được cho mọi document.

Bốn cap lấy các prefix source khác nhau và sinh output khác nhau. Vì vậy
xu hướng theo cap là quan sát đi cùng độ dài và nội dung, chưa là một
can thiệp chỉ thay độ dài. Study dùng một dataset, một cặp checkpoint, một
seed chọn mẫu và output tương đối ngắn; chưa bao phủ các tác vụ khác hoặc
generated history dài hàng nghìn token.

Attention weights mô tả allocation trong forward gốc. Chưa đo tác động
của bỏ key, đổi value hoặc thay context bằng memory tổng hợp lên draft
logits/acceptance. Khi bỏ key, softmax và hidden states các layer sau sẽ
thay đổi; coverage không thể được đọc trực tiếp thành tỷ lệ chất lượng
được giữ lại. Mean head/query cũng có thể che các nhu cầu riêng lẻ.

### Khi loại 16 block key khỏi phép đo

Ở 16K, mean Top-1K cache theo document nằm trong **57,37–59,70% cache
mass**, Top-4K trong **77,60–78,37%**. Phần cache ở 10 document này có độ
tập trung khá gần nhau; chưa chứng minh một danh sách vị trí chung sẽ dùng
được trên các document khác nhau.

Nhánh mới dùng chính các vector dense của study gốc. Chuẩn hóa cache không
loại ảnh hưởng của draft block lên queries/hidden states khi forward. Nó
trả lời “mass gốc phân bố thế nào trên phần cache sẽ nén”, chưa trả lời
“model sẽ dự đoán thế nào khi không có block” hoặc “acceptance sau nén là
bao nhiêu”. Phương pháp giữ block vẫn cần can thiệp cache để đo câu hỏi sau.

## 10. Hàm ý cho module nén toàn bộ context

**Giả định mặc định đã cập nhật:** mỗi lượt giữ đủ 16 vị trí của draft
block hiện tại, gồm anchor và 15 proposal positions. Các vị trí này vẫn
được tính và cập nhật theo forward DFlash gốc; không đưa chúng vào budget
nén cache. Module nén prefix context gồm **prompt + output đã commit**.

Luồng dữ liệu dự kiến:

```text
Target hidden features của prompt + committed output
  → context KV của các draft layer
  → module nén / chọn / tổng hợp cache
  → attention của 16 vị trí draft block với [cache đã nén + block hiện tại]
  → draft proposals → target verification
```

KV dùng trong target verification vẫn giữ đầy đủ theo protocol hiện tại.
Trong mã gốc, `cache_target` và `cache_draft` tách riêng; target features
được chiếu bằng K/V projection của từng draft layer. Phạm vi nén được đề
xuất ở đây là **cache context của drafter tạo từ target features**. Khi
báo VRAM/speedup cần đo lợi ích trên đúng phần cache này và trên toàn hệ
thống; giữ verifier cache đầy đủ chưa tạo lợi ích giảm bộ nhớ ở verifier.

| Quan sát | Hàm ý cần khảo sát |
|---|---|
| Cửa sổ cache kém hơn Top-K cache cùng budget | Memory cần tổng hợp từ nhiều vị trí trong prompt và committed output. |
| Nhóm cache key tập trung vẫn tồn tại sau khi loại block | Có thể kết hợp giữ chi tiết ở nhóm được ưu tiên với tổng hợp thông tin từ phần còn lại. |
| Top-4K cache + block ở 16K còn bỏ ngoài 17,34% tổng mass | Module tạo đại diện mới đáng khảo sát bên cạnh việc giữ nguyên một tập cache key. |
| Tập cache ưu tiên thay đổi theo trạng thái sinh | Khảo sát memory phụ thuộc trạng thái/query, tiếp nhận output mới và cập nhật ưu tiên trên prefix cũ. |
| Ổn định giữa hai lượt liền kề cao hơn giữa các mốc xa | Khảo sát tái sử dụng memory cùng cập nhật nhỏ hoặc refresh theo tín hiệu thay đổi. |
| Một tập cache chung có coverage lệch giữa layer | Theo dõi độ khớp từng layer; coverage tổng cao nhờ block chưa đảm bảo cache được bao phủ tốt. |

Các hàm ý này là giả thuyết thiết kế. Giữ block size và số queries gốc giúp
đánh giá tác động của nén context KV. Chỉ số attention chính vẫn đánh giá
trên tổng mass, với block được giữ sẵn; cache coverage là chẩn đoán phần
module tác động. Khi học memory, có thể khảo sát tái tạo attention/value
output từng layer cho các proposal queries, rồi đối chiếu logits và
acceptance. Vị trí/RoPE và tiến trình decoding cần tuân theo sequence gốc
khi chọn cache key rải rác. Chưa chọn kiến trúc hay loss cụ thể từ study này.

## 11. Đánh giá mức bằng chứng

### Đã xác nhận theo tiêu chí phương pháp — CONFIRMED

Chưa có kết quả xác nhận một module nén đạt tiêu chí acceptance hoặc tăng
tốc: module và tiêu chí chính của phép đánh giá đó chưa được khóa.

### Quan sát thăm dò — EXPLORATORY

Các bảng và hình trong báo cáo mô tả đầy đủ 4.366 lượt của study attention
10 instance. Chúng tạo bằng chứng định lượng cho chênh lệch giữa chọn liên
tiếp/rải rác, phần mass ngoài Top-K, thay đổi tập chọn qua decoding và nhu
cầu khác nhau giữa các layer; chưa chứng minh chất lượng sau nén.

### Chưa hoàn tất hoặc thất bại — FAILED / INCOMPLETE

Thu attention và phân tích đã hoàn tất; không có run inference dở dang
trong study này. Can thiệp KV, huấn luyện module nén và benchmark nén chưa
được thực hiện, nên chưa có kết quả để gọi là thành công hoặc thất bại.

### Mức kiểm chứng cao nhất — HIGHEST VERIFIED RUNG

Mức đã kiểm chứng thực tế là inference quan sát trên 10 instance/40 lần
sinh và đối chiếu toàn bộ NPZ trên CPU. Artifact chứng minh là manifest,
JSONL gốc, `artifact_audit.json`, `full_context_analysis/summary.json` và
`full_context_analysis/analysis_audit.json`, cùng nhánh
`context_cache_analysis/summary.json` và `analysis_audit.json` loại block.
Thang R0–R7 cho training của skill review không áp trực tiếp cho study chỉ
quan sát inference; **chưa
gán một rung GREEN nào cho phương pháp nén mới**, cũng chưa khóa primary
metric/decision rule của phương pháp đó. Không dùng việc hoàn tất study
attention để gọi module nén đã đạt R6/R7.

### Bằng chứng còn thiếu — EVIDENCE GAPS

Chưa có kết quả nén giữ acceptance, chưa có speedup đo với kernel phục vụ
inference, và chưa có kiểm định tính tổng quát ngoài dataset/checkpoint
hiện tại. Chưa biết phần mass trải rộng có thể được tổng hợp tốt đến mức
nào bởi memory mới, hoặc các head/query riêng cần thông tin khác nhau ra sao.

### Thực nghiệm tiếp theo được đề xuất — RECOMMENDED NEXT

Một thực nghiệm can thiệp nhỏ tại 16K: giữ cùng prefix/state, so draft
dense với draft chỉ cho attend **Top-K cache key + toàn bộ 16 key của block
hiện tại**, tại budget cache 1K/4K/8K. Giữ số queries/block size gốc và target
verification đầy đủ; đối chiếu prediction agreement và acceptance. Top-K
cache được tính từ forward dense nên phép này đo tác động của chọn cache,
chưa là phép đo tăng tốc. Nó đóng khoảng trống giữa coverage và chất lượng
drafting cho đúng phạm vi nén; sau đó mới khóa protocol pilot cho module.
Không gán rung training cho can thiệp chỉ inference này.

## 12. Artifact, kiểm tra và tái lập

Artifact của phân tích toàn bộ context nằm tại:

```text
outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/full_context_analysis/
```

- `summary.json`: mean/range từng document, mốc đầu/giữa/cuối, so tập chọn
  giữa các mốc và giữa lượt liền kề, provenance/hash.
- `summary.csv`: metrics mean theo cap, cùng mẫu số toàn bộ mass.
- `round_metrics.jsonl`: 4.366 record phân tích qua `JsonlWriter`, kết thúc
  bằng summary success.
- `analysis_audit.json`: tổng mass, tính đơn điệu theo budget, oracle bounds,
  số record và đối chiếu nguồn.
- `figures/`: bốn hình, mỗi hình có PNG 300 DPI và PDF vector; script vẽ được
  lưu cùng hình tại `gen_fig_full_context.py`.

Nhánh cache loại block nằm cạnh đó, tại:

```text
outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis/
```

- `summary.json` / `summary.csv`: coverage cache và tổng mass giữ khi luôn
  giữ block; per-layer, per-document, mốc thời gian và hash nguồn.
- `round_metrics.jsonl`: 4.366 record và summary success; công thức quy đổi
  được tính riêng từng lượt.
- `analysis_audit.json`: phân rã mass, oracle bounds, budget, matching nguồn,
  consistency theo layer và kiểm tra artifact.
- `figures/`: bốn hình bổ sung, mỗi hình có PNG 300 DPI và PDF vector;
  `gen_fig_context_cache.py` lưu mã vẽ.
- `code/` và `report.md`: bản sao mã phân tích/vẽ và báo cáo hiện tại.

Đã đọc lại 4.366 NPZ. Top-1K tính lại khớp JSONL gốc với sai lệch lớn nhất
`4,11e-15`. Tổng mass gốc sai lệch lớn nhất `4,26e-5` so với 1; phân tích
chuẩn hóa lại từng vector trên toàn bộ key. Mọi chiến lược giữ không quá
oracle Top-K cùng budget, Top-K tăng đơn điệu theo budget, và các bin vị trí
cộng đủ 100%. Các số đo theo layer khớp mean vector trong sai số số học.
Không thay đổi collector hoặc dữ liệu inference gốc.

Nhánh cache đã xử lý đủ cùng 4.366 NPZ. Cache mass + block mass và 10 bin
cache đều cộng thành 1; công thức `R = b + m*C` khớp từng record. Top-K
cache tăng đơn điệu, các chiến lược không vượt oracle cùng budget. Mean
coverage tổng + block theo layer lệch tối đa `3,43e-6` so với vector mean;
coverage điều kiện trên cache từng layer không bị ép khớp mean số học.
Đã xem trực tiếp bốn hình bổ sung và kiểm tra các cặp PNG/PDF.

```bash
python3 scripts/analyze_dflash_full_context.py \
  --input outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006
python3 scripts/plot_dflash_full_context.py \
  --input outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/full_context_analysis
python3 scripts/analyze_dflash_context_cache.py \
  --input outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006
python3 scripts/plot_dflash_context_cache.py \
  --input outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/context_cache_analysis
```

Local CPU có thể dùng `.venv/bin/python` như lần phân tích này. Protocol và
runtime GPU của run gốc nằm trong
[báo cáo thu attention](2026-10-06_dflash_attention_10_instances_full_mass.md).
