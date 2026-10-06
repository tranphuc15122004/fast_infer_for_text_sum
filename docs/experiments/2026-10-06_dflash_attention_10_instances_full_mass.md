# Attention DFlash trên 10 instance — toàn bộ prompt, output và draft block

Ngày: 2026-10-06. Trạng thái: hoàn tất trên Modal L40S, artifact đã tải về local.

Phân tích mới cho module nén dùng tổng mass trên mọi key làm mẫu số thống
nhất: [báo cáo toàn bộ context](2026-10-06_dflash_full_context_compression_analysis.md).
Báo cáo thu attention này giữ các cột prompt-conditioned để truy vết run.

## Thiết lập

- Run: `dflash-attention-fullmass-govreport10-l40s-20261006`.
- Modal app: <https://modal.com/apps/tdphuc-work/main/ap-Wt7jcrHyiZaiDb7fMEUjOI>.
- 10 document GovReport khác nhau; shuffle với seed 42, chỉ chọn document có
  ít nhất 16.384 source token. Sample IDs và hash source nằm trong manifest.
- Mỗi document tạo bốn prompt 3.072 / 5.120 / 8.192 / 16.384 token, cùng
  instruction/template với run trước. Tổng cộng 40 lần sinh.
- Target Qwen3-4B, checkpoint DFlash-b16 năm layer, BF16; target SDPA và
  drafter eager. Giữ feature layers và native block size như run trước.
- Target snapshot: `1cfa9a7208912126459214e8b04321603b3df60c`.
- Draft snapshot: `b74e3a329c4d963783143b1e970d95b002be72bd`.
- Runtime: Python 3.12, torch `2.11.0+cu130`, transformers `5.12.1`.
- Mỗi lần sinh tới EOS hoặc tối đa 512 output token; guard tối đa 512 lượt
  draft. Chỉ thu lượt draft thật, không tiếp tục sau EOS.

## Phạm vi 100% attention mass

Giữ query scope như run đầu: trung bình qua 15 proposal queries, heads và
năm draft layers, loại anchor query. **Key scope gồm toàn bộ key mà các
proposal queries attend**, không chỉ input ban đầu:

1. Prompt ban đầu: report, instruction và chat template.
2. Output đã commit từ các lượt trước.
3. Draft block hiện tại: một anchor token và 15 mask tokens ở thời điểm
   forward của DFlash. Đây là input của block; không thay mask bằng token dự
   đoán sau forward để diễn giải attention.

Ba phần dùng cùng softmax. Các histogram/heatmap full mass không chuẩn hóa
riêng trong prompt; tổng mỗi lượt gần 1 với sai số BF16. Draft block được
tách tiếp thành anchor và mask để thấy mass trên các vị trí nội bộ.

Top-1K toàn bộ chọn 1.024 key lớn nhất từ prompt + generated prefix + draft
block. Báo riêng Top-1K trong prompt để đối chiếu run cũ. Coverage toàn bộ
có thể lớn nhờ output/anchor/mask có mass cao; cần đọc cùng breakdown thay
vì suy trực tiếp rằng 1K source token đủ cho drafting.

## Hình

- Histogram đầu/giữa/cuối: 1K token mỗi vùng prompt, 128 token mỗi vùng output
  và hai cột anchor/mask. Mọi cột cộng đủ mass; các vùng chưa có output có
  giá trị 0. Độ rộng bin khác nhau được ghi trên nhãn.
- Heatmap theo mọi lượt draft của từng instance, cùng thang màu tuyệt đối.
- Histogram trung bình của 10 instance tại các lượt đầu/giữa/cuối, mỗi
  document có trọng số bằng nhau.
- Phân chia mass theo tiến độ sinh: nội suy mỗi trajectory từ output offset
  của lượt đầu tới lượt cuối rồi lấy mean 10 document.
- Phân bố Top-1K coverage và output mass: mean qua lượt trong từng document,
  sau đó tổng hợp các document với trọng số bằng nhau. Không có kiểm định
  thống kê hoặc confidence interval trong bước thăm dò này.

## Artifact

Local và Volume `fast-infer-text-sum-cache`:
`outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/`.

- `attention.jsonl`: record từng lượt và summary cuối qua `JsonlWriter`.
- `manifest.json`, `summary.json`, `aggregate_full_mass.json`.
- `artifact_audit.json`: kết quả đối chiếu file vector và JSONL.
- `code/`: bản sao collector, plotter và launcher của run này.
- `sample_00/` tới `sample_09/`: input/output token IDs, NPZ mỗi lượt chứa
  mean attention qua heads/proposal queries của từng layer **trên mọi key**.
- `figures/`: hình tổng hợp PNG/PDF và histogram/heatmap riêng từng instance.

Hình tổng hợp:

- [Histogram toàn bộ mass, mean 10 instance](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/attention_histograms_full_mean.png).
- [Phân chia mass theo tiến độ output](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/attention_mass_parts_progress.png).
- [Coverage và phân bố giữa các instance](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/attention_concentration_instances.png).

Heatmap riêng từng instance:
[00](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/sample_00/attention_round_heatmaps_full.png),
[01](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/sample_01/attention_round_heatmaps_full.png),
[02](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/sample_02/attention_round_heatmaps_full.png),
[03](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/sample_03/attention_round_heatmaps_full.png),
[04](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/sample_04/attention_round_heatmaps_full.png),
[05](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/sample_05/attention_round_heatmaps_full.png),
[06](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/sample_06/attention_round_heatmaps_full.png),
[07](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/sample_07/attention_round_heatmaps_full.png),
[08](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/sample_08/attention_round_heatmaps_full.png),
[09](../../outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/figures/sample_09/attention_round_heatmaps_full.png).

Mỗi folder `figures/sample_XX/` có thêm histogram đầu/giữa/cuối. Tổng cộng
23 hình, mỗi hình có bản PNG và PDF; các hình tổng hợp đã được mở để kiểm tra
nhãn, màu và thang đo.

## Kết quả

40/40 lần sinh dừng ở EOS, không chạm giới hạn 512 output token hoặc 512 lượt.
Tổng **4.366 lượt draft**, output dài **111–435 token**, gồm EOS.

Mean được tính qua các lượt của mỗi document, rồi lấy mean của 10 document
với trọng số bằng nhau. Breakdown dùng mass tuyệt đối trên mọi key:

| Input ban đầu | Số lượt draft | Prompt mass | Output đã sinh | Draft block | Top-1K mọi key | Top-1K trong prompt |
|---|---:|---:|---:|---:|---:|---:|
| 3K | 818 | 50,40% | 26,77% | 22,82% | 92,38% | 86,85% |
| 5K | 914 | 53,93% | 24,22% | 21,85% | 83,92% | 72,45% |
| 8K | 1.206 | 55,04% | 22,64% | 22,32% | 76,96% | 60,30% |
| 16K | 1.428 | 58,80% | 19,85% | 21,35% | 67,29% | 46,08% |

Top-1K mọi key dùng toàn bộ mass làm denominator. Top-1K trong prompt có
denominator riêng là prompt mass; hai cột không thể so trực tiếp như cùng
một tỷ lệ. Sai lệch tổng các cột breakdown sau làm tròn là sai số hiển thị.

Ở 16K, mean qua 10 document tại lượt đầu/giữa/cuối lần lượt:

| Vị trí trong trajectory | Prompt | Output đã sinh | Draft block |
|---|---:|---:|---:|
| Lượt đầu | 75,63% | 0,00% | 24,37% |
| Lượt giữa | 60,28% | 19,16% | 20,56% |
| Lượt cuối | 56,87% | 22,91% | 20,23% |

Mean số key lớn nhất cần để bao phủ 90% mass của per-token mean vector:
3K khoảng 772; 5K khoảng 1.706; 8K khoảng 3.066; 16K khoảng 7.035.
Đây là số đo coverage của attention, chưa đồng nghĩa số key cần thiết để giữ
acceptance.

### Output từng document

| Sample | ID GovReport | Output 3K | Output 5K | Output 8K | Output 16K |
|---|---|---:|---:|---:|---:|
| 00 | `dc46b17f87454456` | 196 | 221 | 301 | 211 |
| 01 | `27799eeb558e625e` | 297 | 171 | 215 | 353 |
| 02 | `305866a204d9ad75` | 218 | 259 | 281 | 212 |
| 03 | `fecadba94b064797` | 307 | 279 | 435 | 347 |
| 04 | `1eb67252a22d4e5e` | 195 | 251 | 243 | 362 |
| 05 | `f47f03b6b9122915` | 120 | 141 | 133 | 111 |
| 06 | `bf7ab98e72a6a628` | 165 | 215 | 267 | 204 |
| 07 | `59908fcdbd5e47b9` | 209 | 230 | 263 | 219 |
| 08 | `a0ba54a50db00419` | 275 | 168 | 200 | 199 |
| 09 | `e9c52994f2e2a79c` | 265 | 187 | 141 | 270 |

### Phạm vi diễn giải

Đối chiếu theo layer từ JSONL đã lưu, không chạy thêm inference:

| Draft layer (index từ 0) | Top-1K mọi key ở 16K | Top-1K trong prompt ở 16K |
|---|---:|---:|
| 0 | 64,19% | 45,04% |
| 1 | 75,02% | 45,79% |
| 2 | 85,95% | 72,83% |
| 3 | 67,81% | 54,82% |
| 4 | 76,39% | 67,50% |

Top-1K của vector sau mean cả năm layer là 67,29%. Nếu mỗi layer tự chọn
Top-1K riêng rồi lấy mean coverage, kết quả là 73,87%; các tập token được
chọn có thể khác nhau. Chi tiết nằm trong `layer_breakdown.json` ở artifact
directory. Mỗi vector riêng layer vẫn đã mean qua heads và proposal queries.

Hình gộp phù hợp để mô tả phân bổ mass tổng quan và làm proxy ban đầu nếu
muốn chọn cùng một tập context cho các layer. Coverage sau mean không mô tả
độ tập trung của từng layer/head/query. Các nhận định "phân bố trải rộng" và
"khoảng 7K key cho 90% mass ở 16K" ở trên áp dụng cho vector đã tổng hợp.

Phần output và draft block chiếm mass đáng kể. Vì thế histogram chỉ hiển thị
prompt trước đây chưa phản ánh toàn bộ phân bố attention. Ở 16K, Top-1K mọi
key có coverage cao hơn Top-1K trong prompt; các key của output/anchor/mask
đóng góp vào tập chọn trên toàn bộ sequence.

Đây là đo allocation của attention, chưa can thiệp bỏ key hoặc kiểm định
ảnh hưởng tới acceptance. Trung bình qua head/layer/proposal có thể che
specialization; NPZ giữ riêng từng layer để xem tiếp. Prompt gồm format và
instruction, không chỉ evidence của report. Bốn mức input dùng các prefix
source khác nhau nên chưa quy khác biệt chỉ cho độ dài.

### Phân tích bổ sung từ artifact đã lưu

Các số đo dưới đây được tính trên CPU từ JSONL và NPZ của run gốc, không
chạy thêm inference. Artifact bổ sung local:
`additional_attention_analysis.json` và `attention_topk_coverage.json`/`.csv`.

| Input | Prompt mass | Output đã sinh | 15 vị trí mask | Anchor |
|---|---:|---:|---:|---:|
| 3K | 50,40% | 26,77% | 21,54% | 1,29% |
| 5K | 53,93% | 24,22% | 20,73% | 1,12% |
| 8K | 55,04% | 22,64% | 21,25% | 1,07% |
| 16K | 58,80% | 19,85% | 20,35% | 1,00% |

Mask mass gần 20–21% khi thay đổi độ dài prompt; output mass giảm dần theo
các cap. Theo tiến độ sinh, output mass vẫn tăng từ 0 ở lượt đầu lên 22,90%
ở lượt cuối của 16K; mask mass giảm từ 23,32% xuống 19,29%. Vì thế sự ổn
định sau giai đoạn đầu không đồng nghĩa một tỷ lệ cố định cho mọi lượt.

15 vị trí mask ở 16K nhận 20,35% mass, trong khi 16.384 vị trí prompt nhận
58,80%. Tỷ lệ mass trung bình trên một key của hai nhóm chênh khoảng 378
lần. Đây là mức tập trung theo vị trí, chưa đo đóng góp của chúng tới
logits. Hidden states tại vị trí mask tiếp tục được cập nhật qua các layer.
Phân bổ theo layer cũng khác nhau: mask mass của layer 0–4 ở 16K lần lượt
là 7,09% / 38,01% / 35,28% / 13,55% / 7,79%.

Trong prompt 16K, Top-1K token rời rạc chứa 46,08% prompt mass, nhưng cửa
sổ 1K token liên tiếp tốt nhất chỉ chứa 18,02%. Điều này gợi ý các vị trí
có score cao phân bố ở nhiều vùng. Histogram dùng bin 1K không đủ để kết
luận một hoặc vài cửa sổ liên tiếp bao phủ hết các key có score cao.

Đối chiếu tập Top-1K trong cùng prompt qua các mốc của mỗi trajectory:

| Input | Trùng vị trí: đầu–giữa | Trùng vị trí: đầu–cuối | Trùng vị trí: giữa–cuối |
|---|---:|---:|---:|
| 3K | 56,47% | 58,56% | 55,61% |
| 5K | 51,95% | 49,62% | 49,87% |
| 8K | 33,14% | 37,22% | 36,04% |
| 16K | 28,14% | 29,96% | 29,59% |

Overlap ở đây là số vị trí chung chia 1.024, không phải Jaccard. Chọn Top-K
trên vector đã mean layers/heads/proposals; chỉ xét prompt ban đầu và lấy
mean bằng nhau qua 10 document. Các vị trí được chọn có thể rời rạc.

Ở lượt cuối của 16K, giữ tập 1K đã chọn tại lượt giữa chỉ bao phủ 28,96%
prompt mass, so với 44,08% nếu chọn lại Top-1K ở lượt cuối. Chênh lệch này
cho thấy mass tổng của các nhóm có thể ổn định trong khi vị trí được ưu
tiên vẫn thay đổi. Đây là động cơ khảo sát module chọn context phụ thuộc
trạng thái sinh; so sánh ba mốc chưa xác định tần suất cập nhật cần thiết.

### Coverage Top-1K và Top-4K

Đọc lại toàn bộ 4.366 NPZ; với mỗi lượt, mean năm layer rồi chọn 1.024 hoặc
4.096 key có weight cao nhất. Tập chọn được phép khác nhau giữa các lượt;
đây là phép chọn dựa trên attention quan sát được, chưa phải selector triển
khai trước attention. Mean qua lượt của từng document, rồi mean 10 document
với trọng số bằng nhau. Top-1K tính lại khớp JSONL với sai lệch 0.

| Input | Top-1K mọi key / tổng mass | Top-4K mọi key / tổng mass | Top-1K prompt / prompt mass | Top-4K prompt / prompt mass |
|---|---:|---:|---:|---:|
| 3K | 92,38% | 100,00% | 86,85% | 100,00% |
| 5K | 83,92% | 98,89% | 72,45% | 98,29% |
| 8K | 76,96% | 93,59% | 60,30% | 89,02% |
| 16K | 67,29% | 82,61% | 46,08% | 71,06% |

Ở 3K, tổng số key của mỗi lượt nhỏ hơn 4.096 nên Top-4K bao gồm tất cả key.
Hai cặp cột dùng denominator khác nhau. Tập chọn trên mọi key có thể bao
gồm output và block với score cao, vì vậy không thể diễn giải 82,61% ở 16K
là coverage của riêng 4K source token.

Nếu chọn Top-1K hoặc Top-4K **trong prompt** và giữ toàn bộ output cùng draft
block, tổng mass được giữ lại ở 16K lần lượt là 68,39% và 83,04%. Các số này
vẫn là coverage của attention đã tổng hợp, chưa dự đoán acceptance khi bỏ
key. Budget 4K trên prompt 16K tương ứng giữ 25% số vị trí prompt để bao phủ
71,06% prompt mass; phần còn lại vẫn chứa khoảng 28,94% prompt mass.

Tái lập phân tích Top-K bằng script lưu cùng artifact:

```bash
python3 outputs/dflash_attention_probe/dflash-attention-fullmass-govreport10-l40s-20261006/code/analyze_topk.py
```

### Kiểm tra số đo

- Collector đã nhận đủ năm draft layers cho từng lượt, mọi per-key vector
  hữu hạn và không âm.
- 4.366/4.366 record có `key_tokens = prefix_tokens + 16`.
- Sai lệch lớn nhất của tổng mass sau mean layers so với 1: `4,26e-5`,
  tương đương khoảng 0,0043 điểm phần trăm.
- Sai lệch lớn nhất ở từng layer: `1,07e-4`.
- Plotter kiểm tra tổng prompt bins + output bins + anchor + mask bằng
  `mass_sum`, sai lệch không vượt `1e-5`, trước khi vẽ.
- Đã đọc lại toàn bộ 4.366 NPZ: shape `(5, prefix_tokens + 16)` đúng, vector
  hữu hạn/không âm, mean mass khớp record với sai lệch 0. Output token IDs
  khớp chiều dài summary; SHA256 collector khớp manifest.

## Tái lập

```bash
MODAL_GPU=L40S modal run scripts/modal_dflash_attention.py \
  --run-id <run-id-moi> --samples 10 --sample-seed 42 \
  --caps 3072,5120,8192,16384 --max-rounds 512 --max-new-tokens 512
python3 scripts/plot_dflash_attention.py \
  --input outputs/dflash_attention_probe/<run-id-moi>
```

Run đầu trên một instance và tám lượt:
[báo cáo](2026-10-06_dflash_attention_probe_results.md).
