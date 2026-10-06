# Attention mass của DFlash — thăm dò một instance trên Modal

Ngày chạy: 2026-10-06. Trạng thái: hoàn tất, artifact đã tải về local.

## Thiết lập và phạm vi

- Run chính: `dflash-attention-govreport-l40s-20261006`.
- Modal app: <https://modal.com/apps/tdphuc-work/main/ap-gSqAJRwcji1o3XrqZ8zPan>.
- GPU thực tế: NVIDIA L40S. Target Qwen3-4B, DFlash-b16 5 layer, BF16.
- Target snapshot: `1cfa9a7208912126459214e8b04321603b3df60c`.
- Draft snapshot: `b74e3a329c4d963783143b1e970d95b002be72bd`.
- Runtime: Python 3.12, torch `2.11.0+cu130`, transformers `5.12.1`.
- Target dùng SDPA; drafter dùng eager để lấy attention, với feature layers
  `[1,9,17,25,33]`, native block size 16, greedy, no thinking.
- Một instance `gov_report_dc46b17f87454456` từ
  `data/longbench_200/gov_report.jsonl`, source đầy đủ có 52.521 Qwen token.
- Tạo các prefix source từ cùng instance, giữ instruction/template, đạt input
  chính xác 3.072 / 5.120 / 8.192 / 16.384 token. Không dùng padding.
- Mỗi mức thu tám lượt draft đầu tiên, tổng 32 lượt. Output đã commit lần lượt
  32 / 20 / 14 / 15 token; chưa phải toàn bộ summary.

Job A100 ban đầu chưa được cấp container và đã dừng; smoke và run chính chạy
trên L40S. Smoke riêng hai lượt 3K đã hoàn tất trước run chính. Hardware lấy từ
manifest runtime; dòng log launcher cũ ghi default A100 là nhãn sai và đã được
sửa để in tên GPU thực tế cho các run sau.

## Cách tính

Loại query anchor, trung bình attention qua proposal queries, heads và năm
draft layers. Ba phần prompt/generated prefix/draft block dùng cùng softmax;
không renormalize trước khi tính mass tuyệt đối.

Initial prompt gồm report và instruction/chat template. Histogram chia nó
thành vùng 1.024 token. Heatmap chuẩn hóa riêng trong prompt; tổng các vùng
trên một hàng bằng 1. Coverage bên dưới cũng dùng prompt mass làm denominator.

Top-1K là 1.024 token có mean attention lớn nhất, có thể rải rác. Best-window
là một cửa sổ liên tục 1.024 token có mass lớn nhất. Tính từng lượt rồi lấy
trung bình qua tám lượt, không gộp token giữa các lượt trước khi chọn Top-K.

## Số đo

| Context | Prompt mass trung bình, tuyệt đối | Top-1K coverage trong prompt | Cửa sổ 1K tốt nhất, coverage trong prompt |
|---|---:|---:|---:|
| 3K | 60,74% | 90,87% | 65,59% |
| 5K | 64,07% | 80,56% | 49,18% |
| 8K | 68,77% | 73,25% | 31,93% |
| 16K | 67,35% | 51,10% | 20,67% |

Ở 16K, Top-1K coverage đi từ 55,24% ở lượt đầu xuống 45,91% ở lượt tám.
Mass của vùng 15–16K, sau chuẩn hóa trong prompt, giảm từ 28,50% xuống 16,37%;
vùng 0–1K tăng từ 9,86% lên 12,46%. Các vùng giữa vẫn nhận mass.

Trong instance này, phân bố thay đổi qua các lượt và rộng hơn ở context dài.
Chưa có quan sát rằng 1K token chứa gần như toàn bộ prompt attention ở 16K.
Đây là mô tả số đo, không phải kiểm định về việc context nào cần thiết cho
acceptance hoặc về khả năng học một selector.

Các vị trí attention cao còn gồm token format cuối prompt (`\n\n`, `</think>`)
và `<|im_start|>`. Vì thế không đồng nhất tập token attention cao với evidence
nội dung tài liệu. Các vị trí/token hàng đầu đã được ghi trong từng record.

## Hình và artifact

Directory local:
`outputs/dflash_attention_probe/dflash-attention-govreport-l40s-20261006/`.

- [Histogram absolute mass](../../outputs/dflash_attention_probe/dflash-attention-govreport-l40s-20261006/figures/attention_histograms.png).
- [Heatmap qua lượt draft](../../outputs/dflash_attention_probe/dflash-attention-govreport-l40s-20261006/figures/attention_round_heatmaps.png).
- [Coverage và prompt mass theo lượt](../../outputs/dflash_attention_probe/dflash-attention-govreport-l40s-20261006/figures/attention_concentration.png).
- Có bản PDF tương ứng cho cả ba hình.
- `attention.jsonl` chứa 32 record và summary cuối; `.npz` giữ mean attention
  theo token của từng layer/lượt. `manifest.json` chứa snapshot, source/data
  SHA256, runtime và cấu hình; `summary.json` chứa các lần inference.

Raw artifact được lưu trên Volume Modal `fast-infer-text-sum-cache` tại
`outputs/dflash_attention_probe/dflash-attention-govreport-l40s-20261006/`.
Artifact benchmark và figure nằm trong `outputs/`, không commit vào git.

## Kiểm tra và giới hạn

- Ba kiểm tra aggregation CPU qua: loại anchor/phân chia mass, zero prompt
  mass, và phân biệt scattered Top-K với contiguous window.
- Đã chạy smoke model thật và đủ bốn input trên GPU Modal.
- Cả 32 record có đủ năm layer, array hữu hạn và đúng chiều prompt.
  Sai lệch lớn nhất của tổng mass so với 1 là `2,36e-5`.
- Hình đã được mở để kiểm tra nhãn/trục và thang đo.
- Một document, tám lượt đầu mỗi mức; nội dung source khác nhau vì lấy các
  prefix khác nhau. Chưa có kiểm định population, can thiệp bỏ context,
  acceptance retention, speedup hoặc chất lượng summary đầy đủ.
- Eager instrumentation và elapsed time không dùng để claim latency production.

## Tái lập

```bash
MODAL_GPU=L40S modal run scripts/modal_dflash_attention.py --run-id <run-id-moi> \
  --samples 1 --sample-seed 42 --max-rounds 8 --max-new-tokens 128
python3 scripts/plot_dflash_attention.py \
  --input outputs/dflash_attention_probe/<run-id-moi>
```

Thiết kế rút gọn: [protocol](2026-10-06_dflash_context_support_design.md).
Collector hiện đã mở rộng để lưu mọi key và nhiều instance. Xem
[run 10 instance với full attention mass](2026-10-06_dflash_attention_10_instances_full_mass.md).
