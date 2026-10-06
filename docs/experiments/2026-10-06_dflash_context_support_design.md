# Thăm dò attention mass của DFlash trên context dài

Ngày: 2026-10-06. Trạng thái: đã triển khai collector/plotter và launcher
Modal; [run một instance trên L40S đã hoàn tất](2026-10-06_dflash_attention_probe_results.md).
Thiết kế này thay protocol can thiệp/holdout lớn trước đó trong cùng file.

## Mục tiêu

Quan sát drafter phân bổ attention lên context ở đâu, mức tập trung ra sao và
vùng được attend có đổi giữa các lượt draft không. Chạy ít để xem hình trước;
người nghiên cứu sẽ quyết định phép kiểm định và module học ở bước sau.

## Quy mô

- Một instance summarization có đủ ít nhất khoảng 16K token; có thể thêm
  một hoặc hai instance khác nếu cần kiểm tra hình dạng phân bố.
- Dùng cùng instance, tạo bốn input khoảng 3K, 5K, 8K, 16K token sau tokenizer
  và chat template. Các mốc tương ứng 3.072, 5.120, 8.192, 16.384 token.
- Giữ instruction/chat template, lấy prefix source để đạt mỗi mốc; ghi input
  length thực tế. Không pad để tạo context dài giả. Nếu instance không đủ 16K,
  chọn instance dài khác từ dữ liệu đã mirror.
- Target Qwen3-4B và checkpoint DFlash tương ứng đã cache; native block size,
  greedy, no thinking. Ghi model snapshot và tokenizer dùng thực tế.
- Mỗi input chạy tối đa 8 lượt draft, giới hạn tối đa 128 output token; dừng
  tại EOS nếu đến sớm. Thu tất cả lượt thực sự chạy, không ép qua EOS.

Một instance tương ứng bốn lượt inference. Đây là thăm dò hình dạng phân bố,
không phải benchmark latency hay phép kiểm định trên population. Bốn mức lấy
prefix source khác nhau nên nội dung cũng thay đổi; chưa quy khác biệt giữa
các mức chỉ cho độ dài context.

## Thu attention

Đo attention của **drafter** ở tất cả draft layers/query heads, trên các query
dự đoán proposal token; loại query anchor khỏi phép trung bình. Giữ inference
dense thông thường; mỗi lượt draft có query/state thật do pipeline tạo ra.

Target dùng SDPA hoặc backend hiện có. Chỉ DFlash drafter dùng eager attention
để lấy attention weights. Model upstream trả attention ở self-attention nhưng
decoder bỏ weights; collector riêng có thể dùng forward hook trên
`draft.layers[i].self_attn` để aggregate ngay khi forward, tránh sửa baseline.

Với attention weights `a[layer,head,proposal,key]`, báo ba phần mass của mỗi
lượt: initial prompt, generated prefix đã commit và draft block hiện tại.
Tổng ba phần bằng 1, sai số số học được ghi lại. Initial prompt gồm document
và format/instruction; tên trục phải ghi rõ phạm vi này.

Trong phần initial prompt, chia theo absolute position thành các vùng 1.024
token. Mass một vùng là tổng attention lên các key trong vùng, trung bình
qua proposal queries, heads và layers. Trung bình này cho overview; lưu thêm
mass theo layer để xem có bị overview che khuất hay không.

Giữ hai thang đo:

- **Mass tuyệt đối:** tổng các cột prompt bằng prompt attention mass, không
  nhất thiết bằng 1 vì drafter còn attend generated prefix và draft block.
- **Mass điều kiện trên prompt:** chia các cột cho tổng prompt mass, để so
  vùng nào được ưu tiên. Nếu prompt mass bằng 0 thì ghi không xác định.

Lưu thêm mean attention theo từng prompt token. Từ đó tính tỷ lệ prompt mass
nằm trong 1.024 token được attend nhiều nhất; tập này có thể rải rác. Nếu muốn
xem một vùng liên tục, báo riêng cửa sổ 1K có mass cao nhất. Đó là hai số khác
nhau. Không chọn/bỏ context hoặc thay dự đoán trong study này.

## Hình để xem

1. **Histogram theo vị trí context:** cột là vùng 1K, chiều cao là attention
   mass. Với mỗi mức context, hiển thị một vài lượt đầu/giữa/cuối thực tế, ghi
   round ID và output offset; dùng cùng thang Y. Có đường tham chiếu theo số
   token của vùng nếu cần, vì vùng cuối có thể ngắn hơn 1K.
2. **Heatmap theo lượt draft:** X là vị trí/vùng 1K của initial prompt, Y là
   round ID; màu là prompt-normalized mass. Ghi prompt mass tuyệt đối bên
   cạnh để tránh hiểu nhầm một heatmap đậm nghĩa là source đóng góp nhiều.

Trên histogram hoặc bảng nhỏ kèm hình, ghi prompt/generated/draft mass và
Top-1K prompt mass coverage ở mỗi lượt. Chỉ thêm bản theo từng layer khi
overview cho thấy pattern đáng xem. Xuất PNG và PDF bằng matplotlib.

## Artifact và thực thi

Collector: `scripts/probe_dflash_attention.py`; plotter:
`scripts/plot_dflash_attention.py`; launcher: `scripts/modal_dflash_attention.py`.

```bash
modal run scripts/modal_dflash_attention.py --run-id <run-id> \
  --samples 1 --max-rounds 8 --max-new-tokens 128
python3 scripts/plot_dflash_attention.py \
  --input outputs/dflash_attention_probe/<run-id>
```

Launcher dùng Volume `fast-infer-text-sum-cache`, nạp model offline, commit
artifact và tải bản sao về local. GPU mặc định A100-80GB; `MODAL_GPU` override.

Output dưới `outputs/dflash_attention_probe/<run_id>/`, gitignored:
manifest, attention records JSONL qua `io_util.JsonlWriter` kết thúc bằng
summary, mean per-token attention arrays và các figure. Aggregate khi hook
chạy; không lưu toàn bộ tensor layer × head × query × key của mọi lượt.

Chạy trên GPU Modal hoặc server được cấp phát. Máy local chỉ dev CPU và vẽ
hình từ artifact. Eager drafter là instrumentation,
không dùng timing của run này để kết luận tốc độ production.

## Cách đọc kết quả

Nếu mass tập trung vào ít vùng và các vùng thay đổi theo lượt, đó là tín hiệu
để khảo sát module chọn context theo query. Nếu vùng quan trọng ổn định,
selection một lần hoặc refresh thưa có thể đáng thử. Nếu mass trải rộng,
budget 1K có thể chưa phù hợp.

Attention mass chỉ mô tả cách drafter phân bổ trọng số; chưa chứng minh bỏ các
vùng ít mass sẽ giữ acceptance. Phép kiểm định đó được quyết định sau khi xem
hình, đúng phạm vi thăm dò hiện tại.
