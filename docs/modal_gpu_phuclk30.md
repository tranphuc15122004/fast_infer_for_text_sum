# Dùng GPU Modal bằng workspace `phuclk30`

Modal CLI đã được xác thực bằng `modal setup` và token được lưu trong profile
`phuclk30` tại `~/.modal.toml`. Các job GPU của repo cần chọn profile này rõ
ràng để không gửi nhầm sang workspace `tdphuc-work`.

## Kiểm tra profile trước khi chạy

```bash
modal profile list
modal profile current
```

Lệnh thứ hai cần in `phuclk30`. Nếu cần đổi profile mặc định trong terminal:

```bash
modal profile activate phuclk30
```

Khi chạy job, vẫn nên truyền `--profile phuclk30` trực tiếp vào lệnh Modal.
Tham số này làm rõ workspace đích ngay cả khi profile mặc định trong terminal
hoặc shell khác. Không đưa nội dung token, file `~/.modal.toml`, hay URL
token-flow vào repo, log chia sẻ hoặc lệnh lưu trong tài liệu.

## Chạy GPU cho LongBench

Từ thư mục gốc repo, chạy preflight nhẹ trước để kiểm tra cấu hình ứng dụng:

```bash
MODAL_GPU=A100-80GB modal run --profile phuclk30 \
  scripts/modal_longbench.py \
  --mode representative \
  --baselines vanilla_hf \
  --datasets "gov_report lcc" \
  --max-samples 20 \
  --preflight-only
```

Sau đó có thể chạy smoke một mẫu trên GPU:

```bash
MODAL_HF_SECRET=huggingface MODAL_GPU=A100-80GB \
  modal run --profile phuclk30 scripts/modal_longbench.py \
  --mode smoke \
  --baselines vanilla_hf \
  --datasets gov_report \
  --max-samples 1 \
  --max-new-tokens 8 \
  --run-id phuclk30-vanilla-smoke
```

Khi smoke thành công, chạy benchmark representative. Chọn `MODAL_GPU` phù hợp
với baseline và ngân sách workspace:

```bash
MODAL_HF_SECRET=huggingface MODAL_GPU=A100-80GB \
  modal run --profile phuclk30 scripts/modal_longbench.py \
  --mode representative \
  --baselines "vanilla_hf eagle3 dflash" \
  --datasets "gov_report lcc" \
  --max-samples 20 \
  --run-id phuclk30-longbench-representative-01
```

`MODAL_HF_SECRET` chỉ cần thiết khi model/dataset cần Hugging Face token và
Secret `huggingface` đã tồn tại trong workspace `phuclk30`. Tạo Secret trong
workspace được chọn bằng:

```bash
modal secret create --profile phuclk30 huggingface HF_TOKEN=hf_...
```

Không đặt token trực tiếp trong script hoặc file được commit.

## Đọc output từ đúng workspace

Volume và artifact được giới hạn theo workspace. Truy vấn và tải file với cùng
profile đã dùng để chạy:

```bash
modal volume ls --profile phuclk30 fast-infer-text-sum-cache outputs/longbench_100_14k
modal volume get --profile phuclk30 fast-infer-text-sum-cache \
  outputs/longbench_100_14k/phuclk30-longbench-representative-01 \
  outputs/phuclk30-longbench-representative-01
```

Nếu báo không tìm thấy Volume hoặc Secret, kiểm tra profile trước. Đừng tạo
resource trùng tên ngay; resource cùng tên ở `tdphuc-work` và `phuclk30` là
resource của hai workspace khác nhau.

## Script thí nghiệm riêng

Với script Modal khác trong repo, dùng dạng:

```bash
MODAL_GPU=L40S modal run --profile phuclk30 scripts/<ten_script>.py ...
```

Một số script có bước tải artifact bằng `modal volume get` từ Python. Đặt
`MODAL_PROFILE=phuclk30` cho bước đó, ví dụ:

```bash
MODAL_PROFILE=phuclk30 MODAL_GPU=L40S \
  modal run --profile phuclk30 scripts/modal_dflash_target_comparison.py ...
```

Các đường dẫn app `tdphuc-work` trong experiment log cũ là dấu vết của những
lần chạy trước; chúng không xác định profile cho job mới. Luôn xem lệnh chạy
hiện tại và kiểm tra `modal profile current` khi chẩn đoán job.
