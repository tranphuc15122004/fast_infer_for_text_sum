# Hồ sơ server benchmark

Đây là nơi canonical lưu thông tin môi trường server của project. Khi làm
việc trên server, ưu tiên tài liệu này thay vì tự suy đoán đường dẫn từ máy
local.

## Đường dẫn cố định

```text
Repository:
/workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

Shared data/config:
/workspace/storage-shared/nlp/dungdx4/phuc_projects/data

Master config:
/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env

LongBench canonical:
/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/longbench_200

Legacy representative data:
/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/representative_100
```

Tên dataset đúng là `representative_100`.

## Runtime server

- Server benchmark dùng trực tiếp lệnh `python3` từ `PATH`.
- `python3` trên server là Python 3.12.
- Không tạo hoặc activate virtualenv trên server.
- `.venv` trong workspace local chỉ dùng để mô phỏng/debug trước khi đưa code
  lên server.
- Dependency, CUDA kernel, model và checkpoint phải được cài/mirror sẵn vì
  server không có internet trực tiếp.
- Master config phải đặt `FI_PYTHON=python3`, `FI_DEVICE=cuda` và
  `FI_OFFLINE=1`.

## Khởi tạo/kiểm tra

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

# Tạo thư mục/link/master nếu còn thiếu; không chạy preflight.
python3 scripts/setup_server_env.py --init

# Kiểm tra Python 3.12, package, master config và LongBench.
python3 scripts/setup_server_env.py --check

# Hoặc init + check trong một lần.
python3 scripts/setup_server_env.py --all
```

Script không ghi đè `fast_infer_master.env` đã tồn tại và không xoá dataset
đã checkout. Nếu master config đã có, operator chỉnh các đường dẫn model,
draft model, MagicDec `.pth` và SSSD datastore trực tiếp trong file đó.

## Chạy benchmark LongBench

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

python3 scripts/run_longbench_200.sh \
  --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env \
  --mode smoke \
  --run-id b200-smoke-all
```

Sau khi smoke pass, dùng `--mode representative` hoặc `--mode full`. Kết quả
được ghi dưới `outputs/longbench_200/<run-id>/`; xem `run_manifest.json` và
`logs/` để kiểm tra từng cell.

Các lỗi import/tương thích đã được xử lý trong source vendored và adapter nên
không cần cài thêm `dflash`, `MagicDec`, `termcolor` hoặc `fastchat` từ
internet. Sau khi đồng bộ code lên server, chạy lại:

```bash
python3 -m py_compile \
  scripts/infer_dflash.py scripts/infer_magicdec.py scripts/infer_longspec.py \
  scripts/common/model_compat.py
python3 scripts/setup_server_env.py --check
```

SSSD là ngoại lệ về dữ liệu: datastore `.idx` là artifact riêng, không nằm
trong repository. Để smoke không bị bỏ qua, có thể để trống
`SSSD_DATASTORE_PATH`; để benchmark retrieval công bằng, phải điền đường dẫn
`.idx` được build cho Llama 3.1 và tokenizer đang dùng.

## Ghi chú bảo mật

Không commit `HF_TOKEN`, thông tin đăng nhập, hoặc đường dẫn chứa secret vào
repository. Chỉ commit tài liệu path/runtime ổn định và các file cấu hình mẫu.
