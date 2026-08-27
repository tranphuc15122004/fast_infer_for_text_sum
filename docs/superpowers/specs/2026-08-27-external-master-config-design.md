# Thiết kế cấu hình master bên ngoài repository

## Mục tiêu

Thay thế toàn bộ các file `config/<baseline>.env` bằng đúng hai đầu vào cấu
hình do người dùng quản lý:

1. `config/master.path` trong repository, chỉ chứa đường dẫn tới master config.
2. Một file shell-env master đặt ngoài repository, ví dụ
   `/workspace/shared_storage/config/fast_infer_master.env`.

Sau khi thiết lập pointer một lần, việc cập nhật source code không cần copy lại
các model, dataset hoặc nhiều file cấu hình baseline lên server B200.

## Phạm vi

- Tất cả launcher trong `scripts/run_*.sh`, dispatcher, B200 smoke runner và
  representative runner đọc cùng master config.
- Master config dùng cú pháp POSIX/Bash environment assignment, không thêm
  dependency YAML/TOML.
- Config key mới có namespace rõ ràng theo nhóm: `FI_` cho runtime/cache,
  `MODEL_` cho model, `DATA_` cho dữ liệu, `RUN_` cho execution chung và tên
  baseline viết hoa cho tham số riêng baseline.
- Xóa các config per-baseline khỏi repository và khỏi luồng thực thi.
- Giữ các command entrypoint chính như `bash scripts/run.sh dflash` và
  `bash scripts/run_representative_100.sh`.

## Không nằm trong phạm vi

- Không chuyển các tham số inference vào Python code hoặc hard-code model path.
- Không tự tải model, dataset, package hay checkpoint trên server offline.
- Không thay đổi thuật toán, output schema hoặc metric của baseline.
- Không tạo master config thật ngoài workspace sandbox; server operator tạo file
  đó từ template tài liệu và giữ nó ngoài repository.

## Cấu trúc và precedence

`config/master.path` là plain text, không được `source` trực tiếp. Loader đọc
dòng không rỗng đầu tiên, resolve path tương đối theo project root, rồi source
master config trong một shell environment được export.

Precedence của giá trị runtime:

```text
CLI của launcher/infer
  > biến override đã export trong process
  > giá trị canonical trong master config
  > default an toàn của loader/launcher
```

Biến `FAST_INFER_MASTER_CONFIG` có thể override pointer cho test hoặc staging.
Nếu pointer/master thiếu, loader dừng với thông báo có đường dẫn và hướng dẫn
cụ thể; không fallback âm thầm về các config cũ.

## Naming convention

Master config dùng các nhóm sau:

```bash
# Runtime/cache/offline
FI_PYTHON
FI_DEVICE
FI_GPU_IDS
FI_TARGET_GPU
FI_HF_HOME
FI_TRANSFORMERS_CACHE
FI_TRITON_CACHE
FI_FLASHINFER_CACHE
FI_TORCH_EXTENSIONS_CACHE
FI_OFFLINE

# Canonical models
MODEL_TARGET
MODEL_DFLASH_DRAFT
MODEL_EAGLE_DRAFT
MODEL_SPEC_DRAFT
MODEL_LONGSPEC_TARGET
MODEL_LONGSPEC_DRAFT
MODEL_COMPRESSOR
MODEL_EMBEDDING
CHECKPOINT_MAGICDEC

# Shared input/output/execution
DATA_INPUT
DATA_ROOT
OUTPUT_ROOT
RUN_MODE
RUN_SAMPLES
RUN_MAX_NEW_TOKENS
RUN_MAX_INPUT_TOKENS
RUN_TEMPERATURE

# Baseline-specific settings
DFLASH_MODE
DFLASH_BACKEND
DFLASH_DATASET
DFLASH_BLOCK_SIZE
FASTKV_METHOD
FASTKV_ATTN_IMPL
GEMFILTER_TOPK
LLMLINGUA_COMPRESSION_RATE
MINFERENCE_ATTN_TYPE
SPECPREFILL_CONFIG
SEMANTIC_SELECTORS
SEMANTIC_TOKEN_BUDGETS
...
```

Loader map canonical names sang các biến legacy mà Python adapter hiện cần,
ví dụ `MODEL_TARGET` thành `MODEL`, `TARGET_MODEL` hoặc `BASE_MODEL` tùy
baseline; `DATA_INPUT` thành `DATA_FILE`, `DOC_FILE` hoặc `INPUT_FILE`.
Master không chứa các tên legacy generic để tránh xung đột giữa baseline.

## Components

### `scripts/common/config.sh`

API shell công khai:

```bash
fast_infer_load_master
fast_infer_load_config <baseline>
fast_infer_master_path
```

`fast_infer_load_master` đọc pointer, source master một lần và export runtime
compatibility aliases. `fast_infer_load_config` gọi hàm đó rồi map model,
dataset, output, smoke/full và tham số baseline vào biến mà launcher tương ứng
đang dùng. Các biến đã được caller export trước đó được giữ nguyên để runner
có thể tạo override cho từng dataset/sample mà không sinh config file.

### Launcher

Mỗi launcher source `scripts/common/config.sh`, gọi
`fast_infer_load_config <baseline>`, rồi source `runtime.sh`. Không launcher
nào source `config/<baseline>.env` hoặc nhận config per-baseline làm nguồn
chính nữa.

### Representative/B200 runner

- `run_representative_100.sh` đọc các biến `BENCH_*` canonical trong master và
  truyền override bằng environment cho child launcher; không tạo
  `outputs/.../configs/*.env`.
- `run_b200_smoke.sh` đọc cùng master. `b200_smoke.py` truyền child overrides
  trực tiếp qua `env`, vẫn lưu log/output/generated input nhưng không lưu
  overlay config.
- B200 compatibility aliases (`B200_TARGET_MODEL`, `B200_DFLASH_MODEL`,
  `B200_DATA_FILE`, ...) được loader tạo từ canonical master values để preflight
  hiện tại tiếp tục dùng được.

## Offline và validation

Master đặt `FI_OFFLINE=1`; loader export `HF_HUB_OFFLINE=1` và
`TRANSFORMERS_OFFLINE=1`. Loader kiểm tra pointer/master; wrapper kiểm tra
asset bắt buộc khi baseline đã có validation riêng. Preflight B200 vẫn kiểm tra
CUDA, imports, model directories, dataset và cache writable.

Tests phải bao phủ:

- pointer path được đọc đúng và env override có ưu tiên;
- master thiếu hoặc pointer rỗng báo lỗi rõ;
- canonical keys map đúng cho DFlash và ít nhất các nhóm model/data/output;
- caller override không bị master ghi đè;
- tất cả launcher source loader, không source config cũ;
- representative/B200 không còn sinh overlay config;
- shell syntax, Python compile và toàn bộ contract tests.

## Migration policy

Các file cấu hình cũ bị xóa khỏi repository sau khi launcher đã chuyển sang
loader. Documentation chỉ hướng dẫn master config. `bash scripts/run.sh
<baseline>` là entrypoint duy nhất; mode đặc biệt như DFlash GSM8K được chọn
bằng `DFLASH_MODE=gsm8k` trong master.
