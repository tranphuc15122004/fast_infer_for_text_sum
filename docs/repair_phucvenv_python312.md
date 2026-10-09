# Khôi phục phucvenv khi container đổi Python hệ thống

## Nguyên nhân đã xác nhận trên server ngày 09/10/2026

`phucvenv/pyvenv.cfg` ghi Python 3.12.13, `home=/usr/bin` và còn
`lib/python3.12/site-packages`. Nhưng `bin/python3` là symlink tới
`/usr/bin/python3`; container hiện tại resolve đường dẫn này thành Python
3.10.6. Các alias `python` và `python3.12` cùng trỏ tới `python3`, nên tên
`python3.12` vẫn chạy Python 3.10. Không tìm thấy runtime 3.12 ở các vị trí
system/venv đã kiểm tra. Đây là mismatch giữa venv trên shared storage và
interpreter của container hiện tại, không phải lỗi model AMR.

Workspace dev có runtime CPython 3.12.13 Linux x86_64 GNU đầy đủ. Gói offline
`outputs/offline_python312/amr_python312_linux_x86_64_20261009.tar.gz` chứa
runtime này và helper `repair_python312_venv.py`. Base third-party
`site-packages`, pip launcher và các launcher có shebang cũ được loại khỏi
gói; thư viện chuẩn, extension modules, headers và dữ liệu runtime được giữ.
Gói không chứa model, dataset, CUDA stack hoặc dependency ML của server.

## Chạy trên server

Tải gói từ workspace và đưa vào:

```text
/workspace/storage-shared/nlp/dungdx4/phuc_projects/amr_python312_linux_x86_64_20261009.tar.gz
```

Gói dành cho **Linux x86_64 GNU**. Kiểm tra `uname -m` trước khi chạy; không
dùng cho server ARM/aarch64. Các khối dưới dùng ASCII để tránh lỗi paste.

```bash
uname -m
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects
mkdir -p python_runtimes/amr_python312_20261009
tar -xzf amr_python312_linux_x86_64_20261009.tar.gz -C python_runtimes/amr_python312_20261009

AMR_RUNTIME_DIR="$PWD/python_runtimes/amr_python312_20261009"
AMR_BASE_PYTHON="$AMR_RUNTIME_DIR/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12"
"$AMR_BASE_PYTHON" -I -c 'import sys, ssl, ctypes, sqlite3; assert sys.version_info[:3] == (3,12,13); print(sys.version)'
```

Chỉ tiếp tục khi interpreter này chạy được. Không di chuyển hoặc xóa thư mục
runtime sau khi sửa venv: cấu hình/symlink mới sẽ tham chiếu tới nó.

```bash
"$AMR_BASE_PYTHON" -I "$AMR_RUNTIME_DIR/repair_python312_venv.py" --venv /workspace/storage-shared/nlp/dungdx4/phuc_projects/phucvenv
```

Helper kiểm tra venv gốc là 3.12 và còn site-packages; backup `pyvenv.cfg`, các
alias Python và activation scripts; rồi dùng `venv.EnvBuilder` của runtime
3.12 đầy đủ để dựng lại bindings/config. Không dùng `clear`, không cài pip,
không xóa hoặc cài lại dependency ML. Nếu rebuild hoặc probe thất bại, helper
khôi phục các file đã backup. Đường dẫn backup được in trong JSON kết quả.
Probe kiểm tra stdlib, Python version và venv prefix; chưa kiểm tra Torch/GPU.

Kiểm tra dependency của venv đã khôi phục:

```bash
AMR_VENV=/workspace/storage-shared/nlp/dungdx4/phuc_projects/phucvenv
"$AMR_VENV/bin/python" -I -c 'import sys, torch; from transformers import AutoTokenizer, Qwen3ForCausalLM; print("Python:", sys.version); print("Torch:", torch.__version__); print("CUDA:", torch.cuda.is_available())'
```

Lệnh phải import thành công và báo CUDA available. Nếu lỗi dependency/native
library, xử lý đúng package/stack được báo từ venv 3.12; không chạy pip của
system Python 3.10 và không copy `site-packages` 3.10 vào venv này.

Nạp master **trước** rồi chọn executable venv để không bị master cũ ép về
`/usr/bin/python3`:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main
export FAST_INFER_MASTER_CONFIG=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env
source scripts/common/config.sh
fast_infer_load_config amr_dflash
export FAST_INFER_PYTHON=/workspace/storage-shared/nlp/dungdx4/phuc_projects/phucvenv/bin/python
hash -r
"$FAST_INFER_PYTHON" --version
```

Đổi runtime thành công chưa tạo manifest AMR. Run cũ bị overlap vẫn cần
clean validation và prepare-data lại theo
[runbook dữ liệu](amr_dflash_training_data.md), dùng đúng repository path trên
server hiện tại.

## Bằng chứng local

Đã kiểm tra gói giải nén sang một đường dẫn khác: Python 3.12.13 cùng
ssl/ctypes/sqlite3/venv hoạt động. Đã tái hiện venv 3.12 bị symlink sang Python
hệ thống sai phiên bản, chạy helper trong gói, rồi xác nhận ba alias đều chạy
3.12 và module cài sẵn được giữ nguyên. Test riêng kiểm tra rollback khi
rebuild thất bại và từ chối venv thuộc minor version khác. CUDA/B200 và các
dependency thực trong phucvenv server chưa được kiểm chứng từ workspace này.

`scripts/common/runtime.sh` cũng đã sửa để trả failure ngay khi Python không
phải 3.12. Trước đây bước tạo cache sau đó có thể làm source trả success và
launcher vẫn chạy với interpreter sai. Đồng bộ file này khi cập nhật code.
