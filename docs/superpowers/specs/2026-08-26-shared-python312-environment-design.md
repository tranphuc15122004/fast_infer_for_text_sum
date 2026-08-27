# Shared Python 3.12 Environment Migration

## Mục tiêu

Chuyển toàn bộ quy trình chạy chính của `fast_infer_text_sum` sang một venv
Python 3.12 duy nhất, cài dependency từ `requirements.txt` trong điều kiện
server GPU không có kết nối internet trực tiếp, đồng thời loại bỏ các môi
trường uv hiện tại và kiểm tra các entrypoint hiện có.

## Phạm vi

Bao gồm launcher trong `scripts/`, các subprocess do launcher/infer script tạo,
cấu hình setup, tài liệu vận hành và contract tests. Không thay đổi mã nguồn
vendored của từng baseline trừ khi cần để launcher dùng đúng interpreter chung.

Các model, dataset và checkpoint không được tải tự động trong migration. Smoke
test chỉ được coi là pass khi artefact/model cần thiết đã có trong cache; thiếu
cache phải được ghi nhận là blocker riêng.

## Quyết định kiến trúc

### Một venv làm nguồn runtime duy nhất

Venv mặc định đặt tại `.venv/` của repo và phải dùng Python 3.12. Người vận hành
có thể trỏ tới venv khác bằng `FAST_INFER_VENV` hoặc tới executable cụ thể bằng
`FAST_INFER_PYTHON`. Runtime helper kiểm tra Python major/minor trước khi chạy;
interpreter khác 3.12 bị từ chối để tránh chạy nhầm venv.

Thư viện được cài bằng `uv pip` vào venv, nhưng execution dùng trực tiếp
executable của venv. `requirements.txt` là nguồn dependency duy nhất; không còn
root uv project hoặc project uv riêng cho từng baseline.

### Offline-first

Setup không gọi installer qua network và không tự tải Python/package. Nó dùng
Python 3.12 đã có trên máy cùng cache/wheelhouse cục bộ của uv. Các direct URL,
local wheel và editable path trong `requirements.txt` phải tồn tại trên server;
nếu thiếu, setup dừng với thông báo path cụ thể.

### Launcher chung

`scripts/common/runtime.sh` cung cấp việc resolve/validate interpreter. Mọi
`scripts/run_*.sh`, helper Python trong `run_representative_100.sh`, và subprocess
benchmark trong MagicDec/SpecExtend/LongSpec đều phải dùng interpreter đó hoặc
`sys.executable` kế thừa từ nó. `PYTHONPATH`, cwd, config arguments và output
schema của baseline được giữ nguyên.

### Loại bỏ môi trường cũ

Xóa `.venv` Python 3.11 ở root và toàn bộ `envs/*/.venv`. Các manifest/lock
trong `envs/` và root `pyproject.toml` được giữ lại như tư liệu tương thích,
nhưng không còn entrypoint nào sử dụng chúng. `requirements.txt` là manifest
chính; các path/package server-specific được giữ nguyên, còn dependency project
còn thiếu được bổ sung khi preflight xác nhận codebase đang dùng nó.

## Quy trình setup và debug

1. `scripts/setup_venv.sh` kiểm tra Python 3.12, tạo/recreate `.venv` và chạy
   `uv pip install --offline -r requirements.txt` vào đúng venv.
2. `scripts/bootstrap.sh` chỉ kiểm tra uv/Python và gọi setup offline; không
   cài uv bằng `curl`.
3. Contract tests và `bash -n` xác minh toàn bộ launcher không còn gọi
   `uv run`, `--project` hoặc Python ngoài runtime helper.
4. Import preflight kiểm tra các dependency/baseline module cần thiết mà không
   tải model.
5. Smoke/debug chạy các baseline có thể chạy với cache hiện có; mỗi lỗi được
   phân loại thành lỗi launcher, lỗi dependency/API, lỗi binary/CUDA hoặc thiếu
   model/dataset cache.

## Tiêu chí hoàn thành

- Không còn launcher chính nào dùng `uv run --project` hoặc venv riêng.
- Không còn thư mục `.venv` cũ của root hoặc `envs/*`.
- Có một setup path rõ ràng cho venv Python 3.12 từ `requirements.txt`.
- Setup offline không âm thầm truy cập internet.
- Subprocess không thoát khỏi venv chung.
- Contract/static checks pass.
- Runtime smoke được chạy và báo cáo trung thực theo khả năng của môi trường
  mô phỏng; phần không thể chạy vì thiếu Python 3.12 hoặc artefact ngoài repo
  được nêu rõ, không được đánh dấu pass.
