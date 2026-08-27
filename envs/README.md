# Legacy uv environment manifests

Các thư mục `.venv` trong layout cũ đã được xóa. Các `pyproject.toml` và
`uv.lock` còn lại chỉ là tư liệu tương thích lịch sử, không được launcher nào
sử dụng. Runtime hiện tại dùng một venv Python 3.12 tại `.venv/` và
`requirements.txt` ở root.

## Setup server offline

`uv` và Python 3.12 phải được cài sẵn trên server, cùng cache/wheelhouse cho
các package trong `requirements.txt`:

```bash
bash scripts/setup_venv.sh --offline
```

Nếu venv được đặt ngoài repo, dùng `FAST_INFER_VENV=/path/to/venv`. Nếu cần
chỉ định trực tiếp executable, dùng `FAST_INFER_PYTHON=/path/to/python`.

Các manifest cũ không thuộc runtime hiện tại. Nếu baseline cũ không tương thích
với stack Python 3.12 mới, ghi nhận lỗi trong debug report thay vì tạo
environment riêng.
