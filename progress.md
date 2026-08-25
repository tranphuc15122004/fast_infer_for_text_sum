# Progress — bổ sung baseline representative_100

## 2026-08-18

- Đã khảo sát toàn bộ thư mục `externals/`, dispatcher, wrapper và config.
- Đã xác định nhóm thiếu trong runner: dflash và các baseline optional; riêng
  `Sematic_selection` chưa có wrapper.
- Đang thiết kế adapter để không gán nhầm DFlash/GSM8K vào dữ liệu tóm tắt.
- Đã thêm `config/semantic_selection.env`, `scripts/run_semantic_selection.sh`,
  dispatcher case và đưa toàn bộ 13 baseline inference vào default runner.
- Đã thêm collector normalization cho output semantic-selection.
- Contract tests hiện PASS (4/4); dry-run toàn bộ 32 data runs + 5 smoke-only runs PASS.
- Semantic smoke PASS: 1 document tạo 6 rows (`full` + 5 schemes), collector
  tách thành 6 method và sinh đủ JSON/CSV/Markdown.
- Full `pytest -q` không dùng được trong root env vì test upstream trong
  `externals/` yêu cầu các dependency/env riêng; targeted contract test vẫn 4/4.

## 2026-08-18 — chuẩn bị infer 1 sample

- Đã đọc lại runner, wrapper, configs và ma trận model-baseline.
- Đã xác nhận cache local đủ 9 model theo các file trọng số chính.
- Chưa chạy infer; còn phải chốt sample/dataset và tách full inference khỏi
  smoke-only trước khi bắt đầu GPU.
- Lỗi môi trường quan sát được: `hf auth whoami` không phân giải được
  `huggingface.co`; không ảnh hưởng tới việc dùng cache local.

## 2026-08-18 — debug runtime toàn bộ baseline

- Bắt đầu chạy smoke/kernel độc lập cho 13 baseline; giữ nguyên mọi thay đổi chưa commit của người dùng.
- Preflight cho thấy máy không có NVIDIA driver khả dụng; các baseline CUDA sẽ được ghi nhận theo lỗi runtime thực tế.
- Đã chạy entrypoint cho toàn bộ 13 baseline inference độc lập, log tại `outputs/runtime_debug/`.
- PASS trên host CPU: LLMLingua, FastKV smoke tối giản, RocketKV, GemFilter smoke tối giản, HiGOE.
- Có output inference nhưng runner timeout sau khi ghi output: semantic_selection.
- BLOCKED/FAIL do CUDA host: EAGLE-3, DFlash, SpecPrefill, MInference, MagicDec, LongSpec.
- SpecExtend không trả control trong 180 giây ngay cả với fixture 1 câu; chưa có output mới sạch để xác nhận.
- Lỗi chung wrapper đã xác nhận: cần `UV_CACHE_DIR` writable; chỉ workaround bằng env runtime, chưa sửa source.
- Verification cuối: shell syntax pass; artifact assertions pass; pytest bị chặn vì root env thiếu package `pytest`.

## 2026-08-18 — GPU Tesla T4 runtime

- Đã xác nhận GPU ngoài sandbox: Tesla T4 15 GiB, driver 550.163.01.
- Infer thật 1 sample PASS: LLMLingua, FastKV, GemFilter, SpecPrefill, MInference, SpecExtend, EAGLE-3, semantic_selection, DFlash, MagicDec.
- FastKV/GemFilter/SpecPrefill cần retry bằng config TinyLlama local vì config runner mặc định trỏ model thiếu weight/gated; retry đã có output và verification PASS.
- HiGOE/RocketKV/LongSpec chỉ có smoke path trong repo hiện tại; đã chạy path đó trên GPU nhưng chưa gán nhãn là text infer thật.

## 2026-08-21 — chuyển dataset lớn ra cache

- Đã xác nhận thiết kế cache và nhận được phê duyệt từ người dùng.
- Lỗi lần đầu khi chạy session-catchup do quoting shell; chưa làm thay đổi
  repository. Sẽ dùng lệnh tách riêng để tránh lặp lại lỗi.
- Đã thêm `dataset_cache_dir`, cập nhật FastKV/MagicDec resolver, wrapper,
  config và docs; unit test cache path, compile và shell syntax đều pass.
- Đã chuyển 908M FastKV và 99M MagicDec dataset vào
  `~/.cache/huggingface/datasets/fast_infer_text_sum/`.
- `git gc --prune=now` bị chặn vì `.git/gc.pid.lock` nằm trên filesystem
  read-only; object database chưa thể dọn trong sandbox.
