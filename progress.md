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

## 2026-08-25 — bắt đầu profile Qwen3-4B

- Đã đọc protocol/profile code hiện có và xác nhận user chọn target đơn.
- Đang chuẩn bị profiler đo latency breakdown và sinh hình ảnh theo các mốc từ.

## 2026-08-25 — hoàn tất profile Qwen3-4B target đơn

- Đã thêm `scripts/profile_qwen3_long_summary.py` và wrapper/config tương ứng.
- Đã đo 5 mốc GovReport (256/512/1024/2048/3072 từ), 3 repeats/mốc, FP16 + SDPA,
  max 128 output tokens trên Tesla T4.
- Đã sửa helper đọc `DynamicCache.layers[*].keys/values`; chạy lại toàn bộ để KV
  cache bytes không còn bằng 0.
- Artifact PNG/CSV/JSONL đã sinh ở `outputs/qwen3_long_profile/` và được tổ chức
  canonical tại `src/analyze/full_infer/results/`.
- Source profiler canonical là `src/analyze/full_infer/profile_qwen3_long_summary.py`;
  wrapper/config đã được cập nhật để chạy và ghi output tại vị trí này.
- Verification cuối: bộ test của repo (`pytest -q tests`) `33 passed`; Python
  compile, shell syntax và `git diff --check` đều pass. `pytest -q` toàn repo
  vẫn bị giới hạn bởi test upstream trong `externals/` thiếu dependency riêng.

## 2026-08-26 — shared Python 3.12 migration

- Đã ghi spec và implementation plan cho migration.
- Commit spec bị chặn bởi sandbox vì `.git/index.lock` nằm trên filesystem read-only; không stage/commit các thay đổi người dùng.
- Đã thêm `scripts/common/runtime.sh`, `scripts/setup_venv.sh` và preflight `scripts/check_shared_env.py`.
- Đã migrate tất cả shell launcher và child process sang runtime chung; sửa lỗi representative runner tiếp tục sau khi helper fail.
- Đã xóa các `.venv` cũ; giữ manifest/lock legacy sau khi lệnh xóa toàn bộ `envs/` bị lớp an toàn chặn.
- Static verification: 46 tests pass, shell syntax pass, compileall pass.
- Runtime setup còn blocked do sandbox thiếu cả local wheel/path server-specific (`/vllm-workspace/...`) và Python 3.12; không tải các artefact này qua internet.
- Review đã bổ sung kiểm tra direct URL chỉ dùng từ uv cache, cho phép root `.venv` trong test và thêm dependency sentence-transformers còn thiếu.

## 2026-08-26 — smoke 1 sample trên workspace hiện tại

- Đã hoàn tất phân lập smoke CPU/kernel khỏi shared-runtime gate Python 3.12.
- Đã chạy xác nhận lại FastKV và GemFilter inference thực tế trên TinyLlama cache;
  RocketKV và Semantic Selection cũng có artifact smoke PASS.
- Đã sửa các lỗi tương thích Transformers và chuẩn hóa `--smoke` thành 1 sample.
- Runtime hiện tại không có CUDA/GPU; các baseline GPU/server-specific được ghi
  nhận blocked với nguyên nhân package/wheel cụ thể.
- Còn lại: chạy lại `scripts/run.sh <baseline> --smoke` trên server thật sau khi
  cung cấp Python 3.12, wheelhouse `/vllm-workspace` và NVIDIA runtime.

## 2026-08-26 — tái tạo uv lock root từ requirements.txt

- Đã xóa lock root cũ và đặt project pin Python 3.12.
- Đã đưa bước `uv add -r requirements.txt` vào setup offline để lock được tạo
  tự động từ manifest server trên máy đủ điều kiện.
- Lock mới chưa thể sinh tại workspace hiện tại vì thiếu Python 3.12 và local
  artifacts server-specific; trạng thái này cần tiếp tục trên server thật.

## 2026-08-27 — debug và smoke lại bằng `.venv`

- Đã khôi phục các file kế hoạch/log vốn đã được git track sau khi cập nhật nhầm phần đầu phiên; không còn ghi đè lịch sử cũ.
- Đã xác nhận `.venv` Python 3.12.13 và ghi nhận preflight: torch/transformers/vllm/triton/dflash/llmlingua/sentence_transformers pass; flashinfer cache read-only và flash_attn thiếu.
- Đã chạy `bash -n scripts/*.sh scripts/common/*.sh` và `.venv/bin/python -m compileall -q scripts data`; cả hai pass.
- Đang tiếp tục smoke từng baseline và sẽ chỉ sửa sau khi tái hiện + truy nguyên lỗi.
- GemFilter wrapper đã load model và ghi 1/2 record, nhưng timeout 300s trước summary; đang retry khác biệt với 1 run/1 token để tách CPU-duration khỏi lỗi script.
- Đã chạy đủ 14 entrypoint smoke dưới `.venv`: CPU-safe RocketKV/FastKV/LLMLingua PASS; GemFilter minimal PASS; semantic-selection PASS sau fix; các baseline còn lại đã ghi nhận blocker cụ thể.
- Đã sửa và kiểm thử DFlash vendored `PYTHONPATH`, semantic-selection default fixture, RocketKV metadata; regression tests hiện 3/3 pass.
- Toàn bộ test suite hiện `55 passed`; output-contract validator bằng `.venv` pass.
- Diff source cuối chỉ gồm 3 sửa hành vi: DFlash PYTHONPATH, semantic-selection fixture mặc định, RocketKV biến metadata; thêm 3 regression tests và cập nhật expectation test cũ.

## 2026-08-27 — runtime model cho B200

- Người dùng xác nhận B200 production chạy trực tiếp bằng `python3` từ PATH;
  `.venv` hiện tại chỉ mô phỏng server.
- Đã cập nhật spec để tách production interpreter (`python3`) khỏi local
  simulation interpreter (`.venv/bin/python`), không yêu cầu activate venv.

## 2026-08-27 — B200 readiness implementation

- Đã chuẩn hóa runtime nhận `FAST_INFER_PYTHON=python3` qua PATH; nếu production
  không có `.venv`, runtime tự dùng Python system và vẫn gate Python 3.12.
- Đã thêm `config/b200.env`, `scripts/check_b200_env.py` và
  `scripts/run_b200_smoke.sh`/`scripts/b200_smoke.py`.
- Preflight kiểm tra CUDA/device target, CUDA tensor probe, imports, model/cache
  assets và cache JIT writable; cache model chỉ yêu cầu read access.
- Runner dùng overlay config sau baseline config, ép one-sample, lưu log/output
  riêng, tiếp tục sau lỗi và chỉ tổng thể PASS khi preflight B200 PASS.
- Contract tests B200/runtime hiện pass; đã chạy end-to-end overlay RocketKV trên
  `.venv`: baseline CPU smoke pass nhưng tổng thể đúng là BLOCKED vì thiếu CUDA.
- Đã cập nhật docs production B200 dùng `python3`, local simulation dùng `.venv`.
- Đã hoàn tất validation: `68 passed`, shell/compile/diff-check pass; preflight T4
  báo `BLOCKED` đúng vì không có CUDA và không giả lập thành công B200.
