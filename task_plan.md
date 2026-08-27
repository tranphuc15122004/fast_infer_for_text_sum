# Kế hoạch bổ sung baseline representative_100

## Mục tiêu

Cho runner `scripts/run_representative_100.sh` bao phủ toàn bộ baseline có
wrapper inference trong repository; thêm adapter cho `Sematic_selection` và
giữ `SpecForge` ngoài danh sách vì đây là framework hạ tầng.

## Các bước

- [complete] Kiểm kê baseline, wrapper, config và format dữ liệu.
- [complete] Thêm semantic-selection wrapper/config và tích hợp runner.
- [complete] Đưa các baseline đang optional vào chế độ all-by-default, vẫn hỗ trợ nhóm riêng.
- [complete] Cập nhật tài liệu benchmark và cách phân biệt baseline/framework.
- [complete] Chạy syntax, dry-run toàn bộ và smoke có thể chạy trong môi trường hiện tại.

## Tiêu chí hoàn thành

- Mọi baseline có wrapper trong `scripts/` xuất hiện trong runner.
- `SpecForge` được ghi rõ là infrastructure, không chạy như baseline.
- Dry-run không lỗi và smoke tạo output/metrics đúng schema khi dependency/model sẵn có.

## Giới hạn kiểm thử

`pytest -q` toàn repository vẫn thu thập test upstream trong `externals/`
và fail do dependency/env riêng (LLMLingua, MInference, SpecForge). Bộ test
contract của runner chạy độc lập và pass.

## Kế hoạch hiện tại: infer 1 sample với bộ model đã chốt

- [complete] Xác nhận dữ liệu representative và wrapper hiện có.
- [complete] Xác nhận 9 model đã có trong Hugging Face cache.
- [complete] Chốt thiết kế validation: full inference nơi baseline hỗ trợ,
  smoke/kernel cho baseline không nhận dữ liệu tóm tắt trực tiếp.
- [complete] Chạy từng baseline trên đúng 1 sample, lưu log và output riêng.
- [complete] Kiểm tra exit code, output schema, model thực tế và tổng hợp kết quả.

## 2026-08-18 — debug runtime toàn bộ baseline

- [complete] Chạy smoke/kernel đại diện cho 13 baseline qua `scripts/run.sh`.
- [complete] Phân loại root cause theo log và kiểm tra lại các lỗi có thể tái hiện.
- [complete] Tổng hợp PASS/FAIL/BLOCKED cùng artifact log/output.

## Tiêu chí cho lượt chạy

- Một sample cố định từ dữ liệu normalized/representative.
- Không gọi lại Hugging Face nếu model đã có trong cache.
- Ghi rõ `full_infer`, `smoke` hoặc `kernel_only` cho từng baseline.
- Một baseline lỗi không làm mất log/kết quả của các baseline còn lại.

## Kế hoạch hiện tại: chuyển dataset lớn ra HF cache

- [complete] Thêm helper/cache contract và kiểm tra test trước khi sửa.
- [complete] Cập nhật runner/tài liệu để đọc `HF_HOME/datasets/fast_infer_text_sum`.
- [complete] Di chuyển `externals/FastKV/data` và `externals/MagicDec/Data` ra cache.
- [complete] Xoá dữ liệu khỏi Git working tree, kiểm tra tracked files và dung lượng.

## Kế hoạch hiện tại: profile Qwen3-4B long-summary theo độ dài

- [complete] Chốt phạm vi: Qwen3-4B target đơn, không speculative decoding, chạy 1 GPU T4.
- [complete] Kiểm tra backend hiện có và thiết kế phép đo tách model load, tokenize,
  prefill/KV-cache, decode, postprocess và peak VRAM.
- [complete] Thêm profiler chạy các mốc 256/512/1024/2048/3072 từ, sinh CSV/JSONL và PNG.
- [complete] Chạy benchmark trên sample GovReport đủ dài; ghi nhận OOM/giới hạn phần cứng.
- [complete] Kiểm tra artifact, tổng hợp insight về tỷ lệ thời gian và bàn giao.

### Protocol đã chốt

- Input: `data/representative_100/govreport_representative.jsonl`, cắt prefix theo số từ.
- Output: greedy, tối đa 128 token, batch size 1, FP16, SDPA.
- Mỗi mốc có warmup riêng và đo lặp tối thiểu 3 lần nếu VRAM cho phép.
- `model_load` là one-time, không đưa vào tỷ lệ per-sample; báo riêng.
- `prefill` bao gồm forward toàn input và tạo/ghi KV cache lần đầu; không tuyên bố
  đây là một kernel event riêng nếu backend HF không expose event đó.

## Kết quả tổ chức artifact

- [complete] Đưa profiler canonical vào `src/analyze/full_infer/`.
- [complete] Sao chép raw measurements, summary, metadata và các PNG vào
  `src/analyze/full_infer/results/`, giữ nguyên bản gốc trong `outputs/`.
- [complete] Cập nhật wrapper/config để các lần chạy sau dùng source và output
  canonical mới.

## 2026-08-26 — shared Python 3.12 migration

### Mục tiêu và ràng buộc

- [complete] Chốt một runtime duy nhất tại `.venv` bằng Python 3.12 và `requirements.txt`.
- [complete] Chuyển launcher/subprocess sang interpreter chung, không còn project/env execution riêng.
- [complete] Thiết lập offline-first: không cài Python/package qua internet; yêu cầu local cache/wheel/path.

### Thực thi

- [complete] Ghi spec và implementation plan.
- [complete] Thêm `scripts/common/runtime.sh` với `FAST_INFER_PYTHON`/`FAST_INFER_VENV` và gate Python 3.12.
- [complete] Thêm `scripts/setup_venv.sh --offline|--check` và kiểm tra local requirement sources.
- [complete] Migrate 16 launcher chính; hỗ trợ config tùy chọn và truyền extra CLI flags như `--smoke`.
- [complete] Migrate child process của MagicDec/SpecExtend/LongSpec sang `sys.executable`.
- [complete] Thêm `scripts/check_shared_env.py` import-only/offline.
- [complete] Xóa root/group `.venv` cũ; giữ manifest/lock legacy do lớp an toàn chặn xóa artefact ngoài venv.
- [complete] Cập nhật README/docs/AGENTS/envs README theo runtime chung.

### Verification và blocker

- [complete] `pytest -q tests`: 46 passed.
- [complete] `bash -n` toàn bộ shell scripts và `python3 -m compileall -q scripts tests`: pass.
- [complete] Sweep 14 baseline với `--smoke`: 14/14 đi qua đúng shared-runtime gate, không còn lỗi coi `--smoke` là config.
- [blocked] Tạo/cài `.venv` thật và chạy full runtime: sandbox không có Python 3.12 cục bộ; uv chỉ báo bản tải xuống, nhưng mô phỏng offline không được tải.
- [blocked] Import/runtime đầy đủ: các path `/vllm-workspace`, `/workspace/storage-shared` và wheel server-specific chưa có trong sandbox.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| `python3.12: command not found` | 1 | Chưa có interpreter 3.12 trong môi trường mô phỏng; cần server/cache cung cấp Python 3.12. |
| `uv python list` không tạo được file tạm trong `/home/tuantb/.cache/uv` | 1 | Dùng `FAST_INFER_UV_CACHE` trỏ cache writable khi chạy setup. |
| Lệnh xóa toàn bộ `envs/` bị lớp an toàn từ chối | 1 | Xóa các `.venv` thực tế bằng path tường minh, giữ manifest/lock legacy. |
| `scripts/setup_venv.sh --offline` thiếu local wheel `/vllm-workspace/.../deep_ep` | 1 | Setup dừng sớm với path cụ thể; sau khi mount wheelhouse, cần cung cấp thêm Python 3.12 cục bộ. |

## 2026-08-26 — smoke 1 sample trên workspace hiện tại

### Điều tra ban đầu

- [complete] Chạy smoke 1 sample cho từng baseline có wrapper và lưu log riêng.
- [complete] Sửa lỗi code/API có thể tái hiện độc lập với GPU và package server.
- [pending] Xác nhận lại smoke trên Python 3.12 + GPU server sau khi có runtime.

### Môi trường mô phỏng hiện tại

- Python hiện tại: 3.13.9; không có executable Python 3.12.
- `nvidia-smi` không kết nối được NVIDIA driver; CUDA không khả dụng.
- `.venv` chưa được tạo; requirements chứa local wheel `/vllm-workspace/...` chưa mount.
- Wrapper được giữ gate Python 3.12 theo mục tiêu server; smoke trực tiếp sẽ chỉ dùng để phân lập lỗi baseline.

## 2026-08-26 — tái tạo uv project root từ requirements.txt

- [complete] Xóa `uv.lock` root cũ và đổi `.python-version` sang `3.12`.
- [complete] Đặt `pyproject.toml` về project root không package, yêu cầu chính xác
  Python `==3.12.*`; `requirements.txt` là nguồn dependency duy nhất.
- [complete] Cập nhật `scripts/setup_venv.sh` để `uv add -r requirements.txt --no-sync`
  và tự tạo/cập nhật `uv.lock` trước khi tạo venv.
- [blocked] Chưa thể sinh `uv.lock` mới trong workspace: không có interpreter Python
  3.12 và các local source `/vllm-workspace`/`/workspace/storage-shared` không tồn tại.
  Không tạo lock giả hoặc dùng Python 3.13 để resolve thay thế.

### Kết quả smoke và sửa lỗi

- [complete] FastKV và GemFilter chạy inference thực tế trên model cache TinyLlama,
  đúng 1 sample; cả hai sinh được `Paris.` và toàn bộ kiểm tra output/logits PASS.
- [complete] RocketKV chạy kernel prefill/decode smoke trên CPU, budget/shape/finite
  checks PASS.
- [complete] Semantic Selection chạy 1 document với các selector random/lead/tfidf/
  textrank; wrapper đã tự xử lý `--smoke` vì upstream không có cờ này.
- [complete] FastKV được tương thích với Transformers mới: cache API, attention
  registration, legacy attention attributes, position embeddings của Llama/Mistral,
  và short-prefill window.
- [complete] GemFilter được tương thích với Transformers mới cho Llama/Mistral/Phi-3:
  loader và custom attention fallback sang SDPA khi flash-attn unavailable.
- [complete] EAGLE-3 nhận `--smoke`; mọi baseline có dữ liệu ép giới hạn 1 sample
  trong smoke mode.
- [blocked] EAGLE-3, DFlash và FlexPrefill cần CUDA; SpecPrefill/MInference/MagicDec/
  LongSpec/SpecExtend/LLMLingua/HiGOE hiện thiếu package hoặc wheel server trong
  workspace mô phỏng. Đây là blocker môi trường, không phải lỗi dispatcher.

## 2026-08-27 — debug và smoke lại bằng `.venv`

- [complete] Rà soát lại toàn bộ dispatcher/wrapper/config và preflight `.venv`.
- [complete] Chạy lại 14 baseline với đúng 1 sample trên runtime hiện tại.
- [complete] Sửa và regression-test các lỗi script đã tái hiện.

### Verification cuối

- [complete] Toàn bộ `tests/` pass (`55 passed`), static compile/shell/diff-check pass.
- [complete] Kiểm tra output JSONL/summary và xác nhận record count/schema.
- [complete] Phân loại PASS/BLOCKED, ghi rõ blocker môi trường/dependency/GPU.

### Lỗi đã tái hiện trong lượt này

- GemFilter wrapper timeout sau run 0; cần retry với `num_runs=1`/generation ngắn hơn để xác định đây là timeout do khối lượng CPU hay lỗi runtime.

## 2026-08-27 — B200 readiness trên `python3` production

- [complete] Chốt runtime production `python3` từ PATH; `.venv` chỉ là mô phỏng local.
- [complete] Viết spec và implementation plan cho B200 preflight/smoke.
- [complete] Thêm profile `config/b200.env`, preflight text/JSON và one-sample runner.
- [complete] Thêm contract cho command-name resolution, config overlay, B200 CUDA probe,
  preflight hardware block và runner continuation.
- [complete] Chạy full regression/static validation và rà soát artifact/documentation.
