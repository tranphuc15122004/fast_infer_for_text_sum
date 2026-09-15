# MR-DFlash SGLang smoke-debug log — 2026-09-14

## Scope

Mục tiêu là chạy đúng backend `specforge_sglang` cho pipeline cache hidden,
không dùng fallback HF, và ghi lại blocker theo từng lần reproduce.

## Initial evidence

- Máy hiện tại: Tesla T4, driver 550.163.01, host CUDA 12.4.
- Venv: Python 3.12.13, PyTorch `2.11.0+cu130`; CUDA không khởi tạo được trên
  máy này.
- `run_phase1_smoke.py` đang hard-code `--cache-backend hf`, nên smoke Phase 1
  mặc định không kiểm tra SGLang.
- Chạy trực tiếp `cache_target_features.py` với
  `--cache-backend specforge_sglang --attention-backend flashinfer`.

## Reproduction 1

Command dùng artifact đã pass regeneration/validation/tokenization:

```text
cache_target_features.py ... --cache-backend specforge_sglang \
  --attention-backend flashinfer --device cpu
```

Result: fail tại `SpecForge.distributed.init_distributed()`:

```text
ModuleNotFoundError: No module named 'yunchang'
```

Independent probe cũng xác nhận module `sgl_kernel` chưa tồn tại trong venv.

## Fix attempt 1

- Cài `yunchang==0.6.4` vào `.venv` bằng pip từ PyPI.
- Cài thành công, không cài thêm dependency phụ.

## Reproduction 2

Chạy lại cùng backend sau khi cài `yunchang`.

Result: dependency được import xa hơn, sau đó dừng tại:

```text
RuntimeError: The NVIDIA driver on your system is too old (found version 12040)
```

Call site: `yunchang.globals.get_cuda_arch()` gọi
`torch.cuda.get_device_capability()`. Đây là blocker host runtime: máy local có
driver CUDA 12.4, còn PyTorch/FlashInfer stack là cu130.

## Fix attempt 2 — runner selection

- Thêm `cache_backend` vào `SmokeOptions`.
- Thêm CLI `--cache-backend {hf,specforge_sglang}`.
- Truyền backend đã chọn vào cache stage; mặc định vẫn là `hf` để giữ CPU smoke
  portable.
- Thêm regression test kiểm tra stage plan chọn đúng `specforge_sglang`.
- Test đỏ trước sửa: `TypeError: SmokeOptions.__init__() got an unexpected keyword argument 'cache_backend'`.
- Test xanh sau sửa: `5 passed` trong `tests/test_phase1_smoke_pipeline.py`.
- Sau đó thêm test CLI để bắt lỗi `main()` không truyền `args.cache_backend` vào
  `SmokeOptions`; test đỏ vì dry-run vẫn sinh `--cache-backend hf`.
- Sửa wiring trong `main()`; test xanh: `64 passed` cho nhóm pipeline/pilot/
  worker/SpecForge adapter.
- Dry-run cuối xác nhận cache stage sinh:
  `--attention-backend sdpa --cache-backend specforge_sglang`.

## Current hypothesis

SGLang backend hiện đã vượt qua blocker Python đầu tiên sau khi cài
`yunchang`, nhưng không thể chạy trên host local vì CUDA driver không tương
thích. Native `sglang-kernel` cũng chưa có trong venv và cần artifact cu130 từ
B200 server.

## Final status on this host

Không thể đạt full SGLang hidden-cache success trên máy local: `yunchang` cần
`torch.cuda.get_device_capability()`, nhưng driver local chỉ hỗ trợ CUDA 12.4
trong khi venv là cu130. Đây là blocker phần cứng/runtime, không phải lỗi còn
được sửa bằng Python code. Cần chạy command đã được sửa trên B200 server với
driver 580/cu130 và wheel `sglang-kernel` đúng architecture.

## Full runner reproduction after fixes

Chạy `run_phase1_smoke.py` với:

```text
--cache-backend specforge_sglang --cache-attention-backend sdpa
```

Kết quả:

- `regenerate_smoke_train`: pass
- `validate_smoke_train`: pass
- `tokenize_smoke_train`: pass
- `cache_smoke_train`: fail tại CUDA driver trong `yunchang`

Artifact/log: `/tmp/mr_dflash_phase1_sglang_full/`.

## Dependency acquisition follow-up

- PyPI metadata xác nhận có `sglang-kernel==0.4.2` wheel
  `cp310-abi3-manylinux2014_x86_64`, tương thích tag Python 3.12/x86_64.
- Thử tải wheel 326.3 MB; kết nối chỉ đạt khoảng 59.8 MB rồi treo, nên đã
  hủy. Wheel chưa được cài vào venv.
- Server không có internet, vì vậy wheel này cần được mirror/copy vào
  offline wheelhouse của server.

## Full library smoke

Với writable FlashInfer/Triton/Torch extension caches và
`PYTHONPATH=src:externals/SSSD/python:externals/SpecForge`:

- Pass: `torch`, `transformers`, `vllm`, `flashinfer`, `flashinfer_cubin`,
  vendored `sglang`, SGLang request/sampling entry points, `specforge`,
  `specforge.offline_capture` — `10/12` imports.
- Fail: `yunchang` do CUDA driver 12.4 không khởi tạo được PyTorch cu130.
- Fail: `sgl_kernel` vì wheel `sglang-kernel` chưa cài.

Import pass không thay thế được runtime smoke: hidden-cache SGLang vẫn dừng ở
`yunchang` trước khi model load.

## Adaptive batch revision — 2026-09-14

Phản hồi: không muốn dùng batch cố định; cần tăng dần tới ngưỡng an toàn.

Root cause của behavior cũ: SGLang adaptive controller vẫn khởi tạo state
bằng batch lấy từ built-in throughput profile (`64/32/16/4` theo length
bucket). Profile không phải hard cap sau khi controller đã chạy, nhưng nó vẫn
làm batch đầu tiên bị cố định theo bucket.

Đã sửa:

- Thêm `--cache-auto-batch-start-size`, mặc định `1`.
- SGLang adaptive cache bắt đầu từ start size, bỏ qua batch size trong profile
  khi khởi tạo controller.
- Mỗi batch thành công tăng theo `growth_factor`.
- Khi OOM, giữ nguyên buffer CPU, backoff và binary-search vùng giữa batch an
  toàn cuối cùng và batch lỗi.
- Giữ `--cache-auto-batch-max-size` như emergency ceiling; đây không còn là
  batch cố định. `--cache-auto-batch-target-vram-gb 170` vẫn đặt static-pool
  safety target trên mỗi B200.
- Truyền tham số mới qua pipeline chính, parallel workers và regenerate worker;
  ghi cấu hình adaptive vào stats/plan để audit.

TDD/debug evidence:

- Test đỏ trước sửa: `cache_dataset()` và `PipelineOptions` chưa nhận
  `auto_batch_start_size`/`cache_auto_batch_start_size`.
- Test xanh sau sửa: `18 passed` cho cache auto-batch/controller.
- Full nhóm regression: `82 passed`.
- `py_compile` và `git diff --check`: pass.

Ví dụ controller test mô phỏng ngưỡng OOM cho profile batch 64: batch thực tế
được thử theo chuỗi `1 -> 2 -> 4`, OOM ở 4, retry 2, thử 3 để tìm biên, rồi giữ
batch an toàn 2. Full SGLang GPU vẫn cần xác nhận trên B200 vì local host bị
block bởi driver R550/CUDA12.4 với venv cu130.

## Predictive guard and OOM recovery — 2026-09-14

Yêu cầu mới: không chờ tới hard threshold/OOM nếu có thể dự đoán rằng thêm một
sample sẽ vượt ngưỡng.

Đã bổ sung:

- `SpecForgeTargetCapture` đọc `model_runner.max_total_num_tokens` sau khi
  SGLang profile static-pool xong.
- Cache worker tạo `runtime_safe_token_capacity` bằng 95% capacity mặc định
  (`--cache-auto-batch-safety-fraction` điều chỉnh được).
- Trước mỗi forward, batch ceiling được tính theo
  `safe_token_capacity // padded_length`; batch hiện tại + sample kế tiếp nếu
  vượt ceiling sẽ không được đưa vào SGLang.
- Adaptive controller có online predictor theo slope VRAM khi runtime cung cấp
  telemetry; nếu dự đoán sample kế tiếp vượt `target - safety_margin`,
  controller giữ batch hiện tại thay vì tăng. Với SGLang cache, guard chính là
  token-capacity thực tế vì static-pool đã được preallocate.
- Khi OOM, `capture_batch` không ghi feature; buffer CPU vẫn giữ nguyên,
  request/KV pools được clear, `empty_cache()` chạy, controller backoff và
  retry cùng sample. Chỉ batch 1 lỗi mới kết thúc stage vì khi đó sample không
  thể chạy với cấu hình hiện tại.

Regression test mới kiểm tra predictor dừng trước khi thêm sample vượt 170 GB
và runtime token safety ceiling. Controller tests: `7 passed`; nhóm cache/
pipeline đầy đủ sẽ chạy lại sau khi hoàn tất wiring option mới.

## Phase 1 hidden-cache wiring — 2026-09-14

Phát hiện còn một khoảng trống: worker `cache_target_features.py` đã có
adaptive SGLang, nhưng `run_phase1_smoke.py` chưa expose/truyền các option đó.
Vì vậy một Phase 1 smoke SGLang trước đây chưa kiểm tra đúng đường chạy cần
dùng trên B200.

Đã sửa runner Phase 1 để truyền đầy đủ:

- `--cache-auto-batch`;
- `--cache-auto-batch-start-size` (mặc định 1);
- `--cache-auto-batch-safety-fraction` (mặc định 0.95);
- `--cache-auto-batch-target-vram-gb`;
- `--cache-auto-batch-max-size` và `--cache-auto-batch-growth-factor`.

Test đỏ trước sửa: `SmokeOptions` không nhận `cache_auto_batch`.
Test xanh sau sửa: test stage plan mới pass; toàn bộ nhóm regression hiện tại
`85 passed in 4.75s`, `py_compile` và `git diff --check` đều pass.

Do hiện không có quyền truy cập B200, chưa thể tuyên bố runtime server đã
pass. Bước xác nhận duy nhất còn lại là chạy accelerated Phase 1 smoke bằng
`python3` hệ thống trên server với wheel
`sglang_kernel-0.4.2+cu130-...whl`, driver 580/cu130 và model local. Lệnh
chuẩn đã được ghi trong `docs/server_environment.md`; nếu lệnh này pass thì
phase hidden-cache đã được xác nhận end-to-end, không cần sửa thêm code local.

## Modal B200/cu130 upload và debug — 2026-09-15

Để kiểm tra khi chưa truy cập được server thật, đã upload source pipeline,
SpecForge và model local `Qwen3-0.6B` lên Modal. Harness tái lập tại
[`scripts/modal_hidden_cache_smoke.py`](../../scripts/modal_hidden_cache_smoke.py)
và lưu output/log vào Volume `mr-dflash-hidden-cache-debug`.

Image/runtime dùng:

- `nvidia/cuda:13.0.1-devel-ubuntu24.04`, Python 3.12;
- Torch `2.11.0+cu130`, `torch.version.cuda=13.0`;
- FlashInfer `0.6.12`, `flashinfer-jit-cache=0.6.12+cu130`;
- `sglang=0.5.14`, `sglang-kernel=0.4.2`, `yunchang=0.6.4`;
- CUDA tile packages `cuda-tile=1.3.0` và `nvidia-cuda-tileiras=13.2.78`;
- GPU thực tế: NVIDIA B200, compute capability `[10, 0]`.

Các lỗi được reproduce và xử lý tuần tự:

1. thiếu `pybase64`, sau đó thiếu `PIL`, `torchvision`, `openai`,
   `partial_json_parser`, `sentencepiece`, `compressed_tensors`, `gguf` và
   `msgspec`: bổ sung đúng package cần cho import chain SGLang 0.5.14;
2. thiếu `cuda.tile` trong FlashInfer communication path: bổ sung
   `cuda-tile` và `nvidia-cuda-tileiras` theo manifest cu130;
3. source `externals/SSSD/python/sglang` shadow wheel SGLang. Source này trả
   3 giá trị từ `compute_dp_attention_world_info`, còn SpecForge adapter gọi
   contract 4 giá trị của SGLang 0.5.14, gây
   `ValueError: not enough values to unpack (expected 4, got 3)`;
4. bỏ SSSD source khỏi `PYTHONPATH` của SpecForge và sửa
   `_ensure_specforge_importable()` để tự loại path SSSD shadowing. Regression
   test mới bảo đảm helper không ưu tiên source SSSD. SSSD vẫn giữ riêng cho
   baseline SSSD, không dùng chung cho MR-DFlash SpecForge cache.

Lần chạy thành công cuối cùng dùng Modal run
[`ap-Za6huPPzVdbI7H0nMo9hna`](https://modal.com/apps/tdphuc-work/main/ap-Za6huPPzVdbI7H0nMo9hna).
Probe xác nhận tất cả import deep đều từ environment tương thích, trong đó
`sglang` và `sglang.srt.managers.schedule_batch` đến từ
`/usr/local/lib/python3.12/site-packages`, không phải SSSD source.

Kết quả end-to-end:

- `regenerate_smoke_train`: pass;
- `validate_smoke_train`: pass;
- `tokenize_smoke_train`: pass;
- `cache_smoke_train`: pass, `captured=4`, `capture_errors=0`, `skipped=0`;
- hidden-cache manifest: 1 shard, 4/4 sample, layer `[1, 9, 17, 25]`, dtype
  `bfloat16`, feature width `4096`, attention backend `flashinfer`;
- adaptive controller khởi động batch 1, target safety 170 GiB, không phát sinh
  OOM; cache stage hoàn tất khoảng 42 giây trên fixture 4 mẫu.

Artifact xác nhận nằm tại Volume:

```text
mr_dflash_phase1_sglang/target_features_smoke/train/manifest.json
mr_dflash_phase1_sglang/target_features_smoke/train/shard_00000.pt
mr_dflash_phase1_sglang/pipeline_summary.json
```

Sau đó thêm regression guard cho môi trường server thật: Modal retry kế tiếp
mount lại cả SSSD source và cố tình đặt nó vào `PYTHONPATH`, trong khi adapter
phải loại path đó trước khi import. Lần retry này dùng output root riêng
`mr_dflash_phase1_sglang_path_guard`; mục tiêu là xác nhận fix không chỉ thành
công vì SSSD source bị bỏ khỏi image. Run
[`ap-udDvsobvYgEnxLqKUbxV3J`](https://modal.com/apps/tdphuc-work/main/ap-udDvsobvYgEnxLqKUbxV3J)
đã pass: probe vẫn thấy SGLang wheel trong site-packages, cả 4 stage pass và
manifest path-guard ghi `captured=4`, `capture_errors=0`, `skipped=0`.

Validation local sau fix: `28 passed`, `py_compile` và `git diff --check`
pass. Vì vậy blocker còn lại trên server không phải source-code cache nữa;
server cần giữ đúng dependency cu130 và không để package SGLang fork của SSSD
được import cho MR-DFlash SpecForge.
