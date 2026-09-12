# Báo cáo Task 1: token-budget cache scheduler và profile contract

## File đã thay đổi

- `src/MR_DFlash/cache_throughput.py`
  - Thêm `CacheThroughputProfile` và `CacheThroughputBucket` thuần Python.
  - Hỗ trợ bucket liên tục, chọn batch/token budget theo độ dài.
  - Thêm canonical serialization và SHA-256 deterministic.
  - Thêm cap batch theo padded token count thực tế.
  - Profile candidate ban đầu: `4096/8192/16384/32768`, batch
    `64/32/16/4`, token budget `262144/262144/262144/131072`.
- `src/MR_DFlash/tests/test_cache_throughput.py`
  - Test bucket boundary, token budget, malformed payload, validation,
    deterministic serialization/hash, padded-token cap và worker consumption.
- `scripts/mr_dflash/cache_target_features.py`
  - Thêm seam tùy chọn `--throughput-profile` (có alias
    `--cache-throughput-profile` và `--cache-profile`).
  - Khi profile được truyền, worker chọn batch theo profile và cap theo
    padded length; không truyền profile thì HF mode và `--batch-profile` cũ
    giữ nguyên.
  - Ghi path/hash profile vào provenance hiện có của feature manifest.

## Kiểm thử

- RED trước implementation:
  - `pytest -q src/MR_DFlash/tests/test_cache_throughput.py`
  - Kết quả: fail do module `MR_DFlash.cache_throughput` và seam worker chưa
    tồn tại.
- Focused sau implementation:
  - `pytest -q src/MR_DFlash/tests/test_cache_throughput.py`
  - Kết quả: `12 passed`.
- Cache regression:
  - `pytest -q src/MR_DFlash/tests/test_cache_auto_batch.py src/MR_DFlash/tests/test_offline_feature_cache.py tests/test_mr_dflash_cache_progress.py tests/test_dataset_cache.py`
  - Kết quả: `24 passed`, 1 warning CUDA do driver máy dev.
- Combined verification:
  - `python3 -m py_compile src/MR_DFlash/cache_throughput.py scripts/mr_dflash/cache_target_features.py src/MR_DFlash/tests/test_cache_throughput.py`
  - Kết quả: pass.
  - Combined pytest: `37 passed`, 1 warning CUDA do driver máy dev.
  - `git diff --check`: pass.

## Commit

- Implementation commit: `e4a2a10` (`feat(mr-dflash): add token-budget cache profile`).

## Concern và phạm vi

- Task này chỉ tạo profile/scheduler seam và nối vào HF cache worker; chưa
  triển khai SGLang/SpecForge capture backend, high-concurrency engine,
  retry OOM hay orchestration 4 GPU. Các phần đó thuộc các task sau.
- Profile candidate là aggressive và cần được profiler/runtime validation trên
  B200 trước khi chạy full dataset.
- Manifest hiện dùng các trường provenance `cache_batch_profile` và
  `cache_batch_profile_sha256` sẵn có; schema writer chưa được mở rộng trong
  Task 1.
