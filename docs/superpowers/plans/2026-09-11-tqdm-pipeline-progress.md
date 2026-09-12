# TQDM Pipeline Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chuẩn hóa terminal output của toàn bộ MR-DFlash preprocessing pipeline thành progress bar `tqdm` theo số sample, không stream từng dòng log của child process.

**Architecture:** Parent pipeline tạo một progress artifact atomic cho từng stage, truyền path qua `MR_DFLASH_PROGRESS_PATH`, và poll artifact để cập nhật một thanh `tqdm` duy nhất. Mỗi script xử lý sample ghi `completed_samples`/`total_samples` vào artifact; stdout/stderr vẫn được lưu đầy đủ vào `pipeline_logs/` hoặc worker log nhưng không append ra terminal.

**Tech Stack:** Python 3.12, `tqdm`, JSON atomic replace, pytest.

**Spec:** Yêu cầu người dùng ngày 2026-09-11: mọi quá trình log dùng `tqdm`, progress phải theo đúng số lượng sample đã xử lý thay vì append dòng vào terminal.

## Global Constraints

- Runtime production dùng `python3` hệ thống; không tải dependency hoặc model mới.
- Không thay đổi semantics dữ liệu/cache; progress chỉ là observability.
- `progress.json`/`status.json` phải đọc được trong lúc process đang chạy và không được làm hỏng pipeline nếu telemetry lỗi.
- Chi tiết stdout/stderr vẫn phải được lưu vào file log để debug.
- Dev T4 chạy CPU; không dùng GPU local để xác nhận benchmark.

---

### Task 1: Chuẩn hóa stage progress contract

**Files:**
- Modify: `scripts/mr_dflash/progress.py`
- Modify: `scripts/mr_dflash/run_preprocess_pipeline.py`
- Test: `src/MR_DFlash/tests/test_worker_progress.py`
- Test: `src/MR_DFlash/tests/test_pilot_scripts.py`

**Interfaces:**
- `ProgressReporter` tiếp nhận path từ environment khi script không truyền CLI path.
- `run_stage()` tạo progress path, redirect toàn bộ child output vào stage log, và poll `completed_samples` để cập nhật `tqdm`.
- Stage progress spec trả `total_samples` từ input/manifest/requested counts.

- [x] **Step 1: Write the failing tests** cho env path, stage total, và việc stage command không stream child lines ra console.
- [x] **Step 2: Run các test mới** và xác nhận chúng fail vì stage hiện chỉ có progress bar cho parallel stage.
- [x] **Step 3: Implement progress contract** và generic stage polling, giữ riêng aggregate status của parallel stage.
- [x] **Step 4: Run test unit** để xác nhận stage progress bar bắt đầu từ đúng total và kết thúc đúng total khi thành công.

### Task 2: Gắn sample progress cho các phase không có worker heartbeat

**Files:**
- Modify: `scripts/mr_dflash/prepare_server_data.py`
- Modify: `scripts/mr_dflash/prepare_sharegpt.py`
- Modify: `scripts/mr_dflash/prepare_arxiv.py`
- Modify: `scripts/mr_dflash/build_pilot_dataset.py`
- Modify: `scripts/mr_dflash/analyze_pilot_data.py`
- Modify: `scripts/mr_dflash/validate_pilot_dataset.py`
- Modify: `scripts/mr_dflash/tokenize_dataset.py`
- Modify: `scripts/mr_dflash/profile_cache_batches.py`
- Test: `src/MR_DFlash/tests/test_pilot_scripts.py`

**Interfaces:**
- Mỗi script đọc `MR_DFLASH_PROGRESS_PATH` và ghi phase/sample counters qua `ProgressReporter`.
- `completed_samples` tăng một lần cho mỗi input sample đã xử lý, kể cả sample bị skip hợp lệ; total phản ánh đúng workload của phase.
- Profile dùng số sample trong tokenized manifest; các bucket/probe là detail trong file log, không phải sample progress trên terminal.

- [x] **Step 1: Write failing tests** cho counter của analyze/validate/tokenize và cho progress path truyền qua prepare subprocess.
- [x] **Step 2: Run tests** và xác nhận counter/progress artifact chưa tồn tại.
- [x] **Step 3: Add reporters** với update ở phase start, per-sample/per-bucket, done, và failed.
- [x] **Step 4: Run targeted tests** và kiểm tra totals không bị tính theo số shard hoặc số token.

### Task 3: Chuẩn hóa watcher/docs và regression verification

**Files:**
- Modify: `scripts/mr_dflash/watch_parallel_stage.py`
- Modify: `docs/mr_dflash_pilot_pipeline.md`
- Modify: `docs/README.md`
- Test: `tests/test_mr_dflash_cache_progress.py`
- Test: `src/MR_DFlash/tests/test_worker_progress.py`

**Interfaces:**
- Watcher `--tqdm` hiển thị một bar aggregate theo sample; mode text chỉ dành cho chẩn đoán explicit, không được pipeline dùng.
- Documentation mô tả terminal bar và vị trí log chi tiết.

- [x] **Step 1: Write failing tests** cho aggregate watcher và fallback khi status chưa có total.
- [x] **Step 2: Run targeted watcher tests**.
- [x] **Step 3: Update watcher/docs** theo contract mới.
- [x] **Step 4: Run compile, diff check, targeted suite, và regression suite**.
