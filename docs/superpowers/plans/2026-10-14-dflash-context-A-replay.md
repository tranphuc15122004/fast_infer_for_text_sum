# Kế hoạch triển khai sửa replay DFlash Context A

> **Hướng dẫn thực hiện:** Dùng `superpowers:executing-plans` theo từng công việc. Theo dõi tiến độ bằng ô (`- [ ]`).

**Mục tiêu:** Replay mọi prefix A1/A2/A3 trên đúng trajectory full DFlash đã qua parity, kể cả khi trajectory đó khác AR.

**Kiến trúc:** Trace rounds của full adapter là nguồn của round index và acceptance. Harness sẽ chọn `adapter_ids` làm chuỗi token rebuild, đồng thời kiểm tra native DFlash và adapter thật sự trùng nhau trước khi replay. `ar_ids` chỉ còn là chẩn đoán.

**Công nghệ:** Python 3.12, pytest, helper campaign không phụ thuộc PyTorch.

**Đặc tả:** `src/TrainingFree/plans/experiment_plan_A_13_10_26.md`, đặc biệt protocol amendment cho native/full-adapter parity.

## Ràng buộc chung

- Native full DFlash phải trùng full adapter; AR mismatch được lưu làm diagnostic và không chặn A.
- Prefix state, pending anchor và accepted count phải thuộc cùng một trajectory.
- Không mở heldout nếu parity/dev lock chưa đạt.

---

### Công việc 1: Khóa nguồn trajectory khi dựng replay

**Tệp:**
- Modify: `scripts/run_dflash_context_experiments.py`
- Test: `tests/test_dflash_context_campaign.py`
- Update: `src/TrainingFree/docs/dflash_context_interventions_b200.md`

**Giao diện:**
- Đầu vào: trace có `native_ids`, `adapter_ids`, `ar_ids` và parity evidence.
- Đầu ra: `_replay_reference_ids(trace) -> list[int]`, chỉ trả về IDs của adapter sau khi xác nhận parity native/adapter.

- [x] **Bước 1: Viết test lỗi trước** cho trường hợp AR khác DFlash nhưng nguồn replay vẫn là adapter, và trường hợp native/adapter không trùng phải bị từ chối.
- [x] **Bước 2: Chạy test để xác nhận lỗi** vì helper chưa tồn tại.
- [x] **Bước 3: Thêm helper và chuyển mọi nơi dựng prefix/state sang helper**, gồm previous-prefix, age-prefix và replay A1/A2/A3.
- [x] **Bước 4: Ghi rõ nguồn trajectory trong trace và cập nhật runbook.**
- [x] **Bước 5: Chạy test hồi quy cùng kiểm tra cú pháp Python/Bash.**

### Công việc 2: Đưa launcher campaign qua shared runtime

**Tệp:**
- Modify: `scripts/run_dflash_context_A_b200.sh`

**Giao diện:**
- Nạp master config bằng `scripts/common/config.sh` trước khi dùng biến interpreter.
- Kiểm tra Python 3.12 và dùng `FAST_INFER_PYTHON` cho bước xác minh cohort.

- [x] **Bước 1: Chạy toàn suite và ghi nhận test contract đỏ**: các launcher phải nạp `runtime.sh` và không gọi trực tiếp `python3`.
- [x] **Bước 2: Nạp config/shared runtime trong launcher và chuyển inline Python sang `FAST_INFER_PYTHON`.**
- [x] **Bước 3: Chạy lại contract launcher và kiểm tra cú pháp Bash.**

