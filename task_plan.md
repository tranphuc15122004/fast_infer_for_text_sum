# Task Plan: Xây dựng bộ dữ liệu LongBench canonical 1.000 mẫu

## Goal
Thay thế `data/representative_100` bằng bộ test LongBench cố định 5 task × 200 mẫu, có schema JSONL chung, manifest reproducible và adapter đánh giá không phụ thuộc từng baseline.

## Next Step
Bàn giao bộ dữ liệu canonical sau vòng kiểm tra cuối; không commit được trong
sandbox hiện tại vì `.git/index` chỉ đọc.

## Current Phase
Phase 5

## Phases

### Phase 1: Requirements & Discovery
- [x] Đọc yêu cầu và tài liệu đính kèm
- [x] Kiểm tra schema dữ liệu, loader, runner và collector hiện tại
- [x] Ghi nhận ràng buộc offline và khác biệt semantics theo task
- [x] Chốt thiết kế với người dùng
- **Status:** complete

### Phase 2: Planning & Structure
- [x] Viết design spec sau khi được duyệt
- [x] Chọn cấu trúc builder/validator/prompt config
- [x] Viết implementation plan chi tiết
- **Status:** complete

### Phase 3: Small-scale TDD & Analysis
- [x] Tạo fixture local nhỏ, không đụng source cũ
- [x] Viết test schema/sampling/loader trước implementation
- [x] Chạy build nhỏ và phân tích count, length, duplicate, spot-check
- [x] Xin người dùng duyệt kết quả small-scale
- **Status:** complete

### Phase 4: Full-scale Build & Integration
- [x] Build bộ canonical 1.000 mẫu từ source LongBench local/cache
- [x] Cập nhật collector/docs/loader để dùng data set mới
- [x] Kiểm tra completeness theo 5 dataset và determinism full output
- **Status:** complete

### Phase 5: Verification & Delivery
- [x] Chạy focused test suite và validator dữ liệu
- [x] Kiểm tra diff, manifest, tài liệu và stale references
- [x] Bàn giao đường dẫn, lệnh build/chạy và giới hạn còn lại
- **Status:** complete

## Key Questions
1. Người dùng có duyệt bộ LongBench gồm `gov_report`, `qmsum`, `multi_news`, `lcc`, `repobench-p`, mỗi task 200 mẫu, với source local/cache pinned không?
2. Runner có cần chạy cả 5 task ngay không, hay chỉ cần dataset canonical + evaluator chung; các baseline không hỗ trợ task sẽ được đánh dấu unsupported?
3. Tokenizer chuẩn nào đã có sẵn trên server để tính `input_tokens`?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Giữ dataset canonical ở thư mục mới, không overwrite `data/representative_100` trong bước đầu | Có thể rollback và so sánh; chỉ đổi default sau khi validation/duyệt full-scale |
| Dùng JSONL + `manifest.json` | Phù hợp loader hiện tại, dễ stream, dễ audit ID và checksum |
| Giữ `context`/`input`/`answers`, thêm `reference_output` và metadata | Bảo toàn dữ liệu LongBench nhưng cung cấp field chung cho evaluator |
| Tính length bằng tokenizer chuẩn, không bằng số từ | Prefill/KV/verification phụ thuộc LLM token length |
| LCC/RepoBench-P sampling theo 5 length bins, 40 mẫu/bin | Giữ phân phối length thay vì lấy `[:200]` |
| Prompt render riêng theo task tại runtime | Tránh làm chết prompt LongBench vào raw data và tránh dùng prompt summarization cho code-completion |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| Chưa xác định source LongBench local/cache trong repo | 1 | Đã tìm thấy mirror LongBench trong HF cache; builder vẫn nhận explicit `--source-dir` để reproducible |
| Không thể commit spec vì `.git/index.lock` không tạo được (read-only filesystem) | 1 | Giữ spec trong workspace và tiếp tục kiểm tra/triển khai; không lặp lại thao tác commit trong sandbox hiện tại |
| Fixture test duplicate thiếu `manifest.json` | 1 | Bổ sung manifest tối thiểu; giữ validator strict với manifest |
| Rebuild small-scale song song/dự phòng dừng trước RepoBench-P | 1 | Không chạy lặp đồng thời; validator và các file đã hoàn thành vẫn pass, ghi nhận determinism helper thay cho full rebuild comparison ở checkpoint này |
| Test route code-completion giả định output nested | 1 | Sửa expectation về dict metric phẳng; không thay đổi dữ liệu/collector |

## Notes
- Theo data-preparation gate, full-scale chỉ được chạy sau khi test + phân tích small-scale pass và người dùng duyệt kết quả.
- Full output đã được build tại `data/longbench_200/`: 5 × 200 = 1.000 record; validator pass và rebuild độc lập byte-identical.
- `pytest -q tests` là phạm vi test dự án; `pytest -q` toàn repo còn thu thập test vendored cần package tùy chọn không có trong venv local.
- `task_plan.md`, `findings.md`, `progress.md` là ledger làm việc; không phải benchmark artifact.
