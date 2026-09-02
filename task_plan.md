# Kế hoạch: kiểm chứng hypothesis GroundSync

## Mục tiêu

Thực hiện các thí nghiệm kiểm chứng proposal GroundSync bằng một model Qwen3
đã có/được cache, ghi code và kết quả dưới `src/analyze`, và hoàn thiện báo cáo
đánh giá có tiêu chí go/no-go rõ ràng cho từng hypothesis.

## Trạng thái

| Pha | Trạng thái | Bằng chứng cần có |
|---|---|---|
| 1. Khảo sát repo và dữ liệu | hoàn tất | danh sách code/data/model/runtime hiện có |
| 2. Thiết kế experiment | hoàn tất | design doc + protocol + validation scope |
| 3. Pipeline analysis | hoàn tất | unit/synthetic smoke và CLI reproducible |
| 4. Chạy experiment | hoàn tất controlled mở rộng | Qwen3-4B + Qwen3-0.6B, GovReport 99/100 target + 99 proposals + 10 timing rows, CNN/DailyMail 100/100 target + 100 proposals + 12 timing rows, E0 relocation 3/3 cases, multi-start 396/400 rows |
| 5. Báo cáo | hoàn tất bản mở rộng, còn giới hạn production E2E và multi-start timing | report H1–H5, E0 relocation, calibration/sensitivity/controls, position-adjusted hazard + document bootstrap, cross-regime/multi-start metrics, timing, limitation và artifact audit |

## Ràng buộc đã biết

- Code và artifact phân tích phải nằm dưới `src/analyze`.
- Model được phép: Qwen3-4B, Qwen3-1.7B hoặc Qwen3-0.6B.
- Không được gọi kết quả “pass” nếu chưa có số liệu và kiểm tra fresh.
- Dùng một target-only canonical trace để đo source-state và một pipeline
  speculative/controlled phù hợp với model cache thực tế.
- Model-backed GPU đã chạy với Qwen3-4B canonical target và Qwen3-0.6B draft
  bằng `/home/tuantb/miniconda3/bin/python3` ngoài venv trên `cuda:0`; không
  dùng `.venv` cho thực nghiệm T4.

## Errors Encountered

Chưa có lỗi trong pha khảo sát.

## Next

Đã hoàn tất design/protocol, core metrics, target trace adapter, controlled
speculative trace, report và orchestrator theo TDD. Bản mở rộng đã thêm
calibrated positional prior, chunk/sink sensitivity, E0 position relocation,
position-adjusted hazard coefficient với 2.000 document bootstrap, negative controls, adaptive/true-cost
policy, train/dev threshold selection và tách timing khỏi acceptance. Báo cáo
toàn bộ quy trình đã ghi tại
`src/analyze/groundsync/verification_report_2026-08-29.md`.
Kết luận run mở rộng: GovReport H1/H2/H3/H4 `FAIL`, H5 `UNAVAILABLE`; CNN
H1/H3/H4 `FAIL`, H2 `PASS`, H5 `UNAVAILABLE`. Hai regime chưa cho bằng chứng
ổn định để xác nhận claim tổng hợp.
