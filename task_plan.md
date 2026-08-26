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
