# Quyết định nghiên cứu: SpecExtend Horizon-CMR

Ngày: 2026-09-13

## Trạng thái các nhánh trước

Các nhánh dưới đây được archive/đóng trong phạm vi kế hoạch hiện tại:

- DFlash candidate repair / candidate-path selection;
- adaptive semantic budget và SafeBudget/SafeCover;
- source-conditioned residual-cover routing;
- progressive source-KV retirement;
- adaptive output stopping.

Không mở lại hidden-state feature, classifier lớn hơn, threshold sweep hay
E30-C. Các kết quả trước vẫn được giữ làm negative evidence, không được dùng
lẫn vào baseline SpecExtend mới.

## Nhánh đang mở

Chỉ mở nhánh chẩn đoán `SpecExtend Horizon-CMR` với câu hỏi:

> SpecExtend có chọn draft context quá myopic cho toàn bộ speculative block hay
> không?

Thứ tự bắt buộc:

1. kiểm tra runtime và model/checkpoint;
2. chạy baseline upstream không sửa CMR;
3. chỉ khi baseline chạy được mới lưu CMR/attention trace;
4. tính hindsight horizon oracle offline;
5. quyết định PASS/FAIL theo gate định trước.

## Model được chọn cho T4

Implementation summarization chính thức của SpecExtend dùng:

- target: `lmsys/vicuna-7b-v1.5-16k`;
- draft: `double7/vicuna-68m`;
- classic path: `run_classic.py`, `model_name=vicuna_7b`;
- precision: fp16;
- batch size: 1;
- decoding: greedy (`temperature=0`);
- retrieval chunk: 32 tokens;
- retrieval top-k: 32 chunks;
- retrieval interval: 4 verification cycles trong adapter repo.

Đây là cặp model nhỏ nhất phù hợp với implementation classic và đã có local
snapshot. Llama-3.1-8B + EAGLE-3 không được chọn cho T4 vì cần target 8B,
EAGLE-3 và cache/tree memory lớn hơn. Qwen3-4B không tương thích với loader
classic Llama của SpecExtend.

## Quyết định phần cứng

T4 15 GiB chỉ được coi là hợp lệ nếu inference thật khởi tạo được CUDA và
không OOM ở độ dài đang đo. T4 không phải target phù hợp cho SpecExtend 4K/8K
với fallback PyTorch attention trong code hiện tại. Vì vậy:

- 512/1K: smoke/feasibility trên T4;
- 4K: chỉ gọi là baseline đạt nếu chạy thật không OOM;
- 8K: chỉ chạy sau khi 4K ổn định;
- không dùng CPU hoặc fallback để báo cáo GPU speedup.

Nếu CUDA không khả dụng trong runtime, trạng thái phải là `BLOCKED_RUNTIME`,
không phải `PASS`.

## Gate H1

Chỉ mở predictor/architecture sau khi pilot đạt một trong hai điều kiện:

- accepted tokens tăng ít nhất 20%; hoặc
- E2E giảm thêm ít nhất 15% so với SpecExtend hiện tại.

Nếu không đạt, đóng nhánh Horizon-CMR; không thử threshold khác để cứu
hypothesis.

## Ghi chú về branch Git

Tại lần chạy này, thao tác tạo branch `horizon-cmr` bị hệ thống chặn vì
`.git/refs` ở chế độ chỉ đọc. Mã và artifact được cô lập dưới thư mục thí
nghiệm `outputs/specextend_horizon_cmr/`; không reset và không ghi đè thay đổi
có sẵn của người dùng.
