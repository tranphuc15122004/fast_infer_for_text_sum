# E29-A — Future-Use / Source-Retirement Oracle

## Mục tiêu

Kiểm tra xem trong long-document summarization, sau khi summary prefix đã được
sinh, có thể loại bỏ một phần source computation mà vẫn giữ được dự đoán của
target hay không. Đây là oracle screen; không triển khai heuristic eviction và
không thay đổi model/runtime production.

## Hypothesis và gate

Hypothesis: source units có future attention thấp sau một summary prefix có thể
được retire, tạo ra decode-work saving lớn mà không làm thay đổi đáng kể target
next-token distribution.

Gate mở E29-B chỉ khi đồng thời:

- oracle active-source work cho thấy ít nhất 35–40% decode work có thể loại bỏ
  trên ít nhất 2/3 datasets;
- tại cùng prefix, ablation oracle giữ top-1 agreement >=95% hoặc divergence
  continuation rất nhỏ;
- tất cả ngưỡng attention được báo sensitivity, không chọn ngưỡng sau khi xem
  kết quả.

Nếu gate không đạt, E29-B/E29-C không được chạy.

## Thiết kế

- Target: local Qwen3-4B, greedy decoding, thinking disabled, cùng prompt và
  source cap 4096 của E26-R cho GovReport/Multi-News.
- Datasets: CNN/DM, GovReport, Multi-News; initial screen 15 documents/dataset
  (45 documents tổng cộng), chọn deterministic từ representative inputs.
- Output: tối đa 256 tokens; checkpoints 32, 64, 96, 128, 192 tokens nếu
  output đủ dài.
- Source unit: contiguous 128-token chunks, giữ nguyên thứ tự và biên prompt.
- Attention primary: mean của layer/head attention query hiện tại, chỉ giữ
  source-unit mass và tổng source-attention mass; không materialize full
  sequence attention matrix.
- Future-use: với prefix length `t`, tính trung bình attention mass tới từng
  source unit trên các future queries `t:T`; source unit active nếu vượt từng
  ngưỡng preregistered `delta ∈ {0.001, 0.005, 0.01}`.
- Work: tổng active-source fraction trên mọi decode step; báo
  `G_retire = 1 - W_oracle/T` theo document và dataset.
- Safety ablation: dùng cùng generated prefix, giữ các source units active theo
  oracle, forward target lại và so next-token top-1 agreement, KL(full||retained),
  target-token NLL delta; horizon teacher-forcing là metric phụ.

## Controls

- full source, không retire, là reference bắt buộc;
- attention-only oracle và các delta sensitivity;
- source units được retire theo chunk, không token-level;
- không sử dụng future attention để gọi đó là online method; future attention
  chỉ được dùng cho hindsight ceiling.

## Validation scope

L0: py_compile, unit tests cho segmentation/future-use/work/metric aggregation,
finite-value checks và deterministic fixture.

L1: one-document CPU/CUDA smoke nếu runtime có model/device phù hợp; báo rõ
`torch.cuda.is_available()`. Full E29-A chỉ được gắn nhãn GPU khi PyTorch thật
sự thấy CUDA device.

## Artifacts

Artifact root: `outputs/dflash_residual/2026-09-11_e29_source_retirement/`.

Raw traces, per-document oracle rows, aggregate metrics, manifest và báo cáo
được ghi riêng; báo cáo Markdown chỉ chứa bảng tổng hợp, không chèn toàn bộ raw
attention vectors.

## Impact on Plan

- E29-A là oracle-only; không có E29-B/C trước khi gate pass.
- Existing `src/analyze/groundsync/trace_target.py` được tái sử dụng ở phần
  prompt/model loading nhưng E29 lưu thêm unnormalized source mass cần cho
  future-use.
