# Bối cảnh MR-DFlash

## Trạng thái hiện tại

`src/MR_DFlash` đã có implementation V1 của **MR-DFlash** trên bản sao DFlash:
HCA/CSA target memory, learned compressor/indexer, training adapter và
reference speculative inference. Training hiện xử lý block ở batch dimension
để tránh logits liên block; chưa có kết quả GPU về latency, acceptance rate,
ROUGE hoặc speedup.

## Vai trò trong repository

| Đường dẫn | Vai trò | Có dùng để benchmark inference không? |
|---|---|---|
| `externals/dflash` | DFlash upstream dùng cho baseline inference | Có, qua `scripts/run.sh dflash` |
| `externals/SpecForge` | Framework upstream/tham chiếu cho train draft | Không trực tiếp |
| `src/MR_DFlash` | Model + train/inference pipeline MR-DFlash; giữ DFlash compatibility | Chưa đăng ký benchmark |
| `src/SyncSpec` | Một hướng thử nghiệm khác trong repo | Không được mặc định gộp vào MR-DFlash |

`MR_DFlash` không thay thế DFlash baseline và chưa phải một dòng mới trong ma
trận benchmark. Inference hiện có CLI/API riêng trong `src/MR_DFlash`, không
được trộn vào `scripts/run.sh` trước khi benchmark GPU được xác nhận.

## Những gì có thể coi là baseline kỹ thuật hiện tại

Bản copy hiện có giữ các thành phần chính của quy trình DFlash:

- draft model DFlash tự chứa trong `src/MR_DFlash/model.py`;
- wrapper block-parallel và loss trong `training.py`;
- capture hidden states bằng Hugging Face và feature store offline;
- trainer, scheduler, checkpoint và CLI train bằng YAML/CLI;
- CPU smoke test để kiểm tra pipeline tối thiểu.

## Phần MR-DFlash đã triển khai

- `memory.py`: HCA ratio `128`, CSA ratio `4`, local window `128`, complete
  groups + pending cache riêng, per-channel compressor và learned CSA
  score/Top-k tối đa `64`; Adapter có RMSNorm riêng và Indexer dùng
  head-wise `ReLU(dot)`.
- `mr_model.py`: DFlash joint attention với route xen kẽ HCA/CSA, RoPE cho
  memory positions và CSA local+selected trong một softmax. Shared context
  được project một lần; Top-k gather trên projected K/V.
- `training.py`: `OnlineMRDFlashModel` và `MRDFlashTrainStrategy`; anchor,
  label, hard CE, positional decay, accumulation và checkpoint giữ nguyên;
  anchor blocks được reshape thành batch `[B*N,K,H]`, mask cục bộ có shape
  `[B*N,1,K,K]`; indexer có dense warm-up rồi chuyển Top-k theo schedule.
- `checkpoint.py`: hỗ trợ converter DFlash → MR-DFlash cho attention/MLP/norm
  và adapter; native MR load strict, module compressor/indexer mới được giữ
  khởi tạo riêng.
- `inference.py`: prefill, draft block, target greedy verify và chỉ append
  token được accept; full block accept còn commit bonus token, EOS được cắt
  trước khi cập nhật memory; reference verify dùng full-prefix để ưu tiên
  correctness.

Chi tiết mapping file, semantics block/loss và lệnh chạy nằm trong
[`src/MR_DFlash/README.md`](../src/MR_DFlash/README.md). Các thành phần này chỉ
là điểm xuất phát để so sánh trước/sau khi đưa thay đổi MR-DFlash vào.

## Nguyên tắc làm việc cho các thay đổi sau

1. Giữ `externals/dflash` và đường chạy benchmark DFlash độc lập với
   `src/MR_DFlash`.
2. Khi sửa `src/MR_DFlash`, mô tả rõ phần nào là code DFlash được kế thừa và
   phần nào là thay đổi MR-DFlash.
3. Không gọi một checkpoint hoặc kết quả là “MR-DFlash” nếu chưa có thay đổi
   thuật toán được ghi nhận trong tài liệu và kiểm chứng tương ứng.
4. Nếu thay đổi làm mất parity với DFlash gốc, cập nhật README này về invariant
   bị thay đổi và giữ một test/smoke làm mốc hồi quy phù hợp.
5. Không khởi chạy GPU nếu chưa có `CUDA_VISIBLE_DEVICES` được cấp riêng;
   protocol deferred nằm ở [`docs/mr_dflash_gpu_experiments.md`](mr_dflash_gpu_experiments.md).
6. Không bật training lớn nếu chưa kiểm tra memory peak với layout `[B*N,K,H]`;
   `indexer_num_heads=1` là baseline tương thích, `4/8` chỉ là ablation.

## Tài liệu liên quan

- [`src/MR_DFlash/README.md`](../src/MR_DFlash/README.md): cấu trúc code và cách
  chạy bản copy hiện tại.
- [`docs/baselines/dflash.md`](baselines/dflash.md): DFlash inference baseline.
- [`externals/SpecForge`](../externals/SpecForge): upstream framework được dùng
  làm tham chiếu cho pipeline train.
- [`docs/model_baseline_matrix.md`](model_baseline_matrix.md): cặp model của
  DFlash trong benchmark; MR-DFlash chưa được thêm vào ma trận benchmark.
