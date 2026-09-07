# Bối cảnh MR-DFlash

## Trạng thái hiện tại

`src/MR_DFlash` đã có implementation V1 của **MR-DFlash** trên bản sao DFlash:
HCA/CSA target memory, learned compressor/indexer, training adapter và
reference speculative inference. Training hiện xử lý block ở batch dimension
để tránh logits liên block; chưa có kết quả GPU về latency, acceptance rate,
ROUGE hoặc speedup.

## Protocol pilot đã khóa

Thực nghiệm Qwen3-4B mới phải dùng sáu config dưới
`src/MR_DFlash/configs/pilot_qwen3_4b/`, không dùng nhầm các config legacy
(một số config cũ còn phục vụ regression và có init policy khác):

```text
DFlash-2L    : dflash_2l_{3k,8k}.yaml
MR-DFlash-2S : mr_dflash_2s_{3k,8k}.yaml
DFlash-5L    : dflash_5l_{3k,8k}.yaml
```

Chúng cùng target features `[1,9,17,25,33]`, block size `16`, DFlash loss,
`num_anchors=512`, `loss_decay_gamma=7`, seed `42`, `last_assistant`, online
feature extraction và `init_draft_from_target=false`. R2 dùng 10K sample ở
3K token; R3 dùng split 90K/5K/5K ở 8K token. Hướng dẫn prepare/regenerate,
fairness check, train matrix, exactness và report ở
[`mr_dflash_pilot_pipeline.md`](mr_dflash_pilot_pipeline.md).

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

## Khóa công bằng cho thí nghiệm pilot

Trước khi so sánh acceptance hoặc quality, mọi biến thể Qwen3 phải dùng cùng
feature contract:

```text
feature_layer_ids = [1, 9, 17, 25, 33]
block_size = 16
num_anchors = 512
learning_rate = 6e-4
loss_decay_gamma = 7.0
max_length = 3072       # pilot R0; chưa phải long-context claim
seed = 42
```

Feature store một layer cũ không còn tương thích với ma trận này; phải
capture lại và kiểm tra `manifest.json` có đúng năm layer/feature width trước
khi train. `DFlashFeatureDataset` sẽ từ chối cache sai provenance thay vì
âm thầm so sánh khác input.

Ma trận tối thiểu trên Qwen3-4B:

| Run | Config | Architecture/depth | Mục đích |
|---|---|---|---|
| DFlash-1L | `qwen3_4b_dflash_1l.yaml` | DFlash, 1 layer | baseline gốc, cùng 5 feature layers |
| DFlash-2L | `qwen3_4b_dflash_2l.yaml` | DFlash, 2 layers | depth/capacity control |
| MR-HCA+CSA | `qwen3_4b_mr_dflash.yaml` | MR, 2 stages | kiểm tra heterogeneous-resolution memory |

DFlash-2L là baseline khống chế độ sâu, không phải parameter-count matching
tuyệt đối: MR còn có compressor và indexer. Mỗi run phải ghi
`trainable_parameter_count` cùng peak VRAM trước khi diễn giải gain. Nếu
`MR-2stage > DFlash-2L` ở cùng feature/data/loss, bằng chứng cho giả thuyết
memory sẽ mạnh hơn so với chỉ so MR với DFlash-1L.

### Ma trận Llama 3.1 8B Instruct theo ngân sách DFlash-5L

Để benchmark trên target `meta-llama/Meta-Llama-3.1-8B-Instruct`, feature
contract được khóa ở năm layer của target 32 decoder layers:

```text
feature_layer_ids = [1, 8, 15, 22, 29]
```

Ba config liên quan nằm trong `src/MR_DFlash/configs/`:

| Run | Config | Cấu hình draft | Trainable params | Vai trò |
|---|---|---|---:|---|
| DFlash-5L | `llama3_1_8b_dflash_5l.yaml` | DFlash, 5 layer, MLP 12288 | 1,048,626,432 | baseline gốc |
| MR-4S | `llama3_1_8b_mr_dflash.yaml` | HCA→CSA→HCA→CSA, `indexer_dim=4096`, MLP 12288 | 1,040,786,432 (-0.75%) | cấu hình chính, giữ chi phí indexer hợp lý |
| MR-4S exact | `llama3_1_8b_mr_dflash_exact_params.yaml` | như MR-4S, `indexer_dim=5120` | 1,049,175,040 (+0.052%) | ablation kiểm soát tham số chặt |

Các con số trên là số tham số trainable của draft/MR, không tính target 8B bị
freeze. MR 4 stage là lựa chọn gần ngân sách DFlash-5L; 5 stage sẽ vượt ngân
sách. Biến thể `exact_params` gần khớp tuyệt đối nhưng tăng chiều rộng
Indexer, do đó không dùng riêng biến thể này để kết luận latency. Khi so
acceptance/quality nên chạy DFlash-5L, MR-4S và MR-4S exact trên cùng feature
cache, data, loss và seed; khi so latency phải báo riêng ngân sách tham số và
chi phí Indexer.

`mask_token_id: 128002` được ghi explicit; `block_size: 16` được khóa theo
protocol benchmark chung của repo (checkpoint DFlash Llama gốc dùng 10 nhưng
không phải ràng buộc kiến trúc). Trước khi capture thật, phải xác nhận ID này
với tokenizer snapshot
Llama đang được mount. Nếu server dùng model local, override
`--target-model-path` và lưu lại `manifest.json`; không dùng feature cache của
Qwen3 hoặc cache Llama được capture với layer list khác.

CSA Indexer V1 giữ `indexer_num_heads=1` cho run chính. Scale đã khóa theo
`head_dim ** -0.5`; các ablation 4/8 head chỉ chạy sau pilot và không trộn vào
baseline. V1 vẫn dùng score-bias sau Top-k như một cầu differentiable để
Indexer học được; đây là một quyết định engineering được ghi nhận, chưa phải
auxiliary ranking loss hay implementation Lightning production.

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
6. Không bật training lớn nếu chưa kiểm tra memory peak với layout `[B*N,K,H]`
   và log `step_time_s`, `tokens_per_second`, `peak_memory_*_mb` trên GPU;
   `indexer_num_heads=1` là baseline tương thích, `4/8` chỉ là ablation.

## Tài liệu liên quan

- [`src/MR_DFlash/README.md`](../src/MR_DFlash/README.md): cấu trúc code và cách
  chạy bản copy hiện tại.
- [`docs/baselines/dflash.md`](baselines/dflash.md): DFlash inference baseline.
- [`externals/SpecForge`](../externals/SpecForge): upstream framework được dùng
  làm tham chiếu cho pipeline train.
- [`docs/model_baseline_matrix.md`](model_baseline_matrix.md): cặp model của
  DFlash trong benchmark; MR-DFlash chưa được thêm vào ma trận benchmark.
