# MR-DFlash V1 Design — revision 2026-09-07

## Mục tiêu

Triển khai một drafter MR-DFlash trong `src/MR_DFlash` trên nền DFlash hiện
có. DFlash gốc vẫn là baseline hồi quy; MR-DFlash chỉ thay đổi đường context
target và thêm inference memory, không thay đổi verifier target.

## Experiment card R0

```text
Question:         memory target đa phân giải HCA+CSA có tạo được draft block
                  chạy được trong pipeline DFlash mà không phá loss/inference?
Hypothesis:       HCA global memory + CSA learned Top-k memory sẽ giữ chất lượng
                  draft tốt hơn context nén một mức với chi phí memory thấp hơn.
Baseline:         src/MR_DFlash DFlash hiện tại; cùng target, data, block/loss,
                  init và trainer config.
Variant:          chỉ thay target-context path bằng MR-DFlash HCA/CSA; verifier
                  và DFlash block objective giữ nguyên.
Primary metric:   CPU contract: tỉ lệ MR forward/train/inference smoke hoàn tất
                  với loss/logits finite; GPU sau đó đo acceptance rate.
Guardrails:       DFlash smoke không regress; target weights frozen; trainable
                  checkpoint reload được; memory chỉ append token được accept.
Data / split:     synthetic tiny-Qwen3 CPU cho contract; feature offline hiện có
                  cho train; GPU experiment dùng split cố định sau khi cấp GPU.
Seeds:            CPU seed cố định 0/42; GPU ban đầu seed 42, chưa sweep.
Budget:           CPU smoke <= 5 phút; GPU chưa chạy trong lượt này, ghi kế hoạch
                  và chờ GPU được cấp riêng.
Start rung:       R0 CPU unit/forward + train 2-4 steps + một speculative step.
Success criterion: R0 đạt toàn bộ contract; loss finite; checkpoint reload;
                  target frozen; accepted-only memory invariant đúng.
Exploratory-only: acceptance/latency/ROUGE GPU; R0 không xác nhận quality hay
                  speedup thực tế.
```

## Kiến trúc V1 — sau revision correctness

Feature contract giữ nguyên: `hidden_states` có dạng `[B, S, n_layers * H]`,
được concat theo `target_layer_ids`. `TargetFeatureAdapter` chiếu feature này
thành hai không gian cùng chiều draft `H`:

1. HCA memory: learned weighted pooling trên **complete** groups liên tiếp với
   `compression_ratio=128`; incomplete tail được giữ ở `pending_hca` cùng
   `pending_hca_positions`. Local HCA là cửa sổ raw tương đối theo từng anchor
   trong training và tail cửa sổ trong incremental inference.
2. CSA memory: learned weighted pooling trên complete groups với
   `compression_ratio=4`; incomplete tail được giữ độc lập ở `pending_csa` và
   `pending_csa_positions`.
3. Mỗi stage dùng một joint DFlash attention:

```text
Q = draft block
KV = [HCA context ; draft block] -> FFN
Q = draft block
KV = [CSA local + selected context ; draft block] -> FFN
```

Local và selected CSA đi qua cùng một softmax. Context/draft đều dùng cùng
quy ước RoPE theo absolute token position; compressed entry lấy position của
token cuối group.

Khi `sliding_window=None`, block mask là full same-block giống DFlash/SpecForge
gốc; khi sliding attention được bật, draft sub-block dùng lower-triangular
window. Training và reference inference bắt buộc dùng cùng helper/mask policy.
Bản reference dùng SDPA và materialize mask để chạy được trên CPU; tối ưu
kernel/Flex là phần benchmark GPU sau.

## Khởi tạo và tương đương DFlash

Các tham số DFlash hiện tại giữ nguyên: `block_size`, `num_anchors`,
`loss_decay_gamma`, `objective_chunk_blocks`, `loss_type`, learning rate,
warmup, batch size, accumulation, checkpoint và target layer selection.

MR-specific defaults:

```yaml
architecture: mr_dflash
mr_num_stages: 2
hca_compression_ratio: 128
csa_compression_ratio: 4
memory_local_window: 128
csa_top_k: 64
indexer_dim: null       # mặc định H
```

`init_from_target` tiếp tục copy attention/FFN của target layer vào draft
stage. `mr_num_stages` là số MR stages và có route xen kẽ `HCA, CSA, HCA, ...`;
`mr_stage_init_layer_ids` có đúng một layer id cho mỗi stage. Các adapter HCA/CSA
và compressor được khởi tạo ổn định từ trung bình feature; compressor dùng
trọng số theo từng channel và positional bias trong group. Indexer là bản
Lightning-inspired torch thuần (multi-head ReLU interaction + query head
weight), với schedule `dense` warm-up trong `indexer_dense_steps` rồi chuyển
`topk`. Cả dense và top-k đều truyền score qua attention bias để q/k còn
gradient; không tuyên bố đây là full DeepSeek Lightning Indexer production.

Test “initialized draft” kiểm tra copy key và forward hữu hạn, không khẳng định
logits bằng target.

## Inference contract

`MRDFlashInferenceEngine` cung cấp `prefill`, `draft_block`, `verify` và
`generate`. Target HF model được giữ nguyên để verify lossless. `MRMemoryState`
chỉ được cập nhật bằng hidden của token đã được target chấp nhận; token bị
reject không được đưa vào HCA/CSA state. Checkpoint weights-only phải mang theo
resolved model config để inference tự dựng đúng feature layer IDs/ratios/stage
route; không phụ thuộc vào default CLI khác với lúc train.

## Invariants bắt buộc trước serious training

- `build(X[:N])` rồi `append(X[N:])` có cùng complete memory/pending state với
  build một lần trên `X` (trong tolerance số học).
- HCA/CSA pending positions độc lập và chỉ complete group mới được compress.
- Training local window là `[anchor-window, anchor)`, không phải tail sequence.
- Training/inference block mask giống nhau và SDPA chỉ scale đúng một lần.
- DFlash loss tạo gradient hữu hạn tới CSA Indexer ở dense mode và score-bias
  top-k mode.
- Checkpoint reload reconstruct được feature width/layer IDs và không load target
  weights vào draft.

## Ngoài phạm vi V1

- Không sửa `externals/dflash` hoặc đăng ký MR-DFlash thành benchmark baseline.
- Không chạy job GPU trong lượt triển khai này.
- Không claim speedup, acceptance rate hay ROUGE trước GPU experiment.
- Không thêm dependency bắt buộc ngoài stack hiện tại.

## Impact on Plan

- Task memory phải sửa state/pending/tail và thêm parity tests.
- Task model phải thay self-attention + target cross-attention bằng joint
  attention, gộp CSA context và thêm RoPE/context bias.
- Task train/inference phải truyền anchor-relative local windows, thống nhất
  mask/scale và lưu resolved config trong checkpoint.
- Task test phải thêm indexer gradient, mask parity, checkpoint reconstruction,
  verifier EOS/bonus và build-vs-append parity.
