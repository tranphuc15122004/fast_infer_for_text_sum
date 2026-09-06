# MR-DFlash V1 Design

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

## Kiến trúc V1

Feature contract giữ nguyên: `hidden_states` có dạng `[B, S, n_layers * H]`,
được concat theo `target_layer_ids`. `TargetFeatureAdapter` chiếu feature này
thành hai không gian cùng chiều draft `H`:

1. HCA memory: learned weighted pooling trên các nhóm token liên tiếp với
   `compression_ratio=128`, cộng local raw target memory trong cửa sổ `128`.
2. CSA memory: learned weighted pooling với `compression_ratio=4`; một learned
   Q/K indexer nhận query từ trạng thái draft sau HCA và chọn tối đa `64` slot
   bằng `torch.topk`. Local memory luôn được giữ, kể cả khi Top-k nhỏ hơn 64.

Đường draft mặc định là:

```text
block-causal DFlash attention + HCA target attention -> FFN
block-causal DFlash attention + CSA target attention -> FFN
```

Mask block vẫn không cho cross-block leakage. Bản reference dùng SDPA và
materialize mask để chạy được trên CPU; tối ưu kernel/Flex là phần benchmark
GPU sau.

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
layer. Các adapter HCA/CSA và compressor được khởi tạo ổn định từ trung bình
feature; value/output projection gần identity khi có thể. Indexer dùng Xavier
deterministic. Do đó test “initialized draft” kiểm tra việc copy key và
forward hữu hạn, không khẳng định logits bằng target.

## Inference contract

`MRDFlashInferenceEngine` cung cấp `prefill`, `draft_block`, `verify` và
`generate`. Target HF model được giữ nguyên để verify lossless. `MRMemoryState`
chỉ được cập nhật bằng hidden của token đã được target chấp nhận; token bị
reject không được đưa vào HCA/CSA state.

## Ngoài phạm vi V1

- Không sửa `externals/dflash` hoặc đăng ký MR-DFlash thành benchmark baseline.
- Không chạy job GPU trong lượt triển khai này.
- Không claim speedup, acceptance rate hay ROUGE trước GPU experiment.
- Không thêm dependency bắt buộc ngoài stack hiện tại.
