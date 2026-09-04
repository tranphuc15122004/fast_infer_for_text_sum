# MR-DFlash — bản tự đóng gói quy trình train DFlash

Folder này là **bản copy self-contained** của quy trình train model DFlash, port
từ SpecForge (`externals/SpecForge`), dùng làm **gốc cho thí nghiệm MR-DFlash**:
bạn sẽ chỉnh sửa trực tiếp trên code tại đây sau này.

Khác `externals/SpecForge` ở điểm: code **chạy độc lập** (chỉ cần `torch` +
`transformers`), không phụ thuộc SGLang/Mooncake/config schema khổng lồ của
SpecForge. Thuật toán block-parallel + loss được giữ trung thực theo SpecForge.

> ⚠️ Thư mục gốc `src/MR-DFlash` (có dấu `-`) được đổi thành `src/MR_DFlash` vì
> Python không import được package có dấu `-`. Toàn bộ file vẫn nằm ở vị trí cũ.

## Cấu trúc và mapping tới SpecForge

| File | Tương ứng SpecForge | Vai trò |
|---|---|---|
| `model.py` | `modeling/draft/dflash.py`, `dflash_kernels.py` | `DFlashDraftModel`: projector `fc` + `N` decoder layer + RoPE/RMSNorm/SwiGLU tự triển khai (torch thuần) |
| `training.py` | `algorithms/common/dflash_family_model.py`, `training/strategies/base.py` | `OnlineDFlashModel`: sample anchor, noise embedding, block mask, forward song song, loss CE + positional decay; `DFlashTrainStrategy` |
| `data.py` | `data/loss_mask.py`, `algorithms/common/dflash_family_data.py`, `offline_reader`/feature store | loss mask assistant, dataset jsonl, feature store `.ckpt`, normalizer + collator |
| `capture.py` | `scripts/prepare_hidden_states.py` (SGLang → HF) | capture feature target: hidden concat tại `target_layer_ids` |
| `trainer.py` | `training/trainer.py` + `controller.py` + `backend.py` | Trainer: micro-batch, accumulation, AdamW, grad clip, checkpoint |
| `schedule.py` | `training/schedule.py` + `lr_scheduler` | horizon optimizer-step + cosine + warmup |
| `chunking.py` | `core/chunking.py` | `checkpointed_chunk_reduce` |
| `checkpoint.py` | `training/checkpoint.py`, `model_loading.py` | checkpoint full + draft weights-only |
| `config.py` | `config/schema.py` (rút gọn) | `RunConfig`/`ModelConfig`/`DataConfig`/`TrainingConfig` + legacy defaults DFlash |
| `run_train.py` | `specforge train` CLI | entry point end-to-end offline |

## Thuật toán DFlash (tóm tắt)

1. Sample ≤ `num_anchors` anchor/chuỗi tại vị trí có `loss_mask[t]` và
   `loss_mask[t+1]` đều supervise.
2. Mỗi anchor → 1 block `block_size`: vị trí 0 = embedding token anchor, còn lại
   = `mask_token` embedding (từ **embedding target frozen**).
3. Mọi block chạy **song song** qua draft model. Attention: query draft chỉ
   attend context thật `< anchor` + draft trước nó trong cùng block (không
   cross-block) → huấn luyện hiệu quả O(block).
4. Label same-position: vị trí `k` trong block dự đoán token thật tại
   `anchor+k`; `weight = keep × (k>0) × bounds × loss_mask[label]`; loss =
   CE với **label hard** (không dùng target distribution) + tuỳ chọn positional
   decay `loss_decay_gamma`; accuracy telemetry.
5. Draft weight khởi tạo random (SpecForge) hoặc copy từ target layers qua
   `init_draft_from_target` (hook cho MR-DFlash).

## Cách chạy

### 1. CPU smoke (máy dev, không GPU)

```bash
cd src
python MR_DFlash/tests/test_smoke_cpu.py
```

### 2. Capture feature offline (một lần)

```bash
cd src
python -m MR_DFlash.capture \
  --target-model-path Qwen/Qwen3-8B \
  --data-path ../data/user_prompts.jsonl \
  --output-path ../outputs/features_dflash \
  --max-length 3072 \
  --torch-dtype bfloat16 \
  --device cuda
```

Mỗi mẫu lưu `input_ids` / `loss_mask` / `hidden_states` (concat tại các
`target_layer_ids`, tự sinh nếu không truyền `--target-layer-ids`).

### 3. Train offline

```bash
cd src
python -m MR_DFlash.run_train \
  --config MR_DFlash/configs/qwen3_8b_dflash_offline.yaml
```

Nếu `data.hidden_states_path` chưa có, `run_train` tự chạy capture vào
`output_dir/captured_features`. Có thể override CLI:
`--max-steps 100 --batch-size 1 --output-dir out ...`

Hoặc không dùng YAML, truyền thẳng flag: `--target-model-path ... --train-data-path ...`.

### 4. Kết quả

- `metrics.jsonl` — loss/acc/lr/grad_norm theo từng optimizer step.
- `checkpoint_step_*.pt` + `checkpoint_final.pt` — full state (draft weights +
  optimizer/scheduler + config) để resume (`--resume-from`).
- `draft_*.pt` — weights-only, tiện warm-start/export.

## Điểm cần chỉnh khi chạy thật (MR-DFlash)

- `attention_backend`: mặc định `sdpa` (dày đặc, CPU-safe). Với B200 và
  `num_anchors`/`max_length` lớn nên dùng `flex` (Flex Attention, không
  materialize mask) — đã có `build_dflash_flex_block_mask`.
- `mask_token_id`: phải đặt đúng token trong vocab target (Qwen3 VD `151669`).
- `init_draft_from_target=True` copy weight draft layer i từ target layer
  `target_layer_ids[i]` (thường dùng cho DFlash).
- Feature `hidden_states` capture bằng HF = toàn chuỗi (kể cả prompt). Nếu muốn
  tiết kiệm, chỉ capture prefix tới cuối assistant cần thiết.

## Đã lược bỏ so với SpecForge (có thể bổ sung sau)

- Online disaggregated (SGLang capture server + Mooncake) — không tái hiện
  standalone được; seam: thay `capture_dataset` bằng consumer đọc feature.
- EAGLE3/P-EAGLE/Domino/DSpark/D-PACE strategy (chỉ giữ DFlash; D-PACE loss
  dạng `dpace*` đã có trong `OnlineDFlashModel`).
- Liger kernel, config schema đầy đủ, FSDP/DDP multi-rank, vocab mapping.
- `spec_generate` (rollout speculative) — chỉ có training forward.
