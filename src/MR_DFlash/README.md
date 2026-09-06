# MR-DFlash — workspace phát triển trên nền quy trình train DFlash

Folder này là **bản copy self-contained** của quy trình train và model DFlash,
port từ SpecForge (`externals/SpecForge`). Đây là workspace làm **gốc cho ý
tưởng MR-DFlash**: các thay đổi nghiên cứu sau này sẽ được thực hiện trực tiếp
trên code tại đây.

## Trạng thái hiện tại và ranh giới phạm vi

Hiện folder đã có implementation V1 của MR-DFlash trên nền pipeline DFlash:
memory HCA/CSA, learned compressor/indexer, draft forward block-parallel,
training wrapper giữ nguyên DFlash objective, và reference speculative
inference. DFlash gốc vẫn được giữ để hồi quy; chưa có claim speedup/quality
GPU nào ở giai đoạn CPU smoke này.

Phân biệt ba lớp trong repo:

| Thành phần | Vai trò hiện tại |
|---|---|
| `externals/dflash` + `scripts/run.sh dflash` | Baseline **inference** DFlash dùng trong benchmark; không phải nơi phát triển MR-DFlash. |
| `externals/SpecForge` | Nguồn upstream/tham chiếu cho quy trình train DFlash. |
| `src/MR_DFlash` | Model + train/inference pipeline MR-DFlash; giữ DFlash compatibility, chưa đăng ký benchmark. |

Trang bối cảnh cấp repo nằm ở [`docs/mr_dflash.md`](../../docs/mr_dflash.md).
Trang đó là nơi ghi các giả định và trạng thái nghiên cứu; README này tập trung
vào cách đọc/chạy code hiện có.

Khác `externals/SpecForge` ở điểm: code **chạy độc lập** với torch; cần
`transformers` khi capture/nạp target và `pyyaml` khi đọc YAML. Nó không phụ
thuộc SGLang/Mooncake/config schema khổng lồ của SpecForge. Thuật toán
block-parallel + loss hiện có được giữ theo semantics DFlash/SpecForge để làm
mốc so sánh.

> ⚠️ Tên package dùng dấu gạch dưới `src/MR_DFlash` vì Python không import
> được package có dấu gạch ngang. Dùng tên này trong mọi lệnh `python -m` và
> import.

## Cấu trúc và mapping tới SpecForge

| File | Tương ứng SpecForge | Vai trò |
|---|---|---|
| `model.py` | `modeling/draft/dflash.py`, `dflash_kernels.py` | `DFlashDraftModel`: projector `fc` + `N` decoder layer + RoPE/RMSNorm/SwiGLU tự triển khai (torch thuần) |
| `memory.py` | MR-DFlash mới | HCA/CSA weighted pooling, learned `CSAIndexer`, incremental `MRMemoryState` |
| `mr_model.py` | MR-DFlash mới | `MRDFlashDraftModel`: block attention + HCA target attention + CSA Top-k attention |
| `training.py` | `algorithms/common/dflash_family_model.py`, `training/strategies/base.py` | `OnlineDFlashModel`: sample anchor, noise embedding, block mask, forward song song, loss CE + positional decay; `DFlashTrainStrategy` |
| `training.py` | MR-DFlash mới | `OnlineMRDFlashModel`, `MRDFlashTrainStrategy`: thay context path, giữ anchor/label/loss/checkpoint contract |
| `inference.py` | MR-DFlash mới | `MRDFlashInferenceEngine`: prefill, draft block, greedy target verify, accepted-only memory update |
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

## MR-DFlash V1

MR-DFlash vẫn nhận feature contract cũ `hidden_states=[B,S,n_layers*H]`, vì vậy
feature capture và dataset không đổi format. Adapter chiếu feature concat thành
hai view:

1. HCA pool theo nhóm token liên tiếp với ratio `128`, cộng raw local memory
   trong cửa sổ `128`.
2. CSA pool với ratio `4`; indexer học Q/K và chọn tối đa `64` slot cho từng
   draft query. Local CSA memory luôn được đưa vào attention.

Đường forward mặc định gồm hai stage `HCA -> FFN` rồi `CSA -> FFN`, mỗi stage
giữ DFlash block-causal mask. `MRMemoryState.append()` chỉ nhận feature của
token đã được verifier chấp nhận.

## Cách chạy

### 1. CPU smoke (máy dev, không GPU)

```bash
cd src
python MR_DFlash/tests/test_smoke_cpu.py

# Contract smoke MR-DFlash
PYTHONPATH=. pytest -q MR_DFlash/tests/test_mr_memory.py \
  MR_DFlash/tests/test_mr_model_cpu.py \
  MR_DFlash/tests/test_mr_train_smoke.py \
  MR_DFlash/tests/test_mr_inference_cpu.py
```

### 1.1. Real-model smoke Qwen3-4B (offline)

Test end-to-end này không chạy mặc định vì cần nạp khoảng 8 GB weights. Nó
dùng snapshot Qwen3-4B local, capture 5 layer target, train MR-DFlash một
optimizer step, reload `draft_final.pt`, rồi chạy đủ `prefill → draft → verify
→ generate`:

```bash
CUDA_VISIBLE_DEVICES='' \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
MR_DFLASH_RUN_REAL_QWEN3_4B=1 MR_DFLASH_CPU_THREADS=8 \
PYTHONPATH=src .venv/bin/python -m pytest -q -s \
  src/MR_DFlash/tests/test_real_qwen3_4b_smoke.py
```

Có thể chỉ định snapshot khác bằng `MR_DFLASH_QWEN3_4B_PATH`. Test này xác
nhận functional correctness trên model thật; không phải benchmark GPU. Trên
máy dev T4 hiện tại, PyTorch không khởi tạo được CUDA do driver 12.4 không
tương thích torch `cu130`, nên lệnh trên cố ý chạy CPU.

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
Capture đồng thời ghi `manifest.json`; khi train, manifest và feature width
được kiểm tra trước khi chạy để tránh trộn cache khác target/layer.

### 3. Train offline

```bash
cd src
python -m MR_DFlash.run_train \
  --config MR_DFlash/configs/qwen3_8b_mr_dflash.yaml
```

Nếu `data.hidden_states_path` chưa có, `run_train` tự chạy capture vào
`output_dir/captured_features`. Có thể override CLI:
`--max-steps 100 --batch-size 1 --output-dir out ...`

Hoặc không dùng YAML, truyền thẳng flag: `--target-model-path ... --train-data-path ...`.

YAML mẫu hiện bật `architecture: mr_dflash` và `strategy: mr_dflash`; các
tham số DFlash (`learning_rate=6e-4`, `num_anchors=512`, `loss_decay_gamma=7`,
`block_size=16`, scheduler, accumulation, checkpoint) được giữ nguyên.

Trên B200, target bị freeze và draft/head/embed được replicate trên mỗi GPU;
2 GPU dùng DDP data-parallel. `training.batch_size` là batch **mỗi GPU**,
nhưng scheduler/horizon vẫn tính theo global batch. Với config Qwen3-4B mẫu:

```bash
# Một GPU được cấp riêng: giữ batch DFlash gốc = 4.
CUDA_VISIBLE_DEVICES=<GPU> FI_OFFLINE=1 PYTHONPATH=src \
python3 -m MR_DFlash.run_train \
  --config src/MR_DFlash/configs/qwen3_4b_mr_dflash.yaml \
  --device cuda --batch-size 4

# Hai GPU được cấp riêng: mỗi GPU batch 2, global batch vẫn = 4.
CUDA_VISIBLE_DEVICES=<GPU0>,<GPU1> FI_OFFLINE=1 PYTHONPATH=src \
torchrun --standalone --nproc_per_node=2 -m MR_DFlash.run_train \
  --config src/MR_DFlash/configs/qwen3_4b_mr_dflash.yaml \
  --device cuda --batch-size 2
```

Chỉ dùng `torchrun` khi đã được cấp đúng các GPU trong
`CUDA_VISIBLE_DEVICES`. Capture chỉ chạy ở rank 0, các rank còn lại chờ
barrier; checkpoint và metrics chỉ ghi ở rank 0. `eval_data_path` bật
evaluation loss/accuracy cuối run (hoặc định kỳ với
`training.eval_interval > 0`).

### 4. Reference inference

API chính nằm ở `MR_DFlash.inference.MRDFlashInferenceEngine`. CLI:

```bash
cd src
python -m MR_DFlash.inference \
  --target-model-path Qwen/Qwen3-8B \
  --draft-checkpoint-path outputs/mr-dflash-qwen3-8b/draft_final.pt \
  --prompt "Summarize the document" \
  --mask-token-id 151669 \
  --device cuda
```

Reference engine ưu tiên kiểm chứng semantics và hiện verify bằng full prefix;
không dùng lệnh này để kết luận latency GPU trước khi hoàn thành protocol trong
[`docs/mr_dflash_gpu_experiments.md`](../../docs/mr_dflash_gpu_experiments.md).

### 5. Kết quả

- `metrics.jsonl` — loss/acc/lr/grad_norm theo từng optimizer step.
- `checkpoint_step_*.pt` + `checkpoint_final.pt` — full state (draft weights +
  optimizer/scheduler + config) để resume (`--resume-from`).
- `draft_*.pt` — weights-only, tiện warm-start/export.

## Điểm cần chỉnh khi chạy thật (MR-DFlash)

- `attention_backend`: mặc định `sdpa` (dày đặc, CPU-safe). Với B200 và
  `num_anchors`/`max_length` lớn nên dùng `flex` (Flex Attention, không
  materialize mask) — đã có `build_dflash_flex_block_mask`.
- `mask_token_id`: phải đặt đúng token trong vocab target (Qwen3 VD `151669`).
- `feature_layer_ids` là schema cache MR (các layer concat làm context), còn
  `draft_init_layer_ids` là layer target dùng để copy weight khởi tạo draft;
  hai layout này độc lập. `init_draft_from_target=True` dùng một lần nạp target
  để giảm memory/time peak.
- Feature `hidden_states` capture bằng HF = toàn chuỗi (kể cả prompt). Nếu muốn
  tiết kiệm, chỉ capture prefix tới cuối assistant cần thiết.

## Đã lược bỏ so với SpecForge (có thể bổ sung sau)

- Online disaggregated (SGLang capture server + Mooncake) — không tái hiện
  standalone được; seam: thay `capture_dataset` bằng consumer đọc feature.
- EAGLE3/P-EAGLE/Domino/DSpark/D-PACE strategy (chỉ giữ DFlash; D-PACE loss
  dạng `dpace*` đã có trong `OnlineDFlashModel`).
- Liger kernel, config schema đầy đủ, FSDP/model-parallel và vocab mapping.
- DDP hiện chỉ là data-parallel cơ bản qua `torchrun`; chưa có elastic launch,
  resume tự động sau lỗi rank hoặc sharding target model.
- server integration/paged KV và kernel fused — reference inference hiện chạy
  target full-prefix để dễ kiểm chứng, chưa phải implementation benchmark.

Các giới hạn còn lại được ghi ở [`docs/mr_dflash.md`](../../docs/mr_dflash.md)
và protocol GPU deferred ở
[`docs/mr_dflash_gpu_experiments.md`](../../docs/mr_dflash_gpu_experiments.md).
