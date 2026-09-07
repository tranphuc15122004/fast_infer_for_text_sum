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
| `mr_model.py` | MR-DFlash mới | `MRDFlashDraftModel`: DFlash joint attention với HCA/CSA memory + FFN |
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
3. Mọi block chạy **song song** qua draft model. Attention mặc định cho phép
   toàn bộ draft trong cùng block và không cross-block; khi bật
   `sliding_attention` thì dùng causal offset trong block. Cả train và
   inference dùng cùng semantics này.
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

1. HCA pool theo nhóm token **đủ đầy** với ratio `128`; đuôi chưa đủ nhóm
   nằm ở `pending_hca`. Raw local HCA có cửa sổ `128` và được gather tương
   đối theo từng anchor trong training.
2. CSA pool theo nhóm đủ với ratio `4`; `CSAIndexer` chấm điểm từng query.
  Training mặc định chạy dense score-bias trong warm-up rồi chuyển hard
  Top-k tối đa `64` theo `indexer_dense_steps`. Local và selected CSA được
  nối vào cùng một context trước một softmax. Score dùng
  `ReLU(dot(q_head, k_head))`; V1 score-bias là bridge để hard selection vẫn
  có gradient.

Đường forward mặc định gồm các stage xen kẽ `HCA -> CSA -> HCA -> ...`.
Mỗi stage dùng một DFlash joint attention với `KV=[MR context; draft block]`,
áp RoPE cho query, raw local position và compressed group-end position.
`MRMemoryState.append()` chỉ nhận feature của token đã được verifier chấp
nhận; HCA/CSA giữ pending position độc lập.

Trong training, `N` anchor blocks được reshape thành batch `[B*N,K,H]`; local
memory query-relative thành `[B*N,W,H]`, còn global memory giữ batch `[B,C,H]`
và dùng mapping block. Joint attention vì vậy chỉ tạo draft logits `K×K` cho
mỗi block. Shared context được chiếu K/V một lần;
CSA Top-k gather trên projected K/V, không gather hidden-size memory rồi mới
project theo query. Đây là layout bắt buộc cho `num_anchors=512`.

Checkpoint DFlash có thể truyền qua `draft_checkpoint_path` để converter
copy attention/MLP/norm vào mọi MR stage và copy `fc` vào hai adapter. Các
compressor/indexer không có tensor tương ứng nên khởi tạo mới. Native MR
checkpoint được kiểm tra strict khi infer.

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

Để comparison công bằng trên Qwen3-4B, dùng cùng
`feature_layer_ids: [1, 9, 17, 25, 33]` cho ba config:

- `configs/qwen3_4b_dflash_1l.yaml`: DFlash-1L baseline;
- `configs/qwen3_4b_dflash_2l.yaml`: DFlash-2L depth/capacity control;
- `configs/qwen3_4b_mr_dflash.yaml`: MR-HCA+CSA 2-stage.

DFlash-2L kiểm soát số draft layer, nhưng không phải parameter-count matching
tuyệt đối vì MR có thêm compressor/indexer. Cần ghi parameter count và peak
VRAM của từng run trước khi kết luận gain. Các config Qwen3-8B cũng dùng cùng
5 feature layers; `qwen3_8b_dflash_2l.yaml` là control tương ứng.

Với target Llama 3.1 8B Instruct, dùng ma trận theo ngân sách DFlash 5 layer:

| Run | Config | Feature layers | MR stages / indexer | Trainable params |
|---|---|---|---|---:|
| DFlash-5L | `configs/llama3_1_8b_dflash_5l.yaml` | `[1,8,15,22,29]` | 5 DFlash layers, MLP 12288 | 1,048,626,432 |
| MR-4S | `configs/llama3_1_8b_mr_dflash.yaml` | `[1,8,15,22,29]` | 4 / 4096, MLP 12288 | 1,040,786,432 |
| MR-4S exact | `configs/llama3_1_8b_mr_dflash_exact_params.yaml` | `[1,8,15,22,29]` | 4 / 5120, MLP 12288 | 1,049,175,040 |

`MR-4S` là cấu hình chính vì thấp hơn DFlash-5L 0.75% và không nhân đôi
chiều rộng Indexer. `MR-4S exact` chỉ dùng làm ablation parameter-control
(chênh +0.052%); `indexer_dim=5120` làm tăng chi phí retrieval nên không dùng
nó làm headline latency duy nhất. Các số chỉ tính draft parameters; target
Llama 3.1 8B được freeze và không tính vào ngân sách. Cả ba config dùng cùng
feature contract và DFlash objective để loại confound do số layer feature.

Trên server, target thường là snapshot local. Có thể dùng trực tiếp các YAML
trên nếu model ID đã được cache, hoặc override:

```bash
PYTHONPATH=src python3 -m MR_DFlash.run_train \
  --config src/MR_DFlash/configs/llama3_1_8b_mr_dflash.yaml \
  --target-model-path "$MODEL_TARGET" \
  --device cuda
```

Phải capture lại feature với `feature_layer_ids=[1,8,15,22,29]`,
`block_size=16` và kiểm tra
manifest trước khi train; cache Qwen3 hoặc cache Llama khác layer list không
tương thích. `mask_token_id=128002` đã được đặt explicit nhưng vẫn cần xác
nhận lại với tokenizer snapshot local trước run thật.

Cache feature đã capture theo một layer phải được capture lại; manifest và
feature width phải khớp năm layer trên. Dataset loader có kiểm tra
`feature_layer_ids`/target provenance để chặn việc trộn cache cũ vào baseline
mới.

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
  --device cuda --local-files-only
```

Checkpoint mới chứa `config_yaml`, vì vậy CLI tự khôi phục feature layer
IDs, block size, số stage, compression ratio, local window, Top-k và dtype.
Các flag architecture truyền thủ công vẫn được ưu tiên khi cần tương thích
checkpoint cũ không có metadata.

Reference engine ưu tiên kiểm chứng semantics và hiện verify bằng full prefix;
không dùng lệnh này để kết luận latency GPU trước khi hoàn thành protocol trong
[`docs/mr_dflash_gpu_experiments.md`](../../docs/mr_dflash_gpu_experiments.md).
Thêm `--timing-json` để CLI xuất phân rã `prefill_s/draft_s/verify_s/total_s`,
số vòng và số token proposal được accept cho pilot latency.

### 5. Kết quả

- `metrics.jsonl` — loss/acc/lr/grad_norm, trainable parameter count,
  `step_time_s` và `tokens_per_second` theo từng optimizer step; CUDA bổ sung
  peak memory allocated/reserved để kiểm tra pilot.
- `checkpoint_step_*.pt` + `checkpoint_final.pt` — full state (draft weights +
  optimizer/scheduler + config) để resume (`--resume-from`).
- `draft_*.pt` — weights-only, tiện warm-start/export.

## Điểm cần chỉnh khi chạy thật (MR-DFlash)

- `attention_backend`: DFlash gốc hỗ trợ `sdpa`/`flex`. MR-DFlash joint
  attention V1 dùng path tensor chung để giữ correctness và hiện cấu hình
  khuyến nghị là `sdpa`; chưa claim Flex kernel speedup cho MR.
- `mask_token_id`: phải đặt đúng token trong vocab target (Qwen3 VD `151669`).
- `feature_layer_ids` là schema cache MR (các layer concat làm context),
  `draft_init_layer_ids` là layout init DFlash, còn
  `mr_stage_init_layer_ids` có một layer target cho mỗi MR stage; ba layout
  này độc lập. `init_draft_from_target=True` dùng một lần nạp target để giảm
  memory/time peak.
- `CSAIndexer` dùng scale `1/sqrt(head_dim)` khi score theo nhiều head. V1
  mặc định `indexer_num_heads=1` và đưa score vào attention bias để hard Top-k
  vẫn có gradient; 4/8 head chỉ là ablation, không gọi là Lightning Indexer
  faithful.
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
## Pilot data/online training

Pipeline thực nghiệm đã khóa cho câu hỏi MR-DFlash trên long context nằm tại
[`docs/mr_dflash_pilot_pipeline.md`](../../docs/mr_dflash_pilot_pipeline.md).
Pipeline dùng tokenized shard + online hidden features; feature cache offline
chỉ giữ cho smoke/debug. Ba config công bằng là DFlash-2L, MR-DFlash-2S và
DFlash-5L trong `configs/pilot_qwen3_4b/`.
