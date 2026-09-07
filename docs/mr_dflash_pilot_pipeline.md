# Pipeline thực nghiệm MR-DFlash pilot

Profile B200 100 GB hiện hành được mô tả tại
[`docs/mr_dflash_b200_profile.md`](mr_dflash_b200_profile.md). Các config
pilot dùng `batch_size=1`, accumulation 4 và `objective_chunk_blocks=64` để
giữ effective batch/fairness đồng thời chừa headroom VRAM.

Tài liệu này khóa thực nghiệm công bằng trên cùng target `Qwen/Qwen3-4B`:

| Variant | Draft | Feature target | Mục đích |
|---|---:|---|---|
| DFlash-2L | 2 layer | `[1,9,17,25,33]` | capacity-matched baseline |
| MR-DFlash-2S | HCA → CSA | `[1,9,17,25,33]` | proposed model |
| DFlash-5L | 5 layer | `[1,9,17,25,33]` | official-strength baseline |

Ba config đều dùng block size 16, DFlash loss, `num_anchors=512`,
`loss_decay_gamma=7`, seed 42, target-generated trajectory và
`init_draft_from_target=false`. Config pilot không dùng các config Qwen legacy
ở thư mục gốc vì chúng có thể mang init policy khác.

## Các artifact

```text
data/mr_dflash_pilot/
├── normalized/       # prompt-only và split prompt
├── regenerated/      # prompt + assistant do target sinh
├── tokenized_3k/     # R2, max_length=3072
├── tokenized/        # R3, max_length=8192
└── manifests/        # source/split/regeneration/tokenization provenance
```

`raw/` là input do người dùng cung cấp và không bị script ghi đè.
Hidden-state cache chỉ dành cho smoke/debug; pilot 100K dùng `feature_mode:
online`.

## Chuẩn hóa và split

```bash
PYTHONPATH=src python3 scripts/mr_dflash/prepare_sharegpt.py \
  --input data/raw/sharegpt.jsonl \
  --output data/mr_dflash_pilot/normalized/sharegpt_prompts.jsonl

PYTHONPATH=src python3 scripts/mr_dflash/prepare_arxiv.py \
  --input data/raw/arxiv.jsonl \
  --output data/mr_dflash_pilot/normalized/arxiv_prompts.jsonl \
  --tokenizer Qwen/Qwen3-4B

PYTHONPATH=src python3 scripts/mr_dflash/build_pilot_dataset.py \
  --sharegpt data/mr_dflash_pilot/normalized/sharegpt_prompts.jsonl \
  --arxiv data/mr_dflash_pilot/normalized/arxiv_prompts.jsonl \
  --output-root data/mr_dflash_pilot
```

Smoke có thể dùng `--sharegpt-count 4 --arxiv-count 6 --allow-short`.
Split mặc định là 90K/5K/5K với tỷ lệ ShareGPT:ArXiv = 40:60; ArXiv được
phân tầng theo độ dài nguồn.

## Regenerate bằng target

Chạy một lần cho từng split. Cần snapshot local hoặc runtime có quyền đọc
model; script không tự tải dataset.

```bash
for split in train val test; do
  PYTHONPATH=src python3 scripts/mr_dflash/regenerate_pilot.py \
    --input data/mr_dflash_pilot/normalized/${split}_prompts.jsonl \
    --output data/mr_dflash_pilot/regenerated/${split}.jsonl \
    --target-model-path Qwen/Qwen3-4B \
    --max-length 8192 --max-new-tokens 768 \
    --temperature 0 --device cuda --local-files-only \
    --manifest data/mr_dflash_pilot/manifests/regeneration_${split}.json
done
```

Nếu target được chạy bởi service khác, truyền `--responses-jsonl` thay cho
`--target-model-path`; file response phải có `id` và một trong các field
`assistant`, `response` hoặc `text`.

```bash
PYTHONPATH=src python3 scripts/mr_dflash/validate_pilot_dataset.py \
  --input data/mr_dflash_pilot/regenerated/train.jsonl \
  --tokenizer Qwen/Qwen3-4B --max-length 8192 \
  --require-generated
```

## Tokenize một lần cho mỗi context regime

```bash
for split in train val test; do
  PYTHONPATH=src python3 scripts/mr_dflash/tokenize_dataset.py \
    --input data/mr_dflash_pilot/regenerated/${split}.jsonl \
    --output data/mr_dflash_pilot/tokenized/${split} \
    --target-model-path Qwen/Qwen3-4B --max-length 8192 \
    --supervision-mode last_assistant --local-files-only \
    --provenance-manifest data/mr_dflash_pilot/manifests/tokenization_${split}.json
done
```

Lặp lại với `tokenized_3k/${split}` và `--max-length 3072` cho R2.
`attention_mask` được tạo từ độ dài thật, không suy ra bằng `input_ids != 0`.

## Fairness check và smoke train

```bash
PYTHONPATH=src python3 scripts/mr_dflash/check_fairness.py \
  src/MR_DFlash/configs/pilot_qwen3_4b/dflash_2l_8k.yaml \
  src/MR_DFlash/configs/pilot_qwen3_4b/mr_dflash_2s_8k.yaml \
  src/MR_DFlash/configs/pilot_qwen3_4b/dflash_5l_8k.yaml

PYTHONPATH=src python3 scripts/mr_dflash/run_pilot_matrix.py \
  --max-steps 2
```

Lệnh cuối chỉ in command. Muốn chạy thật phải thêm `--run`. Script kế thừa
nguyên `CUDA_VISIBLE_DEVICES`; nó không tự chọn GPU, không kill process và
không chạy đồng thời ba job. Ví dụ trên B200:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python3 \
  scripts/mr_dflash/run_pilot_matrix.py --run --device cuda --max-steps 2
```

Chỉ dùng device đã được allocate riêng. R0 có thể chạy với fixture test
`src/MR_DFlash/tests/test_online_pilot_pipeline.py`; không cần model Qwen thật.

## Các rung thực nghiệm

* R0: khoảng 100 sample, chuẩn hóa → regenerate → tokenize → online 1 step.
* R1: 256–1K sample, overfit để kiểm tra loss giảm và acceptance proxy tăng.
* R2: 10K sample, 3K tokens, chạy ba variant và đo VRAM/loss/acceptance.
* R3: 90K/5K/5K, 8K tokens, dùng checkpoint validation tốt nhất rồi evaluate
  trên holdout theo các bucket `0–2K`, `2–4K`, `4–6K`, `6–8K+`.

Metrics train có `acc_at_1..15` và `accept_ge_1..15`; các trường sau đo
acceptance proxy theo chuỗi prefix chứ không chỉ accuracy token độc lập.

## Evaluation

```bash
PYTHONPATH=src python3 scripts/mr_dflash/evaluate_pilot.py \
  --config src/MR_DFlash/configs/pilot_qwen3_4b/mr_dflash_2s_8k.yaml \
  --checkpoint outputs/mr_dflash_pilot/mr_dflash_2s_8k/checkpoint_final.pt \
  --input data/mr_dflash_pilot/regenerated/test.jsonl \
  --output outputs/mr_dflash_pilot/eval_mr_2s.jsonl \
  --max-new-tokens 128 --exactness-check --device cuda
```

`evaluate_pilot.py` hỗ trợ cả `architecture: dflash` và `mr_dflash`, dùng
target full-prefix verifier để ưu tiên exactness. Vì đây là reference path
Torch/einsum, latency chưa phải speedup production. Nếu bật
`--exactness-check`, vanilla greedy target phải khớp token-for-token; mismatch
làm script dừng.

```bash
PYTHONPATH=src python3 scripts/mr_dflash/summarize_pilot.py \
  --run dflash2=outputs/mr_dflash_pilot/dflash_2l_8k \
  --run mr2=outputs/mr_dflash_pilot/mr_dflash_2s_8k \
  --run dflash5=outputs/mr_dflash_pilot/dflash_5l_8k \
  --eval outputs/mr_dflash_pilot/eval_mr_2s.jsonl \
  --output outputs/mr_dflash_pilot/summary.json
```
