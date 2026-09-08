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
├── regenerated_3k/   # target trajectory với budget 3K
├── regenerated/      # target trajectory với budget 8K
├── tokenized_3k/     # R2, max_length=3072
├── tokenized/        # R3, max_length=8192
└── manifests/        # source/split/regeneration/tokenization provenance
```

`raw/` là input do người dùng cung cấp và không bị script ghi đè.
Hidden-state cache chỉ dành cho smoke/debug; pilot 100K dùng `feature_mode:
online`.

### Cache target trước train hay không?

Với các config pilot hiện tại, **không có phase cache hidden state trước train**:

```text
tokenized shard có sẵn
    -> trainer đọc input_ids/attention_mask
    -> target Qwen frozen forward trong từng batch
    -> hook lấy [1,9,17,25,33], detach
    -> draft model + DFlash/MR-DFlash loss
```

Target vẫn được load một lần và giữ `eval/frozen`, nhưng hidden states chỉ tồn
tại trong batch hiện tại rồi được giải phóng. Cách này tránh feature cache nhiều
terabyte trên 100K mẫu. Nếu đặt `data.feature_mode: offline`,
`run_train.py` sẽ chạy `capture.py` trước train (hoặc dùng
`data.hidden_states_path`) và lưu từng sample vào thư mục
`captured_features`; đó là mode cache trước train, không phải mode pilot chính.

## Dữ liệu server và schema thực tế

`data/train_data_on_server/sample.txt` là file inventory/mẫu, không phải file
dữ liệu để đưa thẳng vào DataLoader. Nó chỉ ra hai source trên B200:

| Source | Format | Quy mô ghi trong inventory | Field dùng cho pilot |
|---|---|---:|---|
| ShareGPT | JSON array `.json` | 64.000 | `id`, `conversations[].from/value` |
| ArXiv | JSONL `.jsonl` | gần 200.000 | `id`, `text[]`; giữ `summary[]`, `label` làm reference/metadata |

ShareGPT có role `human/gpt`. Chuẩn hóa sẽ giữ system và câu hỏi user cuối,
loại các câu trả lời `gpt` cũ; assistant cuối sẽ được sinh lại bởi target.
ArXiv có `text` là danh sách đoạn văn, nên các phần tử được nối bằng hai dòng
trống. `summary` không đi vào loss train; nó được lưu tại
`metadata.reference_summary` để đánh giá ROUGE sau này. `label` được giữ trong
metadata và không được dùng làm token target.

## Chuẩn hóa và split trên B200

Wrapper dưới đây nhận trực tiếp đúng hai path trong `sample.txt`, chỉ đọc raw
source và ghi artifact mới. Truyền tokenizer target để ArXiv được stratify
theo token length thay vì số ký tự:

```bash
export PYTHONPATH=src
export TARGET_MODEL=Qwen/Qwen3-4B
export SHAREGPT_SOURCE=/workspace/storage-shared/nlp/tungdd11/tungdecoder/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json
export ARXIV_SOURCE=/workspace/storage-shared/nlp/dungdx4/datasets/arxiv/train.label.jsonl
```

Lệnh chạy đúng là:

```bash
python3 scripts/mr_dflash/prepare_server_data.py \
  --sharegpt-source "$SHAREGPT_SOURCE" \
  --arxiv-source "$ARXIV_SOURCE" \
  --output-root data/mr_dflash_pilot \
  --sharegpt-count 50000 --arxiv-count 50000 \
  --tokenizer "$TARGET_MODEL"
```

Trước khi scale, chạy fixture nhỏ vào thư mục riêng và xem report:

```bash
python3 scripts/mr_dflash/prepare_server_data.py \
  --sharegpt-source "$SHAREGPT_SOURCE" \
  --arxiv-source "$ARXIV_SOURCE" \
  --output-root data/mr_dflash_pilot_smoke \
  --sharegpt-count 40 --arxiv-count 60 --tokenizer "$TARGET_MODEL"

python3 scripts/mr_dflash/analyze_pilot_data.py \
  --input data/mr_dflash_pilot_smoke/normalized/pilot_prompts.jsonl \
  --output data/mr_dflash_pilot_smoke/manifests/analysis.json \
  --limit 100
```

Kiểm tra report và spot-check prompt trước khi cho phép chạy 50K/50K. Có thể
tiếp tục chuẩn hóa bị gián đoạn bằng `--resume`; không dùng `--resume` nếu
đang trỏ vào artifact của source khác.

Nếu muốn gọi từng bước thay vì wrapper:

```bash
PYTHONPATH=src python3 scripts/mr_dflash/prepare_sharegpt.py \
  --input "$SHAREGPT_SOURCE" \
  --output data/mr_dflash_pilot/normalized/sharegpt_prompts.jsonl

PYTHONPATH=src python3 scripts/mr_dflash/prepare_arxiv.py \
  --input "$ARXIV_SOURCE" \
  --output data/mr_dflash_pilot/normalized/arxiv_prompts.jsonl \
  --tokenizer "$TARGET_MODEL"

PYTHONPATH=src python3 scripts/mr_dflash/build_pilot_dataset.py \
  --sharegpt data/mr_dflash_pilot/normalized/sharegpt_prompts.jsonl \
  --arxiv data/mr_dflash_pilot/normalized/arxiv_prompts.jsonl \
  --output-root data/mr_dflash_pilot
```

Split mặc định là 90K/5K/5K với tỷ lệ ShareGPT:ArXiv = 50:50; ArXiv được
phân tầng theo `source_token_length`. Manifest ghi lại raw path, source count,
split ID và SHA-256 để tránh dùng nhầm dữ liệu.

## Regenerate bằng target

Chạy riêng cho từng context regime. Cần snapshot local hoặc runtime có quyền
đọc model; script không tự tải dataset. Không tái sử dụng trajectory 8K để
tokenize 3K vì truncation sau khi generate có thể cắt response; mỗi regime
được regenerate với prompt budget tương ứng.

```bash
for split in train val test; do
  PYTHONPATH=src python3 scripts/mr_dflash/regenerate_pilot.py \
    --input data/mr_dflash_pilot/normalized/${split}_prompts.jsonl \
    --output data/mr_dflash_pilot/regenerated_3k/${split}.jsonl \
    --target-model-path "$TARGET_MODEL" \
    --max-length 3072 --max-new-tokens 768 \
    --temperature 0 --device cuda --local-files-only \
    --manifest data/mr_dflash_pilot/manifests/regeneration_3k_${split}.json \
    --resume
done

for split in train val test; do
  PYTHONPATH=src python3 scripts/mr_dflash/regenerate_pilot.py \
    --input data/mr_dflash_pilot/normalized/${split}_prompts.jsonl \
    --output data/mr_dflash_pilot/regenerated/${split}.jsonl \
    --target-model-path "$TARGET_MODEL" \
    --max-length 8192 --max-new-tokens 768 \
    --temperature 0 --device cuda --local-files-only \
    --manifest data/mr_dflash_pilot/manifests/regeneration_8k_${split}.json \
    --resume
done
```

Nếu target được chạy bởi service khác, truyền `--responses-jsonl` thay cho
`--target-model-path`; file response phải có `id` và một trong các field
`assistant`, `response` hoặc `text`.

```bash
PYTHONPATH=src python3 scripts/mr_dflash/validate_pilot_dataset.py \
  --input data/mr_dflash_pilot/regenerated_3k/train.jsonl \
  --tokenizer "$TARGET_MODEL" --max-length 3072 \
  --expected-target-model "$TARGET_MODEL" --require-generated

PYTHONPATH=src python3 scripts/mr_dflash/validate_pilot_dataset.py \
  --input data/mr_dflash_pilot/regenerated/train.jsonl \
  --tokenizer "$TARGET_MODEL" --max-length 8192 \
  --expected-target-model "$TARGET_MODEL" \
  --require-generated
```

## Tokenize một lần cho mỗi context regime

```bash
for split in train val test; do
  PYTHONPATH=src python3 scripts/mr_dflash/tokenize_dataset.py \
    --input data/mr_dflash_pilot/regenerated/${split}.jsonl \
    --output data/mr_dflash_pilot/tokenized/${split} \
    --target-model-path "$TARGET_MODEL" --max-length 8192 \
    --supervision-mode last_assistant --local-files-only \
    --provenance-manifest data/mr_dflash_pilot/manifests/tokenization_8k_${split}.json \
    --resume
done

for split in train val test; do
  PYTHONPATH=src python3 scripts/mr_dflash/tokenize_dataset.py \
    --input data/mr_dflash_pilot/regenerated_3k/${split}.jsonl \
    --output data/mr_dflash_pilot/tokenized_3k/${split} \
    --target-model-path "$TARGET_MODEL" --max-length 3072 \
    --supervision-mode last_assistant --local-files-only \
    --provenance-manifest data/mr_dflash_pilot/manifests/tokenization_3k_${split}.json \
    --resume
done
```
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
