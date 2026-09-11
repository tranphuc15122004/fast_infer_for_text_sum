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

## Chạy toàn bộ preprocess bằng một script

`run_preprocess_pipeline.py` là entry point duy nhất cho phase dữ liệu/cache.
Nó không train drafter; sau khi cache hoàn tất, nhiều run train có thể dùng
chung các shard target. Các stage được thực hiện theo thứ tự:

| Stage | Đầu ra chính | Ý nghĩa |
|---|---|---|
| `prepare` | `normalized/*.jsonl`, `source_manifest.json`, `split_manifest.json` | Chuẩn hóa ShareGPT/ArXiv, chọn 50K + 50K và split 90/5/5 |
| `analyze` | `manifests/analysis.json` | Báo cáo nhanh source ratio, độ dài, duplicate và schema |
| `regenerate_{3k,8k}_{split}` | `regenerated_3k/*.jsonl` hoặc `regenerated/*.jsonl`, `*.skipped.jsonl`, regeneration manifests | Sinh assistant response deterministic bằng Qwen3-4B và ghi riêng sample không xử lý được |
| `validate_{3k,8k}_{split}` | `manifests/validation_*.json` | Kiểm tra assistant cuối, target provenance, token length |
| `tokenize_{3k,8k}_{split}` | `tokenized*/{split}/shard_*.pt`, `manifest.json` | Lưu `input_ids`, `loss_mask`, length; không lưu hidden |
| `cache_{3k,8k}_{split}` | `target_features_qwen3_4b_*/{split}/shard_*.pt`, `manifest.json` | Chạy target forward và lưu hidden `[1,9,17,25,33]` tại mọi offset |

Mỗi stage có log riêng tại `pipeline_logs/`, marker `success/failed` tại
`pipeline_state/`, và toàn pipeline có `pipeline_plan.json` cùng
`pipeline_summary.json`. Lỗi hệ thống hoặc CUDA OOM dừng tại stage để bảo toàn
tính đúng đắn; lỗi sample trong chế độ `skip` được ghi vào `*.skipped.jsonl`
và stage tiếp tục. Xem file `*.failed.json` để biết exit code/command và file
`.log` để xem traceback.

Trên server B200, với model đã mount tại path local, chạy:

```bash
cd /workspace/fast_infer_text_sum   # sửa thành thư mục repo thực tế nếu khác
export PYTHONPATH="$PWD/src"

python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --data-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot \
  --device cuda \
  --local-files-only
```

Hai context regime 3K và 8K đều được chuẩn bị. Mặc định cache `train` và
`val`; `test` vẫn được regenerate/tokenize để evaluation nhưng không cache
hidden vì các config train không dùng test cache. Muốn cache cả test, thêm:

```bash
--cache-splits train val test
```

### Chế độ full-context cho dataset/cache gốc

Nếu mục tiêu là giữ input ArXiv và response target đầy đủ trong phase
preprocess, không dùng 3K làm giới hạn sinh. Bật `--full-context`; pipeline sẽ
chỉ tạo một regime full và truyền `--preserve-full-input` cho target
regeneration. Với Qwen3-4B, có thể bắt đầu bằng native context 32K và response
budget 2048:

```bash
python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --data-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot_full \
  --device cuda \
  --local-files-only \
  --full-context \
  --full-context-length 32768 \
  --max-new-tokens 2048 \
  --overflow-policy skip \
  --sample-error-policy skip \
  --resume \
  --cache-splits train val
```

Output của mode này nằm tại `regenerated_full/`, `tokenized_full/` và
`target_features_qwen3_4b_full/`. Prompt không bị cắt một cách âm thầm. Nếu
prompt dài nhưng vẫn còn ít nhất một token trống, response budget sẽ tự giảm
đúng còn phần context còn lại và metadata ghi `generation_budget_clipped=true`.
Nếu prompt tự nó vượt `full-context-length`, sample được ghi vào
`regenerated_full/<split>.skipped.jsonl` kèm ID, số token và lý do; pipeline vẫn
tiếp tục với sample còn lại. B200 chỉ cung cấp VRAM/throughput, không làm tăng
giới hạn context của target.

`--sample-error-policy skip` cũng ghi các lỗi dữ liệu/tokenize/generate theo
từng sample vào cùng skipped report. Lỗi CUDA OOM không bị nuốt vì tiếp tục
trong cùng process có thể làm sai kết quả; stage dừng an toàn và chạy lại với
`--resume` sau khi giảm batch/context. Các JSON/manifest được ghi atomically,
output JSONL có `fsync`, và pipeline có lock theo `data-root` để không có hai
job cùng ghi một cache.

Mode full-context là bộ dữ liệu/cache gốc để audit hoặc train long-context.
Các config pilot 3K/8K hiện tại vẫn dùng artifact regime tương ứng và nên
được chạy riêng khi thực hiện R2/R3 fairness experiment.

### Chạy song song trên ba B200

Với ba GPU vật lý `1,2,3`, thêm `--parallel-gpu-ids 1 2 3`. Pipeline sẽ tạo
một worker process và một bản target model trên mỗi GPU; GPU 0 không bị dùng.
Mỗi worker xử lý shard dòng cố định, sau đó parent kiểm tra đủ ID rồi mới merge
output/cache. Các worker không ghi chung file nên có thể resume an toàn.

```bash
python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --data-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot_full \
  --device cuda \
  --local-files-only \
  --full-context --full-context-length 32768 \
  --max-new-tokens 2048 \
  --overflow-policy skip --sample-error-policy skip \
  --parallel-gpu-ids 1 2 3 \
  --cache-batch-size-long-context 2 \
  --cache-splits train val \
  --resume
```

`--cache-batch-size-long-context 2` nghĩa là mỗi worker/GPU cache đồng thời 2
sample cho mọi regime dài hơn 3K (8K, 16K, 32K, ...). Nó không phải batch size
8.000 sample hay 8.000 token. Đây là điểm bắt đầu phù hợp cho B200 180 GB;
nếu peak VRAM thực tế cao, hạ về `1`, còn không nên tăng cho tới khi pilot đo
xong. Tên cũ `--cache-batch-size-8k` vẫn được hỗ trợ như alias tương thích,
nhưng không nên dùng cho các run 32K.
Mỗi stage có worker log ở `.parallel_*/rank_*/worker.log`; trạng thái live ở
`.parallel_*/status.json`. Nếu một worker lỗi, merge không được publish và
pipeline dừng để bảo toàn coverage; chạy lại đúng lệnh sẽ tiếp tục từng worker.

Trước khi chạy thật, in toàn bộ command mà không đọc source/model:

```bash
python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --data-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot \
  --dry-run
```

Nếu phiên chạy bị ngắt, chạy lại đúng command sẽ skip stage đã có marker và
artifact hợp lệ. Một số cách debug/resume:

```bash
# Chỉ chạy lại một stage sau khi đã hoàn tất các dependency trước đó.
python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --data-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot \
  --only-stage cache_8k_train

# Chạy từ một stage đến hết 8K validation.
python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --data-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot \
  --from-stage regenerate_8k_train --stop-after validate_8k_test
```

`--only-stage` giả định các input/artifact của dependency đã tồn tại. Không
dùng `--no-resume` trên cùng data root nếu chưa chủ động kiểm tra artifact;
flag này dùng khi muốn chạy lại từ đầu với output mới hoặc data root mới.

## Các artifact

```text
/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/
├── normalized/       # prompt-only và split prompt
├── regenerated_3k/   # target trajectory với budget 3K
├── regenerated/      # target trajectory với budget 8K
├── tokenized_3k/     # R2, max_length=3072
├── tokenized/        # R3, max_length=8192
├── target_features_qwen3_4b_3k/
│   ├── train/        # hidden cache shard dùng chung cho 3K matrix
│   └── val/
├── target_features_qwen3_4b_8k/
│   ├── train/        # hidden cache shard dùng chung cho 8K matrix
│   └── val/
├── manifests/        # source/split/regeneration/validation/tokenization provenance
├── pipeline_logs/    # stdout/stderr từng stage, ví dụ cache_8k_train.log
├── pipeline_state/   # *.running/success/failed.json của từng stage
├── pipeline_plan.json
└── pipeline_summary.json
```

`raw/` là input do người dùng cung cấp và không bị script ghi đè.
Target-generated response được lưu trong `regenerated_3k/` và `regenerated/`;
hidden cache được lưu shard ở `target_features_*`. Ba variant trong matrix đọc
cùng một cache của mỗi context regime.

### Cache target trước train hay không?

Có. Config pilot B200 hiện tại dùng **phase cache offline trước train**:

```text
regenerated JSONL (đã có output của Qwen target)
    -> cache_target_features.py
    -> target Qwen frozen forward theo batch
    -> hook lấy [1,9,17,25,33] tại mọi offset hợp lệ
    -> sharded feature cache + manifest
    -> DFlash/MR-DFlash train chỉ đọc cache
```

Cache này giúp các run DFlash-2L, MR-DFlash-2S và DFlash-5L không phải chạy
Qwen target lại. `input_ids`, `loss_mask` và hidden states được lưu cùng
sample; hidden states bao gồm cả prompt và response, không chỉ token có
`loss_mask=1`, vì memory của drafter cần mọi offset trước anchor. Response text
vẫn nằm trong regenerated JSONL để audit/reproduce/evaluate.

Không cache full target logits: objective hiện tại chỉ cần hidden features,
target embedding và frozen lm_head khi train. Full logits sẽ làm dung lượng
tăng thêm nhiều lần mà không đem lại thông tin cần thiết cho pipeline này.

Cache sharded dùng nhiều sample trong một file `.pt`, có `manifest.json`, LRU
reader và `--resume`. Cache `.ckpt` một-file/mẫu trong `capture.py` vẫn được
giữ cho backward compatibility và smoke nhỏ. Khi một config explicit
`data.hidden_states_path` trỏ tới cache chưa hoàn chỉnh, `run_train.py` sẽ
dừng và chỉ rõ lệnh cache; nó không tự chạy một pass target đắt tiền trong lúc
bắt đầu train.

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
  --output-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot \
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
  --output /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/normalized/sharegpt_prompts.jsonl

PYTHONPATH=src python3 scripts/mr_dflash/prepare_arxiv.py \
  --input "$ARXIV_SOURCE" \
  --output /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/normalized/arxiv_prompts.jsonl \
  --tokenizer "$TARGET_MODEL"

PYTHONPATH=src python3 scripts/mr_dflash/build_pilot_dataset.py \
  --sharegpt /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/normalized/sharegpt_prompts.jsonl \
  --arxiv /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/normalized/arxiv_prompts.jsonl \
  --output-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot
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
    --input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/normalized/${split}_prompts.jsonl \
    --output /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated_3k/${split}.jsonl \
    --target-model-path "$TARGET_MODEL" \
    --max-length 3072 --max-new-tokens 768 \
    --temperature 0 --device cuda --local-files-only \
    --manifest /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/manifests/regeneration_3k_${split}.json \
    --resume
done

for split in train val test; do
  PYTHONPATH=src python3 scripts/mr_dflash/regenerate_pilot.py \
    --input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/normalized/${split}_prompts.jsonl \
    --output /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated/${split}.jsonl \
    --target-model-path "$TARGET_MODEL" \
    --max-length 8192 --max-new-tokens 768 \
    --temperature 0 --device cuda --local-files-only \
    --manifest /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/manifests/regeneration_8k_${split}.json \
    --resume
done
```

Nếu target được chạy bởi service khác, truyền `--responses-jsonl` thay cho
`--target-model-path`; file response phải có `id` và một trong các field
`assistant`, `response` hoặc `text`.

```bash
PYTHONPATH=src python3 scripts/mr_dflash/validate_pilot_dataset.py \
  --input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated_3k/train.jsonl \
  --tokenizer "$TARGET_MODEL" --max-length 3072 \
  --expected-target-model "$TARGET_MODEL" --require-generated

PYTHONPATH=src python3 scripts/mr_dflash/validate_pilot_dataset.py \
  --input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated/train.jsonl \
  --tokenizer "$TARGET_MODEL" --max-length 8192 \
  --expected-target-model "$TARGET_MODEL" \
  --require-generated
```

## Tokenize một lần cho mỗi context regime

```bash
for split in train val test; do
  PYTHONPATH=src python3 scripts/mr_dflash/tokenize_dataset.py \
    --input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated/${split}.jsonl \
    --output /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/tokenized/${split} \
    --target-model-path "$TARGET_MODEL" --max-length 8192 \
    --supervision-mode last_assistant --local-files-only \
    --provenance-manifest /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/manifests/tokenization_8k_${split}.json \
    --resume
done

for split in train val test; do
  PYTHONPATH=src python3 scripts/mr_dflash/tokenize_dataset.py \
    --input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated_3k/${split}.jsonl \
    --output /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/tokenized_3k/${split} \
    --target-model-path "$TARGET_MODEL" --max-length 3072 \
    --supervision-mode last_assistant --local-files-only \
    --provenance-manifest /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/manifests/tokenization_3k_${split}.json \
    --resume
done
```
`attention_mask` được tạo từ độ dài thật, không suy ra bằng `input_ids != 0`.

## Cache hidden states một lần trên B200

Chạy sau khi đã validate và regenerate đúng context regime. `--batch-size`
chỉ ảnh hưởng throughput của phase cache, không thay đổi batch/optimizer của
training. Với regime dài hơn 3K, tham số tương ứng trong pipeline là
`--cache-batch-size-long-context`; giá trị của nó là số sample trên mỗi GPU,
không phải độ dài context. Bắt đầu với 2 trên một B200 180 GB cho 8K và với
1 cho 32K, sau đó tăng khi đã kiểm tra peak VRAM.
`--bucket-buffer-size` giúp ghép các sample gần độ dài nhau.

## Benchmark target generation/cache trên B200

Script `scripts/mr_dflash/target_cache_benchmark.py` không ghi đè cache. Nó
chạy bốn phép đo trên cùng một prompt:

* `hf_generate`: target `generate()` với `use_cache=True`;
* `current_causal_lm_full_capture`: full forward bằng
  `AutoModelForCausalLM`, tương ứng phase cache hiện tại;
* `backbone_only_full_capture`: full forward bằng `AutoModel`, không tính
  LM head/logits;
* `fused_generate_capture`: prefill + decode bằng KV cache, capture hidden
  trong cùng một lượt.

Script kiểm tra token output của `hf_generate` và `fused_generate_capture`,
đồng thời so sánh hidden trajectory của full forward với KV path. Có thể chạy
trên sample đã target-generate:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0

python3 scripts/mr_dflash/target_cache_benchmark.py \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --input-jsonl /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot_full/regenerated_full/val.jsonl \
  --sample-index 0 \
  --max-new-tokens 128 \
  --target-layer-ids 1 9 17 25 33 \
  --attn-implementation auto \
  --device cuda --torch-dtype bfloat16 --local-files-only \
  --report /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot_full/manifests/target_cache_benchmark_auto.json
```

Để so sánh backend, chạy lại cùng sample với `--attn-implementation sdpa` và
`--attn-implementation flash_attention_2`. Nếu FlashAttention-2 chưa được
cài hoặc không tương thích với stack CUDA, script sẽ báo lỗi khi load model;
không tự ghi nhận đó là một kết quả hợp lệ. Trước khi dùng tối ưu fused, cần
kiểm tra `comparisons.hf_vs_fused_tokens.exact=true` và hidden comparison nằm
trong tolerance đã chọn (`--atol`, `--rtol`).

```bash
for split in train val test; do
  PYTHONPATH=src python3 scripts/mr_dflash/cache_target_features.py \
    --target-model-path "$TARGET_MODEL" \
    --data-path /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated_3k/${split}.jsonl \
    --output-path /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/target_features_qwen3_4b_3k/${split} \
    --target-layer-ids 1 9 17 25 33 \
    --max-length 3072 --batch-size 2 --bucket-buffer-size 16 \
    --shard-size 64 --supervision-mode last_assistant \
    --device cuda --local-files-only --resume
done

for split in train val test; do
  PYTHONPATH=src python3 scripts/mr_dflash/cache_target_features.py \
    --target-model-path "$TARGET_MODEL" \
    --data-path /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated/${split}.jsonl \
    --output-path /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/target_features_qwen3_4b_8k/${split} \
    --target-layer-ids 1 9 17 25 33 \
    --max-length 8192 --batch-size 1 --bucket-buffer-size 8 \
    --shard-size 32 --supervision-mode last_assistant \
    --device cuda --local-files-only --resume
done
```

Trước khi chạy 50K + 50K, dùng output fixture nhỏ và kiểm tra report; không
đổi target revision, layer IDs, chat template hoặc supervision mode giữa các
cache dùng chung trong một matrix. Với Qwen3-4B, feature width là 12,800 và
BF16 tốn khoảng 25.6 KB/token; 100K sample dài vài nghìn token có thể cần
nhiều TB. Nếu filesystem không đủ, dừng ở R2 hoặc dùng bản copy config online,
không train trên cache ghi dở.

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

### E23/E24 objective screen

Sau khi feature train/val đã tồn tại, chạy cùng một config DFlash-5L để giữ
model, dữ liệu, batch, seed và lịch học cố định:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python3 \
  scripts/mr_dflash/run_e23_e24.py \
  --config src/MR_DFlash/configs/pilot_qwen3_4b/dflash_5l_3k.yaml \
  --output-root outputs/mr_dflash_e23_e24 \
  --eval-input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated_3k/test.jsonl \
  --local-files-only \
  --stop-on-failure
```

E23 chạy `dflash`, `dpace`, `spec-auf`; E24 chạy
`dpace-hard-negative-all` và `dpace-hard-negative-shallow`. E24 dùng
`Top-32` competitor và `lambda=0.25` cho screen đầu tiên. Không trộn các
checkpoint giữa conditions. Mỗi condition phải có `metrics.jsonl`,
`eval_metrics.json`, checkpoint và `run.log`; nếu một condition lỗi, kiểm tra
manifest trước khi kết luận về objective.

Smoke không tải model thật:

```bash
PYTHONPATH=src:. python3 -m pytest -q \
  src/MR_DFlash/tests/test_research_objectives.py \
  src/MR_DFlash/tests/test_research_runner.py
```

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
  --input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated/test.jsonl \
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

## Kiểm chứng hidden state với KV cache

Target causal Transformer phải giữ nguyên hidden state ở các vị trí đã xử lý
khi chuyển từ full forward sang prefill + decode bằng `past_key_values`. Có thể
kiểm chứng trên một sample thật đã regenerate bằng:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=1

python3 scripts/mr_dflash/verify_kv_hidden_equivalence.py \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot_full/regenerated_full/train.jsonl \
  --sample-index 0 \
  --max-length 32768 \
  --decode-tokens 0 \
  --target-layer-ids 1 9 17 25 33 \
  --device cuda \
  --torch-dtype bfloat16 \
  --local-files-only \\
  --report /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot_full/manifests/kv_hidden_check_sample_0.json
```

`--decode-tokens 0` so sánh toàn bộ response; có thể đặt một số dương để
smoke test nhanh. Script dùng đúng continuation đã được target generate, nên
hai đường chạy xử lý cùng một chuỗi token. Kết quả `status=pass` ở các layer
được chọn là điều kiện cần để capture hidden trong lúc decode. Sai số được
đánh giá ở float32 với mặc định `atol=rtol=0.05`, phù hợp với khác biệt số học
có thể có khi chạy BF16. Test này xác nhận tính đúng đắn và loại bỏ forward
lặp lại; nó chưa thay đổi cache pipeline hiện tại.
