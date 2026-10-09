# Phase 1: regenerate, validate, and cache DFlash data

Phase 1 của `src/Finetuning` tạo teacher trajectories từ target Qwen, kiểm tra
chất lượng rồi cache hidden states để trainer DFlash có thể đọc. Launcher này
dừng sau cache; nó không chạy huấn luyện.

Pipeline MR-DFlash không thay thế trực tiếp được quy trình này. MR-DFlash nhận
prompt-only records và tạo cache theo contract riêng; Finetuning cần các record
`id`, `document`, `summary`, giữ reference gốc trong `reference_summary` nếu có,
rồi ghi feature store theo manifest của
`Finetuning.features.OfflineFeatureDataset`.

## Input và thiết lập

Đưa vào hai file UTF-8 JSONL đã được chia sẵn train/eval. Mỗi dòng cần có dạng:

```json
{"id":"vi-001","document":"Văn bản nguồn...","summary":"Tóm tắt tham chiếu..."}
```

Launcher không chuẩn hóa corpus, chia split hay lấy mẫu; hãy chuẩn bị hai file
đó trước. Model phải là snapshot Qwen local có `config.json`; server có thể chạy
offline.

## Tạo input từ dữ liệu phân tầng MR-DFlash đã có

Dataset phân tầng 50K đã build sẵn trên server tại:

```text
/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_50k_50_50_10k/normalized/
```

Trong đó dùng `train_prompts.jsonl` và `val_prompts.jsonl`; không đưa
`test_prompts.jsonl` vào train hay validation. Bộ stratified 50K có 45.000 train,
2.500 validation và 2.500 test. Script dưới đây tạo schema `id/document/summary`,
giữ nguyên hai split, loại prompt trùng bên trong cùng split và ghi số lượng vào
`report.json`. ID lặp vẫn là lỗi; ID/prompt/document trùng chéo train-eval làm
script dừng để tránh leakage. Nó không ghi đè dữ liệu nguồn hoặc output đã tồn tại.

`summary` ở đây chỉ lấy `metadata.reference_summary` gốc cho ArXiv; ShareGPT để
trống vì split prompt-only không có gold summary. Target generation vẫn chạy cho
cả hai nguồn; vì vậy ROUGE validation chỉ có ý nghĩa trên mẫu ArXiv có reference.
Các hội thoại ShareGPT được flatten thành văn bản có nhãn role, nên cần đặt
`data.prompt_template: "{document}"` để không bọc chúng bằng chỉ dẫn tóm tắt
tiếng Việt. Cách biểu diễn văn bản này không giữ nguyên token-level role
structure của prompt gốc; nếu cần parity tuyệt đối cho ShareGPT, pipeline cần
nhận `conversations` trực tiếp.

Trước tiên tạo pilot nhỏ riêng để xem report và spot-check nội dung:

```bash
export MR_STRATIFIED_DIR=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_50k_50_50_10k/normalized
export PHASE1_PILOT_INPUTS="$PWD/outputs/finetuning/phase1_mr_pilot_inputs"

python3 scripts/prepare_finetuning_phase1_inputs.py \
  --stratified-dir "$MR_STRATIFIED_DIR" \
  --output-dir "$PHASE1_PILOT_INPUTS" \
  --max-records-per-split 200 \
  --progress

python3 -m json.tool "$PHASE1_PILOT_INPUTS/report.json"
python3 - "$PHASE1_PILOT_INPUTS/train.jsonl" "$PHASE1_PILOT_INPUTS/eval.jsonl" <<'PY'
import json, sys
for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as handle:
        row = json.loads(next(line for line in handle if line.strip()))
    print(path, "id=", row["id"], "source=", row["metadata"]["source"],
          "document_chars=", len(row["document"]),
          "reference_kind=", row["metadata"]["reference_kind"])
    assert row["document"].strip()
PY
```

Sau khi pilot report/spot-check hợp lệ, tạo bộ đầy đủ vào thư mục mới. Đầu ra có
tối đa 45K train và 2.5K eval; số thực tế sau khi loại prompt trùng nằm trong
`report.json`. 2.5K test vẫn được giữ ngoài run này:

```bash
export PHASE1_INPUTS="$PWD/outputs/finetuning/mr_dflash_50k_phase1_inputs"
python3 scripts/prepare_finetuning_phase1_inputs.py \
  --stratified-dir "$MR_STRATIFIED_DIR" \
  --output-dir "$PHASE1_INPUTS" \
  --progress

export TRAIN_INPUT="$PHASE1_INPUTS/train.jsonl"
export EVAL_INPUT="$PHASE1_INPUTS/eval.jsonl"
```

Dùng config riêng cho run này để giữ prompt và budget phù hợp với corpus phân
tầng 10K input/1K output; file gốc không bị sửa:

```bash
export PHASE1_CONFIG="$PHASE1_INPUTS/qwen3_4b_phase1.yaml"
python3 - "$PWD/src/Finetuning/configs/qwen3_4b.yaml" "$PHASE1_CONFIG" <<'PY'
import sys, yaml
source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
config["data"]["prompt_template"] = "{document}"
config["data"]["max_length"] = 16384
config["data"]["max_source_tokens"] = 10240
config["data"]["max_summary_tokens"] = 1024
with open(destination, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
print(destination)
PY
```

Ngân sách token lấy từ config. Mặc định trong `qwen3_4b.yaml` là tổng 2.048
token, tối đa 1.536 token cho document và 384 token cho summary. Nếu dữ liệu đã
chuẩn bị có document dài hơn, hãy điều chỉnh ba budget trong config để phù hợp
với context của target trước khi chạy; Phase 1 không tự đặt budget 10k/1k.

## Chạy trên hai GPU B200

Trên server, dùng Python 3.12 chung và bật hai GPU muốn cấp cho job. Thay các
đường dẫn train/eval/model theo nơi dữ liệu và model thực sự nằm:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

export FAST_INFER_PYTHON=python3
export CUDA_VISIBLE_DEVICES=0,1
export FINETUNE_GENERATION_BACKEND=hf
export FINETUNE_CAPTURE_BACKEND=hf

bash scripts/run_finetuning_phase1.sh \
  --config "$PWD/src/Finetuning/configs/qwen3_4b.yaml" \
  --train-input /path/to/prepared/train.jsonl \
  --eval-input /path/to/prepared/eval.jsonl \
  --target-model-path /path/to/Qwen3-4B \
  --output-root "$PWD/outputs/finetuning/qwen3_4b_phase1" \
  --run-id qwen3_4b_phase1 \
  --nproc-per-node 2 \
  --adaptive-max-batch-size 128 \
  --max-tokens-per-batch 131072 \
  --target-memory-fraction 0.90 \
  --max-anomaly-rate 0.05
```

Backend HF là đường mặc định: mỗi worker nạp target trên một GPU, chia mẫu theo
rank với `torchrun`, bucket theo độ dài, tự chọn batch theo VRAM và giảm batch khi
gặp OOM. Có thể hạ `--adaptive-max-batch-size` hoặc
`--max-tokens-per-batch` nếu model/context thực tế cần ít VRAM hơn. Không tăng
batch cố định để xử lý OOM.

Trước khi chạy 50k, tạo input pilot nhỏ riêng (ví dụ 100–500 dòng) và dùng một
`--output-root` riêng. Kiểm tra reports, số mẫu generate/cache, thời gian, VRAM
và prompt/token budget; sau khi pilot đạt yêu cầu mới chạy input đầy đủ. Dry-run
chỉ kiểm tra interpreter, config, model snapshot và kế hoạch stage; nó không load
model lên GPU:

```bash
bash scripts/run_finetuning_phase1.sh \
  --config "$PWD/src/Finetuning/configs/qwen3_4b.yaml" \
  --train-input /path/to/pilot/train.jsonl \
  --eval-input /path/to/pilot/eval.jsonl \
  --target-model-path /path/to/Qwen3-4B \
  --output-root "$PWD/outputs/finetuning/qwen3_4b_phase1_dryrun" \
  --nproc-per-node 2 --dry-run
```

## Các stage và quality gate

Launcher luôn chạy stage theo thứ tự:

1. `generate_train`, `generate_eval`: target greedy-generate output. Output giữ
   target ở `summary`; reference đầu vào được ghi ở `reference_summary` và có thể
   rỗng nếu nguồn không có gold.
2. `validate_teacher_train`, `validate_teacher_eval`: ghi ROUGE-1/2/L và audit
   summary rỗng/quá ngắn, tỷ lệ summary/document, lặp n-gram, mojibake và ROUGE
   bằng 0 khi có reference. Nếu anomaly rate hoặc ROUGE không qua ngưỡng, job dừng
   trước cache.
3. `cache_train`, `cache_eval`: render lại đúng prompt contract, lấy hidden states
   của các target layers đã cấu hình và xuất cache cùng manifest.

`--max-anomaly-rate` mặc định là `0.05`, `--min-teacher-rouge1` mặc định `0.0`.
Có thể bật `--filter-anomalies`; khi đó validator ghi file `.filtered.jsonl`
riêng và cache dùng file đó, còn teacher JSONL gốc vẫn được giữ để audit. Quality
gate vẫn áp dụng trên toàn bộ output gốc.

Không tắt validation, không chạy ở chế độ warn-only và không chọn riêng stage:
launcher sẽ từ chối `--skip-validation`, `--validate-warn-only` và `--stages` để
không tạo cache chưa qua gate.

## Output, resume và kiểm tra

Với output root trong ví dụ, artifacts chính là:

```text
outputs/finetuning/qwen3_4b_phase1/
├── teacher/train.jsonl
├── teacher/eval.jsonl
├── teacher/train_validation_report.json
├── teacher/eval_validation_report.json
├── features/train/manifest.json
├── features/eval/manifest.json
├── run_config.yaml
├── run_manifest.json
├── logs/
└── .state/
```

`run_config.yaml` trỏ đến hai feature stores và có thể dùng làm config cho bước
train riêng sau khi review Phase 1. Theo dõi tiến độ và log đầy đủ bằng:

```bash
tail -f outputs/finetuning/qwen3_4b_phase1/logs/run.log
```

Chạy lại cùng output root để resume stage đã hoàn tất. Launcher ghi hash input,
config và contract vào `run_manifest.json`; nếu thay dữ liệu, config, backend,
ngưỡng validation hoặc kiểu capture, hãy dùng output root mới. Worker count và
giới hạn adaptive VRAM có thể điều chỉnh khi tiếp tục cùng contract.
