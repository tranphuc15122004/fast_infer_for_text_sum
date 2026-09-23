# MR-DFlash Phase 1 trên một GPU B200

Tài liệu này hướng dẫn chạy Phase 1 với Qwen3-4B bằng vLLM để regenerate và
HF Transformers để tạo target-feature cache.

Launcher chính:

```text
scripts/mr_dflash/run_b200_vllm_phase1.sh
```

Launcher phải được chạy bằng `bash`, không dùng `source` hoặc dấu `.`.

## Phạm vi và mặc định

Launcher hiện dùng các giá trị sau:

```text
GPU                 0
vLLM port           38147
target model        /workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B
prepared data       /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_20k_sharegpt_30k_arxiv
output root         /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_20k_sharegpt_30k_arxiv_vllm_b200
context length      32768
max new tokens      2048
source count        20K ShareGPT + 30K Arxiv
feature layers      1 9 17 25 33
```

Phase 1 reuse các file prepare hiện có trong `<prepared-root>/normalized/` và
không chạy lại stage `prepare`. Ba file bắt buộc là:

```text
normalized/train_prompts.jsonl
normalized/val_prompts.jsonl
normalized/test_prompts.jsonl
```

## Quét phân phối raw dataset trước khi build

Trước khi quyết định số lượng, split hoặc rule lọc, chạy analyzer độc lập. Nó
duyệt tuần tự toàn bộ raw ShareGPT rồi ArXiv, không random-sample, không gọi
vLLM và không tạo `train/val/test`:

```bash
OUTPUT_ROOT=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_raw_analysis
mkdir -p "$OUTPUT_ROOT"
set -o pipefail
python3 scripts/mr_dflash/analyze_phase1_dataset.py \
  --sharegpt-source /workspace/storage-shared/nlp/tungdd11/tungdecoder/ShareGPT/ShareGPT_V3_unfiltered_cleaned_split.json \
  --arxiv-source /workspace/storage-shared/nlp/dungdx4/datasets/arxiv/train.label.jsonl \
  --output-root "$OUTPUT_ROOT" \
  --tokenizer /workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B \
  --progress-interval-records 500 \
  --progress-interval-seconds 15 \
  --local-files-only \
  2>&1 | tee "$OUTPUT_ROOT/analysis.log"
```

Analyzer hiển thị progress bar riêng cho từng nguồn. Vì không đếm trước tổng
record, thanh báo số đã quét, thời gian và tốc độ thay vì phần trăm/ETA. Log
`START`, `PROGRESS`, `DONE` hiện trực tiếp và được lưu ở `analysis.log`.
Giảm hai interval để log dày hơn. `Ctrl+C` dừng tiến trình; chạy lại sẽ quét
từ đầu và ghi đè `records.jsonl` cùng `issues.jsonl` trong output root.

Kết quả gồm `summary.json`, `records.jsonl`, `issues.jsonl` và bốn hình PNG
trong `figures/`. `--max-records N` chỉ dành cho smoke test; bỏ option này khi
quét thật. Sau khi duyệt report, chạy prepare/build riêng theo rule đã chọn,
rồi truyền `--prepared-root` của artifact đó cho launcher Phase 1.

## Hẹn giờ tạm dừng an toàn

Mode `2` hỗ trợ dừng mềm sau một số phút. Khi đến hạn, launcher gửi stop
signal cho pipeline, dừng trước khi chuyển sang hidden-cache, rồi shutdown
vLLM sạch. Artifact đã ghi vẫn dùng được với `--resume`.

Mặc định không giới hạn thời gian:

```bash
PHASE1_AUTO_PAUSE_MINUTES=0 \
  bash scripts/mr_dflash/run_b200_vllm_phase1.sh 2 \
  --prepared-root /path/to/prepared_dataset
```

Ví dụ dừng sau 4 giờ:

```bash
bash scripts/mr_dflash/run_b200_vllm_phase1.sh 2 \
  --prepared-root /path/to/prepared_dataset \
  --auto-pause-minutes 240
```

Chạy lại đúng lệnh sau khi server rảnh; launcher tự xóa stop signal cũ và
pipeline tiếp tục từ artifact đã hoàn thành. Không dùng `SIGSTOP` hoặc kill
cứng vì process có thể tiếp tục giữ VRAM B200.

## Chuẩn bị môi trường

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main
source /workspace/storage-shared/nlp/dungdx4/phuc_projects/phucvenv/bin/activate
```

Kiểm tra model và dữ liệu prepare:

```bash
test -d /workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B
test -s /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_20k_sharegpt_30k_arxiv/normalized/train_prompts.jsonl
test -s /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_20k_sharegpt_30k_arxiv/normalized/val_prompts.jsonl
test -s /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_20k_sharegpt_30k_arxiv/normalized/test_prompts.jsonl
```

## Cách 1: chạy toàn bộ Phase 1 bằng một terminal

Đây là cách đơn giản nhất:

```bash
bash scripts/mr_dflash/run_b200_vllm_phase1.sh 2
```

Mode `2` thực hiện theo thứ tự:

```text
start/reuse vLLM
  -> regenerate_full_train
  -> regenerate_full_val
  -> regenerate_full_test
  -> stop vLLM
  -> validate_full_{train,val,test}
  -> tokenize_full_{train,val,test}
  -> cache_full_train
  -> cache_full_val
```

vLLM được dừng trước khi cache vì cache HF cần dùng GPU để load target model.
Nếu job bị gián đoạn, chạy lại đúng lệnh trên; pipeline dùng `--resume` và giữ
artifact đã hoàn thành.

## Cách 2: tách thành hai terminal

Terminal 1 chỉ khởi động vLLM:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main
source /workspace/storage-shared/nlp/dungdx4/phuc_projects/phucvenv/bin/activate
bash scripts/mr_dflash/run_b200_vllm_phase1.sh 1
```

Khi thấy:

```text
[b200-vllm] READY pid=... address=http://127.0.0.1:38147/v1
[b200-vllm] mode 1 completed; vLLM remains running
```

Terminal 2 chạy Phase 1:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main
source /workspace/storage-shared/nlp/dungdx4/phuc_projects/phucvenv/bin/activate
bash scripts/mr_dflash/run_b200_vllm_phase1.sh 2
```

Mode `2` nhận diện PID file và reuse server đang chạy; không load model lần hai.

## Kiểm tra vLLM và theo dõi log

Health check:

```bash
curl -sS http://127.0.0.1:38147/health
curl -sS http://127.0.0.1:38147/v1/models
```

Log chi tiết của vLLM:

```bash
tail -f /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_20k_sharegpt_30k_arxiv_vllm_b200/logs/vllm.log
```

PID file:

```text
/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_20k_sharegpt_30k_arxiv_vllm_b200/logs/vllm.pid
```

Log của pipeline:

```text
/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_20k_sharegpt_30k_arxiv_vllm_b200/pipeline_logs/
```

Trạng thái từng stage:

```text
/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_20k_sharegpt_30k_arxiv_vllm_b200/pipeline_state/
```

## Kết quả chính

```text
regenerated_full/train.jsonl
regenerated_full/val.jsonl
regenerated_full/test.jsonl
tokenized_full/{train,val,test}/
target_features_qwen3_4b_full/train/
target_features_qwen3_4b_full/val/
manifests/
pipeline_plan.json
pipeline_summary.json
```

Tất cả nằm dưới:

```text
/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_20k_sharegpt_30k_arxiv_vllm_b200/
```

## Chạy một output root mới

Nếu muốn tách khỏi run cũ, giữ nguyên prepared root nhưng đổi `--run-root`:

```bash
bash scripts/mr_dflash/run_b200_vllm_phase1.sh 2 \
  --run-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_20k_sharegpt_30k_arxiv_vllm_b200_run2
```

Có thể đổi model, prepared root, GPU hoặc port:

```bash
bash scripts/mr_dflash/run_b200_vllm_phase1.sh 2 \
  --target-model /workspace/storage-shared/nlp/dungdx4/BERT/Qwen3-4B \
  --prepared-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_20k_sharegpt_30k_arxiv \
  --gpu-id 0 \
  --port 38147
```

## Dừng vLLM mode 1 thủ công

Chỉ dùng PID lấy từ đúng PID file của run:

```bash
PID_FILE=/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_20k_sharegpt_30k_arxiv_vllm_b200/logs/vllm.pid
kill -TERM "$(cat "$PID_FILE")"
```

Không dùng `killall python`, `killall vllm` hoặc kill toàn bộ GPU vì có thể ảnh
hưởng job khác trên server.

## Lỗi thường gặp

### Sourcing làm thoát terminal

Không chạy:

```bash
. scripts/mr_dflash/run_b200_vllm_phase1.sh 1
```

Phải chạy:

```bash
bash scripts/mr_dflash/run_b200_vllm_phase1.sh 1
```

### Port đã được dùng

Kiểm tra server hiện tại:

```bash
curl -sS http://127.0.0.1:38147/health
```

Nếu đó là server của run khác, dùng port mới và truyền cùng port cho mode `2`.

### Mode 2 báo không có PID file

Điều này nghĩa là port đang có một vLLM bên ngoài launcher. Dừng đúng process
đó hoặc chọn port mới; không để launcher tự dừng một process không xác định.

### Thiếu file prepare

Kiểm tra lại `--prepared-root` và ba file `normalized/*_prompts.jsonl`. Launcher
không tự tạo lại prepare trong Phase 1.
