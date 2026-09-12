# Throughput profile cho phase cache MR-DFlash

## Khuyến nghị cho 4×B200 180GB

Cache nhanh dùng backend offline SGLang theo cơ chế SpecForge. Với dữ liệu
thực tế chủ yếu dưới 8K, worker mặc định dùng profile aggressive:

| Độ dài padded | Batch/GPU | Token budget |
|---:|---:|---:|
| 1–4096 | 64 | 262144 |
| 4097–8192 | 32 | 262144 |
| 8193–16384 | 16 | 262144 |
| 16385–32768 | 4 | 131072 |

Chạy cache trên bốn GPU độc lập:

```bash
PYTHONPATH=src python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --data-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot_full \
  --full-context --full-context-length 32768 --max-new-tokens 2048 \
  --cache-backend specforge_sglang \
  --cache-attention-backend flashinfer \
  --cache-auto-batch --cache-auto-batch-target-vram-gb 170 \
  --cache-auto-batch-max-size 128 \
  --cache-concurrency 64 --cache-max-total-tokens 262144 \
  --cache-memory-fraction 0.99 --cache-startup-stagger-seconds 3 \
  --parallel-gpu-ids 0 1 2 3 --cache-splits train val --resume
```

Khi bật `--cache-auto-batch`, worker tự nâng static-pool lên gần `170/180`
(xấp xỉ `0.944`) dù lệnh vẫn giữ các giá trị compatibility `64` và `262144`;
batch cache tăng theo từng bucket length tới `--cache-auto-batch-max-size`.
Nếu worker bị CUDA OOM hoặc bị hệ
điều hành kill với exit code `-9`, launcher ghi `retry_hint.json` trong
`parallel_cache_<split>/`; các shard đã durable vẫn giữ nguyên. Chạy lại cùng
lệnh với `--resume`, giảm `--cache-concurrency`/`--cache-max-total-tokens`
theo hint. Profile là performance-only nên đổi profile không làm mất cache
đã hoàn tất.

Parent hiển thị một tqdm tổng hợp trên cả bốn GPU; `completed/total` tính theo
sample của toàn input, còn heartbeat chi tiết từng worker nằm trong
`parallel_cache_<split>/rank_*/progress.json`.

## Mục đích

Với backend `specforge_sglang`, `--cache-auto-batch` là adaptive runtime mode:
profile token-budget chỉ làm điểm khởi đầu, sau đó batch tăng dần theo từng
bucket length. Static pool được cấu hình theo mục tiêu VRAM; không cần chạy
stage profiler riêng. `--cache-auto-batch` vẫn là đường tương thích cho HF
backbone cũ và khi đó chạy một stage profile trước `cache_*`. Profiler dùng
tokenized train shard và target model thật trên một GPU, đo lần lượt các batch
candidate (mặc định `1,2,4,8`) tại cận trên của từng bucket độ dài. Bucket chỉ
được chọn batch nếu `peak_memory_reserved` không vượt hard limit VRAM.

Kết quả được ghi tại:

```text
<data-root>/manifests/cache_batch_profile_<regime>.json
```

Sau đó cache worker HF dùng schedule cố định trong toàn bộ run. Không có cơ chế
thử lại bằng batch khác sau khi đã bắt đầu ghi feature shard; nếu batch 1 không
an toàn, profile dừng trước cache. Cache manifest cũng ghi
`cache_batch_profile` để audit/resume không dùng nhầm schedule.

## Lệnh HF legacy cho 4 B200 180 GB

Với dữ liệu đã có `tokenized_full/train`, chạy từ stage profile. `170` là hard
limit mỗi GPU, chừa khoảng 10 GB cho driver/process khác:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main
export DATA_ROOT=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot_full
export TARGET_MODEL=/workspace/storage-shared/models/Qwen3-4B
rm -f "$DATA_ROOT/.stop_parallel"

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src python3 \
  scripts/mr_dflash/run_preprocess_pipeline.py \
    --target-model-path "$TARGET_MODEL" \
    --data-root "$DATA_ROOT" \
    --device cuda \
    --local-files-only \
    --from-stage profile_cache_batch_full \
    --full-context \
    --full-context-length 32768 \
    --max-new-tokens 2048 \
    --parallel-gpu-ids 0 1 2 3 \
    --cache-auto-batch \
    --cache-profile-gpu-id 0 \
    --cache-profile-max-batch-size 8 \
    --cache-profile-vram-limit-gb 170 \
    --cache-profile-bucket-step 8192 \
    --cache-attention-backend sdpa \
    --cache-bucket-buffer-long-context 8 \
    --cache-shard-size-long-context 32 \
    --cache-io-threads 2 \
    --cache-io-queue-size 4 \
    --cache-splits train val \
    --resume
```

`--cache-batch-size-long-context` vẫn được truyền mặc định là 1 như fallback
tương thích, nhưng khi profile hợp lệ thì batch thực tế lấy từ JSON schedule.
`bucket-buffer` là buffer RAM phía producer, không phải số token; cache sẽ tự
mở rộng buffer tới batch lớn nhất trong profile nếu cần.

Nếu chỉ định `--only-stage cache_full_train` hoặc `--from-stage
cache_full_train` cùng `--cache-auto-batch`, pipeline tự thêm
`profile_cache_batch_full` ngay trước cache. Không cần tự gõ tên profile.

## Theo dõi và kiểm tra

Trong lúc profile xem log stage:

```bash
tail -f "$DATA_ROOT/pipeline_logs/profile_cache_batch_full.log"
```

Trong lúc cache xem tổng hợp:

```bash
python3 scripts/mr_dflash/watch_parallel_stage.py \
  --status "$DATA_ROOT/target_features_qwen3_4b_full/parallel_cache_train/status.json" \
  --interval 5 --tqdm
```

Kiểm tra quyết định batch:

```bash
python3 - <<'PY'
import json, os
p = os.path.join(os.environ["DATA_ROOT"], "manifests", "cache_batch_profile_full.json")
payload = json.load(open(p))
for bucket in payload["buckets"]:
    print(bucket["min_length"], bucket["max_length"], "=>", bucket["selected_batch_size"])
PY
```

Muốn dùng profile đã đo cho các lần cache sau, bỏ `--cache-auto-batch` và
truyền trực tiếp:

```text
--cache-batch-profile <data-root>/manifests/cache_batch_profile_full.json
```

Profile phải khớp tuyệt đối target path, layer IDs, max length, dtype, backend
và target revision. Thay đổi một trong các giá trị này sẽ dừng trước capture;
hãy tạo profile mới.
