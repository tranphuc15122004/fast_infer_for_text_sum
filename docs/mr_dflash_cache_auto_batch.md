# Throughput profile cho phase cache MR-DFlash

## Lưu ý trước khi chạy smoke

Backend SpecForge/SGLang trong tài liệu này là tùy chọn tối ưu throughput và
cần thêm bộ kernel CUDA tương ứng. Smoke test portable trên B200 mặc định
dùng `--cache-backend hf --cache-attention-backend sdpa`; đường này chỉ dùng
PyTorch + Transformers. Phase 1 smoke có thể chạy đúng backend tăng tốc bằng
`--cache-backend specforge_sglang`; với runner Phase 1, `sdpa`/`auto` được map
sang `flashinfer`. Không cần cài `flash-attn`, `sglang-kernel` hay `specforge`
cho đường HF.

## Khuyến nghị cho B200 180 GiB

Cache nhanh dùng backend offline SGLang theo cơ chế SpecForge. Với dữ liệu
thực tế chủ yếu dưới 8K, worker dùng token-budget profile làm giới hạn padding
ban đầu, còn batch thực tế được adaptive controller điều chỉnh:

| Độ dài padded | Batch khởi đầu khi adaptive | Token budget hint |
|---:|---:|---:|
| 1–32768 | 1 (cấu hình được) | theo `max_total_tokens`/static-pool |

Chạy cache trên hai GPU độc lập:

```bash
PYTHONPATH=src python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --data-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot_full \
  --full-context --full-context-length 32768 --max-new-tokens 2048 \
  --cache-backend specforge_sglang \
  --cache-attention-backend flashinfer \
  --cache-auto-batch --cache-auto-batch-start-size 1 \
  --cache-auto-batch-safety-fraction 0.95 \
  --cache-auto-batch-target-vram-gb 160 \
  --cache-auto-batch-hard-vram-gb 163 \
  --cache-auto-batch-max-size 128 \
  --cache-concurrency 64 --cache-max-total-tokens 262144 \
  --cache-memory-fraction 0.99 --cache-startup-stagger-seconds 3 \
  --parallel-gpu-ids 0 1 --parallel-scheduler shared_lease \
  --cache-splits train val --resume
```

Khi bật `--cache-auto-batch`, worker bắt đầu từ
`--cache-auto-batch-start-size` (mặc định `1`), không lấy batch `64/32/16/4`
trong profile làm batch cố định. Sau mỗi capture thành công, batch tăng theo
từng bucket length tới `--cache-auto-batch-max-size`. Trước mỗi forward,
worker lấy token capacity thực tế của SGLang static-pool, chừa 5% theo
`--cache-auto-batch-safety-fraction`, rồi kiểm tra cả batch hiện tại cộng thêm
một sample; nếu vượt giới hạn dự đoán thì giữ batch hiện tại. Nếu vẫn gặp OOM,
worker giữ shard CPU, giải phóng request/KV pool, backoff và tìm lại batch an
toàn giữa batch thành công và batch lỗi. Static-pool được giới hạn theo soft
target `160 GiB`, hard cap `163 GiB`; phần còn lại giữ làm headroom cho driver,
allocator và sample bất thường.
Nếu worker bị CUDA OOM hoặc bị hệ
điều hành kill với exit code `-9`, launcher ghi `retry_hint.json` trong
`parallel_cache_<split>/`; các shard đã durable vẫn giữ nguyên. Chạy lại cùng
lệnh với `--resume`, giảm `--cache-concurrency`/`--cache-max-total-tokens`
theo hint. Profile là performance-only nên đổi profile không làm mất cache
đã hoàn tất.

`shared_lease` lưu queue, lease và event log trong `work-root/shared_queue`.
Worker nhận sample theo length tăng dần; mặc định mỗi lease bị giới hạn bởi
`--parallel-queue-quantum-items 512` để GPU xong shard ngắn có thể lấy việc
tiếp theo. Khi process hoặc host bị dừng, lease hết hạn sẽ được host mới
reclaim. Mỗi sample được retry tối đa 3 lần; sample
vẫn lỗi ở batch 1 được ghi vào `quarantine.jsonl`. Merge cuối vẫn giữ thứ tự
canonical của input và không ghi trùng sample.

Trước mỗi forward, cache nhìn trước cả nhóm candidate và dùng padded length của
sample dài nhất trong nhóm, không chỉ dùng length của sample đầu tiên. Khi
length bucket kế tiếp dài hơn, batch được warm-start theo tỷ lệ token của batch
an toàn trước đó và giảm trước khi chạy; điều này tránh thử một batch lớn rồi
mới phát hiện OOM.

Parent hiển thị một tqdm tổng hợp trên cả hai GPU; `completed/total` tính theo
sample của toàn input, còn heartbeat chi tiết từng worker nằm trong
`parallel_cache_<split>/rank_*/progress.json`.

## Mục đích

Với backend `specforge_sglang`, `--cache-auto-batch` là adaptive runtime mode:
`--cache-auto-batch-start-size` là điểm khởi đầu, còn token-budget profile chỉ
là performance hint/fallback và ghi provenance. Batch tăng dần theo từng
bucket length; OOM được coi là tín hiệu vượt ngưỡng và controller
backoff/binary-search vùng an toàn. Predictor token-capacity chặn trước batch
được dự đoán vượt safety threshold. Static pool được cấu hình theo mục tiêu
VRAM; không cần chạy
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

## Lệnh HF legacy cho 2 B200 180 GB

Với dữ liệu đã có `tokenized_full/train`, chạy từ stage profile. `170` là hard
limit mỗi GPU, chừa khoảng 10 GB cho driver/process khác:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main
export DATA_ROOT=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot_full
export TARGET_MODEL=/workspace/storage-shared/models/Qwen3-4B
rm -f "$DATA_ROOT/.stop_parallel"

CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src python3 \
  scripts/mr_dflash/run_preprocess_pipeline.py \
    --target-model-path "$TARGET_MODEL" \
    --data-root "$DATA_ROOT" \
    --device cuda \
    --local-files-only \
    --from-stage profile_cache_batch_full \
    --full-context \
    --full-context-length 32768 \
    --max-new-tokens 2048 \
    --parallel-gpu-ids 0 1 \
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
