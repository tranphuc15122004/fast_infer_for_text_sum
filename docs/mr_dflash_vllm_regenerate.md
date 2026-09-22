# MR-DFlash regenerate bằng vLLM 0.24.0

Worker [`vllm_regenerate.py`](../scripts/mr_dflash/vllm_regenerate.py) giữ
tokenizer/chat-template local, sau đó gửi `prompt` dạng token IDs tới endpoint
OpenAI-compatible `/v1/completions`. vLLM thực hiện continuous batching; worker
admission-control theo số request, tổng `prompt_tokens + generation_budget` và
peak `gpu_cache_usage_perc` lấy từ Prometheus `/metrics`.

## Mô hình triển khai

Chạy một vLLM server trên mỗi GPU độc lập. Không dùng chung GPU giữa vLLM và
worker cache SGLang; server farm phải được dừng trước khi bắt đầu phase hidden
cache nếu hai phase dùng cùng GPU.

Ví dụ với bốn GPU 180 GiB và model fit trên một GPU:

```bash
MODEL=/workspace/storage-shared/models/Qwen3-4B

CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
  --served-model-name qwen3-4b \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 128 --max-num-batched-tokens 262144

CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL" \
  --served-model-name qwen3-4b \
  --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 --max-model-len 32768 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 128 --max-num-batched-tokens 262144
```

Lặp lại cấu hình này cho GPU/port còn lại. Sau khi server khởi động, kiểm tra
endpoint `/metrics`; nếu phiên bản vLLM không expose metric KV-cache, worker
vẫn chạy bằng token-aware admission. Các server phải dùng cùng model revision,
tokenizer và `max-model-len`; `--served-model-name` phải giống
`--vllm-model`.

## Chạy regenerate

Sau khi các endpoint health check thành công, chạy pipeline với backend vLLM:

```bash
PYTHONPATH=src python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  --target-model-path "$MODEL" \
  --data-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_full \
  --full-context --full-context-length 32768 --max-new-tokens 1024 \
  --regenerate-backend vllm \
  --vllm-server-addresses \
    http://127.0.0.1:8000/v1 http://127.0.0.1:8001/v1 \
    http://127.0.0.1:8002/v1 http://127.0.0.1:8003/v1 \
  --vllm-model qwen3-4b \
  --vllm-request-concurrency 64 \
  --vllm-request-concurrency-start 8 \
  --vllm-max-batched-tokens-start 65536 \
  --vllm-max-batched-tokens 262144 \
  --vllm-gpu-cache-target 0.90 \
  --vllm-gpu-cache-hard 0.98 \
  --parallel-gpu-ids 0 1 2 3 \
  --parallel-scheduler shared_lease \
  --parallel-queue-quantum-items 512 \
  --overflow-policy skip --sample-error-policy error --resume
```

`addresses[i]` được gán cho worker rank/GPU thứ `i`. `shared_lease` phát việc
theo quantum hữu hạn để GPU hoàn thành shard ngắn không phải chờ một shard dài
độc quyền; output merge vẫn theo thứ tự input và kiểm tra missing/duplicate ID.

`--vllm-request-concurrency-start` và
`--vllm-max-batched-tokens-start` tăng dần sau các nhóm thành công khi KV-cache
còn dưới target. Mặc định worker giữ target ở 90% và hard backoff ở 98%:

```text
cache_usage < 0.90  → tăng concurrency và token budget
0.90 ≤ cache_usage < 0.98 → giữ mức hiện tại
cache_usage ≥ 0.98 hoặc 429/5xx → giảm một nửa
```

`--vllm-max-batched-tokens` là giới hạn admission phía client, không thay thế
`--max-num-batched-tokens` của server. Nếu server không expose `/metrics`, worker
tự fallback sang token-aware admission, không dừng generation. Có thể truyền
`--vllm-metrics-address` nếu endpoint metrics không nằm ở `/metrics` mặc định.

Khi gặp timeout/429/5xx, worker retry có backoff và giảm concurrency/token
budget; lỗi không retryable tuân theo `--sample-error-policy`.

Với `temperature=0`, worker có thể chạy concurrent requests. Với
`temperature>0`, worker tự hạ concurrency về 1 để giữ contract sampling gần
với đường legacy; vì vậy throughput tối ưu nên dùng generation deterministic
đã được pipeline khóa.

## Chuyển sang cache SGLang

Sau khi regenerate, dừng toàn bộ vLLM server farm nếu cache dùng cùng GPU. Giữ
nguyên các tham số vLLM khi resume để config hash/manifest khớp, nhưng các stage
regenerate đã hoàn tất sẽ được reuse. Chạy tiếp từ validate/tokenize/cache với:

```bash
python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  ...cùng tham số regenerate và data-root... \
  --from-stage validate_full_train \
  --cache-backend specforge_sglang \
  --cache-auto-batch \
  --cache-auto-batch-target-vram-gb 160 \
  --cache-auto-batch-hard-vram-gb 163 \
  --cache-concurrency 64 --cache-max-total-tokens 262144 \
  --cache-memory-fraction 0.99 --resume
```

Đường cache yêu cầu SGLang 0.5.14 và patch/runtime SpecForge tương thích; xem
[`mr_dflash_cache_auto_batch.md`](mr_dflash_cache_auto_batch.md). Không đưa
source SGLang vendored khác vào `PYTHONPATH` để tránh shadow wheel trên server.

## Correctness và đo tốc độ

Trước full run, chạy 20–100 sample và kiểm tra:

- regeneration có cùng tokenizer, revision, max length/budget, seed và schema;
- `validation_*.json` không có missing/duplicate sample;
- cache manifest giữ đúng layer IDs, feature width/dtype và coverage;
- ghi `samples/s`, `tokens/s`, peak VRAM, queue wait, request retry và OOM.

Máy local CPU chỉ kiểm tra protocol/plan/unit tests; chưa có số liệu speedup
vLLM/SGLang nếu chưa chạy smoke trên server GPU thật.
