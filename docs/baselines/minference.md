# MInference

Dynamic sparse attention cho prefill (long context). Dựa trên `externals/MInference`.

## Env & cài đặt

- Env: **venv chung** tại `.venv` (Python 3.12, dependency từ `requirements.txt`).
- `bash scripts/setup_venv.sh --offline`
- Cần CUDA (triton kernels); flash-attn khuyến nghị cho model 7B+.

## Model

Bắt buộc nằm trong danh sách hỗ trợ (`minference.get_support_models()`), ví dụ:
`Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Meta-Llama-3.1-8B-Instruct`,
`microsoft/Phi-3-mini-128k-instruct`, GLM-4-9B-1M...

⚠️ Model ngoài danh sách (vd Qwen2.5-3B) sẽ thiếu sparse-pattern config.

## Chạy smoke / thật

```bash
bash scripts/run.sh minference
```

Cấu hình `config/minference.env`: `MODEL`, `ATTN_TYPE`, `MAX_NEW_TOKENS`,
`MAX_MODEL_LEN`, `DEVICE`, `ATTN_IMPLEMENTATION`. `auto` dùng
FlashAttention-2 trên GPU sm80+ nếu đã cài; trên T4 tự dùng SDPA.

## Dữ liệu của bạn

```bash
DATA_FILE="data/user_prompts.jsonl" MAX_SAMPLES=5 bash scripts/run.sh minference
```

## Output

`outputs/minference_smoke.jsonl` — per record: tokens, e2e, throughput, text.

## Troubleshooting

- Phương pháp tối ưu cho **prefill dài** (64K–1M); smoke ngắn chỉ xác nhận
  patch + kernel chạy đúng.
- MInference build CUDA extension từ source cần nvcc; ưu tiên dùng prebuilt wheel
  hoặc `MINFERENCE_SKIP_CUDA_BUILD=TRUE` nếu chỉ chạy triton path.
- 7B fp16 ~14GB — context ngắn mới vừa T4 16GB.
