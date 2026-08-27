# FastKV

Joint prefill + KV optimization (Llama/Mistral). Dựa trên `externals/FastKV`.

## Env & cài đặt

- Env: **venv chung** tại `.venv` (Python 3.12, dependency từ `requirements.txt`).
- `bash scripts/setup_venv.sh --offline`
- Method **fastkv thật** cần wheel `flash-attn` tương thích CUDA/GPU có sẵn trong
  wheelhouse local; smoke dùng SDPA và không cần flash-attn.

## Model

| Vai trò | Model | Ghi chú |
|---|---|---|
| Target | `mistralai/Mistral-7B-Instruct-v0.3` (mở) hoặc `meta-llama/Meta-Llama-3.1-8B-Instruct` (gated → HF_TOKEN) | chỉ LLaMA/Mistral |

## Chạy smoke / thật

```bash
bash scripts/run.sh fastkv        # mặc định smoke: snapkv + sdpa (không cần flash-attn)
```

Cấu hình trong master: `MODEL_TARGET`, `FASTKV_METHOD`
(fastkv|snapkv|h2o|streamingllm|fullkv), `FASTKV_ATTN_IMPL`
(flash_attention_2|sdpa|eager), `FASTKV_WINDOW_SIZE`, `FASTKV_RETAIN_RATE`,
`FASTKV_EVICTION_MODE`, `FASTKV_NUM_RUNS`, `RUN_MODE`.

- **Full (GPU lớn)**: `SMOKE=0 METHOD=fastkv ATTN_IMPL=flash_attention_2`.

## Dữ liệu của bạn

```bash
DATA_INPUT="data/user_prompts.jsonl" RUN_SAMPLES=5 bash scripts/run.sh fastkv
```

Dữ liệu full của FastKV (`LongBench` và `RULER`) được lưu ngoài repository tại
`${FI_HF_HOME:-$HOME/.cache/huggingface}/datasets/fast_infer_text_sum/FastKV/data`.
Có thể đổi vị trí bằng biến `DATA_ROOT`; các evaluator upstream sẽ tự
dùng biến này khi không truyền `--data_file`.

## Output

`outputs/fastkv_smoke.jsonl` — per record: input/output tokens, e2e, throughput,
`kernel_engaged` (báo method fastkv có thực sự dùng kernel hay không), text.

## Troubleshooting

- `kernel_engaged=false` trong smoke là đúng (smoke dùng snapkv+sdpa vì T4 không
  có flash-attn); muốn fastkv thật phải chạy full trên GPU sm80+.
- T4 16GB: Mistral-7B fp16 ~14GB — dùng 4-bit (bitsandbytes đã có trong env) nếu cần.
