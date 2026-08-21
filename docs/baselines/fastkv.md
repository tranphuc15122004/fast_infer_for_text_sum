# FastKV

Joint prefill + KV optimization (Llama/Mistral). Dựa trên `externals/FastKV`.

## Env & cài đặt

- Env: **`envs/legacy`** (dùng chung với RocketKV/GemFilter/SpecExtend/HiGOE: transformers 4.45.2, torch 2.4.1+cu124).
- `uv sync --project envs/legacy --locked`
- Method **fastkv thật** cần flash-attn: `EXTRA_FLASH=1 bash scripts/setup_envs.sh`
  (chỉ khi GPU ≥ sm80; T4/sm75 phải build từ source).

## Model

| Vai trò | Model | Ghi chú |
|---|---|---|
| Target | `mistralai/Mistral-7B-Instruct-v0.3` (mở) hoặc `meta-llama/Meta-Llama-3.1-8B-Instruct` (gated → HF_TOKEN) | chỉ LLaMA/Mistral |

## Chạy smoke / thật

```bash
bash scripts/run.sh fastkv        # mặc định smoke: snapkv + sdpa (không cần flash-attn)
```

Cấu hình `config/fastkv.env`: `MODEL`, `METHOD` (fastkv|snapkv|h2o|streamingllm|fullkv),
`ATTN_IMPL` (flash_attention_2|sdpa|eager), `WINDOW_SIZE`, `RETAIN_RATE`,
`EVICTION_MODE`, `NUM_RUNS`, `SMOKE`.

- **Full (GPU lớn)**: `SMOKE=0 METHOD=fastkv ATTN_IMPL=flash_attention_2`.

## Dữ liệu của bạn

```bash
DATA_FILE="data/user_prompts.jsonl" MAX_SAMPLES=5 bash scripts/run.sh fastkv
```

Dữ liệu full của FastKV (`LongBench` và `RULER`) được lưu ngoài repository tại
`${HF_HOME:-$HOME/.cache/huggingface}/datasets/fast_infer_text_sum/FastKV/data`.
Có thể đổi vị trí bằng biến `FASTKV_DATA_ROOT`; các evaluator upstream sẽ tự
dùng biến này khi không truyền `--data_file`.

## Output

`outputs/fastkv_smoke.jsonl` — per record: input/output tokens, e2e, throughput,
`kernel_engaged` (báo method fastkv có thực sự dùng kernel hay không), text.

## Troubleshooting

- `kernel_engaged=false` trong smoke là đúng (smoke dùng snapkv+sdpa vì T4 không
  có flash-attn); muốn fastkv thật phải chạy full trên GPU sm80+.
- T4 16GB: Mistral-7B fp16 ~14GB — dùng 4-bit (bitsandbytes đã có trong env) nếu cần.
