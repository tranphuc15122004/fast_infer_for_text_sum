# GemFilter

Early-layer semantic token filtering (self-filtering bằng chính target model).
Dựa trên `externals/GemFilter`.

## Env & cài đặt

- Env: **`envs/legacy`** (dùng chung với FastKV/RocketKV/SpecExtend/HiGOE: transformers 4.45.2 — pin gốc của GemFilter là 4.43.3, các API dùng vẫn tồn tại ở 4.45).
- `uv sync --project envs/legacy --locked`
- Method gemfilter thật (flash_attention_2 path) cần flash-attn trên GPU sm80+:
  `EXTRA_FLASH=1 bash scripts/setup_envs.sh`

## Model (chỉ các model được hỗ trợ)

| Model | VRAM | Ghi chú |
|---|---|---|
| `microsoft/Phi-3.5-mini-instruct` | ~8GB | **smoke T4 khuyên dùng** |
| `mistralai/Mistral-Nemo-Instruct-2407` | ~14GB | |
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | ~16GB | gated → HF_TOKEN |

## Chạy smoke / thật

```bash
bash scripts/run.sh gemfilter      # smoke: Phi-3.5-mini + eager (không cần flash-attn)
```

Cấu hình `config/gemfilter.env`: `MODEL`, `TOPK`, `SELECT_LAYER_IDX`
(13 = Llama-3.1-8B; 19 = Nemo/Phi-3.5), `MAX_GEN_LEN`, `NUM_RUNS`.

## Dữ liệu của bạn

```bash
DATA_FILE="data/user_prompts.jsonl" MAX_SAMPLES=5 bash scripts/run.sh gemfilter
```

## Output

`outputs/gemfilter_smoke.jsonl` — per record: gemfilter vs baseline greedy text +
time, e2e.

## Troubleshooting

- Script chạy **eager attention** (không flash-attn) vì GemFilter patch cả class
  eager; chạy được T4 với Phi-3.5-mini.
- Không phải pip package — chạy từ PYTHONPATH `externals/GemFilter`.
