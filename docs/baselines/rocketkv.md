# RocketKV

KV-cache compression / sparse decode attention (NVlabs). Dựa trên `externals/RocketKV`.

## Env & cài đặt

- Env: **`envs/legacy`** (dùng chung với FastKV/GemFilter/SpecExtend/HiGOE: transformers 4.45.2, torch 2.4.1+cu124).
- `uv sync --project envs/legacy --locked`

## Chạy smoke (kernel, không cần model)

```bash
bash scripts/run.sh rocketkv
```

Verify: `RocketArgs` + `get_params_for_token_budget` sinh budget hợp lệ, và
`RocketAttention` chạy dummy prefill+decode (finite). Không cần GPU lớn / model.

## Full (LongBench pipeline) — cần GPU lớn

Upstream:
```bash
cd externals/RocketKV
export HF_TOKEN=...
bash scripts/longbench/llama3.1-8b-instruct.sh rocket <results_dir> <token_budget>
```
(chạy `python pipeline/inf_stream_llm/main.py --method rocket ...` với config JSON
trong `config/pipeline_config/longbench/`; model Llama-3.1-8B gated cần HF_TOKEN,
`config/access_tokens.py` phải điền token).

## Output

Smoke: `outputs/rocketkv_smoke.jsonl` (cap/r/k + finite từng run).

## Troubleshooting

- Triton (torch.compile) cần `setuptools` — đã có trong env.
- Kernel test dùng **boolean causal mask** cho decode (`torch.where(mask, ...)`).
- Script phải thấy module `rocket` → wrapper đã thêm `externals/RocketKV/gpt-fast` vào PYTHONPATH.
