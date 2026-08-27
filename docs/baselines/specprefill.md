# speculative_prefill

Token-selective prefill (monkey-patch vLLM). Dựa trên `externals/speculative_prefill`.

## Env & cài đặt

- Env: **venv chung** tại `.venv` (Python 3.12, dependency từ `requirements.txt`).
- `bash scripts/setup_venv.sh --offline`

## Model (chỉ Llama family)

| Vai trò | Model | Ghi chú |
|---|---|---|
| Target | `meta-llama/Llama-3.2-3B-Instruct` (smoke T4) / `Meta-Llama-3.1-8B-Instruct` (full) | gated → HF_TOKEN |
| Spec (draft) | `meta-llama/Llama-3.2-1B-Instruct` | |

## Chạy smoke / thật

```bash
bash scripts/run.sh specprefill
```

Cấu hình `config/specprefill.env`: `TARGET_MODEL`, `SPEC_MODEL`, `SPEC_CONFIG`,
`MAX_TOKENS`, `GPU_MEMORY_UTILIZATION`.

Lưu ý: monkey-patch phải được gọi **trước khi import vLLM** (wrapper đã xử lý);
vLLM bắt buộc `enforce_eager=True` + `enable_chunked_prefill=False`.

## Dữ liệu của bạn

```bash
DATA_FILE="data/user_prompts.jsonl" MAX_SAMPLES=5 bash scripts/run.sh specprefill
```
(chạy batch qua vLLM — `batch_size` ghi = số prompt.)

## Output

`outputs/specprefill_smoke.jsonl` — per prompt: tokens, e2e, throughput, text.

## Troubleshooting

- 8B target + 1B draft ~17GB fp16 → smoke dùng 3B target trên T4 16GB.
- vLLM cần CUDA; không có CPU path.
