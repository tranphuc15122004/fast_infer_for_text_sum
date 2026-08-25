# FlexPrefill

Dynamic, context-aware sparse attention cho long-sequence inference (ICLR 2025
Oral, ByteDance). Dựa trên `externals/FlexPrefill` (vendored).

## Env & cài đặt

- Env: **`envs/flexprefill`** (torch 2.4.0 cu124, triton 3.0.0, transformers 4.44.0).
- `uv sync --project envs/flexprefill --locked`
- Không cần flash-attn: script gọi `disable_hf_flash_attention_check()`
  (monkey-patch HF để `_attn_implementation="flash_attention_2"` không cần
  kernel thật — patch_model thay attention bằng kernel triton của FlexPrefill).
- Cần CUDA (triton JIT kernels), batch size 1, bf16.

## Model

Patch model transformers arch **qwen2 / llama / glm** (modules tương ứng).

- **Full mode** (runner): `Llama-3.1-8B-Instruct` (canonical, cùng target với các
  baseline khác) — override qua `REP_FLEXPREFILL_FULL_MODEL`. Cần GPU lớn
  (8B bf16 ≈ 16GB + context).
- **Smoke mode** (T4): `Qwen/Qwen2.5-3B-Instruct` (đã cache local, ~6GB bf16) —
  override qua `REP_FLEXPREFILL_SMOKE_MODEL`.

Pattern: `flex_prefill` (mặc định), ngoài ra có `streaming_llm`,
`vertical_slash`, `minfer`, `default`, `flash`.

## Chạy smoke / thật

```bash
bash scripts/run.sh flexprefill
```

Cấu hình `config/flexprefill.env`: `MODEL`, `PATTERN`, `MAX_NEW_TOKENS`,
`DATA_FILE`, `MAX_SAMPLES`, `MAX_INPUT_TOKENS`, `SKIP_NAIVE`, `SMOKE`.

## Dữ liệu của bạn

```bash
DATA_FILE="data/user_prompts.jsonl" MAX_SAMPLES=5 bash scripts/run.sh flexprefill
```

Script chạy **dense baseline trước khi patch** (paired speedup) rồi patch model
và chạy method; mỗi record ghi `e2e_ms`, `dense_e2e_ms`, `text`, ROUGE.

## Output

`outputs/flexprefill_smoke.jsonl` — schema §13 + summary.

## Troubleshooting

- `patch_model` yêu cầu model arch nằm trong modules (qwen2/llama/glm); model
  khác sẽ không được patch.
- Trên T4, doc dài (govreport 40k+ token) cần `MAX_INPUT_TOKENS` (runner smoke
  tự đặt 4096) để khỏi OOM.
- Kernel triton JIT compile lần đầu chậm; warmup đã được tích hợp trong script.
