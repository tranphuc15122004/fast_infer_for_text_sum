# EAGLE-3

Learned speculative decoding (EAGLE-3, Qwen3) — lossless under target
verification. Dựa trên `externals/EAGLE`.

## Env & cài đặt

- Env: **venv chung** tại `.venv` (Python 3.12, dependency từ `requirements.txt`).
- `bash scripts/setup_venv.sh --offline`.

## Model

| Vai trò | Model | Ghi chú |
|---|---|---|
| Base | `Qwen/Qwen3-4B` | đã cache local, snapshot path trong `config/eagle3_qwen3.env` |
| Draft | `AngelSlim/Qwen3-4B_eagle3` | path trong config |

## Chạy smoke / thật

```bash
bash scripts/run.sh eagle3
```

Cấu hình `config/eagle3_qwen3.env`:
- `BENCH_NAME` = `qa` | `sum` | `gsm8k` | ... (dataset kèm trong repo EAGLE)
- `QUESTION_BEGIN` / `QUESTION_END`, `MAX_NEW_TOKENS`, `TOTAL_TOKEN`/`DEPTH`/`TOP_K`
- `OUTPUT_FILE`

## Dữ liệu của bạn (plug-and-play)

Set `DATA_FILE` trong env (ghi đè question file). Định dạng bắt buộc (EAGLE chat):

```json
{"id": 0, "turns": ["user prompt"]}
```

```bash
DATA_FILE="data/user_prompts.jsonl" bash scripts/run.sh eagle3
```

## Output

`outputs/eagle3_qwen3_qa.jsonl` — per-question: new_tokens, tree_steps,
accept_length, eagle/naive tok/s, speedup + summary cuối.

## Troubleshooting

- Base/draft config mismatch → wrapper tự kiểm tra hidden_size/heads/vocab trước khi load.
- GPU RAM: Qwen3-4B fp16 ~9GB; cần GPU ≥ 12GB.
