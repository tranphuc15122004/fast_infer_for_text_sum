# LLMLingua

Prompt/context compression (LLMLingua-2) — nén input trước khi target LLM sinh.
Dựa trên `externals/LLMLingua`.

## Env & cài đặt

- Env: **venv chung** tại `.venv` (Python 3.12, dependency từ `requirements.txt`).
- `bash scripts/setup_venv.sh --offline`.

## Model

| Vai trò | Model | Ghi chú |
|---|---|---|
| Compressor | `microsoft/llmlingua-2-xlm-roberta-large-meetingbank` | nhẹ, chạy CPU được |
| Target LLM | `Qwen/Qwen2.5-1.5B-Instruct` (mặc định nếu cache) | thay bằng model bạn cần |

## Chạy smoke / thật

```bash
bash scripts/run.sh llmlingua        # mặc định smoke (SMOKE=1)
```

Cấu hình `config/llmlingua.env`: `COMPRESSOR_MODEL`, `TARGET_MODEL`,
`DOC_FILE`, `COMPRESSION_RATE`, `MAX_SAMPLES`, `MAX_NEW_TOKENS`, `DEVICE`.

## Dữ liệu của bạn

Set `DOC_FILE` (jsonl, trường `prompt`/`text`/`turns` đều được):

```json
{"id": 0, "prompt": "Nội dung dài cần nén...", "keyword": "entity_quan_trong"}
```

`DOC_FILE="data/user_prompts.jsonl" bash scripts/run.sh llmlingua`

## Output

`outputs/llmlingua_smoke.jsonl` — origin_tokens, retained_tokens, ratio,
selector_latency, target e2e/throughput, summary + verify retention keyword.

## Troubleshooting

- Keyword nên là **entity đặc trưng** (tên riêng, số, thuật ngữ) — từ chung chung
  dễ bị nén mất (verify sẽ FAIL dù đúng hành vi của LLMLingua).
- Chạy được hoàn toàn trên CPU (chậm hơn CUDA).
