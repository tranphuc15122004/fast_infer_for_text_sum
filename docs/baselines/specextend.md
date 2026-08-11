# SpecExtend

Drop-in long-context speculative decoding (target-guided, training-free) cho
long-document summarization. Dựa trên `externals/SpecExtend`.

## Env & cài đặt

- Env: **`envs/legacy`** (dùng chung với FastKV/RocketKV/GemFilter/HiGOE: transformers 4.45.2 — pin gốc 4.41.0 đã thả lỏng; `tokenizers` lên 0.20.x, `protobuf` lên 4.25.1).
- `uv sync --project envs/legacy --locked`
- Full cần flash-attn (`EXTRA_FLASH=1`, GPU sm80+).

## Model

| Vai trò | Model |
|---|---|
| Target | `lmsys/vicuna-7b-v1.5-16k` (default `vicuna_7b`) hoặc `lmsys/longchat-7b-16k` |
| Draft (classic) | `double7/vicuna-68m` / `JackFram/llama-68m` (tự tải) |

T4 16GB: vicuna-7b fp16 ~13GB — **ranh giới**, chỉ chạy với input ngắn nhất.

## Chạy smoke / thật

```bash
bash scripts/run.sh specextend   # smoke: govreport_512 + vicuna-7b + max_gen_len 64
```

Cấu hình `config/specextend.env`: `SCRIPT` (run_classic.py|run_eagle.py),
`MODEL_NAME`, `INPUT_FILE` (jsonl có trường `text`), `MAX_SAMPLES`,
`MAX_GEN_LEN`, `USE_SPECEXTEND`.

Data kèm sẵn: `externals/SpecExtend/specextend/data/govreport/govreport_{512,1K,2K,4K,8K,16K}.jsonl`
(đã nhúng prompt summarize).

## Dữ liệu của bạn

File jsonl định dạng `{"id":.., "text": "..."}` (prompt summarize sẽ được repo
tự gắn). Đặt vào `data/` và trỏ `INPUT_FILE="data/<file>.jsonl"`.

## Output

`outputs/specextend_smoke.jsonl` — returncode + số dòng summary sinh được.

## Troubleshooting

- Nếu OOM trên T4: giảm `MAX_GEN_LEN`/dùng `govreport_512`, hoặc chuyển GPU lớn.
- `eval_classic.py`/`eval_eagle.py` chạy sweep nhiều độ dài (dành cho GPU lớn).
