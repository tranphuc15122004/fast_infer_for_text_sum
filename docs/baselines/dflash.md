# DFlash

Parallel/diffusion-style drafting (DFlash) — dựa trên `externals/dflash`.

## Env & cài đặt

- Env: **venv chung** tại `.venv` (Python 3.12, dependency từ `requirements.txt`).
- `bash scripts/setup_venv.sh --offline`.

## Model

| Vai trò | Model | Ghi chú |
|---|---|---|
| Target | `Qwen/Qwen3-4B` | đã cache, path trong `config/dflash_gsm8k.env` |
| Draft | `z-lab/Qwen3-4B-DFlash-b16` | đã cache |

## Chạy smoke / thật

```bash
bash scripts/run.sh dflash
```

Cấu hình `config/dflash_gsm8k.env`: `BACKEND` (transformers|sglang|vllm),
`DATASET`, `MAX_SAMPLES`, `MAX_NEW_TOKENS`, `BLOCK_SIZE`.

## Dữ liệu của bạn

⚠️ DFlash benchmark **chỉ hỗ trợ dataset builtin**: `gsm8k | math500 | humaneval |
mbpp | mt-bench` (tự tải về `externals/dflash/cache/`). Muốn dataset riêng phải
thêm entry vào `DATASETS` trong `externals/dflash/dflash/benchmark.py`.

## Output

In bảng speedup + acceptance histogram (không ghi JSONL của riêng script này;
bảng in ra terminal).

## Troubleshooting

- `--enable-thinking` bị cấm với draft Qwen3-4B/8B.
- Backend `transformers` chạy được 1 GPU; `sglang`/`vllm` cần server.
