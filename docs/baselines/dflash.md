# DFlash

Parallel/diffusion-style drafting (DFlash) — dựa trên `externals/dflash`.

## Env & cài đặt

- Env: **venv chung** tại `.venv` (Python 3.12, dependency từ `requirements.txt`).
- `bash scripts/setup_venv.sh --offline`.

## Model

| Vai trò | Model | Ghi chú |
|---|---|---|
| Target | `MODEL_TARGET` | đặt snapshot local trong master nếu server offline |
| Draft | `MODEL_DFLASH_DRAFT` | đặt snapshot local trong master nếu server offline |

## Chạy smoke / thật

```bash
bash scripts/run.sh dflash
```

Cấu hình trong master: `DFLASH_MODE=gsm8k`, `DFLASH_BACKEND`
(transformers|sglang|vllm), `DFLASH_DATASET`, `DFLASH_MAX_SAMPLES`,
`DFLASH_MAX_NEW_TOKENS`, `DFLASH_BLOCK_SIZE`. Dùng
`DFLASH_MODE=representative` để chạy JSONL của repo.

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
