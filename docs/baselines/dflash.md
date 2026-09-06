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

## Phân biệt với MR-DFlash

Doc này mô tả **DFlash inference baseline** ở `externals/dflash`. Nó độc lập
với workspace train [`src/MR_DFlash`](../../src/MR_DFlash/README.md), nơi hiện
chỉ giữ bản sao quy trình/model DFlash để làm gốc cho một ý tưởng mới. Chưa có
thay đổi thuật toán MR-DFlash nào và chưa được dùng như một baseline benchmark
riêng. Xem [bối cảnh MR-DFlash](../mr_dflash.md).

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
