# MagicDec

Long-context self/speculative decoding (SnapKV/StreamingLLM kernels, TP).
Dựa trên `externals/MagicDec`.

## Env & cài đặt

- Env: **`envs/magicdec`** (transformers 4.36.2, torch 2.4.1+cu124, `flashinfer-python`).
- `uv sync --project envs/magicdec --locked`
- Bắt buộc CUDA + NCCL cho distributed; single-GPU smoke chạy trực tiếp bằng
  Python để không cần rendezvous localhost.

## Model (checkpoint phải convert)

MagicDec cần `model.pth` (không phải HF format). Có 2 bước:
1. `download.py --repo_id <hf> --out_dir ...` (tải HF checkpoint)
2. `convert_hf_checkpoint.py --checkpoint_dir ...` → `model.pth`

Wrapper đã có helper tự động:

```bash
PREPARE_CHECKPOINT=1 \
REPO_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
MODEL_KEY="tinyllama" \
bash scripts/run.sh magicdec
```

⚠️ **Tên thư mục chứa checkpoint phải khớp key** trong
`externals/MagicDec/Engine/SnapKV/model.py` (vd `tinyllama`, `llama-3.1-8b`,
`qwen2.5-7b`...) vì `Transformer.from_name(folder_name)` quyết định config.

## Chạy smoke / thật

```bash
bash scripts/run.sh magicdec     # smoke: self-spec, TinyLlama, B=1, prefix 2048
```

Cấu hình `config/magicdec.env`: `MODEL_PTH`, `MODEL_NAME`, `BATCH_SIZE`,
`PREFIX_LEN`, `MAX_LEN` (bội của 128), `SELF_SPEC`/`GAMMA`/`DRAFT_BUDGET`.

- Full (GPU lớn): llama-3.1-8b, TP=8, `--compile`.

## Dữ liệu

Benchmark đọc dataset pg19 từ
`${HF_HOME:-$HOME/.cache/huggingface}/datasets/fast_infer_text_sum/MagicDec/Data/pg19`.
Có thể đổi vị trí bằng biến `MAGICDEC_DATA_ROOT`; không cần đặt dataset trong
`externals/MagicDec/Data/` nữa.

## Output

`outputs/magicdec_smoke.jsonl` — returncode + log tail của benchmark.

## Troubleshooting

- `--max-len` phải `% 128 == 0`.
- flashinfer wheel phải khớp torch/CUDA; index trong `envs/magicdec/pyproject.toml`
  (nếu đổi torch → đổi URL `flashinfer.ai/whl/{cu...}/torch.../`).
- Model nhỏ (TinyLlama/llama-68m) hoặc int8 mới vừa T4 16GB.
