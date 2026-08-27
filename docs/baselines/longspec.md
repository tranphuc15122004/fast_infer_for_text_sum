# LongSpec

Purpose-built long-context learned speculative decoding (Anchor-Offset + Hybrid
Tree Attention). Dựa trên `externals/LongSpec`.

## Env & cài đặt

- Env: **venv chung** tại `.venv` (Python 3.12, dependency từ `requirements.txt`).
- `bash scripts/setup_venv.sh --offline`
- Full inference cần các kernel tương thích trong requirements và **GPU 80GB-class**
  (không chạy được T4).

## Model

Cặp model được hardcode trong `longspec/test/inference_long-bench.py`:
- Target: `llama8b` = `gradientai/Llama-3-8B-Instruct-262k` (gated → HF_TOKEN),
  hoặc `vicuna7b`/`longchat7b`/`qwen`.
- Draft: `sail/longspec-*` trên HF.

## Chạy smoke (không cần GPU lớn)

```bash
bash scripts/run.sh longspec     # smoke mặc định: verify import + triton TreeAttention kernel
```

## Chạy thật (GPU 80GB)

```bash
FULL=1 DATA_PATH_PREFIX="/path/to/longbench_preprocessed" bash scripts/run.sh longspec
```
(wrapper gọi `inference_long-bench.py` bằng interpreter chung với các tham số
`--model_name llama8b --method tree --task gov_report --data_path_prefix ...`)

Cấu hình trong master: `LONGSPEC_MODEL_NAME`, `LONGSPEC_METHOD`,
`LONGSPEC_TASK`, `LONGSPEC_MAX_NEW_TOKENS`, `LONGSPEC_TREE_SHAPE`.

## Dữ liệu

Cần longbench jsonl **đã tiền xử lý** (đường dẫn qua `DATA_PATH_PREFIX`).

## Output

Smoke: `outputs/longspec_smoke.jsonl` (import_ok, kernel_ok).
Full: log inference.

## Troubleshooting

- Triton kernel (`triton_tree_attn.py`) phải khớp phiên bản triton 3.1.0.
- Các kernel tăng tốc phải tương thích với CUDA/GPU và có sẵn trong wheelhouse
  local của server.
