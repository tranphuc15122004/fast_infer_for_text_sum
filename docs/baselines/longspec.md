# LongSpec

Purpose-built long-context learned speculative decoding (Anchor-Offset + Hybrid
Tree Attention). Dựa trên `externals/LongSpec`.

## Env & cài đặt

- Env: **`envs/longspec`** (transformers 4.46.3, torch 2.5.1, triton 3.1.0, liger-kernel).
- `uv sync --project envs/longspec --locked`
- Full inference cần flash-attn + **GPU 80GB-class** (không chạy được T4).

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
(→ `python inference_long-bench.py --model_name llama8b --method tree
--task gov_report --data_path_prefix ...`)

Cấu hình `config/longspec.env`: `MODEL_NAME`, `METHOD`, `TASK`, `MAX_GEN_LEN`,
`TREE_SHAPE`.

## Dữ liệu

Cần longbench jsonl **đã tiền xử lý** (đường dẫn qua `DATA_PATH_PREFIX`).

## Output

Smoke: `outputs/longspec_smoke.jsonl` (import_ok, kernel_ok).
Full: log inference.

## Troubleshooting

- Triton kernel (`triton_tree_attn.py`) phải khớp phiên bản triton 3.1.0.
- flash-attn cần `EXTRA_FLASH=1` (GPU sm80+).
