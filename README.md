# Fast Infer Text Summarization

Efficient inference benchmark cho **long-context text summarization**:
đánh giá công bằng nhiều baseline tăng tốc inference (semantic reduction,
sparse attention, KV optimization, speculative decoding) trên cùng dữ liệu
của bạn.

## Cấu trúc

```
scripts/           # script kiểm chứng từng baseline + common helpers + runner
config/            # cấu hình per baseline (model path, data, tham số)
envs/              # uv env theo nhóm tương thích (mỗi nhóm có uv.lock)
externals/         # các baseline repo (vendored)
data/              # dữ liệu plug-and-play (jsonl) + README định dạng
outputs/           # kết quả JSONL theo schema thống nhất
docs/              # hướng dẫn cài đặt + infer từng baseline
```

## Quick start

```bash
# cài uv
curl -LsSf https://astral.sh/uv/install.sh | sh && export PATH="$HOME/.local/bin:$PATH"

# cài môi trường (tái lập từ uv.lock)
bash scripts/setup_envs.sh
# GPU lớn cần flash-attn:  EXTRA_FLASH=1 bash scripts/setup_envs.sh

# model gated (Llama) cần token
export HF_TOKEN=hf_xxx

# chạy một baseline
bash scripts/run.sh <baseline>        # eagle3 dflash llmlingua fastkv rocketkv
                                      # gemfilter specprefill minference magicdec
                                      # longspec specextend higoe
```

## Tài liệu

- **Hướng dẫn từng baseline (cài đặt + infer)**: [`docs/README.md`](docs/README.md)
  → `docs/baselines/*.md`
- **Định dạng dữ liệu plug-and-play**: [`data/README.md`](data/README.md)
- **Cấu trúc env / portability**: [`envs/README.md`](envs/README.md)
- **Thiết kế thí nghiệm / taxonomy baseline**: [`externals/baseline_repo_guide.md`](externals/baseline_repo_guide.md)

## Nguyên tắc

- Mỗi baseline: `scripts/infer_<b>.py` + `scripts/run_<b>.sh` + `config/<b>.env`,
  có chế độ `--smoke` (T4-safe) và mode full (GPU lớn).
- Dữ liệu của bạn: bỏ jsonl vào `data/`, set `DATA_FILE` trong config → chạy.
- Kết quả: `outputs/*.jsonl` theo schema §13 của `baseline_repo_guide.md`
  (input/retained/output tokens, TTFT/TPOT/E2E, throughput, quality metrics...).
- Môi trường: mỗi nhóm 1 uv venv với `uv.lock` commit → tái lập trên server khác
  bằng `uv sync --locked`.
