# Fast Infer Text Summarization

Efficient inference benchmark cho **long-context text summarization**:
đánh giá công bằng nhiều baseline tăng tốc inference (semantic reduction,
sparse attention, KV optimization, speculative decoding) trên cùng dữ liệu
của bạn.

## Cấu trúc

```
scripts/           # script kiểm chứng từng baseline + common helpers + runner
config/            # cấu hình per baseline (model path, data, tham số)
requirements.txt   # dependency manifest của server GPU Python 3.12
externals/         # các baseline repo (vendored)
data/              # dữ liệu plug-and-play (jsonl) + README định dạng
outputs/           # kết quả JSONL theo schema thống nhất
docs/              # hướng dẫn cài đặt + infer từng baseline
```

## Quick start

```bash
# Python 3.12 phải được cài sẵn trên server offline
python3 --version

# nạp profile B200; không cần activate virtualenv
set -a; source config/b200.env; set +a
python3 scripts/check_b200_env.py --json outputs/b200_preflight.json

# model gated (Llama) cần token
export HF_TOKEN=hf_xxx

# chạy một baseline
bash scripts/run.sh <baseline>        # eagle3 dflash llmlingua fastkv rocketkv
                                      # gemfilter specprefill minference magicdec
                                      # longspec specextend higoe semantic_selection
                                      # flexprefill
```

Smoke toàn bộ 14 baseline trên B200 (mỗi baseline một sample):

```bash
bash scripts/run_b200_smoke.sh
```

Mô phỏng local bằng `.venv`:

```bash
# Nếu cần dựng lại môi trường mô phỏng từ wheel/cache local:
bash scripts/setup_venv.sh --offline
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  bash scripts/run_b200_smoke.sh --preflight-only
```

## Tài liệu

- **Hướng dẫn từng baseline (cài đặt + infer)**: [`docs/README.md`](docs/README.md)
  → `docs/baselines/*.md`
- **Định dạng dữ liệu plug-and-play**: [`data/README.md`](data/README.md)
- **Cấu trúc env / portability**: [`envs/README.md`](envs/README.md)
- **Thiết kế thí nghiệm / taxonomy baseline**: [`externals/baseline_repo_guide.md`](externals/baseline_repo_guide.md)

## Nguyên tắc

- Mỗi baseline: `scripts/infer_<b>.py` + `scripts/run_<b>.sh` + `config/<b>.env`,
  có chế độ `--smoke` và mode full; B200 smoke được điều phối bởi
  `scripts/run_b200_smoke.sh`.
- Dữ liệu của bạn: bỏ jsonl vào `data/`, set `DATA_FILE` trong config → chạy.
- Kết quả: `outputs/*.jsonl` theo schema §13 của `baseline_repo_guide.md`
  (input/retained/output tokens, TTFT/TPOT/E2E, throughput, quality metrics...).
- Môi trường production: `python3` Python 3.12 từ PATH dùng `requirements.txt`;
  launcher không yêu cầu activate venv. `.venv` chỉ dành cho mô phỏng local.
