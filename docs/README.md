# Baseline Inference Guide

Repo này là máy **code/debug**; inference thật chạy trên **server GPU riêng**.
Các hướng dẫn dưới đây được viết để chạy lại được trên server mới (chỉ cần
`git pull` + làm theo từng mục).

## Nội dung

- **Chung** — cài uv, sync env, định dạng dữ liệu, lệnh chạy (mục dưới)
- **Từng baseline** — `docs/baselines/*.md`:

| Baseline | Env | Doc |
|---|---|---|
| EAGLE-3 | core (root) | `docs/baselines/eagle3.md` |
| DFlash | core (root) | `docs/baselines/dflash.md` |
| LLMLingua | core (root) | `docs/baselines/llmlingua.md` |
| FastKV | `envs/legacy` | `docs/baselines/fastkv.md` |
| RocketKV | `envs/legacy` | `docs/baselines/rocketkv.md` |
| GemFilter | `envs/legacy` | `docs/baselines/gemfilter.md` |
| speculative_prefill | `envs/specprefill` | `docs/baselines/specprefill.md` |
| MInference | `envs/specprefill` | `docs/baselines/minference.md` |
| MagicDec | `envs/magicdec` | `docs/baselines/magicdec.md` |
| LongSpec | `envs/longspec` | `docs/baselines/longspec.md` |
| SpecExtend | `envs/legacy` | `docs/baselines/specextend.md` |
| HiGOE | `envs/legacy` | `docs/baselines/higoe.md` |

## Chuẩn bị chung trên server

```bash
# 1) Clone + cài uv
git clone <repo> && cd fast_infer_text_sum
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 2) Sync toàn bộ env từ lock files (tái lập chính xác, không cần đoán version)
bash scripts/setup_envs.sh
#    - GPU lớn (A100/H100) muốn dùng flash-attn cho FastKV/GemFilter/LongSpec/SpecExtend:
#      EXTRA_FLASH=1 bash scripts/setup_envs.sh

# 3) HF token cho model gated (Llama)
export HF_TOKEN=hf_xxxx

# 4) Tải model theo yêu cầu từng baseline (xem doc tương ứng)
```

## Chạy một baseline

```bash
bash scripts/run.sh <baseline> [args...]
```

Mỗi baseline đọc cấu hình `config/<baseline>.env` (model path, data file, tham số).
`bash scripts/run.sh` là dispatcher tự chọn đúng env + wrapper.

## Dữ liệu plug-and-play

Định dạng file jsonl + trạng thái hỗ trợ từng baseline: xem
[`data/README.md`](../data/README.md) và `scripts/common/data_loader.py`.

Tóm tắt: bỏ file jsonl vào `data/`, set `DATA_FILE="data/<file>.jsonl"` (và
`MAX_SAMPLES`) trong `config/<baseline>.env` → chạy `bash scripts/run.sh <baseline>`.

## Output

Mọi script ghi `outputs/<baseline>_*.jsonl` theo schema thống nhất
(`externals/baseline_repo_guide.md` §13) + bản `summary` cuối + verify PASS/FAIL.

## Ghi chú portability

- `uv.lock` đã commit cho từng env → `uv sync --locked` tái lập được trên máy khác.
- Sau khi sửa `envs/<g>/pyproject.toml`: chạy `uv lock --project envs/<g>` và commit cả lock.
- flash-attn (sm80+) chỉ cài khi `EXTRA_FLASH=1`; T4 (sm75) phải build từ source.
