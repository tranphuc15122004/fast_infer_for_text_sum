# Baseline Inference Guide

Repo này là máy **code/debug**; inference thật chạy trên **server GPU riêng**.
Các hướng dẫn dưới đây được viết để chạy lại được trên server mới (chỉ cần
`git pull` + làm theo từng mục).

## Nội dung

- **Chung** — chuẩn bị venv Python 3.12 offline, định dạng dữ liệu, lệnh chạy
- **Từng baseline** — `docs/baselines/*.md`:

| Baseline | Env | Doc |
|---|---|---|
| EAGLE-3 | venv chung | `docs/baselines/eagle3.md` |
| DFlash | venv chung | `docs/baselines/dflash.md` |
| LLMLingua | venv chung | `docs/baselines/llmlingua.md` |
| FastKV | venv chung | `docs/baselines/fastkv.md` |
| RocketKV | venv chung | `docs/baselines/rocketkv.md` |
| GemFilter | venv chung | `docs/baselines/gemfilter.md` |
| speculative_prefill | venv chung | `docs/baselines/specprefill.md` |
| MInference | venv chung | `docs/baselines/minference.md` |
| MagicDec | venv chung | `docs/baselines/magicdec.md` |
| LongSpec | venv chung | `docs/baselines/longspec.md` |
| SpecExtend | venv chung | `docs/baselines/specextend.md` |
| HiGOE | venv chung | `docs/baselines/higoe.md` |
| semantic_selection | venv chung | adapter trong `docs/representative_100_benchmark.md` |
| FlexPrefill | venv chung | `docs/baselines/flexprefill.md` |

## Chuẩn bị chung trên server

```bash
# 1) Clone; uv và Python 3.12 phải có sẵn trên server offline
git clone <repo> && cd fast_infer_text_sum
uv --version

# 2) Tạo venv chung từ cache/wheelhouse local
bash scripts/setup_venv.sh --offline

# 3) HF token cho model gated (Llama)
export HF_TOKEN=hf_xxxx

# 4) Tải model theo yêu cầu từng baseline (xem doc tương ứng)
```

## Chạy một baseline

```bash
bash scripts/run.sh <baseline> [args...]
```

Mỗi baseline đọc cấu hình `config/<baseline>.env` (model path, data file, tham số).
`bash scripts/run.sh` là dispatcher gọi wrapper; mọi wrapper dùng `.venv/bin/python`.

## Dữ liệu plug-and-play

Định dạng file jsonl + trạng thái hỗ trợ từng baseline: xem
[`data/README.md`](../data/README.md) và `scripts/common/data_loader.py`.

Tóm tắt: bỏ file jsonl vào `data/`, set `DATA_FILE="data/<file>.jsonl"` (và
`MAX_SAMPLES`) trong `config/<baseline>.env` → chạy `bash scripts/run.sh <baseline>`.

## Output

Mọi script ghi `outputs/<baseline>_*.jsonl` theo schema thống nhất
(`externals/baseline_repo_guide.md` §13) + bản `summary` cuối + verify PASS/FAIL.

## Chất lượng tóm tắt (ROUGE)

- Triển khai: `scripts/common/rouge.py` — ROUGE-1/2/L pure-Python, không phụ
  thuộc thư viện ngoài (tương thích mọi env đang khóa `--locked`). Thuật toán
  + interface `rouge_all(hyp, ref)` lấy từ
  `PoTR_article_summary/external/HeterSumGraph/tools/utils.py`.
- Khi dữ liệu có trường `reference`/`summary`/`answer` (xem `data/README.md`),
  các script sinh text (`llmlingua`, `fastkv`, `gemfilter`, `minference`,
  `specprefill`, `eagle3`) tự ghi `rouge1/rouge2/rougeL` vào mỗi record và
  `mean_rouge*` vào bản `summary`.
- `externals/Sematic_selection/infer.py` có cờ `--rouge` để tính ROUGE trên
  toàn bộ selector/budget (quality vs retention, RQ3).
- Baseline không sinh text trong smoke probe độc lập (kernel smoke:
  `rocketkv`, `higoe`, `longspec`,
  `magicdec`, `specextend`) không có ROUGE.

## Benchmark baseline có adapter trên representative_100

Runner + collector strict (tốc độ + ROUGE/BLEU) cho các baseline có adapter đọc
trực tiếp `data/representative_100`: [docs/representative_100_benchmark.md](representative_100_benchmark.md).
Các baseline chỉ có kernel/pipeline smoke được tách riêng và không được tính
vào báo cáo representative nếu chưa có adapter dữ liệu.

## Báo cáo kết quả semantic selection

Phân tích latency, memory và ROUGE của các scheme `random`, `lead`, `tfidf`,
`textrank`, `mmr`: [docs/semantic_selection_analysis.md](semantic_selection_analysis.md).

## Ghi chú portability

- `requirements.txt` là nguồn dependency duy nhất; các local wheel/editable path
  trong đó phải tồn tại trên server.
- Setup dùng `uv pip --offline`; không tải Python/package qua internet.
- `setup_venv.sh --check` kiểm tra Python 3.12 và các local source path; dùng
  `check_shared_env.py` để kiểm tra import/version/CUDA sau khi cài.
- Có thể dùng `FAST_INFER_VENV` hoặc `FAST_INFER_PYTHON` để chỉ định venv/interpreter.
