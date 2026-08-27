# Baseline Inference Guide

Repo này là máy **code/debug**; inference thật chạy trên **server GPU riêng**.
Trên server B200, các launcher dùng trực tiếp `python3` từ PATH; `.venv` trong
workspace local chỉ dùng để mô phỏng dependency/API trước khi đưa code lên
server.

## Nội dung

- **Chung** — chuẩn bị venv Python 3.12 offline, định dạng dữ liệu, lệnh chạy
- **Từng baseline** — `docs/baselines/*.md`:

| Baseline | Env | Doc |
|---|---|---|
| EAGLE-3 | `python3` server / `.venv` mô phỏng | `docs/baselines/eagle3.md` |
| DFlash | `python3` server / `.venv` mô phỏng | `docs/baselines/dflash.md` |
| LLMLingua | `python3` server / `.venv` mô phỏng | `docs/baselines/llmlingua.md` |
| FastKV | `python3` server / `.venv` mô phỏng | `docs/baselines/fastkv.md` |
| RocketKV | `python3` server / `.venv` mô phỏng | `docs/baselines/rocketkv.md` |
| GemFilter | `python3` server / `.venv` mô phỏng | `docs/baselines/gemfilter.md` |
| speculative_prefill | `python3` server / `.venv` mô phỏng | `docs/baselines/specprefill.md` |
| MInference | `python3` server / `.venv` mô phỏng | `docs/baselines/minference.md` |
| MagicDec | `python3` server / `.venv` mô phỏng | `docs/baselines/magicdec.md` |
| LongSpec | `python3` server / `.venv` mô phỏng | `docs/baselines/longspec.md` |
| SpecExtend | `python3` server / `.venv` mô phỏng | `docs/baselines/specextend.md` |
| HiGOE | `python3` server / `.venv` mô phỏng | `docs/baselines/higoe.md` |
| semantic_selection | `python3` server / `.venv` mô phỏng | adapter trong `docs/representative_100_benchmark.md` |
| FlexPrefill | `python3` server / `.venv` mô phỏng | `docs/baselines/flexprefill.md` |

## Chuẩn bị chung trên server B200

```bash
# 1) Clone; Python 3.12 phải có sẵn trên server offline
git clone <repo> && cd fast_infer_text_sum
python3 --version

# 2) Đặt master config ngoài repository (config/master.path đã trỏ sẵn tới path này)
#    Lần đầu: cp docs/fast_infer_master.example.env /workspace/shared_storage/config/fast_infer_master.env
export FAST_INFER_MASTER_CONFIG=/workspace/shared_storage/config/fast_infer_master.env
source scripts/common/config.sh && fast_infer_load_master

# 3) Kiểm tra runtime/CUDA/dependency/model cache trước khi chạy
python3 scripts/check_b200_env.py --json outputs/b200_preflight.json

# 4) HF token cho model gated (Llama), nếu server dùng model cần token
export HF_TOKEN=hf_xxxx

# 5) Mirror model/checkpoint và wheel vào cache server theo doc tương ứng
```

Để mô phỏng đúng profile trên máy local, thay interpreter ở command runner:

```bash
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  bash scripts/run_b200_smoke.sh --preflight-only
```

## Chạy một baseline

```bash
bash scripts/run.sh <baseline> [args...]
```

Mọi baseline đọc cùng một master shell-env. Pointer mặc định là
`config/master.path`; có thể override bằng `FAST_INFER_MASTER_CONFIG`. Mẫu đầy
đủ và tên canonical nằm ở [`docs/fast_infer_master.example.env`](fast_infer_master.example.env).
`bash scripts/run.sh` là dispatcher gọi wrapper; production dùng `python3`, còn
local simulation đặt `FAST_INFER_PYTHON` tới `.venv/bin/python`.

## Dữ liệu plug-and-play

Định dạng file jsonl + trạng thái hỗ trợ từng baseline: xem
[`data/README.md`](../data/README.md) và `scripts/common/data_loader.py`.

Tóm tắt: bỏ file jsonl vào `data/`, sửa `DATA_INPUT` và `RUN_SAMPLES` trong
master → chạy `bash scripts/run.sh <baseline>`. Override nhanh một lần vẫn có
thể đặt biến môi trường, ví dụ `DATA_INPUT=data/user.jsonl RUN_SAMPLES=5 ...`.

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
- Có thể dùng `FAST_INFER_VENV` hoặc `FAST_INFER_PYTHON` để chỉ định interpreter;
  giá trị `FAST_INFER_PYTHON=python3` được resolve qua PATH.
- `config/master.path` + master ngoài repo + `scripts/run_b200_smoke.sh` là
  đường chạy one-sample có preflight và summary cho server B200.
