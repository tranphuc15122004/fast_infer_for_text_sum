# Baseline Inference Guide

Repo này là máy **code/debug**; inference thật chạy trên **server GPU riêng**.
Trên server B200, các launcher dùng trực tiếp `python3` từ PATH; `.venv` trong
workspace local chỉ dùng để mô phỏng dependency/API trước khi đưa code lên
server. Thông tin path/runtime canonical: [`docs/server_environment.md`](server_environment.md).

## Nội dung

- **Chung** — chuẩn bị venv Python 3.12 offline, định dạng dữ liệu, lệnh chạy
- **Từng baseline** — `docs/baselines/*.md`:

| Baseline | Env | Doc |
|---|---|---|
| EAGLE-3 | `python3` server / `.venv` mô phỏng | `docs/baselines/eagle3.md` |
| DFlash | `python3` server / `.venv` mô phỏng | `docs/baselines/dflash.md` |
| SSSD | `python3` server / `.venv` mô phỏng | `docs/baselines/sssd.md` |
| FAFO | `python3` server / `.venv` mô phỏng | `docs/baselines/fafo.md` |
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
| semantic_selection | `python3` server / `.venv` mô phỏng | adapter trong `docs/longbench_200_benchmark.md` |
| FlexPrefill | `python3` server / `.venv` mô phỏng | `docs/baselines/flexprefill.md` |
| SyncSpec-v1 | `python3` server / `.venv` mô phỏng | `docs/baselines/syncspec.md` |

`MR_DFlash` không nằm trong bảng baseline trên: đây là workspace phát triển
thuật toán train mới, hiện chỉ là bản sao quy trình/model DFlash để làm gốc.
Xem [bối cảnh MR-DFlash](mr_dflash.md) và
[`src/MR_DFlash/README.md`](../src/MR_DFlash/README.md). Nó chưa có launcher
benchmark inference hay kết quả riêng.

## Chuẩn bị chung trên server B200

```bash
# 1) Clone; Python 3.12 phải có sẵn trên server offline
git clone <repo> && cd fast_infer_text_sum
python3 --version

# 2) Đặt master config ngoài repository (config/master.path đã trỏ sẵn tới path này)
#    Master config ổn định nằm tại:
#    /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env
export FAST_INFER_MASTER_CONFIG=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env
source scripts/common/config.sh && fast_infer_load_master

# 3) Kiểm tra runtime/CUDA/dependency/model cache trước khi chạy
python3 scripts/check_b200_env.py --json outputs/b200_preflight.json

# 4) HF token cho model gated (Llama), nếu server dùng model cần token
export HF_TOKEN=hf_xxxx

# 5) Mirror model/checkpoint và wheel vào cache server theo doc tương ứng
```

## Bootstrap server không dùng venv

Server benchmark dùng trực tiếp `python3` hệ thống (Python 3.12). Script sau
không tạo virtualenv và không cài package; script chỉ khởi tạo cấu trúc dùng
chung, trỏ `config/master.path` tới master config ổn định và chạy preflight:

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

# Tạo thư mục data/config và link còn thiếu; không kiểm tra package/model.
python3 scripts/setup_server_env.py --init

# Sau khi copy đủ longbench_200 và representative_100 vào shared data:
python3 scripts/setup_server_env.py --check

# Hoặc thực hiện init + check trong một lần.
python3 scripts/setup_server_env.py --all
```

Mặc định script dùng:

```text
repository: /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum
shared data: /workspace/storage-shared/nlp/dungdx4/phuc_projects/data
master:      <shared data>/fast_infer_master.env
```

Nếu chỉ muốn kiểm tra filesystem trước khi dataset được copy, dùng
`--check --skip-dependencies --skip-data-validation`. Script không ghi đè
master config đã tồn tại; hãy chỉnh các đường dẫn model/checkpoint trực tiếp
trong `fast_infer_master.env`.

Để mô phỏng đúng profile trên máy local, thay interpreter ở command runner:

```bash
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  bash scripts/run_b200_smoke.sh --preflight-only
```

## Chạy một baseline

```bash
bash scripts/run.sh <baseline> [args...]
```

SyncSpec có thêm preflight/smoke riêng vì cần checkpoint drafter đã train:

```bash
python3 scripts/check_syncspec_b200.py --strict
bash scripts/run_syncspec_b200_smoke.sh
# Smoke toàn chuỗi train + infer (tạo drafter checkpoint mới):
bash scripts/run_syncspec_b200_train_smoke.sh
# Smoke toàn chuỗi CPU deterministic:
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  bash scripts/run_syncspec_cpu_smoke.sh docs/fast_infer_master.example.env
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
- `sssd` và `fafo` là pipeline upstream tích hợp qua adapter. Do upstream chỉ
  trả metric aggregate trong mode benchmark, mỗi lần chạy ghi một record
  aggregate; smoke luôn giới hạn đúng một sample.

## Smoke SSSD và FAFO với Llama 3.1 8B Instruct

Hai launcher dùng cùng `config/master.path`, cùng interpreter Python 3.12 và
cùng `DATA_INPUT`. Cấu hình mẫu đã có các namespace `SSSD_*` và `FAFO_*`.

```bash
export FAST_INFER_MASTER_CONFIG=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env
export MODEL_TARGET=/workspace/shared_storage/model/Llama3.1-8B-Instruct
export DATA_INPUT=data/representative_100/xsum_representative.jsonl
export RUN_SAMPLES=1
export RUN_MAX_NEW_TOKENS=16

bash scripts/run.sh sssd --smoke
bash scripts/run.sh fafo --smoke
```

SSSD cần extension native `sssd_speculator`; có thể truyền datastore đã build
bằng `SSSD_DATASTORE_PATH`. Để chạy SSSD adaptive đặt `SSSD_ADAPTIVE=1`.
FAFO hỗ trợ `FAFO_KV_METHOD=stream-llm` hoặc `quest`; `FAFO_USE_FLASH=1` chỉ
dùng khi stack FlashAttention tương thích. Kết quả nằm ở `OUTPUT_FILE` hoặc
`outputs/<baseline>.jsonl`, còn log/raw result FAFO nằm ở thư mục
`outputs/fafo_runtime/` tương ứng.

## Benchmark baseline trên bộ canonical LongBench

Schema, lệnh build/validate, prompt task-specific và metric cho bộ 5 task nằm ở
[`docs/longbench_200_benchmark.md`](longbench_200_benchmark.md). Collector mặc
định đọc `data/longbench_200`; code-completion dùng exact/edit similarity thay
cho ROUGE. `data/representative_100` được giữ như dữ liệu legacy để tái hiện
run cũ.

Runner ma trận chuẩn dùng `scripts/run_longbench_200.sh` và master config:

```bash
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  bash scripts/run_longbench_200.sh --config /path/to/fast_infer_master.env \
  --mode smoke --preflight-only
```

`smoke`, `representative`, `full` tương ứng lần lượt 1 mẫu, 20 mẫu đại diện và
200 mẫu/dataset; chi tiết output, status và lệnh B200 xem trong
[`docs/longbench_200_benchmark.md`](longbench_200_benchmark.md).

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
