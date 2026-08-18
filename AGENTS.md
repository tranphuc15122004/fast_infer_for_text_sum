# Project Guidelines — fast_infer_text_sum

Benchmark repo cho **long-context text summarization**: so sánh công bằng các
baseline tăng tốc inference (semantic reduction, sparse attention, KV
optimization, speculative decoding) trên cùng dữ liệu/output schema. Toàn bộ
docs tiếng Việt. Python 3.11 + `uv`.

## Cấu trúc folder (hiện tại)

```
fast_infer_text_sum/
├── scripts/                    # script kiểm chứng từng baseline + helpers
│   ├── infer_<baseline>.py     # mỗi baseline 1 file (--smoke + full mode)
│   ├── run_<baseline>.sh       # wrapper: source config/<b>.env → uv run --project <env>
│   ├── run.sh                  # dispatcher: bash scripts/run.sh <baseline>
│   ├── setup_envs.sh           # uv sync --locked cho mọi env (EXTRA_FLASH=1 → flash-attn)
│   ├── bootstrap.sh            # máy mới: uv → sync → smoke llmlingua + rocketkv
│   ├── eagle3_infer_qwen3.py   # EAGLE-3 batch (speedup vs naive)
│   ├── magicdec_prepare_checkpoint.py  # HF → MagicDec model.pth
│   └── common/                 # helpers dùng chung (import as `from common import ...`)
│       ├── paths.py            # ROOT, repo_dir(), hf_token(), snapshot_dir(), gpu_name()
│       ├── io_util.py          # JsonlWriter + schema §13, finalize() ghi summary
│       ├── verify.py           # check PASS/FAIL, set exit code
│       ├── rouge.py            # ROUGE-1/2/L pure-Python (no-deps), add_rouge()/aggregate_rouge()
│       ├── data_loader.py      # chuẩn hóa jsonl → {id, prompt, answer?, reference?, keyword?}
│       └── __init__.py
├── config/                     # cấu hình per baseline: config/<b>.env (+ _smoke/_dense/_prepare/_gsm8k)
│                               # biến: SMOKE, FULL, OUTPUT_FILE, DATA_FILE, MAX_SAMPLES, MAX_NEW_TOKENS...
├── envs/                       # uv env nhóm tương thích (mỗi nhóm pyproject.toml + uv.lock commit)
│   ├── legacy/                 # FastKV, RocketKV, GemFilter, SpecExtend, HiGOE (tf 4.45.2, torch 2.4.1 cu124)
│   │   └── wheels/             # dgl Linux wheel vendored (PyPI chỉ có Windows)
│   ├── specprefill/            # speculative_prefill, MInference (vllm 0.6.3, torch 2.4.0, tilelang)
│   ├── magicdec/               # MagicDec (transformers 4.36.2, flashinfer-python)
│   └── longspec/               # LongSpec (transformers 4.46.3, triton 3.1.0, liger-kernel)
│   # root pyproject.toml = core env: EAGLE, dflash, LLMLingua (tf 4.57.1, torch cu126)
├── externals/                  # baseline repos vendored + guide
│   ├── EAGLE/  dflash/  FastKV/  RocketKV/  GemFilter/  HiGOE/  LLMLingua/
│   ├── LongSpec/  MagicDec/  MInference/  SpecExtend/  SpecForge/  speculative_prefill/
│   └── baseline_repo_guide.md  # taxonomy nghiên cứu + unified schema §13
├── src/text_sum/               # package placeholder (chưa có code)
├── data/                       # dữ liệu plug-and-play jsonl + README định dạng
├── outputs/                    # kết quả JSONL theo schema (GITIGNORED)
├── checkpoints/                # MagicDec checkpoint đã convert (GITIGNORED)
├── docs/                       # docs/README.md (master) + docs/baselines/<b>.md (12 file)
├── main.py                     # stub hello-world
└── pyproject.toml              # project root (core env)
```

## Ánh xạ baseline → env

| Baseline | Env (`--project`) |
|---|---|
| EAGLE-3, dflash, LLMLingua | root |
| FastKV, RocketKV, GemFilter, HiGOE, SpecExtend | `envs/legacy` |
| speculative_prefill, MInference | `envs/specprefill` |
| MagicDec | `envs/magicdec` |
| LongSpec | `envs/longspec` |

## Conventions

- **1 baseline = 1 bộ file**: `scripts/infer_<b>.py` + `scripts/run_<b>.sh` +
  `config/<b>.env` + `docs/baselines/<b>.md`, được nối vào `case` dispatch trong `run.sh`.
- **Smoke vs full**: mặc định `--smoke` (T4-safe: TinyLlama/Qwen2.5-1.5B, sdpa/eager,
  gen ngắn, ít sample); full cần GPU lớn + flash-attn + model thật. `SMOKE=1`/`FULL=1` trong config.
- **Output schema**: mọi record qua `io_util.JsonlWriter` (thêm record bằng `.add()`,
  kết thúc bằng `.finalize()` ghi `{"type": "summary", ...}`). Schema chuẩn ở
  §13 `externals/baseline_repo_guide.md`.
- **Dữ liệu plug-and-play**: bỏ jsonl vào `data/`, set `DATA_FILE`/`DOC_FILE` +
  `MAX_SAMPLES` trong config. Loader ưu tiên `prompt → question → instruction → text → turns[0]`.
  Lưu ý: HiGOE & DFlash KHÔNG đọc `DATA_FILE`.
- **ROUGE quality**: khi data có `reference`/`summary`/`answer`, script sinh text
  gọi `rouge.add_rouge(record, text, ref)` (ghi `rouge1/rouge2/rougeL` vào record)
  và `rouge.aggregate_rouge(records)` cho bản summary. Module `common/rouge.py`
  là pure-Python (không cài thêm package vào env).

## Commands

```bash
bash scripts/run.sh <baseline>        # chạy 1 baseline (eagle3 dflash llmlingua fastkv
                                      # rocketkv gemfilter specprefill minference magicdec
                                      # longspec specextend higoe)
bash scripts/setup_envs.sh            # sync mọi env (EXTRA_FLASH=1 → + flash-attn, sm80+)
uv sync --project envs/<g> --locked   # tái lập 1 env từ lock (luôn dùng --locked)
```

## Gotchas

- **Luôn `--locked`** với uv; nếu sửa `envs/<g>/pyproject.toml` phải chạy
  `uv lock --project envs/<g>` và commit `uv.lock`.
- **Mỗi env là venv riêng** — không giả định package dùng chung giữa các env.
- **flash-attn là optional** (code đã patch import lỗi được); chỉ cài với
  `EXTRA_FLASH=1` trên GPU sm80+ (T4/sm75 phải build từ source).
- **Model gated (Llama)** cần `export HF_TOKEN=hf_xxx`; ưu tiên dùng snapshot
  đã cache local (`paths.snapshot_dir()`).
- Kết quả nằm ở `outputs/` (gitignored) — không commit artifact.

Chi tiết cài đặt/infer từng baseline: `docs/README.md` → `docs/baselines/*.md`.
Định dạng dữ liệu: `data/README.md`. Cấu trúc env/portability: `envs/README.md`.
Thiết kế thí nghiệm/taxonomy/schema: `externals/baseline_repo_guide.md`.
