# Project Guidelines — fast_infer_text_sum

Benchmark repo cho **long-context text summarization**: so sánh công bằng các
baseline tăng tốc inference (semantic reduction, sparse attention, KV
optimization, speculative decoding) trên cùng dữ liệu/output schema. Toàn bộ
docs tiếng Việt. Runtime server chính: Python 3.12 + `requirements.txt` trong
một venv dùng chung; server không có kết nối internet trực tiếp.

## Cấu trúc folder

```
fast_infer_text_sum/
├── scripts/                    # script kiểm chứng từng baseline + helpers
│   ├── infer_<baseline>.py     # mỗi baseline 1 file (--smoke + full mode)
│   ├── run_<baseline>.sh       # wrapper dùng runtime helper chung
│   ├── run.sh                  # dispatcher: bash scripts/run.sh <baseline>
│   ├── setup_venv.sh            # tạo/cài venv Python 3.12 offline
│   ├── check_shared_env.py      # preflight import/version, không tải model
│   └── common/                 # helpers dùng chung
├── config/                     # cấu hình per baseline
├── envs/                       # manifest/lock legacy, không còn venv runtime
├── externals/                  # baseline repos vendored + guide
├── data/                       # dữ liệu plug-and-play jsonl
├── outputs/                    # kết quả JSONL theo schema (GITIGNORED)
├── checkpoints/                # checkpoint đã convert (GITIGNORED)
├── docs/                       # docs master + docs/baselines
├── requirements.txt            # dependency manifest duy nhất của server
└── .venv/                      # venv Python 3.12, GITIGNORED
```

Các `envs/*/pyproject.toml` và `envs/*/uv.lock` chỉ còn để truy vết dependency
cũ. Không tạo hoặc sử dụng venv riêng cho từng baseline.

## Runtime chung

- Mặc định interpreter là `.venv/bin/python`.
- `FAST_INFER_VENV=/path/to/venv` chọn một venv Python 3.12 khác.
- `FAST_INFER_PYTHON=/path/to/python` chọn trực tiếp executable Python 3.12.
- `scripts/common/runtime.sh` kiểm tra major/minor trước khi launcher chạy.
- `uv` chỉ dùng để tạo/cài venv; cài đặt luôn có `--offline`.
- Các local wheel/editable path trong `requirements.txt` phải tồn tại trên server.

## Conventions

- **1 baseline = 1 bộ file**: `scripts/infer_<b>.py` + `scripts/run_<b>.sh` +
  `config/<b>.env` + `docs/baselines/<b>.md`, được nối vào `run.sh`.
- **Smoke vs full**: mặc định `--smoke` (T4-safe khi baseline hỗ trợ); full cần
  GPU lớn, kernel tương thích và model/cache thật. `SMOKE=1`/`FULL=1` trong config.
- **Output schema**: mọi record qua `io_util.JsonlWriter`, kết thúc bằng summary.
- **Dữ liệu plug-and-play**: bỏ jsonl vào `data/`, set `DATA_FILE`/`DOC_FILE` +
  `MAX_SAMPLES` trong config. HiGOE và DFlash vẫn có pipeline riêng.
- **ROUGE quality**: khi có reference, script sinh text gọi `rouge.add_rouge()`
  và summary gọi `rouge.aggregate_rouge()`.

## Commands

```bash
# Server offline: uv và Python 3.12 phải cài sẵn, package phải nằm trong cache/wheelhouse
bash scripts/setup_venv.sh --offline
bash scripts/setup_venv.sh --check

# Kiểm tra interpreter/import không tải model
FAST_INFER_VENV="$PWD/.venv" "$PWD/.venv/bin/python" scripts/check_shared_env.py

# Chạy 1 baseline
bash scripts/run.sh <baseline>
```

Baseline khả dụng: `eagle3 dflash llmlingua fastkv rocketkv gemfilter
specprefill minference magicdec longspec specextend higoe semantic_selection
flexprefill`.

## Gotchas

- Không có internet: không dùng installer online; model/dataset và mọi direct
  wheel URL phải được mirror/cache sẵn trên server.
- `requirements.txt` chứa một số path server-specific (`deep_ep`, `eviseq`,
  `vllm`) nên clone khác phải mount đúng các artefact đó.
- Model gated (Llama) cần `HF_TOKEN`; ưu tiên snapshot local.
- `flash-attn`, `flashinfer`, Triton và vLLM phải khớp torch/CUDA/GPU.
- Kết quả nằm ở `outputs/` (gitignored), không commit artifact.

## Dev CPU local (máy tuantb@teslaT4) — ĐỌC KỸ TRƯỚC KHI DEBUG

- Máy này **KHÔNG có sudo**; driver 550.163 (max CUDA 12.4) KHÔNG chạy được stack
  cu130 → **T4 chỉ dùng để dev/debug trên CPU**. Số liệu benchmark thật chạy trên
  server cu13. `torch.cuda.is_available()` luôn False; đừng cố sửa GPU trên máy này.
- Venv `.venv/` đã cài qua **`requirements.local.txt`** (bản sao của `requirements.txt`
  đã bỏ path server-specific + package không có trên PyPI: deep_ep, eviseq, vllm file://,
  flashinfer-jit-cache, pyrouge, python-apt, dbus-python, PyGObject, mooncake,
  flash_attn). KHÔNG cài trực tiếp từ `requirements.txt` trên máy này.
- Chạy script ở chế độ CPU (llmlingua có sẵn fallback CPU):
  ```bash
  CUDA_VISIBLE_DEVICES="" DEVICE=cpu SMOKE=1 bash scripts/run_llmlingua.sh
  ```
- **Bug đã sửa — đừng tái lập khi sửa script:**
  - Check Python 3.12: dùng `sys.exit(1) if cond else None`, **KHÔNG** dùng
    `raise SystemExit(...) if cond else None` (bẫy cú pháp → `raise None` → TypeError,
    check luôn fail dù đúng 3.12). Đã sửa trong `runtime.sh` + `setup_venv.sh`.
  - transformers 5.x: `apply_chat_template(..., return_tensors="pt")` mặc định trả
    `BatchEncoding` (không còn tensor thô như 4.x) → **phải thêm `return_dict=False`**
    trước khi dùng `.shape[1]`/`.to(device)`/`model.generate()`.
- `setup_venv.sh --check` sẽ báo lỗi thiếu local requirement sources trên máy này
  (deep_ep/eviseq/vllm wheel) — đó là bình thường, không phải lỗi venv.
- Chi tiết debug CPU + checklist: `docs/cpu_dev_workflow.md`.

Chi tiết cài đặt/infer từng baseline: `docs/README.md` → `docs/baselines/*.md`.
Định dạng dữ liệu: `data/README.md`. Taxonomy/schema: `externals/baseline_repo_guide.md`.
