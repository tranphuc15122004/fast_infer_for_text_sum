# Findings — bổ sung baseline representative_100

- `externals/` có các baseline/repo: EAGLE, FastKV, GemFilter, HiGOE,
  LLMLingua, LongSpec, MInference, MagicDec, RocketKV, Sematic_selection,
  SpecExtend, SpecForge, dflash, speculative_prefill.
- Dispatcher hiện đã nối 12 baseline: `eagle3`, `dflash`, `llmlingua`,
  `fastkv`, `rocketkv`, `gemfilter`, `specprefill`, `minference`, `magicdec`,
  `longspec`, `specextend`, `higoe`.
- Runner hiện mặc định chỉ chạy 7 baseline đọc `DATA_FILE`; 5 baseline còn lại
  chỉ chạy với `--include-unsupported`.
- `Sematic_selection/infer.py` nhận JSONL `document/reference`, dùng Qwen3-4B,
  các selector `random lead tfidf textrank mmr`, và tự ghi output JSONL cùng
  summary JSON.
- `SpecForge` là infrastructure theo `externals/baseline_repo_guide.md`,
  không có wrapper inference riêng.
- DFlash dùng dataset GSM8K trong config gốc, không tương thích trực tiếp với
  `representative_100` summarization; cần giữ đường chạy smoke riêng hoặc
  adapter data được xác định rõ.
- Runner mới mặc định gồm 8 baseline đọc dữ liệu/dataset và 5 baseline smoke-only;
  tổng cộng 13 tên baseline inference.
- `SpecForge` vẫn bị loại khỏi runner vì không có wrapper inference độc lập.
- Collector chuẩn hóa schema semantic-selection: `example_id` → `doc_id`,
  `original_tokens`/`selected_tokens` → schema input/retained, và nhóm method
  theo từng selector.

## Infer 1 sample — context hiện tại

- Dữ liệu representative có 4 bộ: `cnn_dailymail`, `govreport`, `multinews`,
  `xsum`; mỗi file có trường `document` và `reference`.
- Cả 9 model đã có snapshot/weight trong cache local.
- `scripts/run_representative_100.sh` mặc định chạy 13 baseline inference;
  nhóm `higoe dflash rocketkv magicdec longspec` chạy smoke riêng vì không
  nhận trực tiếp `DATA_FILE` summarization theo cùng contract.
- `config/semantic_selection.env` vẫn trỏ Qwen3-4B, nên nếu chạy theo bộ model
  đã chốt phải tạo override tạm sang M1, không sửa config gốc trong lượt smoke.
- `eagle3` và `dflash` cũng còn config Qwen3 cũ; cần override model/cặp draft
  cho Llama 3.1 khi chạy full.
- `hf auth whoami` bị lỗi DNS trong sandbox hiện tại, nhưng kiểm tra trực tiếp
  snapshot cho thấy M1 và M2 đã có `config.json`, các model còn lại có weight
  đầy đủ.

## Runtime debug — 2026-08-18

| Baseline | Trạng thái | Bằng chứng chính |
|---|---|---|
| EAGLE-3 | BLOCKED | `EAGLE3 inference requires a visible CUDA GPU` |
| DFlash | BLOCKED | `torch.cuda.set_device` → `No CUDA GPUs are available` |
| LLMLingua | PASS | CPU end-to-end, `[LLMLingua] ALL PASS` |
| FastKV | PASS | TinyLlama/snapkv/sdpa, 1 token/1 run, `[FastKV] ALL PASS` |
| RocketKV | PASS | CPU kernel smoke, `[RocketKV] ALL PASS` |
| GemFilter | PASS | TinyLlama, 1 token/1 run, `[GemFilter] ALL PASS` |
| SpecPrefill | BLOCKED | vLLM `Failed to infer device type` |
| MInference | BLOCKED | import gọi CUDA → `No CUDA GPUs are available` |
| MagicDec | BLOCKED | cache JIT đã tách được; sau đó `get_device_capability` fail vì no CUDA |
| LongSpec | FAIL (GPU) | import pass, Triton forward fail vì no CUDA |
| SpecExtend | TIMEOUT/BLOCKED | không trả control/không ghi output với fixture ngắn trong 180s |
| HiGOE | PASS | CPU Contriever smoke, `[HiGOE] ALL PASS` |
| semantic_selection | PARTIAL | Qwen3-4B CPU load + đủ 6 rows/1 sample; runner exit `124` sau khi ghi output |

### Root causes / notes

- Lượt wrapper đầu tiên của 11 baseline fail trước Python vì wrapper không export `UV_CACHE_DIR`; `uv` cố tạo lock trong cache read-only. Retry dùng cache dưới `/tmp` đã đi vào runtime.
- Host hiện tại không có NVIDIA driver/GPU, nên các baseline CUDA không thể kết luận pass trên GPU.
- MagicDec dùng `FLASHINFER_WORKSPACE_BASE` để chuyển cache JIT; sau khi đặt biến đúng, lỗi còn lại là CUDA.
- Semantic-selection không fail model/inference: log ghi `loaded in 3.20s`, sinh `full` và 5 selector; vấn đề còn lại là process lifecycle/runner timeout.
- Không dùng artifact cũ append trong `outputs/specextend_smoke.jsonl` để kết luận lượt debug mới.
- Verification: `bash -n scripts/run.sh scripts/run_*.sh` pass; artifact/schema assertions pass; root env không có `pytest` nên `uv run ... python -m pytest -q tests` không chạy được (`No module named pytest`).

## GPU runtime debug — Tesla T4 — 2026-08-18

### Infer thật 1 sample đã chạy

| Baseline | Kết quả | Artifact/bằng chứng |
|---|---|---|
| LLMLingua | PASS | runner xsum 1 sample, 15s |
| FastKV | PASS | TinyLlama local, 8 tokens, `FastKV ALL PASS` |
| GemFilter | PASS | TinyLlama local, 8 tokens, `GemFilter ALL PASS` |
| SpecPrefill | PASS | TinyLlama target/draft, 8 tokens, `SpeculativePrefill ALL PASS` |
| MInference | PASS | runner xsum 1 sample, 195s |
| SpecExtend | PASS | runner xsum 1 sample, 95s |
| EAGLE-3 | PASS | runner xsum 1 sample, 30s |
| semantic_selection | PASS | runner xsum 1 sample, 28s |
| DFlash | PASS | GSM8K 1 sample, baseline 18.63 tok/s, DFlash 7.79 tok/s |
| MagicDec | PASS | dense benchmark, returncode 0, timing/output lines |

### Baseline còn thiếu text infer thật

- HiGOE: wrapper hiện chỉ thực hiện Contriever retrieval trên dummy docs; full `eval.py` cần dataset, graph hierarchical, trained weights và LLM/API judge, hiện không có trong checkout.
- RocketKV: wrapper hiện chỉ chạy RocketAttention kernel smoke; upstream full LongBench cần dataset download và model Llama/Mistral local, hiện không có artifact phù hợp.
- LongSpec: wrapper smoke chỉ import/skip kernel trên T4 `sm75`; full inference cần target/draft LongSpec và GPU class 80GB/sm80+, không thể chạy an toàn trên T4 15GB.

### Lượt runner đầu và retry

- Runner đầu của FastKV/GemFilter/SpecPrefill fail do chọn config model gated/weight chưa cache; retry bằng config T4-safe TinyLlama local đã PASS.
- Toàn bộ log/output của lượt GPU nằm trong `outputs/gpu_1sample/`.

## 2026-08-21 — chuyển dataset lớn ra cache

- Cache model hiện dùng `HF_HOME`, mặc định `~/.cache/huggingface`.
- Dữ liệu Git lớn nằm ở `externals/FastKV/data` và `externals/MagicDec/Data`,
  tổng khoảng 1.014 GiB trong HEAD.
- Thiết kế đã được người dùng duyệt: lưu ở
  `${HF_HOME}/datasets/fast_infer_text_sum/{FastKV,MagicDec}/`.
- Repo đang có thay đổi chưa commit ở các file MagicDec; không được reset hoặc
  ghi đè các thay đổi này.

## 2026-08-25 — profile Qwen3-4B long-summary

- User chọn Qwen3-4B target đơn trước; chưa profile EAGLE-3/DFlash.
- GPU mục tiêu: Tesla T4 15,360 MiB; T4 smoke cần dùng FP16 và SDPA.
- Repo đã có số đo Qwen3-4B FP16 + SDPA: peak khoảng 9,744 MB với mean
  1,791.9 source tokens và max output 128 token.
- Các mốc profile dự kiến: 256, 512, 1024, 2048, 3072 từ; mốc cuối có thể
  bị hạ/bỏ nếu input token thực tế hoặc KV cache gây OOM.
- Cần phân biệt model load one-time với per-sample latency; tỷ lệ component chỉ
  tính trên sample path sau warmup.

### Kết quả run trên Tesla T4 — 2026-08-25

- Runtime: Tesla T4 15,360 MiB, PyTorch `2.13.0+cu126`, CUDA available, FP16 + SDPA.
- Model load one-time: khoảng 10.25 s.
- Mỗi mốc chạy 3 repeats, lấy median; output tối đa 128 token; không speculative.
- 5/5 mốc hoàn tất, không OOM. Input thực tế sau chat template là 397 / 746 /
  1,502 / 2,954 / 3,830 tokens cho 256 / 512 / 1,024 / 2,048 / 3,072 từ.
- Decode là thành phần lớn nhất: khoảng 95.3% ở 256 từ, giảm còn 67.3% ở 3,072 từ;
  prefill tăng từ 3.4% lên 31.9%.
- KV cache tăng từ 55.8 MB lên 538.6 MB; peak allocated tăng từ 7.69 GiB lên
  12.42 GiB ở 3,072 từ.
- Artifact canonical: `src/analyze/full_infer/results/{summary.csv,summary.jsonl,metadata.json,*.png}`;
  bản trong `outputs/qwen3_long_profile/` vẫn được giữ nguyên để tương thích.

## 2026-08-26 — shared Python 3.12 migration

- Runtime hiện tại của workspace là Python 3.13.9; không có Python 3.12 executable và uv chỉ báo bản 3.12 là `download available`.
- `requirements.txt` là file untracked do người dùng cung cấp; đã bổ sung `sentence-transformers==5.7.0` vì semantic-selection dùng trực tiếp package này. File có local/direct artefact (`deep_ep`, `eviseq`, `vllm`, `mooncake-transfer-engine`).
- Đã xóa root `.venv` Python 3.11 và các `envs/*/.venv`; giữ manifest/lock legacy vì lớp an toàn chặn xóa toàn bộ env metadata/wheel.
- Tất cả launcher chính dùng `scripts/common/runtime.sh` và trực tiếp gọi `FAST_INFER_PYTHON`.
- MagicDec, SpecExtend, LongSpec và representative runner dùng interpreter kế thừa; representative runner dừng đúng khi runtime thiếu.
- `pytest -q tests`: 46 passed; `bash -n` toàn bộ shell script: pass; compileall scripts/tests: pass.
- `scripts/check_shared_env.py` chạy import-only/offline, kiểm tra dflash/CUDA và vendored LLMLingua; trên workspace báo Python 3.13, thiếu `vllm`/`flashinfer`/`sentence_transformers` và lỗi ABI `flash_attn`.
- `scripts/setup_venv.sh --offline` dừng sớm và đúng vì local wheel `deep_ep` dưới `/vllm-workspace` chưa được mount; sau khi path được cung cấp, gate Python 3.12 vẫn sẽ chặn workspace hiện tại.
- Phát hiện và sửa bug wrapper coi `--smoke` là tên config; sau sửa, sweep 14 baseline đều dừng tại cùng runtime gate với `config_errors=0`.
- Sửa representative dry-run để không cần venv và không gọi Python khi convert/resolve model; test dry-run EAGLE/SpecExtend pass.

## 2026-08-26 — smoke 1 sample trên workspace hiện tại

- Workspace hiện tại chạy Python 3.13.9, không có Python 3.12 và không có NVIDIA driver/GPU.
- `scripts/run.sh <baseline> --smoke` dừng đồng nhất ở shared-runtime gate; đây là blocker môi trường, không phải lỗi riêng baseline.
- Có cache nhiều model HF, nhưng requirements còn cần mount local wheel `/vllm-workspace/...` và các package CUDA/server-specific.

## 2026-08-26 — smoke 1 sample và debug baseline

- Wrapper sweep với `FAST_INFER_PYTHON` mô phỏng Python 3.12 đạt `14/14`, không
  còn lỗi coi `--smoke` là file config; semantic-selection cũng không còn truyền
  cờ lạ xuống upstream và dùng `--limit 1`.
- Smoke thực tế PASS: FastKV trên TinyLlama CPU (modern Llama API), GemFilter trên
  TinyLlama CPU (modern custom attention), RocketKV kernel CPU, Semantic Selection
  trên Qwen2.5-0.5B CPU với 1 document.
- Đã thêm tương thích FastKV cho modern Transformers và Mistral; thêm fallback
  SDPA/position embeddings/cache API cho GemFilter Llama/Mistral/Phi-3.
- Đã chuẩn hóa `--smoke` của DFlash/FastKV/GemFilter/LLMLingua/MInference/
  SpecPrefill/FlexPrefill/SpecExtend/EAGLE-3 thành tối đa đúng 1 sample.
- Các blocker khi chạy trực tiếp workspace hiện tại: Python 3.13.9 thay vì 3.12,
  không có NVIDIA driver; thiếu `vllm`, `flashinfer`, `nltk`, `tilelang`,
  `liger-kernel`, `termcolor`, `faiss`, `dgl`; `flash_attn` lỗi GLIBC ABI.
- `setup_venv.sh --offline` dừng đúng tại local wheel bắt buộc chưa mount:
  `/vllm-workspace/ep_kernels/dist/deep_ep-1.2.1+73b6ea4-cp312-cp312-linux_x86_64.whl`.

## 2026-08-26 — tái tạo uv lock root

- Đã xóa `uv.lock` cũ, đổi `.python-version` từ `3.11` sang `3.12`, và đặt
  `pyproject.toml` yêu cầu `==3.12.*`.
- `scripts/setup_venv.sh` hiện tự chạy `uv add --requirements requirements.txt
  --no-sync --offline --no-python-downloads --python <python3.12>` để tạo/cập nhật
  lock chung trước khi cài venv.
- Không thể sinh lock trong workspace hiện tại: uv báo không tìm thấy Python 3.12;
  sau đó setup cũng sẽ bị chặn bởi local wheel `deep_ep` và editable path server.
