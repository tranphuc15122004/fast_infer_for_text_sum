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
