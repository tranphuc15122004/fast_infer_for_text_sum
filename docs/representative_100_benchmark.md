# Benchmark representative_100 (baseline có adapter dữ liệu)

Script chạy infer cho các baseline trong `externals/` trên bộ dữ liệu đại diện
`data/representative_100/` (4 dataset × 100 mẫu: govreport, multinews,
cnn_dailymail, xsum — xem `data/README.md`) và thu thập **toàn bộ metric**:
tốc độ infer + semantic (ROUGE-1/2/L, ROUGE-Lsum, BLEU-1..4, ...).

## 1. Chuẩn bị và runner

Trên server B200 offline, dùng trực tiếp Python 3.12 từ PATH; không cần tạo hoặc
activate virtualenv:

```bash
python3 --version
set -a
source config/b200.env
set +a
python3 scripts/check_b200_env.py --json outputs/b200_preflight.json
```

Local simulation dùng `.venv` bằng cách đặt `FAST_INFER_PYTHON` rõ ràng.

```bash
# full: 100 mẫu / dataset, model canonical M1-M9 (mặc định trên server GPU lớn)
FAST_INFER_PYTHON=python3 bash scripts/run_representative_100.sh

# smoke: 5 mẫu / (baseline, dataset), cấu hình thận trọng theo từng baseline
FAST_INFER_PYTHON=python3 bash scripts/run_representative_100.sh --mode smoke

# chạy 1 nhóm baseline / 1 dataset / giới hạn mẫu
bash scripts/run_representative_100.sh --baselines "llmlingua minference" --datasets "cnn_dailymail xsum" --max-samples 20

# xem kế hoạch mà không chạy
bash scripts/run_representative_100.sh --dry-run

# chọn riêng nhóm semantic-selection
bash scripts/run_representative_100.sh --baselines semantic_selection --datasets xsum
```

Smoke toàn bộ 14 baseline với profile B200, một sample/baseline:

```bash
set -a; source config/b200.env; set +a
bash scripts/run_b200_smoke.sh --output-dir outputs/b200_smoke
```

Trên local T4, thay `python3` bằng
`FAST_INFER_PYTHON="$PWD/.venv/bin/python"`; nếu không có CUDA, các baseline
GPU-only phải xuất hiện là `BLOCKED`, không phải `PASS` giả.

Các tùy chọn đầy đủ: `--baselines`, `--datasets`, `--max-samples`,
`--max-new-tokens`, `--mode smoke|full`, `--config FILE`, `--output-dir DIR`,
`--include-unsupported`, `--dry-run`, `--skip-collect`. Defaults có thể đặt
trong `config/representative_100.env`. Runner mặc định chỉ chọn baseline có
adapter đọc `representative_100`; `--include-unsupported` chỉ chạy thêm smoke
probe vào thư mục `smoke/`, không được tính vào benchmark/metric.

### Nhóm baseline

| Nhóm | Baseline | Ghi chú |
|---|---|---|
| Đọc dữ liệu representative (chạy mặc định) | llmlingua, fastkv, gemfilter, specprefill, minference, specextend, eagle3, semantic_selection, dflash, longspec | Runner sinh config full canonical riêng cho từng (baseline, dataset) trong `<output-dir>/configs/` |
| Chưa có adapter representative | higoe, rocketkv, magicdec | Chỉ chạy khi có `--include-unsupported`; là smoke probe riêng, không đọc dataset và không được đưa vào metric |

`semantic_selection` chạy cùng target canonical M1 và embedding M6 cho `full`, `random`, `lead`, `tfidf`,
`textrank`, `mmr`; các scheme được tách riêng khi collector tổng hợp metric.
`SpecForge` không chạy trong bảng vì là infrastructure/framework, không phải
algorithmic inference baseline.

Data được convert tự động cho baseline có format riêng (vào
`<output-dir>/data/`):

- **eagle3**: record dạng `{"question_id", "turns": [...], "reference"}`
  (EAGLE chat format).
- **specextend**: record dạng `{"text": "<prompt wrapper + document>"}`
  (đúng format `run_eagle.py` đọc), chạy target M1 + EAGLE-3 draft M3.
- **dflash**: đọc trực tiếp unified `{id, document, reference}` JSONL và dùng
  target `Llama-3.1-8B-Instruct` cùng draft `LLaMA3.1-8B-Instruct-DFlash-UltraChat`.
- **longspec**: đọc trực tiếp unified JSONL và dùng cặp chính thức
  `lmsys/vicuna-7b-v1.5-16k` + `sail/longspec-vicuna-7b-v1.5-16k`.

### Output

Mỗi run ghi `<output-dir>/<baseline>_<dataset>.jsonl` (schema §13) + log
`<output-dir>/logs/<baseline>_<dataset>.log`. Runner in bảng PASS/FAIL và tự
động gọi collector strict ở cuối. Collector chỉ ghi report khi mọi cặp
baseline/dataset có đủ số sample duy nhất theo `--max-samples`; kết quả partial
sẽ làm runner exit khác 0.

## 2. Collector (thu thập metric)

```bash
# tổng hợp thư mục đã chạy đủ ma trận (runner tự truyền các cờ strict)
"$FAST_INFER_PYTHON" scripts/collect_metrics.py

# kiểm tra strict thủ công
"$FAST_INFER_PYTHON" scripts/collect_metrics.py \
  --strict \
  --expected-baselines "llmlingua fastkv gemfilter specprefill minference specextend eagle3 semantic_selection dflash longspec" \
  --expected-datasets "cnn_dailymail govreport multinews xsum" \
  --expected-samples 100

# chỉ 1 file output, ghi report vào chỗ khác
"$FAST_INFER_PYTHON" scripts/collect_metrics.py \
  --outputs-dir outputs/representative_100/llmlingua_cnn_dailymail.jsonl \
  --out /tmp/m.json --csv /tmp/m.csv --md /tmp/m.md
```

Sinh ra trong `<output-dir>`:

| File | Nội dung |
|---|---|
| `metrics_summary.json` | đầy đủ, per (dataset × method): `speed` (mean/median/p90/std), `speculative`, `semantic` (mean) + `overall` gộp mọi dataset |
| `metrics_summary.csv` | bảng rộng 1 dòng / (dataset, method), cột `metric_stat` |
| `metrics_summary.md` | báo cáo markdown đọc được (tốc độ + semantic theo dataset và overall) |

### Metric tốc độ (schema §13)

`input_tokens`, `retained_tokens`, `output_tokens`, `selector_latency_ms`,
`prefill_ms`, `decode_ms`, `ttft_ms`, `tpot_ms`, `e2e_ms`, `pipeline_e2e_ms`,
`throughput_tok_s`, `qps`, `peak_memory_gb`
+ `retained_ratio`, `compression_ratio` (suy ra) + key speculative
(`avg_accept_length`, `acceptance_rate`, `draft_latency_ms`,
`verification_latency_ms`, `rejected_draft_ratio`).

Mỗi key báo **mean / median / p90 / std**. Key không được baseline ghi
(vd `ttft_ms` của script dùng transformers) tự động vắng mặt — không phải lỗi.

### Paired speedup

Khi mỗi record có timing của dense/reference run tương ứng, collector thêm
`speedup` vào JSON và các cột `*_ratio` vào CSV:

- **ESR** (`esr`): `mean(dense_e2e_ms) / mean(method_pipeline_e2e_ms)`;
- **DSR** (`dsr`): `mean(dense_decode_ms) / mean(method_decode_ms)`;
- **Prefill speedup** (`prefill_speedup`):
  `mean(dense_prefill_ms) / mean(method_prefill_ms)`;
- **TTFT speedup** (`ttft_speedup`):
  `mean(dense_ttft_ms) / mean(method_pipeline_ttft_ms)`.

Các tỷ số dùng ratio của mean trên những record có đủ hai vế, không phải mean
của các tỷ số từng record. Giá trị lớn hơn `1.0` nghĩa là method nhanh hơn.
Nếu baseline không đo được component tương ứng, metric đó được bỏ qua thay vì
coi timing thiếu là `0`.

Các adapter hiện có paired reference cho GemFilter, EAGLE-3, semantic
selection, DFlash và LongSpec; LLMLingua đo thêm dense target E2E. Các
baseline chỉ ghi một E2E chung sẽ chỉ có ESR khi có `dense_e2e_ms`, còn
DSR/prefill cần instrument riêng.

### Metric semantic

Collector join reference theo record id (`doc_id`/`sample_id`/`question_id`/`id`)
từ `data/representative_100/*_representative.jsonl`, tính trên mọi text key
có trong record (`summary`, `text`, `answer`, `gemfilter_text`; `base_text`
của GemFilter dùng prefix `base_`):

- **ROUGE-1/2/L** precision / recall / F1 (`rouge1_p/r/f`, `rouge2_*`, `rougeL_*`)
- **ROUGE-Lsum** (LCS trên chuỗi câu, chuẩn summarization) — `rougeLsum_f`
- **BLEU-1..4** (smoothed + brevity penalty)
- **length_ratio** (hyp_tokens / ref_tokens)

Triển khai: `scripts/common/metrics.py` (pure-Python, chạy trực tiếp trong venv
chung). ROUGE base: `scripts/common/rouge.py`.

Lưu ý: baseline không ghi text sinh ra vào record (specextend, rocketkv,
higoe, magicdec và LongSpec kernel smoke độc lập) sẽ không có metric semantic;
LongSpec representative adapter có ghi text và được tính quality.

## 3. Kiểm chứng nhanh metric

```bash
"$FAST_INFER_PYTHON" - <<'PY'
from common import metrics
s = metrics.semantic_scores("the cat sat on the mat", "the cat sat on the mat today")
print(s)
PY
```
