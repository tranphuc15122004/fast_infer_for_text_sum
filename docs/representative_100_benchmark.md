# Benchmark representative_100 (baseline có adapter dữ liệu)

Script chạy infer cho các baseline trong `externals/` trên bộ dữ liệu đại diện
`data/representative_100/` (4 dataset × 100 mẫu: govreport, multinews,
cnn_dailymail, xsum — xem `data/README.md`) và thu thập **toàn bộ metric**:
tốc độ infer + semantic (ROUGE-1/2/L, ROUGE-Lsum, BLEU-1..4, ...).

## 1. Runner

```bash
# full: 100 mẫu / dataset, model canonical M1-M9 (mặc định trên server GPU lớn)
bash scripts/run_representative_100.sh

# smoke: 5 mẫu / (baseline, dataset), cấu hình T4-safe
bash scripts/run_representative_100.sh --mode smoke

# chạy 1 nhóm baseline / 1 dataset / giới hạn mẫu
bash scripts/run_representative_100.sh --baselines "llmlingua minference" --datasets "cnn_dailymail xsum" --max-samples 20

# xem kế hoạch mà không chạy
bash scripts/run_representative_100.sh --dry-run

# chọn riêng nhóm semantic-selection
bash scripts/run_representative_100.sh --baselines semantic_selection --datasets xsum
```

Các tùy chọn đầy đủ: `--baselines`, `--datasets`, `--max-samples`,
`--max-new-tokens`, `--mode smoke|full`, `--config FILE`, `--output-dir DIR`,
`--include-unsupported`, `--dry-run`, `--skip-collect`. Defaults có thể đặt
trong `config/representative_100.env`. Runner mặc định chỉ chọn baseline có
adapter đọc `representative_100`; `--include-unsupported` chỉ chạy thêm smoke
probe vào thư mục `smoke/`, không được tính vào benchmark/metric.

### Nhóm baseline

| Nhóm | Baseline | Ghi chú |
|---|---|---|
| Đọc dữ liệu representative (chạy mặc định) | llmlingua, fastkv, gemfilter, specprefill, minference, specextend, eagle3, semantic_selection | Runner sinh config full canonical riêng cho từng (baseline, dataset) trong `<output-dir>/configs/` |
| Chưa có adapter representative | higoe, dflash, rocketkv, magicdec, longspec | Chỉ chạy khi có `--include-unsupported`; là smoke probe riêng, không đọc dataset và không được đưa vào metric |

`semantic_selection` chạy cùng target canonical M1 và embedding M6 cho `full`, `random`, `lead`, `tfidf`,
`textrank`, `mmr`; các scheme được tách riêng khi collector tổng hợp metric.
`SpecForge` không chạy trong bảng vì là infrastructure/framework, không phải
algorithmic inference baseline.

Data được convert tự động cho baseline có format riêng (vào
`<output-dir>/data/`):

- **eagle3**: record dạng `{"question_id", "turns": [...], "reference"}`
  (EAGLE chat format).
- **specextend**: record dạng `{"text": "<prompt wrapper + document>"}`
  (đúng format `run_classic.py` đọc).

### Output

Mỗi run ghi `<output-dir>/<baseline>_<dataset>.jsonl` (schema §13) + log
`<output-dir>/logs/<baseline>_<dataset>.log`. Runner in bảng PASS/FAIL và tự
động gọi collector strict ở cuối. Collector chỉ ghi report khi mọi cặp
baseline/dataset có đủ số sample duy nhất theo `--max-samples`; kết quả partial
sẽ làm runner exit khác 0.

## 2. Collector (thu thập metric)

```bash
# tổng hợp thư mục đã chạy đủ ma trận (runner tự truyền các cờ strict)
uv run --project . --locked python scripts/collect_metrics.py

# kiểm tra strict thủ công
uv run --project . --locked python scripts/collect_metrics.py \
  --strict \
  --expected-baselines "llmlingua fastkv gemfilter specprefill minference specextend eagle3 semantic_selection" \
  --expected-datasets "cnn_dailymail govreport multinews xsum" \
  --expected-samples 100

# chỉ 1 file output, ghi report vào chỗ khác
uv run --project . --locked python scripts/collect_metrics.py \
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
`ttft_ms`, `tpot_ms`, `e2e_ms`, `throughput_tok_s`, `qps`, `peak_memory_gb`
+ `retained_ratio`, `compression_ratio` (suy ra) + key speculative
(`avg_accept_length`, `acceptance_rate`, `draft_latency_ms`,
`verification_latency_ms`, `rejected_draft_ratio`).

Mỗi key báo **mean / median / p90 / std**. Key không được baseline ghi
(vd `ttft_ms` của script dùng transformers) tự động vắng mặt — không phải lỗi.

### Metric semantic

Collector join reference theo record id (`doc_id`/`sample_id`/`question_id`/`id`)
từ `data/representative_100/*_representative.jsonl`, tính trên mọi text key
có trong record (`summary`, `text`, `answer`, `gemfilter_text`; `base_text`
của GemFilter dùng prefix `base_`):

- **ROUGE-1/2/L** precision / recall / F1 (`rouge1_p/r/f`, `rouge2_*`, `rougeL_*`)
- **ROUGE-Lsum** (LCS trên chuỗi câu, chuẩn summarization) — `rougeLsum_f`
- **BLEU-1..4** (smoothed + brevity penalty)
- **length_ratio** (hyp_tokens / ref_tokens)

Triển khai: `scripts/common/metrics.py` (pure-Python, chạy được trong mọi uv-env
đang khóa `--locked`). ROUGE base: `scripts/common/rouge.py`.

Lưu ý: baseline không ghi text sinh ra vào record (specextend, rocketkv,
higoe, magicdec, longspec smoke) sẽ không có metric semantic — collector chỉ
báo phần dữ liệu có.

## 3. Kiểm chứng nhanh metric

```bash
uv run --project . --locked python - <<'PY'
from common import metrics
s = metrics.semantic_scores("the cat sat on the mat", "the cat sat on the mat today")
print(s)
PY
```
