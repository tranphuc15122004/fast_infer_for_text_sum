# Benchmark LongBench canonical 1.000 mẫu

## Mục đích

`data/longbench_200/` là test set cố định dùng chung cho các lần đo inference.
Mọi output phải giữ `id` của record đầu vào để ghép tốc độ và chất lượng theo
cùng request.

## Thành phần

| Dataset | Task type | Source test | Chọn vào benchmark |
|---|---|---:|---:|
| `gov_report` | `summarization` | 200 | 200 |
| `qmsum` | `summarization` | 200 | 200 |
| `multi_news` | `summarization` | 200 | 200 |
| `lcc` | `code_completion` | 500 | 200 |
| `repobench-p` | `code_completion` | 500 | 200 |

Ba task đầu giữ toàn bộ test set LongBench. LCC và RepoBench-P được xếp theo
`input_tokens`, chia 5 bin và lấy 40 mẫu/bin với seed 42. Danh sách ID không
được random lại trong lúc chạy baseline.

## Build offline

Source phải là mirror local có năm file `<dataset>.jsonl`, mỗi dòng là record
LongBench gốc. Tokenizer phải là cùng tokenizer dùng khi benchmark; không dùng
`len(text.split())`.

```bash
python scripts/build_longbench_200.py \
  --source-dir /path/to/LongBench \
  --tokenizer /path/to/Meta-Llama-3.1-8B-Instruct \
  --output-dir data/longbench_200 --seed 42
python scripts/validate_longbench_200.py \
  --data-dir data/longbench_200 --expected-count 200
python scripts/analyze_longbench_200.py \
  --data-dir data/longbench_200 --spot-checks 2
```

Builder chỉ đọc local JSONL, từ chối ghi đè thư mục không rỗng nếu thiếu
`--force`, ghi manifest checkpoint sau mỗi dataset và lưu checksum file.

## Xem dữ liệu trực quan

Dùng script sau để xem bảng tổng quan và sample preview của từng dataset:

```bash
python scripts/show_longbench_200.py --samples 1
```

Script hiển thị `context`, `input`, `reference_output`, số token và
`length_bin`; text dài được rút gọn để terminal dễ đọc. Có thể chọn riêng
dataset bằng cách lặp `--dataset`, đồng thời chỉnh `--context-chars` và
`--field-chars`.

## Schema chung

Field bắt buộc: `id`, `dataset`, `source_split`, `source_index`, `task_type`,
`context`, `input`, `answers`, `reference_output`, `input_tokens`,
`length_bin`. `metadata` giữ các field tùy chọn như `language` và
`all_classes` của source.

Prompt không lưu trong raw record. `scripts/common/data_loader.py` tự render
prompt từ `scripts/common/longbench_prompts.json` khi record canonical được đưa qua
loader chung; các adapter đọc trực tiếp `context` cần dùng cùng renderer.

## Metric

- `summarization`: ROUGE-1/2/L và các metric semantic hiện có.
- `code_completion`: `code_exact_match` và `code_edit_similarity` sau khi
  chuẩn hóa line ending, trailing whitespace và code fence.
- Tốc độ vẫn dùng các field chung như `input_tokens`, `retained_tokens`,
  `ttft_ms`, `e2e_ms`, `throughput_tok_s`.

Collector đọc cả tên file canonical `<dataset>.jsonl` và tên legacy
`<dataset>_representative.jsonl`; mặc định mới là `data/longbench_200`.

## Chạy baseline

Ví dụ chạy một dataset bằng loader chung:

```bash
DATA_INPUT=data/longbench_200/gov_report.jsonl \
RUN_SAMPLES=200 \
bash scripts/run.sh <baseline>
```

Trước khi chạy toàn ma trận, kiểm tra baseline có adapter phù hợp với
`task_type`. Không dùng prompt summarization cho LCC/RepoBench-P và không đưa
ROUGE vào báo cáo code-completion.

## Orchestrator 3 profile

Toàn bộ ma trận dùng một master shell-env ngoài repository, được trỏ bởi
`config/master.path` hoặc override bằng `FAST_INFER_MASTER_CONFIG`. Interpreter
được chọn bằng `FAST_INFER_PYTHON`/`FAST_INFER_VENV`; trên máy local dùng `.venv`:

```bash
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  bash scripts/run_longbench_200.sh \
  --config /workspace/shared_storage/config/fast_infer_master.env \
  --mode smoke
```

Các profile có ý nghĩa sau:

| Profile | Phạm vi mặc định | Chính sách |
|---|---|---|
| `smoke` | 9 baseline × 5 dataset × 1 mẫu | CPU/T4 chỉ preflight; B200 chạy inference ngắn |
| `representative` | 9 baseline × `gov_report,lcc` × 20 mẫu | cần CUDA; 20 mẫu giữ phân tầng 5 length-bin |
| `full` | 9 baseline × 5 dataset × 200 mẫu | cần CUDA; strict mặc định |

`--preflight-only` tạo đủ file status và manifest mà không load model. Đây là
chế độ phù hợp để kiểm tra máy T4/CPU. Không được diễn giải
`unsupported_cpu`, `missing_checkpoint`, `missing_dependency` hoặc
`unsupported_dataset` thành số đo tốc độ; các field timing của chúng là
`null`.

Ví dụ kiểm tra đầy đủ pipeline local:

```bash
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  "$PWD/.venv/bin/python" scripts/run_longbench_200.py \
  --mode smoke --preflight-only \
  --baselines "vanilla_hf vanilla_fa magicdec longspec eagle3 dflash specextend sssd fafo" \
  --datasets "gov_report qmsum multi_news lcc repobench-p" \
  --output-dir /tmp/longbench_smoke

python scripts/collect_metrics.py \
  --outputs-dir /tmp/longbench_smoke/<run_id> \
  --data-dir data/longbench_200
```

Mỗi run lưu `run_manifest.json`, input subset bất biến, log child process và
`<baseline>/<dataset>.jsonl`. Record thành công có input/output tokens,
model-load/prefill/TTFT/decode/E2E, TPOT, throughput, QPS, peak GPU memory,
dtype/backend, seed và cấu hình generation. Collector tính ESR/DSR sau khi có
đủ cặp timing; không suy ra metric từ status record.

MagicDec dùng nhánh canonical bổ sung trong `infer_magicdec.py`, gọi trực tiếp
SnapKV engine với checkpoint `.pth` đã convert và tokenizer của model. Vì vậy
master phải khai báo `LONG_BENCH_MAGICDEC_MODEL_PTH` cho checkpoint tương ứng;
thiếu checkpoint/dependency sẽ thành status lỗi rõ ràng. SSSD và FAFO được ghi
`scope=aggregate` khi upstream chỉ trả timing gộp; không nhân bản timing đó cho
từng sample.
