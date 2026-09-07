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
`config/master.path` (mặc định là master trên server B200:
`/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env`)
hoặc override bằng `FAST_INFER_MASTER_CONFIG`. Interpreter được chọn bằng
`FAST_INFER_PYTHON`/`FAST_INFER_VENV`; trên máy local dùng `.venv`:

```bash
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  bash scripts/run_longbench_200.sh \
  --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env \
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

Smoke có guard an toàn mặc định `LONG_BENCH_SMOKE_MAX_INPUT_TOKENS=4096`.
Mẫu đầu của `gov_report` dài khoảng 10k token; với `vanilla_hf` eager
attention, bộ nhớ attention tăng theo `L²` nên chạy nguyên mẫu có thể OOM ngay
cả trên B200. Profile `representative`/`full` vẫn giữ mặc định
`LONG_BENCH_MAX_INPUT_TOKENS=0`; muốn đo context dài phải đặt giới hạn phù hợp
với baseline và VRAM rồi ghi rõ trong manifest.

Runner cũng kiểm tra VRAM trống trước khi tạo process con. Mặc định cần ít nhất
32 GiB trên GPU được chọn (`LONG_BENCH_MIN_FREE_GB=32`). Nếu `nvidia-smi` cho
thấy GPU chỉ còn vài GiB, runner dừng sớm với hướng dẫn chọn GPU khác hoặc tắt
process đang chiếm VRAM; dùng `--min-free-gb 0` chỉ khi đã hiểu rủi ro.

### Compatibility với runtime server

Các adapter đã có lớp tương thích cho đúng stack Python 3.12/Transformers
trên server:

- DFlash và MagicDec tự đăng ký source vendored vào `sys.path`; không cần
  `pip install` hai package này.
- LongSpec và EAGLE-3 xử lý RoPE schema mới của Llama 3.1. EAGLE-3 dùng
  frequency-dependent Llama 3.1 RoPE, không đổi thành linear/dynamic RoPE
  chỉ để né lỗi `KeyError: type`.
- SpecExtend không bắt buộc `termcolor`; FAFO không bắt buộc FastChat và có
  fallback cho các alias Transformers cũ đã bị bỏ.
- SSSD cho phép datastore rỗng khi smoke để kiểm tra wiring prompt/self-output.
  Muốn đo đúng retrieval SSSD phải đặt `SSSD_DATASTORE_PATH` trỏ tới `.idx`
  đã build cho đúng tokenizer/model; nếu path đã khai báo nhưng không tồn tại,
  preflight vẫn dừng cell với `missing_checkpoint`.
- SSSD cần binary wheel `sglang-kernel==0.4.1` khớp CUDA/GPU. Nếu
  `sgl_kernel` không import được, preflight ghi `missing_dependency` và không
  khởi chạy child process để tránh traceback import sâu.

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

### Chọn GPU trên máy nhiều GPU (B200)

Orchestrator chọn GPU theo thứ tự ưu tiên: flag `--gpu-ids` > env
`LONG_BENCH_GPU_IDS` > `FI_GPU_IDS` > `CUDA_VISIBLE_DEVICES`. Giá trị là danh
sách index **vật lý** (phân tách bằng dấu phẩy hoặc space), ví dụ `0`, `2`
hoặc `0,1` khi baseline cần nhiều GPU. GPU được chọn được áp dụng lên
`CUDA_VISIBLE_DEVICES` trước khi torch được import, nên cả orchestrator lẫn
mọi child process đều thấy cùng tập device.

Xem nhanh GPU trên host, lựa chọn hiện tại và mapping torch nhìn thấy (không
cần data, không load model, thoát ngay):

```bash
bash scripts/run_longbench_200.sh --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env --list-gpus
```

Trước khi launch một run tốn GPU, kiểm tra **GPU id + VRAM còn trống** bằng
script chuyên dụng (chỉ dùng `nvidia-smi`, không load model, chạy bằng python3
hệ thống):

```bash
bash scripts/run_gpu_check.sh --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env
```

Báo cáo liệt kê từng GPU vật lý (total/used/free VRAM, utilization, nhiệt độ,
tiến trình đang chiếm), đánh dấu `*` đúng GPU job sẽ dùng (theo
`LONG_BENCH_GPU_IDS` như khi launch thật) và gợi ý GPU trống nhiều nhất. Muốn
chặn job khi không đủ VRAM, dùng ngưỡng `--min-free-gb`:

```bash
bash scripts/run_gpu_check.sh --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env --gpu-ids 3 --min-free-gb 120
# exit 0: GPU 3 đủ VRAM | exit 2: thiếu VRAM (dừng, chọn GPU khác)
bash scripts/run_gpu_check.sh --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env --json /tmp/gpu_report.json  # automation
```

Kiểm tra với đúng ngưỡng mặc định của runner:

```bash
bash scripts/run_gpu_check.sh \
  --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env \
  --gpu-ids 0 --min-free-gb 32
```

File liên quan: `scripts/check_gpu_vram.py` (logic) + `scripts/run_gpu_check.sh`
(wrapper load master profile `longbench`).

Chạy một run trên GPU vật lý số 2 (hai cách tương đương):

```bash
LONG_BENCH_GPU_IDS=2 bash scripts/run_longbench_200.sh --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env --mode full
# hoặc dùng flag CLI (ưu tiên cao nhất, ghi đè mọi env):
bash scripts/run_longbench_200.sh --config /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env --mode full --gpu-ids 2
```

Khi launch, runner in banner `[gpu] ...` cho biết host có bao nhiêu GPU, GPU
được chọn và số device torch nhìn thấy; nếu index yêu cầu không tồn tại trên
host sẽ có cảnh báo trên stderr. Lựa chọn và snapshot đầy đủ (host GPU + bản
đồ visible) được ghi vào `run_manifest.json` ở các field `gpu_ids`/`gpu`. Khi
`LONG_BENCH_DEVICE=cpu` (dev CPU) GPU vẫn được ghi lại nhưng compute chạy CPU.

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
