# Benchmark LongBench canonical 500 mẫu, tối đa 14k token

## Mục đích

`data/longbench_100_14k/` là test set cố định dùng chung cho các lần đo inference.
Mọi output phải giữ `id` của record đầu vào để ghép tốc độ và chất lượng theo
cùng request.

## Thành phần

| Dataset | Task type | Source test | Chọn vào benchmark |
|---|---|---:|---:|
| `gov_report` | `summarization` | 200 | 100 |
| `qmsum` | `summarization` | 200 | 100 |
| `multi_news` | `summarization` | 200 | 100 |
| `lcc` | `code_completion` | 500 | 100 |
| `repobench-p` | `code_completion` | 500 | 100 |

Ba task đầu giữ toàn bộ test set LongBench. LCC và RepoBench-P được xếp theo
`input_tokens`, giới hạn 14k token, chia 5 bin và lấy 20 mẫu/bin với seed 42. Danh sách ID không
được random lại trong lúc chạy baseline.

## Build offline

Source phải là mirror local có năm file `<dataset>.jsonl`, mỗi dòng là record
LongBench gốc. Tokenizer phải là cùng tokenizer dùng khi benchmark; không dùng
`len(text.split())`.

```bash
python scripts/build_longbench_200.py \
  --source-dir /path/to/LongBench \
  --tokenizer /path/to/Meta-Llama-3.1-8B-Instruct \
  --output-dir data/longbench_100_14k --samples-per-dataset 100 \
  --max-input-tokens 14000 --allow-partial --seed 42
python scripts/validate_longbench_200.py \
  --data-dir data/longbench_100_14k --expected-count 100
python scripts/analyze_longbench_200.py \
  --data-dir data/longbench_100_14k --spot-checks 2
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
`<dataset>_representative.jsonl`; mặc định mới là `data/longbench_100_14k`.

## Chạy baseline

Ví dụ chạy một dataset bằng loader chung:

```bash
DATA_INPUT=data/longbench_100_14k/gov_report.jsonl \
RUN_SAMPLES=100 \
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
| `smoke` | 7 baseline × 5 dataset × 1 mẫu | CPU/T4 chỉ preflight; B200 chạy inference ngắn |
| `representative` | 7 baseline × `gov_report,lcc` × 20 mẫu | cần CUDA; 20 mẫu giữ phân tầng 5 length-bin |
| `full` | 7 baseline × 5 dataset × 100 mẫu | cần CUDA; strict mặc định |

LongSpec và SSSD không nằm trong ma trận này: LongSpec hiện không chạy offline
đầy đủ và SSSD còn thiếu native extension. Adapter standalone vẫn được giữ để
debug riêng. Representative/full mặc định sinh tối đa 2048 token; smoke chỉ
dùng tối đa 8 token để kiểm tra wiring nhanh. Có thể override bằng
`--max-new-tokens` hoặc `LONG_BENCH_MAX_NEW_TOKENS`.

Vanilla HF và Vanilla FA đều chạy target-only với `batch_size=1`. Khi cả hai
đã có output, các baseline speculative (`eagle3`, `dflash`, `specextend`)
không chạy thêm naive/reference inference; runner mặc định dùng Vanilla FA,
fallback sang Vanilla HF, rồi join timing theo `sample_id`. Các trường được
ghi là `external_decode_speedup`/`external_e2e_speedup` với
`speedup_scope=external_reference`. Có thể chọn reference bằng
`LONG_BENCH_REFERENCE_BASELINE=vanilla_hf`.

`--preflight-only` tạo đủ file status và manifest mà không load model. Đây là
chế độ phù hợp để kiểm tra máy T4/CPU. Không được diễn giải
`unsupported_cpu`, `missing_checkpoint`, `missing_dependency` hoặc
`unsupported_dataset` thành số đo tốc độ; các field timing của chúng là
`null`.

### Theo dõi log trong runtime

Runner tạo `run_manifest.json` và thư mục `logs/` ngay khi bắt đầu run. Trước
mỗi baseline/dataset, terminal in đường dẫn live log tương ứng, ví dụ:

```text
outputs/longbench_100_14k/<run-id>/logs/vanilla_hf_gov_report.log
```

Child process được stream đồng thời ra terminal và file log; không cần chờ
baseline kết thúc. Có thể theo dõi một cell bằng:

```bash
tail -F outputs/longbench_100_14k/<run-id>/logs/vanilla_hf_gov_report.log
```

Log giữ nguyên output gốc của baseline, còn terminal thêm prefix
`[<baseline>_<dataset>]` để phân biệt cell. Runner đặt `PYTHONUNBUFFERED=1`
cho child Python và vẫn áp dụng timeout của master config. Nếu baseline bị
treo, xem log live trước khi timeout; sau timeout trạng thái và `log_tail`
được ghi vào `run_manifest.json`.

Với `vanilla_hf` và `vanilla_fa`, log từng mẫu có thêm
`prefill_ms`, `decode_ms`, `decode_tok_s`, `cache` và `attn`. Hai baseline này dùng
`StaticCache` khi Transformers hỗ trợ, đồng thời cấp phát attention mask một
lần cho cả request. `cache=static` là đường chạy tối ưu; nếu thấy
`cache=generate` thì runtime đã rơi về compatibility fallback và cần kiểm tra
version Transformers trước khi benchmark dài. `attn` cho biết backend thực tế
được model resolve, giúp phát hiện trường hợp `vanilla_fa` được yêu cầu nhưng
không chạy bằng FlashAttention-2.

### Tổng hợp metric tự động trong run

Mỗi run không chỉ ghi raw JSONL: khi run kết thúc (không phải
`--preflight-only`), orchestrator **tự chạy collector** trên chính run đó và
ghi vào run dir ba file tổng hợp `metrics_summary.json` (đầy đủ),
`metrics_summary.csv` (bảng rộng) và `metrics_summary.md` (báo cáo đọc được).
Kết quả bước tổng hợp (status, exit code, đường dẫn file, log) được ghi vào
`run_manifest.json` ở field `aggregate`; nếu run sạch (không cell nào fail),
collector chạy ở chế độ `--strict` để xác nhận đủ 100 mẫu cho mỗi
(baseline, dataset) — thiếu mẫu sẽ làm exit code của run khác 0.

- Tắt tổng hợp tự động khi cần: `--no-collect` (hoặc `LONG_BENCH_COLLECT=0`).
  Muốn chạy lại tay: `python scripts/collect_metrics.py --outputs-dir <run_dir>
  --data-dir data/longbench_100_14k`.
- `--preflight-only` luôn bỏ qua tổng hợp (chỉ có status rows, không có
  inference records); để kiểm tra pipeline local đầy đủ phải chạy collector
  tay như ví dụ bên dưới.
- Tổng hợp là best-effort: collector fail không làm mất raw JSONL đã ghi, chỉ
  được ghi nhận trong manifest và (khi chạy strict) làm exit code khác 0.

Smoke có guard an toàn mặc định `LONG_BENCH_SMOKE_MAX_INPUT_TOKENS=4096`.
Mẫu đầu của `gov_report` dài khoảng 10k token; với `vanilla_hf` eager
attention, bộ nhớ attention tăng theo `L²` nên chạy nguyên mẫu có thể OOM ngay
cả trên B200. Profile `representative`/`full` vẫn giữ mặc định
`LONG_BENCH_MAX_INPUT_TOKENS=0`; muốn đo context dài phải đặt giới hạn phù hợp
với baseline và VRAM rồi ghi rõ trong manifest.

Runner dùng `LONG_BENCH_SEED` (mặc định `42`) cho việc chọn sample và truyền
tiếp vào mọi adapter target/drafter. Các adapter reset Python/NumPy/Torch RNG
trước mỗi generation tương ứng; `LONG_BENCH_TEMPERATURE=0` giữ decoding ở chế
độ greedy. Seed chung không đảm bảo bit-identical giữa các kernel/precision
khác nhau, nhưng loại bỏ khác biệt do sampling RNG.

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
- SSSD cần binary wheel `sglang-kernel==0.4.2` khớp Torch 2.11/CUDA/GPU và
  package `gguf==0.19.0`. Nếu
  `sgl_kernel` không import được, preflight ghi `missing_dependency` và không
  khởi chạy child process để tránh traceback import sâu.

Ví dụ kiểm tra đầy đủ pipeline local (preflight-only nên **không** tự tổng
hợp; phải chạy collector tay):

```bash
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  "$PWD/.venv/bin/python" scripts/run_longbench_200.py \
  --mode smoke --preflight-only \
  --baselines "vanilla_hf vanilla_fa magicdec longspec eagle3 dflash specextend sssd fafo" \
  --datasets "gov_report qmsum multi_news lcc repobench-p" \
  --output-dir /tmp/longbench_smoke

python scripts/collect_metrics.py \
  --outputs-dir /tmp/longbench_smoke/<run_id> \
  --data-dir data/longbench_100_14k
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

### Data-parallel: batch size 1 trên nhiều GPU

Mặc định runner chạy tuần tự từng cell `(baseline, dataset)` trong **một**
process batch size 1. Muốn dùng nhiều card cho cùng một cell (vẫn batch size 1
trên mỗi card), bật `--data-parallel`:

```bash
# 4 card, mỗi card 1 shard batch size 1
bash scripts/run_longbench_200.sh --config <master> --mode full \
  --gpu-ids 0,1,2,3 --data-parallel

# hoặc qua env trong master: LONG_BENCH_DATA_PARALLEL=1 + LONG_BENCH_GPU_IDS=0,1,2,3
```

Cơ chế:

- Pool GPU lấy từ `--gpu-ids` / `LONG_BENCH_GPU_IDS` / `FI_GPU_IDS` /
  `CUDA_VISIBLE_DEVICES`; nếu không khai báo thì dùng toàn bộ device torch
  nhìn thấy. Pool được chia thành các **nhóm device**, mỗi nhóm 1 shard
  (`--dp-gpus-per-shard N`, mặc định 1 = 1 card/shard; `N>1` dành cho baseline
  cần song song trong cùng process). Số card phải chia hết cho `N`.
- Mỗi cell được chia thành N shard, cân bằng theo **độ dài input token**
  (greedy longest-first, deterministic) chứ không chia đều số mẫu, vì mỗi row
  LongBench dài ngắn rất khác nhau.
- Mỗi shard là một child process riêng, `CUDA_VISIBLE_DEVICES` chỉ chứa đúng
  nhóm GPU của nó, đọc file input riêng
  (`<run>/inputs/shards/<dataset>.dp<k>.jsonl`) và ghi output riêng
  (`<run>/<baseline>/shards/<dataset>.dp<k>.jsonl`, log
  `<run>/logs/<baseline>_<dataset>.dp<k>.log`, stdout prefix
  `[<baseline>_<dataset>.dp<k>]`). Các shard chạy **đồng thời**; timeout của
  cell áp cho cả cell như trước.
- Sau khi tất cả shard xong, runner gộp thành file canonical
  `<run>/<baseline>/<dataset>.jsonl` (đúng thứ tự mẫu gốc) và ghi một summary
  duy nhất, trong đó `shard_count` là số shard của cell còn `shards_merged` là
  số file shard thực sự gộp (khác nhau khi có shard lỗi, kèm `shards_missing`).
  Summary của từng shard được giữ nguyên trong field `shard_summaries` (không
  cộng/trung bình các key đặc thù của từng baseline để tránh tạo ra metric
  giả); metric canonical vẫn do `collect_metrics.py` tính từ các sample row.
- Manifest ghi `data_parallel`, `dp_world_size`, `dp_gpu_groups`,
  `dp_gpus_per_shard` và danh sách `shards` (GPU, số mẫu, status, log, elapsed)
  trong từng cell.

Điểm cần lưu ý:

- DP chỉ giảm **wall-clock**; mỗi sample vẫn chạy batch size 1 trên một card nên
  latency/throughput mỗi sample giữ nguyên ý nghĩa. Model được load một lần cho
  mỗi shard → cần N× VRAM; guard `--min-free-gb` được kiểm tra trên **từng** GPU
  được chọn.
- FAFO và SSSD chỉ trả `scope=aggregate` cho cả process: khi bật DP, mỗi shard
  báo aggregate cho riêng batch của nó. Summary gộp đánh dấu
  `shard_aggregate_semantics: "per_shard"` — **không** so sánh throughput của
  shard nhỏ với aggregate của cả cell chạy tuần tự.
- Cell chỉ `success` khi **mọi** shard thành công. Shard lỗi/timeout làm cell
  `failed` (record của các shard còn lại vẫn được gộp và giữ để điều tra), và
  chế độ `--strict` sẽ báo cell thiếu mẫu.
- Trên máy không có CUDA (dev CPU) hoặc khi chỉ có 1 nhóm GPU, runner in cảnh
  báo `[parallel] ... inactive` rồi tự fallback về 1 process batch size 1 như cũ.

#### Tận dụng VRAM: nhiều process trên mỗi card + ngân sách an toàn

Một cell chỉ dùng ~20-25 GiB/card (batch 1, context ≤14k), nên card 180 GiB còn
rất nhiều chỗ trống. Muốn dùng phần đó để rút ngắn sweep, cho nhiều process
batch-1 **chia sẻ** một card:

```bash
# 4 card × 4 process/card = 16 shard song song, trần sử dụng 170 GiB/card
bash scripts/run_longbench_200.sh --config <master> --mode full \
  --gpu-ids 0,1,2,3 --data-parallel --dp-processes-per-gpu 4

# xem trước kế hoạch trên host mà không chạy gì
bash scripts/run_longbench_200.sh --config <master> --list-gpus \
  --data-parallel --dp-processes-per-gpu 4
```

Cách tính concurrency (mỗi lần cell bắt đầu, dựa trên `nvidia-smi` sống):

```text
usable    = min(--vram-budget-gb, tổng VRAM thật của card) - --vram-headroom-gb
K         = min(--dp-processes-per-gpu, floor((usable - VRAM đang bị chiếm) / --child-vram-gb))
```

Với mặc định `budget=170`, `headroom=10`, `child-vram-gb=40` trên card 180 GiB
đang trống → `K = 3..4` process/card; trần sử dụng luôn ≤170 GiB vì planner
không bao giờ xếp quá `usable`. Nếu muốn pack dày hơn, giảm `--child-vram-gb`
(ví dụ 25 với Llama-3.1-8B + DFlash ở 14k) — nhưng chỉ nên làm **sau khi** xem
`peak_memory_gb` đo được của cell đầu tiên.

**Cơ chế bảo đảm không OOM và không job nào bị kill giữa chừng:**

1. Planner chỉ launch khi phần bộ nhớ đó thực sự trống; `--vram-budget-gb` bị
   cap bởi tổng VRAM của card nên không thể "tiêu" ngân sách của card lớn trên
   card nhỏ.
2. Nếu không đủ chỗ cho dù chỉ 1 child, runner **chờ** (`--vram-wait-seconds`,
   mặc định 600s) và poll lại, thay vì launch liều rồi OOM.
3. Hết thời gian chờ, runner ghi cell đó là `vram_blocked` (chỉ cell đó; sweep
   vẫn chạy tiếp) và **không kill** bất kỳ process nào — kể cả process của
   người khác đang chiếm card.
4. Nếu shard vẫn OOM (reserve thấp hơn thực tế), runner **retry riêng shard đó**
   sau khi các shard anh em đã thoát (`--oom-retries`, mặc định 1, chạy lần lượt
   từng shard) và dùng output của lần retry để gộp. Mọi lần thử được ghi ở
   `shards[].retries`, `oom_retry_rounds`, `retried_shards` trong manifest; file
   partial của lần OOM được giữ lại nhưng **không** được gộp.
5. `--vram-budget-gb 0` tắt hẳn planner (không đọc `nvidia-smi`, không chờ) và
   dùng đúng `--dp-processes-per-gpu`; khi đó retry OOM là lưới an toàn duy nhất.

**Hệ quả đo lường:** `K>1` làm nhiều process tranh chấp SM/băng thông bộ nhớ
trên cùng card, nên latency/throughput từng sample **không còn so sánh được**
với run 1 process/card. Runner in warning khi bắt đầu, gắn
`shared_gpu_concurrency` vào từng record và `measurement_note` vào summary. Chỉ
dùng `K>1` khi ưu tiên wall-clock (ví dụ chạy full matrix), không dùng cho số
liệu head-line.

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
