# Thiết kế pipeline infer LongBench cho 9 baseline

## Mục tiêu

Xây pipeline inference dùng Llama 3.1 8B Instruct trên năm dataset canonical
trong `data/longbench_200`, cho phép chạy ba mức `smoke`, `representative` và
`full` trên cùng môi trường `.venv`/master config. Pipeline phải ghi đủ số
liệu thô để tính speedup, ESR, DSR và các metric tổng hợp ở bước sau.

Chín baseline trong phạm vi:

```text
vanilla_hf vanilla_fa magicdec longspec eagle3 dflash specextend sssd fafo
```

## Nguyên tắc thực nghiệm

- Independent variable: baseline/attention hoặc speculative/KV strategy.
- Control variables: model target Llama 3.1 8B Instruct, dataset và sample ID,
  seed, temperature, batch size, max new tokens, warmup policy, device và
  output schema.
- `vanilla_hf` dùng Hugging Face Transformers với PyTorch eager attention.
- `vanilla_fa` dùng cùng model/generation path nhưng
  `flash_attention_2`; thiếu đúng flash-attn thì fail rõ ràng, không fallback.
- Không đo thời gian load model vào latency per-sample; ghi riêng
  `model_load_ms` và environment metadata.
- Không sinh số liệu giả. CPU/T4 không chạy giả lập GPU performance cho
  baseline CUDA-only.
- `answers`/`reference_output` chỉ dùng cho quality evaluation, tuyệt đối
  không đưa vào prompt model.

## Entry point và master config

Entry point chính là `scripts/run_longbench_200.sh`. Wrapper đọc
`FAST_INFER_MASTER_CONFIG` hoặc pointer `config/master.path`, gọi
`fast_infer_load_config longbench`, kiểm tra runtime Python 3.12 rồi gọi
`scripts/run_longbench_200.py`.

Master config là nguồn mặc định duy nhất:

```bash
LONG_BENCH_DATA_DIR="data/longbench_200"
LONG_BENCH_OUTPUT_DIR="outputs/longbench_200"
LONG_BENCH_MODEL="$MODEL_TARGET"
LONG_BENCH_BASELINES="vanilla_hf vanilla_fa magicdec longspec eagle3 dflash specextend sssd fafo"
LONG_BENCH_DATASETS="gov_report qmsum multi_news lcc repobench-p"
LONG_BENCH_MODE="smoke"
LONG_BENCH_SMOKE_SAMPLES=1
LONG_BENCH_REPRESENTATIVE_SAMPLES=20
LONG_BENCH_FULL_SAMPLES=200
LONG_BENCH_REPRESENTATIVE_DATASETS="gov_report lcc"
LONG_BENCH_MAX_NEW_TOKENS=64
LONG_BENCH_SMOKE_MAX_NEW_TOKENS=8
LONG_BENCH_TEMPERATURE=0
LONG_BENCH_WARMUP_RUNS=3
LONG_BENCH_SEED=42
LONG_BENCH_LOCAL_FILES_ONLY=1
```

Model/checkpoint riêng của từng baseline cũng nằm trong master, với fallback
đến các biến model đã tồn tại: `LONG_BENCH_EAGLE_MODEL`,
`LONG_BENCH_DFLASH_MODEL`, `LONG_BENCH_LONGSPEC_TARGET_MODEL`,
`LONG_BENCH_LONGSPEC_DRAFT_MODEL`, `LONG_BENCH_SPECEXTEND_DRAFT_MODEL`,
`LONG_BENCH_SSSD_DATASTORE_PATH` và `LONG_BENCH_MAGICDEC_MODEL_PTH`.

CLI chỉ là override có chủ đích, ví dụ `--mode full`, `--baselines ...`,
`--datasets ...`, `--samples-per-dataset ...`; mọi config không override vẫn
đến từ master.

## Ba profile

### Smoke

Mặc định chọn một mẫu mỗi dataset và giới hạn output ngắn. Trên máy không có
CUDA, pipeline chạy preflight/adapter validation và ghi trạng thái
`preflight_only` hoặc `unsupported_cpu`; không ghi throughput/latency như thể
đã chạy GPU. Trên B200, cùng lệnh chạy inference thật cho mọi baseline có đủ
dependency và checkpoint.

### Representative

Mặc định chạy `gov_report` và `lcc`, mỗi dataset 20 mẫu. Hai dataset đại diện
cho summarization và code completion; danh sách có thể override bằng
`--datasets`. Profile này dùng để kiểm tra end-to-end và lấy số liệu nhanh
trước full matrix.

### Full

Chạy năm dataset, 200 mẫu/dataset, cho các baseline được chọn. Nếu thiếu
model/checkpoint/dependency, baseline được đánh dấu trong manifest và pipeline
fail ở chế độ strict; tùy chọn `--allow-unsupported` cho phép chạy phần còn
lại nhưng không được coi là ma trận đầy đủ.

## Luồng chạy

```text
master config
    ↓
run_longbench_200.sh
    ↓ load + runtime/preflight
run_longbench_200.py
    ↓ chọn profile / dataset / baseline
baseline adapter
    ↓ prompt canonical + generation
per-sample JSONL + run_manifest.json
    ↓
collect_metrics.py: speedup, ESR/DSR và quality metric sau
```

Model được load một lần cho mỗi process/baseline-dataset run. Warmup không
tính vào sample timing. Trước/sau generation phải synchronize CUDA; peak
memory được reset trước mỗi sample và đọc sau generation.

## Output schema

Mỗi output JSONL có record per sample và một record `type=summary` cuối file.
Các trường chuẩn gồm:

```text
run_id, method, dataset, sample_id, model, status,
input_tokens, output_tokens, retained_tokens,
model_load_ms, warmup_runs, prefill_ms, ttft_ms, decode_ms, tpot_ms, e2e_ms,
throughput_tok_s, qps, peak_memory_gb, batch_size,
device, gpu_name, dtype, attention_backend, seed, temperature, max_new_tokens,
text, reference_output
```

Thông số riêng được đặt trong `extra_metrics`, ví dụ acceptance length/rate,
draft latency, verification latency, compression ratio. `status` phải phân
biệt `success`, `preflight_only`, `unsupported_cpu`, `missing_dependency`,
`missing_checkpoint` và `failed`.

Output không ghi quality metric trong infer; evaluator sau đó route
`summarization` đến ROUGE/BLEU và `code_completion` đến exact/edit similarity.

Mỗi run có:

```text
outputs/longbench_200/<run_id>/
├── run_manifest.json
├── vanilla_hf/<dataset>.jsonl
├── vanilla_fa/<dataset>.jsonl
└── ...
```

`run_manifest.json` lưu snapshot config đã dùng, source manifest/checksum,
command, git revision nếu có, Python/PyTorch/CUDA/Transformers/flash-attn
version, GPU, profile, seed, sample counts và trạng thái từng baseline.

## Failure handling

- Master/config/data/model path thiếu: fail trước khi load model.
- Dataset schema/ID không hợp lệ: fail trước khi infer.
- Dependency hoặc kernel không tương thích: ghi failure có nguyên nhân; strict
  mode dừng baseline đó, không fallback âm thầm.
- CUDA không có: chỉ preflight trong smoke; representative/full yêu cầu GPU.
- Process crash hoặc output thiếu: giữ log tail, exit code và trạng thái trong
  manifest; không đánh dấu success.
- Output đang tồn tại: tạo `run_id` mới hoặc yêu cầu `--force`, không ghi đè
  run cũ mặc định.

## Validation scope

- L0 static: Python compile, CLI help, master-config contract, baseline
  registry, schema/output contract, no-network/local-files-only checks.
- L1 runtime local: `.venv` preflight trên T4/CPU cho cả 9 baseline; Vanilla
  adapter có thể dùng fixture nhẹ để kiểm tra mapping, nhưng không báo cáo
  GPU performance.
- L1 runtime server: smoke inference thật một mẫu/dataset/baseline trên B200,
  kiểm tra output không rỗng, timing hữu hạn, token count hợp lệ, memory và
  artifact JSONL/manifest.

## Acceptance criteria

- Một lệnh duy nhất dùng master có thể chọn cả ba profile.
- Không baseline nào dùng hard-coded model/data/output mặc định thay cho master.
- Vanilla HF và Vanilla FA khác đúng attention backend.
- Mọi record thành công giữ đúng dataset/sample ID canonical.
- Các trường timing/token/memory cần cho speedup/ESR/DSR có mặt hoặc `null`
  với lý do rõ ràng.
- Smoke local không tạo số liệu GPU giả; full B200 có thể chạy lại cùng config.
