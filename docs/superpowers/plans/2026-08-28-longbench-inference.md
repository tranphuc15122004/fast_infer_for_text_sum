# Implementation Plan: Pipeline infer LongBench cho 9 baseline

> **For agentic workers:** Thực hiện từng task theo TDD; không ghi số liệu benchmark giả và không chạy full matrix trên môi trường không có CUDA.

**Mục tiêu:** Xây entry point dùng master config để chạy `vanilla_hf`, `vanilla_fa`, `magicdec`, `longspec`, `eagle3`, `dflash`, `specextend`, `sssd`, `fafo` trên `data/longbench_200` với ba profile `smoke`, `representative`, `full` và output timing thống nhất.

**Experiment directory:** repository root `/home/tuantb/fast_infer_text_sum`

**Hypothesis:** Giữ cố định Llama 3.1 8B Instruct, dataset/sample ID, generation config và môi trường; khi đó khác biệt attention/speculative/KV implementation có thể được đo bằng timing/token/memory từ cùng output contract.

**Validation scope:** L0 static và contract checks trên `.venv`; L1 local preflight trên T4/CPU; L1 inference smoke thật trên B200 khi có CUDA, model và checkpoint local. Không báo cáo GPU performance từ CPU.

**Evaluation design:** Infer ghi raw metrics per sample khi adapter hỗ trợ; adapter upstream chỉ trả aggregate phải ghi `scope=aggregate`, `sample_ids` và lý do. Quality/speedup/ESR/DSR tính sau bởi collector. Smoke có phase log, timeout và status rõ ràng; lỗi model/dependency/kernel không được fallback âm thầm.

**Kiến trúc:** `run_longbench_200.sh` load master qua `common/config.sh`, kiểm tra Python 3.12 rồi gọi `run_longbench_200.py`. Orchestrator resolve profile/dataset/baseline, tạo output run riêng và gọi adapter registry. Vanilla adapters dùng chung measurement helper; các baseline vendored giữ implementation upstream và được gọi qua adapter/converter, không sửa trực tiếp source vendored trừ khi contract hiện tại bắt buộc.

**Tech stack:** Bash master-config loader, Python 3.12, PyTorch, Transformers, CUDA/flash-attn, JSONL; test bằng pytest.

**Spec:** `docs/superpowers/specs/2026-08-28-longbench-inference-design.md`

## Ràng buộc chung

- Mọi default model/data/output/profile lấy từ master external qua `config/master.path` hoặc `FAST_INFER_MASTER_CONFIG`.
- Runtime mặc định là `.venv/bin/python`; launcher phải kiểm tra Python 3.12.
- `FI_OFFLINE=1` đặt `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`; không tải internet.
- Model target là Llama 3.1 8B Instruct local; draft/checkpoint riêng lấy từ master.
- `temperature=0`, seed cố định, batch size mặc định 1 và cùng `max_new_tokens` theo profile.
- `vanilla_hf` dùng `eager`; `vanilla_fa` dùng `flash_attention_2`, thiếu flash-attn thì status lỗi rõ ràng.
- Không dùng `answers`/`reference_output` làm input model.
- Không ghi đè output run cũ mặc định; dùng run ID mới.
- Không commit trong sandbox nếu `.git/index` không ghi được; ghi nhận kết quả vào progress ledger.

## Shared Scaffold

### Existing infra giữ nguyên

- Canonical data: `data/longbench_200/*.jsonl`, `manifest.json`.
- Prompt/schema helpers: `scripts/common/benchmark_data.py`, `scripts/common/data_loader.py`.
- Runtime/config loader: `scripts/common/config.sh`, `scripts/common/runtime.sh`.
- Output writer: `scripts/common/io_util.py`.
- Existing baseline adapters: `scripts/infer_*.py`, `scripts/eagle3_infer_qwen3.py`, `scripts/run_*.sh`.
- Existing collector: `scripts/collect_metrics.py`.

### Files cần tạo/chỉnh

- Tạo `scripts/run_longbench_200.sh`: master-config entry point.
- Tạo `scripts/run_longbench_200.py`: orchestration/profile/run manifest.
- Tạo `scripts/infer_vanilla_hf.py`: native PyTorch eager baseline.
- Tạo `scripts/infer_vanilla_fa.py`: flash-attention baseline.
- Tạo `scripts/common/benchmark_runtime.py`: metadata, synchronized timing, memory, status/output helpers.
- Tạo `scripts/common/longbench_adapter.py`: canonical-to-upstream converters và adapter command registry.
- Chỉnh `scripts/common/config.sh`: namespace `longbench` và alias master variables.
- Chỉnh `docs/fast_infer_master.example.env`: `LONG_BENCH_*` defaults/checkpoint paths.
- Chỉnh `scripts/collect_metrics.py`: recursive run layout, status/scope và metric raw fields.
- Chỉnh `docs/longbench_200_benchmark.md`, `docs/README.md`: lệnh chạy ba profile.
- Tạo test trong `tests/test_longbench_inference.py` và các test contract nhỏ nếu cần.

## Task 1: Master config và entry point

**Vai trò:** Bảo đảm mọi profile infer dùng đúng external master config và cùng `.venv`.

**Files:**

- Modify: `scripts/common/config.sh`
- Create: `scripts/run_longbench_200.sh`
- Modify: `docs/fast_infer_master.example.env`
- Test: `tests/test_longbench_inference.py`

**Interface:**

- Bash function mới: `fast_infer__load_longbench`.
- Case mới: `fast_infer_load_config longbench`.
- Launcher nhận `--mode`, `--baselines`, `--datasets`, `--config`/master override và truyền phần còn lại cho Python.
- Export tối thiểu: `LONG_BENCH_DATA_DIR`, `LONG_BENCH_OUTPUT_DIR`, `LONG_BENCH_MODEL`, `LONG_BENCH_BASELINES`, `LONG_BENCH_DATASETS`, `LONG_BENCH_MODE`, sample counts, token/generation/warmup/seed và checkpoint paths.

- [ ] **Step 1: Viết test fail cho config namespace và launcher**

```python
def test_longbench_loader_maps_master_values(tmp_path):
    master = tmp_path / "master.env"
    master.write_text(
        'LONG_BENCH_DATA_DIR="data/fixture"\n'
        'LONG_BENCH_OUTPUT_DIR="outputs/fixture"\n'
        'LONG_BENCH_MODEL="/models/llama"\n'
        'LONG_BENCH_BASELINES="vanilla_hf vanilla_fa"\n',
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", "-c", "source scripts/common/config.sh; fast_infer_load_config longbench; printf '%s\\n' \"$LONG_BENCH_MODEL\" \"$LONG_BENCH_BASELINES\""],
        cwd=ROOT,
        env={**os.environ, "FAST_INFER_MASTER_CONFIG": str(master), "FAST_INFER_PYTHON": sys.executable},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "/models/llama" in result.stdout
    assert "vanilla_hf vanilla_fa" in result.stdout


def test_longbench_launcher_requires_shared_python312():
    text = (ROOT / "scripts/run_longbench_200.sh").read_text(encoding="utf-8")
    assert "fast_infer_load_config longbench" in text
    assert "scripts/common/runtime.sh" in text
    assert "run_longbench_200.py" in text
```

- [ ] **Step 2: Chạy test để xác nhận fail**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'loader or launcher'`

Expected: FAIL vì chưa có namespace `longbench` và launcher.

- [ ] **Step 3: Implement config mapping và launcher**

Thêm `fast_infer__load_longbench`/case vào `config.sh`; launcher source config, gọi `fast_infer_load_config longbench`, source runtime và execute Python bằng `$FAST_INFER_PYTHON`. Không source một config riêng trong repository.

- [ ] **Step 4: Chạy test pass**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'loader or launcher'`

Expected: PASS.

- [ ] **Step 5: Cập nhật master example**

Thêm block `LONG_BENCH_*` và fallback model/checkpoint names; không thay đổi pointer `config/master.path`.

## Task 2: Timing, metadata và output contract

**Vai trò:** Cung cấp một measurement core không phụ thuộc baseline để output có đủ thông số cho speedup/ESR/DSR.

**Files:**

- Create: `scripts/common/benchmark_runtime.py`
- Test: `tests/test_longbench_inference.py`

**Interface:**

- `runtime_metadata() -> dict[str, object]`
- `measure_call(fn, *, device: torch.device, reset_peak_memory: bool = True) -> tuple[object, dict]`
- `build_sample_record(...) -> dict`
- `build_status_record(...) -> dict`
- `append_jsonl(path: Path, record: Mapping) -> None`

Timing contract:

- `e2e_ms` đo từ trước generation đến sau generation.
- CUDA có `torch.cuda.synchronize()` trước/sau; CPU dùng `perf_counter`.
- `decode_ms`/`prefill_ms` chỉ ghi khi adapter đo được; không suy diễn thành số chính xác nếu upstream không cung cấp.
- `throughput_tok_s = output_tokens / (e2e_ms / 1000)` và ghi thêm `decode_throughput_tok_s` nếu có decode time.
- `peak_memory_gb` là `max_memory_allocated / 2**30` sau reset; CPU là `None`.

- [ ] **Step 1: Viết test fail cho timing/schema**

```python
def test_measure_call_returns_elapsed_and_output():
    value, timing = measure_call(lambda: "ok", device=torch.device("cpu"))
    assert value == "ok"
    assert timing["e2e_ms"] >= 0
    assert timing["device"] == "cpu"


def test_build_status_record_never_invents_performance_metrics():
    row = build_status_record(method="vanilla_fa", dataset="lcc", sample_id="x", status="unsupported_cpu", reason="CUDA unavailable")
    assert row["status"] == "unsupported_cpu"
    assert row["e2e_ms"] is None
    assert row["throughput_tok_s"] is None
```

- [ ] **Step 2: Chạy test fail**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'measure or status_record'`

Expected: FAIL vì module/helper chưa tồn tại.

- [ ] **Step 3: Implement helper tối thiểu**

Implement synchronization an toàn, metadata version/GPU, finite checks, throughput derivation và status record. Dùng `io_util._json_safe` hoặc một hàm tương đương để tránh tensor không serialize được.

- [ ] **Step 4: Chạy test pass**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'measure or status_record'`

Expected: PASS.

## Task 3: Vanilla HF và Vanilla FA adapters

**Vai trò:** Cung cấp hai control baselines cùng model/generation path, khác duy nhất attention backend.

**Files:**

- Create: `scripts/infer_vanilla_hf.py`
- Create: `scripts/infer_vanilla_fa.py`
- Test: `tests/test_longbench_inference.py`

**Interface:** Cả hai script nhận `--model`, `--data-file`, `--max-samples`, `--max-new-tokens`, `--temperature`, `--seed`, `--warmup-runs`, `--max-input-tokens`, `--output`, `--smoke`.

Implementation:

- Load tokenizer/model một lần cho một dataset run, `local_files_only` theo master.
- Render canonical prompt thông qua `load_records`; không dùng reference làm prompt.
- `vanilla_hf`: `attn_implementation="eager"`.
- `vanilla_fa`: kiểm tra import `flash_attn`, load `attn_implementation="flash_attention_2"`, fail rõ ràng nếu thiếu.
- Warmup riêng; đo generation bằng `model.generate` với `do_sample=False`, `use_cache=True`.
- Ghi `model_load_ms`, input/output token, e2e, throughput, peak memory, text, `sample_id`, reference và backend.
- Ghi summary cuối file.

- [ ] **Step 1: Viết test fail**

```python
def test_vanilla_parser_exposes_distinct_attention_defaults():
    from infer_vanilla_hf import build_parser as hf_parser
    from infer_vanilla_fa import build_parser as fa_parser
    assert hf_parser().parse_args(["--output", "x"]).attention_backend == "eager"
    assert fa_parser().parse_args(["--output", "x"]).attention_backend == "flash_attention_2"


def test_vanilla_record_contains_shared_timing_fields():
    record = build_sample_record(method="vanilla_hf", dataset="lcc", sample_id="id", model="m", input_tokens=10, output_tokens=2, timing={"e2e_ms": 4.0, "prefill_ms": 1.0, "decode_ms": 3.0}, config={"attention_backend": "eager"}, text="x", reference_output="y")
    assert record["throughput_tok_s"] == 500.0
    assert record["attention_backend"] == "eager"
```

- [ ] **Step 2: Chạy test fail**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'vanilla'`

Expected: FAIL vì parser/adapters chưa tồn tại.

- [ ] **Step 3: Implement hai adapter**

Dùng một hàm nội bộ dùng chung trong `benchmark_runtime.py` để tránh sai khác measurement; giữ hai file entry point để baseline registry nhận diện rõ ràng.

- [ ] **Step 4: Chạy test pass và CLI help**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'vanilla' && .venv/bin/python scripts/infer_vanilla_hf.py --help && .venv/bin/python scripts/infer_vanilla_fa.py --help`

Expected: PASS, help không load model.

## Task 4: Adapter registry, canonical conversion và preflight

**Vai trò:** Chuẩn hóa invocation của 9 baseline và bảo đảm các upstream input format vẫn giữ sample ID.

**Files:**

- Create: `scripts/common/longbench_adapter.py`
- Test: `tests/test_longbench_inference.py`

**Interface:**

- `BASELINES: tuple[str, ...]`
- `build_adapter_command(baseline: str, config: Mapping, data_file: Path, output: Path, mode: str) -> list[str]`
- `preflight_baseline(baseline: str, config: Mapping, *, device: str) -> dict`
- `convert_records_for_baseline(baseline: str, records: Sequence[Mapping], output: Path) -> Path | None`

Registry mapping:

- Vanilla: direct Python adapters.
- DFlash/LongSpec/FAFO/SSSD: existing adapters with canonical data path and master-derived args.
- EAGLE3: temporary question JSONL with `question_id=sample_id`, `turns=[rendered_prompt]`, `answer/reference`; no sample ID loss.
- SpecExtend: temporary upstream-compatible input generated from canonical prompt/reference.
- MagicDec: canonical branch gọi trực tiếp SnapKV engine trên prompt tùy ý;
  checkpoint `.pth` và tokenizer path lấy từ master.
- Preflight validates local file paths, imports and CUDA requirement without loading 8B model.

- [ ] **Step 1: Viết test fail**

```python
def test_registry_contains_exactly_requested_baselines():
    assert BASELINES == ("vanilla_hf", "vanilla_fa", "magicdec", "longspec", "eagle3", "dflash", "specextend", "sssd", "fafo")


def test_eagle_converter_preserves_canonical_id_and_reference(tmp_path):
    output = tmp_path / "eagle.jsonl"
    convert_records_for_baseline("eagle3", [{"id": "lcc_1", "prompt": "code", "reference": "next"}], output)
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["question_id"] == "lcc_1"
    assert row["turns"] == ["code"]
    assert row["answer"] == "next"
```

- [ ] **Step 2: Chạy test fail**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'registry or converter'`

Expected: FAIL vì registry/converter chưa tồn tại.

- [ ] **Step 3: Implement registry/preflight/converters**

Mỗi command phải truyền model/checkpoint từ config; không thêm default model hard-coded trong registry. Converter ghi temp file trong run directory và được cleanup an toàn sau process.

- [ ] **Step 4: Chạy test pass**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'registry or converter'`

Expected: PASS.

## Task 5: Orchestrator và ba profile [INTEGRATION]

**Vai trò:** Ghép master config, profiles, registry, subprocess, run manifest và output layout thành pipeline chạy được.

**Files:**

- Create: `scripts/run_longbench_200.py`
- Modify: `scripts/run_longbench_200.sh`
- Test: `tests/test_longbench_inference.py`

**Interface:**

```text
python scripts/run_longbench_200.py
  [--mode smoke|representative|full]
  [--baselines BASELINE ...]
  [--datasets DATASET ...]
  [--samples-per-dataset N]
  [--max-new-tokens N]
  [--output-dir DIR]
  [--run-id ID]
  [--preflight-only]
  [--allow-unsupported]
```

Profile resolution:

- `smoke`: 1 sample/dataset, smoke token budget; if no CUDA, preflight only with status records.
- `representative`: config datasets default `gov_report lcc`, 20 samples/dataset.
- `full`: config datasets default all five, 200 samples/dataset; strict by default.

Output:

```text
outputs/longbench_200/<run_id>/
├── run_manifest.json
├── vanilla_hf/gov_report.jsonl
└── ...
```

Run manifest records resolved config, environment, source manifest checksum, exact commands, counts, status and error tail. Every subprocess gets a bounded timeout from master; output files are checked for summary/status before success.

- [ ] **Step 1: Viết integration test fail**

```python
def test_orchestrator_smoke_preflight_writes_manifest_without_loading_model(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/run_longbench_200.py", "--mode", "smoke", "--preflight-only", "--baselines", "vanilla_hf", "--datasets", "lcc", "--output-dir", str(tmp_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    manifests = list(tmp_path.glob("*/run_manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["mode"] == "smoke"
    assert manifest["preflight_only"] is True


def test_orchestrator_rejects_full_mode_without_cuda():
    # Uses monkeypatched/injected config path and preflight so no model loads.
    with pytest.raises(SystemExit):
        resolve_profile(mode="full", cuda_available=False, allow_unsupported=False)
```

- [ ] **Step 2: Chạy test fail**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'orchestrator'`

Expected: FAIL vì orchestrator chưa tồn tại.

- [ ] **Step 3: Implement orchestrator**

Load canonical manifest before dispatch; resolve paths relative to repo root; create unique run ID; run each baseline/dataset command; stream phase logs; record subprocess status and log tail. In local CPU smoke, call preflight and write `preflight_only` records; do not call 8B inference. In B200 smoke/full, call real adapter commands.

- [ ] **Step 4: Chạy integration test pass**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'orchestrator'`

Expected: PASS.

- [ ] **Step 5: Chạy L0 static validation**

Run: `.venv/bin/python -m py_compile scripts/run_longbench_200.py scripts/infer_vanilla_hf.py scripts/infer_vanilla_fa.py scripts/common/benchmark_runtime.py scripts/common/longbench_adapter.py`

Expected: PASS.

- [ ] **Step 6: Chạy L1 local preflight**

Run: `FAST_INFER_PYTHON="$PWD/.venv/bin/python" bash scripts/run_longbench_200.sh --mode smoke --preflight-only`

Expected: master/config/data/dependency checks complete; no fabricated latency or throughput.

## Task 6: Collector, metric readiness và documentation

**Vai trò:** Cho phép tính metric sau khi infer mà không trộn code-completion với ROUGE.

**Files:**

- Modify: `scripts/collect_metrics.py`
- Modify: `data/README.md`
- Modify: `docs/longbench_200_benchmark.md`
- Modify: `docs/README.md`
- Test: `tests/test_longbench_inference.py`

**Requirements:**

- Collector nhận `outputs/longbench_200/<run_id>/` và recursive baseline/dataset files.
- Chỉ record `status=success` dùng cho speed/quality aggregate; status khác nằm trong report nhưng không được tính như zero.
- Dùng `task_type`/dataset để route summarization đến ROUGE/BLEU và code completion đến exact/edit similarity.
- Speedup/ESR/DSR tính từ paired records có cùng `sample_id`, `max_new_tokens`, model và run config; thiếu pair phải báo missing pair.
- Markdown report hiển thị số record success/unsupported/failed và coverage.

- [ ] **Step 1: Viết test fail**

```python
def test_collector_ignores_preflight_records_for_speed_aggregates(tmp_path):
    path = tmp_path / "vanilla_fa" / "lcc.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"status": "unsupported_cpu", "method": "vanilla_fa", "dataset": "lcc", "e2e_ms": None},
        {"type": "summary", "status": "preflight_only"},
    ]
    path.write_text("".join(json.dumps(row) + "\\n" for row in rows), encoding="utf-8")
    result = load_run_records(tmp_path)
    assert result["coverage"]["success"] == 0


def test_code_completion_aggregate_excludes_rouge_keys():
    result = aggregate_run_group([{"status": "success", "task_type": "code_completion", "text": "return x", "reference_output": "return x"}])
    assert "rouge1_f" not in result["quality"]
    assert result["quality"]["code_exact_match"] == 1.0
```

- [ ] **Step 2: Chạy test fail**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'collector'`

Expected: FAIL vì collector chưa hiểu run layout/status contract.

- [ ] **Step 3: Implement collector/docs**

Giữ backward compatibility với output legacy; thêm loader recursive canonical run; không coi `None` timing là zero; ghi rõ coverage và missing pairs.

- [ ] **Step 4: Chạy test pass**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_inference.py -k 'collector'`

Expected: PASS.

## Task 7: Regression và delivery verification

**Vai trò:** Xác nhận thay đổi không phá contract hiện tại và artifact có thể bàn giao.

**Files:**

- Modify: `progress.md`, `findings.md` nếu cần ghi kết quả.
- Test: toàn bộ `tests/` và focused tests.

- [ ] **Step 1: Chạy focused suite**

Run: `.venv/bin/python -m pytest -q tests/test_longbench_dataset.py tests/test_longbench_inference.py`

Expected: PASS.

- [ ] **Step 2: Chạy toàn bộ project tests**

Run: `.venv/bin/python -m pytest -q tests`

Expected: test mới và test hiện có pass; nếu lỗi baseline ngoài phạm vi, ghi rõ test/path và không che giấu.

- [ ] **Step 3: Kiểm tra CLI/config contract**

Run: `FAST_INFER_PYTHON="$PWD/.venv/bin/python" bash scripts/run_longbench_200.sh --help` và `bash scripts/run_longbench_200.sh --mode smoke --preflight-only`.

Expected: help/entry point dùng `.venv`, master pointer và không load model trong preflight.

- [ ] **Step 4: Kiểm tra canonical data trước khi benchmark**

Run: `.venv/bin/python scripts/validate_longbench_200.py --data-dir data/longbench_200 --expected-count 200`.

Expected: `VALID: 1000 records`.

- [ ] **Step 5: Kiểm tra diff và artifact**

Run: `git diff --check`; kiểm tra không có output benchmark thật bị commit vào `outputs/`/checkpoint.

Expected: không có whitespace error; report cuối nêu rõ smoke local là preflight-only nếu không có CUDA và full B200 chưa được chạy trong local environment.

## Kết quả thực hiện

Tasks 1–6 đã được triển khai. Task 7 đã kiểm tra: `pytest -q tests` đạt 108
tests; `py_compile` và Bash syntax checks pass; smoke CPU tạo đủ 45 cell status
và không có timing giả. Full inference trên B200 chưa được chạy trong phiên này
vì môi trường hiện tại không có CUDA khả dụng. MagicDec có nhánh canonical
LongBench dùng SnapKV engine và checkpoint `.pth`; SSSD/FAFO vẫn là aggregate
scope theo giới hạn upstream. Lưu ý khi đọc output: EAGLE3 upstream đo
decode-only nên record giữ `measurement_scope=decode_only`, không được gọi là
E2E.
