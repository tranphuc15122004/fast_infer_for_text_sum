# DFlash residual experiments Implementation Plan

> **For agentic workers:** thực hiện theo TDD; mỗi task là một thay đổi nhỏ có
> test độc lập và không được đưa số liệu GPU giả vào artifact.

**Goal:** Xây pipeline P0–P4 dưới `src/analyze/dflash_residual` để phân biệt
candidate-generation failure với residual candidate-selection failure của
DFlash/DFlash2 trên long-document summarization.

**Experiment directory:** `src/analyze/dflash_residual/`

**Hypothesis:** Khi context tăng, Recall@16 và/hoặc tỷ lệ recovery của DFlash2
trên oracle Top-16 giảm; interaction `log(context) × draft_position` giúp
kiểm tra context-induced suffix decay.

**Validation scope:** TDD unit tests, CPU synthetic end-to-end smoke, static
compile/diff checks; GPU collector là external handoff vì host T4 không có
CUDA. Không có training loop nên không chạy Validation Pyramid training.

**Evaluation design:** Các phase là CLI analysis batch, chạy theo P0→P4 một lần
trên artifact trace; mỗi phase ghi phase-start/phase-end, status, counts,
metrics và output paths. Missing/invalid input trả `UNAVAILABLE` có lý do,
không thay bằng zero. Output không cần checkpoint/in-memory evaluator.

**Architecture:** Core metrics dependency-light và độc lập collector. Trace
collector mirror target verifier semantics của DFlash; analyzer nhận JSONL
để có thể dùng cả official runner và custom runner, join DFlash2 selection khi
artifact thật đã có, và sinh bảng/report/plot reproducible.

---

## Shared Scaffold

### Existing infra (không sửa)

- DFlash model/semantics: `externals/dflash/dflash/model.py`.
- Benchmark data loader: `scripts/common/data_loader.py`.
- GroundSync trace conventions tham khảo: `src/analyze/groundsync/`.
- Runtime master config: `config/master.path` và `docs/server_environment.md`.

### Needs setup

- Tạo package `src/analyze/dflash_residual/` và tests nội bộ.
- Không sửa các file đang dirty của SyncSpec.
- Không commit generated outputs lớn; output mặc định nằm trong run directory
  do người dùng chọn.

## Subtask 1: Trace schema và metrics core

**Role:** Định nghĩa contract chống lệch target/candidate và mọi metric nền
cho P1–P4.

**Files:**

- Create: `src/analyze/dflash_residual/__init__.py`
- Create: `src/analyze/dflash_residual/schema.py`
- Create: `src/analyze/dflash_residual/metrics.py`
- Create: `src/analyze/dflash_residual/tests/test_metrics.py`

**Interfaces:** `normalize_trace_row`, `validate_trace_row`,
`recall_at_k`, `survival`, `prefix_match_length`, `oracle_prefix_length`,
`summarize_p1`, `summarize_headroom`, `fit_context_depth_interaction`.

- [x] Viết test fail-first cho 1-based `draft_position`, Recall@M, candidate
  miss, prefix oracle, rho khi denominator bằng 0, MAT/survival và logistic
  interaction có document bootstrap.
- [x] Chạy `python3 -m pytest src/analyze/dflash_residual/tests/test_metrics.py -q`
  và xác nhận fail vì package/function chưa tồn tại.
- [x] Implement schema immutable-safe và metrics deterministic; target token
  ngoài candidate phải là miss thật, không inject vào candidate list.
- [x] Chạy lại test và full core test.

## Subtask 2: JSONL adapters, joins và report tables

**Role:** Nạp trace official/custom, chuyển benchmark acceptance legacy thành
round rows, join DFlash2 selection và ghi JSON/CSV/Markdown.

**Files:**

- Create: `src/analyze/dflash_residual/io.py`
- Create: `src/analyze/dflash_residual/report.py`
- Create: `src/analyze/dflash_residual/tests/test_io_report.py`

**Interfaces:** `read_trace_jsonl`, `read_legacy_acceptance_jsonl`,
`join_selection_trace`, `write_metrics_bundle`, `render_markdown_report`.

- [x] Viết test với mixed valid/error rows, legacy `acceptance_lengths`, join
  key mismatch và missing DFlash2.
- [x] Chạy test để quan sát RED.
- [x] Implement strict validation, duplicate-key rejection cho selection,
  deterministic CSV columns và report status `UNAVAILABLE`.
- [x] Chạy test IO/report và core regression.

## Subtask 3: P0–P2 analyzer và plots

**Role:** Chạy alignment gate, task-regime table và Recall@M anatomy/heatmap.

**Files:**

- Create: `src/analyze/dflash_residual/alignment.py`
- Create: `src/analyze/dflash_residual/plotting.py`
- Create: `src/analyze/dflash_residual/p0_alignment.py`
- Create: `src/analyze/dflash_residual/p1_task_regime.py`
- Create: `src/analyze/dflash_residual/p2_coverage.py`
- Create: `src/analyze/dflash_residual/tests/test_phases.py`

- [x] Viết test P0 matching/mismatch và P1/P2 context-depth grouping.
- [x] Chạy test RED.
- [x] Implement phase functions, plots optional, và gate conservative.
- [x] Chạy test GREEN; kiểm tra output không có claim positive khi thiếu rows.

## Subtask 4: P3/P4 decomposition và logistic interaction

**Role:** Tách candidate-generation/selection headroom và kiểm định
context-induced suffix decay.

**Files:**

- Create: `src/analyze/dflash_residual/p3_headroom.py`
- Create: `src/analyze/dflash_residual/p4_interaction.py`
- Extend: `src/analyze/dflash_residual/tests/test_phases.py`

- [x] Viết test oracle Top-16 longest prefix, DFlash2 unavailable, rho clipping
  không tự ý clip số liệu, và bootstrap CI theo document.
- [x] Chạy test RED.
- [x] Implement P3/P4 và decision metadata; phân biệt `candidate_miss`,
  `selection_error`, `zero_headroom`.
- [x] Chạy test GREEN và full package tests.

## Subtask 5: GPU DFlash collector

**Role:** Sinh candidate lattice + target verifier posterior trên canonical
setup, hỗ trợ context caps và native block metadata.

**Files:**

- Create: `src/analyze/dflash_residual/trace_dflash.py`
- Create: `src/analyze/dflash_residual/tests/test_trace_dflash_contract.py`
- Create: `src/analyze/dflash_residual/README.md`

- [x] Viết contract test parser/config không load model và test helper target
  token alignment bằng fake tensor/model.
- [x] Chạy RED.
- [x] Implement collector lazy-import Transformers/DFlash, mirror
  `dflash_generate`, ghi candidate Top-M, verifier target, accepted draft len,
  context cap/truncate metadata và error rows.
- [x] Chạy CPU contract tests; CUDA execution chỉ document handoff.

## Subtask 6: CLI integration `[INTEGRATION]`

**Role:** Assemble `p0`, `p1`, `p2`, `p3`, `p4`, `all` thành command reproducible
và CPU synthetic fixture.

**Files:**

- Create: `src/analyze/dflash_residual/run.py`
- Extend: `src/analyze/dflash_residual/README.md`
- Create: `src/analyze/dflash_residual/tests/test_run.py`

- [x] Viết test CLI synthetic all, manifest preservation, phase-specific
  output và no-DFlash2 conservative report.
- [x] Chạy RED.
- [x] Implement argparse, phase dispatch, run manifest, progress messages,
  JSON/CSV/Markdown bundle và exit code 2 cho unavailable input.
- [x] Chạy package tests, compileall và synthetic CLI.
- [x] Chạy full repository regression chỉ khi không đụng các thay đổi dirty
  hiện có; ghi rõ mọi failure ngoài package này.
