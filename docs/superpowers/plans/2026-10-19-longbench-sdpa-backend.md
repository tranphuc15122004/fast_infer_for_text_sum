# LongBench SDPA Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit SDPA backend mode to the canonical LongBench runner, preserving native mode as the default and preventing unsupported baselines from being reported as SDPA.

**Architecture:** The orchestrator accepts an attention-backend policy and passes it through adapter configuration. Only adapters that actually run model attention through PyTorch SDPA may execute in strict SDPA mode; unsupported baseline rows are written with an explicit backend status. Inference records and the run manifest retain the requested and effective backend.

**Tech Stack:** Python 3.12, PyTorch, Transformers, Bash launchers, Markdown documentation.

**Spec:** `docs/longbench_200_benchmark.md` and the user's request to run the full baseline matrix with SDPA.

## Global Constraints

- Production server uses the system Python 3.12 runtime and pre-mirrored offline dependencies.
- Full LongBench uses the canonical adapter registry and JSONL schema.
- Native baseline behavior remains the default unless SDPA is explicitly selected.
- A baseline that cannot use SDPA must not be silently run with a different kernel under an SDPA label.
- The local T4 machine is CPU-only for development; full GPU inference runs on the configured server.

---

### Task 1: Add an explicit backend policy to the LongBench orchestrator

**Files:**
- Modify: `scripts/run_longbench_200.py`
- Modify: `scripts/common/longbench_adapter.py`

**Interfaces:**
- Consumes: `--attention-backend {native,sdpa}` and the existing baseline configuration.
- Produces: `attention_backend` in each baseline's adapter config, plus backend compatibility status from preflight.

- [x] Add `--attention-backend` with default `native` and record the selected value in the run manifest.
- [x] Thread the selected value through bundled and per-cell execution paths.
- [x] Add an explicit compatibility registry; strict SDPA mode marks unsupported adapters as `unsupported_backend` before launch.
- [x] Preserve each unsupported reason in status JSONL and run manifest cells.

### Task 2: Implement SDPA for compatible LongBench adapters

**Files:**
- Modify: `scripts/common/vanilla_inference.py`
- Modify: `scripts/common/longbench_adapter.py`
- Modify: `scripts/infer_dflash.py`

**Interfaces:**
- Consumes: `attention_backend="sdpa"` from Task 1.
- Produces: adapter commands explicitly request `sdpa`; records report requested and effective backend.

- [x] Allow `sdpa` in the `vanilla_hf` parser and pass it explicitly from the adapter.
- [x] Add an explicit SDPA override to DFlash while retaining its existing automatic backend selection in native mode.
- [x] Keep output record metadata and DFlash's internal run-config hash tied to the effective backend.

### Task 3: Document server execution and backend limits

**Files:**
- Modify: `docs/longbench_200_benchmark.md`
- Modify: `docs/README.md`

**Interfaces:**
- Consumes: the `--attention-backend sdpa` runner interface from Tasks 1–2.
- Produces: a server command for SDPA preflight and full runs, plus the supported-baseline list and expected unsupported statuses.

- [x] Document a preflight-only command using the server master config.
- [x] Document the full-run command, GPU selection, output location, and how to inspect actual backend metadata.
- [x] State that the T4 local CPU environment cannot be used to claim server GPU performance.

### Task 4: Review implementation without GPU inference

**Files:**
- Review: all files changed above.

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: syntax-checked scripts and a reviewed server procedure.

- [x] Check changed Python files for syntax errors.
- [x] Review the help text and command construction paths for bundled and non-bundled full runs.
- [x] Do not claim a GPU benchmark ran on the CPU-only development host.

---

## Extension: SDPA for the `src/Benchmark` native method matrix

The canonical `run_longbench_200.py` adapter registry does not include Domino or
DSpark. The relevant method runner is the existing native Transformers pipeline
in `src/Benchmark/native_fa4_benchmark.py`; its default remains FA4. vLLM 0.30
does not provide `TORCH_SDPA` as a decoder attention backend, so this extension
adds an explicit native SDPA mode and does not route SDPA requests through
`vllm_all`.

### Task 5: Add an explicit SDPA native benchmark mode

**Files:**
- Modify: `src/Benchmark/fa4_server.py`
- Modify: `src/Benchmark/native_fa4_benchmark.py`
- Modify: `src/Benchmark/common/flashattn_runtime.py`
- Modify: `scripts/run.sh`
- Modify: `scripts/run_fa4_benchmark.sh`

**Interfaces:**
- Consumes: `--attention-backend sdpa` or `scripts/run.sh sdpa_native`, plus the
  existing master configuration and native method names.
- Produces: a separate `outputs/sdpa_native_benchmark/<run-id>/` result tree;
  records include resolved backend and dispatch proof for target and draft.
- `vanilla_hf` and `dflash` remain required as the reference pair; the requested
  speculative methods are `dflash,eagle3,domino,dspark`.

- [x] Keep FA4 as the default and gate its Blackwell/import checks to FA4 mode.
- [x] Add a CUDA SDPA helper for EAGLE target/draft attention that preserves the
  verifier tree mask and handles one-token cached decode without top-left causal
  masking.
- [x] Parameterize attention dispatch tracking and runtime validation so SDPA
  mode requires target and draft SDPA calls and rejects alternate dispatch.
- [x] Add the `sdpa_native` dispatcher alias and backend-specific output root.

### Task 6: Document the server run on physical GPU 1

**Files:**
- Modify: `src/Benchmark/README.md`
- Modify: `docs/longbench_200_benchmark.md`

**Interfaces:**
- Consumes: `scripts/run.sh sdpa_native <master-config> --mode ...` and
  `FI_GPU_IDS=1`.
- Produces: preflight, smoke, and full commands for the five-method paired
  comparison with `vanilla_hf` as the quality/parity reference.

- [x] State that `vllm_all` stays on its own selected backend and is not the SDPA
  execution path for the vLLM 0.30 decoder.
- [x] Document physical GPU selection, required reference method, output path,
  and metadata fields used to confirm SDPA dispatch.

### Task 7: Static review

**Files:**
- Review: all files changed in Tasks 5–6.

**Interfaces:**
- Produces: syntax-checked Python, shell syntax-checked launchers, and clean diff
  formatting. No model inference or tests are run on the CPU-only host.

- [x] Run Python compilation, shell syntax checks, `git diff --check`, and inspect
  CLI help/dispatch construction.
- [x] Report that GPU smoke and full execution must run on the configured server.
