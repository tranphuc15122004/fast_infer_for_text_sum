# DFlash Phase A Adaptive Budget for FA4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, per-request adaptive context budget to native FA4 DFlash Phase A, with live budget traces and a B200 full-run command.

**Architecture:** Keep `dflash_phase_a` fixed mode as the control and select adaptive behavior through `DFLASH_PHASE_A_BUDGET_MODE=adaptive`. Add a scalar controller that moves the budget in 128-token increments based on recent measured round time per committed token, and wire it into the vendored DFlash verify/draft loop. Emit the budget path and per-round observations in sample records; leave vLLM untouched.

**Tech Stack:** Python 3.12, PyTorch CUDA events, existing native FA4 benchmark, DFlash vendored Transformers model.

**Spec:** `docs/superpowers/specs/2026-10-19-dflash-phase-a-adaptive-budget-fa4-design.md`

## Global Constraints

- Keep dense `dflash` and fixed `dflash_phase_a` behavior unchanged unless shared correctness fixes are required by the adaptive path.
- Target verification always uses full context; Phase A changes only the context read by the drafter.
- Preserve the full draft KV bank so budget can rise again later in the same request.
- Keep target layer 18, M1 max-pooling, chunk size 128, sink 16, recent 32, and gamma 15 as defaults.
- Do not modify `src/Benchmark/common/vllm_pilot.py`; it has pre-existing user changes.
- Do not add or run tests in this task; run syntax/static checks only. A B200 runtime is unavailable in this workspace.

---

### Task 1: Per-request chunk-step budget controller

**Files:**
- Create: `src/Benchmark/common/phase_a_budget.py`
- Modify: `src/Benchmark/common/phase_a_selector.py`

**Interfaces:**
- `AdaptiveBudgetController(initial_budget, min_budget, max_budget, chunk_size=128, window_rounds=3, min_gain=0.03)`
- `observe(budget, round_ms, committed_tokens, context_length, draft_ms=None, verification_ms=None)` records only the executed action.
- `choose(context_length) -> tuple[int, str]` returns the next budget and a decision reason.
- Budgets are integer multiples of 128 relative to the configured starting point and are clamped to configured bounds/context length.
- `PhaseAConfig` gains budget-mode and adaptive-bound environment settings. Its default remains fixed.

- [x] Implement config parsing with defaults: mode `fixed`, initial 2048, minimum 512, maximum 8192, chunk step 128, window 3, minimum predicted gain 3%.
- [x] Implement discounted per-budget timing/commit statistics, a local direction search with adaptive step size, a 3-round dwell, and a stable incumbent when neighboring actions do not improve cost per committed token.
- [x] Expose a budget override in the selector without mutating the frozen config or changing fixed-mode defaults.
- [x] Validate malformed mode, bounds, and window settings with actionable `ValueError` messages.

### Task 2: FA4 DFlash round feedback and selector correctness

**Files:**
- Modify: `externals/dflash/dflash/model.py`
- Modify: `src/Benchmark/common/phase_a_selector.py`

**Interfaces:**
- `PhaseAContextSelector.end_verification_round(..., budget_override=None, prefill=False)` returns indices selected for the next draft.
- The generation loop observes the actually executed budget and accepted commit count, then chooses the next budget before selecting its context.
- Every Phase A round trace records executed/next budgets, selected-token count, committed tokens, round wall time, draft CUDA time, verification CUDA time, and controller decision.

- [x] At prompt prefill, score the final prompt query and initialize the first draft context.
- [x] At target verification, score exactly the parent plus accepted query rows and mask key positions causally before pooling.
- [x] Measure each round with `perf_counter`; retain existing CUDA event draft/verify measurements and read them only after `query()` reports completion.
- [x] For adaptive mode, call the controller with the executed budget, wall interval and commits; select next-round indices with the returned budget.
- [x] Preserve the complete draft KV bank and keep all changes behind an active Phase A selector.
- [x] Return the adaptive configuration, budget trajectory and cumulative controller metrics with generation stats.

### Task 3: Benchmark record, server invocation, and operator guide

**Files:**
- Modify: `src/Benchmark/native_fa4_benchmark.py`
- Modify: `docs/baselines/dflash_phase_a.md`
- Modify: `src/Benchmark/README.md`

**Interfaces:**
- Adaptive mode is enabled using `DFLASH_PHASE_A_BUDGET_MODE=adaptive`; fixed mode remains the default.
- Per-sample record includes `phase_a_budget_mode`, initial/min/max budget, selected-token count, budget trajectory, and per-round controller trace.
- Summary names the active Phase A policy and configured bounds.

- [x] Add adaptive controller configuration and runtime policy to sample and summary metadata without changing the common schema envelope.
- [x] Add a representative command and the full B200 command using `bash scripts/run_fa4_benchmark.sh --full --methods vanilla_hf,dflash,dflash_phase_a` with adaptive env settings.
- [x] Document the expected round trace, fixed-mode control command, and limitations on short outputs that provide too few rounds to adapt.
- [x] Run Python syntax compilation on touched Python files, `git diff --check`, inspect CLI/config forwarding, and report that GPU parity/speedup still require the server run.
