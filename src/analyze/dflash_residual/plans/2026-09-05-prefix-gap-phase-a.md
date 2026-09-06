# Prefix-Gap Phase A Implementation Plan

> **For Codex:** Execute this plan with TDD; keep all Phase A analysis offline on the existing T4 traces.

**Goal:** Determine whether summarization has a large gap between marginal Top-K candidate recall and usable joint speculative-prefix survival, and whether the gap is explained by target-token rank/logit ambiguity.

**Experiment directory:** `src/analyze/dflash_residual/`

**Hypothesis:** Summarization preserves many correct Top-16 candidates but produces fragile joint prefixes and deeper/less confident target ranks than canonical workloads.

**Validation scope:** R1 static/unit validation plus offline R5 representative-trace analysis. No new GPU run is required for E1–E3; E4 remains gated on the results and a DFlash2 selector trace.

**Status:** E1, E2/E2b and E3 implemented and completed on the real trace. Unit/compile validation passed (`41 passed`). E2 opens the E4 gate; E4 itself is intentionally not included in this phase.

**Evaluation design:** Use the existing validated trace; filter E1 to `context_cap=1024`; compute E2/E2b per block and draft position for K∈{1,4,8,16}; compute E3 on rows whose target is in Top-16; use deterministic document/block aggregation and explicit empty/sparse handling. Outputs must include JSON, CSV and Markdown reports.

**Architecture:** Add dependency-light metric functions in a new `prefix_gap.py` module, expose E1/E2/E3 through the existing `run.py` CLI, and add phase-specific report metadata without changing the GPU collector contract. Preserve P0–P4 behavior.

---

## Shared Scaffold

### Existing infra

- Trace reader/schema: `src/analyze/dflash_residual/io.py`, `schema.py`
- Existing block grouping and MAT/recall metrics: `metrics.py`
- Existing orchestrator: `run.py`
- Existing report writer: `report.py`, `io.write_metrics_bundle`
- Existing tests: `src/analyze/dflash_residual/tests/`
- Validated trace: `outputs/dflash_residual/2026-09-04_t4_full/traces/canonical_cnn_gov_multi_clean_trace.jsonl`

### Needs setup

- `prefix_gap.py`: E1 matched-context, E2/E2b prefix oracle/independence/conditional survival, E3 rank/logit/entropy metrics.
- `run.py`: CLI phases `e1`, `e2`, `e3`, and `phase next` convenience mode if useful; preserve existing phases.
- `report.py`: human-readable phase labels and decision interpretation.
- Tests for every new public metric and CLI output.
- Output directories under `outputs/dflash_residual/2026-09-05_prefix_gap/`.

## Subtask 1: Prefix survival and oracle metrics

**Role:** Implement the central E2/E2b quantities without GPU.

**Implementation:** Add functions for block-level hit sequences, `S_K(j)`, independence baseline `S_K^ind(j)`, conditional survival `c_K(j)`, Top-K oracle prefix length, `MAT_O_K`, and `MAT_D`. Keep K validation and missing-row behavior explicit.

**Unit Tests:** Construct deterministic two-block fixtures where joint survival is lower than, equal to, and higher than the independence baseline; verify longest-prefix semantics and MAT calculations.

### Steps

1. Write failing tests in `tests/test_prefix_gap.py`.
2. Run the focused tests and verify the expected missing-symbol failure.
3. Implement the minimal metrics in `prefix_gap.py`.
4. Run focused tests, then the existing metrics/phase tests.

## Subtask 2: E1 matched-context and E3 rank ambiguity

**Role:** Implement matched-context task comparison and candidate ranking/logit anatomy.

**Implementation:** Add context-cap filtering; compute canonical-vs-summarization MAT comparison at cap 1K with document-bootstrap CIs; compute target rank distribution, MRR, target logit deficit, Top-16 entropy, Top-1/Top-2 margin, and per-regime summaries.

**Unit Tests:** Verify context filtering, bootstrap determinism, rank/deficit/entropy on hand-built logits, and correct exclusion of target-miss rows from rank-conditioned metrics.

### Steps

1. Extend the failing test file with E1/E3 fixtures.
2. Run tests to confirm failures.
3. Implement E1/E3 functions.
4. Run focused tests and existing package tests.

## Subtask 3: CLI integration and artifacts [INTEGRATION]

**Hypothesis:** The assembled offline analysis will determine whether summarization is prefix-fragile and/or rank-ambiguous using the existing real trace.

**Components consumed:** `prefix_gap.py`, existing `run.py`, `io.py`, `report.py`, validated trace artifacts.

**Implementation:** Add `e1`, `e2`, `e3` phase dispatch, output JSON/CSV/Markdown, optional combined `next` run, and a manifest recording trace, context filter, K values, seed and bootstrap count. Do not modify P0–P4 metric semantics.

**Integration Tests:** Run the synthetic trace through each new phase and verify artifact existence, deterministic metrics, and nonzero exit only for unavailable input.

**Validation Pyramid:** R1 package tests/compileall; R5 offline analysis on the 110K-row real trace. E4 GPU work is explicitly not part of this integration task.

### Steps

1. Add parser/dispatch tests before implementation.
2. Run them red.
3. Wire phase dispatch and report artifacts.
4. Run tests green and compileall.
5. Run E1/E2/E3 on the real trace with Conda `myenv`.
6. Review metrics for empty groups, context leakage, and sample-size accounting.
7. Write the Phase A decision report and identify whether E4 is justified.
