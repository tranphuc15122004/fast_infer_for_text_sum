# Source-Conditioned Candidate Disambiguation Implementation Plan

> **For Codex:** Execute this plan with TDD; keep the diagnostic selector frozen-lattice and do not claim official DFlash2 results.

**Goal:** Test whether source evidence provides incremental signal for disambiguating already-generated DFlash candidates on summarization.

**Experiment directory:** `src/analyze/dflash_residual/`

**Hypothesis:** Summarization creates source-dependent candidate ambiguity; source lexical/semantic evidence should improve frozen-lattice accepted-prefix recovery beyond DFlash unary ranking, especially on source-novel positions.

**Validation scope:** Offline R5 analysis on the validated 110K-row T4 trace for E6–E8/E10, plus a new T4 pilot trace for E9 that records target logits on DFlash Top-16 candidates. No DFlash2 checkpoint is assumed or emulated.

**Evaluation design:** E6 stratifies rows by exact source-token support. E7 compares unary DFlash ranking with source lexical diagnostic scores. E8 measures frozen-lattice prefix MAT and oracle-normalized recovery. E9 uses target-side candidate logits on 20–30 documents per dataset. E10 performs fixed-protocol leave-one-dataset-out reporting; no learned selector is introduced.

**Architecture:** Add a dependency-light `source_disambiguation.py` analysis module, optional source metadata joins from the representative JSONL, and a collector flag for target logits on existing candidate IDs. Keep the original DFlash trace schema backward-compatible and label all new selector numbers as diagnostic frozen-lattice results.

---

## Shared Scaffold

### Existing infra

- Trace reader/schema: `src/analyze/dflash_residual/io.py`, `schema.py`
- Existing candidate/MAT metrics: `metrics.py`, `prefix_gap.py`
- GPU collector: `trace_dflash.py`
- CLI/reporting: `run.py`, `report.py`
- Existing validated trace: `outputs/dflash_residual/2026-09-04_t4_full/traces/canonical_cnn_gov_multi_clean_trace.jsonl`
- Source records: `data/representative_100/{cnn_dailymail,govreport,multinews}_representative.jsonl`

### Needs setup

- `source_disambiguation.py`: source token index, E6 strata, lexical source scores, diagnostic selector and frozen-lattice path metrics.
- `trace_dflash.py`/`schema.py`: optional `target_candidate_logits` field and collector flag; preserve old rows.
- `run.py`: phases `e6`, `e7`, `e8`, `e9`, `e10`, and `source-next` convenience mode.
- Tests for exact token support, tie handling, selector determinism, path accounting and target-logit extraction.
- Outputs under `outputs/dflash_residual/2026-09-05_source_disambiguation/`.

## Subtask 1: Source alignment and H6 stratification

**Role:** Build reproducible source-copyable versus source-novel labels without changing the GPU trace.

**Implementation:** Load source documents by `sample_id`/record ID, tokenize with the cached Qwen3-4B tokenizer when available, compute exact target-token support for `n=1`, and expose an explicit `source_metadata`/tokenizer provenance field. Treat n-gram labels as unavailable unless the trace contains a valid contiguous target continuation; do not infer multi-token continuation from masked block rows.

**Unit Tests:** Source token present/absent, punctuation and whitespace token handling, missing document behavior, dataset ID normalization, and deterministic joins.

### Steps

1. Write failing source-alignment tests.
2. Run focused tests and verify failure.
3. Implement source index and E6 metric functions.
4. Run focused tests and existing package tests.

## Subtask 2: E7/E8 diagnostic source scoring

**Role:** Determine whether source evidence adds signal beyond DFlash unary ranking.

**Implementation:** Define transparent, non-trained scores:

- `U`: DFlash candidate logit/rank baseline;
- `S_lex`: exact source token support/frequency and optional source phrase support when valid;
- `U+S_lex`: deterministic normalized score with fixed lambda grid.

For every frozen block state, select a candidate from the recorded Top-16 lattice and compute target-prefix survival against recorded verifier target IDs. Report `MAT_U`, `MAT_S`, `MAT_U+S`, `MAT_O16`, `rho_M`, and copyable/source-novel strata. Explicitly label this as a diagnostic on frozen DFlash states, not end-to-end speculative decoding.

**Unit Tests:** Candidate tie-breaking, target-membership accounting, oracle upper bound, per-stratum MAT, lambda determinism and no candidate outside Top-16.

### Steps

1. Write failing metric/selector tests.
2. Run tests red.
3. Implement E7/E8 metrics.
4. Run focused and full package tests.

## Subtask 3: Target-side logits for E9

**Role:** Test whether DFlash mismatches are target near-ties rather than strong target disagreements.

**Implementation:** Add an opt-in collector field `target_candidate_logits` by indexing target verifier logits at the recorded DFlash Top-16 candidate IDs. Do not change the default trace output or candidate generation. Run a pilot of 20–30 documents per summarization dataset at cap 1K with the external T4 Conda env.

**Unit Tests:** Correct candidate-logit indexing, shape validation, finite-value validation and backward-compatible serialization when the flag is disabled.

### Steps

1. Add failing collector contract tests.
2. Run tests red.
3. Implement optional target-logit capture.
4. Run collector tests and a one-sample GPU smoke trace.
5. Run the E9 pilot on T4 and validate manifests.

## Subtask 4: CLI/report integration [INTEGRATION]

**Hypothesis:** Source evidence improves candidate-to-path recovery beyond unary DFlash ranking and the gain is strongest on source-novel spans.

**Components consumed:** Subtasks 1–3, existing trace reader, prefix metrics, report writer and validated T4 artifacts.

**Implementation:** Wire E6–E10 phases, fixed lambda grid, per-dataset and pooled reports, source-stratum tables, frozen-lattice caveat, and leave-one-dataset-out summary. Do not call any diagnostic selector DFlash2.

**Integration Tests:** Synthetic trace plus synthetic source records should generate all JSON/CSV/Markdown artifacts, preserve old phase behavior, and fail explicitly when source documents or target-side logits are unavailable.

**Validation Pyramid:** R1 tests/compileall and R5 real-trace analysis on T4 artifacts. The T4 GPU pilot is required only for E9 target logits.

### Steps

1. Add CLI and report tests before implementation.
2. Run them red.
3. Integrate phases and manifests.
4. Run tests green and compileall.
5. Run E6–E8 and E10 offline on the full validated trace.
6. Run E9 pilot on T4, then analyze it offline.
7. Apply gates: source-novel headroom, `rho_{U+S}-rho_U`, overall recoverability and cross-dataset stability.
8. Write a sober decision report separating evidence from proposal.
