# Phase 1 Dataset Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full-source, non-sampling Phase 1 dataset analyzer with reproducible distribution figures and remove implicit prepare/build execution from the vLLM launcher.

**Architecture:** `analyze_phase1_dataset.py` streams each raw source independently and writes analysis-only artifacts. The existing prepare/build path remains available for a later user-approved dataset construction step. The B200 vLLM launcher validates an already prepared root and starts at regeneration, so no sampling can happen implicitly before vLLM/cache.

**Tech Stack:** Python 3.12, standard library JSON/CSV/statistics, local Transformers tokenizer when provided, matplotlib/seaborn for PNG figures, pytest fixtures.

**Spec:** `docs/superpowers/specs/2026-09-22-phase1-dataset-analysis-design.md`

## Global Constraints

- No internet and no model download; tokenizer must use `local_files_only=True`.
- Read raw inputs in streaming mode; never load the full raw dataset into memory.
- Do not random sample or create train/val/test artifacts during analysis.
- Do not run vLLM, regeneration, tokenization, or hidden-cache from the analyzer.
- Existing prepare/build behavior remains available for a later explicit decision.
- Use a small fixture test before any full server scan.

### Task 1: Analyzer contract tests

**Files:**
- Create: `tests/test_phase1_dataset_analysis.py`
- Create: `scripts/mr_dflash/analyze_phase1_dataset.py`

**Interfaces:**
- `analyze_source(path, source, tokenizer=None) -> iterator[dict]`
- `analyze_dataset(sharegpt_source, arxiv_source, output_root, tokenizer=None, max_records=None) -> dict`
- CLI arguments `--sharegpt-source`, `--arxiv-source`, `--output-root`, `--tokenizer`, `--max-records` (smoke-only limit), and `--local-files-only`.

- [x] Write fixture tests for full deterministic scan, source-specific fields, duplicate/malformed rows, quantiles, and no train/val/test outputs.
- [x] Run the focused tests and confirm the expected missing-module failure.
- [x] Implement streaming record normalization and report aggregation.
- [x] Run the focused tests to green.

### Task 2: Reproducible distribution figures

**Files:**
- Modify: `scripts/mr_dflash/analyze_phase1_dataset.py`
- Test: `tests/test_phase1_dataset_analysis.py`

- [x] Add token-length histogram/ECDF, source-composition, and conversation-structure figures.
- [x] Use deterministic bins and a non-interactive matplotlib backend.
- [x] Assert figure files exist and are non-empty when matplotlib is available.

### Task 3: Separate launcher prepare/build from regenerate/cache

**Files:**
- Modify: `scripts/mr_dflash/run_b200_vllm_phase1.sh`
- Modify: `tests/test_b200_launcher_contract.py`

- [x] Add an explicit comment and validation that the launcher consumes prepared artifacts only.
- [x] Ensure no launcher command invokes `prepare_server_data.py` or `build_pilot_dataset.py`.
- [x] Keep the existing symlink/check behavior and `--from-stage regenerate_full_train` boundary.
- [x] Add a launcher contract assertion for the separation.

### Task 4: Documentation and smoke validation

**Files:**
- Modify: `scripts/mr_dflash/README_b200_vllm_phase1.md`
- Modify: `docs/mr_dflash_phase1_pipeline_v2.md`

- [x] Document the analysis-only command and output layout.
- [x] Document that sampling/split happens only after user approval through `prepare_server_data.py`.
- [x] Run focused pytest, shell syntax checks, and the small fixture analyzer.
- [x] Check raw server source paths; run the full scan only when mounted and report output/figures.

Current workspace does not have the two canonical `/workspace/storage-shared/...`
raw paths mounted, so the full scan is intentionally not run here.
