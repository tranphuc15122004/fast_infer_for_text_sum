# E27-A Policy Learnability Screen Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:ml-subagent-dev to implement this plan task-by-task.

**Goal:** Determine whether source-only document features can recover a useful fraction of the E26-R adaptive context-budget headroom without using target outputs, hidden states, or new inference.

**Experiment directory:** `src/analyze/safe_budget/`

**Hypothesis:** A source-only multi-action risk/cost router can capture at least 60% of the hindsight adaptive savings under `epsilon=0.02`, `alpha=0.05` on at least two datasets; if it cannot, the SafeBudget controller branch is closed before E27-B training/calibration.

**Validation scope:** Offline policy-learning screen over the 90 paired documents and 540 scored action outcomes from E26-R. Variants are evaluated with repeated document-level stratified splits. Primary comparison is best fixed, length-only, a deterministic length heuristic, learned cheap-feature risk/cost routing, and hindsight adaptive oracle. No target model inference is run.

**Evaluation design:** One shared evaluator computes per-document quality violation, measured action cost, policy risk, capture, and normalized cost. The outer evaluation uses 3-fold stratified document splits repeated over fixed seeds; models fit only on the training documents in each fold. The fixed test-set optimum and adaptive oracle are reported as upper-bound references, while train-selected fixed policy is reported as a deployment-like control. Quality uses both ROUGE-L and BERTScore F1 with `epsilon=0.02`; risk is the fraction of documents violating either metric, with `alpha=0.05` as primary and sensitivity over existing E26-R contracts.

**Architecture:** A pure offline evaluator extracts source-only cheap features (length, sentence statistics, lexical redundancy, and section structure). A length-only quantile policy and a linear per-action risk/cost router are compared. The learned router selects the cheapest action whose predicted violation risk is no greater than alpha; if none is eligible it selects full context. No conformal guarantee is claimed at this stage.

---

## Shared Scaffold

### Existing infra (don't touch, advise if problems found)

- E26-R paired outcomes: `outputs/safe_budget_sum/2026-09-11_e26r_full/scored/*.jsonl`
- E26-R quality/cost schema and risk definition: `src/analyze/safe_budget/e26r_risk_oracle.py`
- Original documents: `data/representative_100/*_representative.jsonl`
- Local Qwen tokenizer for source feature tokenization.

### Needs setup

- `src/analyze/safe_budget/e27a_policy_screen.py`: feature extraction, fold fitting, policy evaluation, bootstrap/CV aggregation, report generation.
- `src/analyze/safe_budget/tests/test_e27a_policy_screen.py`: deterministic unit tests for features, risk labels, policy selection, and capture.
- `outputs/safe_budget_sum/2026-09-11_e27a_policy_screen/`: JSON, CSV, report, feature snapshot, and manifest.

## Subtask 1: Offline policy evaluator

**Role:** Build the reusable evaluator for paired E26-R action outcomes and source-only features.

**Implementation:** Load one full row and five MMR rows per document; join documents by ID; compute quality violation from both required metrics; implement fixed, length-only, heuristic, and learned per-action risk/cost policies. Use only train-fold rows for fitting; use actual test-fold action cost only after an action is selected. Report both test-set best-fixed/oracle references and train-selected controls.

**Unit Tests:** Feature determinism, all-full fallback, violation OR semantics, no test-row fitting, and capture denominator handling.

**Expected Conclusion:** Evaluator produces finite results for a synthetic paired dataset and rejects missing action/quality rows.

## Subtask 2: E27-A report and integration [INTEGRATION]

**Hypothesis:** At least one source-only router captures `>=60%` of the E26-R oracle-vs-fixed headroom while satisfying empirical test risk on at least two datasets.

**Components consumed:** Subtask 1 evaluator; E26-R scored JSONL; representative source JSONL.

**Implementation:** Add CLI, feature snapshot, repeated 3-fold document-level evaluation, sensitivity contracts, JSON/CSV/Markdown outputs, exact seeds, and leakage audit. The report must contain all fold-level and pooled metrics, action counts, quality/risk definitions, and explicit gate decisions.

**Integration Tests:** Run the CLI on a tiny synthetic dataset and on the real 540-row inputs; verify all variants are finite, all test rows are assigned exactly one action, and the report includes the primary contract and limitations.

**Validation:** Offline only; no GPU/model generation. Primary contract `epsilon=0.02`, `alpha=0.05`; sensitivity `epsilon∈{0.01,0.02,0.05}`, `alpha∈{0,0.05,0.10}`. Report mean risk/cost/capture and 95% bootstrap CI over outer-fold document results.

**Expected Conclusion:** Continue to E27-B only if a learned source-only policy meets `capture>=60%` on at least two datasets and empirical risk does not exceed alpha; otherwise close the adaptive-controller branch while retaining E26-R fixed-compression results.

