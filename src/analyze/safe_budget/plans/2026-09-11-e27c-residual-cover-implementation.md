# E27-C0/C1 Residual-Cover Implementation Plan

## Task 1 — Offline MMR feature reconstruction

- [x] Load E26-R scored JSONL and source JSONL.
- [x] Verify 90 joined documents and six actions per document.
- [x] Reproduce E26-R source cap, Qwen tokenizer, MMR lambda and selected
  indices.
- [x] Compute source-only, front/redundancy, facility and residual features.
- [x] Save deterministic feature cache and manifest.

## Task 2 — C0 policy screen

- [x] Reuse E27-A risk/cost policy semantics without changing action space or
  quality/cost definitions.
- [x] Evaluate all signal families under 20 repeated 3-fold CV.
- [x] Report primary and sensitivity contracts, risk, cost, capture and CIs.
- [x] Compute prior-vs-residual gate per dataset and pooled scope.

## Task 3 — Tests and report

- [x] Add unit tests for residual/facility calculations and no-leakage joins.
- [x] Run compile, focused tests and independent output integrity checks.
- [x] Write a Vietnamese report with direct tables and explicit gate decision.

## Task 4 — Conditional C1

- [x] Do not implement C1: C0 failed the locked gate on all three datasets.
- [x] Mark C1 not run by design and retain C0 as the terminal screen.
