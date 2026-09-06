# Joint Candidate-Lattice Degradation Implementation Plan

> **For Codex:** execute with TDD; do not infer a method before E14/E14b/E15 gates.

**Goal:** Decompose summarization MAT_O16 degradation into marginal recall, joint prefix coherence and intrinsic target entropy, then test minimal task-matched DFlash adaptation only if warranted.

**Experiment directory:** `src/analyze/dflash_residual/`

**Hypothesis:** Summarization reduces contiguous Top-16 prefix survival beyond what marginal candidate recall and target entropy explain; original-objective task adaptation is the minimal causal diagnostic.

**Validation scope:** offline unit/compile tests, E14 on existing valid traces, E14b on T4 target-entropy traces, and E15 T4 training/inference only if E14b passes its continuation gate. B200 is not used.

**Evaluation design:** document-level aggregation, fixed candidate lattice, explicit finite-value checks, reproducible seeds, held-out dataset evaluation for E15, and a consolidated Vietnamese report.

**Architecture:** E14/E14b are analysis-only. E15 reuses `src/MR_DFlash` with original DFlash CE/position decay, warm-started from the existing Qwen3-4B DFlash checkpoint; no new selector, source feature, or prefix loss.

## Shared scaffold

- Existing valid traces: `outputs/dflash_residual/2026-09-05_prefix_utility/e11_*.jsonl`.
- Existing trace collector: `src/analyze/dflash_residual/trace_dflash.py`.
- Existing DFlash training copy: `src/MR_DFlash/`.
- Output directory: `outputs/dflash_residual/2026-09-06_joint_lattice/`.

## Subtask 1: E14 marginal/joint decomposition

Implement `joint_lattice.py` and CLI with `R_j`, `J_j`, `C_j`, canonical-coherence counterfactual, decomposition fractions and document bootstrap. Add synthetic tests for independent/perfectly coherent blocks and conservation of decomposition.

## Subtask 2: E14b target-entropy control

Extend trace schema/collector with finite scalar `target_entropy` and `target_top1_probability`. Collect 8 canonical + 50 documents per summarization dataset at 1K/Top-16/bfloat16. Implement shared entropy-bin standardization for prefix survival and MAT. Add tests for bin overlap, entropy metric and deterministic standardization.

## Subtask 3: E15 minimal adaptation diagnostic

Prepare reference-grounded summarization conversations for `src/MR_DFlash`, capture frozen target features, warm-start the original DFlash draft, train only the original objective for a short T4-safe run, export a serving-compatible draft checkpoint, and run held-out lattice traces. Reject the run if checkpoint/key/finite/parity contracts fail.

## Subtask 4: E14–E15 integration artifacts [INTEGRATION]

Run E14/E14b, apply gates, conditionally run E15, aggregate metrics/manifest/logs, write `joint_lattice_report.md`, run full tests/compile/diff checks, and record whether the training branch is opened or closed.

