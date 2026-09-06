# Prefix-Utility Alignment Phase Implementation Plan

> **For Codex:** Execute with TDD. Keep target logits as training/analysis labels only; never use target logits as test-time selector features.

**Goal:** Test whether DFlash summarization degradation is explained by target–draft ordering misalignment and whether a tiny frozen-lattice probe can recover meaningful Top-16 prefix utility.

**Experiment directory:** `src/analyze/dflash_residual/`

**Hypothesis:** Summarization preserves candidate coverage but DFlash does not preserve the target preference structure; alignment errors near early prefix positions explain accepted-prefix loss.

**Validation scope:** R1 unit/compile validation, R5 GPU target-logit collection on external T4, offline E11/E12 analysis, and document-disjoint E13 probe training. Canonical control is limited to the eight available GSM8K samples; summarization uses 100 documents per dataset when target-logit collection completes.

**Evaluation design:** E11 reports Kendall tau, Spearman rho, pairwise inversion, JS divergence and target-in-lattice rates. E12 correlates block alignment with MAT_D and oracle gap and reports first-rejection hazards. E13 compares pointwise, pairwise, listwise and prefix-utility-weighted tiny probes using held-out documents; primary metrics are MAT and oracle recovery.

**Architecture:** Add dependency-light alignment metrics and a small probe module. The probe uses only DFlash-side row features at test time; target candidate logits provide supervision during training. No full model fine-tuning and no DFlash2 emulation.

---

## Subtask 1: E11/E12 target–draft alignment metrics

**Implementation:** Add stable rank statistics, JS divergence, per-block alignment summaries, MAT/oracle-gap correlations and first-rejection hazard. Handle target-miss rows explicitly and preserve document/block aggregation.

**Tests:** Hand-built ranking permutations, ties, target-miss rows, block alignment, hazard and deterministic aggregation.

## Subtask 2: GPU target-logit collection

**Implementation:** Reuse `trace_dflash.py --record-target-candidate-logits`. Run 100 summarization documents/dataset and all available eight canonical documents at 1K, Top-16, max-new 16, external T4 Conda env. Validate manifests and no-error rows.

**Tests:** Existing target-logit contract tests plus one-sample smoke.

## Subtask 3: E13 tiny probe objectives

**Implementation:** Add document-disjoint tiny candidate scorer with four fixed objectives: pointwise CE, target-logit pairwise ranking, listwise KL, and prefix-utility weighted listwise KL. Test only on held-out documents; report train/test rows, MAT_D, MAT_probe, MAT_O16 and recovery.

**Tests:** No target-logit feature leakage, document split determinism, candidate restriction, lambda/objective reproducibility and MAT prefix semantics.

## Subtask 4: E11–E13 artifacts [INTEGRATION]

**Implementation:** Add CLI commands, JSON/CSV/Markdown reports and a consolidated report under `outputs/dflash_residual/2026-09-05_prefix_utility/`. Apply gates: E11 alignment gap, E12 alignment–utility relation and E13 recovery >30% before opening a method proposal.

**Expected conclusion:** If alignment is not task-specific, close ranking branch. If alignment is task-specific but probes recover <10%, treat Top-16 oracle as hindsight ceiling. If prefix-utility probe exceeds listwise by >10 points and recovery >30%, open a training-objective proposal only then.
