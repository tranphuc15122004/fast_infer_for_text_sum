# E22–E24 Rank-Band Repair Experiments

## Hypothesis

Summarization degradation in DFlash is amplified by a small number of
prefix-critical target tokens whose draft rank falls just outside the fixed
Top-16 candidate budget. Repairing only these shallow rank-tail errors may
recover more accepted prefix than their marginal frequency suggests.

## Scope

The experiment is staged and gate-based:

1. **E22 — Rank-Band Repair Oracle:** offline oracle on the existing fixed
   on-policy traces; no model or training changes.
2. **E23 — Frontier training screen:** only if E22 has headroom, compare the
   existing fixed-decay objective with available D-PACE and Spec-AUF
   implementations under identical data, compute, and inference settings.
3. **E24 — Boundary-aware hard-negative prototype:** only if E23 leaves
   shallow-tail headroom; compare all-mismatch and rank-band-selected
   hard-negative losses against the same frontier baseline.

The current implementation starts with E22. E23/E24 must not be represented
as completed until their training code, checkpoints, held-out evaluation, and
GPU runs exist.

## Variables and controls

### E22 independent variable

The repaired target-rank band, run independently as:

- `[2, 16]`
- `[17, 32]`
- `[33, 64]`
- `>64`

Rank 1 is not repaired. Rows with a missing full-vocabulary rank are excluded
from E22 rather than silently assigned to a band.

### E22 dependent variables

- observed `MAT_D` from the original selected draft path;
- repaired `MAT` after replacing only the selected token at the chosen band;
- absolute and relative MAT gain;
- fraction of rows in the repaired band;
- per-document bootstrap 95% CI for MAT gain and relative gain;
- `R@1`, `R@16`, and `R@16(3:8)` before and after repair.

The repaired prefix is recomputed from the complete recorded block. It is not
computed by adding the number of repaired rows, because longest-prefix
verification makes the effect non-additive.

### Controls

Target model, drafter checkpoint, tokenizer, trace state, block grouping,
context, dtype, and candidate rank instrumentation remain fixed. Band results
are mutually exclusive single-band interventions, not cumulative repairs.

## Data and scale

The first run uses the existing E20 traces:

- `multi_news`, `govreport`, and `cnn_dm`;
- 30 documents per dataset;
- fixed on-policy state with `reveal_count=0`;
- native DFlash block size 16;
- 1,350 rows and 90 blocks per dataset in the current artifacts.

## E22 gate

E22 passes if relative `MAT_[17,32]` gain exceeds 20% on at least two of the
three datasets. The other bands are controls. A pass only justifies moving to
the training screen; it is not a training result or a serving-speed claim.

## E23 design after E22 pass

Use the same target, data split, block size, training token budget, optimizer
budget, and seed set for:

- fixed-decay DFlash historical baseline;
- D-PACE, if its exact implementation is available;
- Spec-AUF, if its exact implementation is available.

Primary metrics are `MAT_D`, `R@1`, `P(17<=rank<=32)`, and `R@16(3:8)`.
Evaluation is checkpoint-based and held-out, with a step-based cadence during
training and final full validation. No custom substitute objective is labeled
D-PACE or Spec-AUF without an explicit implementation note.

## E24 design after E23 pass

Compare under identical training budget and seeds:

1. frontier baseline;
2. frontier + hard-negative loss on all mismatches;
3. frontier + hard-negative loss only for detached ranks 17–32.

The shallow-boundary variant must beat the all-mismatch variant to support the
claim that repairability selection, rather than generic hard-negative mining,
matters.

## Validation scope

- E22: deterministic unit tests, malformed/missing-rank tests, artifact
  validation, and a real-trace CPU run.
- E23/E24: training smoke test, finite-loss/backward check, checkpoint reload,
  held-out evaluation, and B200 run with logs and fixed seeds.
- No tree, wider Top-K serving, causal head, or on-policy adaptation is added
  by this experiment.

## Review criteria

- E22 report must contain all four bands and document-bootstrap intervals.
- E22 gate decision must be derived from the generated metrics, not entered by
  hand.
- E23/E24 must report no NaN/Inf, reproducible seeds, training/evaluation
  budget, and inference cost.
- A proposed method is stopped if screen-1 MAT gain is below 5%; 5–10% is
  reported as weak; only >=10% proceeds to three seeds, with >=15–20% on at
  least two summary datasets considered proposal-level evidence.
