# Implementation plan — E19–E21 gain-first causal screening

## Subtask 1: Trace contract for rank and fixed-state metadata

Modify `trace_dflash.py` so a collector can record the full-vocabulary rank of
the target token as a scalar, without storing full logits. Preserve existing
Top-M contract and add deterministic tests for rank fields.

## Subtask 2: Fixed state-bank preparation and GPU collector

Add `causal_screening.py` with:

- state-bank preparation from real representative JSONL;
- target-greedy on-policy prefix generation;
- reference-prefix token construction;
- one-block fixed-state collection;
- reveal conditions `r={0,1,2,4}`;
- manifest/progress/error accounting.

Use local snapshots and external Conda runtime. Every output row records
`fixed_state_id`, `state_mode`, `prefix_length`, `reveal_count`, and rank.

## Subtask 3: Offline E19–E21 analysis

Add `causal_screening_analysis.py` with:

- fixed-state paired bootstrap;
- candidate-depth sweep K=16/32/64/128;
- conditional reveal-prefix oracle;
- training topology audit from real loss masks;
- JSON/CSV/Markdown report with explicit gates.

## Subtask 4: Integration run

Prepare 30-document banks for CNN/DM, GovReport and Multi-News; collect on-policy
and reference fixed states plus reveal sweep on T4; run all offline analyzers;
write `metrics.json`, `metrics.csv`, `report.md`, `run_manifest.json` under
`outputs/dflash_residual/2026-09-07_causal_screening/`.

E22 is not part of this integration run. It is conditionally closed until the
screening report passes the locked gate.

## Verification

Run `py_compile`, targeted unit tests, full `src/analyze/dflash_residual/tests`,
manifest row/error checks, and verify all GPU manifests use external Conda,
T4, bf16/SDPA, and zero error rows.
