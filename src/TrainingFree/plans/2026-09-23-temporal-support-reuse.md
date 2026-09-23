# E43 Temporal Support Reuse — Implementation Plan

**Goal:** Kiểm tra liệu exact source support ở refresh token có thể tái sử dụng cho các token kế tiếp với GQA-aware cost dưới 50% và rủi ro attention thấp hay không.

**Experiment directory:** src/TrainingFree/

**Hypothesis:** Previous exact support remains locally persistent for 4–8 decode steps; periodic refresh plus per-head adaptive K95 blockification therefore yields recurrent source-cost ratio <0.50, mean missed mass ≤1%, and P99 missed mass ≤5% on the 40-document development cohort.

**Validation scope:** R0/R1 local deterministic tests, R5 Modal/trace pilot on Qwen3-0.6B layer 27, 20 gov_report + 20 multi_news, max 32 generated tokens. No physical KV mutation unless the recurrent gate passes.

**Evaluation design:** One shared temporal analyzer is used by offline trace-compatible code and the Modal collector. The primary decision is recurrent GQA-aware source-cost ratio at R∈{4,8} under adaptive K95 block cache; fixed-budget block cache is a targeted follow-up because it is a separate policy variant. Lag/budget/GQA union are diagnostics that explain the decision. Actual masked generation is launched only for a candidate that passes the recurrent gate.

**Architecture:** Keep full dense attention as the reference. At each refresh, select per-query-head support using alpha × K95, map supports through the 16-query/8-KV GQA mapping, union/round to contiguous blocks, and reuse the resulting KV-group working support until the next refresh. The first phase computes exact current attention only for audit; it never changes model execution.

---

## Experiment card

| Field | Locked value |
|---|---|
| Research question | Does previous exact source support transfer across decode steps under GQA? |
| Baseline | Dense exact attention from existing e42-oracle-hybrid-head-20260921 protocol |
| Variants | lag 1,2,4,8,16; fixed source budgets 10,20,30,40%; adaptive alpha×K95, alpha 1.0,1.25,1.5; refresh R=2,4,8,16; block 16,32,64 |
| Primary metric | Recurrent GQA-aware source cost ratio |
| Guardrails | Mean missed mass ≤1%, P99 missed mass ≤5%, no invalid/empty supports; report attention-context error when available |
| Data | Same 40-document dev cohort; no holdout claim |
| Seeds | Deterministic seed 42; attention traces are greedy |
| Budget | One Modal A10 pilot, max 32 tokens/document; no second GPU run unless trace instrumentation is insufficient |
| Success criterion | At least one fixed configuration with cost <0.50 at R=4 or R=8 and both missed-mass guardrails |
| Stop rule | If no configuration passes, close source-selection branch and do not implement physical KV |
| Exploratory-only | Lag overlap, query-head oracle, GQA union oracle, and context error do not prove E2E speedup |

## Subtask 1: Temporal/GQA analyzer core

**Role:** Deterministically compute support selection, GQA union, lag transfer, adaptive K95, blockification, and recurrent metrics from one attention step.

**Files:** src/TrainingFree/temporal.py, src/TrainingFree/tests/test_temporal.py.

**Expected conclusion:** Core metrics pass deterministic unit tests, including GQA union being at least as large as a query-head support.

## Subtask 2: Collector integration and trace contract

**Role:** Compute E43 metrics during the existing eager-attention collection without changing model KV execution.

**Files:** src/TrainingFree/hierarchy_collector.py, src/TrainingFree/schema.py, src/TrainingFree/run.py, src/TrainingFree/tests/test_hierarchy_collector.py, src/TrainingFree/tests/test_hierarchy_schema.py.

**Expected conclusion:** Existing V3 behavior remains unchanged when temporal analysis is disabled; temporal traces validate when enabled.

## Subtask 3: Modal launcher and report analyzer

**Role:** Add a temporal experiment mode, forward the locked sweep configuration, and write a compact report with per-dataset and per-configuration gates.

**Files:** scripts/modal_trainingfree.py, scripts/analyze_trainingfree_e43.py, tests/test_modal_trainingfree_contract.py, tests/test_trainingfree_e43.py.

**Expected conclusion:** One Modal command can run the entire sweep and produce reproducible JSON/CSV/Markdown artifacts.

## Subtask 4: E43 integrated Modal experiment [INTEGRATION]

**Hypothesis:** Periodic exact refresh plus adaptive GQA-aware block reuse yields a sub-50% source-cost configuration without violating mass guardrails.

**Components consumed:** temporal analyzer, collector integration, schema, runner, Modal launcher.

**Implementation:** Run the locked 40-document pilot on Modal using existing Qwen3-0.6B checkpoint and layer 27. Do not modify physical KV. If a candidate passes, run a separate 10+10 masked-generation validation; otherwise stop before physicalization.

**Validation:** R0/R1 tests, then Modal R5 pilot. Record run ID, GPU, model, seed, config, sample coverage, per-dataset metrics, preemption/retry events, and final decision.

**Expected conclusion:** Either promote one candidate to masked-generation validation or close the source-selection branch with evidence.

## Acceptance checklist

- [x] GQA-aware oracle union is measured, not inferred from query-head expansion.
- [x] Temporal lag metrics never use current exact support as the policy support.
- [x] Refresh steps are the only steps allowed to inspect exact current source attention.
- [x] Recurrent cost includes dense refresh cost and GQA block union cost.
- [x] No physical KV mutation is implemented before the locked success criterion passes.
- [x] Final report separates confirmed, exploratory, failed/incomplete, highest verified rung, evidence gaps, and one recommended next experiment.

## Execution status (2026-09-23)

- [x] Subtask 1: analyzer core and unit tests.
- [x] Subtask 2: temporal collector, schema and runner integration.
- [x] Subtask 3: Modal launcher and E43 analyzer.
- [x] Subtask 4: 40-document adaptive pilot plus 10+10 fixed-budget targeted pilot.
- [ ] Masked generation and physical KV: intentionally not run because no recurrent candidate passed the locked gate.
