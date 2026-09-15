# E30-B Online Output-Stopping Policy Screen

> **For Codex:** use the experiment-planning skill. This plan has exactly one
> `[INTEGRATION]` subtask.

**Goal:** Kiểm tra xem một policy dừng summary tại sentence boundary có thể khai
thác phần headroom của E30-A bằng các tín hiệu có thể quan sát online hay không.

**Experiment directory:** `outputs/safe_budget_sum/2026-09-13_e30b_policy_screen/`

**Primary hypothesis:** prefix hiện tại và source+prefix coverage dự đoán được
điểm dừng an toàn tốt hơn source length hoặc các heuristic generic; policy cuối
phải đạt ít nhất 60% oracle capture ở contract `(epsilon=0.02, alpha=0.05)` trên
ít nhất 2/3 datasets.

**Scope:** offline, document-grouped cross-validation trên 30 documents/dataset
và các prefix rows đã sinh bởi E30-A. Không train target model, không chạy
inference mới, không dùng reference để tạo feature runtime. Reference chỉ dùng
để tạo nhãn oracle trong calibration/evaluation.

## Experimental contract

- Quality: ROUGE-L recall và local BERTScore recall là metric audit; policy gate
  chính tiếp tục dùng ROUGE-L và BERTScore F1 để giữ khả năng so sánh với E30-A.
- Primary risk contract: `epsilon=0.02`, `alpha=0.05`; report sensitivity cho
  `epsilon in {0.01, 0.02, 0.05}` và `alpha in {0, 0.05, 0.10}` khi khả thi.
- Cost: projected E30-A cost at the selected prefix boundary. This is a
  screening estimate, not fresh online latency.
- Splitting: grouped document-level repeated 5-fold CV; no prefix from a held
  out document enters training/calibration. A deterministic 70/30 holdout is
  also reported for the primary contract.
- Capture: `(C_fixed - C_policy) / (C_fixed - C_oracle)`; report undefined when
  the denominator is non-positive.

## Data and feature availability audit

- Join E30-A `prefix_metrics.csv` with E26-R full rows and the matching
  `data/representative_100/*.jsonl` source documents by `example_id`.
- Available online features: current prefix token count, sentence index,
  current-sentence length, prefix lexical novelty, self-redundancy, semantic
  drift proxy from TF-IDF/cosine, source token length, source sentence count,
  and source-prefix TF-IDF coverage.
- Not available without new inference: EOS probability, token confidence,
  target hidden state, target entropy, and decoder attention. These baselines
  must be reported as unavailable rather than approximated from reference.

## Subtask 1: Metric audit and joined state table

Create a deterministic state table with one row per document × sentence prefix,
including ROUGE-L recall, BERTScore recall, F1 deltas, projected cost, source
features, and all prefix-only/source+prefix features. Validate that the full
prefix is the last row and that source joins cover every E26-R document.

**Expected conclusion:** The quality audit does not silently change the E30-A
contract, and all deployable feature families are explicitly identified.

## Subtask 2: Grouped policy families

Implement and evaluate:

1. best fixed output action;
2. source-length-only budget regression policy;
3. generic prefix heuristics: self-redundancy and semantic-drift proxy;
4. prefix-only supervised stop classifier;
5. source+prefix supervised stop classifier;
6. oracle adaptive policy as an upper bound.

For each policy, select thresholds on training documents only under the primary
risk contract, then apply the selected policy to held-out documents. Use simple
logistic/Ridge models and deterministic feature scaling; avoid a feature search
that would overfit 30 documents per dataset.

**Expected conclusion:** Determine whether state information available after
generation is predictive enough to recover a meaningful fraction of E30-A
headroom.

## Subtask 3: Cross-dataset and sensitivity validation

Report per-dataset risk, cost, output token saving, capture, and bootstrap or
fold-level uncertainty. Run leave-one-dataset-out transfer only for the best
within-dataset policy after the primary screen. Preserve negative results for
unavailable EOS/hidden-state baselines.

**Kill gates:**

- candidate source+prefix capture must be at least 60% on at least 2/3 datasets;
- source+prefix must beat the strongest executable prior/generic policy by at
  least 10 capture points on at least 2 datasets;
- projected E2E saving must be at least 15% over the best deployable fixed
  policy, under the same risk contract.

## [INTEGRATION] Final E30-B pipeline and report

Integrate the joined-state builder, metric audit, grouped policy selection,
primary/sensitivity evaluation, and report writer into one deterministic CLI.
Write `state_table.csv`, `policy_metrics.csv`, `fold_metrics.csv`,
`dataset_summary.json`, `run_manifest.json`, and a detailed `report.md` that
separates confirmed, exploratory, unavailable, and failed/incomplete claims.

**No E30-C runtime inference is allowed unless the E30-B gates pass.**
