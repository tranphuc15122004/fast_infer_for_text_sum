# E30-A Output-Prefix Utility Oracle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:ml-subagent-dev to implement this plan task-by-task.

**Goal:** Định lượng headroom của việc dừng summary tại sentence boundary theo oracle, so với các fixed output budgets dưới cùng quality-risk contract.

**Experiment directory:** `outputs/safe_budget_sum/2026-09-12_e30a_output_prefix_oracle/`

**Hypothesis:** Một số summary có prefix ngắn đạt chất lượng gần full output; policy adaptive theo document có thể tiết kiệm E2E decode cost nhiều hơn best fixed output budget.

**Validation scope:** E30-A offline trên 30 documents/dataset của E26-R (`cnn_dailymail`, `govreport`, `multi_news`), dùng full Qwen3-4B generations đã đo trên T4; quality gồm ROUGE-L và local BERTScore F1; có bootstrap document-level và deterministic 70/30 holdout.

**Evaluation design:** Sentence-boundary prefixes; fixed actions `{64,96,128,160,192,224,full}`; contracts `epsilon={0.01,0.02,0.05}`, `alpha={0,0.05,0.10}`; cost projected từ full measured prefill/decode; adaptive oracle chọn prefix sớm nhất an toàn cho từng document; fixed chọn một action chung có empirical risk thấp nhất dưới alpha.

**Architecture:** Không train và không chạy inference mới. Loader đọc E26-R full rows, segment summary, tokenize prefix bằng tokenizer Qwen3-4B local, chấm ROUGE-L và local BERTScore F1, sau đó tổng hợp oracle/fixed/CI/holdout vào JSON/CSV/Markdown. BERTScore implementation giữ đúng implementation local đã dùng trong E26-R để tránh đổi metric giữa phases.

---

## Shared Scaffold

### Existing infra (don't touch, advise if problems found)
- E26-R full scored JSONL: `outputs/safe_budget_sum/2026-09-11_e26r_full/scored/*.jsonl`
- ROUGE implementation: `scripts/common/rouge.py`
- Local BERTScore implementation: `src/analyze/safe_budget/augment_bertscore.py`
- Local Qwen tokenizer: `/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`
- Local BERTScore encoder: `/home/tuantb/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b`

### Needs setup
- `src/analyze/output_stopping/e30a_output_prefix_oracle.py`: core offline analysis and report writer.
- `src/analyze/output_stopping/tests/test_e30a_output_prefix_oracle.py`: deterministic tests for segmentation, boundary policy, safety, fixed selection, and bootstrap primitives.
- Output artifacts under `outputs/safe_budget_sum/2026-09-12_e30a_output_prefix_oracle/`.

## Subtask 1: Prefix extraction and fixed/adaptive oracle core

**Role:** Chuyển full summary thành prefix records, tính quality/cost/risk cho từng document và policy.

**Implementation:** Implement pure functions in `e30a_output_prefix_oracle.py`; keep all thresholds explicit and serialize fallback/overflow diagnostics.

**Unit Tests:** Sentence spans, cumulative prefix lengths, fixed budget selection, epsilon safety and document violation.

**Expected Conclusion:** Core functions deterministic and unit-tested.

## Subtask 2: Metric scoring and aggregate validation

**Role:** Chấm ROUGE-L/local BERTScore, bootstrap document CI và holdout analysis.

**Implementation:** Reuse `scripts.common.rouge` and `LocalBERTScorer`; preserve full prefill/decode timing and calculate projected cost exactly as declared.

**Unit Tests:** Quality delta, projected cost, percentile CI, fixed policy risk, and adaptive-vs-fixed headroom.

**Expected Conclusion:** Offline analyzer produces complete raw and aggregate artifacts without model inference.

## Subtask 3: Final E30-A pipeline and report [INTEGRATION]

**Hypothesis:** Oracle sentence-boundary stopping has at least 20% projected E2E savings on two datasets and exceeds best risk-matched fixed output budget by at least 10–15%.

**Components consumed:** Subtasks 1–2 in `src/analyze/output_stopping/e30a_output_prefix_oracle.py`.

**Implementation:** Add CLI, load the three E26-R JSONL files, compute all contracts, bootstrap/holdout, write `metrics.json`, `metrics.csv`, `run_manifest.json`, and detailed `report.md`. No E30-B/C is run automatically.

**Integration Tests:** Synthetic JSONL end-to-end test plus real artifact preflight validating 30 complete full rows per dataset.

**Validation Pyramid:** Offline equivalent: 90-document primary analysis, bootstrap CI, 70/30 holdout; no training or new target inference is claimed.

**Evaluation contract:** Primary contract `(epsilon=0.02, alpha=0.05)`; all other epsilon/alpha cells are sensitivity analysis. Fixed policy is selected once per dataset/contract; adaptive oracle is hindsight upper bound and is not a deployable controller.

**Expected Conclusion:** Pass only if both oracle savings and adaptive-over-fixed headroom gates are met on at least two datasets; otherwise close E30 stopping branch and retain the negative/feasibility result.

