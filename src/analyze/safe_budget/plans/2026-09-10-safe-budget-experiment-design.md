# Risk-Calibrated Semantic Budgeting — Experiment Design

**Goal:** Determine whether per-document input-context budgets provide
substantial compute savings beyond the best fixed semantic-selection budget,
while preserving a pre-specified quality tolerance against full-context
summarization.

**Experiment directory:** `src/analyze/safe_budget/`

**Hypothesis:** Long-document summarization instances have heterogeneous safe
source-context requirements. An oracle that chooses the smallest quality-safe
budget per document should save at least 20% target compute relative to the
best fixed budget at the same quality tolerance.

**Current execution constraint:** The repository contains real Qwen3-4B
semantic-selection outputs, but the available files currently cover only
`full` and `tokens_512` for the MMR selector on five documents per dataset.
They do not cover the requested six-point budget grid. The current Conda
environment has PyTorch CUDA unavailable despite `nvidia-smi` seeing a T4, and
there is no local Qwen3-4B target checkpoint for a new sweep. Therefore the
first executable artifact is an E26 pilot/preflight, not a full E26 claim.

## E26-R corrected experiment card — 11/09/2026

**Question:** Khi fixed và adaptive bị ràng buộc bởi cùng một document-level
risk contract, adaptive per-document context budget có giảm selector-inclusive
E2E cost đáng kể hơn best fixed policy hay không?

**Hypothesis:** Với cùng

\[
P(\text{quality violation})\le \alpha,
\]

oracle adaptive sẽ có mean `pipeline_e2e_ms` thấp hơn fixed policy và tạo ít
nhất 15–20% headroom trên tối thiểu hai dataset.

**Baseline:** MMR semantic selection với cùng target model, tokenizer, chat
template, greedy decoding, source budget grid và `pipeline_e2e_ms` như cost.

**Variant:** Chỉ thay policy budget: một label cố định cho fixed, hoặc một
action/document cho adaptive oracle. Không thay selector/model trong cùng run.

**Primary metric:** Mean selector-inclusive `pipeline_e2e_ms` dưới risk
contract; adaptive headroom là metric quyết định được suy ra từ primary.

**Guardrails:** empirical document risk không vượt `alpha`; từng quality
metric không được giảm quá `epsilon`; không dùng mean quality để thay risk.

**Data/split:** tối thiểu 30 documents/dataset, cùng document IDs giữa mọi
budget, retention labels `full`, `ratio_0.25`, `ratio_0.4`, `ratio_0.55`,
`ratio_0.7`, `ratio_0.85`. Không được dùng các record thiếu ROUGE-L hoặc
BERTScore.

**Seeds:** inference greedy và selector deterministic; full study cần ghi seed
và paired document IDs. Pilot hiện tại không đủ để ước lượng CI đáng tin cậy.

**Budget:** phân tích offline không GPU; inference full study phải chạy trên
Conda T4 khi CUDA/runtime/model contract hoạt động. Output target tối thiểu
128 hoặc 256 token cho long-document quality audit.

**Cheapest useful rung:** chạy strict schema preflight với ROUGE-L+BERTScore;
nếu thiếu metric thì dừng strict rung, sau đó chạy diagnostic ROUGE-L-only với
nhãn rõ ràng.

**Success criterion:** E26-R full chỉ pass nếu đủ sample/grid/quality và
adaptive headroom ≥15–20% trên ít nhất hai dataset ở risk-matched condition.

**Exploratory-only:** Kết quả 5 document/2 budget, ROUGE-L-only, speedup pilot,
và hindsight oracle không phải bằng chứng cho conformal guarantee hay policy
triển khai.

---

## Experiment stages

### E26 — Dynamic-budget oracle

**Independent variables:** selector and source budget. The planned primary
selector is existing MMR; the planned budget grid is 25%, 40%, 55%, 70%, 85%,
100% of each document's tokenized source, rounded to the runner's valid budget.

**Quality:** primary per-document ROUGE-L F1 against the supplied reference.
ROUGE-1 and ROUGE-2 are secondary. The default tolerance is
`epsilon_rougeL = 0.02` absolute ROUGE-L below the same document's full-context
output. Sensitivity is reported for 0.00, 0.01, 0.02 and 0.05.

**Cost:** selector-inclusive `pipeline_e2e_ms`, with
`pipeline_ttft_ms` as a secondary cost. Full-context output is the per-document
reference cost.

For document `i`, the observed safe budget is the smallest budget `b` for which
every observed budget `b' >= b` satisfies:

\[
Q_{i,b'} \ge Q_{i,full}-\epsilon.
\]

This conservative closure prevents selecting a budget merely because of a
non-monotonic generation fluctuation. If the largest available budget fails,
the document is not assigned a safe budget.

Primary oracle comparison:

- `best fixed`: smallest budget whose quality constraint holds for every
  document in the evaluation set;
- `oracle adaptive`: per-document conservative safe budget;
- adaptive headroom:

\[
H_{adapt} =
\frac{C_{fixed}-C_{oracle}}{C_{fixed}}.
\]

The E26 gate is `H_adapt >= 20%` on at least two summary datasets, with the
same selector and quality tolerance. The full E26 also requires at least six
budget levels and at least 30 documents per dataset; the current pilot does
not satisfy this completeness requirement.

### E27 — Safe-budget predictor

Only open after a passing full E26. Use source-only features (length,
redundancy, sentence count/structure and selector score statistics) and a
small regressor. Fit on a calibration split and add a one-sided conformal
upper residual margin. Compare best fixed, uncalibrated predictor and
conformal predictor using average source tokens, quality-risk violations and
selector-inclusive E2E latency.

E27 gate: capture at least 70% of oracle adaptive savings while keeping the
pre-specified quality-risk violation rate inside the calibration contract.

### E28 — Systems evaluation

Only open after E27 passes. Compare full-context + modern decode baseline,
best fixed compression + the same decode backend, and safe-budget policy +
the same backend. Report selector cost, target prefill, decode, TTFT, E2E
latency, peak memory, ROUGE and tokens/s. A compression ratio alone is not an
E28 result.

---

## Controls and reproducibility

- Same document IDs and references across all budget conditions.
- Same target model, tokenizer, chat template, decoding mode and output limit.
- No quality comparison between different selector algorithms in the primary
  E26 gate.
- Per-document pairing is mandatory; aggregate means alone are insufficient.
- Report count, missing conditions, budget coverage, quality deltas and cost
  deltas explicitly.
- E26 offline analysis must not be described as a new target-model inference
  run.
- E27/E28 require real checkpoints and logs; dry-run/config validation is not
  evidence of method quality.

## Validation scope

- E26 analyzer: deterministic unit tests, schema/missing-budget tests and a
  real-output pilot.
- E27: calibration leakage test, split reproducibility, finite predictions and
  held-out risk audit.
- E28: target/checkpoint compatibility, exact output contract, warmup policy,
  selector-inclusive timing and repeated latency measurements.

## Expected decision

If the complete six-budget E26 does not show 20% adaptive headroom on two
datasets, close the dynamic-budget branch. If it passes, implement E27. No
E27/E28 results may be inferred from the current MMR 512 pilot.
