# E27-C0/C1 Residual-Cover Design

## Mục tiêu

Kiểm tra liệu **residual semantic structure** của context được chọn có cung
cấp signal mới để dự đoán downstream summarization risk hay không, vượt qua
các signal đã có trong E27-A và các signal gần với Budget-Aware Routing.

E27-C0 là offline screen trên chính 540 outcomes của E26-R; không chạy target
inference mới. E27-C1 chỉ được mở nếu C0 đạt novelty/signal gate.

## Hypothesis

Với document (D) và selected context (S_b), định nghĩa:

\[
R(D,S_b)=\frac1n\sum_i\left(1-\max_{j\in S_b}\operatorname{sim}(e_i,e_j)\right).
\]

Hypothesis chính:

> Residual-cover và marginal residual-cover signal dự đoán được quality
> violation ở action/budget tốt hơn rõ rệt so với length, lexical redundancy,
> front-loading và facility-location signal.

## Biến thực nghiệm

**Biến độc lập:** signal family dùng cho policy:

1. `cheap`: tám source-only features của E27-A.
2. `front_redundancy`: source length, front-loading và redundancy statistics.
3. `facility`: source length, front-loading, redundancy và facility-location
   coverage.
4. `residual`: residual cover, marginal residual gain và các feature cấu trúc
   cần thiết.
5. `residual_plus_cheap`: residual signal cộng cheap source features, chỉ là
   exploratory control.

**Biến phụ thuộc:** mean selector-inclusive `pipeline_e2e_ms`, empirical risk,
risk-contract pass fraction, cost gain so với best fixed và oracle capture.

**Biến kiểm soát:** cùng E26-R outcomes, cùng six actions, epsilon/alpha,
document IDs, target quality definition, cost metric, splits, seed và policy
selection rule.

## Cách dựng signal

Tái dựng MMR selection với:

- Qwen3-4B tokenizer local;
- source cap 4096 cho GovReport/Multi-News;
- MMR lambda 0.7;
- all-MiniLM-L6-v2 local embedding;
- cùng budget labels `full`, `ratio_0.25`, `ratio_0.4`, `ratio_0.55`,
  `ratio_0.7`, `ratio_0.85`.

Các embeddings chỉ dùng để tính feature offline; chúng không được coi là
downstream quality hoặc target feature.

`facility` dùng mean maximum cosine support từ mỗi source sentence tới selected
sentence set. `residual` dùng (1-support). `marginal_residual` là giảm
residual khi chuyển từ action trước đó sang action hiện tại. Với budget full,
residual bằng 0 vì toàn bộ source sentence được chọn.

## Policy và đánh giá

Mỗi signal family được dùng trong cùng policy learner như E27-A:

- LogisticRegression cho violation probability theo từng action;
- Ridge cho cost theo từng action;
- chọn action rẻ nhất có predicted risk không vượt alpha;
- fallback full nếu không có action eligible.

Đánh giá bằng 20 repeats của 3-fold document-level CV, stratified theo dataset.
Mọi train-derived model chỉ fit trên train folds. Best-fixed và adaptive oracle
là hindsight references, không phải deployment policy.

Primary contract:

\[
\epsilon=0.02,\qquad\alpha=0.05.
\]

Sensitivity giữ lại (epsilon\in\{0.01,0.02,0.05\}) và
(alpha\in\{0,0.05,0.10\}).

## Gate

So sánh residual với signal prior mạnh nhất trên từng dataset. C0 pass khi:

\[
Capture_{residual}-Capture_{prior}\ge 10\text{--}15\text{ percentage points}
\]

và:

\[
Capture_{residual}\ge30\text{--}40\%
\]

trên ít nhất hai dataset tại primary contract, đồng thời risk không vượt alpha.

Nếu không đạt, đóng SafeCover residual-cover branch và không chạy C1.

## E27-C1 dự kiến nếu C0 pass

E27-C1 sẽ tạo policy stopping dựa trên residual curve:

\[
S_1\subset S_2\subset\dots\subset S_t
\]

và chọn action nhỏ nhất thỏa residual threshold/learned risk rule trong
train-fold. C1 phải so với best fixed MMR, length/redundancy router, facility
router và residual-cover stopping dưới cùng outcome grid. Không được gọi đây là
conformal guarantee nếu chưa có held-out calibration.

## Validation scope

- Unit tests: cosine support, residual monotonicity trên nested selection,
  marginal gain, missing/duplicate outcome handling, no leakage và policy
  fallback.
- Integration smoke: một dataset nhỏ hoặc fixture synthetic, kiểm tra đủ
  signal families và finite metrics.
- Full C0: real 90-document E26-R outcome grid, offline CPU feature
  reconstruction, 20 repeated 3-fold CV.
- E27-C1 chỉ chạy sau khi C0 gate pass; nếu pass sẽ cần thêm held-out policy
  screen nhưng chưa phải conformal deployment study.

## Artifact contract

Output directory:

`outputs/safe_budget_sum/2026-09-11_e27c_residual_cover/`

Ghi `features.json`, `metrics.json`, `metrics.csv`, `report.md` và manifest
input/seed/protocol. Báo cáo phải tách CONFIRMED, EXPLORATORY,
FAILED/INCOMPLETE, EVIDENCE GAPS và RECOMMENDED NEXT.
