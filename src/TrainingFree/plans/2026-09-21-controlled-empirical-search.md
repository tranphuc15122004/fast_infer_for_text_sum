# Controlled Empirical Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:ml-subagent-dev to implement this plan task-by-task.

**Goal:** Kiểm tra có cấu trúc head/source nào tạo được headroom thực sự cho Training-Free trước khi đầu tư vào router, sketch hoặc physical KV executor.

**Experiment directory:** `src/TrainingFree/`

**Hypothesis:** Một subset attention head cần global source access, trong khi các head còn lại có thể route theo block với expansion thấp mà vẫn giữ missed mass và output error trong gate.

**Validation scope:** L0 unit/static; L1 Modal GPU trên dev-search hiện tại gồm 20 `gov_report` + 20 `multi_news`, Qwen3-0.6B, 32 output tokens. Chưa dùng final evaluation.

**Evaluation design:** E41 là bắt buộc trước. E42 chỉ chạy nếu E41 cho thấy heterogeneity; E43 chỉ chạy nếu E42 oracle hybrid có headroom. Mỗi run lưu config, per-head metrics, P95/P99 và trạng thái gate; không dùng partial/OOM mean như full result.

**Execution status (2026-09-21):** E41 đã PASS heterogeneity; E42 đã chạy đủ
40/40 mẫu nhưng không có candidate nào đạt đồng thời các gate trên cả hai
dataset. Theo protocol, dừng ở `STOP_BEFORE_E43`; chưa viết router/sketch hay
physical executor. Bảng và artifact được ghi tại
[`docs/experiments/2026-09-21_trainingfree_controlled_search.md`](../../../docs/experiments/2026-09-21_trainingfree_controlled_search.md).

**Architecture:** Collector chỉ bổ sung summary per-head từ attention weights, không thay đổi model/KV path. Analyzer độc lập đọc JSONL trace và sinh JSON/CSV/Markdown; core Training-Free không import analyzer/test code.

---

## Shared Scaffold

### Existing infra (don't touch, advise if problems found)

- V3 collector: `src/TrainingFree/hierarchy_collector.py`
- V3 trace schema: `src/TrainingFree/schema.py`
- Modal runner: `scripts/modal_trainingfree.py`
- V3 report/evaluator: `src/TrainingFree/hierarchy_evaluation.py`
- Existing dev-search data: `data/longbench_100_14k/{gov_report,multi_news}.jsonl`

### Needs setup

- Per-head concentration summary in hierarchy traces.
- Offline E41 analyzer: `scripts/analyze_trainingfree_e41.py`.
- Deterministic unit tests for K90/K95/K99 and aggregation.
- Dedicated Modal output root/run ID; raw traces remain on Modal Volume.

## Subtask 1: Per-head concentration metrics

**Role:** Bổ sung observability cần thiết cho E41 mà không thay đổi routing.

**Implementation:** Thêm helper tính source-normalized K90/K95/K99 và source
mass theo head; collector ghi `head_concentration` ở mỗi decode step, gồm
`layer`, `head`, `source_mass`, `k90/k95/k99`, và fraction tương ứng.

**Unit Tests:** `src/TrainingFree/tests/test_hierarchy_collector.py` và test
schema cho metric mới; kiểm tra mass concentrated, diffuse, source span và
finite values.

**Expected Conclusion:** Collector tạo được summary per-head hợp lệ; không
thay đổi `exact_expansion_fraction` V3.

## Subtask 2: E41 analyzer [INTEGRATION]

**Hypothesis:** Distribution `K95/source_tokens` khác biệt đáng kể giữa heads,
layer hoặc dataset; nếu mọi head diffuse thì dừng head-routing.

**Components consumed:** Subtask 1, V3 hierarchy JSONL, `data/longbench_100_14k`.

**Implementation:** Tạo `scripts/analyze_trainingfree_e41.py` để aggregate
mean/variance/P95/P99 theo layer-head và dataset, tính
`GlobalHeadFraction(tau)` cho `tau ∈ {0.1,0.25,0.5}`, sinh CSV/JSON/Markdown
và dữ liệu heatmap. Không sinh physical KV hay claim speedup.

**Integration Tests:** Trace fixture → analyzer → report; kiểm tra global-head
fraction và dataset grouping.

**Validation Pyramid:** L0 unit/static; L1 Modal `hierarchy` trên 40 doc,
`max_new_tokens=32`, layer cuối, lưu output dưới run ID riêng.

**Expected Conclusion:**

- `heterogeneous`: chạy Subtask 3 E42.
- `diffuse`: dừng head-routing và ghi negative result.
- `inconclusive`: tăng trace coverage, không tune threshold trên holdout.

## Subtask 3: E42 oracle hybrid-head sweep

**Role:** Đo upper bound của ý tưởng global-head + routed-head trước khi viết router.

**Gate:** Chỉ thực hiện nếu E41 cho thấy heterogeneity.

**Implementation:** Với attention oracle hiện có, sweep global-head fraction
`{0.1, 0.2, 0.3, 0.4}` và routed source fraction
`{0.05, 0.10, 0.20, 0.30}`; report missed mass, exact expansion và attention
output error. Đây là offline oracle, không đại diện physical latency.

**Expected Conclusion:** Nếu oracle không đạt gate, dừng router; nếu đạt,
chạy Subtask 4.

## Subtask 4: E43 exact block-score oracle

**Role:** Phân biệt lỗi granularity/head diffusion với lỗi score sketch.

**Gate:** Chỉ thực hiện nếu E42 pass.

**Implementation:** Sweep block size `{32, 64, 128}`, tính exact block score,
đo số block cần giữ để đạt quality/missed-mass gate. Không triển khai sketch.

**Expected Conclusion:** Nếu cần hơn 50% block, bỏ low-rank search; nếu có
headroom, mở E44/E45.

## Search gate và handoff

Candidate chỉ được promote khi trên dev-search đạt đồng thời:

- exact expansion ≤ 30%;
- mean missed mass ≤ 1%; P99 missed mass ≤ 5%;
- index overhead ≤ 10%;
- effective cost ratio < 0.5 nếu đã có routing proxy;
- top-1 agreement ≥ 99% nếu logit simulation khả dụng.

Khi một candidate pass, freeze config và chạy internal holdout 40 document
mới. Không tune trên holdout; sau holdout mới thử Qwen3-4B.
