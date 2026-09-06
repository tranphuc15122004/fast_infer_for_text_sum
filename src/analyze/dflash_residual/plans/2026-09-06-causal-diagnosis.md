# Causal Diagnosis E16–E18 Implementation Plan

**Goal:** Xác định bằng intervention bounded liệu prefix-critical candidate valley của DFlash trên summarization đến từ state-distribution mismatch hay training-utility mismatch.

**Experiment directory:** `outputs/dflash_residual/2026-09-06_causal_diagnosis/`

**Hypothesis:** Nếu valley vị trí 3–8 tái lập, một trong hai intervention tối thiểu — on-policy state alignment hoặc utility-aligned training — sẽ tăng `MAT_O16` trên held-out summarization.

**Validation scope:** E16/E17 offline/bounded diagnostics; E18 T4 short pilot nếu gate trước đó pass. Không dùng B200. Không mở architecture/source/selector branch.

**Evaluation design:** document-level aggregation; primary `MAT_O16`; secondary `MAT_D`, `R16(3:8)`, `J16(3:8)` và runtime. E18 phải đánh giá held-out Multi-News, checkpoint và in-memory output nếu có; collector có phase/progress/error summary.

## Shared scaffold

- Existing traces: `outputs/dflash_residual/2026-09-05_prefix_utility/e11_*.jsonl`.
- Existing collector: `src/analyze/dflash_residual/trace_dflash.py`.
- Existing lattice metrics: `src/analyze/dflash_residual/joint_lattice.py`.
- Existing DFlash training: `src/MR_DFlash/training.py`, `src/MR_DFlash/run_train.py`.
- Existing checkpoint/data preparation: `prepare_e15.py`, `export_e15_checkpoint.py`.

## Subtask 1: E16 canonical expansion and valley gate

**Role:** Kiểm tra nhanh phenomenon trước khi chạy causal intervention.

**Implementation:** Reuse collector; prepare 50–100 canonical documents if local canonical source is available; otherwise record the exact sample limitation and use the largest reproducible canonical subset. Run matched 1K/Top-16/block16 traces and offline `R16`, `J16`, `MAT_O16`, document bootstrap.

**Tests:** schema validation, finite metrics, deterministic bootstrap fixture.

**Expected conclusion:** PASS only if the position 3–8 valley is stable against expanded canonical and appears on at least two summarization datasets.

## Subtask 2: E17-A reference-vs-on-policy state audit

**Role:** Tách deployment-state mismatch khỏi candidate-generation quality.

**Implementation:** Add an analysis/collector mode that uses the same source/document and target/draft checkpoint but constructs reference-prefix and target-on-policy prefix states. Record `state_mode` in trace; compare per-position `R16`, `J16`, `MAT_O16`, first rejection and bootstrap.

**Tests:** synthetic state-mode trace contract; assert all non-state protocol fields match.

**Expected conclusion:** State mismatch is supported only if the on-policy/reference difference is reproducible and material at positions 3–8.

## Subtask 3: E17-B effective utility audit

**Role:** Đo chính xác training emphasis thay vì suy luận từ `gamma`.

**Implementation:** Add offline utility audit that computes theoretical decay weights, observed supervised counts, effective weighted mass, per-position loss/accuracy from a deterministic batch, and verifier utility `U_j` from E16 traces. Keep training objective unchanged.

**Tests:** gamma-weight formula, mask/exposure accounting, conservation of weighted mass.

**Expected conclusion:** Utility mismatch is supported only if observed emphasis differs materially from measured `U_j` in the prefix-critical region.

## Subtask 4: E18 selected minimal intervention [INTEGRATION]

**Hypothesis:** Correcting the winning mechanism increases held-out `MAT_O16` by at least 10% without changing inference architecture.

**Implementation:** Run only the intervention(s) selected by E17 gates. Use original checkpoint/target, fixed block16/Top16, task-disjoint train/eval, one seed, explicit checkpoint/export parity, progress logging and finite checks. For both mechanisms, preserve a no-change baseline from the same held-out documents.

**Integration tests:** one-step train, checkpoint load, trace collection, metric/report generation and failure-on-nonfinite.

**Validation Pyramid:** R0–R2 before GPU; R5 short pilot on T4. Success requires `MAT_O16` delta >10% on held-out Multi-News; >20% triggers proposal review.

**Expected conclusion:** Either one mechanism passes and becomes the only proposal candidate, or the DFlash branch is closed.
