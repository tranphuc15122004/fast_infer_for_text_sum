# GroundSync Hypothesis Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:ml-subagent-dev to implement this plan task-by-task.

**Goal:** Xây dựng và chạy pipeline tái lập để đo H1–H5 của GroundSync, lưu raw traces/metrics/plots và sinh báo cáo kết luận có thể audit.

**Experiment directory:** `src/analyze/groundsync/`

**Hypothesis:** Source-utilization state của target có persistence; transition của nó dự báo speculative rejection, tạo oracle utility và có thể dự đoán online.

**Validation scope:** Không có training pipeline lớn nên không chạy training VP; dùng static checks, unit tests, synthetic CPU smoke và model-backed local-files-only smoke. CUDA E2E chỉ báo cáo khi server có target/draft cache.

**Evaluation design:** Evaluation là batch analysis, không có epoch/cadence. Mỗi phase có log bắt đầu/kết thúc, progress theo document, counts, error status và summary. Một evaluator core dùng chung cho synthetic/model traces; hỗ trợ đọc trace file và in-memory rows. Missing model, OOM, empty trace, non-finite metric và timing unavailable phải thành trạng thái rõ ràng.

**Architecture:** `core.py` thuần dữ liệu chứa metric/statistics/policy logic; `trace_target.py` và `trace_speculative.py` là adapters model độc lập; `run_experiment.py` điều phối và ghi manifest/raw artifacts; `report.py` aggregate và sinh Markdown/CSV/PNG. Core không import test/validation code và không ghi vào artifact cũ.

---

## Shared Scaffold

### Existing infra (không sửa)

- Data records: `data/representative_100/*.jsonl` và loader schema trong `scripts/common/data_loader.py`.
- Qwen3 runtime precedent: `src/analyze/full_infer/profile_qwen3_long_summary.py`.
- Runtime/server convention: `docs/server_environment.md`, `scripts/common/runtime.sh`.
- Existing EAGLE benchmark: `scripts/eagle3_infer_qwen3.py` — chỉ tham khảo/benchmark E2E, không phụ thuộc cho controlled trace.

### Needs setup

- `src/analyze/groundsync/__init__.py`
- `src/analyze/groundsync/core.py`
- `src/analyze/groundsync/trace_target.py`
- `src/analyze/groundsync/trace_speculative.py`
- `src/analyze/groundsync/run_experiment.py`
- `src/analyze/groundsync/report.py`
- `src/analyze/groundsync/tests/test_core.py`
- `src/analyze/groundsync/tests/test_pipeline.py`
- `src/analyze/groundsync/README.md`
- `src/analyze/groundsync/results/.gitkeep` nếu cần giữ thư mục; run thực tế ghi vào `results/<run_id>/`.

## Subtask 1: Core trace schema và hypothesis metrics

**Role:** Cung cấp các hàm deterministic để mọi runner và report dùng chung.

**Implementation:** Viết schema validation cho target/spec rows; chunk aggregation; normalize distribution; Jensen–Shannon divergence; lag similarity; segment length/horizon; acceptance/drift join; document-level grouped summaries; stable bootstrap; feature matrix cho H3 và oracle/policy replay cho H4.

**Unit Tests:** Zero/negative distribution, JS symmetry/bounds, normalized chunk mass, lag null/shuffle, horizon threshold, first-rejection alignment, document-level aggregation không coi token là independent, non-finite rejection.

**Expected Conclusion:** Core chạy được bằng Python 3.12, không cần torch/Transformers/model.

### Steps

1. Viết `test_core.py` với synthetic traces và chạy `pytest src/analyze/groundsync/tests/test_core.py -q`; kỳ vọng fail vì module chưa tồn tại.
2. Implement `core.py` với type hints, JSON-safe outputs và explicit `ValueError` cho input malformed.
3. Chạy lại test; kỳ vọng toàn bộ core tests pass.
4. Chạy `python3 -m py_compile src/analyze/groundsync/core.py`.

### Subtask 1 Conclusion (code)

**Role:** Core metric độc lập model cho H1/H2/H4.
**Result:** implemented.
**Evidence:** 12 unit tests passing; source `core.py` và `tests/test_core.py`.

## Subtask 2: Target canonical trace adapter

**Role:** Sinh canonical greedy target trace cho H1/E0 và các feature target dùng H3.

**Implementation:** `trace_target.py` nhận model/tokenizer path, input JSONL, sample/context/output limits, device/dtype/attention backend và output dir. Render prompt nhất quán; incremental cached forward; lấy attention query hiện tại, chỉ aggregate source chunks; lưu raw/no-sink/calibrated variants, entropy, canonical output và metadata. Có model loader injectable để test fake model/tokenizer; dùng `local_files_only`; không lưu full `L x L` attention.

**Unit Tests:** Prompt/source offset, chunk boundaries, no-sink masks, schema serialization, fake model one-step generation và failure status. Không gọi network.

**Expected Conclusion:** Target trace có thể chạy trên CPU/GPU khi snapshot tồn tại; thiếu snapshot trở thành `unavailable` có lý do.

### Steps

1. Thêm fake backend tests và chạy chúng trước implementation.
2. Implement adapter, CLI và JSONL writer; giữ Qwen3 thinking disabled và Transformers 5.x `return_dict=False`/mapping compatibility.
3. Chạy tests + `py_compile`.
4. Chạy synthetic/no-model CLI smoke để xác minh manifest/error contract.

### Subtask 2 Conclusion (code)

**Role:** Canonical target trace và E0/H1 signal.
**Result:** implemented.
**Evidence:** target/core tests trong tổng `33 passed`; `trace_target.py` dùng
incremental cached forward, eager attention vector và local-files-only loader.

## Subtask 3: Controlled speculative trace và policy replay

**Role:** Đo H2/H4 bằng target canonical continuation và draft Qwen3 nhỏ; không phụ thuộc EAGLE CUDA.

**Implementation:** `trace_speculative.py` load target/draft local snapshots; lấy các prefix đã chọn trong target trace; draft greedy sinh `Kmax`; so continuation với canonical target để tính accepted prefix/first rejection; ghi draft confidence, drift-at-relative-position và optional draft/verification timing. Implement fixed-k, entropy/history adaptive, oracle horizon và true-cost policy replay trong core/report. Tách rõ exact controlled acceptance khỏi E2E EAGLE benchmark.

**Unit Tests:** Accepted-prefix edge cases (full accept, first-token reject, EOS), prefix bounds, deterministic draft, policy clipping, unavailable draft/GPU path.

**Expected Conclusion:** Có thể đánh giá H2/H4 từ JSONL ngay cả khi EAGLE không chạy; timing proxy không bị gắn nhãn E2E.

### Steps

1. Viết unit tests acceptance/policy và chạy test fail-first.
2. Implement controlled runner và cost fields với CUDA synchronization khi có CUDA.
3. Chạy tests + `py_compile`.
4. Chạy fake backend smoke và kiểm tra không ghi artifact ngoài experiment dir.

### Subtask 3 Conclusion (code)

**Role:** Controlled draft–target acceptance, drift và timing theo block length.
**Result:** implemented.
**Evidence:** acceptance/policy tests trong tổng `38 passed`; runner lưu
`draft_time_by_k_ms`, optional `verification_time_by_k_ms` và tách timing khỏi
acceptance-only. Target, draft và verifier có chunked causal prefill cho T4
long-context.

## Subtask 4: H3 predictor và report aggregation

**Role:** Kiểm tra incremental signal và biến raw metrics thành kết luận từng hypothesis.

**Implementation:** Trong `core.py` hoặc module report-only, fit regularized logistic/survival baseline bằng dependency sẵn có hoặc fallback thuần torch/numpy; split theo documents; tính AUROC/AUPRC/log-loss/Brier/calibration và drift coefficient. `report.py` sinh `metrics.json`, `metrics.csv`, plots tối thiểu persistence/drift/rejection/horizon, và Markdown report gồm coverage, controls, confidence/cluster unit, PASS/FAIL/INCONCLUSIVE/UNAVAILABLE. Không gọi hypothesis pass khi thiếu data.

**Unit Tests:** Logistic trên fixture separable/non-separable, document split disjoint, metric finite, empty/unavailable report, status decision thresholds.

**Expected Conclusion:** Report có thể đọc cả trace thật lẫn fixture và không overclaim.

### Steps

1. Viết tests report/predictor và chạy fail-first.
2. Implement predictor, aggregation và Markdown/CSV/PNG outputs.
3. Chạy unit tests, kiểm tra artifact schema và `py_compile`.

### Subtask 4 Conclusion (code)

**Role:** H1–H5 aggregation, document-split predictor và report artifacts.
**Result:** implemented.
**Evidence:** report/pipeline tests nằm trong tổng `34 passed`; evaluator sinh
`metrics.json`, `metrics.csv`, `hypothesis_report.md` và PNG.

## Subtask 5: Experiment orchestrator and reproducible docs [INTEGRATION]

**Hypothesis:** Pipeline assembled từ target trace + controlled draft trace + evaluator sẽ tạo bằng chứng đủ để quyết định H1–H5, hoặc ghi rõ hypothesis nào chưa thể kết luận.

**Components consumed:** `core.py`, `trace_target.py`, `trace_speculative.py`, `report.py`.

**Implementation:** `run_experiment.py` có CLI phase `target`, `speculative`, `analyze`, `all`; model/draft/data/output args; seed; sample limits; run manifest; phase logs/progress; resume/skip nếu raw trace đã tồn tại; error isolation theo document; không ghi đè kết quả cũ. `README.md` mô tả lệnh CPU smoke, server CUDA discovery và schema.

**Integration Tests:** Synthetic JSONL → core analysis → `hypothesis_report.md`, kiểm tra target-only mode không cần draft, unavailable mode không crash, repeated run giữ manifest deterministic.

**Validation:**

- Static: `python3 -m compileall -q src/analyze/groundsync` và import check.
- Runtime smoke: `pytest src/analyze/groundsync/tests -q` với fixture, không tải model.
- Model smoke: chỉ khi Qwen snapshot local tồn tại; ghi model revision/path basename và device.
- E2E timing: chỉ trên CUDA server; phân biệt measured từ analytical proxy.

**Expected Conclusion:** Có raw trace/metrics/report reproducible; mỗi H1–H5 có coverage, result, decision và limitation.

### Steps

1. Viết integration tests và chạy fail-first.
2. Assemble orchestrator, report wiring, README và output manifest.
3. Chạy integration tests và full compile/tests.
4. Chạy CPU synthetic end-to-end.
5. Chạy model-backed smoke nếu preflight cho phép; nếu không, ghi `UNAVAILABLE` trong artifact.
6. Chạy static/runtime validation; kiểm tra toàn bộ artifact bằng fresh commands.

### Subtask 5 Conclusion (integration)

**Role:** Orchestrator và documentation.
**Result:** implemented and audited.
**Evidence:** synthetic fixture `synthetic-20260829-v2` chạy end-to-end;
GPU smoke và discovery `qwen3-4b-gov25-gpu-all-20260829` chạy bằng Python
miniconda ngoài venv trên T4, lần lượt có 1 và 25 target documents; discovery
ghi 50 proposals có measured verifier timing.

## Execution order và commit checkpoints

1. Subtask 1 → commit core metrics.
2. Subtask 2 → commit target trace.
3. Subtask 3 → commit controlled speculative trace.
4. Subtask 4 → commit predictor/report.
5. Subtask 5 → commit integration pipeline and docs.

Mỗi checkpoint phải có test output fresh; không dùng kết quả cũ để khẳng định
phase pass.

## Post-plan execution audit 2026-08-30

Các mở rộng sau khi chạy plan ban đầu đã được thực hiện và ghi trong
`verification_report_2026-08-29.md`: positional calibration/sensitivity,
position-relocation E0 fixture, position-adjusted H2 hazard với document
bootstrap, H3 negative controls, adaptive/true-cost policy và H5 threshold
selection trên train/dev. Model-backed evidence dùng Qwen3-4B/Qwen3-0.6B trên
T4 ngoài venv. CNN/DailyMail đã được chạy controlled cross-regime, không chỉ
target-side.

Các giới hạn còn lại là có chủ ý: main timing vẫn một start/document và
multi-start bổ sung chưa có verifier timing; timing chưa phải EAGLE/vLLM
production serving; E0 relocation mới là fixture 3 case. Những giới hạn này
được báo cáo thành evidence scope, không chuyển thành `PASS` giả.

## P0 decision extension — 2026-09-02

### Subtask 6: Corrected transition hazard, oracle ladder và burstiness

**Role:** Tách ba câu hỏi quyết định khỏi H2/H4 legacy: transition tại từng
relative position có liên hệ với rejection không; oracle opportunity có nằm ở
admission/`k=0`/`k=16` không; acceptance có burst/persistence không.

**Implementation:** Thêm `p0_decision.py`. Xây risk-set theo từng proposal và
`j`, tính `d_transition=JS(g[t+j-1],g[t+j])`, fit regularized logistic hazard
với controls entropy, draft confidence, relative position và absolute output
position, bootstrap theo document. Sửa semantic horizon: không thấy transition
trong `Kmax` trả về `Kmax`. Tune threshold trên train/dev bằng utility rồi
freeze. Replay ladder `0,2,4,8,16`, entry oracle chỉ dùng
`accepted_len > 0`, và burstiness within/across round.

**Tests:** Transition index/risk-set, `NULL -> Kmax`, k=0 cost, ladder
coverage, admission oracle chỉ đọc first-token bit, hazard `h_j`, persistence
delta và document split không leak.

### Subtask 7: Đo chi phí AR và block `Kmax=16`

**Role:** Cung cấp timing hợp lệ cho `k=0` và `k=16` trên target/draft thật.

**Implementation:** Mở rộng `trace_speculative.py` ghi
`autoregressive_time_ms` cho target cached one-token check và cho phép timing
arrays tới `max_k=16`; giữ chunked prefill, CUDA synchronize và local-only
loading. Không chạy venv trên T4; nếu PyTorch không expose CUDA thì ghi rõ
UNAVAILABLE thay vì chạy CPU và gắn nhãn GPU.

### Subtask 8: P0 CLI/report artifacts

**Role:** Chạy tất cả P0 trên raw traces hiện tại và traces mới, lưu cùng thư
mục experiment.

**Implementation:** CLI nhận target/spec/timing paths theo dataset, chọn
train/dev/test theo document, ghi `p0_metrics.json`, `p0_metrics.csv`,
`p0_decision_report.md`, `p0_manifest.json`, và các plot hazard,
ladder/admission, burstiness. Mỗi kết luận giữ
`PASS/FAIL/INCONCLUSIVE/UNAVAILABLE`, coverage và timing basis.

### Subtask 9: Execution and audit

**Execution:** Chạy test CPU trước; sau đó chạy acceptance `max_k=16` tối đa
100 documents/dataset và timing tối đa 50 documents/dataset trên T4 ngoài
venv. Bổ sung multi-start acceptance-only với 9 starts
`1,4,7,10,13,16,19,22,25` (`max_k=4`) để đo đủ round offset `1,2,4,8`;
persistence có document-bootstrap CI.

**Acceptance:** Corrected H2 dùng transition hazard, corrected H4 tune
train/dev, ladder có `k=0/16`, entry oracle có recovery, burstiness có cả hai
loại xác suất; mọi raw output và report nằm dưới
`src/analyze/groundsync/results/<run_id>/`.

## Subtask 10: Cheap online admission predictor

**Role:** Kiểm tra P1 bằng một policy nhỏ có thể dùng online, không dùng
grounding/future attention/accepted length tại thời điểm ra quyết định.

**Implementation:** Thêm `p1_predictor.py`. Dùng standardized ridge logistic
regression trên bốn feature có thật trong schema; split document 60/20/20; fit
trên train, chọn `k`/threshold trên calibration train+dev, đánh giá test. Tính
AUROC/AUPRC/log-loss/Brier/ECE và replay policy với AR `k=0`, fixed selected
`k`, first-token admission oracle, cùng descriptive true-cost oracle. Báo cáo
recovery của entry-oracle gap và không overclaim khi test one-class.

**Tests/validation:** CPU fixture có hai class, split không giao nhau, missing
feature, one-class inconclusive; chạy trên combined 56 GovReport và 50 CNN rows.

## Subtask 11: Strong-drafter replication

**Role:** Loại weak-drafter artifact bằng EAGLE-3 Qwen3-4B local head.

**Implementation:** Thêm `p1_strong_drafter.py`. Load EAGLE qua vendored loader
ngoài venv trên CUDA; ghi per-document acceptance lengths, chuẩn hóa accepted
draft tokens, exact schema và within/across-round burstiness. Dùng cùng 50
documents/dataset, `max_new_tokens=32`, input cap 1024 và document bootstrap.

**Tests/validation:** acceptance fallback conversion, persistence/burst summary,
CPU E2E summary test; GPU smoke trước full 50-document runs. Nếu cache hoặc CUDA
không tồn tại, artifact phải có trạng thái unavailable và nguyên nhân.

## Subtask 12: Paired direct E2E benchmark [INTEGRATION]

**Role:** Đo speedup thực tế ở model-level sau khi P0/P1 đã có kết quả; không
nhầm direct EAGLE với serving API.

**Implementation:** Runner Subtask 11 cùng lúc chạy EAGLE và greedy AR trên
cùng prompt, warmup trước đo, prefill loại khỏi decode timing, output exact-match
và aggregate tokens/s/speedup. Sinh `strong_drafter_metrics.json`, raw JSONL và
Markdown report trong run directory; nếu cần vLLM/API riêng mà không có package,
ghi `UNAVAILABLE` thay vì suy diễn từ direct run.

**Validation:** EAGLE smoke, paired exact-match, VRAM/process audit, `compileall`,
full pytest và kiểm tra artifact paths. Chỉ gọi P2 direct E2E `PASS` khi có đủ
paired timing và exact-match guardrail; serving production vẫn là một status
riêng.

## Target-KV implementation revision — 2026-09-03

### Subtask TK-1: E0 data manifest và context bucketing

Viết loader chỉ đọc local JSONL, chuẩn hóa `id`, dataset, context và token
length, lọc rõ mẫu vượt giới hạn model. Không padding main result. Sinh manifest
bao gồm số mẫu theo bucket, loại trừ, lý do loại trừ và hash input.

### Subtask TK-2: E0 DFlash trace runner

Tái sử dụng loader DFlash chính thức nhưng bổ sung raw round trace cho
K=`{4,8,16}`. Ghi acceptance length, first rejection, survival rows, target/draft
timing đã synchronize CUDA, peak memory và exact output guardrail. Chạy
`--smoke` trước; mọi lỗi OOM/compatibility là status có nguyên nhân.

### Subtask TK-3: E0 analyzer và decision report

Thêm metrics aggregation theo dataset/context/K, document bootstrap CI,
interaction effect và heatmap. Kill gate được tính máy móc từ manifest/metrics,
không nhập tay. Sinh JSON/CSV/PNG/Markdown dưới một run directory trong
`src/analyze/groundsync/results/`.

### Subtask TK-4: E1 feature extraction/cache

Target frozen, chạy từng anchor batch 1. Trích hidden/KV không có future-token
leak, lưu shard CPU/đĩa theo representation và split. Kiểm tra shape, dtype,
finite values, token alignment, document split và không giữ model graph.

### Subtask TK-5: E1 matched probe

Viết probe nhỏ dùng chung decoder/interface, train trên cached features với
document-disjoint split. Ghi parameter budget, seed, loss curves, CE/Acc@1/5,
survival-by-position và negative controls. Dừng/ghi `INCONCLUSIVE` khi test
không có đủ class hoặc bucket.

### Subtask TK-6: T4 smoke, pilot và gate

Dùng Python có CUDA ngoài `.venv`, FP16, batch 1. Smoke gồm short + long
representative; pilot E0 tối đa 200/200/100 records và E1 bắt đầu 5.000
anchors/dataset. Chỉ khi E0/E1 đạt gates mới thêm E2/E3. Không gọi kết quả
inference-only là serving benchmark.

### Impact on existing plan

Subtask 1–12 và các artifact GroundSync/P0/P1/P2 giữ nguyên. TK-1–TK-6 là
nhánh Target-KV độc lập, không sửa raw results cũ. Validation scope mở rộng từ
evaluation-only sang feature/probe training nên bắt buộc TDD cho parser/metric,
L0 static checks và L1 GPU smoke cho probe; full training không được coi là đạt
nếu chưa có document-disjoint audit.
