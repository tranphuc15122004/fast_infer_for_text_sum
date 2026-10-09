# Training-Free: Context-Adaptive DFlash

Ngày cập nhật: **2026-10-09**. Trạng thái: **đã sửa 10 lỗi rà soát; 45/45 ca hồi quy CPU đạt; đã bổ sung tài liệu thực nghiệm, parity GPU và benchmark vẫn pending**.

Mục tiêu là cùng thích nghi context budget của drafter và số speculative tokens trên DFlash đã train, cho long-context summarization. Target giữ full context và thực hiện exact verification. Trọng số target, DFlash, embedding và LM head không thay đổi.

## Tài liệu canonical và thứ tự đọc

| Tài liệu | Nội dung |
|---|---|
| [Paper story](docs/context_adaptive_dflash/paper_story.md) | Động cơ, prior work, giả thuyết và contribution dự kiến |
| [Đặc tả phương pháp](docs/context_adaptive_dflash/design.md) | A/B/C, budget, signals, controller và fallback |
| [Hợp đồng tích hợp](docs/context_adaptive_dflash/integration.md) | Giao diện cần xây, logical positions, KV bank và verifier |
| [Hợp đồng đo lường](docs/context_adaptive_dflash/measurement_contract.md) | Manifest, request/round schema, timing, correctness và aggregation |
| [Protocol thực nghiệm](docs/context_adaptive_dflash/experiment_protocol.md) | Split, calibration, ablation, fairness và các gate |
| [Ma trận và kế hoạch thực nghiệm](docs/context_adaptive_dflash/experiment_matrix.md) | Preregistration, exposure audit, tuning/controls, lịch chạy, gate và artifact paper |
| [Kiểm chứng GPU](docs/context_adaptive_dflash/gpu_validation.md) | G0/G1/G2 trên checkpoint thật, cases parity/cache/positions/EOS và harness còn cần |
| [Runbook](docs/context_adaptive_dflash/runbook.md) | Runtime offline và các lệnh chạy prepare→preflight→calibrate→dev→lock→test→report |
| [Mẫu báo cáo và bàn giao](docs/context_adaptive_dflash/results_template.md) | Coverage/exactness, calibration support, paired CI, latency/quality/memory và verdict G0–G6 |
| [Kế hoạch triển khai](plans/2026-10-08-context-adaptive-dflash-implementation.md) | Các task, file cần tạo và điều kiện hoàn tất |
| [Rà soát và khắc phục 2026-10-08](docs/context_adaptive_dflash/implementation_review_2026-10-08.md) | 10 lỗi đã sửa, kết quả CPU và bước kiểm chứng GPU còn lại |

Ghi chép ban đầu `plans/RECENT_idea.md` hiện không có trong checkout này. Định nghĩa phương pháp và protocol thực nghiệm canonical nằm trong bộ tài liệu ở bảng trên.

## Ba thành phần

- **A:** chọn context có ích cho drafter; giữ instructions, source anchors, recent output và các source chunks được xếp hạng.
- **B:** chọn số speculative tokens bằng uncertainty, acceptance history và chi phí.
- **C:** chọn cặp `(draft_context_budget, draft_tokens)` theo estimated milliseconds trên mỗi token mới được commit.

Selective context có cơ sở thực nghiệm từ SpecExtend. Chuyển cơ chế sang DFlash, giá trị của entropy có sẵn trước round và lợi ích của joint adaptation cần được đo theo protocol.

## Phạm vi code

Các module hiện tại như `policy.py`, `collector.py`, `run.py`, `lease*.py`, `hierarchy*.py`, `temporal*.py` và `fidelity*.py` thuộc **RECAP-KV và các nghiên cứu chẩn đoán trước đó**. `python -m TrainingFree.run` chạy pipeline RECAP-KV, không chạy phương pháp mới.

Executor mới nằm trong [`src/TrainingFree/context_adaptive/`](context_adaptive/), CLI tại [`scripts/infer_context_adaptive_dflash.py`](../../scripts/infer_context_adaptive_dflash.py), runner tại [`scripts/runners/run_context_adaptive_dflash.sh`](../../scripts/runners/run_context_adaptive_dflash.sh) và đã nối dispatcher `bash scripts/run.sh context_adaptive_dflash`. Nó có action replay calibration, adaptive generation, target-only AR, output manifests/JSONL và paired report. Không đổi tên schema RECAP-KV hoặc nhập core inference từ test/probe.

Các nguồn có thể tham khảo:

- [DFlash baseline](../../scripts/infer_dflash.py) và [drafter vendored](../../externals/dflash/dflash/model.py).
- [Parent target attention → next draft](../../docs/experiments/2026-10-06_dflash_parent_attention_next_draft.md).
- [Cơ chế dùng attention DFlash và dense refresh](../../docs/experiments/2026-10-06_dflash_context_execution_mechanism.md).
- [Phân tích context/memory đã có](../../docs/experiments/2026-10-06_dflash_context_memory_paper_analysis.md).

Các báo cáo attention trên cohort đã dùng để phát triển ý tưởng là bằng chứng chẩn đoán, không phải holdout hoặc kết quả can thiệp của phương pháp mới.

## Trạng thái kiểm chứng

1. Source code có full/sparse DFlash bank, exact target verifier, logical-position masks, target-parent/draft-refresh signals, controller variants và action replay calibration.
2. CLI có source-group split, model/data signatures, request-local calibration validation, resumable JSONL, output-token audit, ROUGE và paired source-cluster report.
3. `verify_prefix` và `async_event` bị từ chối rõ ràng; V1 chỉ hỗ trợ `draft_shape`, `frozen_cost`, batch 1. Calibration priors hiện yêu cầu greedy.
4. Đợt khắc phục sau rà soát: **45/45 ca hồi quy CPU đạt**, contract master config **10/10 đạt**, prepare corpus thành công. Xem [báo cáo](docs/context_adaptive_dflash/implementation_review_2026-10-08.md).
5. **Chưa chạy** parity trên checkpoint/GPU thật, CUDA memory/timing, smoke server hoặc G0–G6. Local host CPU-only; chưa có số liệu hay claim speedup.

Các lỗi trong báo cáo đã sửa và kiểm tra CPU đạt. Tiếp theo chạy preflight với checkpoint thật và G0/G1/G2 trên GPU server trước khi mở matrix lớn. Chưa coi phương pháp là baseline đã validated cho đến khi có parity/backend và kết quả GPU tương ứng.

Runtime server là Python 3.12 hệ thống; máy local chỉ dev/debug CPU. Xem [hồ sơ server](../../docs/server_environment.md) và [workflow CPU](../../docs/cpu_dev_workflow.md). Artifact đặt dưới `outputs/context_adaptive_dflash/`, được gitignore; chưa tạo artifact giả hoặc công bố speedup từ đặc tả.
