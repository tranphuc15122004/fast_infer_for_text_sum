# Training-Free: Context-Adaptive DFlash

Ngày cập nhật: **2026-10-08**. Trạng thái: **executor đã có; rà soát CPU phát hiện lỗi chặn, chưa sẵn sàng chạy matrix lớn; GPU parity và benchmark vẫn pending**.

Mục tiêu là cùng thích nghi context budget của drafter và số speculative tokens trên DFlash đã train, cho long-context summarization. Target giữ full context và thực hiện exact verification. Trọng số target, DFlash, embedding và LM head không thay đổi.

## Tài liệu canonical và thứ tự đọc

| Tài liệu | Nội dung |
|---|---|
| [Paper story](docs/context_adaptive_dflash/paper_story.md) | Động cơ, prior work, giả thuyết và contribution dự kiến |
| [Đặc tả phương pháp](docs/context_adaptive_dflash/design.md) | A/B/C, budget, signals, controller và fallback |
| [Hợp đồng tích hợp](docs/context_adaptive_dflash/integration.md) | Giao diện cần xây, logical positions, KV bank và verifier |
| [Hợp đồng đo lường](docs/context_adaptive_dflash/measurement_contract.md) | Manifest, request/round schema, timing, correctness và aggregation |
| [Protocol thực nghiệm](docs/context_adaptive_dflash/experiment_protocol.md) | Split, calibration, ablation, fairness và các gate |
| [Runbook](docs/context_adaptive_dflash/runbook.md) | Runtime offline và các lệnh chạy prepare→preflight→calibrate→dev→lock→test→report |
| [Kế hoạch triển khai](plans/2026-10-08-context-adaptive-dflash-implementation.md) | Các task, file cần tạo và điều kiện hoàn tất |
| [Rà soát triển khai 2026-10-08](docs/context_adaptive_dflash/implementation_review_2026-10-08.md) | 10 vấn đề đã tái hiện, kết quả 45 ca CPU và thứ tự xử lý |

[RECENT_idea.md](plans/RECENT_idea.md) được giữ như ghi chép hình thành ý tưởng. Khi định nghĩa hoặc policy trong ghi chép khác bộ tài liệu này, dùng đặc tả canonical ở bảng trên.

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
4. Rà soát CPU ngày 2026-10-08: **31/45 ca đạt, 14 ca thất bại**, tương ứng 7 vấn đề P1 và 3 P2. Full/sparse adapter và greedy rollout khớp trên mô hình nhỏ; checkpoint validation, AR schema, sampling, launcher và resume còn lỗi. Xem [báo cáo](docs/context_adaptive_dflash/implementation_review_2026-10-08.md).
5. **Chưa chạy** parity trên checkpoint/GPU thật, CUDA memory/timing, smoke server hoặc G0–G6. Local host CPU-only; không có số liệu hay claim speedup được tạo.

Cần sửa các vấn đề trong báo cáo và chạy lại kiểm tra CPU trước server preflight/smoke. Sau đó thực hiện G0/G1/G2 trên checkpoint thật trước khi mở matrix lớn. Trạng thái triển khai chưa được coi là baseline đã validated hoặc phương pháp đã pass correctness.

Runtime server là Python 3.12 hệ thống; máy local chỉ dev/debug CPU. Xem [hồ sơ server](../../docs/server_environment.md) và [workflow CPU](../../docs/cpu_dev_workflow.md). Artifact đặt dưới `outputs/context_adaptive_dflash/`, được gitignore; chưa tạo artifact giả hoặc công bố speedup từ đặc tả.
