# Training-Free: Context-Adaptive DFlash

Ngày cập nhật: **2026-10-08**. Trạng thái: **đã đặc tả phương pháp và protocol để triển khai; chưa có executor Context-Adaptive DFlash hoặc kết quả benchmark của phương pháp mới**.

Mục tiêu là cùng thích nghi context budget của drafter và số speculative tokens trên DFlash đã train, cho long-context summarization. Target giữ full context và thực hiện exact verification. Trọng số target, DFlash, embedding và LM head không thay đổi.

## Tài liệu canonical và thứ tự đọc

| Tài liệu | Nội dung |
|---|---|
| [Paper story](docs/context_adaptive_dflash/paper_story.md) | Động cơ, prior work, giả thuyết và contribution dự kiến |
| [Đặc tả phương pháp](docs/context_adaptive_dflash/design.md) | A/B/C, budget, signals, controller và fallback |
| [Hợp đồng tích hợp](docs/context_adaptive_dflash/integration.md) | Giao diện cần xây, logical positions, KV bank và verifier |
| [Hợp đồng đo lường](docs/context_adaptive_dflash/measurement_contract.md) | Manifest, request/round schema, timing, correctness và aggregation |
| [Protocol thực nghiệm](docs/context_adaptive_dflash/experiment_protocol.md) | Split, calibration, ablation, fairness và các gate |
| [Runbook](docs/context_adaptive_dflash/runbook.md) | Runtime offline, cấu hình master và CLI dự kiến |
| [Kế hoạch triển khai](plans/2026-10-08-context-adaptive-dflash-implementation.md) | Các task, file cần tạo và điều kiện hoàn tất |

[RECENT_idea.md](plans/RECENT_idea.md) được giữ như ghi chép hình thành ý tưởng. Khi định nghĩa hoặc policy trong ghi chép khác bộ tài liệu này, dùng đặc tả canonical ở bảng trên.

## Ba thành phần

- **A:** chọn context có ích cho drafter; giữ instructions, source anchors, recent output và các source chunks được xếp hạng.
- **B:** chọn số speculative tokens bằng uncertainty, acceptance history và chi phí.
- **C:** chọn cặp `(draft_context_budget, draft_tokens)` theo estimated milliseconds trên mỗi token mới được commit.

Selective context có cơ sở thực nghiệm từ SpecExtend. Chuyển cơ chế sang DFlash, giá trị của entropy có sẵn trước round và lợi ích của joint adaptation cần được đo theo protocol.

## Phạm vi code hiện có

Các module hiện tại như `policy.py`, `collector.py`, `run.py`, `lease*.py`, `hierarchy*.py`, `temporal*.py` và `fidelity*.py` thuộc **RECAP-KV và các nghiên cứu chẩn đoán trước đó**. `python -m TrainingFree.run` chạy pipeline RECAP-KV, không chạy phương pháp mới.

Implementation mới được đặt trong `src/TrainingFree/context_adaptive/` theo kế hoạch; đường dẫn này và launcher mới là **deliverable cần xây**, chưa phải API đang có. Không đổi tên schema RECAP-KV hoặc nhập core inference từ test/probe.

Các nguồn có thể tham khảo:

- [DFlash baseline](../../scripts/infer_dflash.py) và [drafter vendored](../../externals/dflash/dflash/model.py).
- [Parent target attention → next draft](../../docs/experiments/2026-10-06_dflash_parent_attention_next_draft.md).
- [Cơ chế dùng attention DFlash và dense refresh](../../docs/experiments/2026-10-06_dflash_context_execution_mechanism.md).
- [Phân tích context/memory đã có](../../docs/experiments/2026-10-06_dflash_context_memory_paper_analysis.md).

Các báo cáo attention trên cohort đã dùng để phát triển ý tưởng là bằng chứng chẩn đoán, không phải holdout hoặc kết quả can thiệp của phương pháp mới.

## Điều kiện sẵn sàng chạy thực nghiệm

1. Adapter full-context tương đương DFlash vendored và greedy target-only.
2. Sparse draft giữ original positions, target cache đầy đủ và rollback đúng.
3. Signals chỉ dùng dữ liệu đã có trước action; sampler giữ target distribution.
4. Schema/timing bao gồm bootstrap, retrieval, gather, refresh và controller.
5. Calibration/dev/test được khóa bằng manifest, không dùng reference để điều khiển inference.
6. Các gate G0–G6 trong protocol có artifact tương ứng; smoke không thay thế kết quả heldout.

Runtime server là Python 3.12 hệ thống; máy local chỉ dev/debug CPU. Xem [hồ sơ server](../../docs/server_environment.md) và [workflow CPU](../../docs/cpu_dev_workflow.md). Artifact đặt dưới `outputs/context_adaptive_dflash/`, được gitignore; chưa tạo artifact giả hoặc công bố speedup từ đặc tả.
