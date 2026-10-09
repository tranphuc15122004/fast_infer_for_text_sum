# Context-Adaptive DFlash — đặc tả nghiên cứu

Ngày: **2026-10-09**. Trạng thái: **runner và tài liệu thực nghiệm đã có; chưa xác nhận parity/correctness trên GPU và chưa đủ điều kiện đưa vào baseline matrix chính thức**.

Phương pháp training-free cùng điều chỉnh draft-context budget và speculative block length trên pretrained DFlash. Target giữ full context và exact verification; model weights không thay đổi.

Điểm vào tài liệu: [TrainingFree README](../../src/TrainingFree/README.md). Các contract:

- [Thuật toán và policy](../../src/TrainingFree/docs/context_adaptive_dflash/design.md).
- [Tích hợp/cache/verifier](../../src/TrainingFree/docs/context_adaptive_dflash/integration.md).
- [Protocol và gate](../../src/TrainingFree/docs/context_adaptive_dflash/experiment_protocol.md).
- [Ma trận, exposure audit và tuning](../../src/TrainingFree/docs/context_adaptive_dflash/experiment_matrix.md).
- [Kiểm chứng G0/G1/G2 trên GPU](../../src/TrainingFree/docs/context_adaptive_dflash/gpu_validation.md).
- [Mẫu báo cáo thực nghiệm](../../src/TrainingFree/docs/context_adaptive_dflash/results_template.md).
- [Output/timing](../../src/TrainingFree/docs/context_adaptive_dflash/measurement_contract.md).
- [Runbook/master-env/CLI](../../src/TrainingFree/docs/context_adaptive_dflash/runbook.md).
- [Kế hoạch task/file cụ thể](../../src/TrainingFree/plans/2026-10-08-context-adaptive-dflash-implementation.md).

Dispatcher: `bash scripts/run.sh context_adaptive_dflash`. Lệnh này dùng CLI mới, khác [DFlash baseline](dflash.md) và RECAP-KV. Bắt đầu bằng `prepare`, `preflight --model-check`, rồi server smoke; không đưa vào benchmark matrix hoặc trích dẫn speedup trước khi G0/G1 và các correctness gates đạt.

Runbook stage chỉ ba datasets summarization và xuất dev-only corpus cho smoke: phase smoke hiện chưa tự lọc dev theo split manifest. Chốt exposure audit trước heldout; report/lock chưa tự chứng nhận mọi scientific gates.

Runtime theo [server_environment](../server_environment.md): Python3.12 hệ thống, model/dependency offline. Local CPU dành cho policy/schema checks; GPU thực nghiệm ở server. V1 batch1, full draft bank + gather, nên không mặc định giảm resident VRAM.

Artifacts nằm dưới `outputs/context_adaptive_dflash/`, dùng writer/schema chung, summary, ROUGE khi có reference và paired report. V1 hỗ trợ `draft_shape`, `frozen_cost`, batch 1; `verify_prefix` và `async_event` chưa được triển khai. Local chỉ dành cho CPU/static checks; model parity và latency thực hiện trên server GPU. Chưa có số đo speedup.
