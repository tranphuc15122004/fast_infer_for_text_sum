# Context-Adaptive DFlash — đặc tả nghiên cứu

Ngày: **2026-10-08**. Trạng thái: **đã có thiết kế và protocol; chưa có launcher/executor, chưa là baseline khả dụng của matrix**.

Phương pháp training-free cùng điều chỉnh draft-context budget và speculative block length trên pretrained DFlash. Target giữ full context và exact verification; model weights không thay đổi.

Điểm vào tài liệu: [TrainingFree README](../../src/TrainingFree/README.md). Các contract:

- [Thuật toán và policy](../../src/TrainingFree/docs/context_adaptive_dflash/design.md).
- [Tích hợp/cache/verifier](../../src/TrainingFree/docs/context_adaptive_dflash/integration.md).
- [Protocol và gate](../../src/TrainingFree/docs/context_adaptive_dflash/experiment_protocol.md).
- [Output/timing](../../src/TrainingFree/docs/context_adaptive_dflash/measurement_contract.md).
- [Runbook/master-env/CLI dự kiến](../../src/TrainingFree/docs/context_adaptive_dflash/runbook.md).
- [Kế hoạch task/file cụ thể](../../src/TrainingFree/plans/2026-10-08-context-adaptive-dflash-implementation.md).

Tên dispatcher dự kiến là `context_adaptive_dflash`. Chỉ thêm nó vào danh sách baseline thực nghiệm sau khi implementation đạt correctness/preflight gates. Hiện có thể chạy [DFlash baseline](dflash.md) để lấy control; không coi RECAP-KV launcher là phương pháp này.

Runtime theo [server_environment](../server_environment.md): Python3.12 hệ thống, model/dependency offline. Local CPU dành cho policy/schema checks; GPU thực nghiệm ở server. V1 batch1, full draft bank + gather, nên không mặc định giảm resident VRAM.

Kết quả tương lai ở `outputs/context_adaptive_dflash/`, dùng writer/schema chung và summary, ROUGE khi có reference; không commit artifacts. Bộ tài liệu chỉ mô tả điều kiện triển khai và thực nghiệm, không cung cấp speedup đã đo.
