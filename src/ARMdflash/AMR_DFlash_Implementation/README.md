# Bộ tài liệu triển khai AMR-DFlash

Ngày: **08/10/2026**. Trạng thái: **đặc tả và kế hoạch; chưa triển khai AMR, chưa chạy training/GPU benchmark**.

Nguồn quyết định nghiên cứu là [proposal AMR-DFlash ngày 08/10](../AMR_DFlash_Research_Proposal_and_Paper_Story_2026-10-08.md). Bộ này chuyển proposal thành hợp đồng triển khai có thể kiểm tra; các lựa chọn V0 là cấu hình khởi đầu cho pilot, không phải kết quả khoa học.

## Thứ tự đọc

| Tài liệu | Dùng khi nào |
|---|---|
| [Đặc tả kỹ thuật](../../../docs/superpowers/specs/2026-10-08-amr-dflash-design.md) | Xây module, quyết định memory interface, vị trí và verifier |
| [Dữ liệu và huấn luyện](data_training_contract.md) | Capture state, tạo nhãn, train và resume |
| [Giao thức thực nghiệm](experiment_protocol.md) | Chọn cohort, chạy pilot, đo cost và đánh giá giả thuyết |
| [Kế hoạch triển khai](../../../docs/superpowers/plans/2026-10-08-amr-dflash-implementation.md) | Thực hiện từng task với đầu vào, đầu ra và kiểm thử cụ thể |
| [Checklist nghiệm thu](acceptance_checklist.md) | Xác định milestone đã đạt và claim được phép |

## Phạm vi đã chốt

- Acceptance-first: accepted prefix là cơ chế; committed tokens/s và E2E là kết quả cuối.
- Giữ pretrained DFlash **5 layer**, **16 live positions = 1 anchor + 15 proposals**.
- Thêm local exact memory, selector học theo state, compressor bổ trợ và gate theo cost.
- Target và DFlash frozen ban đầu; attention warm-up không bắt buộc.
- Full integrated pilot sớm; selector/compressor/gate/reuse có thể tắt độc lập.
- Target giữ full context. V0 triển khai **greedy**; sampling chỉ được mở sau khi có verifier đúng phân phối.

## Nơi triển khai dự kiến

Core mới: `src/AMR_DFlash/`. Đây là đường dẫn **sẽ tạo ở giai đoạn code**, không phải package đã tồn tại. Không chuyển kiến trúc HCA/CSA hai stage cũ thành AMR bằng cách đổi tên.

Entrypoint dự kiến: `scripts/amr_dflash/` cho pipeline nghiên cứu; `scripts/infer_amr_dflash.py` + `scripts/runners/run_amr_dflash.sh` + `docs/baselines/amr_dflash.md` cho benchmark khi integration đạt gate.

Artifact sinh ra nằm ở `outputs/amr_dflash/<run_id>/`, checkpoint ở `checkpoints/amr_dflash/<run_id>/`. Không commit tensors, outputs hoặc checkpoint. Launcher dùng master-env qua `config/master.path`; cấu hình YAML chỉ chứa hyperparameter và tên biến môi trường, không tạo master config thứ hai.

## Ranh giới với code đã có

| Thành phần hiện có | Cách sử dụng |
|---|---|
| [DFlash pretrained](../../../externals/dflash/dflash/model.py) | Nguồn backbone, projections, normalization và dense generator |
| [Probe attention](../../../scripts/probe_dflash_attention.py) | Tham khảo scope/query/feature alignment và candidate offline |
| [Engine reference MR/DFlash](../../../src/MR_DFlash/inference.py) | Tham khảo greedy correction/EOS; không dùng làm production timing |
| [Feature manifest](../../../src/Finetuning/features.py) | Tham khảo fingerprint, lazy load và validation; AMR bổ sung state/candidate schema riêng |
| [JSONL writer](../../../scripts/common/io_util.py) và [ROUGE](../../../scripts/common/rouge.py) | Dùng trực tiếp cho record/summary và task metrics |

## Trình tự thực hiện

`T1–T3` khóa backbone và sparse interface; `T4–T6` tạo state/preference và selector; `T7–T8` thêm compressor và evaluator; `T9` tích hợp train; `T10–T12` hoàn thiện cached execution, launcher và pilot/ablation.

Tất cả task bắt đầu ở trạng thái chưa thực hiện. CPU smoke kiểm tra semantics/gradient; chỉ server GPU mới xác nhận acceptance trên checkpoint thật và latency vật lý. Thông tin runtime: [hồ sơ server](../../../docs/server_environment.md), [CPU workflow](../../../docs/cpu_dev_workflow.md).
