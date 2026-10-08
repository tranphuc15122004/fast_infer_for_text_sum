# Bộ tài liệu triển khai AMR-DFlash

Ngày: **08/10/2026**. Trạng thái: **V0 code và launcher đã triển khai; CPU contract tests pass; model thật/B200 training và benchmark chưa chạy**.

Nguồn quyết định nghiên cứu là [proposal AMR-DFlash ngày 08/10](../AMR_DFlash_Research_Proposal_and_Paper_Story_2026-10-08.md). Bộ này chuyển proposal thành hợp đồng triển khai có thể kiểm tra; các lựa chọn V0 là cấu hình khởi đầu cho pilot, không phải kết quả khoa học.

## Thứ tự đọc

| Tài liệu | Dùng khi nào |
|---|---|
| [Đặc tả kỹ thuật](../../../docs/superpowers/specs/2026-10-08-amr-dflash-design.md) | Xây module, quyết định memory interface, vị trí và verifier |
| [Dữ liệu và huấn luyện](data_training_contract.md) | Capture state, tạo nhãn, train và artifact hiện có |
| [Giao thức thực nghiệm](experiment_protocol.md) | Chọn cohort, chạy pilot, đo cost và đánh giá giả thuyết |
| [Kế hoạch triển khai](../../../docs/superpowers/plans/2026-10-08-amr-dflash-implementation.md) | Thực hiện từng task với đầu vào, đầu ra và kiểm thử cụ thể |
| [Checklist nghiệm thu](acceptance_checklist.md) | Xác định milestone đã đạt và claim được phép |

## Phạm vi đã chốt

- Acceptance-first: accepted prefix là cơ chế; committed tokens/s và E2E là kết quả cuối.
- Giữ pretrained DFlash **5 layer**, **16 live positions = 1 anchor + 15 proposals**.
- Thêm local exact memory, selector học theo state, compressor bổ trợ và gate ngưỡng context.
- Target và DFlash frozen; attention warm-up không bắt buộc.
- V0 có dense/selection/compressor/hybrid ablation và gate override. Memory reuse policy chưa triển khai.
- Target giữ full context. V0 triển khai **greedy**; sampling chỉ được mở sau khi có verifier đúng phân phối.

## Nơi triển khai dự kiến

Core mới nằm ở `src/AMR_DFlash/`. Không chuyển kiến trúc HCA/CSA hai stage cũ thành AMR bằng cách đổi tên.

Entrypoint pipeline là `scripts/amr_dflash/cli.py`; baseline được dispatch qua `scripts/infer_amr_dflash.py`, `scripts/runners/run_amr_dflash.sh` và `bash scripts/run.sh amr_dflash`. Preflight kiểm tra local assets và B200 capability mà không load model weights.

Artifact sinh ra nằm ở `outputs/amr_dflash/<run_id>/`, checkpoint ở `checkpoints/amr_dflash/<run_id>/`. Không commit tensors, outputs hoặc checkpoint. Launcher dùng master-env qua `config/master.path`; cấu hình YAML chỉ chứa hyperparameter và tên biến môi trường, không tạo master config thứ hai.

V0 có `dense`, `selection`, `compressor` và `amr` modes; capture target trajectory, tạo support candidates, target-verifier labels/preferences, train selector/compressor, fixed-state evaluation và rollout inference. Các acceptance/evidence gates trong checklist vẫn mở cho model thật, GPU, train-time validation cadence, cost calibration và holdout results.

## Ranh giới với code đã có

| Thành phần hiện có | Cách sử dụng |
|---|---|
| [DFlash pretrained](../../../externals/dflash/dflash/model.py) | Nguồn backbone, projections, normalization và dense generator |
| [Probe attention](../../../scripts/probe_dflash_attention.py) | Tham khảo scope/query/feature alignment và candidate offline |
| [Engine reference MR/DFlash](../../../src/MR_DFlash/inference.py) | Tham khảo greedy correction/EOS; không dùng làm production timing |
| [Feature manifest](../../../src/Finetuning/features.py) | Tham khảo fingerprint, lazy load và validation; AMR bổ sung state/candidate schema riêng |
| [JSONL writer](../../../scripts/common/io_util.py) và [ROUGE](../../../scripts/common/rouge.py) | Dùng trực tiếp cho record/summary và task metrics |

## Trình tự thực hiện

`T1–T11` có hiện thực V0 rút gọn nhưng nhiều acceptance gate vẫn mở; `T12` pilot/ablation và mọi GPU experiment chưa thực hiện. CPU tests kiểm tra semantics/gradient trên tiny synthetic Qwen3; chỉ server GPU mới xác nhận checkpoint thật, fit, latency vật lý và quality. Thông tin runtime: [hồ sơ server](../../../docs/server_environment.md), [CPU workflow](../../../docs/cpu_dev_workflow.md).
