# Tiến độ thực nghiệm Context-Adaptive DFlash

Ngày: **2026-10-09**. Đây là trạng thái của lần chạy pilot; chưa phải kết quả heldout hoặc claim speedup.

## Trạng thái hiện tại

| Bước | Trạng thái | Bằng chứng |
|---|---|---|
| R0: protocol/metric | pass | [Ma trận](experiment_matrix.md): paired geometric macro E2E speedup; thresholds G3–G5 đã đăng ký trong protocol |
| R1: interpreter/config/CLI | pass | Local Python 3.12.13; CLI/import/preflight; `evidence/r1_cpu_preflight.json` |
| Corpus primary/exposure/split | pass cục bộ | 300 records, 220 source groups; split complete; calibration/dev/test = 44/44/132 records. Exposure audit ghi 10 probe IDs `absent_verified` trong primary local; cần giữ audit cùng run khi đổi server/code snapshot. |
| G0: asset/config/runtime | pass một phần | Modal preflight đọc config; 36 target layers, 5 draft target layers, vocab/hidden size khớp; 10 file target từ snapshot ghim khớp SHA-256 với Modal volume. Runtime thật là L40S/BF16/SDPA. Collector/kernel và parity chưa được chứng nhận. |
| R2: forward thật | pass | Smoke AR và DFlash fixed chạy đủ 6 dev requests, không lỗi runtime |
| R4: GPU smoke | pass | Qwen3-4B + Qwen3-4B-DFlash-b16 trên Modal L40S; Python 3.12.10, Torch 2.13.0+cu130, Transformers 5.12.1; backend SDPA |
| R5: calibration/dev pilot | đã chạy, exploratory | 6 calibration records, 72 replay action rows; paired dev pilot 12 records × AR/DFlash fixed/joint. Cap 128 token, 1 repetition. Xem [kết quả pilot Modal](modal_pilot_results_2026-10-09.md). |
| G1/G2: exactness/cache/boundary | **fail/pending** | Report thấy 5/12 output token-ID mismatches ở cả DFlash fixed và `joint` so với AR. Harness tensor/cache/boundary đầy đủ chưa chạy; chưa biết mismatch đến từ verifier/cache hay numerical backend. |
| G3–G6, heldout, R7 | pending | Pilot report `headline_valid=false`; chưa lock config hoặc chạy test split. |

## Artifact local

Artifacts dữ liệu/split local nằm ở `.gitignore` tại `outputs/context_adaptive_dflash/m0_local_primary_cpu_20261009/`. Modal pilot và kết quả tải về nằm ở `.gitignore` tại `outputs/modal_cadflash_pilot_20261009/`, gồm bundle JSONL/model/runtime evidence và report. Entrypoint chạy lại pilot là [modal_context_adaptive_pilot.py](../../../../scripts/modal_context_adaptive_pilot.py).

Primary file SHA-256 dùng trong pilot vẫn khớp corpus đã chuẩn bị:

| Dataset | SHA-256 |
|---|---|
| GovReport | `3dbba6ee4874e2a00b9d0c706524ea2dfb47a60169491a54464b686189299813` |
| Multi-News | `0bc56649bac6b80b622f11e655ad1887316242523c12e45e219d72bb01557d6a` |
| QMSum | `3ebf2c39a0a3a4ffc5a8ff367837b550e2b017bc125791c3ebd13fab06f9a689` |

## Bước tiếp theo

1. Dừng tuning performance và chạy checkpoint-level G1/G2 harness trên các mismatch: so logits AR với verifier tại cùng prefix/position, kiểm cache commit/crop và branch reject; lưu first divergence và top-1 margin. DFlash fixed cũng mismatch nên cần kiểm shared verifier/cache/backend path trước khi kết luận lỗi riêng của controller.
2. Khi greedy output đạt exactness, tăng calibration support theo [runbook](runbook.md): tối thiểu cohort đã đăng ký, 3 repetitions, ghi coverage theo state/cost bucket; xử lý fallback `full_fallback_uncalibrated_cost`.
3. Chạy lại performance trên cùng GPU/backend với output cap dài/natural EOS, đủ warmup theo shape và 3 paired repetitions. Dùng đầy đủ dev; chọn cấu hình rồi lock trước khi chạm test.
4. Ghi rõ khác biệt runtime so với server mục tiêu. L40S pilot dùng SDPA; nếu mục tiêu dùng Blackwell/FA4 thì cần lặp lại G0/G1/G2 và đo chính trên backend đó.

Kết luận hiện tại: **đã xác nhận đường chạy GPU và có tín hiệu acceptance thuận lợi khi chọn context 1024 trong calibration replay nhỏ; phương pháp chưa khả thi để claim inference speedup vì exact output parity chưa đạt và controller fallback phần lớn về full context.**
