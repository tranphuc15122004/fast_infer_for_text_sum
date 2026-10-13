# Tiến độ thực nghiệm Context-Adaptive DFlash

Ngày: **2026-10-09**. Đây là trạng thái của lần chạy pilot; chưa phải kết quả heldout hoặc claim speedup.

## Trạng thái hiện tại

| Bước | Trạng thái | Bằng chứng |
|---|---|---|
| R0: protocol/metric | pass | [Ma trận](experiment_matrix.md): paired geometric macro E2E speedup; thresholds G3–G5 đã đăng ký trong protocol |
| R1: interpreter/config/CLI | pass | Local Python 3.12.13; CLI/import/preflight; `evidence/r1_cpu_preflight.json` |
| Corpus primary/exposure/split | pass cục bộ | 300 records, 220 source groups; split complete; calibration/dev/test = 44/44/132 records. Exposure audit ghi 10 probe IDs `absent_verified` trong primary local; cần giữ audit cùng run khi đổi server/code snapshot. |
| G0: asset/config/runtime | pass cho pilot FA4 | Modal H100 80 GB; checkpoint target khớp SHA-256; FA4 import/kernel probe qua; cả 9 dev manifests ghi `flash_attention_4`. Runtime chi tiết và artifact ở [kết quả FA4/H100](modal_fa4_h100_results_2026-10-09.md). |
| R2: forward thật | pass | Smoke AR và DFlash fixed chạy đủ 6 dev requests, không lỗi runtime |
| R4: GPU smoke | pass | Qwen3-4B + Qwen3-4B-DFlash-b16 đã smoke trên Modal L40S/SDPA và Modal H100/FA4; run FA4 đủ AR/DFlash smoke, calibration và dev. |
| R5: calibration/dev performance | đã chạy, exploratory | Run FA4: 30 calibration docs, 15 dev docs; 9 variants × 45 request, 3 repetitions; 405/405 chạm cap 2048 token. DFlash đạt 2.49× AR; `joint` đạt 1.10× DFlash; `a_only` đạt 1.19× DFlash. Xem [kết quả FA4/H100](modal_fa4_h100_results_2026-10-09.md). |
| G1/G2: exactness/cache/boundary | **fail/pending** | L40S pilot mismatch 5/12. Trong run FA4, DFlash full fixed chỉ khớp AR 6/45; `joint` khớp AR 3/45. Harness tensor/cache/boundary đầy đủ chưa chạy; chưa xác định root cause. |
| G3–G6, heldout, R7 | pending | Đã chạy các ablation hiệu năng nhưng report `headline_valid=false`; chưa lock config, chưa đánh giá quality, chưa chạy test split. |

## Artifact local

Artifacts dữ liệu/split local nằm ở `.gitignore` tại `outputs/context_adaptive_dflash/m0_local_primary_cpu_20261009/`. Modal pilot L40S ban đầu ở `outputs/modal_cadflash_pilot_20261009/`; FA4/H100 raw JSONL, report, metrics và self-contained bundle ở `.gitignore` tại `outputs/cadflash_lengthbins_modal_fa4_h100_20261009/`. Bundle cũng được lưu trong Modal volume `cadflash-lengthbins-20261009` tại `assets/fa4_complete_results_bundle.tar.gz`. Chi tiết run ID, SHA-256 và lệnh query nằm trong [artifact record FA4/H100](modal_fa4_h100_results_2026-10-09.md) và `outputs/cadflash_lengthbins_modal_fa4_h100_20261009/artifact_manifest.json`.

Primary file SHA-256 dùng trong pilot vẫn khớp corpus đã chuẩn bị:

| Dataset | SHA-256 |
|---|---|
| GovReport | `3dbba6ee4874e2a00b9d0c706524ea2dfb47a60169491a54464b686189299813` |
| Multi-News | `0bc56649bac6b80b622f11e655ad1887316242523c12e45e219d72bb01557d6a` |
| QMSum | `3ebf2c39a0a3a4ffc5a8ff367837b550e2b017bc125791c3ebd13fab06f9a689` |

## Bước tiếp theo

1. Ưu tiên checkpoint-level G1/G2 harness trên H100/FA4: so logits AR với verifier tại cùng prefix/position, kiểm cache commit/crop và reject branch; lưu first divergence và top-1 margin. DFlash fixed cũng mismatch nên kiểm shared verifier/cache path trước khi quy lỗi cho controller.
2. Điều tra vì sao cả 405 dev request đều chạm cap 2048; xác nhận EOS token và dừng sinh tự nhiên trước khi benchmark dài hơn.
3. Sau khi parity đạt, chạy lại performance với nhiều source groups, context/output dài tự nhiên, đủ warmup theo shape và paired repetitions; chọn config rồi lock trước test.
4. Chỉ dùng H100/FA4 result làm performance pilot. Không tuyên bố quality hoặc exact acceleration khi chưa qua correctness gate và heldout.

Kết luận hiện tại: **đường chạy FA4/H100 hoạt động và pilot cho thấy `a_only`/`joint` có throughput cao hơn DFlash ở nhiều bin dài; tuy nhiên parity với AR chưa đạt, toàn bộ output chạm cap, và `a_only` vượt `joint` pooled. Vì vậy chưa đủ điều kiện claim tăng tốc suy luận giữ nguyên output hoặc kết luận quality.**

## Bổ sung 2026-10-13: exposure cho campaign A trên B200

Đã gộp probe, split exploration cũ và các Modal pilot vào
[`data/dflash_context_A_audit/combined_exposure_manifest.json`](../../../../data/dflash_context_A_audit/combined_exposure_manifest.json),
kèm [`exposure_audit.json`](../../../../data/dflash_context_A_audit/exposure_audit.json).
Đối soát theo ID/hash/fingerprint cho kết quả 135 sample identities khớp corpus
campaign và 29 identities được xác minh thuộc LongBench-200 nhưng không thuộc
corpus `longbench_100_14k`. Kết quả này chỉ tạo cohort; chưa chạy GPU B200.

Sau khi loại toàn bộ nhóm calibration/dev cũ và mọi pilot exposure, test còn
37 GovReport, 33 Multi-News, 3 QMSum groups. Vì thế campaign B200 tự chạy confirm
trên 18/18/3 groups và ghi partial quota; không claim đạt mục tiêu ban đầu
18/18/12. Runbook B200 hiện có hai lệnh `check` và `start`.
