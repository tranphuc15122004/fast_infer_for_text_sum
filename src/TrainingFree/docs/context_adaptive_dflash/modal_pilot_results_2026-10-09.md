# Kết quả pilot Context-Adaptive DFlash trên Modal

Ngày chạy: **2026-10-09**. Đây là pilot kỹ thuật và performance trên dev; không phải heldout result. Report tự sinh đặt `headline_valid=false` vì không đạt exact greedy parity.

## Kết luận

- Runner nạp được Qwen3-4B và Qwen3-4B-DFlash-b16 trên GPU, qua model-config preflight và chạy đủ smoke/calibration/dev mà không có inference error hoặc OOM.
- Replay ghép cặp cho thấy context budget 1024 có tín hiệu acceptance thuận lợi hơn full context trên cohort nhỏ: **52/300 proposal tokens được accept (17.3%) so với 34/300 (11.3%)**. Chi phí action trung bình giảm từ 50.77 ms xuống 49.47 ms, khoảng 2.6%.
- Tín hiệu đó chưa thành speedup E2E đáng tin. Adaptive `joint` fallback về full context trong **521/657 speculative rounds (79.3%)** do `full_fallback_uncalibrated_cost`; chỉ 134/657 rounds dùng budget 1024.
- Có **5/12 output token-ID mismatches** giữa AR và cả DFlash full fixed lẫn `joint`. Vì exactness là điều kiện bắt buộc, hai speedup point estimate trong report đều không hợp lệ để claim. Cần sửa/giải thích parity trước khi mở rộng benchmark.

## Thiết lập

| Hạng mục | Thiết lập |
|---|---|
| GPU | NVIDIA L40S, 46 GiB; capability 8.9 |
| Runtime | Python 3.12.10; Torch 2.13.0+cu130; Transformers 5.12.1; Hugging Face Hub 1.33.0 |
| Attention backend | SDPA; `flash_attn` không cài trong Modal image |
| Models | Qwen3-4B + Qwen3-4B-DFlash-b16; 10 file target cần cho snapshot ghim khớp SHA-256 với local snapshot |
| Data | GovReport, Multi-News, QMSum; 300 records; SHA-256 trùng với corpus chuẩn bị cục bộ |
| Split | Seed 42; calibration/dev/test = 44/44/132 records |
| Calibration pilot | 6 records (2 mỗi dataset), 1 repetition, checkpoints 0 và 64, 6 action choices; 72 replay rows |
| Dev pilot | 12 records (4 mỗi dataset), 1 repetition, 1 warmup, greedy decode, cap 128 tokens, temperature 0 |
| Variants | AR, DFlash full fixed (gamma 15), adaptive `joint`; budgets 1024/full, gammas 3/7/15 |

Mọi dev request sinh đúng 128 token; chúng đều chạm output cap. Vì vậy các timing này mô tả decode ngắn bị cap, không đại diện output dài tới EOS. Model load, hash/provenance và warmup được tính ngoài E2E theo runner.

## Dev E2E và exactness

| Variant | Requests thành công | Mean E2E | Median E2E | Acceptance rate | Avg. accepted length | Mean ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| AR | 12/12 | 3051.9 ms | 2878.4 ms | — | — | 0.12337 |
| DFlash full fixed | 12/12 | 3205.2 ms | 2498.5 ms | 0.1096 | 2.527 | 0.12355 |
| Adaptive `joint` | 12/12 | 3025.6 ms | 2500.4 ms | 0.1209 | 2.589 | 0.12223 |

Report paired geometric macro speedup là **1.105×** cho DFlash fixed, CI95 **0.885–1.373**, và **1.144×** cho `joint`, CI95 **0.923–1.410**. Bootstrap dùng 1000 samples. Cả hai so sánh có `all_greedy_exact=false`, `headline_valid=false`; CI cũng chứa 1. Arithmetic mean E2E của DFlash fixed còn cao hơn AR trong pilot này. Một repetition và 12 prompts không đủ để kết luận tốc độ.

Các output mismatch đầu tiên xuất hiện tại generated-token offsets 4, 39, 52, 53 và 71 trên 5 prompts:

- `gov_report_cd1b0d330f6ad3ec` (offset 4)
- `multi_news_59607a52b75b5e14` (offset 39)
- `multi_news_bec584228962f753` (offset 52)
- `qmsum_5cc9b09dd9cc9a0f` (offset 53 với DFlash fixed)
- `qmsum_201110c13428e207` (offset 71)

Mismatch cũng có ở DFlash full fixed, nên không thể quy lỗi riêng cho policy chọn context của `joint`. Root cause chưa xác định; SDPA với shape verify khác AR là một giả thuyết cần kiểm tra bằng logits/cache trace, chưa phải kết luận.

## Acceptance và lựa chọn context

Trong 72 calibration replay rows, có **36 cặp** so action budget 1024 với full tại cùng sample/checkpoint/gamma trên **6 source groups**. Các cặp dùng cùng causal prefix và cùng gamma:

| Gamma | Chênh accepted candidates/action (1024 − full) | 1024 tốt hơn / hòa / kém | Geometric action-cost ratio (full / 1024) |
|---:|---:|---:|---:|
| 3 | +0.50 | 4 / 8 / 0 | 1.007× |
| 7 | +0.75 | 4 / 8 / 0 | 1.029× |
| 15 | +0.25 | 3 / 7 / 2 | 1.028× |

Trong các replay rows, budget 1024 giữ khoảng 988 context tokens/layer so với 7457 của full. Nó accept nhiều hơn 18 proposal tokens trên tổng 300 proposal, trong khi action timing chỉ đổi nhẹ. Đây là tín hiệu ban đầu rằng context được chọn theo attention có thể giữ hoặc tăng acceptance ở các state này; sáu calibration documents chưa đủ để kết luận hiệu ứng tổng quát.

On dev, `joint` used budget 1024 in 134 of 657 speculative rounds (20.4%). It fell back to full context in 521 rounds due to uncalibrated cost support. Mean request acceptance moved from 0.1096 to 0.1209 and average accepted length from 2.527 to 2.589 relative to DFlash fixed, but these are one-repetition descriptive summaries on a cap-limited cohort and the output exactness gate failed.

## Đánh giá khả thi và thứ tự cải thiện

1. **Ưu tiên correctness.** Thêm checkpoint-level GPU harness cho các mismatch: replay đúng prefix, so target AR next-token logits với từng verifier row, ghi top-1/top-2 margin, accepted-prefix length, cache length/position và token đầu tiên khác. Force reject đầu, reject giữa block, accept hết và boundary gamma. Chạy DFlash fixed trước, sau đó `joint` để tách shared verifier/cache path khỏi policy.
2. **Tăng cost/state support.** Sau khi exactness đạt, calibration theo protocol với nhiều source groups và 3 repetitions; kiểm cost buckets mà controller dùng. Pilot hiện có 6 calibration records nhưng `joint` vẫn fallback uncalibrated cost ở 79.3% speculative rounds. Chưa hạ `min_state_support` để che thiếu dữ liệu; cần cải thiện coverage hoặc prior/fallback cost model có kiểm chứng.
3. **Đo lại E2E theo chiều dài thật.** Dùng full dev split, output cap dài hơn hoặc natural EOS, shape warmup, ít nhất 3 paired repetitions và thứ tự cell đã đăng ký. Báo latency, accepted tokens, effective acceptance, context bytes và controller/collector overhead cùng nhau.
4. **Khớp backend đích.** Pilot chạy L40S + SDPA. Nếu cấu hình server mục tiêu là Blackwell/FlashAttention-4, lặp G0/G1/G2 và performance trên đúng GPU/backend trước khi chuyển số liệu này thành dự đoán production.

Đánh giá: **ý tưởng có một tín hiệu cơ chế đáng theo đuổi ở calibration replay, nhưng phương pháp hiện chưa chứng minh được cải thiện inference khả thi**. Rào cản đầu tiên là greedy output mismatch; rào cản tiếp theo là cost support thấp khiến policy thường xuyên dùng full context. Chưa chạy test split và không dùng các con số pilot này làm paper claim.

## Artifacts và tái lập

- Modal runner: [modal_context_adaptive_pilot.py](../../../../scripts/modal_context_adaptive_pilot.py)
- Artifact root `.gitignore`: `outputs/modal_cadflash_pilot_20261009/`
- Full downloaded bundle: `outputs/modal_cadflash_pilot_20261009/results_bundle.tar.gz`
- Extracted result tree: `outputs/modal_cadflash_pilot_20261009/retrieved_full/runs/cadflash_modal_pilot_20261009/`
- Các file chính: `preflight.json`, `evidence/modal_hardware.json`, `calibration/calibration.json`, `calibration/action_observations.jsonl`, `dev/online/comparison.json`, `dev/online/report.md`, cùng request/round/output-token JSONL cho từng variant.
- Volume asset tạm đã được xóa sau khi archive được tải về; các Modal volumes có sẵn trước pilot vẫn còn nguyên. Local staging trong `outputs/modal_cadflash_pilot_20261009/assets/` cho phép dựng lại volume nếu cần chạy lại.

Local asset staging dùng lệnh:

```bash
modal volume create cadflash-context-adaptive-pilot-20261009
modal volume put cadflash-context-adaptive-pilot-20261009 \
  outputs/modal_cadflash_pilot_20261009/assets /
modal run --profile tdphuc-work scripts/modal_context_adaptive_pilot.py
```

Runner chỉ đọc target từ volume `fast-infer-text-sum-mr-dflash-timer`; pilot không ghi vào volume target. Xem `target_volume_verification.json` trước khi chạy lại để bảo đảm file target ghim khớp.
