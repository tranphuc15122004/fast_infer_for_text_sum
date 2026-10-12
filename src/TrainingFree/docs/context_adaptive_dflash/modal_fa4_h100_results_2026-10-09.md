# Modal FA4/H100 benchmark artifacts — 2026-10-09

Run này chạy Context-Adaptive DFlash và các ablation trên Modal H100 với FlashAttention-4. Đây là pilot hiệu năng, không phải heldout result hoặc quality claim.

## Run và môi trường

| Hạng mục | Giá trị |
|---|---|
| Run ID | `cadflash_lengthbins_modal_fa4_h100_20261009_001` |
| Modal app | `ap-Jc2sQuY2dxmCC7e3vb2eoZ` |
| GPU | NVIDIA H100 80 GB HBM3, compute capability 9.0 |
| Attention backend | `flash_attention_4`; FA4 import/kernel probe thành công, cả 9 dev manifests đều ghi backend này |
| Runtime | Python 3.12.10; Torch 2.13.0+cu130; CUDA 13.0; Transformers 5.12.1; FlashAttention-4 4.0.0b19; CUTLASS DSL 4.7.1 |
| Models | Qwen3-4B target + Qwen3-4B-DFlash-b16 draft, BF16 |
| Dữ liệu | `data/length_bins/`: `gov_report`, `multi_news`, `qmsum`; loại code-completion |
| Split | 30 calibration documents + 15 dev documents; 3 dev documents mỗi length bin; seed 42, chia theo source group |
| Dev protocol | Greedy, 3 warmups, 3 repetitions, batch size 1, `max_new_tokens=2048` |
| Variant coverage | 9 variants × 15 dev documents × 3 repetitions = 405 requests; đủ 45 request cho mỗi variant |

## Kết quả pooled

Speedup throughput = `mean tok/s(variant) / mean tok/s(baseline)`. Acceptance length và tok/s là trung bình request-level.

| Variant | Mean acceptance length | Mean tok/s | So với DFlash | So với AR | Exact token IDs vs DFlash |
|---|---:|---:|---:|---:|---:|
| AR | — | 31.63 | 0.40× | 1.00× | 6/45 |
| DFlash full fixed | 3.819 | 78.63 | 1.00× | 2.49× | 45/45 |
| `a_only` | 4.083 | 93.31 | 1.19× | 2.95× | 45/45 |
| `b_only_history` | 3.856 | 85.94 | 1.09× | 2.72× | 39/45 |
| `b_only_entropy` | 3.805 | 87.91 | 1.12× | 2.78× | 6/45 |
| `independent_ab` | 4.007 | 80.51 | 1.02× | 2.55× | 15/45 |
| `joint_no_entropy` | 3.702 | 89.06 | 1.13× | 2.82× | 30/45 |
| `joint_no_source_relevance` | 3.702 | 87.12 | 1.11× | 2.75× | 30/45 |
| `joint` | 3.715 | 86.46 | 1.10× | 2.73× | 9/45 |

### Joint theo length bin

| Prompt length | Joint mean acceptance length | tok/s so với DFlash | tok/s so với AR |
|---|---:|---:|---:|
| 0–2K | 2.513 | 0.81× | 2.16× |
| 2–4K | 3.388 | 0.99× | 2.82× |
| 4–8K | 3.077 | 1.13× | 2.43× |
| 8–12K | 4.525 | 1.20× | 3.20× |
| 12–16K | 5.070 | 1.47× | 3.06× |

## Diễn giải và validity gates

- DFlash gốc đạt 78.63 tok/s so với AR 31.63 tok/s, tức 2.49× trên run H100/FA4 này. `joint` đạt 1.10× DFlash pooled, với lợi ích lớn hơn ở prompt 8–16K; hai bin ngắn nhất không cải thiện so với DFlash.
- `a_only` là biến thể nhanh nhất: +18.7% throughput so với DFlash, mean acceptance length tăng từ 3.819 lên 4.083. `joint` không vượt `a_only` pooled; interaction giữa A/B chưa được chứng minh là có lợi.
- **Chưa đạt exact greedy parity với AR.** DFlash full fixed chỉ khớp AR 6/45 request; `joint` khớp AR 3/45. Vì thế speedup so AR chỉ là throughput measurement, chưa phải claim tăng tốc giữ nguyên output. `a_only` khớp DFlash 45/45 nhưng vẫn chỉ khớp AR 6/45.
- **405/405 request đều sinh đủ 2.048 token và chạm cap.** Dù EOS stopping được bật, run này chưa đánh giá output kết thúc tự nhiên. Chưa có ROUGE/quality evaluation và mỗi bin chỉ có 3 tài liệu dev độc lập.
- Report gate còn pending correctness; không dùng số liệu này làm paper headline. Cần xử lý parity shared DFlash/AR và kiểm tra EOS/cap trước khi mở rộng heldout.

## Artifact locations và tra cứu

`outputs/` được gitignore theo quy ước repo. Artifact được giữ trong workspace hiện tại và trong Modal volume `cadflash-lengthbins-20261009`.

| Artifact | Local path | Modal volume path |
|---|---|---|
| Self-contained bundle gồm raw run, sample/split manifests, metrics và report | `outputs/cadflash_lengthbins_modal_fa4_h100_20261009/fa4_complete_results_bundle.tar.gz` | `assets/fa4_complete_results_bundle.tar.gz` |
| Metrics JSON | `outputs/cadflash_lengthbins_modal_fa4_h100_20261009/lengthbin_metrics.json` | Được chứa trong bundle |
| Full report theo bins/ablations | `outputs/cadflash_lengthbins_modal_fa4_h100_20261009/lengthbin_results.md` | Được chứa trong bundle |
| Extracted raw results | `outputs/cadflash_lengthbins_modal_fa4_h100_20261009/retrieved/runs/cadflash_lengthbins_modal_fa4_h100_20261009_001/` | Được chứa trong bundle |
| Artifact manifest | `outputs/cadflash_lengthbins_modal_fa4_h100_20261009/artifact_manifest.json` | `assets/fa4_run_20261009/artifact_manifest.json` |

SHA-256:

- Bundle: `05275ad0c803ad3517a6e6145ec3978c753e5830b16309b23199a2abaf1d81e4`
- Metrics JSON: `b2a6c6691e61eac9ff4928f2cd6b9589f28f5b545c817ab774107aca5832ac80`
- Report Markdown: `c518f52b62c1ccb19d0c60459362170a66cb6aa615f3b8d7aa9de4c9d6b68ace`
- Input data: `a251a82be7f4343bba89d9f52b91424a3f0e49f7872e8ec70a4318e08b0585f0`
- Split manifest: `24bc79a8a4c48bdff14209899b5ac005686d8205cf88e5b5b83340f015fa5d83`

Đọc nhanh metrics pooled:

```bash
jq '.pooled | {ar, dflash_full_fixed, a_only, joint}' \
  outputs/cadflash_lengthbins_modal_fa4_h100_20261009/lengthbin_metrics.json
```

Đọc kết quả `joint` theo bin:

```bash
jq '.bins | map_values(.variants.joint)' \
  outputs/cadflash_lengthbins_modal_fa4_h100_20261009/lengthbin_metrics.json
```

Tái tạo report từ raw JSONL đã giải nén:

```bash
python3 scripts/analyze_cadflash_lengthbin_pilot.py \
  --root outputs/cadflash_lengthbins_modal_fa4_h100_20261009/retrieved \
  --run-id cadflash_lengthbins_modal_fa4_h100_20261009_001
```

Run protocol và resume-aware Modal runner: [runner](../../../../scripts/modal_cadflash_lengthbin_fa4_h100.py). Kết quả pilot Modal trước đó trên L40S/SDPA được lưu riêng trong [bản ghi pilot ban đầu](modal_pilot_results_2026-10-09.md).
