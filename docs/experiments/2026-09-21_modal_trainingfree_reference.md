# Reference record — Modal baseline và Training-Free

Đây là snapshot đối chiếu cho các phát triển Training-Free về sau. Không sửa
các số trong record này; nếu chạy lại với config khác, tạo record mới và ghi
liên hệ với `record_id` này.

## Identity

- `record_id`: `modal-trainingfree-reference-20260921`
- Ngày chạy: `2026-09-21`
- Baseline run: `tf-screen-l50-20260921-core`
- Training-Free run: `recap-v3-screen-20260921`
- Baseline raw artifacts: Modal Volume `fast-infer-text-sum-cache`,
  `outputs/longbench_100_14k/tf-screen-l50-20260921-core/`
- Training-Free raw artifacts: Modal Volume `fast-infer-text-sum-cache`,
  `outputs/recap_kv_v3/recap-v3-screen-20260921/`

## Config baseline

| Trường | Giá trị |
|---|---|
| GPU | Modal A100-SXM4-80GB, 1 GPU |
| Model target | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| EAGLE draft | `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` |
| DFlash draft | `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` |
| Dataset | `gov_report`, `multi_news`, `qmsum` |
| Dataset profile | `data/longbench_100_14k`, 100 rows/dataset |
| Samples | 20/dataset, deterministic theo seed 42 |
| Max output | 128 tokens |
| Temperature | 0.0 |
| Warmup | 1 |
| Timeout | 3600 s/cell |
| Runtime | Python 3.12.10, torch 2.11.0+cu130, transformers 5.12.1, CUDA 13.0 |
| Preflight | `modal-longbench-core`; dependency riêng kiểm tra tại từng baseline |

Lệnh tái hiện:

```bash
MODAL_GPU=A100-80GB modal run scripts/modal_longbench.py \
  --mode representative \
  --baselines 'vanilla_hf,eagle3,dflash' \
  --datasets 'gov_report,multi_news,qmsum' \
  --max-samples 20 \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --run-id tf-screen-l50-20260921-core \
  --allow-unsupported
```

Manifest ghi nhận `source_manifest_sha256` là
`794c688f49ed896abcfc1c27b9944037d0e52dcb7d954c3b2091356125cdb7cb`, seed
42, `max_input_tokens=0` (dùng giới hạn mặc định của profile), và
`min_free_gb=32`.

## Bảng baseline reference

Các số là mean của các record hợp lệ. `E2E` tính bằng milliseconds; ROUGE là
F1. `n` phải được đọc cùng trạng thái coverage, không được xem các mean partial
là kết quả đầy đủ.

| Dataset | Method | n | E2E ms | tok/s | ROUGE-2 F | ROUGE-L F | Coverage |
|---|---|---:|---:|---:|---:|---:|---|
| `gov_report` | Vanilla HF | 20 | 6663.0 | 22.26 | 0.0903 | 0.1400 | complete |
| `gov_report` | EAGLE-3 | 20 | 4802.3 | 32.19 | 0.0862 | 0.1242 | complete; accept 4.23 |
| `gov_report` | DFlash | 20 | 3465.7 | 41.32 | 0.0960 | 0.1441 | complete; output checks pass |
| `multi_news` | Vanilla HF | 20 | 3037.2 | 43.17 | 0.0759 | 0.1526 | complete |
| `multi_news` | EAGLE-3 | 20 | 1872.0 | 73.45 | 0.0573 | 0.1404 | complete; accept 4.25 |
| `multi_news` | DFlash | 20 | 1488.0 | 95.74 | 0.0791 | 0.1538 | complete; output checks pass |
| `qmsum` | Vanilla HF | 4 | 6643.5 | 20.30 | 0.0915 | 0.2087 | OOM từ mẫu 5 |
| `qmsum` | EAGLE-3 | 5 | 5967.1 | 19.50 | 0.0634 | 0.1835 | OOM từ mẫu 6 |
| `qmsum` | DFlash | 20 | 4593.4 | 29.17 | 0.0829 | 0.2032 | complete; output checks pass |

Speedup ghép cặp có coverage đủ: DFlash so với Vanilla HF là `1.92x` trên
`gov_report` và `2.04x` trên `multi_news`; EAGLE-3 lần lượt là `1.39x` và
`1.62x`. Không claim speedup đầy đủ trên `qmsum` vì Vanilla/EAGLE bị OOM.

## Training-Free reference

RECAP-KV V3 chạy trên target khác nên là reference riêng, không gộp vào bảng
Llama 3.1 8B.

| Trường | Giá trị |
|---|---|
| Target | Qwen3-0.6B |
| GPU | Modal A10G |
| Dataset | `gov_report`, `multi_news`, `qmsum`, 20 mẫu/dataset |
| Max output | 32 tokens |
| Samples | 60/60, error 0, inconclusive 0 |
| Missed attention mass | 0 |
| Upper-bound violations | 0 |
| Exact expansion fraction | 1.0 |
| Active QK / full QK | 7303.6 / 7303.6 |
| Index overhead | 0.03359 |
| Routing fraction | 0.03614 |
| Aggregate status | `STOP_BEFORE_PHYSICAL` |

Diễn giải chuẩn: audit chính xác và routing chạy được, nhưng chưa có KV
physical reduction, VRAM reduction, latency speedup hoặc quality comparison.

## Quy tắc dùng làm mốc phát triển

1. Khi đánh giá Training-Free mới, giữ target model, tokenizer, dataset subset,
   seed, output budget và GPU profile cố định; nếu thay đổi thì tạo cột/config
   mới, không overwrite record này.
2. Báo cáo riêng `prefill`, `decode`, `E2E`, throughput, peak memory và quality;
   không dùng một metric để đại diện cho toàn bộ hệ thống.
3. Giữ các ô OOM/blocked/partial bằng trạng thái rõ ràng. Không điền số thay
   thế để làm đủ bảng.
4. RECAP-KV chỉ được đưa vào bảng speedup chung sau khi chạy cùng target Llama
   3.1 8B hoặc có một bảng Qwen3 riêng cho toàn bộ baseline tương thích.
5. Raw JSONL/manifest vẫn giữ trên Modal Volume; `metrics_summary.json`, CSV
   và Markdown của run là nguồn số liệu chi tiết khi cần audit.

## Verification và giới hạn

- Modal orchestrator ghi `cell_count=9`, `failure_count=2`; aggregate report
  được tạo thành công.
- Hai failure là OOM ở `vanilla_hf/qmsum` và `eagle3/qmsum`.
- `vanilla_fa` chưa nằm trong record vì FlashAttention source build không hoàn
  tất; RocketKV chưa có adapter trong canonical Modal matrix.
- Sau khi tách preflight core khỏi dependency baseline-specific, contract tests
  liên quan đạt `33 passed`.

