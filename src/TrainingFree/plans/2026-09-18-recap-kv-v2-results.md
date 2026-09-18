# RECAP-KV V2 Source-State Lease — kết quả thực nghiệm

**Ngày:** 2026-09-18
**Mục tiêu:** kiểm tra liệu query-drift certificate có tạo được lease đủ dài
để giảm full source-K rescoring hay không.
**Phạm vi:** certificate/headroom instrumentation; chưa physical eviction,
quantization hoặc offload.

## 1. Các lần chạy

### Smoke

- Run ID: `recap-v2-lease-smoke-qwen06b`
- GPU: Modal A100-80GB request; CUDA hoạt động.
- Mẫu: 1 GovReport, tối đa 32 token.
- Kết quả runtime: `1/1` sample hợp lệ, schema `recap.lease.trace.v1`.
- Kết quả sample: lease length `1`, certified cold fraction `0.195876`,
  violation rate `0.0`, mean bound `0.944811`, actual cold mass `0.005449`.
- Scientific status: `INCONCLUSIVE` vì smoke không đủ hai dataset/10 document.

### Pilot chính

- Run ID: `recap-v2-lease-pilot-qwen06b-a10g-rerun`
- Modal app: https://modal.com/apps/tdphuc-work/main/ap-H2cQoFCfppXivbP2djSdRg
- Checkpoint: `/opt/models/Qwen3-0.6B` từ checkpoint local hiện có.
- GPU thực tế theo manifest: `NVIDIA A10`.
- Venv: `/mnt/fast-in/venv/bin/python`.
- Dữ liệu: 20 GovReport + 20 Multi-News.
- Decode cap: 32 token/sample. Đây là pilot headroom rút gọn; chưa phải
  full 128-token study vì instrumentation phải full source-score lại khi lease
  expire.
- Kết quả runtime: `40/40` sample hợp lệ, `0` lỗi, `0` inconclusive.
- Thời gian: `1463.396s`.
- Decode steps trung bình: `31.1`/document.

Trace compact và manifest nằm trong Modal Volume:

```text
outputs/recap_kv_v2/recap-v2-lease-pilot-qwen06b-a10g-rerun/
├── lease_trace.jsonl
├── lease_metrics.jsonl
├── lease_metrics.json
├── manifest.json
└── lease_report.md
```

## 2. Cách tính metric

Tại anchor, source block được chọn vào hot set để giữ ít nhất 95% source mass.
Với query drift `D`, certificate dùng:

```text
U_C(D) = sum exp(log_Z0[i] + D * kappa[i])       for cold blocks
L_H(D) = sum exp(log_Z0[i] - D * kappa[i])       for hot blocks
bound  = U_C / (U_C + L_H + exact_live_Z)
```

Lease expire khi `bound > delta_cert=0.10`. Khi expire, bước đó được tính là
refresh/full source score. Numerical tolerance của actual-vs-bound là `1e-4` do
Q/K được đọc từ bfloat16 cache.

Các metric dưới đây được recompute từ `lease_trace.jsonl` bằng evaluator sau
patch event accounting. Bản `lease_metrics.json` được ghi trong lần chạy đã
được tạo trước patch đánh dấu expiry refresh, nên không dùng refresh rate raw
của file đó.

## 3. Kết quả aggregate — 40 document

| Metric | Giá trị |
|---|---:|
| Certificate violation rate | 0.000000 |
| Mean bound | 0.918639 |
| Mean actual cold mass | 0.011961 |
| Mean bound tightness | 0.906678 |
| Median lease length (mean theo document) | 1.525000 |
| Valid steps (mean theo document) | 2.475000 |
| Certified cold fraction | 0.164768 |
| Refresh rate | 0.951643 |
| Source-score avoidance proxy | 0.048357 |

### Kill gate

| Criterion | Threshold | Result | Pass |
|---|---:|---:|:---:|
| Certificate violation rate | `<= 0.001` | 0.000000 | yes |
| Certified cold fraction | `>= 0.20` | 0.164768 | no |
| Median lease length | `>= 4` | 1.525000 | no |
| Refresh saving | `> 0` | 0.048357 | yes |

**Gate decision: `STOP_BEFORE_PHYSICAL`.**

## 4. Kết quả theo dataset

| Dataset | N | Median lease | Cold fraction | Violation | Refresh rate | Score avoidance |
|---|---:|---:|---:|---:|---:|---:|
| GovReport | 20 | 0.000000 | 0.180093 | 0.000000 | 1.000000 | 0.000000 |
| Multi-News | 20 | 3.050000 | 0.149443 | 0.000000 | 0.903287 | 0.096713 |

GovReport có lease gần như expire ngay sau anchor. Multi-News ổn định hơn một
chút nhưng vẫn dưới ngưỡng lease 4 token và cold fraction 20%.

## 5. Đánh giá khoa học

### Điều được hỗ trợ

1. Query/source-K tracing chạy được end-to-end trên checkpoint Qwen3-0.6B với
   DynamicCache, GQA mapping và tất cả layer được instrument.
2. Certificate không có violation sau tolerance số học trên pilot cuối.
3. Có một lượng cold fraction không bằng 0, đặc biệt trên GovReport/Multi-News,
   nên method không bị chết vì không có source block nào để demote.

### Điều không được hỗ trợ

1. Certificate hiện tại quá conservative: actual cold mass trung bình chỉ
   `0.011961` nhưng bound trung bình là `0.918639`.
2. Lease không đủ dài: median aggregate `1.525` token; GovReport median theo
   document là `0` vì nhiều anchor expire ngay.
3. Refresh rate `95.16%` nghĩa là gần như mỗi decode step vẫn cần full source
   score. Source-score avoidance `4.84%` chưa có systems value đáng kể.
4. Không có cơ sở để triển khai physical KV tiering, low-bit K hoặc offload.

## 6. Lỗi instrumentation và xử lý

Một pilot đầu tiên (`recap-v2-lease-pilot-qwen06b-32`) có `40/40` lỗi do
vectorized geometry xử lý sai partial final block. Regression test đã tái hiện
lỗi, patch đã sửa indexing, và pilot cuối chạy `40/40` thành công.

Đây là lỗi runtime đã được sửa, không phải scientific result; metric của lần
chạy lỗi không được dùng.

Ngoài ra, trace đầu tiên không đánh dấu expiry row là refresh. Evaluator hiện
đã fallback theo `expired=True`, và kết quả trên báo cáo này đã tính lại đúng
refresh event từ compact trace.

## 7. HIGHEST VERIFIED RUNG

**R5 — short pilot trên real data và Modal GPU.** V2 đã vượt smoke/runtime
validation và hoàn tất 40 document không lỗi. Đây chưa phải R6 full study:
decode cap chỉ 32 token, một target model, một seed và chưa có physical quality
ablation.

## 8. Quyết định tiếp theo

Không mở physical tiering ở trạng thái hiện tại. Hướng sửa hợp lý nhất là làm
certificate tight hơn trước khi chạy lại:

- dùng bound theo từng layer/head thay vì trung bình geometry có thể làm mất
  cấu trúc;
- tách positional/RoPE drift khỏi semantic query drift;
- dùng lower bound denominator chặt hơn cho live/hot contribution;
- đánh giá block size nhỏ hơn và so sánh bound với actual theo từng head.

Chỉ khi lease length và certified cold fraction cùng vượt kill gate mới triển
khai KV precision lease hoặc offload.
