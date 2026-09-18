# RECAP-KV V0 — kết quả chạy Modal

## Runtime

- Target frozen: `/opt/models/Qwen3-0.6B`, mount từ checkpoint local hiện có.
- GPU: Modal `A100-80GB`; `cuda_available=true`.
- Python: `/mnt/fast-in/venv/bin/python` trong Volume
  `fast-infer-text-sum-cache`.
- Dữ liệu: 20 mẫu `GovReport` + 20 mẫu `Multi-News`.
- Cấu hình: block 128, segment 16 token, tối đa 128 token sinh, seed 42,
  budget `K={1,2,4,8}`.
- Kết quả chạy: `40/40` mẫu hợp lệ, `0` lỗi, runtime khoảng `202.4s`.
- Run ID: `recap-v0-pilot-qwen06b-ablation`.

Artifact được lưu trong Modal Volume tại:
`outputs/recap_kv_v0/recap-v0-pilot-qwen06b-ablation/`.

## Kết quả chính

Scientific gate: **`GATE_FAIL`**.

Delta là `policy - current attention`; số âm nghĩa là kém baseline.

| K | RECAP hidden Recall Δ | RECAP hidden NDCG Δ | Attention-residual Recall Δ | Historical Recall Δ |
|---:|---:|---:|---:|---:|
| 1 | -0.2969 | -0.2969 | -0.0121 | -0.0902 |
| 2 | -0.3073 | -0.3041 | -0.0139 | -0.0695 |
| 4 | -0.3160 | -0.3102 | -0.0141 | -0.0224 |
| 8 | -0.2911 | -0.2982 | -0.0050 | +0.0040 |

## Đánh giá

1. V0 đã chạy được end-to-end trên GPU với checkpoint và venv hiện có; trace,
   metrics, manifest và report đều được sinh.
2. Hypothesis chưa được ủng hộ trên pilot. Biến thể RECAP dùng hidden-state
   semantic index kém current attention ở mọi budget.
3. Ablation `current attention + residual` gần baseline hơn nhiều (kém khoảng
   0.5–1.4 điểm phần trăm), cho thấy semantic index/prototype là điểm hỏng
   chính của V0. Đây là chẩn đoán, chưa phải chứng minh nhân quả.
4. Do gate fail, chưa triển khai physical KV eviction/offload và chưa claim
   speedup. Bước hợp lý tiếp theo là thay proxy hidden-state bằng tín hiệu
   target-key/query hoặc thiết kế consumption không phụ thuộc cosine giữa
   summary hidden và source hidden, rồi chạy lại cùng pilot.

