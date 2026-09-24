# FidelityKV — E44–E47 experiment card

## Câu hỏi và giả thuyết

Sau E43, chuyển từ bỏ source KV sang giữ toàn bộ source nhưng hạ fidelity có
chọn lọc. Câu hỏi đầu tiên: fake-quantize source K/V ở 4–8 bit có giữ được
logit behavior khi generated KV vẫn BF16 không?

## Protocol đã khóa

- Model: checkpoint local `Qwen3-0.6B`, 28 layers, 16 query heads / 8 KV heads.
- Cohort: đúng 40 mẫu đầu của E43 — 20 `gov_report`, rồi 20 `multi_news`;
  dùng các `generated_token_ids` trong E43 làm teacher-forced decode prefixes.
- Decode: tối đa 32 bước, greedy reference, seed 42; dừng theo EOS như E43.
- Baseline: dense BF16 source + generated KV. Candidate chỉ fake-quantize source
  cache, giữ phần prefix ngoài document và generated KV ở BF16.
- Fake quant: symmetric uniform quantization rồi dequantize về BF16; scale
  riêng cho mỗi layer/KV-head/head-dimension, hiệu chuẩn min/max trên source
  sequence. Không có kernel hoặc physical storage claim ở E44.
- Metrics chính: token Top-1 agreement và relative ΔNLL trên cùng dense-reference
  prefixes. Metrics phụ: forward KL `D_KL(p_dense || p_quant)`, layer attention
  output relative-L2 distortion, reconstruction error, và ước lượng bit-rate.
- Search gate: Top-1 agreement ≥99%, relative ΔNLL ≤1%, mean attention-output
  relative-L2 ≤1% và P99 ≤5%. Đây là ngưỡng sàng lọc nội bộ, chưa phải ngưỡng
  paper-quality.

## Ma trận tuần tự

1. E44 baseline + one-sided: K16/V16, K16/V8, K16/V4, K8/V16, K4/V16.
2. Chỉ khi precision một phía đạt gate, chạy joint combinations cần thiết
   (ví dụ K8/V4, K4/V8, K8/V8, K4/V4).
3. Nếu không có uniform candidate đạt gate, chạy E45 layer sensitivity bằng
   counterfactual local attention-output error, sau đó E46 per-KV-head map.
   Mixed-precision proposal chỉ được promote khi replay toàn model trên cùng
   cohort đạt lại search gate và average source precision ≤8 bits/component.
4. E47 chỉ chạy khi có candidate hỗn hợp đủ tốt; đối chiếu sensitivity với
   prefill-only K/V norms và reconstruction error. K95/entropy từ decode được
   xem là comparator hậu nghiệm, không được gọi là prefill signal.
5. Fused physical prototype chỉ khi có candidate đạt gate. Dequantize-to-BF16
   trước attention không được tính là bandwidth speedup.

## Feedback ladder / stop rules

- R0: card này khóa câu hỏi, baseline, metrics và gate.
- R1: import/CLI/config/path/schema checks.
- R2: CPU synthetic quantization/cache-span checks.
- R4: Modal A10G smoke trên 1 mẫu, tối đa 4 token; chỉ promote khi baseline và
  mọi candidate hoàn tất, metrics hữu hạn, dense reference tái lập được.
- R5: 4 mẫu (2/dataset), 32 token; promote khi không có lỗi số học/cache và
  không thấy candidate rõ ràng fail cả hai primary metrics.
- R6: 40 mẫu hoàn chỉnh. Mẫu lỗi không được âm thầm bỏ khỏi denominator.
- R7: quyết định E44 → dừng, E45/E46, hoặc physical prototype; báo cáo riêng
  quality headroom và physical speedup (nếu có).

## Reproducibility

E43 source trace: Modal profile `tdphuc-work`, Volume
`fast-infer-text-sum-cache`, run
`e43-temporal-pilot-qwen06b-a10g-20260923`. E44 outputs dùng run ID mới, không
ghi đè artifact E43. Hardware dự kiến: Modal A10G, Python 3.12 persistent venv.
