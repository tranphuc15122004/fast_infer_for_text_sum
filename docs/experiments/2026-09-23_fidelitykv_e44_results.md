# FidelityKV — báo cáo E44

- Ngày chạy: 2026-09-23
- Model: `Qwen3-0.6B`, BF16
- Runtime: Modal A10G, Python 3.12, PyTorch `2.11.0+cu130`, Transformers `5.12.1`, CUDA 13.0

## Kết luận điều hành

E44 xác nhận có headroom ở mức logits khi chỉ nén source KV: K8/V16 đạt
Top-1 `99.196%`, ΔNLL `+0.334%`; K16/V8 đạt Top-1 `98.875%`, ΔNLL
`+0.556%`. Tuy nhiên, không cấu hình uniform nào qua toàn bộ search gate đã
khóa trước. K8V8 — cấu hình có compression lý thuyết 2× — giữ ΔNLL rất thấp
(`+0.024%`) nhưng Top-1 chỉ `98.875%`, attention-output error trung bình
`2.906%` và P99 `6.367%`.

Vì vậy E44 cho thấy quantization 8-bit là một tín hiệu đáng nghiên cứu, nhưng
chưa đủ căn cứ để physicalize hoặc tuyên bố speedup. Theo protocol, bước tiếp
theo là sensitivity theo layer/head và thử mixed precision có điều kiện.

## Câu hỏi và phương pháp

E44 kiểm tra liệu có thể giữ toàn bộ source states nhưng biểu diễn source K/V
bằng ít bit hơn, trong khi generated KV vẫn BF16. Cohort gồm 40 tài liệu giống
E43: 20 GovReport và 20 Multi-News; mỗi continuation dùng token IDs từ trace
E43 teacher-forced, tối đa 32 token. Có 1,244 target tokens tổng cộng.

Với mỗi mẫu, dense BF16 được chạy lại làm tham chiếu trên cùng prefix. Sau đó
source cache được fake-quantize rồi dequantize về BF16; prefix ngoài document
và generated KV không bị quantize. Scale đối xứng được tính độc lập theo
layer/KV-head/head-dimension trên chiều source. Đây là phép thử chất lượng
offline, không phải packed cache hay kernel nén.

Đo: Top-1 agreement so với dense replay mới, relative ΔNLL trên token tiếp
theo, forward KL, và relative-L2 distortion của projected attention output.
Gate sàng lọc đã khóa trước: Top-1 ≥99%, ΔNLL ≤1%, attention-output error
mean ≤1%, P99 ≤5%. Ngưỡng distortion là lựa chọn vận hành bảo thủ cho bước
search; không phải con số được khẳng định trong ý tưởng gốc hay ngưỡng
paper-quality.

## Kết quả toàn cohort

| Cấu hình | Top-1 | ΔNLL | KL mean | Attn error mean | Attn error P99 | KV bits/component | Compression lý thuyết | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| K16V16 | 100.000% | 0.000% | 0 | 0.000% | 0.000% | 16 | 1.00× | Qua |
| K16V8 | 98.875% | +0.556% | 0.001543 | 2.812% | 6.341% | 12 | 1.33× | Trượt |
| K16V4 | 97.508% | +2.348% | 0.005383 | 6.810% | 17.598% | 10 | 1.60× | Trượt |
| K8V16 | 99.196% | +0.334% | 0.001599 | 2.912% | 6.555% | 12 | 1.33× | Trượt |
| K4V16 | 95.257% | +9.923% | 0.027867 | 11.223% | 30.185% | 10 | 1.60× | Trượt |
| K8V8 | 98.875% | +0.024% | 0.001616 | 2.906% | 6.367% | 8 | 2.00× | Trượt |

Compression là tỷ lệ bit lý thuyết của source K+V so với BF16, chưa tính scale,
alignment, metadata hay chi phí dequantization. K8V8 là 2× cho source KV; phần
generated cache vẫn BF16 nên đây không phải 2× toàn bộ KV.

### Theo dataset

K8V8:

| Dataset | Số mẫu | Target tokens | Top-1 | ΔNLL | Attn error mean | Attn error P99 |
|---|---:|---:|---:|---:|---:|---:|
| GovReport | 20 | 626 | 98.243% | −0.225% | 3.000% | 6.521% |
| Multi-News | 20 | 618 | 99.515% | +0.259% | 2.810% | 6.189% |

Tín hiệu Top-1 yếu hơn trên GovReport; ΔNLL nhỏ ở cả hai dataset. Việc ΔNLL
âm trên GovReport không đồng nghĩa candidate “tốt hơn” nói chung: đây là
teacher-forced likelihood trên một cohort nhỏ, trong khi Top-1 và distortion
vẫn không qua gate.

## Diễn giải

- V-only INT8 (K16V8) và K-only INT8 (K8V16) giữ ΔNLL dưới 1%; chỉ K8V16
  đồng thời vượt ngưỡng Top-1 99% sau khi khôi phục đủ mẫu. Cả hai đều không
  đạt ngưỡng attention-output distortion.
- Ghép K8V8 giữ ΔNLL gần 0 nhưng Top-1 tụt xuống 98.875%; hai phép lượng tử
  hóa cùng lúc không thể được quảng bá chỉ dựa trên NLL.
- INT4 uniform bị loại trong cohort này: K16V4 có ΔNLL +2.35%; K4V16 có
  ΔNLL +9.92% và Top-1 95.26%.
- Chưa có evidence về free-running generation, ROUGE, memory bandwidth,
  TPOT, throughput hoặc packed-cache capacity. Fake quantization vẫn giữ
  tensor trong BF16 sau dequantization.

Dense replay khớp token dự đoán E43 ở `98.553%` trên lượt K8V8. Do đó các
candidate metrics trong bảng được tính so với dense BF16 replay mới trên
cùng teacher-forced prefixes, không so trực tiếp với logits cũ từ E43.

## Tính đầy đủ và sự cố chạy

Lượt one-sided ban đầu hoàn tất 39/40; một mẫu GovReport dài 11,962 token gặp
CUDA OOM khi chạy ma trận nhiều candidate. Mẫu này được chạy lại riêng thành
công trên cùng A10, không đổi input hay cấu hình. Metrics trong bảng được
tính bằng 39 row ban đầu cộng row replay thay cho row lỗi; denominator cuối là
đủ 40 mẫu/1,244 token. Lượt K8V8 chạy riêng hoàn tất 40/40, không có sample
error. OOM không tái hiện ở lượt retry hoặc lượt K8V8; xem đây là peak tài
nguyên nhất thời, nhưng không đủ dữ liệu để kết luận nguyên nhân sâu hơn.

## Artifact và khả năng tái lập

- E44 one-sided: Modal volume `fast-infer-text-sum-cache`, run
  `e44-full-one-sided-qwen06b-a10g-20260923`; row retry:
  `e44-oom-replay2-qwen06b-a10g-20260923`.
- E44 joint K8V8: run `e44-full-k8v8-qwen06b-a10g-20260923`.
- Trace tham chiếu E43:
  `outputs/recap_kv_e43/e43-temporal-pilot-qwen06b-a10g-20260923/temporal_trace.jsonl`.
- Dataset SHA256 được ghi trong manifest; GovReport
  `3dbba6ee4874e2a00b9d0c706524ea2dfb47a60169491a54464b686189299813`,
  Multi-News
  `0bc56649bac6b80b622f11e655ad1887316242523c12e45e219d72bb01557d6a`.
- Runner và phép lượng tử hóa: `src/TrainingFree/fidelity_run.py`,
  `src/TrainingFree/fidelity.py`; launcher Modal:
  `scripts/modal_trainingfree_fidelity.py`.
- Kiểm tra CPU hiện tại: 13 test FidelityKV pass. Parser sample-ID được bổ
  sung sau khi targeted replay bắt lỗi chuỗi ID bị duyệt theo từng ký tự.

## Quyết định

E44 không đạt điều kiện mở physical prototype. Kết quả giữ lại một tín hiệu
đủ tốt để thử phân bổ fidelity không đồng nhất, nhưng mixed candidate chỉ
được promotion nếu replay trên cohort 40 mẫu đạt lại gate. Phạm vi E45/E46
đang chờ xác nhận lựa chọn staged scan hay full KV-head sweep; E47 và physical
kernel chưa chạy.
