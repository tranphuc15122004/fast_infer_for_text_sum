# Khảo sát lại: tăng tốc speculative decoding cho summary ngắn

Ngày: 2026-09-28. Đây là đề xuất và gate thực nghiệm, chưa phải kết quả speedup của phương pháp mới.

## Sửa giả định từ các lần thử trước

Coverage prefill trên Modal không thắng Lead ở GovReport và chỉ ngang Lead ở QMSum; xem [báo cáo](2026-09-28_modal_coverage_prefill.md). Ở workload Qwen3-4B, output 128 token, decode chiếm phần lớn E2E; do đó một selector prefill mới có trần lợi ích nhỏ. SSSD đã được người dùng tái lập và không phải hướng ưu tiên. Các nhánh source routing, candidate repair và output stopping đã có negative evidence; không mở lại dưới tên khác.

Benchmark B200 hiện có cho DFlash dùng Llama 3.1 8B, block size 10 và `max_new_tokens=2048`. Từ 100 trace mỗi dataset trong `outputs/Benchmark_results/longbench_full/b200-full-5datasets/dflash/`, tính lại acceptance theo **vị trí đầu ra**. Mỗi hàng dưới đây chỉ lấy các round bắt đầu trước mốc token; round cắt qua mốc vẫn được tính nguyên. Đây là phân tích trace cũ, không phải lần chạy 128/256 token mới.

| Dataset | Mean acceptance, 128 token đầu | P(accept=1), 128 đầu | Mean acceptance, cả trace 2048 |
|---|---:|---:|---:|
| GovReport | 2.10 | 0.482 | 4.56 |
| MultiNews | 2.95 | 0.320 | 5.18 |
| QMSum | 1.84 | 0.517 | 5.58 |

Vì nhiều output chạm trần 2048, mean acceptance toàn trace dễ làm ta đánh giá quá cao hiệu quả DFlash cho summary 128–256 token. Không suy ra nguyên nhân (ví dụ lặp lại cuối output) chỉ từ acceptance. Cần kiểm tra text và chạy mới ở output budget phù hợp. Bảng B200 hiện tại là provisional; E2E DFlash mạnh hơn Vanilla FA ở toàn trace nhưng không chứng minh cùng mức speedup trên summary ngắn. Reference Modal 128 token cho DFlash 1.92× GovReport và 2.04× MultiNews so với **Vanilla HF**, không phải Vanilla FA; xem [reference](2026-09-21_modal_trainingfree_reference.md).

## Đề xuất chính: thích nghi drafter DFlash vào 256 token summary đầu

Giữ target và verifier exact. Fine-tune **chỉ drafter DFlash** bằng teacher trajectories của chính target trên GovReport, MultiNews, QMSum. Dùng prompt/source distribution, độ dài source và output budget giống đánh giá. Giới hạn training trajectories ở 256 hoặc 512 output token để không tối ưu cho phần output dài ít liên quan. Baseline đầu tiên là DFlash pretrained; control bắt buộc thứ hai là DFlash fine-tune chuẩn trên **cùng** trajectories. Biến thể nghiên cứu chỉ khác phân bổ anchor/loss: ưu tiên response positions 0–256 và first-failure positions trong draft block, với trọng số chuẩn hóa để tổng loss không tự tăng. Chọn trọng số trên dev; test holdout khóa trước. Nếu fine-tune chuẩn đã giải quyết phần lớn vấn đề, báo đó là kết quả domain adaptation, không gán gain cho loss mới.

Lý do chọn: QMSum và GovReport trong 128 token đầu chỉ nhận trung bình khoảng 1.8–2.1 token/round dù verifier xử lý block 10. Tăng acceptance ở đoạn này có thể giảm số target verification rounds, là phần chi phí liên quan trực tiếp tới E2E. Training drafter dùng lại target-hidden feature cache, objective và inference contract trong `src/Finetuning/` hoặc `src/MR_DFlash/`; không train target. Cần đo chi phí capture/train và kích thước checkpoint. Không giả định LoRA đủ tốt: trước tiên đo một pilot với adapter nhỏ; chỉ dùng full draft training nếu pilot và chi phí cho phép.

Liên hệ tài liệu: [DFlash](https://arxiv.org/abs/2602.06036) cho thấy parallel drafting và exact verification; [domain draft training](https://arxiv.org/abs/2503.07807) ủng hộ distillation trên distribution thực; [AdaSPEC](https://arxiv.org/abs/2510.19779) chỉ ra loss tổng quát có thể lệch khỏi objective acceptance; [SpecBlock](https://arxiv.org/abs/2605.07243) dùng valid-prefix training để tránh dạy các suffix không còn đúng sau mismatch. Đây là tổ hợp áp dụng cho summary đầu ra ngắn, không phải tuyên bố thuật toán gốc chưa từng có.

## Gate training-free trước khi tốn GPU train

Chạy DFlash checkpoint gốc ở `k ∈ {2,4,6,8,10}` trên **cùng** 30 document/dataset, `max_new_tokens=128` và 256, cùng GPU/runtime/prompt. DFlash draft phụ thuộc số noise positions nên không thể dùng trace k=10 để kết luận counterfactual k=4. Đo CUDA event draft/verify, TTFT, decode, E2E, output length, exact token equality với target greedy và ROUGE. Chọn best static k trên dev, không trên test. Đây là baseline mạnh và có thể tự mang lại gain không cần train.

Chỉ thử dynamic k nếu sweep cho thấy verify k ngắn rẻ hơn đáng kể và các trạng thái dự đoán acceptance hữu ích ở **128/256 token đầu**. Trace 2048 cho thấy acceptance regime bền theo round, nhưng ở 128 token đầu rất ít round có acceptance ≥8; vì thế không dùng toàn trace để biện minh cho controller. Nếu thử, chọn k bằng previous acceptance và output position, so với best static k, và giữ verifier exact. [DISCO](https://arxiv.org/abs/2405.04304), [SVIP](https://arxiv.org/abs/2411.18462) và [DSpark](https://arxiv.org/abs/2607.05147) đã nghiên cứu adaptive lookahead/verification; đóng góp ở đây chỉ là chi phí và tín hiệu đo được cho DFlash summarization cụ thể.

## Protocol quyết định

1. Khóa target/draft snapshot, prompt, tokenizer, GPU, attention backend, dataset split theo document, seed và decoding greedy. Dùng 128/256 token làm primary; 512 token là secondary. Báo EOS và output-length distribution; nếu đa số chạm max, quality cần được diễn giải cùng độ dài.
2. So sánh Vanilla FA, DFlash pretrained best static k, DFlash fine-tune chuẩn best static k, và biến thể response-position/first-failure best static k. So với HF riêng, không trộn denominator.
3. Gate triển khai: exact token equality 100% ở greedy; paired E2E nhanh hơn control DFlash **ít nhất 10%**, với 95% bootstrap CI theo document có cận dưới >1.00× trên ít nhất 2/3 dataset; không giảm ROUGE vì output exact. Báo thêm TTFT, TPOT, draft/verify time, memory và training cost. Nếu chỉ tăng acceptance mà E2E không qua gate, dừng.
4. Holdout tối thiểu 100 document/dataset sau khi chọn cấu hình trên dev; không dùng 100 trace B200 đã khảo sát làm holdout. Nếu model Llama không tương thích pipeline train hiện có, dùng Qwen3-4B + draft checkpoint phù hợp nhưng phải thu trace 128/256 mới trước khi áp dụng giả thuyết về early acceptance.

Rủi ro chính: early acceptance thấp có thể do target entropy cao, nên distillation không sửa được; k ngắn có thể tăng số round; draft training có thể tăng latency hoặc quá khớp prompt. Cả ba đều được phát hiện ở gate trên trước khi đưa vào benchmark lớn.
