# Thực nghiệm Qwen3-4B paired online

Launcher ngắn gọn:

```bash
bash scripts/runners/run_qwen3_4b_paired.sh \
  --mode smoke \
  --target-model /models/Qwen3-4B \
  --dflash-model /models/Qwen3-4B-DFlash \
  --domino-model /models/Qwen3-4B-Domino-b16 \
  --eagle-model /models/Qwen3-4B-EAGLE3 \
  --gpu-id 0
```

Chạy đầy đủ 100 mẫu trên mỗi dataset:

```bash
bash scripts/runners/run_qwen3_4b_paired.sh --mode full \
  --target-model /models/Qwen3-4B \
  --dflash-model /models/Qwen3-4B-DFlash \
  --domino-model /models/Qwen3-4B-Domino-b16 \
  --eagle-model /models/Qwen3-4B-EAGLE3 \
  --gpu-id 0
```

Mặc định launcher chạy `gov_report`, `qmsum`, `multi_news`, `lcc` và
`repobench-p`; baseline gồm `vanilla_hf`, `vanilla_fa`, `dflash`, `domino`
và `eagle3`. Dùng `--dry-run` để kiểm tra toàn bộ command mà không load model.

Contract cố định:

- batch size = 1;
- cùng file JSONL, sample ID và prompt tokenized;
- input cap = 14.000 token, truncation head+tail deterministic;
- chỉ có một lượt generation fixed-K: không dừng tại EOS, 32 token ở smoke,
  1.024 token ở full;
- Vanilla HF chạy trước và tạo `references/<dataset>.jsonl`;
- mỗi baseline đọc sidecar ngay trong run và ghi `dense_*`, `esr`, `dsr`,
  `speedup_valid` cùng raw timing.
- token IDs đầy đủ, EOS đầu tiên, `quality_text` (chỉ phần trước EOS) và
  `full_output_text` đều được lưu; vì vậy metric chất lượng có thể tính offline
  mà không chạy model thêm lần nữa.
- chuỗi output giữa các baseline không bắt buộc giống nhau. ESR/DSR chỉ hợp lệ
  khi ghép đúng input/config và cùng K token; output content được dùng cho
  đánh giá chất lượng riêng, không phải điều kiện ghép timing.
- record đồng thời ghi exact-match/token-match của continuation và phần output
  trước EOS so với Vanilla HF; đây là correctness/quality audit, không phải
  điều kiện để speedup pair hợp lệ.
- runner kiểm tra batch=1, đúng số mẫu, đúng fixed-K, raw token IDs và ESR/DSR
  hợp lệ; sau đó tự tạo `metrics_summary.json`, `.csv`, `.md`.
- `decode tok/s` trong bảng chính được tính tổng output token chia tổng decode
  time (weighted theo thời gian); mean/p50/p90 tốc độ từng mẫu vẫn được lưu
  trong JSON/CSV để xem phân bố.

Artifact được ghi vào `outputs/Benchmark_results/qwen3_4b_paired_online/<run-id>/`.
`run_manifest.json` lưu hash dữ liệu, cấu hình/hash từng dataset, model paths,
runtime, command và trạng thái kiểm tra. `inputs/source/` giữ nguyên các mẫu
đã chọn cùng reference để có thể tái tổng hợp độc lập với thư mục data nguồn.
JSONL từng baseline lưu timing, raw
continuation, acceptance trace, model/tokenizer revision cùng online ESR/DSR.
Sidecar trong `references/` chỉ phục vụ ghép pair và không bị collector đếm như
một lần chạy Vanilla HF thứ hai.

Không cần chạy lại generation để cập nhật bảng hay tính ROUGE/BLEU/τ từ các
record đã lưu. Nếu một pair ESR/DSR không hợp lệ, runner đánh dấu run thất bại;
không thay bằng phép ghép offline giữa output khác.
