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
- speed profile không dừng EOS: 32 token ở smoke, 1.024 token ở full;
- Vanilla HF chạy trước và tạo `references/<dataset>.jsonl`;
- mỗi baseline đọc sidecar ngay trong run và ghi `dense_*`, `esr`, `dsr`,
  `speedup_valid` cùng raw timing.

Artifact được ghi vào `outputs/Benchmark_results/qwen3_4b_paired_online/<run-id>/`.
`run_manifest.json` lưu input distribution và command đầy đủ; các JSONL của
từng baseline là nguồn để tạo bảng kết quả chính.
