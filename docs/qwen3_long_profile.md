# Profile Qwen3-4B cho tóm tắt dài

Profiler này đo **Qwen3-4B target đơn**, không dùng EAGLE-3/DFlash hay
speculative decoding. Mặc định chạy trên một GPU CUDA, FP16 + SDPA, batch size
1, với các mốc 256/512/1024/2048/3072 từ và tối đa 128 token output.

## Chạy

```bash
bash scripts/run_qwen3_long_profile.sh config/qwen3_long_profile.env
```

Source canonical nằm ở `src/analyze/full_infer/profile_qwen3_long_summary.py`.
Artifact của lần chạy Qwen3-4B/Tesla T4 được lưu ở
`src/analyze/full_infer/results/`:

- `measurements.jsonl`: từng repeat;
- `summary.csv` / `summary.jsonl`: median của 3 repeats mỗi mốc;
- `metadata.json`: model load, protocol và định nghĩa phase;
- `phase_time_stacked.png`: thời gian tuyệt đối theo phase;
- `phase_share_100pct.png`: tỷ lệ phần trăm mỗi phase;
- `phase_time_by_length.png`: xu hướng prefill/decode theo độ dài;
- `memory_by_length.png`: peak VRAM và kích thước KV cache;
- `model_load_vs_sample_total.png`: model load one-time so với một sample.

## Ý nghĩa phép đo

`prefill_ms` là forward đầu tiên trên toàn input, bao gồm tạo và ghi KV cache.
Transformers không expose một event riêng cho “KV-cache load”; vì vậy
`kv_cache_first_read_ms` là forward decode một token đầu tiên đọc cache đã có,
còn `decode_rest_ms` là các forward decode còn lại. Tỷ lệ trong chart loại
`model_load_ms` khỏi per-sample total vì model chỉ load một lần; model load được
báo riêng trong `metadata.json` và chart amortization.

Đây là profiler latency thực tế với EOS: nếu model kết thúc sớm, số output token
có thể nhỏ hơn 128. Khi so sánh riêng ảnh hưởng của input length, cần đọc thêm
`output_tokens`; khi so sánh workload cố định, nên bổ sung chế độ fixed-output
tokens trong một lượt thí nghiệm riêng.
