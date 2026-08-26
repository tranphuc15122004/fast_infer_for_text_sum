# Qwen3-4B full-inference profiling

Thư mục này chứa source profiler và artifact của thực nghiệm baseline Qwen3-4B
không dùng speculative decoding trên Tesla T4 16 GB.

- Code: `profile_qwen3_long_summary.py`
- Raw measurements: `results/measurements.jsonl`
- Median summary: `results/summary.csv` và `results/summary.jsonl`
- Runtime metadata: `results/metadata.json`
- Hình phân tích: các file `results/*.png`

Chạy lại từ root repo:

```bash
bash scripts/run_qwen3_long_profile.sh config/qwen3_long_profile.env
```

Config mặc định dùng snapshot local của `Qwen/Qwen3-4B`, các mốc 256, 512,
1024, 2048 và 3072 từ `govreport_representative.jsonl`, tối đa 128 token đầu
ra, ba lần đo sau một lần warm-up.

Lưu ý: `kv_cache_first_read_ms` là thời gian forward của token decode đầu tiên
có đọc KV cache hiện có; Transformers không phát ra một event độc lập cho
"KV-cache load". `model_load_ms` là chi phí một lần và được ghi riêng trong
`metadata.json`, không cộng vào tỷ lệ phase của từng sample.
