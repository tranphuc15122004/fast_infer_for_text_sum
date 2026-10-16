# Đánh giá đồng bộ các baseline bằng vLLM (Synchronized vLLM Evaluation)

`scripts/run.sh vllm_all` chạy vanilla, EAGLE3, DFlash, Domino và DSpark lần
lượt trên cùng target model, cùng prompt token IDs, cùng tham số greedy và cùng
phiên bản vLLM. Mỗi lần chỉ nạp một draft model để tránh chiếm VRAM đồng thời.
Runner dùng checkpoint local trong master config và bật chế độ offline; nó không
tải model từ Hub.

## Chạy trên server

```bash
# Preflight: kiểm tra package, GPU, target/tokenizer và config các draft.
bash scripts/run.sh vllm_all --preflight-only

# Smoke: 2 mẫu chung để kiểm tra runtime và output schema.
bash scripts/run.sh vllm_all --smoke

# Full throughput: batching tối đa 16 request; --full ghi đè RUN_MODE/SMOKE.
bash scripts/run.sh vllm_all --full --measurement throughput

# Latency/speedup speculative batch 1, chạy riêng trên GPU không tranh chấp.
bash scripts/run.sh vllm_all --full --measurement latency
```

Mặc định config được resolve qua `config/master.path`. Có thể override một lần:

```bash
bash scripts/run.sh vllm_all \
  --config /path/to/fast_infer_master.env \
  --max-new-tokens 512
```

Các biến `VLLM_*` có thể override model, data, output, phương pháp, dtype,
context limit và mức dùng VRAM. Model/data mặc định lấy từ master config:

| Biến | Master config mặc định |
|---|---|
| `VLLM_TARGET_MODEL` | `MODEL_TARGET` |
| `VLLM_EAGLE3_MODEL` | `MODEL_EAGLE_DRAFT` |
| `VLLM_DFLASH_MODEL` | `MODEL_DFLASH_DRAFT` |
| `VLLM_DOMINO_MODEL` | `MODEL_DOMINO_DRAFT` |
| `VLLM_DSPARK_MODEL` | `MODEL_DSPARK_DRAFT` |
| `VLLM_DATA_FILE` | `LONG_BENCH_DATA_FILE`, rồi `DATA_INPUT` |

Mặc định runner dùng tối đa 8.192 input tokens, 12.288 tổng context tokens,
512 output tokens, BF16 và greedy decode. Mặc định `throughput` gửi các nhóm
tối đa 16 prompt vào một lần `LLM.generate`, để scheduler vLLM thực hiện batching.
`latency` luôn đặt `max_num_seqs=1`. Warmup tối đa ba prompt đại diện theo độ dài,
mỗi prompt sinh tối đa 64 token, một lần sau khi nạp engine; không warmup toàn corpus.
Warmup hữu hạn có thể chưa bao phủ mọi shape/kernel; report giữ cấu hình để đọc đúng phép đo. Các mẫu vượt giới hạn input chung bị
bỏ khỏi toàn bộ method và được ghi vào manifest; mọi method còn lại nhận cùng
sample IDs và token IDs. Nên chạy khi GPU không có workload khác để tránh nhiễu timing.
`--max-samples N` giới hạn số mẫu đầu tiên đủ điều kiện; mặc định `0` nghĩa là chạy hết.
`--preflight-only` xác nhận model path, Python package, CUDA và GPU trước khi bắt đầu full run.

Các biến điều khiển execution (CLI cuối lệnh có ưu tiên):

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `VLLM_MEASUREMENT` | `throughput` | `latency` hoặc `throughput` |
| `VLLM_MAX_NUM_SEQS` | `16` | Số request tối đa mỗi nhóm throughput; latency luôn 1 |
| `VLLM_MAX_NUM_BATCHED_TOKENS` | vLLM tự chọn | Token budget scheduler; không giảm context/output |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.85` | Ngân sách cấp phát engine |
| `VLLM_WARMUP_PROMPTS` | `3` | Giới hạn prompt warmup, đặt 0 để tắt |
| `VLLM_WARMUP_TOKENS` | `64` | Output token tối đa mỗi warmup |
| `VLLM_DIAGNOSTICS` | `report-only` | `strict` làm quality/parity diagnostic ảnh hưởng exit code |

Backend dùng API công khai của vLLM 0.30.0 trong `requirements.txt`, cùng
PyTorch 2.13.0 và transformers 5.12.1. Kiểm thử CPU engine giả không xác nhận
CUDA/kernel hoặc GPU parity. Runner này phục vụ các baseline native; các arm
can thiệp attention/KV nghiên cứu cần adapter riêng, không được đổi tên DFlash
chuẩn thành arm nghiên cứu.

## Artifact và tiêu chí đọc kết quả

Mỗi lần chạy tạo thư mục riêng dưới `outputs/vllm_unified/<run-id>/`:

- `console.log`: toàn bộ stdout/stderr của launcher và evaluator, bao gồm stack trace nếu lỗi.
- `results.jsonl`: record từng method/sample, token IDs và text, timing, metric chất lượng, parity, bộ nhớ GPU, RequestOutput/metrics thô của vLLM và speculative metrics thô.
- `samples.jsonl`: hàng dữ liệu đầu vào gốc, prompt đã render, prompt token IDs, reference và số từ.
- `excluded_samples.jsonl`: hàng gốc và lý do mẫu bị loại.
- `warmup.jsonl`: output/token IDs, RequestOutput thô, thời gian và bộ nhớ của từng request warmup.
- `events.jsonl`: timeline JSONL có UTC timestamp và elapsed time; ghi config nạp model/generation, warmup, request, GPU snapshot, lỗi, tổng hợp và hoàn tất.
- `run_report.json`: config đầy đủ, config target/draft, SamplingParams, phiên bản thư viện, hash dữ liệu, tokenizer, thông tin môi trường và metric tổng hợp.
- `report_vi.md`: bảng TPOT, DSR, ESR, ROUGE-L, repetition và token overlap.
- `progress.json`: method/sample đang chạy và số record đã lưu, cập nhật khi request bắt đầu/kết thúc.
- `results.partial.jsonl`: checkpoint theo từng request; được xóa sau khi `results.jsonl` hoàn tất, nhưng giữ lại nếu tiến trình bị ngắt.

Chế độ throughput báo `output_throughput_tok_s`, `request_throughput_qps` từ tổng
output token/request hoàn tất chia tổng makespan các nhóm đo, không gồm load/warmup.
Latency p50/p95 lấy timestamp của từng request; không chia wall time nhóm cho số
request. `peak_active_requests_observed` lấy overlap timestamp khi có;
`submitted_concurrency` chỉ là số request gửi chung, không tự coi là concurrency
thực tế. Các nhóm cuối có thể nhỏ hơn giới hạn.

DSR/ESR chỉ được tính ở chế độ latency khi cả method và vanilla có timing hợp lệ trên cùng tập sample IDs. `correctness_pass=false` báo có method không khớp greedy vanilla; khi đó không diễn giải speedup của method ấy như hiệu năng đã xác nhận. Speedup dưới 1 vẫn được báo cáo nếu đầu ra hợp lệ và phép đo hoàn tất. Mặc định `report-only`, quality/parity/timing diagnostic không chặn campaign hay
làm exit code khác 0. Record vẫn giữ `status=invalid` nếu phép kiểm chứng không
đạt; `execution_status` cho biết engine đã hoàn tất generation hay thất bại.
`--diagnostics strict` làm quality/parity diagnostic ảnh hưởng exit code.
Lỗi execution/schema vẫn trả mã lỗi; artifact được giữ. Batch lỗi được ghi thất
bại cho các request bị ảnh hưởng rồi tiếp tục nhóm tiếp theo; runner native chưa
tự restart engine hoặc giảm concurrency khi CUDA/OOM. Khi gặp lỗi loại này,
chạy lại với concurrency thấp hơn và đo lại cùng cấu hình cho mọi method.
