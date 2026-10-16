# Ma trận nghiên cứu DFlash trên vLLM

`vllm_research` nối planner A1–A3/B0–B4 với một runtime Python chung trên
server. Target forward, KV paging, speculative verification và batching thuộc
vLLM; adapter chỉ thay đổi đường đọc KV của drafter hoặc độ dài proposal khi
đã có worker hook tương ứng. Runtime khóa theo `requirements.txt`: Python
3.12, vLLM 0.30.0, torch 2.13.0 và transformers 5.12.1. Không cần cài thêm
package hoặc tải model từ Internet.

## Chạy campaign

```bash
# Xem trước toàn ma trận và khả năng của vLLM cài trên server; không nạp model.
bash scripts/run.sh vllm_research --plan-only --experiments ALL

# Chạy mọi mẫu, một lượt đo, latency batch 1 (mặc định).
bash scripts/run.sh vllm_research --full --experiments ALL

# Chạy phần E2E theo continuous batching; các policy bắt buộc batch 1 sẽ được
# ghi unsupported, còn baseline/A3 tiếp tục.
bash scripts/run.sh vllm_research --full --experiments E2E --measurement throughput \
  --max-num-seqs 16

# Chọn lưới A3/đổi budget; đây là các giá trị thực nghiệm, không giảm input/output.
bash scripts/run.sh vllm_research --full --experiments A3 \
  --selectors previous_dflash,recency --budgets 1024,2048,4096 \
  --periods 2,4,8,16 --measurement latency

# Tiếp tục campaign bị ngắt; cell đã hoàn tất được giữ nguyên.
bash scripts/run.sh vllm_research --resume 20261015T120000Z-1234
```

Model và dữ liệu lấy từ cùng master config của các runner vLLM:
`MODEL_TARGET`, `MODEL_DFLASH_DRAFT` và `DATA_INPUT`/`LONG_BENCH_DATA_FILE`.
Có thể override bằng `--target-model`, `--draft-model`, `--data-file` hoặc
các biến `VLLM_*`. Không có venv riêng. `--full` chỉ bỏ sample cap; grid
method vẫn do `--experiments`, `--selectors`, `--budgets`, `--periods` và
`--gammas` xác định. Mặc định smoke giới hạn hai mẫu, một repetition, một
budget A3 và một warmup tối đa ba prompt × 64 token cho mỗi engine.

`latency` là phép đo batch 1 và mặc định để B3 variable-k chạy đúng chế độ.
`throughput` dùng `LLM.generate` với continuous batching, tối đa 16 request
mặc định. Engine giữ sống qua toàn bộ corpus của một cấu hình; các cấu hình
khác nhau vẫn có engine riêng để bootstrap hook đúng từ đầu. Warmup hữu hạn
không chạy lại trên từng sample. Không có gate ROUGE/parity/acceptance để dừng
job hoặc campaign; record quality và execution được báo riêng.

## Capability hiện thực

Worker tùy biến khóa theo source vLLM 0.30.0 và yêu cầu Qwen3 target + DFlash,
TP=1, eager, model runner MRv1, `FLASH_ATTN`, KV không lượng tử hóa, attention
không sliding-window/sink. A3 dùng vị trí token nguồn lấy từ offset tokenizer
trên prompt gốc. B3 fixed-k draft 15 rồi cắt proposal trước native verification;
B4 adapter chọn gamma trước khi native DFlash tạo query. B3/B4 adaptive chưa
được nối fit artifact/cost profile từ planner vào worker và được ghi
`unsupported`; fixed-k/fixed-gamma vẫn chạy. Backend khác không được đổi tên
thành một arm tương đương.

Stock vLLM 0.30.0 không có giao dịch fork/rollback đồng thời target KV,
DFlash KV, partial tail page, request metadata, RNG và scheduler state. Vì vậy
A1/A2, A3 same-prefix replay, B1 gamma sweep và B1-V chưa được giả làm bằng
các lần `generate` độc lập; planner ghi `unsupported` hoặc
`dependency_missing` kèm nguyên nhân và tiếp tục các arm độc lập. Kết quả
`gpu_available`/`worker_started` chỉ được ghi sau khi engine khởi tạo trên
CUDA. `semantic_parity_verified` vẫn false cho tới khi có lượt xác minh riêng;
khởi tạo engine không tự chứng minh parity. Lệnh full luôn finalize coverage của toàn ma trận; trạng
thái campaign `finished` không đồng nghĩa mọi cell đã chạy thành công.
B0 full-generation trace ghép proposal/verification liền kề để tạo candidate
signals cho các fit B2 dùng tín hiệu drafter. Các biến thể cần parent target
logits được ghi thiếu evidence; việc này không thay gamma sweep hoặc
same-prefix replay.

## Artifact

Mỗi campaign nằm dưới `outputs/vllm_research/<run-id>/`:

- `plan.json`: cell/config dự kiến, input/model settings, capability và hash.
- `prompt_layouts.json`: offset nguồn đã token hóa, dùng chung giữa worker.
- `jobs/<job-id>/cells/*.json`: checkpoint từng sample/repetition để resume.
- `jobs/<job-id>/results.jsonl`: record generation và summary của job.
- `jobs/<job-id>/traces/`: proposal/verification trace của adapter.
- `results.jsonl`: terminal record từng job và summary toàn campaign.
- `report.json`, `runtime.json`, `progress.json`, `excluded_samples.json`,
  `console.log`: coverage, phiên bản runtime, tiến độ, input loại và log.

`unsupported` chỉ ra capability code không có; `dependency_missing` chỉ ra
evidence/model fit prerequisite chưa thành công; `failed` là lỗi thực thi.
Chất lượng đầu ra không quyết định trạng thái execution. Nếu CUDA OOM hoặc
engine lỗi, các cell đã lưu được giữ lại; bản hiện tại chưa tự restart engine
hoặc tự hạ concurrency.
