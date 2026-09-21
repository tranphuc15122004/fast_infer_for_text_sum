# Thiết kế tăng tốc MR-DFlash bằng vLLM và SGLang

**Ngày:** 2026-09-20  
**Phạm vi:** regenerate target trajectories bằng vLLM 0.24.0 và hidden-state cache bằng SpecForge/SGLang 0.5.14.  
**Trạng thái:** đã được người dùng duyệt để triển khai.

## Mục tiêu

Giảm wall-clock của hai phase target inference trên server GPU 180 GiB bằng
cách dùng vLLM cho text generation và giữ SGLang/SpecForge cho hidden-state
capture. Output text, feature schema, layer IDs, resume và coverage phải tương
thích với pipeline MR-DFlash hiện tại.

## Bối cảnh và giả định

- Server có vLLM 0.24.0 và SGLang 0.5.14 đã cài sẵn.
- Model target fit trên một GPU 180 GiB; mặc định dùng một replica/GPU (TP=1,
  DP=N). Nếu model không fit, caller phải chạy server theo nhóm TP riêng.
- vLLM regenerate chạy qua OpenAI-compatible `/v1/completions` và truyền prompt
  dưới dạng token IDs để không render lại chat template ở server.
- Cache dùng `specforge_sglang`, không dùng vLLM để lấy hidden states trung gian.
- Không sửa `externals/SpecForge`; chỉ gọi boundary adapter hiện có.

## Thiết kế

### Regenerate bằng vLLM

Thêm worker `vllm_regenerate.py` dùng tokenizer local để chuẩn bị prompt,
truncate và tính generation budget. Worker gửi nhiều request đồng thời tới một
vLLM server được gán cho GPU/worker đó. vLLM chịu trách nhiệm continuous
batching; worker chỉ giới hạn số request và tổng token đang in-flight.

Request scheduler:

```text
admission_cost = prompt_tokens + generation_budget
submit khi in_flight_tokens + admission_cost <= max_batched_tokens
```

Worker dùng request concurrency bắt đầu nhỏ, tăng theo batch thành công và giảm
khi server trả lỗi 5xx/timeout/OOM. Với `temperature > 0`, giữ một request mỗi
lượt để không thay đổi sampling contract của pipeline hiện tại.

### Cache bằng SGLang

Giữ nguyên `cache_target_features.py` và `specforge_capture.py` hiện có:

- `--cache-backend specforge_sglang`;
- adaptive batch theo length bucket;
- static token pool với soft target 160 GiB và hard cap 163 GiB;
- async shard writer, shared lease và resume.

SGLang 0.5.14 phải được import từ wheel/patch tương thích SpecForge; không đưa
source SGLang khác vào `PYTHONPATH`.

### Multi-GPU

Pipeline truyền danh sách vLLM server addresses theo thứ tự GPU. Mỗi worker
parallel xử lý một server address và một lease riêng. Shared lease hiện có được
giữ lại; thêm giới hạn lease quantum để giảm thời gian GPU chờ khi độ dài sample
không đồng đều.

### Correctness contract

- Regenerate: giữ model revision, tokenizer, prompt token IDs, max length,
  generation budget, temperature và seed. Parity test so sánh output/token IDs
  trên fixture trước khi chạy full.
- Cache: giữ layer IDs, input IDs, loss mask, feature width/dtype và sample IDs.
  SGLang hidden states phải có shape đúng và nằm trong tolerance đã chọn so với
  HF fixture.
- Không claim speedup nếu parity hoặc coverage chưa pass.

## Validation

- L0: unit tests cho vLLM request payload, token admission, retry/backoff, CLI
  propagation và cache flag preservation.
- L1 server: smoke 20–100 samples với vLLM regenerate và SGLang cache.
- Multi-GPU: kiểm tra mỗi server address nhận đúng worker, không duplicate/missing
  sample và merge canonical.
- Performance: wall-clock, samples/s, tokens/s, peak VRAM, queue wait, OOM,
  I/O backlog.

## Ngoài phạm vi

- Không thay đổi trainer, MR-DFlash model/loss hoặc feature schema.
- Không triển khai vLLM hidden-state capture.
- Không tự động khởi động/dừng vLLM server; server farm phải được vận hành bên
  ngoài pipeline.
