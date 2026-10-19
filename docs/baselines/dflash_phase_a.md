# DFlash Phase A trên native FA4

Phase A chọn một tập context nhỏ hơn cho **draft model** của DFlash. Target
vẫn xác thực trên toàn bộ context. Bản adaptive hiện chỉ chạy trên pipeline
native Transformers + FlashAttention-4; không thay đổi Phase A trên vLLM.

Selector giữ các mặc định của Phase A: Query ở target layer 18, M1 max-pooling,
chunk 128 token, 16 sink token và 32 recent token. Budget mặc định 2048 vẫn là
chế độ đối chứng cố định. Chế độ adaptive khởi động ở 2048 rồi tìm budget theo
các bước 128 token trong khoảng 512–8192.

## Cách điều khiển budget

Controller `AdaptiveBudgetController` tối ưu surrogate
`round wall-ms / committed-token`. Mỗi action được đo là thời gian từ khi
controller chọn budget và dựng context cho draft kế tiếp đến lúc round draft +
target verify kết thúc. Số token commit gồm parent/current token và các draft
token được chấp nhận, có chặn theo output limit.

Controller cần 3 round đo được ở cùng một budget trước khi đánh giá action đó.
Round sinh đầu tiên được ghi là bootstrap và không đưa vào thống kê vì nó còn
chịu phần prefill. Sau đó controller thử lân cận theo hướng giảm budget trước;
nếu một mức tốt hơn incumbent ít nhất 3%, controller nhận mức đó và tăng bước
tìm kiếm (tối đa 1024). Nếu không cải thiện, controller đảo chiều và giảm bước,
tối thiểu 128. Khi cả hai phía ở bước nhỏ nhất đều không thắng, controller giữ
incumbent và kiểm tra lại sau mỗi 12 round.

Đây là tìm kiếm online theo request, không dùng prior đã calibration và không
ngoại suy kết quả cho budget chưa chạy. Request ngắn có thể kết thúc trước khi
controller học đủ; budget trajectory và từng round được ghi vào JSONL để kiểm
tra điều đó. Wall time của toàn generation vẫn là thước đo cuối, vì nó tính cả
bootstrap, thử action, controller và tail.

## Cấu hình

| Biến môi trường | Mặc định | Ý nghĩa |
|---|---:|---|
| `DFLASH_PHASE_A_BUDGET_MODE` | `fixed` | `fixed` hoặc `adaptive` |
| `DFLASH_PHASE_A_BUDGET` | `2048` | Budget ban đầu và budget cố định |
| `DFLASH_PHASE_A_MIN_BUDGET` | `512` | Cận dưới adaptive |
| `DFLASH_PHASE_A_MAX_BUDGET` | `8192` | Cận trên adaptive |
| `DFLASH_PHASE_A_CHUNK_SIZE` | `128` | Độ lớn bước và chunk selector |
| `DFLASH_PHASE_A_WINDOW_ROUNDS` | `3` | Số round ở một budget trước khi đánh giá |
| `DFLASH_PHASE_A_MIN_GAIN` | `0.03` | Mức cải thiện tối thiểu để nhận budget mới |
| `DFLASH_PHASE_A_DECAY` | `0.9` | Hệ số giảm trọng số thống kê cũ mỗi round |
| `DFLASH_PHASE_A_RECHECK_ROUNDS` | `12` | Chu kỳ thử lại khi đã giữ incumbent |
| `DFLASH_PHASE_A_LAYER` | `18` | Target layer dùng lấy Query |
| `DFLASH_PHASE_A_SINK` | `16` | Số sink token giữ lại |
| `DFLASH_PHASE_A_RECENT` | `32` | Số recent token giữ lại |

## Chạy full trên server B200

Launcher sẽ lấy model, data, Python và GPU từ master config như các benchmark
FA4 khác. Lệnh sau chạy đối chứng vanilla + DFlash dense + Phase A adaptive:

```bash
DFLASH_PHASE_A_BUDGET_MODE=adaptive \
DFLASH_PHASE_A_BUDGET=2048 \
DFLASH_PHASE_A_MIN_BUDGET=512 \
DFLASH_PHASE_A_MAX_BUDGET=8192 \
FI_GPU_IDS=0 \
bash scripts/run.sh fa4_native --full \
  --methods vanilla_hf,dflash,dflash_phase_a
```

Để chạy đối chứng fixed với cùng cấu hình dữ liệu/GPU, thay mode bằng
`DFLASH_PHASE_A_BUDGET_MODE=fixed`. Có thể truyền biến môi trường trong lệnh
hoặc khai báo chúng trong master config. Trước full run, kiểm tra runtime và
đường dẫn model bằng:

```bash
bash scripts/run.sh fa4_native --preflight-only \
  --methods vanilla_hf,dflash,dflash_phase_a
```

Kết quả nằm dưới `outputs/fa4_native_benchmark/<run-id>/`. Record của
`dflash_phase_a` có `phase_a_budget_trajectory`, `phase_a_round_trace` và
`phase_a_controller`; mỗi round ghi budget dùng, budget kế tiếp, context token
được chọn, commit, wall time, CUDA draft/verify time, quyết định và lý do
fallback nếu có. Summary ghi policy cùng các cận đã cấu hình.

## Đọc kết quả

So sánh `dflash_phase_a` adaptive với cả `dflash_phase_a` fixed và `dflash`
dense trên cùng input. Dùng end-to-end latency/throughput làm kết luận speedup;
`round_ms / committed_tokens` chỉ là tín hiệu điều khiển. Đồng thời kiểm tra
greedy parity và quality gate. Controller không đảm bảo speedup: nếu exploration
và selector overhead lớn hơn phần draft attention tiết kiệm được, full-run có
thể không nhanh hơn.
