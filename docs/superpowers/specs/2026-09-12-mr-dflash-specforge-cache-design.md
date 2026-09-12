# Thiết kế cache MR-DFlash theo cơ chế SpecForge

**Ngày:** 2026-09-12  
**Phạm vi:** phase cache target hidden states; không thay đổi trainer hoặc feature contract.  
**Trạng thái:** đã được người dùng duyệt; L0 đã triển khai, L1 B200 đang chờ chạy trên server.

## Mục tiêu

Thay implementation cache HF tuần tự/batch nhỏ hiện tại bằng offline SGLang
capture theo cơ chế SpecForge, tận dụng 4 GPU B200 180GB và dữ liệu thực tế
chủ yếu dưới 8K token. Cache phải nhanh hơn rõ rệt nhưng vẫn đọc được bởi
`ShardedDFlashFeatureDataset`, resume an toàn và kiểm tra đủ sample.

## Bối cảnh và giả định

- Target mặc định là Qwen3-4B local; đường dẫn model vẫn do CLI truyền vào.
- Một stage chạy data-parallel trên 4 GPU, mỗi GPU một target engine (`TP=1`).
- `feature_layer_ids=[1, 9, 17, 25, 33]` và feature concat theo mọi token hợp lệ
  được giữ nguyên; không chuyển sang suffix-only capture ngầm.
- `input_ids`, `loss_mask`, độ dài và thứ tự sample vẫn giữ nguyên.
- Không gọi network; SGLang, model và kernel phải có sẵn trên server.
- Dữ liệu được tokenize trước và cache đọc từ tokenized manifest để không render
  chat template lặp lại trên GPU worker.

## Giả thuyết throughput

Nếu target capture dùng offline SGLang/FlashInfer với request batching theo
`max_total_tokens`, thay vì HF backbone forward một batch cố định, cùng với
producer/consumer I/O bất đồng bộ, throughput cache sẽ tăng đáng kể trên B200
trong khi peak VRAM được giới hạn trước khi chạy full dataset.

## Thiết kế được chọn

### 1. SGLang capture adapter

Tạo adapter nội bộ gọi boundary `SpecForge` offline capture với
`capture_method="dflash"`. Adapter chỉ nhận batch tokenized đã pad, gọi target
engine một lần và chuyển `aux_hidden_states` thành tensor
`[sample, sequence, feature_width]` tương thích với MR-DFlash.

Không import test code và không sửa `externals/SpecForge`. Nếu dependency hoặc
capture hook không có trên server, chương trình dừng rõ ràng với hướng dẫn
preflight thay vì tự động rơi về HF chậm mà không báo.

### 2. Scheduler theo độ dài thực tế

Worker gom sample theo bucket độ dài và giới hạn đồng thời bằng tổng token,
không dùng một batch cố định cho mọi sample:

```text
1..4096       : tối đa 64 sample hoặc 262144 token
4097..8192    : tối đa 32 sample hoặc 262144 token
8193..16384   : tối đa 16 sample hoặc 262144 token
16385..32768  : tối đa 4 sample hoặc 131072 token
```

Các giá trị trên là candidate aggressive cho B200 180GB. Profiler được phép
thử từ mức này xuống các mức thấp hơn khi peak VRAM vượt giới hạn. Mặc định
profile dùng khoảng 98--99% VRAM khả dụng; không cố chừa 10--15% nếu GPU đã
được cấp riêng cho job. Request pending có thể cao, nhưng số token active trên
GPU luôn bị giới hạn bởi `max_total_tokens`, tránh lỗi kiểu 128 sequence × 32K.

Do dữ liệu hầu hết dưới 8K, hai bucket đầu là đường chạy chính. Sample dài hơn
8K vẫn được xử lý đúng, không bị cắt ngoài `max_length` đã cấu hình.

### 3. Data-parallel topology và resume

Giữ parent/worker topology hiện tại để hạn chế rủi ro output collision:

- parent launch một worker trên mỗi GPU được cấp phát;
- mỗi worker dùng một SGLang engine local với `CUDA_VISIBLE_DEVICES` riêng;
- worker ghi shard riêng bằng `ShardedFeatureWriter`;
- parent chỉ merge sau khi tất cả worker hoàn tất và kiểm tra ID coverage;
- resume dựa trên manifest/sample ID, không dựa trên số dòng.

Khởi động engine được stagger và có health/VRAM check để tránh 4 process cùng
load checkpoint làm host bị `SIGKILL`. Một worker bị OOM sẽ ghi chẩn đoán gồm
GPU, bucket, batch, token budget và peak memory. Nếu bị `SIGKILL` ở ngoài
Python, parent ghi retry hint; người dùng có thể hạ token budget/profile rồi
chạy lại cùng output với `--resume`.

### 4. I/O và dung lượng

Capture GPU được tách khỏi serialization bằng writer bất đồng bộ. Cache dùng
shard lớn hơn, queue đủ sâu và số I/O thread cấu hình được. Không `fsync` mỗi
sample; shard được ghi atomically, sau đó manifest mới publish.

Pipeline phải ước tính `valid_tokens` và dung lượng BF16 trước capture. Với
Qwen3-4B và 5 layer concat, feature width hiện khoảng 12,800, tương đương
25.6KB/token. Nếu output dự kiến vượt dung lượng còn lại của storage, stage
dừng trước capture hoặc yêu cầu chọn staging directory local/NVMe.

## Invariants không được thay đổi

- Feature schema `mr_dflash_feature_sharded_v1`.
- Layer IDs, feature width, dtype provenance và max length trong manifest.
- Input/loss mask cùng chiều với hidden states.
- Không duplicate/missing sample sau merge.
- `--resume` không xử lý lại sample đã durable.
- Cache backend và correctness metadata được ghi vào manifest để không resume
  nhầm. Batch profile/token budget là metadata hiệu năng, được phép thay đổi
  sau OOM khi resume và manifest giữ `profile_history`.

## Validation scope

- **L0:** compile/import, CLI plan, schema/provenance, scheduler budget, resume
  và merge coverage bằng test CPU/fake capture.
- **L1:** smoke thật trên B200 với 1 GPU/128 sample, sau đó 4 GPU/512 sample;
  đo samples/s, tokens/s, peak VRAM, thời gian engine warm-up và I/O backlog.
- **Parity:** so sánh SGLang với HF capture trên fixture nhỏ cùng model/layer/mask;
  cho phép sai số số học đã ghi rõ, nhưng không cho phép sai shape/độ dài/ID.
- **Full run:** chỉ chạy sau khi L0/L1 pass; báo riêng compute throughput và
  storage throughput vì feature cache có thể lớn hơn nhiều so với JSONL.

## Tiêu chí chấp nhận

- Không còn khởi động 4 HF full-forward worker cho mode SpecForge cache.
- 4 GPU có aggregate progress đúng tổng sample/token và không duplicate/missing.
- Cấu hình profile đã chọn không OOM trong L1; nếu full run OOM, stage dừng
  sạch và tạo retry hint có batch/token budget thấp hơn, không làm hỏng shard
  đã durable.
- SGLang output đọc được bởi trainer hiện tại mà không đổi config feature.
- Có benchmark so sánh cùng model, max length, layer IDs và dataset subset với
  cache HF cũ; không claim speedup nếu chỉ đo thời gian ghi hoặc khác workload.

## Ngoài phạm vi

- Không thay đổi regenerate trong task này.
- Không thay đổi MR-DFlash model/loss/trainer.
- Không giảm layer count, không capture suffix-only và không nén mất mát nếu
  chưa có thí nghiệm parity riêng.
