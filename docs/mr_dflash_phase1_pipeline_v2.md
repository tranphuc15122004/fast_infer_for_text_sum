# MR-DFlash Phase 1 — pipeline source-to-cache mới

Tài liệu này mô tả pipeline Phase 1 hiện hành cho server B200 CUDA 13.0.
Pipeline tạo dữ liệu prompt, sinh assistant trajectory, kiểm tra, tokenize và
cache hidden states của target model. Phase này chưa train drafter.

## 1. Luồng xử lý

```text
ShareGPT + ArXiv
      │
      ▼
prepare: normalized/*.jsonl + split manifests
      │
      ▼
analyze: length statistics + length_manifest.jsonl
      │
      ▼
regenerate_full_{train,val,test}
      │
      ▼
validate_full_* → tokenize_full_*
      │
      ▼
cache_full_{train,val,test}: target hidden states
```

Lệnh chính là `scripts/mr_dflash/run_preprocess_pipeline.py`. Khi chạy với
`--full-context --full-context-length 32768`, pipeline dùng một regime 32K và
giữ tối đa input hợp lệ thay vì cắt về regime 3K/8K.

`--resume` chỉ reuse stage khi success marker, config hash và toàn bộ artifact
đúng. Vì vậy có thể dừng rồi chạy lại cùng `data-root` mà không tạo lại phần đã
hoàn tất.

## 2. Phân tích và sắp xếp theo độ dài

Stage `analyze` mặc định quét toàn bộ input, không còn giới hạn 1.000 dòng.
Nó ghi:

```text
<data-root>/manifests/analysis.json
<data-root>/manifests/length_manifest.jsonl
```

`length_manifest.jsonl` có `sample_id`, `index`, `length`, `source` và được
sắp xếp tăng dần theo length. Scheduler dùng metadata này để giao sample ngắn
trước; cache worker tiếp tục sắp xếp từng CPU buffer theo padded token length.
Thứ tự output cuối vẫn là thứ tự canonical của input ban đầu.

## 3. Adaptive batch trên B200

Giá trị khuyến nghị hiện tại:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| soft target | 160 GiB | điểm dừng tăng batch thông thường |
| hard cap | 163 GiB | giới hạn cấu hình/telemetry, không chạy sát toàn bộ VRAM |
| batch bắt đầu | 1 | khởi động an toàn cho context dài |
| growth factor | 2.0 | tăng nhanh khi batch trước thành công |
| max batch | 128 | trần khẩn cấp, không phải batch cố định |
| cache safety fraction | 0.95 | dự phòng cho token pool/allocator |
| length look-ahead safety | 0.85 | dự phòng khi sample kế tiếp dài hơn |

### Regeneration

Regeneration đo peak allocated/reserved VRAM sau mỗi lần `generate`. Controller
tăng batch theo từng bucket `prompt length + generation budget`. Trước batch kế
tiếp, controller dự đoán mức VRAM nếu thêm sample; nếu vượt ngưỡng an toàn thì
giữ batch hiện tại.

Nếu batch lớn bị OOM, cùng nhóm sample được giữ lại và thử lại với batch nhỏ
hơn. Sau OOM, controller tìm kiếm giữa batch an toàn và batch lỗi thay vì luôn
quay về batch 1.

Trước khi forward, regeneration nhìn trước các sample ngắn nhất sẽ được đưa vào
batch tiếp theo và ước lượng effective length:

```text
effective_length = prompt_tokens + generation_budget
batch_next <= batch_safe × length_safe / length_next × 0.85
```

Khi length tăng trong cùng bucket hoặc chuyển sang bucket mới, batch được giảm
trước khi chạy. Sample ngắn đầu run vẫn cho phép bước nhảy lớn; khi sample dài
dần, controller tự chuyển sang bước nhỏ hơn.

### Hidden cache

Cache dùng token capacity của static pool SGLang/SpecForge và giới hạn:

```text
batch <= auto_batch_max_size
batch <= max_total_tokens / padded_length
batch <= runtime_token_capacity / padded_length
```

Cache bắt đầu từ batch 1 theo từng length bucket. OOM giữ nguyên CPU buffer,
giảm batch, giải phóng cache nếu backend hỗ trợ rồi capture lại đúng sample.

Cache không chỉ nhìn `buffer[0]`: trước mỗi forward nó tính padded length của
toàn bộ nhóm candidate. Nếu nhóm sắp tới chứa sample dài hơn, batch được cap
theo padded length và token capacity trước khi gọi SGLang. Sau mỗi batch, length
thực tế được lưu làm mốc để warm-start bucket kế tiếp.

Hard cap được kiểm tra khi parse cấu hình và truyền vào worker. Runtime adaptive
dừng theo soft target; allocator/OOM là hàng rào cuối cùng.

## 4. Shared lease scheduler

Khi truyền `--parallel-gpu-ids`, pipeline mặc định dùng
`--parallel-scheduler shared_lease`. Queue chỉ dùng file trên shared
filesystem, không cần database hoặc NCCL:

```text
parallel_<mode>_<split>/
├── shared_queue/items.jsonl
├── shared_queue/meta.json
├── shared_queue/events.jsonl
├── shared_queue/.scheduler.lock/
├── rank_00/ ... rank_NN/
└── quarantine.jsonl
```

Mỗi GPU nhận một lease gồm các sample được sắp xếp theo length. Parent heartbeat
lease trong lúc worker chạy. Nếu parent/host bị kill, lease hết hạn và host mới
có thể reclaim.

Các nguyên tắc resume:

- chỉ chạy một coordinator trên cùng một `data-root` tại một thời điểm;
- dừng job cũ trước khi chuyển host nếu có thể;
- dùng cùng input, model path và `data-root`;
- có thể thay đổi số lượng hoặc ID GPU khi chạy lại;
- không trộn queue của model/context/layer configuration khác nhau.

Nếu worker chết sau khi đã flush một phần, các sample đã có artifact được commit
trước; sample chưa có artifact được trả về queue. Sample lỗi tối đa 3 lần sẽ
được ghi vào `quarantine.jsonl` và pipeline tiếp tục. Với regeneration, sample
quarantine cũng xuất hiện trong `*.skipped.jsonl`; với cache, merge ghi rõ các
ID bị quarantine trong manifest.

## 5. Lệnh chạy full trên B200

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="$PWD/src:$PWD/externals/SpecForge"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FI_OFFLINE=1

python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  --data-root /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_phase1_full \
  --target-model-path /workspace/storage-shared/models/Qwen3-4B \
  --sharegpt-count 50000 \
  --arxiv-count 50000 \
  --full-context \
  --full-context-length 32768 \
  --max-new-tokens 1024 \
  --device cuda \
  --torch-dtype bfloat16 \
  --local-files-only \
  --cache-backend specforge_sglang \
  --cache-attention-backend sdpa \
  --regenerate-auto-batch \
  --regenerate-auto-batch-target-vram-gb 160 \
  --regenerate-auto-batch-hard-vram-gb 163 \
  --regenerate-auto-batch-max-size 128 \
  --cache-auto-batch \
  --cache-auto-batch-target-vram-gb 160 \
  --cache-auto-batch-hard-vram-gb 163 \
  --cache-auto-batch-max-size 128 \
  --parallel-gpu-ids 0 1 2 3 \
  --parallel-scheduler shared_lease \
  --overflow-policy skip \
  --sample-error-policy error \
  --resume
```

Nếu source đã được chuẩn bị trong cùng `data-root`, `prepare` sẽ được reuse
nhờ `--resume`; pipeline sẽ chuyển đến stage còn thiếu. Muốn bắt đầu từ
regeneration sau khi `normalized/` và manifest đã sẵn sàng:

```bash
python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  ...các tham số của lệnh full... \
  --from-stage regenerate_full_train \
  --resume
```

## 6. Chuyển từ 4 GPU sang 2 GPU/host khác

Dừng parent cũ bằng `Ctrl-C` hoặc stop file. Sau đó trên host mới, dùng cùng
`data-root` và chạy lại với GPU mới:

```bash
export CUDA_VISIBLE_DEVICES=0,1

python3 scripts/mr_dflash/run_preprocess_pipeline.py \
  ...các tham số cũ... \
  --parallel-gpu-ids 0 1 \
  --parallel-scheduler shared_lease \
  --resume
```

Lease mặc định có TTL 300 giây. Nếu host cũ bị kill đột ngột, chờ lease hết
hạn trước khi host mới reclaim phần việc đang dở.

## 7. Theo dõi và kiểm tra artifact

Các file cần kiểm tra:

```text
<data-root>/pipeline_plan.json
<data-root>/pipeline_summary.json
<data-root>/pipeline_logs/<stage>.log
<data-root>/pipeline_state/<stage>.success.json
<data-root>/manifests/analysis.json
<data-root>/manifests/length_manifest.jsonl
<data-root>/**/parallel_*/status.json
<data-root>/**/parallel_*/quarantine.jsonl
```

Trạng thái GPU worker có thể xem bằng:

```bash
python3 scripts/mr_dflash/watch_parallel_stage.py \
  --status /path/to/parallel_regenerate_train/status.json \
  --interval 5 --tqdm
```

Một lần chạy được xem là thành công khi `pipeline_summary.json` là `success`
hoặc stage parallel là `success_with_quarantine`, tất cả artifact bắt buộc có
đủ và mọi sample thiếu đều được liệt kê rõ trong `quarantine.jsonl`.

## 8. Giới hạn môi trường

Runbook này dành cho server B200 với CUDA 13.0, Python `python3` hệ thống và
model snapshot local. Máy CPU/local chỉ có thể chạy unit test, dry-run hoặc
test doubles; không dùng kết quả CPU để kết luận throughput SGLang/B200.

Chi tiết backend cache xem
[`mr_dflash_cache_auto_batch.md`](mr_dflash_cache_auto_batch.md). Chi tiết path
và runtime server xem [`server_environment.md`](server_environment.md).
