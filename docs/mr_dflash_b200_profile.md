# Cấu hình MR-DFlash cho B200 100 GB

Các config pilot Qwen3-4B trong
[`src/MR_DFlash/configs/pilot_qwen3_4b/`](../src/MR_DFlash/configs/pilot_qwen3_4b/)
đã được khóa cho profile B200 100 GB. Profile này ưu tiên đúng protocol và
headroom bộ nhớ hơn việc đẩy batch lên tối đa.

Dataset artifacts của pilot được đặt mặc định tại
`/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot`,
không nằm trong working tree của repo. Vì vậy có thể chạy config trực tiếp từ
repo sau khi hoàn tất các phase chuẩn hóa, regenerate và cache.

## Profile đã áp dụng

Tất cả sáu config pilot (`3k` và `8k`) dùng chung:

```text
dtype                  bfloat16
batch_size             1 sample / rank
accumulation_steps     4
effective batch        4 sample / optimizer step trên 1 GPU
block_size             16
num_anchors            512
objective_chunk_blocks 64
attention_backend      sdpa
feature cache           offline, sharded, dùng chung theo target/context
```

`objective_chunk_blocks=64` chỉ chia nhỏ phần tính objective và target LM
head. Nó không thay đổi anchor sampling, block size, loss, thứ tự dữ liệu hay
đầu ra của model; vì vậy vẫn giữ được so sánh công bằng giữa DFlash-2L,
MR-DFlash-2S và DFlash-5L. Giá trị 64 giảm peak activation so với 128, nhất là
khi vocab target lớn.

Không tăng `batch_size` mặc định lên 2 hoặc 4. DFlash baseline hiện vẫn có
nhánh attention reference toàn draft sequence ở training; DFlash-5L/8K là
trường hợp cần đo peak VRAM đầu tiên. Nếu chỉ tăng batch cho MR-DFlash thì kết
quả không còn là pilot công bằng.

Các config Llama 3.1 8B benchmark cũng dùng `batch_size=1`, accumulation 4 và
`objective_chunk_blocks=64`; target lớn hơn nên không dùng batch 4 trực tiếp dù
B200 có 100 GB. Các config legacy Qwen ở thư mục gốc vẫn được giữ nguyên để
không phá các run cũ; khi chạy protocol pilot hãy dùng đúng thư mục
`pilot_qwen3_4b` hoặc config Llama benchmark tương ứng.

Trước khi train, phải tạo target feature cache theo từng context regime. Ví dụ
với 8K:

```bash
export TARGET_MODEL=Qwen/Qwen3-4B
for split in train val; do
  python3 scripts/mr_dflash/cache_target_features.py \
    --target-model-path "$TARGET_MODEL" \
    --data-path /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/regenerated/${split}.jsonl \
    --output-path /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/mr_dflash_pilot/target_features_qwen3_4b_8k/${split} \
    --target-layer-ids 1 9 17 25 33 --max-length 8192 \
    --batch-size 1 --bucket-buffer-size 8 --shard-size 32 \
    --device cuda --local-files-only --resume
done
```

Các config pilot đã trỏ tới thư mục này. `run_train.py` kiểm tra manifest,
target model, layer IDs, feature width và max length; nếu cache thiếu hoặc
không khớp, run dừng trước khi nạp draft để không train nhầm. Cache 3K dùng
thư mục `target_features_qwen3_4b_3k` tương ứng. Ước tính dung lượng trước khi
scale: Qwen3-4B có feature width 12,800 nên BF16 cần khoảng 25.6 KB/token.

## Chạy trên một B200

Chỉ truyền GPU đã được allocate cho tiến trình hiện tại; launcher không tự
chọn GPU và không dừng job khác:

```bash
export PYTHONPATH="$PWD/src"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0   # thay bằng GPU được cấp phát thực tế

python3 scripts/mr_dflash/check_fairness.py \
  src/MR_DFlash/configs/pilot_qwen3_4b/dflash_2l_8k.yaml \
  src/MR_DFlash/configs/pilot_qwen3_4b/mr_dflash_2s_8k.yaml \
  src/MR_DFlash/configs/pilot_qwen3_4b/dflash_5l_8k.yaml

python3 scripts/mr_dflash/run_pilot_matrix.py \
  --only mr_dflash_2s_8k.yaml --device cuda --max-steps 2 --run
```

Sau smoke 2 step, chạy lần lượt DFlash-2L, MR-DFlash-2S rồi DFlash-5L. Không
chạy song song trên cùng GPU. Theo dõi `peak_allocated_gb` và
`peak_reserved_gb` trong log/checkpoint metrics; không coi một run là hợp lệ
nếu có OOM hoặc peak vượt headroom vận hành.

## Chạy hai B200 và giữ effective batch

Config được viết cho một rank/GPU. Khi dùng hai GPU DDP, để giữ effective
batch bằng 4 của protocol một GPU, dùng:

```text
batch_size=1/rank
world_size=2
accumulation_steps=2
```

vì:

```text
1 sample/rank × 2 rank × 2 accumulation = 4 sample/update.
```

Đặt `training.dp_world_size: 2` trong bản copy config dành cho run hai GPU,
hoặc truyền thông số tương ứng theo launcher DDP đang dùng. Không dùng
`accumulation_steps=4` trên hai GPU nếu muốn so sánh trực tiếp với run một
GPU, vì khi đó effective batch sẽ tăng thành 8.

## Fallback khi DFlash-5L/8K vượt 100 GB

Thực hiện theo thứ tự, ghi lại thay đổi trong run manifest:

1. Xác nhận chỉ có GPU được cấp phát trong `CUDA_VISIBLE_DEVICES` và bật
   `expandable_segments`.
2. Giữ `batch_size=1`; giảm `objective_chunk_blocks` từ 64 xuống 32 cho cả
   ba variant của cùng experiment matrix.
3. Nếu vẫn OOM, dừng run 8K và ghi nhận rằng implementation DFlash reference
   hiện chưa có block-packed/FlashAttention training kernel. Không tự giảm
   `num_anchors`, `block_size` hoặc chỉ giảm riêng một baseline, vì như vậy
   sẽ thay đổi protocol hoặc phá fairness.

`num_anchors=512`, `block_size=16`, năm target feature layers và `max_length`
không được thay đổi trong pilot chính. Đây là các thành phần của experimental
contract, không phải knob memory tùy ý.

## Quy trình khuyến nghị

```text
R0: --max-steps 2 trên từng variant
R1: overfit 256–1K sample
R2: 10K sample / 3K tokens
R3: 90K train + 5K val + 5K holdout / 8K tokens
```

Chỉ tiến sang rung kế tiếp sau khi rung trước không có OOM, loss hữu hạn,
checkpoint load được và acceptance proxy có giá trị. Đo latency production
chưa phải mục tiêu của profile này vì MR-DFlash hiện vẫn dùng reference
Torch/einsum path.
