# AMR-DFlash

AMR-DFlash đặt một bộ nhớ đa phân giải trước pretrained DFlash-5L: giữ một
ngân sách raw gồm recent guard và các vị trí do selector xếp hạng, thêm global
slots tùy chọn, rồi dùng nguyên DFlash 16 vị trí để draft. Target vẫn xác minh
trên toàn bộ prefix bằng KV cache. Bản V0 chỉ hỗ trợ greedy, batch 1 và cần
checkpoint AMR đã fingerprint khi bật selector/compressor.

## Chuẩn bị trên B200

Nếu đã có corpus ShareGPT/ArXiv regenerate, dùng `prepare-data` để tạo manifest
giữ hội thoại nhiều lượt và split gốc, rồi capture/label supervision AMR mới.
Lệnh pilot và các điều kiện kiểm tra nằm trong
[hướng dẫn chuẩn bị dữ liệu train](../amr_dflash_training_data.md).

Đặt `TARGET_MODEL`, `DRAFT_MODEL`, `DATA_INPUT` và `FI_DEVICE=cuda` trong master
env ngoài repository. Model/tokenizer phải là snapshot local; không có bước
tải model hoặc cài package từ Internet. Config mặc định nằm ở
`src/AMR_DFlash/configs/pilot.yaml`; có thể đặt `AMR_CONFIG` để chọn config
khác. Các budget trong YAML là điểm bắt đầu pilot, chưa được hiệu chỉnh bằng
benchmark.

Chạy preflight trước khi chiếm GPU cho model:

```bash
bash scripts/run.sh amr_dflash preflight --require-b200
```

Lệnh kiểm tra Python/package, B200, target/draft config và weights, tokenizer
local, cùng input JSONL; nó không load model weights.

## Tạo state và acceptance labels

Chọn corpus huấn luyện JSONL đã cố định split theo document. Mỗi record cần có
`prompt` hoặc field mà `scripts/common/data_loader.py` hỗ trợ; có thể thêm
`id`, `split` (`train`, `validation`, `holdout`) và `reference`. Nếu thiếu
split, code gán split ổn định theo document ID. Ví dụ:

ID phải duy nhất trên toàn manifest. Nếu nhiều prompt thuộc cùng tài liệu,
đặt `source_document_id` (hoặc `document_id`/`source_id`) và giữ cùng split.
Code kiểm tra trùng ID, source và nội dung giao giữa các split trên toàn file,
kể cả khi chỉ chạy `--max-samples 1`.

```bash
export AMR_RUN_ID=pilot_2026_10_08
export AMR_RUN_ROOT="$PWD/outputs/amr_dflash/$AMR_RUN_ID"
export AMR_DATA_MANIFEST=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/amr_train.jsonl

bash scripts/run.sh amr_dflash capture --max-samples 200 --max-new-tokens 256 \
  --max-states-per-document 6
bash scripts/run.sh amr_dflash candidates
bash scripts/run.sh amr_dflash label
```

`capture` sinh trajectory greedy có verifier, lưu một feature tensor theo
document và tối đa sáu state cách đều (mặc định) cùng state index. `label` tạo
các candidate support có cùng raw budget,
khôi phục target cache một lần/state rồi crop về prefix sau từng candidate.
Labels ghi accepted prefix, survival, tie/censor status và candidate proposal
IDs. Với candidate tốt nhất chưa censor, lưu thêm target logits để train slot
compressor. Dữ liệu tạm và feature bundle theo document nằm trong `outputs/`.
Sau label, `training_signal.json` đếm preference/teacher hợp lệ theo split;
train cần có cả train preferences và train uncensored teachers.

Feature bundle khóa SHA-256 của cả target lẫn drafter tạo projection, feature
layers, dtype và attention backend. Manifest ghi `capture_contract` version 2
và digest; state, candidate, label, preference, teacher và checkpoint mang
digest để kiểm tra nguồn gốc. Tên file tensor có SHA-256 của toàn ID nên các
state có ID dài hoặc bị sanitize không ghi đè nhau.

`capture --resume` chỉ tiếp tục với cùng input hash, model bytes, seed,
memory config, giới hạn token/state và split policy. Có thể tăng
`--max-samples` hoặc chuyển snapshot sang mount khác nếu bytes giữ nguyên;
manifest gốc được giữ nguyên. Artifact thiếu hợp đồng mới phải capture lại
trong run root mới, rồi candidates/label/train lại; không thêm hash thủ công
để hợp thức hóa feature cũ.

## Train và đánh giá

```bash
bash scripts/run.sh amr_dflash train-selector --steps 200
export AMR_CHECKPOINT="$PWD/checkpoints/amr_dflash/$AMR_RUN_ID/selector.pt"
bash scripts/run.sh amr_dflash train-compressor --checkpoint-in "$AMR_CHECKPOINT" --steps 200
export AMR_CHECKPOINT="$PWD/checkpoints/amr_dflash/$AMR_RUN_ID/compressor.pt"
bash scripts/run.sh amr_dflash infer --split holdout --max-samples 100 --max-new-tokens 256 --overwrite
```

Selector học preference theo accepted proposals trong cùng state/budget và bỏ
tie/censor. Query dùng anchor embedding, feature context mới nhất và trung bình
16 feature context gần nhất; query chỉ đọc trạng thái đã commit. Compressor học
candidate-verifier-logit KL surrogate với backbone đóng băng; proposal history
được cố định tại lúc label và ghi trong manifest. Chạy thêm ablation cùng
raw-token budget:

```bash
bash scripts/run.sh amr_dflash infer --split holdout --mode dense --output outputs/amr_dflash/dense.jsonl --overwrite
bash scripts/run.sh amr_dflash infer --split holdout --mode selection --checkpoint "$AMR_CHECKPOINT" \
  --output outputs/amr_dflash/selection.jsonl --overwrite
bash scripts/run.sh amr_dflash infer --split holdout --mode amr --checkpoint "$AMR_CHECKPOINT" \
  --output outputs/amr_dflash/hybrid.jsonl --overwrite
```

`infer --split train|validation|holdout|all` lọc manifest theo document split
trước khi áp dụng `--max-samples`; mặc định là `all` để giữ tương thích. Dùng
`holdout` cho lần đánh giá confirmatory sau khi đã khóa checkpoint và ngưỡng.

Mỗi generation record theo schema chung, gồm timings, A/G acceptance, output
token IDs, state trace và ROUGE khi có reference; file kết thúc bằng summary.
`memory_build_latency_ms` (đồng thời map vào base field
`selector_latency_ms`) đo key-index/update, slot build và working-memory
assembly; không được hiểu là selector MLP riêng.
`memory_update_latency_ms` tách phần cập nhật keys/slots/bank sau verification;
`feature_projection_latency_ms` đo extraction/projection riêng. Prefix LM
head chỉ tính logits cuối, verifier vẫn giữ đủ 16 vị trí.

`ttft_ms` kết thúc khi token đầu và initial feature bank đã sẵn sàng;
`prefill_ms` chỉ đo target forward. `decode_ms`/`decode_time_ms` bằng E2E trừ
TTFT. `tpot_ms` bằng decode time chia cho `output_tokens - 1`, trả null khi chỉ
có một token. CUDA dùng events cho các stage và đồng bộ ở ranh giới sample/
token đầu; CPU dùng wall clock. Tổng các stage không bắt buộc bằng E2E vì
E2E còn có CPU bookkeeping và các thao tác cache/commit.

Generation có `sample_id`, `document_id`, `source_document_id`, `split`, hash
của tokenized prompt, input manifest, effective run config và checkpoint.
`workload_hash` giữ giống nhau giữa các mode có cùng workload;
`run_config_hash` còn khóa mode/checkpoint/cost gate. Ghép sample theo ID,
prompt hash và workload hash; luôn kiểm tra model/dtype/backend và timing
policy trước khi tính speedup. `avg_accept_length` là raw accepted proposals
cộng một trước EOS/cap; dùng committed counts để tính throughput thực.
Summary tính `decode_committed_tok_s` bằng tổng decode tokens chia tổng decode
time; các `mean_*` là thống kê mô tả từng sample.
Chỉ số latency được đo trên server B200 mới có ý nghĩa cho paper.

## Hợp đồng và giới hạn V0

- `dense`: full-context DFlash control, checkpoint AMR không cần.
- `selection`: selector raw memory, không có slots.
- `compressor`: slots, không có dynamic selector/raw selection.
- `amr`: selector + raw memory + learned slots.
- Target luôn full-context; verifier correction/EOS giữ target greedy output.
- Slots là compressed features trong không gian DFlash đã project, gắn vị trí
  theo centroid vị trí quan sát và nhận learned additive gate.
- `min_context_tokens` bật bypass dense cho prompt ngắn; đây là ngưỡng pilot,
  chưa phải cost-model đã hiệu chuẩn.
- Tập candidate giữ cùng raw-token budget và recent guard. Attention oracle,
  sampling, multi-GPU batch, layer-specific routing và measured speedup chưa
  được triển khai/khẳng định ở V0.
- Checkpoint từ model/config/tokenizer khác bị từ chối qua SHA-256 fingerprints.
  Mount path chỉ là provenance, không tham gia so sánh identity. Seed được
  đặt trước khi khởi tạo module; checkpoint ghi seed và trạng thái deterministic
  algorithms. Kết quả CUDA vẫn cần kiểm chứng trên stack server thực.
- `evaluate-fixed` chạy same-state hard-policy acceptance trên split đã capture;
  nó chưa được tự động gọi trong training cadence. GPU-hour ledger V0 tính
  capture/label/train wall time trên CUDA; fixed-state/rollout evaluation không
  nằm trong adaptation cap. V0 chưa có optimizer resume, calibrated cost
  crossover, resident-memory breakdown hay B200 run artifact.

Checkpoint fingerprint khóa toàn bộ memory config. Muốn so raw/slot budgets
khác nhau cần capture/label/train một run riêng cho từng config; các mode ở ví
dụ trên chỉ so cùng raw budget và chưa phải matched-total-budget ablation.

Xem [proposal](../../src/ARMdflash/AMR_DFlash_Research_Proposal_and_Paper_Story_2026-10-08.md),
[data contract](../../src/ARMdflash/AMR_DFlash_Implementation/data_training_contract.md)
và [acceptance checklist](../../src/ARMdflash/AMR_DFlash_Implementation/acceptance_checklist.md).
