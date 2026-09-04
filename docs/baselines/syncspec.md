# SyncSpec-v1

SyncSpec-v1 là speculative decoding lossless theo target full-context. Drafter
diffusion sinh candidate lattice, selector chuẩn hóa `q` trên Top-M, survival
head dự báo prefix sống sót, controller chọn `K_v` sau khi đã quyết định `K_d`,
và target xác minh chính xác bằng cache transaction. Target không bị prune và
không evict KV.

Khi toàn bộ proposal được accept, cache/logits/hidden của block target đã tính
được commit trực tiếp; chỉ round có rejection mới cần commit lại prefix tuần
tự. Điều này tránh forward duplicate trong đường speculative chính.

## Chạy CPU synthetic

```bash
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  "$PWD/.venv/bin/python" scripts/infer_syncspec.py \
  --backend synthetic --smoke --device cpu --output /tmp/syncspec.jsonl
```

GPU engine smoke không cần model/checkpoint, nhưng phải chạy trên CUDA thật:

```bash
bash scripts/run_syncspec_cuda_smoke.sh
```

Trên máy không có CUDA, lệnh kết thúc với status `BLOCKED` có cấu trúc và mã
thoát 2; đây không được tính là GPU pass. Smoke với target/checkpoint thật trên
B200 dùng các wrapper ở phần dưới.

Các CLI `infer_syncspec.py`, `build_syncspec_trajectories.py` và
`train_syncspec.py` cũng từ chối rõ ràng `--device cuda` khi CUDA không khả
dụng; chúng không tự rơi về CPU. Vì vậy một lệnh GPU gọi trực tiếp không thể
vô tình được ghi nhận như GPU pass trên máy dev.

Smoke CPU toàn chuỗi (Stage 0 → joint drafter/selector/survival → profile →
batch inference + vanilla-AR exactness) dùng backend synthetic deterministic:

```bash
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  SYNCSPEC_CPU_SMOKE_DIR=/tmp/syncspec_cpu_smoke \
  bash scripts/run_syncspec_cpu_smoke.sh docs/fast_infer_master.example.env
```

Runner này giữ artifact trong thư mục `SYNCSPEC_CPU_SMOKE_DIR` và kiểm tra đủ
ba checkpoint component trước khi profile/infer. Đây là smoke của orchestration
và correctness, không dùng timing CPU để kết luận speedup B200.

## Stage 0 → train drafter → inference

SyncSpec dùng cùng semantics block với DFlash: `kd` vẫn là số proposal đưa
qua selector/verifier, nhưng input vật lý của drafter có `kd + 1` slot. Slot
đầu là token cuối cùng đã được target commit (thường là bonus token sau một
block được accept hoàn toàn, hoặc correction token khi block bị reject), các
slot còn lại là mask. Target hidden và recent-hidden sau commit được dùng làm
điều kiện cho block kế tiếp. Slot anchor không được đưa vào candidate lattice
hoặc diffusion loss.

```bash
python3 scripts/build_syncspec_trajectories.py \
  --backend transformers --target-model "$MODEL_TARGET" \
  --input data/your_records.jsonl --output /tmp/syncspec-trajectories.jsonl \
  --device cuda --local-files-only --include-target-features \
  --include-source-memory --source-chunk-size 128 --num-anchors 512 \
  --seed 42 --resume
python3 scripts/train_syncspec.py --stage diffusion \
  --data /tmp/syncspec-trajectories.jsonl --output-dir checkpoints/syncspec \
  --target-model "$MODEL_TARGET" --device cuda --kd 16 --steps 1000 \
  --train-batch-size 1 --num-anchors 512 --position-decay 7 \
  --attention-backend flash --seed 42
# Training ghi loss, step time và token throughput vào
# checkpoints/syncspec/training_steps.jsonl.
# Mỗi forward chọn ngẫu nhiên tối đa 512 eligible anchors từ mỗi trajectory;
# anchor thiếu đủ suffix ``kd`` hoặc vượt physical positional capacity bị loại.
# Nếu toàn bộ trajectory bị cắt bởi EOS trước ``kd``, pipeline dùng fallback
# loss-mask cho phần suffix còn lại và ghi ``truncated_suffix_fallback``.
# ``flash`` ép ưu tiên FlashAttention qua SDPA dispatcher; nếu kernel không
# khả dụng thì tự động fallback sang efficient/math SDPA.
# Với cache torch binary, dùng output path kết thúc bằng `.pt` và truyền cùng
# path đó cho mọi training stage; fingerprint/resume được kiểm tra như nhau.
# Ví dụ: --output /tmp/syncspec-trajectories.pt và --data /tmp/syncspec-trajectories.pt
# Tiếp tục stage này từ model + optimizer checkpoint trước đó bằng:
#   --init-checkpoint checkpoints/syncspec
# Tùy chọn nếu Stage 0 cũng dùng --include-logits:
#   --kl-weight 0.1 --rank-weight 0.05 --rank-margin 0.2 --rank-top-m 16
python3 scripts/train_syncspec.py --stage selector \
  --data /tmp/syncspec-trajectories.jsonl --output-dir checkpoints/syncspec-selector \
  --target-model "$MODEL_TARGET" --init-checkpoint checkpoints/syncspec \
  --device cuda --kd 16 --steps 1000 --train-batch-size 1
python3 scripts/train_syncspec.py --stage survival \
  --data /tmp/syncspec-trajectories.jsonl --output-dir checkpoints/syncspec-survival \
  --target-model "$MODEL_TARGET" --init-checkpoint checkpoints/syncspec \
  --selector-checkpoint checkpoints/syncspec-selector \
  --device cuda --kd 16 --steps 1000 --train-batch-size 1
# Tùy chọn Stage 4: lệnh `joint` chạy các stage và sau đó tinh chỉnh joint
# trong cùng một output checkpoint; dùng --init-checkpoint nếu cần tiếp tục
# từ một diffusion checkpoint đã có.
python3 scripts/train_syncspec.py --stage joint --joint-finetune \
  --data /tmp/syncspec-trajectories.jsonl --output-dir checkpoints/syncspec-joint \
  --target-model "$MODEL_TARGET" --device cuda --kd 16 --steps 1000 \
  --joint-steps 200 --joint-learning-rate 1e-5 --train-batch-size 1
python3 scripts/infer_syncspec.py --backend transformers \
  --target-model "$MODEL_TARGET" \
  --drafter-checkpoint checkpoints/syncspec \
  --selector-checkpoint checkpoints/syncspec-selector \
  --survival-checkpoint checkpoints/syncspec-survival \
  --input data/your_records.jsonl --output outputs/syncspec.jsonl \
  --device cuda --batch-size 4 --kd 16 --kv 8 --local-files-only
```

Để bật đầy đủ pre-draft adaptation, profile và infer cùng một finite set:

```bash
PROFILES="8:4,8:8,16:4,16:8,16:12,16:16"
python3 scripts/profile_syncspec.py --backend transformers \
  --target-model "$MODEL_TARGET" --drafter-checkpoint checkpoints/syncspec \
  --selector-checkpoint checkpoints/syncspec-selector \
  --survival-checkpoint checkpoints/syncspec-survival \
  --input data/your_records.jsonl --output outputs/syncspec-profile.json \
  --device cuda --dtype bfloat16 --batch-size 4 \
  --budget-profiles "$PROFILES" --local-files-only
python3 scripts/infer_syncspec.py --backend transformers \
  --target-model "$MODEL_TARGET" --drafter-checkpoint checkpoints/syncspec \
  --selector-checkpoint checkpoints/syncspec-selector \
  --survival-checkpoint checkpoints/syncspec-survival \
  --input data/your_records.jsonl --output outputs/syncspec.jsonl \
  --device cuda --batch-size 4 --budget-profiles "$PROFILES" \
  --profile outputs/syncspec-profile.json --local-files-only
```

Profile thật trên B200 (profile này phải dùng đúng target/drafter đang benchmark):

```bash
python3 scripts/profile_syncspec.py --backend transformers \
  --target-model "$MODEL_TARGET" \
  --drafter-checkpoint checkpoints/syncspec \
  --selector-checkpoint checkpoints/syncspec-selector \
  --survival-checkpoint checkpoints/syncspec-survival \
  --input data/your_records.jsonl --output outputs/syncspec-profile.json \
  --device cuda --dtype bfloat16 --batch-size 4 \
  --budget-profiles "8:4,8:8,16:4,16:8,16:12,16:16" \
  --local-files-only
```

Profile lưu cả `target_ar`, p50/p95 cho từng component và peak GPU memory;
controller tính chi phí một round từ draft + selector + survival + verify +
scheduler (profile legacy chỉ có `verify` vẫn được đọc). CUDA inference chỉ
bật speculation khi profile khớp target/drafter/selector/survival checkpoint,
GPU/precision/context/batch/Kd/Kv. Nếu thiếu hoặc lệch profile, engine dùng AR
fallback an toàn. Với
finite adaptive set, profile JSON có thể là list gồm nhiều bản ghi và phải
được tạo bằng cùng `--budget-profiles`; `--kd/--kv` vẫn là chế độ fixed-profile
khi cần tái lập một cặp duy nhất.

`target_ar` trong profile mới được đo cho một decode token sau prefill của mỗi
request (`target_ar_tokens=batch_size` trong profile batch; bằng `1` ở scalar);
prefill được ghi riêng. Các component cost shared của microbatch cũng được
chuẩn hóa theo batch trước khi controller so sánh utility speculative với AR
cùng đơn vị. Nếu speculative utility không vượt safety margin so với AR, engine
vẫn đã trả tiền draft nhưng commit một token AR và ghi `kv=0` trong budget.
Fallback này cũng cập nhật acceptance EMA bằng zero để pre-draft gate không
lặp lại chi phí draft vô ích cho cùng request.

Profile dùng để gate CUDA phải có `schema_version=1` và
`source="measured"`; preflight và engine đều từ chối profile synthetic hoặc
không rõ nguồn. B200 preflight còn đối chiếu profile với target model,
drafter/selector/survival checkpoint, precision, GPU và `batch_bin` đang chạy; profile CPU hoặc
khác batch sẽ bị từ chối trước inference. Điều này ngăn calibration
placeholder vô tình bật speculation trên B200 hoặc khiến smoke rơi im lặng về
AR.

Engine cũng giữ runtime feedback theo từng request: EMA của acceptance prefix,
độ dài committed và latency draft/selector/survival/verify/scheduler. Sau một
round có acceptance thấp, pre-draft prior được giảm bảo thủ; nếu utility rơi
dưới margin, request chuyển sang AR fallback. State này được ghi trong trường
`runtime_feedback` của output record để audit scheduler behavior.

Inference Transformers production bắt buộc truyền cả
`--selector-checkpoint` và `--survival-checkpoint`; nếu thiếu, CLI sẽ dừng
trước khi load model để tránh chạy bằng component random. Cờ
`--allow-untrained-components` chỉ dành cho development/diagnostic; profile
được tạo bằng cờ này mang `source="diagnostic"` và bị preflight/engine từ chối
khi dùng để gate production CUDA.

Adapter Transformers lấy hidden cuối qua final-normalization hook trong
prefill/verify để tránh materialize hidden state của mọi layer trên context
dài; chỉ model local không có final norm/hidden output mới dùng compatibility
fallback `output_hidden_states=True`.

Để kiểm chứng lossless trên đúng target/checkpoint đang chạy, thêm
`--check-exactness` ở một smoke run greedy. CLI sẽ chạy một vanilla target-AR
reference độc lập cho từng request, ghi `exact_match_vanilla_ar` ở mỗi record
và `exactness_failures` ở summary; nếu có mismatch, process trả mã lỗi. Cờ này
không bật trong benchmark timing thông thường vì reference làm thêm một lần
target generation. Không kết hợp cờ này với `--stochastic`, vì stochastic
exactness cần kiểm định phân phối thay vì so sánh với greedy output.

Model, tokenizer, dataset và checkpoint phải tồn tại local; launcher không
được phép tự tải internet trên server. Thông số B200 canonical nằm trong
`docs/server_environment.md` và master config ngoài repository.
Trước smoke B200, chạy preflight với các asset thật; preflight sẽ resolve HF
repo ID sang snapshot cache local và kiểm tra config/tokenizer cũng như
`vocab_size`/`hidden_size` giữa target và drafter; preflight cũng từ chối
drafter có `max_positions` nhỏ hơn `max_position_embeddings` của target và
selector có `vocab_size/hidden_size` không khớp. Nó cũng yêu cầu tokenizer
artifact thực (không chỉ `tokenizer_config.json`) để bảo đảm offline loading.
Phase `infer` cũng yêu cầu measured profile; phase `train` không yêu cầu vì
profile được tạo sau khi smoke sinh checkpoint.
Với `--strict`, preflight trả exit `0` khi `PASS`, `2` khi môi trường bị
`BLOCKED` (có thể retry trên server đúng), và `1` khi cấu hình/asset `FAIL`.

Stage 1 dùng Anchor-Offset: mỗi anchor được truyền vào drafter với vị trí
tuyệt đối `prompt_len - 1 + anchor`, vì block vật lý bắt đầu tại token anchor
đã commit, để backbone thấy đúng vị trí decode khi chạy
long-context. Batch có thể dùng offset khác nhau theo từng dòng. Các slot
`[MASK]` dùng một sentinel embedding học được riêng; `mask_token_id` chỉ là ID
placeholder trong tensor, không làm thay đổi embedding/LM head frozen của
target.

Stage 0 chỉ lưu `target_features` tại các anchor được chọn, kèm metadata
`target_feature_positions`; điều này tránh đưa toàn bộ suffix hidden state vào
JSON khi train long-context. Cache cũ lưu feature ở mọi target position vẫn
được đọc bằng fallback tương thích. `--include-logits` vẫn là tùy chọn riêng vì
full-vocabulary logits có thể rất lớn và chỉ nên bật khi cần KL.

Với train target thật, bật `--include-source-memory` để Stage 0 lưu mean
pooled final-hidden descriptor cho từng source chunk. Stage 1/2/joint dùng lại
đúng descriptor target-derived này như serving; nếu bỏ cờ, cache cũ vẫn chạy
nhưng training dùng embedding fallback và không phản ánh đầy đủ source-memory
path. Descriptor có kích thước theo số chunk, không lưu toàn bộ source hidden.

Trajectory path `.pt` bật torch cache binary (atomic write, fingerprint và
resume theo `sample_id`); `.jsonl` tiếp tục là format dễ kiểm tra bằng mắt.
Fingerprint bao gồm cấu hình CLI, tokenizer và manifest nội dung artifact
local của target (hash đầy đủ metadata nhỏ, lấy mẫu đầu/cuối của shard lớn),
nên cache sẽ không bị dùng lại âm thầm khi model tại cùng path thay đổi.
Model ID chưa resolve thành path local vẫn được giữ như một định danh ổn định;
trên server nên truyền snapshot local canonical để có provenance đầy đủ.

Khi train drafter với target thật, `max_positions` mặc định kế thừa
`max_position_embeddings` của target để không âm thầm wrap vị trí ở 4096 trong
long-context. Nếu checkpoint đã tồn tại nhưng khác capacity, CLI dừng với lỗi
không tương thích thay vì tiếp tục train sai topology.

Khi train với `--target-model`, drafter checkpoint chỉ lưu các trọng số
shallow drafter; embedding và LM head frozen của target được đánh dấu trong
`checkpoint_metadata.json` và tie lại lúc selector/survival/inference load.
Checkpoint synthetic standalone không bị rút gọn.

Stage 4 pre-gate calibration dùng trace ghép cặp có
`realized_gain`, hoặc `throughput_tok_s` và `ar_throughput_tok_s`:

```bash
python3 scripts/calibrate_syncspec_gate.py \
  --input outputs/syncspec_paired_traces.jsonl \
  --output checkpoints/syncspec-gate.json
python3 scripts/infer_syncspec.py --backend transformers \
  --gate-table checkpoints/syncspec-gate.json \
  --target-model "$MODEL_TARGET" --drafter-checkpoint checkpoints/syncspec \
  --selector-checkpoint checkpoints/syncspec-selector \
  --survival-checkpoint checkpoints/syncspec-survival \
  --input data/your_records.jsonl --output outputs/syncspec.jsonl \
  --device cuda --kd 16 --kv 8 --local-files-only
```

Nếu trace có thêm trường `kd` (hoặc `budget.kd`), calibrator giữ riêng trục
`K_d` với key dạng `long:batch1:kd8`. Pre-gate sẽ chọn profile có utility gain
dự báo cao nhất vượt margin; khi bảng đã có dữ liệu theo `K_d`, profile chưa
được đo sẽ không được tự động bật bằng `default_gain`. Trace cũ chỉ có
`context_bin/batch` vẫn tương thích và giữ hành vi fallback theo bảng tổng.

Smoke toàn chuỗi train trên B200 (Stage 0 → joint drafter/selector/survival →
infer) dùng thư mục output riêng mặc định:

```bash
bash scripts/run_syncspec_b200_train_smoke.sh
```

Lệnh này dùng `SYNCSPEC_TRAIN_*` trong master config; `--phase train` chỉ yêu
cầu target model và data vì drafter checkpoint sẽ được tạo trong smoke. Checkpoint
mặc định nằm trong `checkpoints/`, còn trajectory/preflight/profile/output nằm
trong `outputs/`; trajectory mặc định là binary
`outputs/syncspec_b200_train_smoke/trajectories.pt` (đặt
`SYNCSPEC_TRAIN_TRAJECTORY` thành `.jsonl` nếu cần đọc trực tiếp). Wrapper train profile đúng target/drafter/selector/survival
trước infer và chạy infer preflight để xác nhận đủ artifact. Sau khi smoke pass,
đặt `SYNCSPEC_TRAIN_STEPS`, `SYNCSPEC_TRAIN_KD`,
`SYNCSPEC_TRAIN_BATCH_SIZE` và output checkpoint phù hợp cho run train thật.

`--train-batch-size` là số trajectory/lattice rows trong mỗi forward và mỗi
optimizer step của các stage diffusion, selector, survival và joint. Mặc định
là `1`, phù hợp với long-context trên B200; tăng lên chỉ sau khi profile memory
và throughput. Stage selector cũng chunk forward tạo Top-M lattice theo cùng
giá trị này, nên không giữ toàn bộ trajectory trong một activation graph.

## Output và correctness

Mỗi dòng sample có timing draft/selector/survival/verify, `K_d/K_v`, accepted
length và token IDs; dòng cuối là summary. Greedy output phải bằng vanilla AR
trên cùng target/prompt/EOS/precision. Benchmark speed chỉ có ý nghĩa sau khi
đã có profile thật cho GPU/context/batch, không dùng timing CPU.
Verifier cắt proposal và transaction payload tại EOS trước khi commit; đường
AR fallback cũng dừng ngay tại EOS nên không sinh token thừa sau kết thúc.
Nếu input có `reference`, inference tự ghi `rouge1/rouge2/rougeL` trên sample
và trung bình tương ứng ở summary cuối.

`--batch-size` là microbatch thực: các request tương thích được gom cho shallow
drafter forward và target block verification; request khác độ dài vẫn được
regroup theo full-context length và giữ cache riêng. Prefill hiện vẫn diễn ra
theo từng request, vì vậy phải profile đúng batch/context trên phần cứng đích
trước khi claim throughput continuous serving.
