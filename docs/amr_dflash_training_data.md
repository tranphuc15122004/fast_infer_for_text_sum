# Dữ liệu và huấn luyện AMR-DFlash

Cập nhật: **09/10/2026**. Phương pháp hiện tại là **memory adapter nhẹ,
frozen target + frozen pretrained DFlash-5L**.
[Hồ sơ ý tưởng và paper story](amr_dflash_paper_story.md) là tài liệu chính
kiểm soát method, recipe training và claims. [Đặc tả kỹ thuật](superpowers/specs/2026-10-08-amr-dflash-design.md)
ghi contracts kiến trúc/loss; [kế hoạch](superpowers/plans/2026-10-08-amr-dflash-implementation.md)
ghi các phần cần triển khai.

**Trạng thái:** converter response/token/cache, compact teacher collector và
trainer adapter-only mới chưa có CLI. Các lệnh prepare-data/capture/candidates/
label/train-selector/train-compressor bên dưới chỉ dành cho **V0 lịch sử**.
Không dùng manifest prompt-only hoặc candidate teachers của V0 như dataset
đã sẵn sàng cho prediction training mới.

**Cập nhật artifact 09/10/2026:** người vận hành xác nhận trên server đã có
regenerated train/val và target-feature cache 50K được tạo với Qwen3-4B.
Artifact hiện có; khả năng reuse cho AMR vẫn cần audit fingerprints và parity.

## 1. Dữ liệu của phương pháp chính

~~~text
Regenerated prompt + target response
              + audited target/projected feature cache
              -> random response anchors + valid block labels
              -> shared memory adapter -> frozen DFlash
              -> weighted draft CE + indexer KL trên labeled anchors
~~~

Một optimizer cập nhật global pooling, indexer và gate. DFlash fc/norm,
attention/FFN, target embeddings và output head đều frozen.
Không cần tạo candidate preferences hoặc regenerate trajectories lần nữa
để xây main prediction labels. Full-context verifier vẫn dùng trong evaluation.

| Thành phần dữ liệu | Nội dung cần có |
|---|---|
| Document identity | id, source_document_id, source, split, full-history content hash |
| Prompt | Toàn bộ messages trước assistant response cuối; template/token IDs đúng cache |
| Response | Target-regenerated assistant tokens, EOS/stop policy và generation metadata |
| Features | Raw concatenated target features hoặc projected H; vị trí và độ dài khớp tokens |
| Anchor manifest | document_id, absolute anchor position, valid proposal length, seed |
| Teacher subset | Train-only anchor ID, eligible fine-group IDs, compact attention distribution |
| Reference | ArXiv reference gốc để ROUGE; không dùng làm target draft labels |

Không nối gold response vào prefix của mọi block. Anchor a chỉ dùng features
ở positions [0,a); input là token x_a và 15 masks. Labels nằm ở a+1...a+15,
cắt valid mask tại EOS/response end. Features của những proposal positions
không vào compressor, indexer hoặc local memory của block đó.

Target có thể xử lý whole clean sequence bằng causal teacher forcing một lần.
Nhiều random blocks được train chung, với intra-block bidirectional attention
và không có inter-block leakage. Đây là cách xây dữ liệu của
[DFlash](https://arxiv.org/html/2602.06036v1#S4.SS2), không phải rollout từng
sample qua candidate verifier.

Regenerate ở temperature khác zero vẫn cung cấp distillation tokens, nhưng
teacher-token prefix match không tự là actual greedy acceptance.
Giữ decoding metadata và đo actual A bằng target verifier trên validation/holdout.

## 2. Tận dụng corpus/cache 50K hiện có

Dữ liệu server do người vận hành cung cấp:

~~~text
/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_50k_50_50_10k_b200_gpu0_ctx16k_out1k_20260928T132928Z/regenerated_full/
/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_50k_50_50_10k_b200_gpu0_ctx16k_out1k_20260928T132928Z/target_features_qwen3_4b_full/
~~~

Trong thư mục đầu có train.jsonl và val.jsonl; thư mục thứ hai chứa cache
features target Qwen3-4B theo train/val. Giữ nguyên artifacts nguồn và split.
Config MR-DFlash khai báo tên cache này; workspace local không đọc trực tiếp
artifact trên server.

**Artifact tồn tại không đồng nghĩa đã tương thích.** Audit ngày 05/10 của
cache thuộc run 28/09 ghi hidden recompute fail 32/32 mẫu train và 32/32 val;
manifest không có target-weight fingerprint. Chưa có report audit mới sau lần
fail đó. Vì vậy cache đã có nhưng chưa được xác nhận để reuse với AMR.
Audit lại theo từng document trước khi nhập:

1. So target snapshot bytes, tokenizer/template, hidden layer IDs và offset
   hidden_states[layer_id + 1], dtype/backend policy.
2. So chính xác token IDs/absolute positions của sequence dùng để tạo cache;
   xác định prompt/response boundary, response EOS và truncation.
3. Với projected cache, so thêm draft fc + hidden_norm fingerprint. Không
   project thêm lần nữa. Raw target features được project một lần bằng frozen
   fc/norm rồi cache để tránh gọi target trong main training.
4. Recompute một subset train/validation trên stack server theo tolerance đã
   khóa, kiểm tra shape, finite values và feature agreement.
5. Ghi audit manifest, số pass/fail, cache hashes và lý do loại; cache mismatch
   bị từ chối. Nếu phải recompute, làm một lần offline theo document, teacher
   forcing từ regenerated sequence, không chạy full autoregressive generation.

Không thể nhận cache chỉ từ tên “50K”, shape hoặc cùng model path.
[Audit MR ngày 05/10](mr_dflash_gpu_experiments.md) đã ghi hidden recompute
fail 32/32 train và 32/32 validation. Kết quả này chưa chứng minh cache người
vận hành đang dùng đã được tái tạo và vượt parity. Artifact hiện hữu theo xác
nhận ngày 09/10; chỉ sau audit lại mới quyết định reuse/recompute.
Hiện chưa có importer được kiểm chứng cho adapter mới.

Nếu thay template hoặc head/tail truncation làm đổi prefix, cached features
cũ không còn hợp lệ. Đối với greedy-aligned evaluation, regenerate response
cũng phải khớp target conditioning. Chọn anchors có context length trong cap
đã khóa; không thay prefix âm thầm để vừa batch.

## 3. Làm sạch split và chọn pilot

Source document và full-history content không được giao giữa train,
validation và holdout. Kiểm tra cả source file trước khi chọn subset; loại
overlap bằng quy tắc đã ghi, giữ report và không đổi train labels theo validation.

Báo cáo **người vận hành cung cấp ngày 09/10/2026**, chưa là artifact B200
đã tải về workspace:

- Validation cleaning giữ 2.399 records, bỏ 101 records content-overlap.
- V0 prepare chọn 200 train + 50 validation trong run
  amr_pilot_20261009T180405Z.
- Train gồm 100 ArXiv + 100 ShareGPT; validation gồm 44 ArXiv + 6 ShareGPT.
- Với prompt >=4097 tokens: chỉ 139 ShareGPT train và 6 ShareGPT validation
  eligible; số này không đại diện mọi anchors trong response.
- Prompt selected dài 4.140–10.234 tokens, mean 6.473,772.
  Báo cáo này chưa chứng minh coverage 16K, completion capture, training
  hoặc speedup.

Adapter pilot mới dùng 2–5K train documents **khi corpus có đủ mẫu phù hợp**.
Khóa validation riêng, điểm bắt đầu 50–100 documents cùng fixed anchor manifest.
Stratify theo source và context length tại anchor, báo số eligible/selected
thật; không hứa 50/50 ở long context nếu ShareGPT không đủ.
Training tập trung context nơi AMR active; short context dùng để kiểm tra
dense bypass/parity và calibration.

Không còn điều kiện “prompt phải dài hơn raw budget để tạo preference”.
Mọi anchor phải có valid proposals, prefix features hợp lệ và adaptation
path active. Mask toàn bộ hoặc dense bypass hoàn toàn thì skip với reason,
tránh backward trên một loss không có trainable graph.

## 4. Compact teacher cho indexer

Chọn subset từ **train split**, anchors cố định theo seed, độc lập validation.
Chạy frozen dense DFlash trên cached context và cùng masked blocks.
Aggregate attention qua layers/heads/proposal positions; dùng early-position
weights như main loss, gộp raw-token mass theo fine groups.

Chỉ lấy groups hoàn tất, trước anchor và nằm ngoài local guard; chuẩn hóa
trên đúng eligible domain. Mỗi teacher record lưu group IDs/distribution,
document/anchor identity, visibility policy, dense snapshot và collector backend.

Các điểm bắt buộc:

- Teacher distribution là drafter attention proxy, không phải target attention
  hoặc actual acceptance utility.
- Không lưu full attention tensors hoặc [15,V] logits cho toàn bộ 50K.
- Empty/zero-mass eligible domain được skip có reason, không gán reward giả.
- Collector xử lý anchor chunks để hạn chế peak VRAM; thời gian collector
  được tính vào adaptation cost.
- Main sampler tiếp tục đưa teacher-labeled anchors vào batch, tỷ lệ pilot
  25% khi đủ dữ liệu; anchors chưa có teacher chỉ dùng prediction CE.
- Không lấy validation attention để train scorer, không dùng inference-time
  attention teacher làm một “deployable selector”.

Indexer KL truyền gradient trực tiếp vào scores; hard Top-K vẫn không truyền
gradient qua indices. Tiền lệ distillation của
[DSA](https://arxiv.org/html/2512.02556v1#S2.SS1.SSS1) không thay actual
acceptance validation cho AMR.

## 5. Warm-up và main training

| Giai đoạn | Dữ liệu/loss | Cách tiết kiệm |
|---|---|---|
| Baseline/audit | Dense DFlash và cache recompute subset | Khóa identity trước khi train |
| Warm-up, tối đa 200 steps pilot | Weighted CE + indexer KL, pooling uniform, memory budget rộng | Backbone frozen; teacher subset |
| Main adapter training | Cùng loss; final group/budget config | Cache features, nhiều anchors/forward |
| Validation/calibration | Prediction + actual greedy rollout | Cadence riêng, cost được ghi |
| Holdout | Config/checkpoint đã khóa | Không cập nhật weights |

Main config đề xuất: fine groups m=4, K=256; global c=64; local L=128;
index dim 64, pooling MLP hidden 32, eight anchors/document mỗi lượt,
AdamW LR 1e-4, seed 17. Warm-up dùng c=32 và K<=1024, chuyển về final
budget qua validation. Đây là thiết kế, chưa là YAML hoặc flags của V0.

Khởi tạo global pooling gần mean và finite global logit bias -4.
Một optimizer chứa pooling/indexer/gate; target và toàn bộ DFlash có
requires_grad=False và không nằm trong optimizer. Chỉ feature extraction
được no_grad; frozen draft/head forward giữ graph để compressor học.

Dùng loader lazy theo document, bounded RAM cache/LRU và prefetch.
Không nhân bản toàn bộ feature bank cho mỗi anchor; packing dùng masks đúng.
Chunk logits/CE theo anchor blocks để tránh tensor [B,anchors,15,V] quá lớn.
Freeze backbone vẫn cần input gradients qua năm layers; profile forward,
backward, cache I/O và teacher collection, không suy speedup từ số params.

Evaluation dùng full validation anchor manifest mỗi 50 steps; actual
full-validation rollout mỗi 500 steps và cuối phase. Evaluator hỗ trợ
checkpoint/in-memory, progress riêng, summary thời gian/metrics và fail
khi thiếu checkpoint, dataset rỗng, NaN hoặc contracts sai.
Mean accepted-proposal retention >=95% ở nonzero dense bins và throughput
>1x là **gate pilot đề xuất**, chưa là kết quả; xem
[validation criteria](superpowers/specs/2026-10-08-amr-dflash-design.md#8-validation-reproducibility-và-chi-phí).

Checkpoint mới lưu adapter + optimizer/scheduler/RNG/sampler, phase/step,
budget schedule, token/cache/teacher contracts và eval history.
V0 preferences/teacher-logit/checkpoint schemas không được nhập ngầm.
Prefix loss mặc định tắt; LoRA/full backbone training không thuộc recipe chính.

## 6. Thứ tự triển khai và readiness

1. Converter giữ response và cache audit/importer.
2. Anchor sampler/packing, grouped memory và frozen-model gradient checks.
3. Compact dense attention collector và indexer KL.
4. Một adapter trainer, checkpoint/resume, shared evaluator và profiling.
5. B200 runtime ngắn, pilot 2–5K, rồi mở rộng 50K khi acceptance/cost đạt yêu cầu.

Các tasks nằm trong [kế hoạch triển khai](superpowers/plans/2026-10-08-amr-dflash-implementation.md).
Hiện chưa có lệnh chạy main adapter training mới. Không thêm một subcommand
giả vào hướng dẫn server; các lệnh tồn tại được giữ trong phụ lục V0 dưới đây.

## Phụ lục: pipeline V0 hiện chạy được

Phụ lục giữ phương pháp cũ để replay/debug hoặc làm control. Preference
labels và candidate-verifier-logit KL không còn là main training của ý tưởng
hiện tại. Historical CPU tests chỉ xác nhận V0 trong phạm vi được ghi.

<details>
<summary>Mở hướng dẫn prepare/capture/label/train V0</summary>

### Dữ liệu cần tạo V0

AMR giữ nguyên target và pretrained DFlash-5L. Selector học từ chênh lệch
acceptance giữa các support cùng state/budget; compressor học từ logits của
target verifier. Vì vậy dữ liệu train gồm:

| Artifact | Nguồn tạo | Mục đích |
|---|---|---|
| Manifest prompt, source ID, split | Corpus regenerate hiện có | Đầu vào capture |
| Projected features và state snapshots | `capture` với snapshot target/drafter đã khóa | Context trước anchor |
| Candidate supports | `candidates` | So các tập raw token cùng budget |
| Acceptance preferences | `label` với full target verifier | Huấn luyện selector |
| Teacher logits và proposal history | Cùng lượt `label` | Huấn luyện compressor |

Corpus regenerate đủ làm nguồn prompt. Câu trả lời regenerate cuối cùng được
loại khỏi prompt; những lượt assistant trước đó vẫn thuộc lịch sử hội thoại.
AMR tạo trajectory greedy mới và supervision riêng. Tóm tắt gốc ArXiv ở
`metadata.reference_summary` được giữ làm reference; câu trả lời regenerate
không được dùng làm reference ROUGE.

Target-feature cache 50K từ Qwen3-4B đang có trên server theo xác nhận người
vận hành. Audit run 50K ngày 05/10 được ghi trong
[audit MR-DFlash](mr_dflash_gpu_experiments.md): hidden recompute fail 32/32
mẫu train và 32/32 mẫu val. Chưa có artifact audit mới để xác nhận cache đó
đã được sửa/tạo lại; importer dành cho trainer AMR mới cũng chưa có.
Sau khi audit, có thể dùng lại response JSONL và cache đạt parity; nếu cache
không đạt thì rebuild features bằng causal teacher forcing, không regenerate
autoregressive toàn bộ 50K.

### Pilot V0 đề nghị

Chọn tối đa **200 document train + 50 document validation**. Giữ nguyên split
nguồn; phân tầng ShareGPT/ArXiv bằng seed cố định. Chọn prompt sau chat template
dài ít nhất **4.097 token**, rồi giới hạn input ở **16.384 token** theo cùng
head/tail truncation với capture. Với raw budget 4.096, prompt quá ngắn thường
làm các candidate trùng nhau và không có preference để học.

Nếu ShareGPT không có đủ mẫu dài, report sẽ cho thấy mất cân bằng còn lại;
không tăng chiều dài bằng padding hay đưa response regenerate vào prompt.
Hai limit là trần số mẫu, không đảm bảo corpus có đủ mẫu đạt bộ lọc. Nếu budget
khác 4.096, đổi ngưỡng tối thiểu tương ứng và khóa config trước capture.

### 1. Tạo manifest V0 bằng tokenizer local

Chạy trong repository trên server; production dùng `python3` hệ thống thông
qua shared runtime. Master phải trỏ tới đúng snapshot target và pretrained
DFlash **5 layer, block size 16**. Tokenizer dùng khi chuẩn bị phải thuộc
snapshot target sẽ capture. Lệnh `prepare-data` chỉ load tokenizer local,
không load weights, không cần GPU và không tải tài nguyên.

```bash
cd /workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_for_text_sum-main
export FAST_INFER_MASTER_CONFIG=/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env
source scripts/common/config.sh
fast_infer_load_config amr_dflash
export FAST_INFER_PYTHON="$(command -v python3)"
"$FAST_INFER_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'
set -euo pipefail

# Thay bằng thư mục regenerate thực tế nếu corpus đang nằm ở run khác.
MR_REGENERATED_DIR=/workspace/storage-shared/nlp/dungdx4/phuc_projects/outputs/mr_dflash_phase1_50k_50_50_10k_b200_gpu0_ctx16k_out1k_20260928T132928Z/regenerated_full
AMR_PILOT_ID="amr_v0_$(date -u +%Y%m%dT%H%M%SZ)"
# File validation da loai overlap trong run do nguoi van hanh cung cap.
# Doi path neu ban chuan bi mot clean split khac.
AMR_VALIDATION_INPUT="$PWD/outputs/amr_dflash/clean/amr_pilot_20261009T180405Z/val.jsonl"
test -f "$AMR_VALIDATION_INPUT"
AMR_PREPARED_DIR="$PWD/outputs/amr_dflash/prepared/$AMR_PILOT_ID"
AMR_PILOT_ROOT="$PWD/outputs/amr_dflash/$AMR_PILOT_ID"

bash scripts/run.sh amr_dflash prepare-data \
  --train-input "$MR_REGENERATED_DIR/train.jsonl" \
  --validation-input "$AMR_VALIDATION_INPUT" \
  --tokenizer-path "$TARGET_MODEL" \
  --output-dir "$AMR_PREPARED_DIR" \
  --train-samples 200 --validation-samples 50 \
  --min-input-tokens 4097 --max-input-tokens 16384 --seed 17

export AMR_DATA_MANIFEST="$AMR_PREPARED_DIR/manifest.jsonl"
python3 -m json.tool "$AMR_PREPARED_DIR/report.json"
```

`--train-input`/`--validation-input` nhận nhiều lần nếu hai nguồn nằm trong
file riêng. Input phải là JSONL regenerate có `id`, `source` bằng `sharegpt`
hoặc `arxiv`, và `conversations` kết thúc bằng câu trả lời assistant không rỗng.
Role `human`/`gpt` cũng được chuẩn hóa. Record lỗi báo rõ file, dòng, sample ID;
không âm thầm đưa record lỗi vào manifest.

Đầu ra gồm `manifest.jsonl`, `report.json`, `preparation_contract.json` và
`token_lengths.sqlite3`. Report ghi số đọc/đủ điều kiện/được chọn theo
split và source, số bỏ vì ngắn/trùng, token-length statistics, số reference,
hash input/tokenizer và seed. `training_labels_ready=false` là đúng ở giai đoạn
này: manifest chưa có supervision AMR.

Tool quét toàn bộ ID và nội dung prompt trước khi chốt subset, để phát hiện
overlap nằm ngoài limit. Nội dung được so trên toàn lịch sử, gồm role và các
lượt assistant cũ. Nếu nguồn có `source_document_id`, `document_id` hoặc
`metadata.original_id`, các cửa sổ của cùng document cũng không được nằm ở
hai split. Tool bỏ duplicate trong cùng split, giữ nguyên JSONL đầu vào và
giới hạn số prompt giữ trong RAM theo limit/source.

Không ghi đè thư mục có sẵn. Nếu bị gián đoạn, chạy lại **cùng lệnh** thêm
`--resume`: tiếp tục quét nhưng dùng lại token lengths đã checkpoint. Resume
đòi cùng input bytes, tokenizer, seed và các limit. Muốn đổi subset/config,
dùng ID/thư mục mới. Progress audit hiện ở terminal.

### 2. Capture V0 state và features mới trên B200

Sau khi review report và spot-check vài hội thoại trong manifest, dùng GPU
đã được cấp trong master/allocation hiện tại. Preflight kiểm tra đúng model,
tokenizer, input và B200; không chạy trên host T4 local.

```bash
bash scripts/run.sh amr_dflash preflight --require-b200

bash scripts/run.sh amr_dflash capture \
  --input "$AMR_DATA_MANIFEST" --run-root "$AMR_PILOT_ROOT" \
  --max-input-tokens 16384 --max-new-tokens 256 \
  --max-states-per-document 3
```

Không thêm `--max-samples 200` ở bước này: manifest đã giới hạn số mẫu theo
split và đặt train trước validation; cắt toàn manifest có thể làm mất validation.
Giới hạn ba state/document là mức bắt đầu để kiểm tra tín hiệu/chi phí. Có thể
đổi lên sáu trong một run mới sau khi pilot cho thấy cần thêm state.

Capture lưu `documents.jsonl`, `states.jsonl`, feature bundles và
`manifest.json` khóa fingerprint target/drafter, dtype/backend, memory config,
input/output caps, prompt policy, seed và split. Snapshot model phải giữ
nguyên qua label/train. `capture --resume` yêu cầu cùng các giá trị này.
Prompt policy hiện hỗ trợ structured messages version 2; capture cũ có policy
version 1 không được resume bằng hợp đồng mới, cần run root mới khi recapture.

### 3. Tạo labels huấn luyện V0

```bash
bash scripts/run.sh amr_dflash candidates --run-root "$AMR_PILOT_ROOT"
bash scripts/run.sh amr_dflash label --run-root "$AMR_PILOT_ROOT"
python3 -m json.tool "$AMR_PILOT_ROOT/training_signal.json"
```

Kiểm tra summary của capture/label để xác nhận job hoàn tất, số documents/states
theo split đúng và không hết adaptation budget giữa chừng. Supervision nằm ở
`preferences.jsonl`, `candidate_labels.jsonl`, `teacher/*.pt`; states và
features được tham chiếu từ cùng capture run.

Trong `training_signal.json`:

- `train_selector_signal_present=true`: train có preference không hòa do
  verifier tạo; fixture tổng hợp không được đếm trong báo cáo tín hiệu.
- `train_compressor_signal_present=true`: train có teacher logits từ candidate
  thành công, chưa censor.
- `by_split` ghi số states có preference, số pairs, teacher rows và censor.
  Validation cần đủ states có tín hiệu để đánh giá; không lấy validation
  pairs để bù thiếu train pairs.

Hai cờ trên chỉ xác nhận có supervision, chưa chứng minh số mẫu đủ hoặc model
học tốt. Nếu train preferences bằng 0, xem candidate có bị degenerate do
context ≤ raw budget, reward có hòa hết, hay quá nhiều state cuối bị censor.
Chọn thêm document dài/đa dạng hoặc điều chỉnh state/candidate/config rồi
tạo run mới. Không gán reward giả và không phá tie thành preference.

### 4. Kiểm chứng train V0 ngắn trước khi mở rộng

Khi có tín hiệu thật và artifact hoàn tất, chạy 20 bước mỗi pha:

```bash
AMR_PILOT_CHECKPOINT_DIR="$PWD/checkpoints/amr_dflash/$AMR_PILOT_ID"
bash scripts/run.sh amr_dflash train-selector \
  --run-root "$AMR_PILOT_ROOT" --steps 20 \
  --checkpoint-out "$AMR_PILOT_CHECKPOINT_DIR/selector_20.pt"

bash scripts/run.sh amr_dflash train-compressor \
  --run-root "$AMR_PILOT_ROOT" --steps 20 \
  --checkpoint-in "$AMR_PILOT_CHECKPOINT_DIR/selector_20.pt" \
  --checkpoint-out "$AMR_PILOT_CHECKPOINT_DIR/compressor_20.pt"

bash scripts/run.sh amr_dflash evaluate-fixed \
  --run-root "$AMR_PILOT_ROOT" --split validation --mode amr \
  --checkpoint "$AMR_PILOT_CHECKPOINT_DIR/compressor_20.pt" \
  --output "$AMR_PILOT_ROOT/evaluation/amr_validation_20.jsonl"
```

Kiểm tra loss/gradient hữu hạn, cập nhật weights, save/load checkpoint và
same-state acceptance trên validation, so với dense/selection. Dùng cùng
max-input cap khi chạy rollout trên manifest này. Với pilot đạt yêu cầu, mới
chọn một run lớn hơn, chẳng hạn 500–1.000 document train, và tính tiếp theo
RAM, disk, GPU-hour ledger và tỷ lệ state có preference. Train V0 giữ feature
bundles/teachers trong cache RAM; số document lớn cần xem lại loader/cache trước
khi dùng hết 50k.

Giữ holdout riêng theo source ID/nội dung cho đánh giá cuối. Có thể cung cấp
`--holdout-input` và `--holdout-samples` nếu đã có file test riêng; không đổi
validation thành holdout sau khi đã dùng để chọn config/checkpoint.

Kiểm chứng local ngày 09/10/2026: **68 tests passed** cho converter, AMR contracts,
regressions và launcher. Sample tổng hợp 120 dòng chọn 20 train + 10 validation,
mỗi split cân bằng hai source, giữ 15 reference gốc ArXiv và không đưa response
regenerate cuối vào manifest. Test với Qwen/DFlash nhỏ thật trên CPU xác nhận
manifest nhiều lượt đi qua capture/candidates/verifier labeling và tạo teacher.
Các regression train dùng preferences có kiểm soát để kiểm tra gradient; không
xem đó là bằng chứng preference tự nhiên của corpus thật.

Corpus 50K và target-feature cache hiện diện trên B200 theo xác nhận người vận
hành, nhưng chưa được tải vào workspace này. Report pilot trên server là bằng
chứng về phân phối, số preference tự nhiên và chi phí thực tế của dữ liệu.

</details>
