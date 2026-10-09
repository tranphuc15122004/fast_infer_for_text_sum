# Kế hoạch triển khai AMR-DFlash — memory adapter nhẹ

Ngày tạo: **08/10/2026**. Revision: **09/10/2026**.

**Mục tiêu:** triển khai thiết kế memory adapter nhẹ đã thống nhất:
frozen target + frozen pretrained DFlash-5L, global grouped pooling nhỏ,
fine block retrieval và exact local context. Train bằng weighted draft
prediction CE + compact attention indexer KL từ regenerate/cache.

**Trạng thái:** các tasks dưới đây là phần chuyển đổi cần làm, chưa được
nghiệm thu chỉ vì V0 có file cùng tên. Code V0 vẫn chạy
capture → candidates → label → train-selector → train-compressor.
Đợt cập nhật này thay tài liệu, chưa hiện thực trainer mới.

Nguồn ý tưởng/paper story: [tài liệu chính](../../amr_dflash_paper_story.md).
Contracts triển khai: [đặc tả](../specs/2026-10-08-amr-dflash-design.md),
[dữ liệu/training](../../amr_dflash_training_data.md),
[CLI V0](../../baselines/amr_dflash.md).
[Review V0](../../reviews/2026-10-08_amr_dflash_implementation_review.md)
giữ bằng chứng sửa lỗi/CPU của pipeline cũ.

## 1. Phạm vi và những phần tái sử dụng

Giữ target verifier full-context, pretrained depth/block/weights, projected
feature space, original positions, A/G accounting, local model loading,
offline Python 3.12 và output qua JsonlWriter/ROUGE. Không sửa vendored DFlash
hoặc default dense baseline. Local chỉ CPU; L1/pilot chạy server B200.

Tái sử dụng sau regression checks:

- model.py: frozen projection và five-layer compact-context forward.
- inference.py/evaluation.py: greedy verifier, pending anchor, cache crop và parity.
- artifacts.py/checkpoint.py: atomic artifacts, identity/fingerprints, mở schema mới.
- common runtime/config/schema và AMR dispatcher: giữ V0 compatibility.

Phần thay: prompt-only training preparation, candidate/preference teacher,
token-level selector fit, full-rank fixed slots, candidate-logit KL main loss,
unbounded in-RAM state/teacher cache và phase-specific V0 trainer.

LoRA, full backbone fine-tuning, native layer-alternating HCA/CSA,
adaptive block size và speculative sampling không thuộc triển khai chính.

## 2. Task graph và nguyên tắc kiểm chứng

~~~mermaid
flowchart LR
    T1["T1 contracts/config"] --> T2["T2 frozen backbone"]
    T1 --> T4["T4 response + cache + anchors"]
    T2 --> T3["T3 memory interface"]
    T4 --> T5["T5 compact teacher"]
    T5 --> T6["T6 block indexer"]
    T3 --> T7["T7 grouped pooling"]
    T6 --> T8["T8 loss + evaluator"]
    T7 --> T8
    T4 --> T9["T9 adapter trainer: integration"]
    T8 --> T9
    T9 --> T10["T10 cache + profiling"]
    T10 --> T11["T11 CLI/runtime"]
    T11 --> T12["T12 B200 pilot + ablations"]
~~~

Các semantic tests phải kiểm tra causal visibility, label shift, optimizer
freeze và checkpoint consistency. Không viết test chỉ xác nhận một flag/config
được gán đúng. T9 là assembled training integration duy nhất chạy L0/L1;
T12 là scientific validation, không là một trainer khác.

Các filenames bổ sung dưới đây là **dự kiến**, không phải module/CLI đã tồn tại.
Giữ tests hiện có trong tests/; không hướng dẫn chạy src/AMR_DFlash/tests
khi thư mục đó chưa được tạo.

## T1. Config, versions và accounting

Sửa core.py/config.py, thêm adapter schema và các semantic contracts.

- [ ] Config m=4, c=64, K=256, L=128, index dim 64, pooling hidden 32.
- [ ] Assert L>=c, valid group sizes, supported depth/block/dtype/mode.
- [ ] Context budget tính actual unique raw + completed global entries;
      live block 16 ngoài budget.
- [ ] Schema tách adapter-data/teacher/checkpoint khỏi V0; không implicit conversion.
- [ ] Check config serializable, fingerprint stable, selected groups/positions
      unique và padding/future groups bị từ chối.
- [ ] Log actual trainable params; optimizer whitelist là pooling/indexer/gate.

Nghiệm thu: config mới không được V0 hiểu nhầm; no-future và budget tests pass.

## T2. Frozen backbone và dense identity

Sửa model.py/runtime.py, giữ local pretrained loading.

- [ ] Freeze target, embedding/head, fc/norm và toàn bộ năm draft layers.
- [ ] All-raw, global-off forward khớp dense baseline với cùng backend/dtype,
      gồm logits tolerance và argmax.
- [ ] Projected features không chạy fc/norm lần thứ hai.
- [ ] Pooling gradient xuyên frozen draft/head khác zero và hữu hạn;
      backbone gradients None và hash weights không đổi sau optimizer step.
- [ ] Bypass không khởi tạo memory/index hoặc thêm teacher branch vào timing.

Nghiệm thu: attach adapter không đổi baseline; test không download model.

## T3. Position-safe shared memory interface

Sửa memory.py và DFlash adapter interface.

- [ ] Fine selection trả groups rồi gather raw features; không dùng mean value
      làm raw feature.
- [ ] Raw/local union, loại trùng/sort, giữ absolute positions.
- [ ] Global entries midpoint/RoPE riêng; finite shared additive bias.
- [ ] Giữ full live-block K/V và bidirectional attention; không prune anchor/masks.
- [ ] Test completed-group visibility cho anchors gần boundaries;
      perturb future features không đổi working memory.
- [ ] Multi-anchor packing giữ mỗi block memory/positions/mask độc lập.

Nghiệm thu: compact bank được thực sự gather; dense N-key mask không là sparse executor.

## T4. Regenerate converter, cache audit và anchor store

Thêm adapter-specific data preparation/cache importer; V0 prepare-data giữ
semantics prompt-only.

- [ ] Giữ final regenerated response, tokenize prompt/response boundary đúng.
- [ ] Lock source/full-history split trên toàn corpus; không lấy validation để train.
- [ ] Cache import so token IDs/positions/template/truncation, target/layers và
      frozen projection fingerprint; mismatch fail có report.
- [ ] Raw features project offline một lần; audit recompute subset trước import.
- [ ] Manifest lưu response EOS/decoding metadata và references ArXiv riêng.
- [ ] Random anchors train, fixed anchors validation; valid labels sau anchor,
      không dùng features của anchor pending hoặc proposal suffix.
- [ ] Lazy per-document loading, bounded cache/LRU/prefetch; không clone whole bank
      cho từng anchor.
- [ ] Atomic shard/index checkpoint; interrupted preparation resume từ artifact
      đã hoàn tất và cùng contract.

Nghiệm thu: token/cache audit pass; converter không làm response biến thành
conditioning future; V0 prompt manifests bị từ chối cho main prediction training.

## T5. Compact dense-attention teacher

Thay vai trò candidate labeling trong phương pháp chính bằng teacher collector
train-only; giữ V0 candidates/label làm legacy.

- [ ] Dense DFlash cùng prefix/anchor/masks, collect theo anchor chunks.
- [ ] Aggregate heads/layers/valid proposal queries với early-position weights,
      group raw attention mass và loại local/live/ineligible groups.
- [ ] Normalize đúng domain; zero mass/empty domain skip có reason.
- [ ] Save compact group distribution + teacher/visibility/anchor fingerprints,
      không save full attention matrices hoặc full-vocabulary teachers toàn corpus.
- [ ] Collector backend/timing được ghi; không dùng instrumentation cho speedup.
- [ ] Teacher anchors thuộc train; mọi overlap/duplicate/fingerprint mismatch fail.
- [ ] Report teacher coverage theo source/length; collector resumable per shard.

Nghiệm thu: ranking supervision có provenance, không được gọi là verifier utility.

## T6. Indexer nhỏ theo draft block

Sửa/reuse query scorer trong core.py, thêm group indexing.

- [ ] Fine keys từ mean pooling cố định d→64; query từ anchor/latest/recent H.
- [ ] Một score/support mỗi block, dùng chung năm layers.
- [ ] Eligible groups wholly outside local guard và trước anchor;
      min(K,valid groups), stable tie order.
- [ ] KL vào scores truyền gradient; hard gather không được gán gradient giả.
- [ ] Main sampler giữ teacher-labeled anchors, quota pilot 25% khi đủ;
      log actual quota và skip reasons.
- [ ] Toy two-state ranking học hai supports khác nhau; compare fixed/random
      cùng memory budget.

Nghiệm thu: scorer tiếp tục được supervision trong main training.

## T7. Grouped scalar weighted pooling

Thay learned fixed slots/full-rank value projection trong recipe chính.

- [ ] MLP d→32→1 tạo scalar scores, values dùng nguyên projected H.
- [ ] Uniform init, completed groups căn absolute position 0, L>=c.
- [ ] FP32 log-sum-exp pooling, completed-entry cache + incomplete-tail accumulator.
- [ ] Streaming/full-build parity qua chunks, group boundaries và varying N.
- [ ] Chỉ append processed committed features; rejected proposals không cập nhật.
- [ ] Invalidate cached memory khi model weights đổi; training rebuild có graph.
- [ ] Learned global bias finite -4; global-off bỏ branch thật cho dense identity.

Nghiệm thu: gradient nonzero, memory growth N/c đúng, không resident-bank reduction claim.

## T8. Prediction loss và shared evaluator

Thêm loss/evaluation core dùng cả checkpoint/in-memory.

- [ ] Weighted CE trên tối đa 15 valid labels, gamma=7, đúng label shift/EOS/padding.
- [ ] Indexer KL trên eligible groups và samples có teacher; frozen teacher.
- [ ] Prefix loss mặc định off; teacher-output distillation optional subset
      không là bắt buộc của pipeline.
- [ ] Prediction eval toàn fixed validation anchors mỗi 50 optimizer steps.
- [ ] Actual full-validation greedy rollout mỗi 500 steps và cuối phase.
- [ ] Teacher-token prefix match có label riêng khi chưa xác minh greedy alignment.
- [ ] Checkpoint/in-memory cùng metrics; aggregate sum(G)/sum(T), tách A_raw/A_committed.
- [ ] Missing/corrupt checkpoint, empty validation, NaN, fingerprint mismatch
      hoặc partial run fail; progress/phase/heartbeat/error artifact rõ.
- [ ] Output schema và ROUGE reference đúng policy.

Nghiệm thu: loss giảm không thay actual acceptance; evaluator không im lặng
suốt một phase dài hoặc báo partial evaluation là complete.

## T9. Unified adapter trainer [INTEGRATION]

Thêm một trainer warm-up/main; optimizer chỉ cập nhật adapter.

- [ ] Warm-up pilot <=200 steps: c=32, K<=1024; main c=64, K=256,
      chuyển budget có validation.
- [ ] Một optimizer cho pooling/indexer/gate; CE + beta*indexer KL, beta pilot 1.
- [ ] Packing ban đầu tám anchors/document, nhiều blocks/forward; chunk logits
      theo VRAM, gradient accumulation không đổi loss normalization.
- [ ] Deterministic train sampling, teacher quota và fixed val anchors.
- [ ] Checkpoint adapter/optimizer/scheduler/RNG/sampler/phase/budget/eval history;
      atomic completion và resume khớp uninterrupted tiny CPU run.
- [ ] Log losses, gradient norm, params, blocks/s, step/cache times, VRAM, GPU-hours.
- [ ] Full validation theo T8; không tự mở backbone khi retention kém.
- [ ] L0 static + meaningful CPU integration tests.
- [ ] L1 trên B200 real-data trong khoảng năm phút, profile forward/backward/I/O;
      không chạy GPU local.

Nghiệm thu: frozen-weight hash giữ nguyên, cached data đi xuyên tới loss/backward/
step/eval/save/restore. L1 pass chưa là scientific success.

## T10. Inference cache, profiling và calibration

Sửa/reuse cached inference engine, thêm grouped memory lifecycle/profile.

- [ ] Cache fine index keys/completed global entries; update committed tail.
- [ ] Original positions không lấy từ compact bank length.
- [ ] Pending anchor, reject crop, EOS/cap và target AR parity qua nhiều rounds.
- [ ] Force dense/AMR cùng backend/verifier/workload; đo build/gather/update/indexer/
      draft/verification/TTFT/E2E và memory breakdown.
- [ ] Full resident raw/KV/features bank được báo; không đồng nhất keys với VRAM.
- [ ] Short-context crossover đo sau model training; fingerprint calibration.
- [ ] Force mode không bị auto gate đổi; ngoài vùng calibration có fallback reason.

Nghiệm thu: measured physical execution, không claim attention-key ratio là speedup.

## T11. CLI, config và docs migration

Mở CLI adapter mới sau khi T4–T10 có code; giữ V0 subcommands.

- [ ] Names/schema/version tách V0, help rõ method/training recipe.
- [ ] Master env/offline Python 3.12, selected interpreter và allocation được tôn trọng.
- [ ] Preflight kiểm cache contract/teacher coverage/optimizer freeze/data splits.
- [ ] Không nhận V0 checkpoint như adapter checkpoint; budget mới không đổi
      num_slots/raw_budget âm thầm.
- [ ] Document lệnh **chạy được** sau khi launcher tồn tại, cập nhật readiness.
- [ ] Regression V0/DFlash dispatcher/output schema/runtime.

Nghiệm thu: smoke/full flags đúng scope; không thêm fake commands vào docs
trước implementation. Batch-1 V0 config không tự trở thành multi-anchor trainer.

## T12. B200 pilot và paper evidence

- [ ] Audit/baseline trước; train pilot 2–5K documents nếu eligible đủ,
      validation 50–100 docs khóa riêng; báo source/context coverage thật.
- [ ] Actual acceptance retention gate đề xuất 95% ở bins có dense A>0;
      nếu không đạt, tăng budget/đơn giản hóa trong run mới hoặc no-go.
- [ ] Force AMR ở final budget; total decode throughput > dense là pilot goal,
      short dense routing giữ parity; không dựa warm-up dense-like budget.
- [ ] Local+fine, local+global, full adapter, mean/learned pooling,
      fixed/random/learned indexer, raw/compressed fine entries, warm-up/no-warm-up.
- [ ] Matched-total-key và matched-adaptation-compute comparisons riêng;
      prefix objective so dense tương ứng nếu mở extension.
- [ ] Tổng adaptation cost gồm cache audit/recompute cần thiết, teacher,
      train, validation/calibration; report wall time và GPU-hours.
- [ ] Khóa config/checkpoint/gate trước holdout; paired document IDs/prompt hashes,
      failed samples và document-bootstrap CI.
- [ ] Expand 50K sau pilot/runtime checks; no-gain là kết luận hợp lệ.

Nghiệm thu: throughput, acceptance, parity và chi phí có raw server artifacts.
Không hứa thời gian train/speedup trước profile.

## 3. Tác động so với kế hoạch V0

| Phần cũ | Revision hiện tại |
|---|---|
| T4 trajectory capture training | Response/cache/anchor preparation |
| T5 candidate search/preferences | Compact dense-DFlash attention teacher |
| T6 token preference selector | Block indexer + auxiliary KL |
| T7 fixed slots/full-rank transforms | Grouped scalar weighted pooling |
| T8 candidate-verifier-logit KL | Weighted prediction CE + explicit indexer KL |
| T9 selector/compressor train riêng | Unified adapter-only optimizer, multi-anchor/resume |
| T12 acceptance-first interventions | Retention/cost trade-off và lightweight adaptation evidence |

Các artifacts V0 không bị xóa và có thể replay theo
[phụ lục dữ liệu](../../amr_dflash_training_data.md#phụ-lục-pipeline-v0-hiện-chạy-được).
Các CPU/B200 kết quả cũ chỉ được gắn với schema/model thật của run đó.
Khi đóng task, ghi files/commands/exit codes/artifact/limitations; chỉ stage
đúng files nếu được thực hiện commit, không gom working-tree changes khác.
