# Findings — GroundSync experiment

## Repo hiện tại

- Git worktree sạch trên nhánh `main`.
- `src/analyze` hiện chỉ có `full_infer/profile_qwen3_long_summary.py`, README
  và kết quả profiler Qwen3 cũ; chưa có pipeline GroundSync/hypothesis analysis.
- Repo đã có các helper chung ở `scripts/common`, profiler Qwen3 ở
  `src/analyze/full_infer`, và wrapper `scripts/run_qwen3_long_profile.sh`.
- Dữ liệu plug-and-play đã có: `data/representative_100/` gồm GovReport,
  Multi-News, CNN/DailyMail, XSum; `data/longbench_200/` có GovReport,
  Multi-News, QMSum và các task khác.

## Proposal cần triển khai

- H1/E1: source-utilization state có temporal persistence trên canonical target
  AR trace.
- H2/E2: source-state drift/transition dự báo first speculative rejection.
- H3/E3: tín hiệu còn predictive sau entropy, draft confidence, acceptance
  history, position, sentence boundary và copyability controls.
- H4/E4: oracle source-state horizon tạo utility về committed tokens/sec hoặc
  cost trên mỗi committed token.
- H5/E5: horizon có thể dự đoán online từ tín hiệu hiện tại/quá khứ.
- Attention sink/position bias phải là confounder được đo và kiểm soát, không
  được gọi raw attention là ground truth.

## Tín hiệu từ repo

- `docs/qwen3_long_profile.md` mô tả profiler Qwen3-4B target-only và cấu hình
  local-files-only; artifact cũ nằm ở `src/analyze/full_infer/results`.
- `docs/model_baseline_matrix.md` ghi Qwen3-4B là cấu hình hiện có cho EAGLE-3
  và DFlash; `docs/baselines/eagle3.md` mô tả chi phí/Qwen3 pairing.
- `scripts/eagle3_infer_qwen3.py` và `scripts/run_eagle3_qwen3.sh` là ứng viên
  để kiểm tra speculative trace nếu model/drafter cache tồn tại.

## Điểm đã chốt trước khi code

- Experiment directory là `src/analyze/groundsync`, đúng yêu cầu artifact.
- Protocol ưu tiên GovReport/representative và có synthetic CPU smoke để kiểm
  tra evaluator trước khi đụng model.
- H4 chỉ được kết luận speed khi có timing đo theo từng `k`; nếu thiếu thì chỉ
  giữ acceptance-only.

## Khảo sát bổ sung

- Profiler hiện tại dùng Transformers target-only, greedy, `output_attentions`/
  `output_hidden_states` chưa được triển khai; cần một pipeline trace riêng để
  giữ attention/hidden-state theo output position.
- EAGLE script hiện tại là benchmark decode-only và yêu cầu CUDA; không phù hợp
  làm nguồn duy nhất cho H2 trên máy CPU. Có thể dùng một controlled speculative
  trace bằng target-vs-draft greedy logits trước, sau đó chạy E2E nếu có GPU và
  checkpoint draft tương thích.
- `src/analyze/full_infer/results` là artifact cũ, không nên ghi đè; experiment
  mới cần thư mục riêng.
- Repo có cấu hình EAGLE Qwen3-4B nhưng chưa có bằng chứng trong khảo sát rằng
  Qwen3-1.7B/0.6B hoặc drafter tương ứng đã được cache.

## Runtime hiện tại

- Local `.venv` là Python 3.12.13 với torch `2.11.0+cu130`, Transformers
  `5.12.1`; trong sandbox CUDA không được expose. Khi tắt venv và chạy trực
  tiếp `/home/tuantb/miniconda3/bin/python3` (Python 3.13, torch `2.6.0+cu124`),
  host thấy Tesla T4, driver `550.163.01`, CUDA `12.4` và
  `torch.cuda.is_available()=True`.
- Cache hợp lệ Qwen3-4B đã được tìm thấy tại
  `/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c`;
  config là `Qwen3ForCausalLM`, 36 layers và 3 shard safetensors.
  `/home/tuantb/models/Qwen3-4B_eagle3` chỉ là EAGLE head
  `Eagle3LlamaForCausalLM`, không được dùng làm canonical target.
- Qwen3-0.6B ban đầu chưa có nên đã tải vào `/home/tuantb/models/Qwen3-0.6B`;
  config là `Qwen3ForCausalLM`, 28 layers và safetensors đã được mở/kiểm tra
  thành công. Qwen3-1.7B chưa cần tải.
- `torch.cuda.is_available()` vẫn false vì driver local không tương thích;
  discovery model-backed dưới đây chạy CPU. H4/E2E throughput trên server CUDA
  vẫn phải tách khỏi controlled acceptance.

## Kết quả thực thi 2026-08-29

- `synthetic-20260829-final-v2`: evaluator chạy đầy đủ 12 target fixture và 120
  controlled rows, sinh JSONL/CSV/4 PNG/Markdown. Đây là test đường ống, không
  phải bằng chứng Qwen.
- `qwen3-local-smoke-final`: đã thử Qwen3-1.7B target + Qwen3-0.6B draft với
  `local_files_only=True` trên CPU; snapshot không có, nên H1–H5 đều
  `UNAVAILABLE`. Không dùng `/home/tuantb/models/Qwen3-4B_eagle3` vì đó là
  EAGLE head một layer, không phải canonical Qwen3 target.
- Final validation fresh: `34 passed`, `compileall` pass và `git diff --check`
  pass.

## Kết quả model-backed mới nhất

- `qwen3-4b-06b-actual-smoke-20260829`: 1 tài liệu, target Qwen3-4B và draft
  Qwen3-0.6B đều chạy; verifier target đo được timing theo block `k`. Đây là
  pipeline smoke, chưa đủ coverage để kết luận H1–H5.
- `qwen3-4b-cnn10-target-20260829`: 10 tài liệu CNN/DailyMail, 80 target token
  steps và 20 controlled proposals (`max_k=2`, hai prefix mỗi tài liệu). Report:
  - H1 `FAIL` theo gate định trước: persistence excess `0.00306`, bootstrap
    95% CI `[0.00175, 0.00442]`, thấp hơn ngưỡng `0.02` dù adjacent similarity
    nhỉnh hơn shuffle null.
  - H2 `FAIL`: high-drift rejection rate `0.20` thấp hơn low-drift `0.40`,
    chênh lệch `-0.20`.
  - H3 `UNAVAILABLE`: document-split/predictor chưa đủ coverage và class
    variation hữu ích.
  - H4 `UNAVAILABLE`: discovery dùng draft-only, không có verifier timing cho
    mọi row; smoke riêng chỉ xác nhận đường đo timing hoạt động.
  - H5 `UNAVAILABLE`: grounding horizon không tạo nhãn dương ở threshold hiện
    tại, nên chưa đánh giá được predictor.
- Kết luận hiện tại: hypothesis chưa được kiểm chứng thành công trên run
  model-backed này; H1/H2 có bằng chứng không ủng hộ claim dưới gate hiện hành,
  H3–H5 cần coverage/điều kiện đo bổ sung. Synthetic PASS không phải bằng chứng
  Qwen.

## Kết quả GPU T4 mới nhất

- Run `qwen3-4b-gov25-gpu-all-20260829` dùng Python miniconda ngoài venv,
  `cuda:0`, FP16, Qwen3-4B target và Qwen3-0.6B draft trên 25 GovReport.
  Target đạt `25/25`, speculative đạt `50/50`; context sau tokenizer là
  `2.609–22.630` token. Tất cả 50 row có draft/verifier timing.
- H1 `FAIL`: adjacent similarity `0.98997`, shuffle null `0.97113`, excess
  `0.01884`, bootstrap 95% CI `[0.01598, 0.02188]`; CI không vượt gate
  `0.02`.
- H2 `FAIL`: rejection low-drift `1.00`, high-drift `0.8333`, chênh lệch
  high-minus-low `-0.1667`.
- H3 `UNAVAILABLE`: document split đã có 15/5/5 documents nhưng test labels
  chỉ có một class, nên AUROC/gain không hợp lệ.
- H4 `FAIL`: timing thật đã đo; fixed policy đạt `0.0003616` committed-token/ms,
  oracle `0.0002885`, speed gain `-0.2021`, không đạt mục tiêu `>= 0.08`.
- H5 `UNAVAILABLE`: có 375 horizon rows nhưng grounding horizon không tạo
  class variation dương ở threshold `0.2`.
- Run đầu tiên với eager prefill bị OOM vì T4 `sm75` không được native Flash
  SDP của torch cu124 hỗ trợ. Adapter đã được sửa và test hồi quy thêm
  chunked causal prefill; run GPU chính sau đó không OOM. Đây là sửa hạ tầng đo,
  không phải thay đổi hypothesis.

## Mở rộng protocol 2026-08-30

- TDD regression suite cuối: **50 passed**, compileall pass. Bổ sung
  calibration positional prior 32 bins, chunk/sink sensitivity, first-reject
  hazard, position-adjusted hazard coefficient với 2.000 document-bootstrap
  resamples, H3 controls, adaptive entropy/history, true-cost policy và H5
  threshold selection train/dev.
- `qwen3-4b-gov100-gpu-protocol-20260830`: 100 GovReport target requests,
  99 ok/1 OOM; 99 draft-only proposals start=1/kmax=8; timing test-ID subset
  11 ok/1 OOM, 10 rows phủ đủ k=8.
- `qwen3-4b-cnn100-gpu-protocol-20260830`: CNN/DailyMail target 100/100 ok,
  100 controlled proposals và 12 timing rows phủ đủ k=8; dùng kiểm tra
  cross-regime H1–H5.
- GovReport H1 no-sink excess 0,023892 (CI lower 0,021996) nhưng calibrated
  CI lower 0,018257; CNN CI lower 0,010580. Composite H1 FAIL.
- H2 GovReport 99 proposals: drift high-minus-low rejection -0,022857; hazard
  coefficient `-0,0657`, CI document-bootstrap `[-0,0664; -0,0535]`, FAIL.
  CNN/DailyMail 100 proposals: coefficient point `0,00054`, CI `[0,0181;
  0,0301]`, PASS ở regime này. Hai regime khác chiều nên chưa có claim tổng
  quát.
- H3 99 rows, split 59/19/21 documents: baseline/full cùng AUROC 1, gain 0;
  full cải thiện log-loss/Brier nhưng không đạt incremental AUROC gate.
- H4 10 rows timing: grounding horizon oracle 0,0002649 token/ms so với fixed
  k=8 0,0004758, speed gain -0,4433; true-cost hindsight oracle 0,0005333.
- H5 chọn threshold 0,05 từ train/dev, predictor test AUROC 0,35; predicted
  policy 0,0002969 thấp hơn fixed k=4 0,0004111. Grounding oracle chậm hơn
  fixed nên oracle-gain recovery để None, decision UNAVAILABLE.
- CNN/DailyMail H3 no-sink AUROC gain `-0,0267` (calibrated sensitivity
  `+0,0267`), H4 speed gain `-0,4386`, H5 predictor AUROC `0,6875` nhưng
  oracle horizon vẫn chậm hơn fixed; H5 UNAVAILABLE.
- E0 position-relocation Qwen3-4B fixture gồm 3 case cùng evidence ở đầu/
  giữa/cuối. Raw mass lần lượt `0,5029/0,1170/0,1929`, no-sink lần lượt
  `0,5185/0,2297/0,2296`; no-sink giảm nhưng không loại bỏ position
  confounder. Artifact ở `results/e0-position-relocation-qwen3-4b-20260830`.
- Multi-start controlled run với `start=1,6,11,16` tạo 396 GovReport và 400
  CNN/DailyMail proposal rows. GovReport H2 FAIL (CI `[-0,0901;-0,0841]`),
  H3 AUROC gain `-0,0053`; CNN H2 PASS (CI `[0,0061;0,0106]`), H3 gain
  `+0,0086`. Không có verifier timing nên H4 unavailable và H5 inconclusive
  ở hai run.

## P0 decision extension 2026-09-02

- User yêu cầu kiểm định quyết định: corrected H2 dùng within-block transition;
  corrected H4 coi horizon không thấy transition là `Kmax`; oracle ladder thêm
  `k=0` và `k=16`; first-token admission oracle; within/across-round burstiness.
- Artifact phải co-locate dưới `src/analyze/groundsync/`. Thiết kế được thêm vào
  `plans/2026-08-29-groundsync-hypotheses-design.md` và implementation plan
  được mở rộng trong file plan hiện tại.
- Timing hiện tại không có `autoregressive_time_ms` và chỉ phủ `max_k=8`, nên
  không được suy diễn chi phí `k=0/16` từ artifact cũ; cần runner mới và manifest
  riêng. Existing multi-start có thể dùng cho persistence nhưng không có verifier
  timing.
- CUDA kiểm tra ngoài sandbox bằng `/home/tuantb/miniconda3/bin/python3` trả
  `torch.cuda.is_available()=True`, Tesla T4/driver 550.163.01/CUDA 12.4;
  trong sandbox cùng interpreter trả False. Run GPU phải dùng execution escalated
  ngoài venv.

## P0 source-of-truth — 2026-09-02

- Raw P0 acceptance/timing/multi-start và analyzer report được co-locate dưới
  `src/analyze/groundsync/`. Source-of-truth cuối là
  `results/p0-decision-final9-20260902/`; report toàn bộ là
  `p0_final_report_2026-09-02.md`.
- Corrected H2 dùng within-block transition drift và document-bootstrap CI:
  GovReport beta `0.24378`, CI `[0.01101, 0.59689]` PASS; CNN/DailyMail beta
  `0.09078`, CI `[-0.29372, 0.36339]` FAIL; cross `MIXED`.
- Corrected H4 tune threshold train/dev rồi test cùng common timing population:
  GovReport threshold `0.01`, gain so với best available `-5.24%`; CNN threshold
  `0.005`, gain `-33.40%`; cả hai FAIL. Adaptive-only comparison dương nhưng
  fixed baseline nhanh hơn và là baseline deploy hợp lệ.
- Oracle ladder `k={0,2,4,8,16}` có headroom O3 lần lượt `+72.3%` GovReport và
  `+23.5%` CNN/DailyMail, chứng minh ceiling/opportunity chứ chưa chứng minh
  online policy. First-token admission recovery tốt nhất `73.8%` GovReport và
  `30.3%` CNN/DailyMail; cross `MIXED`.
- P0-5 dùng 9 start/document (`1,4,7,10,13,16,19,22,25`), 891 GovReport và
  900 CNN rows. Sau document-bootstrap, GovReport có CI dương ở delta=1 và
  CNN/DailyMail ở delta=4; within-block ratio lần lượt `1.5623` và `1.0452`;
  P0-5 PASS theo gate định trước.
- Overall P0 decision: `NO_GO_GROUNDSYNC_GENERAL; conditional BurstSpec
  follow-up only where admission passes`. Chưa chạy P1/strong-drafter/Multi-News/
  production serving; đây là giới hạn có chủ ý sau P0.

## P1/P2 source-of-truth — 2026-09-02

- P1 predictor code/artifact: `p1_predictor.py` và
  `results/p1-cheap-admission-20260902/`. Dùng bốn feature causal có trong
  schema; GovReport recovery 0% FAIL, CNN/DailyMail 71.2% PASS, Multi-News
  INCONCLUSIVE vì test 2 documents; cross MIXED.
- Multi-News P0 confirmatory dùng 50 target/spec, 10 timing và 450 multistart
  rows. H2 coefficient 0.2590 nhưng CI `[-0.2339, 0.7848]`; H4 -45.91%; O3
  +43.69%; admission k4 77.73%; P0-5 FAIL vì persistence CI đều cắt 0.
- EAGLE-3 Qwen3-4B head đã smoke/full chạy được với target Qwen3-4B trên T4;
  results `p1p2-eagle3-{gov50,cnn50,multinews50}-20260902`. Strong burstiness
  FAIL với h1/later `0.455/0.527/0.513`; persistence không có CI dương đầy đủ.
- Direct paired E2E speedup lần lượt `1.8185x/1.9055x/1.8137x`, exact-match
  `0.96/0.98/0.94`; P2 direct FAIL guardrail exact-match 1.0. Đây là direct
  model timing, chưa phải vLLM/API serving.
- Serving preflight `results/p2-serving-preflight-20260902/` ghi UNAVAILABLE vì
  server mount thiếu; không điền số liệu serving từ direct run.
- Kết luận hợp nhất: `NO_GO_GROUNDSYNC_GENERAL` và
  `NO_GO_BURSTSPEC_GENERAL`; oracle ladder chỉ chứng minh hindsight ceiling,
  không chứng minh policy online/deployable.

## SyncSpec-v1 reading — 2026-09-02

- `src/SyncSpec/SyncSpec_v1_design_complete.md` là living design doc revision
  v1.1, mục tiêu lossless speculative decoding tương đối với full-context
  target model.
- Phần đầu định nghĩa objective là committed-token utility/CTC, không tối ưu
  acceptance rate riêng lẻ. Decomposition hiện tại gồm exact target prefill,
  long-context target-memory reuse, parallel diffusion candidates,
  source-coherent selector, prefix-survival head, analytical cost controller,
  batch/context-aware profiles và exact target verification.
- Pipeline round hiện được mô tả theo state gồm target KV/hidden state, source
  n-gram index, source memory bank, acceptance/runtime history và hardware
  profile; sau pre-draft gate là diffusion top-M, selector, survival estimate,
  controller chọn K và verifier target.
- Conversation share chưa đọc được bằng web reader: `Cache miss` khi mở URL.
  Đây là lỗi truy cập nguồn ngoài, không phải nội dung conversation đã được
  xác nhận là rỗng.

### Conversation share đã giải mã

- Read-only HTML request ngoài sandbox thành công; title là `Đề xuất
  DiffuRoute`. React payload parse được thành danh sách 11.064 phần tử, trong
  đó có các message dài tại indices `397, 576, 860, 1518, 1530, 2197, 3078,
  3559, 3707, 4618, 5611` (cùng các tool/query records).
- Chronology chính: GroundSync bị đánh giá không robust → pivot sang
  DiffuRoute (utility-guided context-adaptive diffusion) → SyncSpec được mở
  rộng thành framework co-design cross-phase → thêm candidate lattice/source
  selector → survival + cost controller → formal v1.1.
- Message cuối trước v1.1 nhấn mạnh không train ngay; cần P0–P4/oracle trước.
  Message v1.1 sau đó bổ sung formal contracts, tách `K_d`/`K_v`, exact
  algorithms, training objectives, serving và reproducibility.
- Một tool message chứa script sửa file `/mnt/data/SyncSpec_v1_design.md` và
  gặp `PermissionError`; không được coi đó là thay đổi thành công lên repo.

### Design sections 1–29 đã đọc

- Target path là full-context exact prefill/KV; mọi approximate/context
  selection chỉ ở drafter. Source n-gram index (n=2..6) chỉ rerank candidate
  đã có, không sinh token mới.
- Drafter khuyến nghị DFlash2-style block diffusion (pilot 2–3 layer, full
  khoảng 5), shared target embedding/LM head, target feature conditioning,
  bounded source memory (chunk 128, local window 256, retrieve R=8), dynamic
  causal convolution và candidate lattice Top-M=16.
- Selector dùng diffusion logit + predecessor bilinear coherence + gated
  source n-gram continuation; V1 tuần tự nhẹ qua top-M sets, greedy hoặc
  normalized stochastic proposal. Survival head dự đoán hazard để tạo dãy
  đơn điệu `S_j=P(A>=j)`, không trực tiếp dự đoán K.
- Controller chọn verification budget từ expected committed tokens /
  measured `T_D+T_S+T_V(k,L,B)`, có pre-draft AR/spec gate và AR fallback.
  Exact verifier luôn dùng full target KV; greedy dùng longest matching prefix,
  stochastic dùng rejection correction `min(1,p/q)`.

### Design sections 30–62 đã đọc

- Training tách 4 stage: target trajectory generation; diffusion backbone;
  source-coherent selector; on-policy survival; joint fine-tuning chỉ optional.
  Target-generated trajectory là teacher chính, không trực tiếp imitate human
  reference; dùng datasets CNN/DM, XSum, GovReport, Multi-News, QMSum để phủ
  extractive/abstractive regimes.
- Diffusion training ưu tiên candidate Recall@M với random anchors,
  prefix-weighted CE/KL và Anchor-Offset positional training. Selector freeze
  backbone lúc đầu, teacher token ngoài Top-16 chỉ được thay tạm trong early
  training, validation không replacement; giảm teacher forcing dần.
- Survival labels là `1[A>=j]`, cần calibration (ECE/Brier); tuyệt đối không
  train trực tiếp lớp phân loại K. Controller dùng survival + measured cost.
- Pilot gates trước full training: P0 Recall@1/4/8/16 và oracle headroom; P1
  source n-gram recoverability; P2 reranking oracle; P3 adaptive-K oracle với
  mục tiêu khoảng 10% headroom; P4 kiểm tra phụ thuộc context/batch.
- H1–H5 bản conceptual: candidate headroom; source selector recovery;
  adaptive verification utility; regime dependence; cross-phase synergy.
  Lossless chỉ được quy cho target exact verification và standard speculative
  acceptance/correction. Roadmap triển khai là A (oracle không train) → B
  selector → C survival/controller → D long-context drafter → E serving.

### Design revision v1.1 sections 63–86 đã đọc

- v1.1 chính thức tách action thành `(K_d, K_v, R)`: `K_d` phải quyết định
  trước khi trả diffusion cost; `K_v <= K_d` quyết định sau khi có candidate
  và survival. `K_d=0` là AR fallback. Pre-gate dùng calibrated empirical
  table/lightweight state model; post-gate dùng `argmax` committed progress /
  measured latency trên profile rời rạc.
- Drafter contract: same-width shallow model `d_D=d_T`, block tensor
  `[B,K_d,d]`; self-attention bidirectional trong block, dynamic causal conv,
  cross-attention đến target anchor/recent ring buffer/selected source memory,
  rồi MLP. Không tạo full independent source KV cache.
- Source memory pooling là compressed semantic evidence; n-gram index là exact
  lexical evidence. Retrieval được phép imperfect và raw target attention chỉ
  là analysis feature, không phải supervision chính.
- Selector phải phát ra proposal distribution chuẩn hóa `q_j` trên Top-M để
  stochastic speculative correction hợp lệ. Main selector loss gồm hard
  target alignment + optional target-distribution KD trên candidate set; n-gram
  feature vector dùng longest suffix/count/continuity và cheap lexical flags.
- Survival cho greedy dùng hard `1[A>=j]`; stochastic có soft survival từ tích
  acceptance probabilities. Survival features nhận tín hiệu từ tất cả phase.
  Batch-aware scheduler giới hạn finite profiles để tránh phá kernel/batch
  regularity; exact algorithm nêu rõ full target verify, commit correction và
  rollback uncommitted KV.

### Design revision v1.1 sections 87–105 đã đọc

- Verification phải là transaction: ghi nhớ committed KV length, chỉ giữ
  accepted/correction state và rollback toàn bộ speculative suffix chưa commit.
- Stage 0–3 có pseudocode rõ: target trajectories dùng đúng prompt/tokenizer,
  target-generated output (human reference chỉ evaluation), diffusion →
  selector → on-policy survival; pre-gate calibration chỉ sau khi post-draft
  chạy được, ưu tiên empirical table/isotonic trước MLP/RL.
- Correctness tách greedy exact (`accelerated_token_ids == vanilla`) và
  stochastic exact (normalized q, residual correction, statistical tests).
  Profiler phải tách target AR/draft/selector/survival/verify/scheduler,
  context+batch bins và warmup.
- G0–G4 là điều kiện giữ/bỏ thành phần: candidate headroom, source
  recoverability, oracle adaptive headroom (khuyến nghị >=10%), controller
  recover >=70% oracle headroom, và end-to-end gain sau khi cộng latency của
  chính component. Đây là các claim cần test chứ chưa phải kết quả.

### Design revision v1.1 sections 106–124 đã đọc

- Có hai track: validate mechanism trên cặp DFlash/DFlash2 được hỗ trợ dễ
  trước, sau đó mới tái lập trên target cuối của project (tài liệu cũ giả định
  Llama-3.1-8B). Repo structure đề xuất là một package `syncspec/` riêng với
  data/models/training/inference/experiments/tests.
- Cache schemas đã định nghĩa cho target trajectory, anchor, diffusion Top-M
  và rollout/timing. Correctness track gồm greedy exact equality và stochastic
  distributional equivalence; benchmark phải cùng target/prompt/precision/EOS/
  hardware/engine/warmup với baseline.
- Novelty boundary: không claim DFlash block diffusion, DFlash2 conv/selector,
  AdaFlash/adaptive SD, SSSD n-gram, LongSpec memory/Anchor-Offset hay exact
  verification. Candidate claims chỉ là N1 source-coherent reranking trong
  diffusion lattice, N2 survival/cost decoupling, N3 tách hai budget
  `K_d/K_v`, N4 cross-phase co-design nếu interaction ablation chứng minh.
- Failure policy rõ: source miss → gate 0/local selector; survival thấp → K_v
  nhỏ; pre-utility thấp → AR; retrieval lỗi → recent/anchor memory; batching
  phân mảnh → shrink profiles/AR. Immediate next step vẫn là freeze design,
  chọn pair, trace collector và chạy P0–P4 oracle trước training.

### Repo gap sau khảo sát

- `src/SyncSpec/` hiện chỉ có `SyncSpec_v1_design_complete.md`; chưa có
  package/model/inference/training/tests cho SyncSpec.
- Các match SyncSpec còn lại chỉ nằm trong design doc; implementation hiện có
  dưới `src/analyze/groundsync/` là pipeline phân tích GroundSync/P0/P1/P2,
  không phải engine SyncSpec.
- Vì vậy việc “triển khai ý tưởng” nếu được yêu cầu sẽ là xây một subsystem mới,
  không phải chỉnh một flow SyncSpec có sẵn; theo brainstorming gate cần chốt
  phạm vi/thiết kế và được người dùng duyệt trước khi viết code.

### Điểm cần phân biệt giữa conversation và file hiện tại

- Một message trung gian từng đề xuất thu hẹp pre-training study thành ba
  oracle: reduced draft context, joint `(R,K)` allocation và regime dependence
  theo `(L,B)`, với ngưỡng joint headroom khoảng 10–15%.
- File v1.1 hiện tại đã làm rõ hơn và đặt `R` cố định lúc đầu; hai adaptive axes
  chính là `K_d` (trước diffusion) và `K_v` (sau selector/survival). Vì file
  được message cuối chỉ định làm source of truth, không nên tự đưa joint-RK
  oracle cũ vào implementation V1 nếu chưa có quyết định sửa spec.
- Conversation cũng nêu rõ design concept đã gần hoàn chỉnh nhưng training/model
  spec chỉ khoảng 70–80%: còn target pair, exact DFlash/DFlash2 corruption,
  pilot dimensions/fusion, canonical losses, data scale và compute budget.

### Asset/runtime implication

- Repo có `externals/dflash`, `scripts/infer_dflash.py` và wrapper/config DFlash;
  smoke pair canonical hiện là Qwen3-4B target + Qwen3-4B-DFlash-b16. Đây là
  điểm bắt đầu tự nhiên cho Milestone A (trace/oracle), nhưng chưa phải
  SyncSpec drafter vì chưa có source selector/survival/two-level controller.
- DFlash adapter hiện là benchmark runner; không thấy file `src/SyncSpec/*.py`
  hay engine package để mở rộng trực tiếp. Implementation mới cần bọc/reuse
  DFlash ở tầng trace/candidate lattice, giữ config/runtime chung của repo.

### Implementation scope confirmed — 2026-09-02

- Yêu cầu hiện tại là triển khai đầy đủ, không chỉ oracle/P0. Implementation
  plan được ghi ở `docs/superpowers/plans/2026-09-02-syncspec-pipeline.md`.
- Kiến trúc sẽ dùng interface target/drafter/selector/verifier để synthetic
  CPU, CUDA toy và Transformers offline chia sẻ cùng engine. Đây là cách kiểm
  thử được correctness trên host hiện tại dù host không có B200, đồng thời
  không giả lập kết quả GPU server.
- Native drafter sẽ là shallow DFlash2-style model có checkpoint riêng; adapter
  target thật dùng full-context cache transaction. Các module mới phải có
  contract test trước implementation theo TDD.

### SyncSpec implementation evidence — 2026-09-02

- Full source-of-truth implementation hiện ở `src/SyncSpec/`; không sửa design
  v1.1. `scripts/infer_syncspec.py` dùng cùng engine cho synthetic CPU/CUDA và
  Transformers offline. Native drafter checkpoint gồm `config.json` và
  `pytorch_model.bin`; selector/survival có checkpoint riêng.
- Lỗi cache thực tế đã được phát hiện bằng LlamaForCausalLM nhỏ: PyTorch không
  cho deepcopy non-leaf tensors trong Transformers 5 `DynamicCache`. Clone
  detached từng DynamicLayer đã làm logits sau commit khớp full recompute với
  tolerance `1e-5`.
- Output smoke sample có đủ base/spec schema keys của repo, timing draft/
  selector/survival/verify, budgets `K_d/K_v`, acceptance và summary cuối.
- `profile_syncspec.py` ghi key model/checkpoint/GPU/precision/context/batch/
  Kd/Kv và measured component timings; engine ưu tiên verify cost profile rồi
  mới fallback heuristic. CPU profile không được dùng để claim B200 speedup.
- Full repository validation sau implementation: `153 passed, 2 skipped`;
  static checks pass. Local CPU chain và dispatcher wrapper đã chạy thật.
- Giới hạn còn lại là external hardware evidence: host hiện tại là Tesla T4,
  `torch.cuda.is_available()` false với stack cu130, nên CUDA/B200 conditional
  tests skip và `check_syncspec_b200.py --strict` trả `BLOCKED`. Cần chạy trên
  canonical B200 mount với `python3`, local target model, drafter checkpoint,
  data và cache writable trước khi báo GPU smoke PASS.

### Final implementation audit — 2026-09-02

- Selector stage đã được căn theo serving contract: candidate lattice là Top-M
  nguyên bản của drafter, target miss được mask trong loss; trainer hỗ trợ
  nhiều sample/lattice thay vì chỉ record đầu tiên.
- Survival stage không còn dùng target length làm proxy khi target thật được
  cung cấp; nó thu acceptance labels từ rollout on-policy. Joint CLI là pipeline
  tuần tự ba stage và lưu diffusion/selector/survival artifacts.
- CUDA timing được synchronize có điều kiện tại engine boundaries. Đây là
  instrumentation cho profile/serving, không phải speedup claim.
- Preflight bổ sung snapshot-ID resolution và structural target/drafter checks:
  tokenizer files, `config.json`, vocab/hidden width. Host local đã test được
  snapshot resolution bằng fake cache; hardware check vẫn đúng là BLOCKED.
- Full regression sau audit: **158 passed, 2 skipped**. Native Llama cache test
  vẫn khớp vanilla AR; không có B200 GPU evidence trong session này.

### Post-audit verification — 2026-09-02

- Edge contract `max_new_tokens=0` đã được sửa ở engine và có regression test;
  `max_new_tokens < 0` trả `ValueError`.
- Full suite mới nhất đạt **165 passed, 3 skipped, 21 warnings**; compileall,
  `bash -n` và `git diff --check` đều pass.
- Real local tiny-Llama path đã chạy được Stage 0, diffusion, selector,
  survival, infer với selector/survival checkpoints và measured profile. Test
  bfloat16 target/drafter kiểm tra cả native draft lẫn engine.
- Không được diễn giải kết quả CPU/T4 thành B200 benchmark. Strict preflight
  hiện vẫn trả `BLOCKED: hardware_unavailable`; cần server B200 và asset local
  thật để hoàn tất evidence hardware.

### Mở rộng pipeline sau audit — 2026-09-02

- Có thêm `run_syncspec_b200_train_smoke.sh`: preflight phase `train` chỉ yêu
  cầu target/data, tạo trajectory, chạy joint train và infer bằng cùng thư mục
  checkpoint.
- Trainer hiện có gradient accumulation, clipping, AMP BF16 trên CUDA, seed;
  Stage 4 low-LR joint refinement là opt-in qua `--joint-finetune`.
- Pre-gate calibration empirical nhóm theo `context_bin:batchN`, đọc paired
  serving traces và nạp qua `--gate-table`; không dùng CPU timing để tạo GPU
  profile.
- Preflight đã kiểm tra cache writable và compute capability tối thiểu; trên
  host hiện tại cache cũng read-only, nhưng nguyên nhân chính vẫn là CUDA
  unavailable.

### Final regression sau mở rộng — 2026-09-02

- Full suite đạt **175 passed, 5 skipped, 23 warnings** trong 3:37. Các skip
  đều là CUDA-conditional do host T4 không expose CUDA; không chuyển thành
  kết quả GPU giả.
- `run_syncspec_b200_train_smoke.sh` chạy strict train preflight, Stage 0,
  `--stage joint`, rồi infer bằng checkpoint vừa tạo. Stage 4 refinement dùng
  `--joint-finetune`; pre-gate calibration dùng `calibrate_syncspec_gate.py`.
- Checkpoint target-tied được rút gọn bằng `checkpoint_metadata.json`, tránh
  lưu lại target lexical weights lớn; loader yêu cầu tie trước forward và đã
  có regression test.
- B200 evidence còn thiếu đúng một phần external: phải chạy preflight, train
  smoke và infer smoke trên canonical B200 với model/data/cache thực. Local
  strict preflight hiện trả `BLOCKED` vì CUDA false và cache read-only.
- Wrapper train smoke có hai gate: phase `train` trước Stage 0 và phase
  `infer` sau joint checkpoint, nên không âm thầm chuyển artifact thiếu sang
  bước inference.

### Handoff hardening — 2026-09-02

- `batch_size` được ghi vào `InferenceResult`/JSONL và dùng khi chọn pre-gate
  cũng như matching profile. Bản ghi cũ từng chặn batch>1 trước khi có
  `generate_batch`; giới hạn đó đã được thay bằng execution path microbatch ở
  mục audit bên dưới.
- Full suite gần nhất vẫn **175 passed, 5 skipped, 23 warnings**; sau đó
  wrapper contract bổ sung infer preflight đã đạt `7 passed` và static checks
  pass.
- Scope còn cần external verification: actual batched serving và B200 GPU
  train/infer chưa thể xác nhận trên host T4; không được gọi đây là completed
  hardware gate.

### Microbatch serving audit — 2026-09-02

- `SyncSpecEngine.generate_batch` hiện đã thay thế giới hạn batch-1 trước đó:
  native drafter chạy batch theo nhóm tương thích, target block verification
  stack cache tạm thời, sau đó commit riêng từng request; prompt length khác
  nhau được regroup và adapter không hỗ trợ batch có scalar fallback exact.
- CLI inference và profiler đã dùng execution path batch thật cho
  `--batch-size > 1`. Regression thêm cho equal/mixed lengths, synthetic CLI
  và batch profile; full repository validation hiện đạt **180 passed, 5
  skipped, 23 warnings**.
- Prefill vẫn tuần tự theo request. Vì vậy actual batched serving đã có code và
  CPU evidence, nhưng GPU throughput/B200 train-infer vẫn cần external run trên
  canonical server với model, checkpoint, data và cache writable.
- Train-smoke B200 truyền `BATCH_SIZE` từ master config vào infer sau train; nếu
  đặt batch lớn hơn một, wrapper không còn âm thầm hạ về batch một.
- Stage 1 đã khớp hơn với design: random anchor sampling reproducible, tùy chọn
  teacher-logit KL/Top-M rank margin/position weighting; Stage 2 không còn chỉ
  học anchor đầu tiên. Stage 3 riêng nhận `--selector-checkpoint` để labels
  on-policy dùng đúng selector đã train.
- Target adapter tái sử dụng transaction cache khi cả block accept, tránh
  forward duplicate trong commit; rejection vẫn dùng path tuần tự an toàn.
  Test DynamicCache thật trên Transformers 5 khớp full recompute.

### Final verification — 2026-09-02

- Profile key hiện bao gồm context bin thực tế sau tokenization; engine kiểm
  tra đồng thời model/checkpoint/precision/GPU/context/batch/Kd trước khi dùng
  cost đo được. Batch ragged bị từ chối ở profiler để không trộn context bins.
- Full repository test: **192 passed, 5 skipped, 23 warnings**. Compileall,
  shell syntax và `git diff --check` pass.
- Kết luận evidence không đổi: implementation/reference path và CPU/tiny
  Transformers validation đã hoàn tất; B200 real-model train/infer/profile
  chưa thể xác nhận trên host T4 không có CUDA và phải được chạy external.
- B200 train-smoke hiện tự tạo measured profile sau checkpoint, preflight nhận
  repo-relative paths và infer dùng AR fallback nếu profile không khớp. Engine
  cũng commit correction/bonus token đúng semantics standard speculative.

### Final implementation hardening — 2026-09-02

- Anchor-Offset training không còn chỉ tồn tại ở inference: helper tính vị trí
  target tuyệt đối từ metadata Stage 0 và truyền tensor offset theo từng batch
  row vào drafter trong Stage 1/selector/joint.
- Learned mask sentinel tách khỏi target lexical embedding; checkpoint loader
  chấp nhận cả checkpoint mới và checkpoint cũ chưa có `mask_embedding`.
- GPU engine smoke có launcher độc lập, kiểm tra CUDA trước master production
  để môi trường dev trả `BLOCKED` có cấu trúc; không biến CPU thành evidence
  GPU/B200.

### Verification sau hardening — 2026-09-02

- Full repository: **196 passed, 5 skipped, 23 warnings**; warning còn lại do
  CUDA/NVML/Triton cache của host dev, không phải test failure.
- GPU smoke launcher trên host này trả đúng JSON `status=BLOCKED`, exit code 2;
  real CUDA/B200 train-infer vẫn cần canonical server.

### Regression sau offset-topology fix — 2026-09-02

- Full repository mới nhất: **197 passed, 5 skipped, 23 warnings**; thêm
  regression cho selector-stage anchor rebinding và tất cả test vẫn xanh.

### Production preflight gate — 2026-09-02

- Inference Transformers thiếu selector hoặc survival giờ fail-fast thay vì
  dùng module random; preflight infer yêu cầu cả hai checkpoint.
- Preflight offline xác minh các file weights/component/profile cần thiết mà
  không load model hoặc download internet.
- Full repository mới nhất: **200 passed, 5 skipped, 23 warnings**; CUDA/B200
  runtime vẫn chưa có evidence vì host dev không có CUDA.

### Budget/profile consistency — 2026-09-02

- Phát hiện và sửa mismatch tích hợp: profiler/train-smoke đo mặc định
  `K_d=16,K_v=8` nhưng inference trước đó chỉ có candidate `4/2,4/4`, khiến
  measured profile không match và controller fallback AR trên CUDA.
- Budget hiện đi từ master config/CLI đến engine; inference, B200 smoke và
  train-smoke dùng cùng cặp `K_d/K_v` với profile. Test wrapper synthetic đã
  xác nhận output giữ nguyên budget explicit.
- Evidence cục bộ cuối: `201 passed, 5 skipped, 23 warnings`; không coi các
  skip CUDA trên T4 là GPU pass. Preflight strict và CUDA launcher vẫn ghi
  `BLOCKED` đúng điều kiện host.

### Implementation audit mở rộng — 2026-09-02

- Stage 2 trước đây chưa truyền source-memory và dùng source-only history;
  đã chuyển sang source-memory cùng prefix target và thêm schedule
  teacher-forcing → self-conditioning.
- Selector hiện dùng learned predecessor/successor token embeddings theo
  công thức low-rank của design, với vocab size ghi trong checkpoint metadata.
- Engine phát hiện EOS giữa block cần cắt proposal/cache trước commit; đã thêm
  regression cho scalar và microbatch để tránh commit token sau EOS.
- Đang bổ sung phép tính round cost từ toàn bộ component profile; trước audit
  controller chỉ dùng verify latency, không phản ánh công thức của design.

### Implementation audit hoàn tất — 2026-09-02

- Controller hiện dùng tổng chi phí measured của một speculative round:
  draft + selector + survival + verify + scheduler; profile legacy chỉ có
  verify/e2e vẫn đọc được để không phá artifact cũ.
- Khi phần output còn lại ngắn hơn `K_v`, pre-gate/controller lọc candidate
  theo `max_kv`; profile có `kv` khác với candidate serving không thể vô tình
  bật speculation.
- EOS được xử lý trước commit transaction ở block speculative và ngay sau
  token ở AR fallback; không còn token/bonus sau EOS.
- Full suite cuối: **213 passed, 5 skipped, 23 warnings**. CUDA smoke trả
  structured `BLOCKED` exit 2; preflight strict trả `BLOCKED` vì hardware,
  assets và cache local, đúng với môi trường dev.
- Config drafter và preflight hiện fail-fast khi attention/group/positional
  capacity không hợp lệ, drafter ngắn hơn target long-context, hoặc selector
  không khớp width/vocab với target; preflight không coi riêng
  `tokenizer_config.json` là đủ cho offline loading.

### Evidence handoff — 2026-09-03

- Requirement train/infer pipeline đã có evidence CPU fresh và full regression;
  các guard mới cũng đã được test trực tiếp qua preflight contracts.
- Requirement GPU/B200 chưa đạt theo tiêu chí nghiêm ngặt: launcher chỉ xác
  nhận được `cuda_unavailable`, thiết bị hiện tại là Tesla T4, thiếu mount
  model/data/checkpoint/cache canonical. Cần external run trên B200 để đóng
  gate này; không suy diễn từ CPU hoặc T4.

## SyncSpec training memory audit — 2026-09-03

- Audit phát hiện trainer cũ đưa toàn bộ trajectory/anchor rows vào một
  forward; điều này không phù hợp với long-context hoặc cache nhiều mẫu dù
  inference đã có microbatch.
- Đã thêm batch contract xuyên suốt: `SyncSpecTrainer` chọn deterministic
  cyclic mini-batch cho diffusion/selector/survival/joint; selector CLI chunk
  cả drafter forward và sau đó huấn luyện selector theo cùng batch size.
- Bằng chứng fresh: targeted training tests pass; tiny Transformers
  Stage-2 regression pass sau khi sửa lỗi truyền nhầm hidden chunk cuối; full
  suite `214 passed, 5 skipped, 24 warnings`.
- `--train-batch-size` mặc định `1` trong CLI/wrapper. Đây là guard memory và
  không phải claim throughput; B200 vẫn phải chạy profile thật để chọn batch
  lớn hơn.
- Independent smoke chain sau hardening xác nhận training summary của
  diffusion/selector/survival/joint đều báo batch `1`, inference batch có
  `2/2 ok`, và profiler ghi `source=measured` với `K_d=4,K_v=2`.

## Survival semantics correction — 2026-09-03

- Design định nghĩa head output là hazard `h_j`, nhưng supervision là
  cumulative prefix survival `z_j`; implementation cũ đã BCE nhầm raw hazard
  với `z_j`, làm controller nhận curve không được train đúng mục tiêu.
- Đã sửa `survival_loss` để tính BCE trên `survival_from_hazard(hazard)` và
  sửa CLI calibration dùng cùng đại lượng `S_j`; engine semantics hazard →
  cumulative survival được giữ nguyên.
- Test đỏ xác nhận lỗi cũ, test xanh sau sửa; full suite hiện
  `215 passed, 5 skipped, 24 warnings`. Smoke chain mới cũng pass end-to-end
  trên CPU với joint training, batch infer và measured profile.

## Final contract hardening — 2026-09-03

- `top_m` hiện được kiểm tra dương ở cả config và helper chọn candidate; test
  regression xác nhận input bằng không bị từ chối thay vì rơi vào `topk(k=0)`.
- Full suite và static checks sau thay đổi đều pass; không có thêm thay đổi
  nào làm thay đổi kết luận về việc cần chạy trên B200 thật.

## Long-context position hardening — 2026-09-03

- Vị trí draft vượt `max_positions` trước đây bị modulo-wrap; hiện bị từ chối
  rõ ràng cho cả scalar và batch offset, phù hợp với contract positional
  capacity của preflight.
- Regression và full suite sau sửa đều pass (`216 passed, 5 skipped, 24
  warnings`).

## Post-hardening smoke evidence — 2026-09-03

- Synthetic CPU infer và measured CPU profile đều chạy lại thành công sau
  runtime guard; profile xác nhận batch `2` với `K_d=4,K_v=2`.
- CUDA smoke vẫn dừng đúng contract ở `cuda_unavailable` trên host T4; chưa có
  cơ sở để đánh dấu GPU/B200 requirement hoàn thành.

## Context-safe serving and staged-train contracts — 2026-09-03

- Engine hiện giới hạn sinh theo context headroom của target, kể cả với
  microbatch có prompt dài ngắn khác nhau.
- Staged training bắt buộc checkpoint của stage trước; full regression sau
  hardening là `220 passed, 5 skipped, 24 warnings`.

## Context-safe trajectory/profile completion — 2026-09-03

- Stage 0 trajectory và vanilla-AR profiling reference dùng cùng context cap
  với engine; full suite hiện `222 passed, 5 skipped, 24 warnings`.
- Fresh CPU infer/profile pass; CUDA path chỉ có thể xác nhận `BLOCKED` trên
  host T4 hiện tại.

## Latest verification handoff — 2026-09-03

- Bằng chứng hiện tại: full suite `222 passed, 5 skipped, 24 warnings`, CPU
  infer/profile fresh và static checks pass.
- Chưa đánh dấu hoàn tất mục tiêu vì GPU smoke thật và train/profile/infer trên
  canonical B200 chưa thể chạy trong môi trường session này.

## Anchor-only trajectory cache — 2026-09-03

- Target hidden features trong Stage 0 được lưu theo anchor thay vì toàn bộ
  suffix; metadata định vị giúp `_target_anchor_batch` đọc đúng anchor và vẫn
  tương thích cache format cũ.
- Full suite sau thay đổi: `223 passed, 5 skipped, 24 warnings`.

## Binary trajectory cache — 2026-09-03

- Thêm binary torch cache `.pt/.pth/.torch`, có fingerprint/schema, atomic
  replace và resume idempotent; train CLI tự nhận diện theo suffix.
- Unit/CLI integration và full suite pass: `225 passed, 5 skipped, 24 warnings`.

## B200 wrapper binary-cache default — 2026-09-03

- Train-smoke B200 mặc định đã chuyển sang `.pt` để tránh JSON tensor phình
  trên long-context; biến `SYNCSPEC_TRAIN_TRAJECTORY` vẫn override được.
- Launcher contract/static validation pass; không thay đổi kết luận B200 thật
  còn cần external run.

## Full-block survival-label rollout — 2026-09-03

- Survival training label collection giờ bypass controller choice chưa được
  train bằng `force_kv=K_d`, nhưng vẫn chạy selector/drafter và exact target
  verification; fresh joint CPU smoke pass.
- Full suite sau thay đổi: `226 passed, 5 skipped, 24 warnings`.

## Runtime exactness guard — 2026-09-03

- Bổ sung guard runtime để B200 smoke không chỉ kiểm tra schema mà còn so sánh
  output greedy SyncSpec với vanilla AR fresh-state trên cùng target.
- Mismatch được ghi trong record/summary và trả exit code `1`; không bật cờ này
  trong benchmark bình thường vì sẽ trả thêm chi phí target-AR.
- Guard không áp dụng cho stochastic decoding; stochastic exactness cần kiểm
  định phân phối proposal/residual riêng.
- Full repository sau guard: **229 passed, 5 skipped, 24 warnings**; static
  checks pass. GPU evidence vẫn chưa có vì môi trường hiện tại không có CUDA.

## B200 master-cache default audit — 2026-09-03

- Có mismatch giữa fallback `.pt` trong train wrapper và biến mặc định trong
  master example; giá trị từ master được source sau đó sẽ thắng fallback.
- Đã sửa master example sang `.pt` và thêm test bảo vệ; JSONL chỉ còn là lựa
  chọn override có chủ đích.

## Runtime feedback loop — 2026-09-03

- Engine hiện cập nhật EMA acceptance và component latency sau mỗi speculative
  verification, đồng thời ghi state để kiểm toán.
- Acceptance thấp làm giảm pre-draft gain prior; regression test xác nhận safe
  degradation sang AR thay vì tiếp tục trả draft cost vô điều kiện.
- Full repository sau patch: **231 passed, 5 skipped, 24 warnings**.

## Stochastic exactness distribution test — 2026-09-03

- Kiểm định mới sample proposal từ `q`, chạy rejection/residual và so sánh
  histogram output với `p_target` trên 12.000 trial; sai số tối đa được giữ
  dưới `0.025`.
- Full repository sau test: **232 passed, 5 skipped, 24 warnings**.

## Runtime profile provenance/schema guard — 2026-09-03

- Profile dùng để gate CUDA không còn được chấp nhận chỉ dựa trên các trường
  benchmark; bắt buộc `schema_version=1` và `source="measured"`.
- Cả B200 preflight và engine đều enforce guard này, tránh dùng calibration
  synthetic/không rõ nguồn để bật speculation.
- Full regression sau hardening: **236 passed, 5 skipped, 24 warnings**.

## Post-draft measured AR utility gate — 2026-09-03

- Trước patch, post-draft controller chỉ tối ưu trong các profile speculative;
  nó không biết chi phí AR nên có thể trả draft dù utility thấp hơn AR.
- Đã bổ sung `ar_cost_from_profile`, chuẩn hóa `target_ar` về latency/token và
  fallback sau draft ở cả scalar/batch engine path.
- Fallback ghi acceptance zero vào request-local feedback, nên round sau chuyển
  sang AR thay vì trả draft cost lặp lại.
- Full regression sau patch: **241 passed, 5 skipped, 25 warnings**.

## Stage-0 artifact provenance hardening — 2026-09-03

- Cache key trước đây chỉ dựa trên model identifier và CLI options; cùng một
  local path nhưng artifact bị thay có thể tái sử dụng trajectory cũ.
- `artifact_fingerprint()` hiện ghi nhận manifest local và bounded content
  digest; Stage 0 lưu fingerprint này trong metadata, đồng thời fingerprint
  theo `--seed` để resume không trộn các trajectory khác seed.
- Regression cuối sau hardening: **243 passed, 5 skipped, 25 warnings**; static
  checks pass.

## B200 train seed handoff — 2026-09-03

- Train wrapper giờ truyền một `SYNCSPEC_TRAIN_SEED` chung cho trajectory và
  drafter training; default là `42`, có thể override trong master config.

## Long-context hidden-state memory hardening — 2026-09-03

- `output_hidden_states=True` trong prefill/verify có thể giữ hidden của mọi
  layer, không phù hợp với context dài dù SyncSpec chỉ cần hidden cuối.
- Adapter hiện hook final normalization để lấy hidden cuối với
  `output_hidden_states=False`; test custom model buộc cờ này là false và test
  real Llama cache/native E2E đều pass.
- Full repository final: **243 passed, 5 skipped, 25 warnings**; static checks
  pass.

## Target-derived source-memory cache + profiler boundary — 2026-09-03

- Train và serving trước đây không cùng source-memory distribution: train
  fallback sang drafter token embedding, còn serving lấy final hidden của
  target prefill.
- Stage 0 đã bổ sung descriptor cache theo chunk (`source_memory`) và metadata
  offset; `.pt` giữ descriptor dạng tensor, JSONL giữ format dễ inspect. Các
  stage train ưu tiên cache target-derived, cache legacy vẫn chạy fallback.
- Profiler đã đặt CUDA synchronization boundary sau AR prefill trước timer;
  `target_ar` hiện đo đúng một token sau prefill.
- Targeted tests pass; phải cập nhật số liệu full suite sau regression cuối.

## Regression sau source-memory alignment — 2026-09-03

- Full regression cuối: **247 passed, 5 skipped, 25 warnings**; compileall,
  shell syntax và `git diff --check` đều pass.
- Tiny local Transformers đã chạy joint diffusion/selector/survival + optional
  refinement, profile và greedy exactness thành công.
- GPU smoke exit `2` với `status=BLOCKED, reason=cuda_unavailable`; B200 strict
  preflight vẫn bị block bởi hardware/assets/cache ngoài canonical server.

## Profile-aware pre-draft budget selection — 2026-09-03

- Pre-gate trước đây chỉ nhận một scalar predicted gain và chọn profile
  speculative có `K_d` lớn nhất, dù design yêu cầu so sánh các profile trước
  khi trả chi phí diffusion.
- `fit_empirical_gate_table` nay giữ key profile-specific khi trace cung cấp
  `kd`/`budget.kd`; `PreDraftGate` tối ưu gain theo từng `K_d` và không dùng
  default cho profile chưa có calibration nếu bảng đã có trục này.
- Engine truyền mapping priors theo `K_d` cho scalar và batch path. Đây là
  empirical pre-gate tương thích với trace cũ; utility/speedup thực tế vẫn chỉ
  được kết luận từ profile measured trên B200.

## Final regression sau profile-aware gate — 2026-09-03

- Full repository pass: **252 passed, 5 skipped, 25 warnings** (`162.51s`).
- CPU synthetic end-to-end wrapper và vanilla-AR exactness pass; static checks
  (`compileall`, shell syntax, diff check) pass.
- CUDA smoke không bị coi là pass giả: exit `2` với JSON
  `status=BLOCKED/cuda_unavailable`. Strict B200 preflight cũng block do host
  không có CUDA, không có các asset/profile canonical và cache ngoài workspace.

## Final adaptive-profile/batch-cost audit — 2026-09-03

- Controller giữ calibration riêng theo `context:batch:kd`, không dùng
  `default_gain` để bật một `K_d` chưa đo khi bảng đã có profile-specific
  entries; nếu profile ưu tiên không có measurement, engine thử profile đã đo
  khác trước khi rơi về AR.
- Chi phí component shared của profile batch được chuẩn hóa theo batch trước
  khi so utility trên từng request; `target_ar_tokens` ghi rõ số request được
  đo trong batch.
- Full regression cuối: **263 passed, 5 skipped, 25 warnings**; compileall,
  shell syntax và `git diff --check` pass. CPU adaptive wrapper exactness pass.
- CUDA/B200 chưa thể xác nhận tại host dev: CUDA smoke `BLOCKED` exit `2`,
  preflight strict `BLOCKED` vì thiếu CUDA, model/checkpoint/data/profile và
  cache writable. Không dùng kết quả CPU để kết luận speedup B200.
- Hai wrapper B200 đã được chạy trực tiếp với example master; cả train và
  infer đều dừng ở strict preflight với JSON `status=BLOCKED`, không đi tiếp
  vào model/checkpoint generation khi hardware hoặc asset thiếu.
- Launcher infer B200 đã được sửa để dùng `KD/KV` sau normalize của
  `config.sh`; test contract bắt lỗi truyền đồng thời fixed budget và finite
  profile khi override bằng tên biến chung.

## CPU full-chain smoke runner — 2026-09-03

- Runner mới thực thi trực tiếp Stage 0 binary trajectory, joint training của
  drafter/selector/survival, kiểm tra checkpoint, profiler multi-profile và
  batch inference exactness.
- Fresh run `/tmp/syncspec_cpu_smoke_final` tạo `trajectories.pt`,
  `pytorch_model.bin`, `selector.pt`, `survival.pt`, hai profile `4:2/4:4`
  và output `records=2` với `exactness_failures=0`.
- Test local Tiny Llama qua CLI Transformers cũng bật `--check-exactness` và
  xác nhận `exact_match_vanilla_ar=true`, `exactness_failures=0`.
- Timing trong run này chỉ là correctness/orchestration smoke trên CPU; không
  được dùng làm bằng chứng speedup GPU.
- Full regression sau khi runner source shared config/runtime: **263 passed,
  5 skipped, 25 warnings**; bốn launcher-contract failures do helper chưa
  source runtime đã được sửa.

## Transformers dtype loading hardening — 2026-09-03

- Adapter ưu tiên keyword `dtype` của Transformers 5, đồng thời fallback có
  điều kiện sang `torch_dtype` cho runtime legacy; không còn phát sinh cảnh
  báo deprecation từ đường load model hiện hành.
- Có regression test riêng cho cả hai nhánh; full repository sau patch đạt
  **265 passed, 5 skipped, 25 warnings**.
- Re-run inference trên Qwen3-0.6B thật sau patch vẫn trả `status=ok`,
  `exactness_failures=0`; output không còn warning `torch_dtype`.

## B200 batch/sample smoke contract — 2026-09-03

- B200 infer-smoke và train-smoke hiện fail-fast nếu `MAX_SAMPLES < BATCH_SIZE`,
  tránh profile batch-N rồi vô tình infer batch-1 và rơi im lặng về AR.
- Contract test và direct wrapper test với `MAX_SAMPLES=1/BATCH_SIZE=4` đều
  xác nhận exit `2` cùng thông báo `max_samples must be >= batch size`, trước
  preflight/model loading.
- Full regression sau guard: **266 passed, 5 skipped, 25 warnings**.

## B200 preflight exit-code contract — 2026-09-03

- `check_syncspec_b200.py --strict` hiện phân biệt `PASS=0`,
  `BLOCKED=2` (thiếu hardware/external environment) và `FAIL=1` (asset/config
  không hợp lệ), trong khi vẫn ghi structured JSON report.
- Regression preflight không-CUDA pass với `BLOCKED` và exit `2`; full suite
  sau patch: **266 passed, 5 skipped, 25 warnings**.

## Real-model batch exactness audit và CUDA fail-fast — 2026-09-03

- Qwen3-0.6B local chạy `SyncSpecEngine.generate_batch()` với hai request và
  checkpoint thật; cả hai output đều khớp vanilla target AR, `batch_size=2`.
- Inference/trajectory/train CLI nay fail-fast khi yêu cầu CUDA nhưng host
  không có CUDA, thay vì synthetic backend tự fallback CPU. Nhóm regression
  liên quan đạt **31 passed**.
- Kết quả trên không thay thế GPU evidence; B200 canonical vẫn là bước handoff
  bên ngoài workspace hiện tại.

## Final local regression sau GPU-integrity hardening — 2026-09-03

- Full regression đạt **270 passed, 5 skipped, 25 warnings**; compileall,
  `bash -n` và `git diff --check` đều sạch.
- CUDA smoke trả structured `BLOCKED` với exit `2`; strict B200 preflight cũng
  trả `BLOCKED`/exit `2` do CUDA và canonical assets không có trên host dev.

## CPU full-chain recheck sau hardening — 2026-09-03

- Runner fresh `/tmp/syncspec_cpu_smoke_final_gpu_guard` vẫn chạy xuyên suốt
  Stage 0, joint train, profile nhiều budget và batch inference.
- Output đạt `status=ok`, `records=2`, `exactness_failures=0`; profile batch
  ghi đúng `target_ar_tokens=2`.

## B200 profile provenance guard — 2026-09-03

- Chỉ kiểm tra schema là chưa đủ: profile CPU hoặc khác batch có thể khiến
  serving B200 fallback AR mà smoke vẫn nhìn như thành công. Preflight nay
  đối chiếu model/checkpoint/GPU/precision/batch và từ chối profile lệch.
- Hai wrapper B200 truyền các trục runtime này vào preflight; targeted tests
  đạt **22 passed**.

## Final regression sau profile provenance guard — 2026-09-03

- Full regression đạt **271 passed, 5 skipped, 25 warnings**; static checks
  vẫn sạch.
- Preflight đầy đủ artifact Qwen local đã bắt profile CPU bằng
  `runtime_profile_invalid` cùng `hardware_unavailable`.

## Production profiler component guard — 2026-09-03

- Không nên phát hành measured profile từ selector/survival random. Profiler
  target thật nay fail-fast khi thiếu hai checkpoint; diagnostic phải ghi rõ
  `--allow-untrained-components`.
- Local CLI/profile regression sau thay đổi: **26 passed**.

## Standalone staged-training audit — 2026-09-03

- Fresh CPU chain riêng biệt đã chạy Stage 0 → diffusion → selector →
  survival, mỗi stage một bước trên cache synthetic có target features và
  source-memory descriptors.
- Các artifact nối tiếp đều được tạo thành công: `pytorch_model.bin`,
  `selector.pt`, `selector_config.json` và `survival.pt`; diffusion/survival
  loss hữu hạn và fingerprint cache nhất quán.
- Đây là kiểm chứng wiring/training contract; số liệu loss và timing synthetic
  không đại diện cho chất lượng hay speedup trên B200.

## Real Qwen3 offline Stage-0 audit — 2026-09-03

- Snapshot local `/home/tuantb/models/Qwen3-0.6B` đã chạy qua
  `build_syncspec_trajectories.py --backend transformers --local-files-only`
  trên một mẫu dữ liệu thật.
- Artifact `.pt` đọc lại qua fingerprint guard thành công: `1` record, `128`
  source tokens, `1` target token, `2` source-memory chunks và metadata
  `source_memory_source=target_final_hidden`.
- Đây xác nhận adapter Qwen3/Transformers và target-derived source memory ở
  model thật; không phải GPU evidence và không thay thế B200 smoke.

## Real Qwen3 offline train/profile/infer audit — 2026-09-03

- Snapshot Qwen3-0.6B đã chạy trọn đường production CPU: Stage 0 → `joint`
  train → checkpoint drafter/selector/survival → measured profile → infer.
- Joint training một bước tạo đủ checkpoint; profile ghi đầy đủ component
  costs và `target_ar_tokens=1`.
- Inference với `--check-exactness` trả `status=ok`,
  `exactness_failures=0`, `exact_match_vanilla_ar=true`; output decode được
  từ tokenizer thật (`"to"`).
- Đây là validation adapter/model artifact offline trên CPU; không dùng timing
  này để claim GPU speedup và vẫn cần real B200 run.

## Final local verification — 2026-09-03

- Regression cuối sau production profiler guard: **272 passed, 5 skipped,
  25 warnings**.
- Compileall Python, shell syntax của các wrapper SyncSpec và `git diff --check`
  đều pass.
- Không có bằng chứng benchmark GPU trong run này; CUDA/B200 vẫn cần chạy trên
  canonical server.

## Profile measurement/provenance hardening — 2026-09-03

- Measured profile gắn với target, drafter, selector, survival, GPU, precision,
  context và batch; mismatch bị AR fallback hoặc preflight reject.
- Profile diagnostic không còn mang nhãn measured.
- Profile timing được giới hạn một round cố định với `K_v` đã chọn, phù hợp
  với cost model của controller.
- Full regression sau thay đổi: **277 passed, 5 skipped, 25 warnings**.
