# Progress log — GroundSync

## 2026-08-29

- Đọc proposal GroundSync từ shared conversation và xác định mạch H1–H5/E0–E5.
- Đọc `AGENTS.md`; giữ nguyên quy ước Python/runtime/data của repo.
- Khảo sát repo: worktree sạch; xác nhận `src/analyze` mới có profiler Qwen3
  target-only, chưa có implementation GroundSync.
- Xác nhận các dataset representative/LongBench và các script Qwen3/EAGLE hiện có.
- Chưa sửa code; đang chờ chốt design experiment theo ML brainstorming gate.
- Design và implementation plan đã ghi dưới `src/analyze/groundsync/plans/`.
- Subtask 1: viết core metric tests (RED) và implement `core.py`; fresh result
  `12 passed in 0.07s`.
- Subtask 2: viết target adapter tests (RED) và implement `trace_target.py`;
  fresh result combined core/target `15 passed`, compile pass.
- Subtask 3: thêm controlled draft/target acceptance, drift, grounding horizon
  và timing arrays theo `k`; fresh result toàn bộ tests `31 passed`.
- Subtask 4: thêm document-split predictor, bootstrap CI, H1–H5 aggregation,
  CSV/PNG/Markdown artifacts; report tests và integration tests nằm trong tổng
  `31 passed`.
- Subtask 5: thêm `run_experiment.py`, synthetic fixture và README. Synthetic
  run `synthetic-20260829-v2` tạo raw JSONL, metrics, CSV, PNG và report.
- Model-backed local-only smoke `qwen3-local-smoke-20260829` đã thử với
  Qwen3-1.7B/0.6B trên CPU; snapshot không có trong cache, nên target/draft và
  H1–H5 đều ghi `UNAVAILABLE`. Không dùng Qwen2.5-VL hay EAGLE head thay thế.
- Final validation: `34 passed`, `compileall` và `git diff --check` đều pass.
  Run `synthetic-20260829-audit` có đủ 4 PNG + JSON/CSV/Markdown; run
  `qwen3-local-smoke-audit` tái hiện rõ 5/5 hypothesis `UNAVAILABLE` vì thiếu
  Qwen3 snapshot local.
- Audit continuation: verifier H4 đã được sửa để đo block `k` trong một
  forward; full test/compile lại pass và run
  `synthetic-20260829-blocked-audit` chạy thành công. Đã kiểm tra thêm các
  mount/path hệ thống, không có full Qwen3 snapshot; blocker cần external model
  state hoặc user cung cấp đường dẫn checkpoint.
- Cache audit mới: xác nhận full Qwen3-4B canonical target trong Hugging Face
  cache; Qwen3-4B EAGLE head bị loại vì sai architecture. Qwen3-0.6B đã được
  tải local và validate bằng config/tokenizer/safetensors.
- Model smoke mới `qwen3-4b-06b-actual-smoke-20260829` chạy end-to-end trên
  CPU với target Qwen3-4B + draft/verifier Qwen3-0.6B, gồm timing verifier.
- Discovery mới `qwen3-4b-cnn10-target-20260829` chạy 10 mẫu CNN/DailyMail:
  target 10/10, controlled proposal 20/20, report mới cho H1 `FAIL`, H2
  `FAIL`, H3/H4/H5 `UNAVAILABLE` theo coverage/điều kiện đo; không overclaim
  hypothesis từ run này.
- Theo yêu cầu dùng môi trường máy, đã xác minh trực tiếp ngoài sandbox:
  `/home/tuantb/miniconda3/bin/python3` thấy Tesla T4 và CUDA 12.4; không dùng
  `.venv` cho thực nghiệm GPU.
- Discovery GovReport đầu tiên lộ OOM do T4 `sm75` không có native Flash SDP
  phù hợp với torch cu124. Đã thêm chunked causal prefill (mask bottom-right)
  cho target, draft và verifier, kèm regression tests; fresh suite sau patch
  đạt `38 passed`.
- Run GPU chính `qwen3-4b-gov25-gpu-all-20260829` chạy 25/25 target và 50/50
  speculative rows, mọi row có timing. Report mới: H1/H2/H4 `FAIL`, H3/H5
  `UNAVAILABLE`; không có OOM sau patch.
- Đã viết báo cáo diễn giải toàn bộ quy trình, runtime, cache audit, các lần
  OOM và sửa chunked prefill, lệnh tái lập, kết quả H1–H5 và giới hạn tại
  `src/analyze/groundsync/verification_report_2026-08-29.md`.
- Đã audit lần hai theo yêu cầu: bổ sung protocol-versus-implementation
  deviations, gate versioning, inventory 18 run directories, cache download
  event, artifact/log gaps và các metric H1 lag/segment vào báo cáo; không đổi
  raw result hay status H1–H5.

## Bổ sung protocol và kết quả 2026-08-30

- Theo TDD, thêm test/implementation cho positional calibration 32-bin,
  sensitivity chunk 64/128/256, sink 4/8/16, first-rejection hazard theo
  relative draft position, position-adjusted hazard coefficient với 2.000
  document-bootstrap resamples, fixed/adaptive/true-cost policy sweep, H3
  negative controls và H5 threshold selection train/dev. Fresh suite đạt
  **50 passed**;
  `compileall` pass.
- Thêm `start-offset` và `sample-ids` cho speculative runner để không lấy duy
  nhất position 0 và để tạo timing subset đúng document test; mỗi speculative
  output có manifest coverage/timing basis.
- `qwen3-4b-gov100-gpu-protocol-20260830`: 100 GovReport target request,
  99 `ok`, 1 OOM; 99 draft-only proposals tại start=1, kmax=8; 12 test-ID
  timing requests, 11 `ok`, lọc còn 10 rows phủ đủ k=8.
- `qwen3-4b-cnn100-gpu-protocol-20260830`: 100/100 CNN/DailyMail target
  `ok`, 100 controlled proposals và 12 timing rows phủ đủ k=8; dùng làm
  cross-regime H1–H5.
- Kết quả run mở rộng: H1 composite `FAIL` (no-sink CI lower 0,021996 nhưng
  calibrated 0,018257; CNN CI lower 0,010580); H2 GovReport `FAIL` với
  coefficient -0,0657 và CI [-0,0664; -0,0535], CNN/DailyMail `PASS` với CI
  [0,0181; 0,0301]; hai regime không cùng chiều. H3 primary `FAIL` (Gov
  AUROC gain 0; CNN no-sink gain -0,0267 dù calibrated sensitivity +0,0267);
  H4 `FAIL` (grounding oracle speed gain -0,4433 Gov/ -0,4386 CNN); H5
  `UNAVAILABLE` (predictor có metric nhưng grounding oracle chậm hơn fixed,
  nên oracle-gain recovery không được định nghĩa ở cả hai regime).
- Báo cáo chính đã cập nhật thành bản mở rộng, giữ run 25 mẫu là lịch sử và
  link artifact mới; đã ghi thêm hazard coefficient/bootstrap và kết quả
  cross-regime H2–H5. Chưa claim EAGLE/vLLM production throughput hoặc causal
  attribution.
- Đã bổ sung E0 model-backed position-relocation fixture bằng Qwen3-4B trên
  T4: cùng evidence ở đầu/giữa/cuối, 3/3 case `ok`; raw mass
  `0,5029/0,1170/0,1929`, no-sink mass `0,5185/0,2297/0,2296`. No-sink giảm
  nhưng không loại bỏ positional confounder.
- Đã bổ sung lag-drift history và chạy multi-start draft-only trên 100 target
  traces mỗi regime, bốn start `1,6,11,16`: GovReport 396 rows, CNN/DailyMail
  400 rows. H2 giữ cùng pattern theo regime; H3 no-sink gains lần lượt
  `-0,0053` và `+0,0086`, đều dưới gate `0,02`; thiếu verifier timing nên H4
  `UNAVAILABLE`, H5 `INCONCLUSIVE` cho các run bổ sung.

## P0 decision extension — 2026-09-02

- Đã đọc lại AGENTS, design/implementation plan và artifact schema. Đã chốt
  implementation plan cho corrected H2/H4, oracle ladder `k=0/2/4/8/16`,
  first-token admission và burstiness; code/results sẽ co-locate dưới
  `src/analyze/groundsync/`.
- Kiểm tra runtime: `nvidia-smi` thấy Tesla T4 15 GiB, driver 550.163.01,
  compute capability 7.5; Python ngoài venv trong sandbox không expose CUDA,
  nhưng kiểm tra escalated xác nhận CUDA hoạt động. Không dùng `.venv` cho run
  GPU.
- Trạng thái hiện tại: trước khi chạy model-backed P0; đang triển khai metric
  code/test và instrumentation timing AR/k=16.
- Đã hoàn tất TDD cho P0 metric module `p0_decision.py` và AR timing field trong
  `trace_speculative.py`; fresh suite đạt **55 passed**, `compileall` và
  `git diff --check` pass. Chưa có raw P0 model run mới; bước kế tiếp là
  acceptance/timing `max_k=16` ngoài venv.

## P0 execution complete — 2026-09-02

- Đã chạy acceptance `Kmax=16` trên 100 request/dataset bằng Qwen3-4B target và
  Qwen3-0.6B draft ngoài venv trên T4. GovReport target/acceptance đạt 99 row
  hợp lệ do một target trace thiếu; CNN/DailyMail đạt 100 row.
- Đã sửa semantics timing: AR cached one-token cho `k=0`, draft incremental
  decode và cached target block verification cho `k>0`; loại prefill khỏi chi
  phí round và dùng common rows đủ `k=0,2,4,8,16`. Timing hoàn chỉnh là
  55 GovReport và 50 CNN/DailyMail.
- Đã chạy multi-start cuối `1,4,7,10,13,16,19,22,25`, `max_k=4`, đủ 9 round/
  document để đo `delta=1,2,4,8`: GovReport 891 row/99 documents, CNN/DailyMail
  900 row/100 documents. Early EOS được censor, không tính là rejection giả.
- Analyzer source-of-truth là `results/p0-decision-final9-20260902/`; báo cáo đầy
  đủ là `p0_final_report_2026-09-02.md`. Kết luận cross-regime: P0-1 `MIXED`,
  P0-2 `FAIL`, P0-3 `PASS`, P0-4 `MIXED`, P0-5 `PASS`; overall
  `NO_GO_GROUNDSYNC_GENERAL; conditional BurstSpec follow-up only where admission passes`.
- Validation cuối sau các sửa: **57 passed**, compileall pass và `git diff --check`
  pass. Chưa chạy P1 predictor, strong-drafter replication, Multi-News hoặc
  EAGLE/vLLM production vì các bước đó chỉ được phép sau P0 gate.
-

## Hoàn tất P1/P2 mở rộng — 2026-09-02

- Đã thêm `p1_predictor.py`: predictor admission causal với document split,
  threshold/policy chọn trước test, metrics AUROC/AUPRC/log-loss/Brier/ECE và
  realized tokens/ms. Run `p1-cheap-admission-20260902`: GovReport FAIL
  (recovery 0%), CNN/DailyMail PASS (71.2%), Multi-News INCONCLUSIVE do chỉ có
  2 test documents.
- Đã thêm `p1_strong_drafter.py`, smoke và full EAGLE-3 Qwen3-4B head trên
  50 GovReport, 50 CNN/DailyMail và 50 Multi-News; acceptance được trừ một
  target fallback token. Burstiness gate FAIL cả ba regime.
- Paired direct E2E cùng runner có speedup aggregate Gov 1.819x, CNN 1.905x,
  Multi-News 1.814x; exact-match với greedy AR lần lượt 96%, 98%, 94%, nên
  P2 direct FAIL lossless guardrail dù tốc độ dương.
- Đã thêm `p2_serving_preflight.py`. Canonical server mount không tồn tại;
  runtime GPU miniconda không có vLLM, `.venv` có vLLM nhưng CUDA không khả
  dụng. P2 serving/API ghi `UNAVAILABLE`, không thay bằng direct E2E.
- Đã chạy Multi-News P0 confirmatory 50 target/spec, 10 timing, 450
  multistart rows: H2/H4/P0-5 FAIL, O3 +43.7%, admission k4 recovery 77.7%.
- Báo cáo master: `src/analyze/groundsync/final_decision_report_2026-09-02.md`.
- Validation cuối fresh: **66 passed**, compileall pass, `git diff --check`
  pass và final artifact/decision audit pass.

## Đọc SyncSpec-v1 — 2026-09-02

- Bắt đầu đọc `src/SyncSpec/SyncSpec_v1_design_complete.md`; file có 4.818
  dòng, revision v1.1.
- Đã đọc phần mở đầu và ghi nhận objective/decomposition/pipeline vào
  `findings.md`.
- Web reader không fetch được ChatGPT share URL do `Cache miss`; chưa coi
  conversation là đã đọc.
- Chưa sửa implementation; đang ở pha làm rõ design và phạm vi theo
  brainstorming gate.
- Đã truy cập được shared conversation bằng read-only `curl` ngoài sandbox và
  parse React payload thành JSON; đã xác định chronology GroundSync →
  DiffuRoute → SyncSpec và các message dài cần đối chiếu.
- Đã ghi nhận một script trong conversation từng lỗi `PermissionError` khi
  ghi `/mnt/data`; lỗi này thuộc session share, không phải repo workspace.
- Đã đọc đầy đủ phần còn lại của design v1.1 (sections 63–124), giải mã các
  message chính của conversation và khảo sát asset DFlash trong repo. Kết luận:
  SyncSpec là subsystem mới; design là source of truth, còn DFlash adapter chỉ
  là baseline/điểm tựa cho Milestone A. Chi tiết nằm trong `findings.md`.

## Bắt đầu triển khai toàn bộ SyncSpec-v1 — 2026-09-02

- Người dùng đã mở rộng yêu cầu từ đọc/hiểu proposal sang thiết kế và triển
  khai toàn bộ pipeline: source evidence, drafter training, selector/survival,
  exact verification, end-to-end inference, profiling và B200 smoke.
- Đã chốt implementation plan tại
  `docs/superpowers/plans/2026-09-02-syncspec-pipeline.md`; giữ design v1.1
  làm source of truth và không đưa joint-RK oracle cũ vào v1 serving.
- Chưa claim implementation hoàn tất; đang bắt đầu TDD từ contract/config,
  source evidence và output schema.

## SyncSpec pipeline implementation complete locally — 2026-09-02

- Đã triển khai package mới `src/SyncSpec`: config/schema; SourceNgramIndex
  n=2..6; bounded SourceMemoryBank chunk/top-R; native shallow
  DFlash2-style block drafter với bidirectional block attention, grouped
  causal convolution, anchor/recent/source conditioning, Top-M checkpoint;
  selector source-coherent normalized q; hazard/survival; pre/post controller.
- Đã triển khai exact target path: greedy longest-prefix/correction và
  stochastic rejection-correction; `TransformersTargetAdapter` full-context
  prefill + transaction cache. Đã sửa tương thích Transformers 5/PyTorch 2.11
  bằng clone detached DynamicCache layers thay vì deepcopy non-leaf tensors.
- Đã triển khai Stage 0 target trajectory JSONL fingerprint, tùy chọn target
  feature cache; train CLI Stage 1 diffusion, Stage 2 selector cập nhật
  selector weights thật, Stage 3 on-policy survival rollout/Calibration và
  joint flag; checkpoint resume/tie target embedding+LM head frozen.
- Đã triển khai end-to-end `scripts/infer_syncspec.py`, wrapper/dispatcher
  `scripts/run_syncspec.sh`, profile measured component costs và
  `scripts/check_syncspec_b200.py` + `run_syncspec_b200_smoke.sh`; output có
  schema benchmark + summary.
- Fresh full validation: **153 passed, 2 skipped**; `compileall`, shell syntax
  và `git diff --check` pass. CPU live chain Stage0 → 3 training stages →
  inference/profile pass; dispatcher synthetic smoke pass.
- Host hardware evidence: `nvidia-smi` là Tesla T4 15 GiB; theo AGENTS không
  dùng stack CUDA cu130 trên host này. SyncSpec CUDA/B200 tests skip có lý do;
  preflight fresh trả `BLOCKED`, không giả lập kết quả B200. Real B200 command
  đã sẵn sàng nhưng cần chạy trên canonical server với local model/drafter
  checkpoint.

## Target-KV experiments complete on T4 — 2026-09-03

- Đã triển khai E0 Target-KV DFlash failure map và E1 representation probe
  trong `src/analyze/groundsync/`; model-backed execution dùng Miniconda CUDA
  ngoài `.venv`, local-only Qwen3-4B + Qwen3-4B-DFlash-b16.
- E0 FP16 pilot: GovReport 13 documents/39 K-record, Multi-News 49/147,
  CNN/DailyMail 30/90; mọi observed round có `accepted_draft_tokens=0`.
  E0 max-new32 confirmation thêm 5 document mỗi GovReport/Multi-News, 480
  round mỗi dataset, vẫn zero acceptance. Official DFlash cross-check cũng
  ghi raw acceptance `[1]*8`, tức 0 draft token sau khi trừ fallback.
- E0 long-context gate được sửa để trả `INCONCLUSIVE` khi thiếu natural long
  bucket; 8-bit feasibility 11K/16K/28K chạy được nhưng không trộn vào FP16.
- E1 extraction/probe: GovReport cap4K 17 documents/34 anchors; Multi-News
  cap8K 49/98. KV không vượt `hidden_sequence` về CE/accuracy ở cả hai run.
- Đã sửa hai lỗi runtime E1: chuyển hidden chunks/KV feature về CPU và bọc
  forward bằng `torch.inference_mode()`; sau sửa không còn artificial autograd
  OOM ở cap đã chọn.
- Fresh validation riêng ground-sync: **94 passed**, `compileall` và
  `git diff --check` pass. Full repo suite vẫn có một failure cũ ở
  `common/model_compat.py` (`64 passed, 1 failed`).
- Báo cáo master: `src/analyze/groundsync/target_kv_decision_report_2026-09-03.md`.

## SyncSpec final audit — 2026-09-02

- Đã sửa selector training để dùng toàn bộ batch/lattice, giữ đúng serving
  Top-M và mask candidate miss; không chèn target token vào candidate set.
- Stage survival khi có target thật giờ thu nhãn bằng on-policy rollout qua
  target + native drafter + selector + exact verifier. `--stage joint` chạy đủ
  diffusion → selector → survival và lưu các checkpoint tương ứng.
- Preflight B200 resolve được HF repo ID từ snapshot cache offline, kiểm tra
  tokenizer/config và tương thích `vocab_size`/`hidden_size` giữa target và
  drafter; engine đồng bộ CUDA quanh timing component để profile B200 đo
  wall-clock đúng hơn.
- Fresh validation: **158 passed, 2 skipped, 17 warnings**; compileall,
  `bash -n` và `git diff --check` pass. CPU infer/profile/dispatcher smoke đều
  pass; profile có đủ `draft`, `selector`, `survival`, `verify`, `scheduler`.
- Preflight hiện tại vẫn `BLOCKED: hardware_unavailable` vì host là Tesla T4,
  CUDA false; chưa được phép claim B200 real-model smoke cho tới khi chạy trên
  canonical server.

## SyncSpec post-audit verification — 2026-09-02

- Sửa API override `max_new_tokens=0`: giá trị zero giờ được tôn trọng; giá
  trị âm bị từ chối rõ ràng thay vì bị thay bằng config mặc định.
- Regression suite mới nhất: **165 passed, 3 skipped, 21 warnings** trong
  3:29. Compileall, shell syntax và `git diff --check` đều pass.
- Suite đã bao phủ thêm cache trajectory append/resume idempotence, ROUGE
  record/summary, real-target selector/survival/joint CLI, real Transformers
  profile và đường dtype bfloat16 cho native drafter + engine.
- Cảnh báo còn lại là CUDA/NVML trên máy T4 và DeepSpeed/Triton ghi autotune
  cache read-only lúc atexit; không có test failure. B200 smoke thật vẫn
  pending canonical server.
- Đã bổ sung train-phase B200 preflight và wrapper smoke toàn chuỗi tạo
  checkpoint mới, cùng kiểm tra cache writable và compute capability. Trainer
  hỗ trợ accumulation, clipping, AMP BF16 trên CUDA, seed và opt-in Stage-4
  low-LR joint refinement; empirical gate calibration đã có CLI riêng.

## SyncSpec final regression — 2026-09-02

- Thêm compact tied-target checkpoint: khi train với target thật, checkpoint
  không serialize embedding/LM head frozen; metadata buộc tie lại trước
  forward. Standalone native checkpoint vẫn tương thích.
- Thêm Stage-4 empirical gate calibration (`calibrate_syncspec_gate.py`) và
  nạp bảng qua `--gate-table`; thêm validation cho `max_new_tokens=0` và
  trajectory limit zero.
- Full regression cuối: **175 passed, 5 skipped, 23 warnings** trong 3:37;
  compileall, shell syntax và `git diff --check` pass. CUDA-conditional tests
  skip trên T4 vì `torch.cuda.is_available() == false`, không được tính là
  GPU pass.
- Preflight local ghi `BLOCKED` đúng vì hardware unavailable; đồng thời phát
  hiện cache local read-only. B200 train/infer smoke wrapper đã sẵn sàng nhưng
  vẫn cần chạy trên canonical server để có evidence GPU thật.
- Train wrapper đã thêm infer-phase preflight sau khi tạo checkpoint, kiểm tra
  lại drafter/selector/survival trước bước infer.

## SyncSpec handoff hardening — 2026-09-02

- Engine và output đã nhận `batch_size` hiệu dụng để pre-gate/profile lookup
  không bỏ qua trục batch. (Giới hạn profiler batch>1 ở bản ghi handoff cũ đã
  được thay thế bởi microbatch engine ở audit bên dưới.)
- B200 train wrapper hiện có hai preflight artifact: train trước Stage 0 và
  infer sau joint checkpoint. Contract tests sau thay đổi đạt `7 passed`;
  compileall, `bash -n`, `git diff --check` vẫn pass.
- Đây là handoff-ready reference path, chưa phải bằng chứng benchmark batch
  hoặc B200: hardware test thật vẫn phải chạy trên canonical server.

## SyncSpec microbatch serving audit — 2026-09-02

- Đã bổ sung `SyncSpecEngine.generate_batch`: group theo full-context/source
  length, draft native theo `[B,K_d]`, target verify theo `[B,K_v]` với cache
  stack tạm thời, rồi commit độc lập vào cache từng request. Mixed prompt
  lengths được regroup; adapter legacy/synthetic có scalar fallback giữ exactness.
- CLI inference giờ gọi batch engine khi `--batch-size > 1`; profiler cũng đo
  và lưu profile `batchN`, không còn chỉ ghi batch metadata hoặc chặn batch >1.
- Regression mới: native target/drafter batch, equal/mixed prompt lengths,
  synthetic CLI microbatch và batch profile đều pass. Fresh SyncSpec suite sau
  audit: **60 passed, 5 skipped, 22 warnings**; toàn bộ repo đạt **180 passed,
  5 skipped, 23 warnings**; warnings còn lại do CUDA/NVML
  T4 và Triton/DeepSpeed atexit không ghi được cache read-only.
- Prefill hiện vẫn tuần tự theo request; chỉ draft/block verification được gom
  batch. Vì vậy profile B200 phải đo đúng batch/context/model/checkpoint trước
  khi kết luận throughput continuous serving.
- Train-smoke wrapper đã truyền tiếp `BATCH_SIZE` vào infer cuối và có contract
  test tương ứng; cấu hình batch trong master không còn bị bỏ qua.
- Stage 1 giờ sample anchor reproducible ở mỗi optimizer step; hỗ trợ cached
  teacher KL, Top-M rank-margin và exponential position weighting tùy chọn.
  Stage 2 dùng toàn bộ anchor của trajectory và Stage 3 riêng có thể nạp
  selector checkpoint đã train.
- Exact target transaction đã được tối ưu: all-accepted verification tái sử
  dụng cache/logits/hidden đã tính, batch DynamicCache được slice theo request;
  regression Llama Transformers 5 xác nhận next logits khớp full recompute.

## SyncSpec final verification — 2026-09-02

- Sửa profiler để `ProfileKey.context_bin` lấy từ độ dài prompt sau
  tokenization; batch profile từ chối input ragged thay vì ghi profile sai
  context. Engine cũng bỏ qua profile có context bin không khớp.
- Full repository validation mới nhất: **192 passed, 5 skipped, 23 warnings**;
  `compileall`, `bash -n` và `git diff --check` đều pass.
- Local CPU synthetic inference/profile, microbatch path, staged training
  contracts, tiny Transformers path và exact DynamicCache transaction đều
  pass. Năm skip vẫn là CUDA-conditional trên host Tesla T4.
- Handoff hardening bổ sung target bonus-token commit sau block accept, yêu cầu
  measured profile cho CUDA/Transformers trước khi bật speculation, profile
  target-AR/p95/peak-memory, optimizer resume, positional offset/dynamic gate,
  candidate-dependent selector coherence và trajectory metadata/fingerprint.
- Không có thay đổi nào biến CPU/T4 thành B200 evidence. Strict B200 preflight
  và train/infer smoke thật vẫn cần chạy trên canonical B200 với model, data,
  checkpoint/cache local; trạng thái external hiện là `BLOCKED` vì CUDA không
  khả dụng và cache local read-only.

### Anchor-Offset và GPU smoke contract — 2026-09-02

- Stage 1, selector và joint training truyền vị trí tuyệt đối `prompt_len +
  anchor` theo từng dòng batch; trajectory metadata lưu
  `anchor_position_offsets`.
- Drafter dùng learned sentinel embedding cho slot `[MASK]`, không dùng nhầm
  embedding token cuối vocabulary target; loader vẫn nhận checkpoint legacy.
- Thêm `scripts/run_syncspec_cuda_smoke.sh`: chạy synthetic microbatch trên
  CUDA thật, hoặc trả structured `BLOCKED` exit code 2 khi host không có CUDA.

### Verification sau hardening — 2026-09-02

- Full repository regression: **196 passed, 5 skipped, 23 warnings** trong
  3:47; các skip là CUDA-conditional trên host T4.
- Anchor-Offset, learned mask sentinel, legacy checkpoint loading và CUDA
  smoke guard đều có test pass; chưa có GPU/B200 runtime evidence.

### Regression sau offset-topology fix — 2026-09-02

- Full repository: **197 passed, 5 skipped, 23 warnings** trong 3:47.
- Đã kiểm tra riêng trường hợp Stage 2 rebinding `anchors=[anchor]` không lấy
  nhầm offset từ metadata của anchor cũ.

### Production preflight gate — 2026-09-02

- Transformers production inference bắt buộc selector/survival checkpoint;
  có cờ explicit `--allow-untrained-components` chỉ cho development.
- B200 preflight kiểm tra model weights, drafter weights, selector/survival
  artifacts và schema measured profile trước khi cho phép infer.
- Full regression sau gate: **200 passed, 5 skipped, 23 warnings** trong 3:51.

### Budget/profile consistency — 2026-09-02

- Inference nhận `--kd/--kv` explicit; Transformers production mặc định 16/8,
  còn synthetic giữ profile 4/2 và 4/4 khi không override.
- `run_syncspec.sh`, B200 smoke và B200 train-smoke truyền cùng budget đã dùng
  để profile, tránh profile 16/8 nhưng engine chọn candidate 4/2 rồi fallback
  AR âm thầm. Master config có `SYNCSPEC_KD/SYNCSPEC_KV`.
- Full repository sau fix: **201 passed, 5 skipped, 23 warnings**. Compileall,
  shell syntax và `git diff --check` pass; wrapper synthetic ghi đúng budget
  explicit. CUDA smoke trên host T4 trả `BLOCKED` exit 2 như thiết kế.
- B200 real-model/profile/train/infer vẫn pending canonical server; strict
  preflight local tiếp tục `BLOCKED` vì CUDA, asset và cache writable.

### Training memory hardening — 2026-09-03

- Theo TDD, bổ sung `--train-batch-size` và mini-batch thật cho diffusion,
  selector, survival và joint fine-tuning; selector stage còn chunk luôn
  forward tạo Top-M lattice để không materialize toàn bộ long-context cache
  trong một activation graph.
- Wrapper B200 train-smoke truyền `SYNCSPEC_TRAIN_BATCH_SIZE`, master example
  mặc định `1`; docs ghi rõ chỉ tăng sau khi profile memory/throughput.
- Regression mới: microbatch unit/integration và tiny Transformers; full
  repository **214 passed, 5 skipped, 24 warnings**. `compileall`, `bash -n`
  và `git diff --check` pass.
- Một regression trung gian do selector dùng nhầm hidden chunk cuối đã được
  tái hiện, sửa và xác minh lại bằng targeted test và full suite.
- Không thay đổi kết luận phần cứng: CUDA smoke trên host Tesla T4 vẫn
  `BLOCKED`, chưa có train/profile/infer evidence thật trên canonical B200.
- Smoke chain độc lập sau patch cũng pass: Stage 0 → joint với cả bốn summary
  batch size bằng `1` → synthetic batch infer `2/2 ok` → measured profile
  `K_d=4,K_v=2`.

### Survival semantics audit — 2026-09-03

- Sửa lỗi trainer dùng BCE trực tiếp trên hazard với nhãn cumulative survival;
  loss hiện tính đúng `S_j=prod_i(1-h_i)` theo design v1.1, và calibration
  cũng đánh giá `head.survival(...)` thay vì raw hazard.
- TDD regression mới chuyển full repository thành **215 passed, 5 skipped,
  24 warnings**; compileall, shell syntax và diff check vẫn pass.
- Smoke chain mới sau fix xác nhận joint training đủ diffusion/selector/
  survival/refinement, bốn stage batch size `1`, inference batch `2/2 ok` và
  measured profile `K_d=4,K_v=2`.
- B200 GPU evidence vẫn là phần external: máy hiện tại T4/CUDA unavailable,
  `/workspace` chưa mount.

### Final contract hardening — 2026-09-03

- Bổ sung fail-fast cho `top_m <= 0` ở cả `SyncSpecDrafterConfig` và
  `top_m_candidates`; regression tests đã xanh.
- Full repository sau patch: **215 passed, 5 skipped, 24 warnings**; thêm
  `compileall`, `bash -n` cho các launcher SyncSpec và `git diff --check`, đều
  pass.
- Kết luận không đổi: CPU end-to-end đã có evidence; train/profile/infer thật
  trên canonical B200 vẫn là gate external chưa được xác minh.

### Long-context position hardening — 2026-09-03

- Drafter không còn âm thầm modulo-wrap `position_offset`; scalar và per-row
  offsets phải nằm trọn trong `max_positions`, nếu không sẽ fail-fast.
- Regression mới và full repository sau patch: **216 passed, 5 skipped, 24
  warnings**.
- Static checks vẫn pass; B200 GPU evidence tiếp tục cần chạy trên server thật.

### Post-hardening smoke evidence — 2026-09-03

- CPU synthetic inference fresh sau patch: `status=ok`, `records=1`.
- CPU measured profile fresh: `batch2`, `K_d=4,K_v=2`, đầy đủ component timing;
  static checks tiếp tục pass.
- GPU-path launcher fresh trên host hiện tại trả structured
  `status=BLOCKED, reason=cuda_unavailable`, exit `2`; đây là bằng chứng
  môi trường chưa đủ, không phải GPU pass.

### Context-safe serving and staged-train contracts — 2026-09-03

- Engine đọc context headroom từ target adapter và tự cap `max_new_tokens`;
  batch cap riêng từng request, tránh target commit/drafter vượt context.
- CLI train fail-fast nếu selector thiếu diffusion checkpoint hoặc survival
  thiếu diffusion/selector checkpoint, ngăn pipeline production dùng component
  random ngoài ý muốn.
- Regression sau hai thay đổi: **220 passed, 5 skipped, 24 warnings**.

### Context-safe trajectory/profile completion — 2026-09-03

- Stage 0 và vanilla-AR reference cũng cap theo target context headroom; không
  còn đường phụ nào cố commit/generate vượt context.
- Full repository sau toàn bộ hardening: **222 passed, 5 skipped, 24 warnings**.
- Fresh CPU inference/profile và static checks pass; CUDA launcher vẫn trả
  structured `BLOCKED` vì host dev không có CUDA.

### Latest verification handoff — 2026-09-03

- Full suite cuối cùng trong session: **222 passed, 5 skipped, 24 warnings**;
  exit code `0`.
- Fresh CPU synthetic infer (`records=1`) và CPU measured profile (`batch2`,
  `K_d=4,K_v=2`) pass sau toàn bộ context/checkpoint hardening.
- `run_syncspec_cuda_smoke.sh` trả `status=BLOCKED, reason=cuda_unavailable`,
  exit `2`; canonical B200 train/profile/infer vẫn là external gate duy nhất.

### Anchor-only trajectory cache — 2026-09-03

- Stage 0 chỉ serialize target hidden features tại các anchor được chọn,
  kèm `target_feature_positions`; loader/trainer fallback tương thích cache
  cũ lưu feature theo mọi target position.
- Mục tiêu là tránh phình RAM/disk không cần thiết trên long-context; full
  regression sau patch: **223 passed, 5 skipped, 24 warnings**.

### Binary trajectory cache — 2026-09-03

- Bổ sung `TrajectoryCache` format `.pt/.pth/.torch` với schema/fingerprint,
  atomic write và resume theo `sample_id`; JSONL vẫn là format mặc định tương
  thích ngược.
- `_load_records` và CLI train đã đọc được cả hai format; full suite sau patch
  **225 passed, 5 skipped, 24 warnings**.
- Fresh CLI Stage 0 binary smoke tạo thành công cache schema `1` với `2` records;
  static checks pass.

### B200 wrapper binary-cache default — 2026-09-03

- `run_syncspec_b200_train_smoke.sh` mặc định dùng
  `outputs/.../trajectories.pt`, nhưng vẫn cho phép override
  `SYNCSPEC_TRAIN_TRAJECTORY` sang JSONL.
- Contract test launcher và static checks pass; full regression giữ ở
  **225 passed, 5 skipped, 24 warnings**.

### Full-block survival-label rollout — 2026-09-03

- Stage 3 on-policy rollout có `force_kv=K_d`, thu label cho toàn bộ draft
  block bằng exact verifier thay vì phụ thuộc survival head random chưa train.
- Fresh CPU `.pt` Stage 0 → joint diffusion/selector/survival → optional
  refinement pass; summary ghi đủ bốn stage và `batch_size=1`.
- Full repository sau patch: **226 passed, 5 skipped, 24 warnings**.

### Runtime exactness guard — 2026-09-03

- Thêm `--check-exactness` cho inference greedy: chạy vanilla target-AR độc
  lập trên cùng input, ghi `exact_match_vanilla_ar` per record và fail process
  nếu có mismatch.
- Hai B200 smoke wrapper tự bật guard này; wrapper inference thường không bật
  để không làm sai timing benchmark. Stochastic mode không dùng guard token-
  equality này.
- TDD red → green: test mới ban đầu fail do CLI chưa có cờ, sau triển khai
  đạt `4 passed`; các static/contract checks liên quan cũng pass.
- Full repository sau guard: **229 passed, 5 skipped, 24 warnings**; `compileall`,
  shell syntax và `git diff --check` pass. CUDA launcher vẫn trả structured
  `BLOCKED/cuda_unavailable` trên host T4.

### B200 master-cache default audit — 2026-09-03

- Phát hiện master example còn override fallback wrapper bằng
  `trajectories.jsonl`; đã sửa thành `trajectories.pt` để đường chạy B200 mặc
  định thực sự dùng binary cache.
- Vẫn giữ `SYNCSPEC_TRAIN_TRAJECTORY` làm override rõ ràng cho JSONL; shell
  expansion được kiểm chứng ra đúng `.../trajectories.pt`.
- Thêm launcher contract test; test pass cùng static checks.

### Runtime feedback loop — 2026-09-03

- Bổ sung request-local EMA cho acceptance prefix và latency từng component;
  state được ghi vào `InferenceResult.runtime_feedback`/output record.
- Pre-draft gate dùng prior đã điều chỉnh theo acceptance quan sát được; test
  zero-acceptance xác nhận request chuyển sang AR fallback ở các round sau.
- Bổ sung empirical stochastic rejection/residual test với 12.000 mẫu, target
  distribution khớp trong `±0.025`.
- Fresh targeted tests pass; full regression sau patch: **232 passed, 5
  skipped, 24 warnings**.

### Runtime profile provenance/schema guard — 2026-09-03

- B200 preflight và engine hiện yêu cầu profile có `schema_version=1` cùng
  `source="measured"`; profile synthetic/không rõ nguồn bị từ chối.
- Thêm regression tests cho schema/provenance ở preflight và CUDA engine.
- Full regression sau hardening: **236 passed, 5 skipped, 24 warnings**.
- GPU evidence vẫn chờ canonical B200: host hiện tại là T4 CPU-only.

### Post-draft measured AR utility gate — 2026-09-03

- Controller hiện nhận `target_ar` measured per-token và so sánh với expected
  committed tokens/round; speculative utility thua safety margin sẽ fallback
  đúng một token AR.
- Profile CLI đổi baseline AR sang `max_new_tokens=1` và ghi
  `target_ar_tokens=1`; helper vẫn đọc profile cũ bằng fallback chia theo `kv`.
- Post-draft fallback ghi acceptance zero vào runtime feedback; round kế tiếp
  chuyển sang AR thay vì lặp lại draft cost.
- Thêm regression cho scalar controller/engine và profile measurement; full
  repository sau sửa batch regression: **241 passed, 5 skipped, 25 warnings**.
- GPU evidence vẫn chờ canonical B200: host hiện tại là T4 CPU-only.

### Stage-0 artifact provenance hardening — 2026-09-03

- Bổ sung fingerprint bounded cho artifact local: file nhỏ được hash toàn bộ,
  shard lớn dùng manifest kích thước cùng chunk đầu/cuối để không reread toàn
  bộ model long-context khi tạo cache key.
- Stage 0 nhận `--seed`, đưa seed và fingerprint artifact vào cache key/metadata;
  fingerprint unresolved model ID vẫn ổn định nhưng docs yêu cầu snapshot local
  canonical trên server.
- TDD targeted pass; full regression sau patch: **243 passed, 5 skipped, 25
  warnings**. B200 real-model train/profile/infer vẫn đang chờ external handoff
  vì host hiện tại không có CUDA.

### B200 train seed handoff — 2026-09-03

- Thêm `SYNCSPEC_TRAIN_SEED` vào master example và truyền cùng seed vào Stage 0
  lẫn `train_syncspec.py`, giúp trajectory/checkpoint smoke tái lập trên B200.

### Long-context hidden-state memory hardening — 2026-09-03

- Transformers adapter capture final hidden qua final-normalization hook cho
  prefill, block verify và rejection commit; tránh yêu cầu all-layer hidden
  states trong đường chạy chuẩn Llama/Qwen.
- Compatibility fallback vẫn giữ cho model local custom không có final norm;
  regression adapter/native E2E pass, nhưng B200 real run vẫn là bằng chứng
  cần chạy trên canonical server.
- Full repository final sau hardening: **243 passed, 5 skipped, 25 warnings**;
  compileall, shell syntax và `git diff --check` pass.

### Target-derived source-memory cache + profiler boundary — 2026-09-03

- Phát hiện train target thật đang dựng source memory từ token embedding,
  khác với final-hidden descriptors mà serving dùng.
- Stage 0 nay có `--include-source-memory`, lưu mean-pooled descriptor theo
  source chunk trong cả JSONL/`.pt`; train Stage 1/2/joint dùng lại descriptor
  này, cache cũ vẫn có embedding fallback tương thích.
- B200 train wrapper bật source-memory cache mặc định và dùng chung
  `SYNCSPEC_SOURCE_CHUNK_SIZE`.
- Sửa profiler đồng bộ CUDA sau prefill trước khi đo `target_ar`, tránh tính
  nhầm phần prefill còn pending vào opportunity cost AR; thêm regression thứ tự
  đồng bộ.
- Targeted source-memory/profile regression pass; full repository cần chạy lại
  sau patch. GPU/B200 real evidence vẫn chờ canonical server.

### Regression sau source-memory alignment — 2026-09-03

- Full repository: **247 passed, 5 skipped, 25 warnings** trong `163.10s`.
- Tiny Transformers joint train với optional refinement → measured profile →
  exactness inference pass; wrapper synthetic exactness cũng pass.
- CUDA smoke vẫn trả `BLOCKED/cuda_unavailable` (exit 2); B200 preflight strict
  vẫn `BLOCKED` vì host không có CUDA và không có asset/cache canonical.

### Profile-aware pre-draft budget selection — 2026-09-03

- Phát hiện pre-gate cũ dùng một gain tổng rồi luôn chọn `K_d` lớn nhất, chưa
  phản ánh lựa chọn utility giữa các profile theo §66–67.
- Calibrator nay giữ trục `K_d` khi trace có `kd`/`budget.kd`, với key dạng
  `context:batchN:kdM`; pre-gate chọn gain lớn nhất vượt safety margin.
- Khi bảng đã có profile-specific entries, `K_d` chưa được đo không dùng
  `default_gain` để tự động bật; bảng cũ không có `kd` vẫn tương thích.
- Engine truyền prior riêng theo từng `K_d`, kể cả khi gom batch; controller
  regression pass. Full suite cuối cần chạy lại sau thay đổi này.

### Final regression sau profile-aware gate — 2026-09-03

- Full repository: **252 passed, 5 skipped, 25 warnings** trong `162.51s`.
- CPU synthetic wrapper với `--check-exactness`: `status=ok`,
  `records=1`, `exactness_failures=0`.
- `compileall`, `bash -n` và `git diff --check` pass.
- CUDA smoke đúng contract trả exit `2`, `status=BLOCKED`,
  `reason=cuda_unavailable`; strict B200 preflight trả `BLOCKED` vì hardware
  và asset/cache canonical chưa có trên host dev.

### Final adaptive-profile/batch-cost audit — 2026-09-03

- Sau khi bổ sung calibration bảo thủ theo profile và fallback sang profile có
  measurement, full repository đạt **263 passed, 5 skipped, 25 warnings**
  trong `164.71s`.
- CPU synthetic wrapper với finite profiles adaptive và `--check-exactness`
  đạt `status=ok`, `records=1`, `exactness_failures=0`.
- `compileall`, `bash -n` và `git diff --check` đều pass. CUDA smoke vẫn trả
  đúng `BLOCKED/cuda_unavailable` (exit `2`); strict B200 preflight trả
  `BLOCKED` vì host dev không có CUDA và chưa mount asset/cache canonical.
- Các phần còn thiếu duy nhất là external evidence: chạy CUDA smoke, train
  Stage 0 → joint drafter/selector/survival, profile từng `(K_d,K_v)`, rồi
  inference exact trên canonical B200.
- Chạy trực tiếp cả `run_syncspec_b200_train_smoke.sh` và
  `run_syncspec_b200_smoke.sh` với example master đã dừng ở strict preflight,
  trả JSON `status=BLOCKED` trước khi tạo/chạm vào checkpoint.
- Sửa launcher B200 dùng `KD/KV` sau normalize của `config.sh`; regression
  mới bảo đảm override fixed profile không xung đột với `BUDGET_PROFILES`.

### CPU full-chain smoke runner — 2026-09-03

- Thêm `scripts/run_syncspec_cpu_smoke.sh` để chạy một lệnh toàn chuỗi:
  Stage 0 `.pt` → joint train → kiểm tra drafter/selector/survival artifacts →
  multi-profile CPU measurement → batch inference và `--check-exactness`.
- Fresh run với `2` samples, `1` training step, profiles `4:2,4:4` đạt
  `status=ok`, `records=2`, `exactness_failures=0`; checkpoint có đủ ba file
  component.
- Full regression sau khi runner source shared config/runtime: **263 passed,
  5 skipped, 25 warnings**.

### Standalone staged-training audit — 2026-09-03

- Fresh CPU run riêng đã nối Stage 0 → diffusion → selector → survival; mỗi
  stage tạo artifact hợp lệ (`pytorch_model.bin`, `selector.pt`,
  `survival.pt`) với fingerprint cache nhất quán.
- Loss diffusion/survival hữu hạn; đây chỉ là wiring smoke, không phải bằng
  chứng chất lượng hoặc throughput B200.

### Real Qwen3 offline Stage-0 audit — 2026-09-03

- Snapshot local `Qwen3-0.6B` chạy thành công qua Transformers offline trên
  một mẫu thật; cache `.pt` đọc lại qua fingerprint guard.
- Artifact có `1` record, `128` source tokens, `1` target token, `2`
  source-memory chunks và `source_memory_source=target_final_hidden`.
- Đây là model-architecture/adapter evidence trên CPU; B200 GPU evidence vẫn
  chờ canonical server.

### Real Qwen3 offline train/profile/infer audit — 2026-09-03

- Qwen3-0.6B local đã chạy trọn Stage 0 → joint train → checkpoint → measured
  profile → Transformers inference trên CPU.
- Inference trả `status=ok`, `exactness_failures=0` và khớp vanilla AR; profile
  có `target_ar_tokens=1` cùng component costs.
- Đây là offline CPU validation, không thay thế GPU/B200 evidence.

### Transformers dtype loading hardening — 2026-09-03

- Adapter dùng `dtype` trên Transformers 5 và fallback có điều kiện cho
  runtime legacy; test cả hai nhánh đều pass.
- Full regression sau patch: **265 passed, 5 skipped, 25 warnings**.
- Qwen3-0.6B inference re-run sau hardening pass exactness; warning
  `torch_dtype` đã biến mất.

### B200 batch/sample smoke contract — 2026-09-03

- Thêm fail-fast guard cho hai B200 wrapper khi `MAX_SAMPLES` nhỏ hơn
  `BATCH_SIZE`; direct test và contract test đều pass với exit `2` rõ ràng.
- Full regression sau guard: **266 passed, 5 skipped, 25 warnings**.

### B200 preflight exit-code contract — 2026-09-03

- Chuẩn hóa `--strict`: `PASS=0`, `BLOCKED=2`, `FAIL=1`; structured JSON
  report vẫn được ghi đầy đủ.
- Regression preflight không-CUDA và full suite đều pass (`266 passed, 5
  skipped, 25 warnings`).

### Real-model batch exactness audit và CUDA fail-fast — 2026-09-03

- Chạy batch serving offline với Qwen3-0.6B local, checkpoint drafter/
  selector/survival đã train: hai request cùng batch đều khớp vanilla AR,
  `batch_size=2`, không có token sai.
- Bổ sung guard cho inference, trajectory builder và training CLI: truyền
  `--device cuda` trên host không có CUDA phải dừng với lỗi rõ ràng, không được
  âm thầm chuyển sang CPU. Regression nhóm CLI/Transformers: **31 passed**.
- Đây vẫn là bằng chứng CPU/offline; CUDA smoke và train/profile/infer thật
  trên canonical B200 chưa thể chạy vì driver/asset server chưa hiện diện.

### Final local regression sau GPU-integrity hardening — 2026-09-03

- Full regression: **270 passed, 5 skipped, 25 warnings**.
- `git diff --check`, `compileall`, `bash -n` cho toàn bộ SyncSpec launcher
  đều pass.
- CUDA smoke thực tế trả JSON `status=BLOCKED`, exit `2`; strict B200
  preflight cũng trả `status=BLOCKED`, exit `2` trên host không có CUDA.

### CPU full-chain recheck sau hardening — 2026-09-03

- Fresh runner `/tmp/syncspec_cpu_smoke_final_gpu_guard` chạy lại Stage 0 →
  joint train → multi-profile profile → batch inference.
- Kết quả `status=ok`, `records=2`, `exactness_failures=0`; checkpoint đủ
  drafter/selector/survival và profile ghi `target_ar_tokens=2` cho batch 2.

### B200 profile provenance guard — 2026-09-03

- Preflight nay đối chiếu measured profile với model, drafter checkpoint,
  GPU, precision và batch đang chạy; profile CPU/khác regime bị loại trước
  inference.
- Thêm contract test và nối `--precision`/`--batch-size` vào cả B200 infer
  smoke và train-smoke preflight.
- Targeted B200/launcher regression: **22 passed**.

### Final regression sau profile provenance guard — 2026-09-03

- Full regression mới nhất: **271 passed, 5 skipped, 25 warnings**.
- Compileall, shell syntax và `git diff --check` tiếp tục pass sau khi nối
  profile validation vào launcher.
- Preflight với đầy đủ Qwen artifact nhưng profile CPU báo đúng
  `runtime_profile_invalid` cùng `hardware_unavailable`.

### Production profiler component guard — 2026-09-03

- Profiler Transformers nay yêu cầu selector và survival checkpoint đã train;
  chỉ cờ diagnostic `--allow-untrained-components` mới cho phép component
  ngẫu nhiên.
- Cập nhật test full local-profile path; targeted CLI/profile regression:
  **26 passed**.

### Final verification after production profiler guard — 2026-09-03

- Full repository regression: **272 passed, 5 skipped, 25 warnings**.
- `compileall`, `bash -n` cho toàn bộ SyncSpec launchers và `git diff --check`
  đều pass.
- Warning còn lại chỉ là môi trường dev: driver CUDA cũ/không có NVML và
  DeepSpeed/Triton atexit không ghi được cache read-only; không làm test fail.
- Canonical B200 execution vẫn là external handoff chưa thực hiện được trên
  host hiện tại.

### Profile provenance và single-round measurement hardening — 2026-09-03

- Profile key nay lưu cả selector/survival checkpoint; engine và B200
  preflight từ chối profile đo từ component khác.
- Profile diagnostic tạo bởi `--allow-untrained-components` được đánh dấu
  `source="diagnostic"`, không thể gate production CUDA.
- Profiler ép `force_kv` và `max_rounds=1`, bảo đảm cost component là của đúng
  một serving round, không cộng dồn nhiều vòng decode.
- CUDA smoke probe tôn trọng `FAST_INFER_VENV` thay vì probe nhầm interpreter.
- Full regression sau hardening: **277 passed, 5 skipped, 25 warnings**;
  CPU full-chain vẫn pass với exactness failure bằng `0`.

### Implementation audit mở rộng — 2026-09-02

- Selector Stage 2 đã được nối source-memory và đúng prefix target trước khi
  huấn luyện; selector có learned token-pair embeddings và schedule
  teacher-forcing/self-conditioning.
- Đã sửa EOS giữa block trên cả scalar/batch path; test xác nhận không append
  token hoặc bonus sau EOS.
- Phát hiện controller đang chỉ dùng `verify` thay vì tổng draft/selector/
  survival/verify/scheduler theo profile; bước tiếp theo là sửa bằng test
  regression, giữ compatibility với profile legacy.

### Implementation audit hoàn tất — 2026-09-02

- Đã hoàn tất round-cost extraction từ các component draft/selector/survival/
  verify/scheduler, giữ tương thích profile legacy verify/e2e.
- Đã giới hạn candidate `K_v` theo phần ngân sách còn lại và chỉ dùng measured
  profile đúng `K_v`; nếu không có thì pre-gate chuyển về AR.
- Đã sửa AR fallback dừng ngay tại EOS, cùng với EOS trim đã có ở scalar/batch
  speculative path.
- Full repository mới nhất: **212 passed, 5 skipped, 23 warnings**; static
  checks pass. CUDA smoke exit 2 với JSON `status=BLOCKED` do CUDA unavailable.
- Đã thêm validation fail-fast cho `heads/groups/max_positions` của drafter và
  preflight compatibility cho `max_positions` so với target context capacity,
  đồng thời đối chiếu width/vocab của selector.
- B200 train/profile/infer thật vẫn chưa chạy trong session vì host dev T4,
  CUDA false và asset/cache canonical không được mount.

### Evidence handoff — 2026-09-03

- Fresh CPU chain đã chạy thành công Stage 0 → joint training → batch infer →
  profile; checkpoint có drafter, selector và survival artifacts.
- Full regression mới nhất: **213 passed, 5 skipped, 23 warnings**; compileall,
  shell syntax và `git diff --check` pass.
- CUDA smoke trả `BLOCKED` exit 2; `nvidia-smi` chỉ thấy Tesla T4 và `/workspace`
  canonical B200 không được mount. Preflight cũng yêu cầu tokenizer artifact
  thực. Không dùng kết quả này làm GPU evidence.
