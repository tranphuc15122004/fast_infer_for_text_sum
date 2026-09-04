# Kế hoạch: kiểm chứng hypothesis GroundSync

## Mục tiêu

Thực hiện các thí nghiệm kiểm chứng proposal GroundSync bằng một model Qwen3
đã có/được cache, ghi code và kết quả dưới `src/analyze`, và hoàn thiện báo cáo
đánh giá có tiêu chí go/no-go rõ ràng cho từng hypothesis.

## Trạng thái

| Pha | Trạng thái | Bằng chứng cần có |
|---|---|---|
| 1. Khảo sát repo và dữ liệu | hoàn tất | danh sách code/data/model/runtime hiện có |
| 2. Thiết kế experiment | hoàn tất | design doc + protocol + validation scope |
| 3. Pipeline analysis | hoàn tất | unit/synthetic smoke và CLI reproducible |
| 4. Chạy experiment | hoàn tất P0/P1 và P1 strong/P2 direct; P2 serving unavailable | P0 trên GovReport/CNN-DM/Multi-News; cheap predictor; EAGLE-3 strong-drafter; paired direct E2E |
| 5. Báo cáo | hoàn tất | Báo cáo master tại `src/analyze/groundsync/final_decision_report_2026-09-02.md`, kèm P0/P1/P2 artifacts |

## Ràng buộc đã biết

- Code và artifact phân tích phải nằm dưới `src/analyze`.
- Model được phép: Qwen3-4B, Qwen3-1.7B hoặc Qwen3-0.6B.
- Không được gọi kết quả “pass” nếu chưa có số liệu và kiểm tra fresh.
- Dùng một target-only canonical trace để đo source-state và một pipeline
  speculative/controlled phù hợp với model cache thực tế.
- Model-backed GPU đã chạy với Qwen3-4B canonical target và Qwen3-0.6B draft
  bằng `/home/tuantb/miniconda3/bin/python3` ngoài venv trên `cuda:0`; không
  dùng `.venv` cho thực nghiệm T4.

## Errors Encountered

- EAGLE smoke ban đầu lỗi reference list trong script benchmark cũ; runner mới
  không phụ thuộc reference và đã chạy lại thành công.
- Nhánh naive của runner mới ban đầu unpack sai số giá trị trả về; đã sửa và
  validation smoke/full pass.
- Serving API không thể chạy: canonical server mount không có trên host và
  runtime GPU miniconda không import được vLLM; artifact ghi `UNAVAILABLE`.

## Next

Đã hoàn tất design/protocol, core metrics, target trace adapter, controlled
speculative trace, report và orchestrator theo TDD. Bản mở rộng đã thêm
calibrated positional prior, chunk/sink sensitivity, E0 position relocation,
position-adjusted hazard coefficient với 2.000 document bootstrap, negative controls, adaptive/true-cost
policy, train/dev threshold selection và tách timing khỏi acceptance. Báo cáo
toàn bộ quy trình đã ghi tại
`src/analyze/groundsync/verification_report_2026-08-29.md`.
Kết luận run mở rộng: GovReport H1/H2/H3/H4 `FAIL`, H5 `UNAVAILABLE`; CNN
H1/H3/H4 `FAIL`, H2 `PASS`, H5 `UNAVAILABLE`. Hai regime chưa cho bằng chứng
ổn định để xác nhận claim tổng hợp. P0 decision extension đã hoàn tất:
P0-1 `MIXED`, P0-2 `FAIL`, P0-3 `PASS`, P0-4 `MIXED`, P0-5 `PASS`, với overall
decision `NO_GO_GROUNDSYNC_GENERAL; conditional BurstSpec follow-up only where
admission passes`. Sau khi chạy P1/P2, strong-drafter burstiness fail cả ba
regime và exact-match direct E2E là 94–98%, nên quyết định cuối được nâng lên
`NO_GO_BURSTSPEC_GENERAL`; chỉ giữ oracle headroom như ceiling phân tích.

## Tiếp nối: đọc SyncSpec-v1 — 2026-09-02

- Mục tiêu hiện tại: đối chiếu conversation share với
  `src/SyncSpec/SyncSpec_v1_design_complete.md` để xác định phương pháp đề
  xuất và phạm vi triển khai thực sự.
- Đã đọc phần đầu design doc: SyncSpec-v1 kết hợp target full-context exact,
  target-memory reuse, diffusion candidate generation, source-coherent
  selection, prefix-survival modeling, cost-optimal adaptive verification và
  exact target verification; tối ưu CTC/committed tokens thay vì acceptance
  rate đơn thuần.
- URL ChatGPT share chưa tải được qua web reader (`Cache miss`); cần thử các
  cách truy cập an toàn khác trước khi kết luận conversation không khả dụng.
- Không được sửa code trước khi làm rõ phạm vi và trình bày thiết kế ngắn để
  người dùng phê duyệt theo brainstorming gate.

### Trạng thái

Pha đọc/đối chiếu đang thực hiện.

### Next

Đọc mục lục/các phần contract, algorithm, training và evaluation; thử truy
cập share URL bằng endpoint phù hợp; sau đó tổng hợp gap với repo và đề xuất
phạm vi triển khai.

### Phát hiện truy cập conversation

HTML share đã lấy được bằng read-only request escalated và payload React đã
parse được. Conversation có nhiều revision; cần ưu tiên message v1.1/final và
đối chiếu với file repo, không nhập nhầm các proposal cũ DiffuRoute/GroundSync
vào implementation hiện tại.

### Pha đọc/đối chiếu: hoàn tất

- Đã đọc toàn bộ 124 section của design v1.1 theo các block có heading.
- Đã giải mã và đọc các message chính của conversation: tóm tắt SyncSpec,
  đánh giá mức hoàn thiện, bản v1.1, các oracle cũ và message pivot từ
  GroundSync/DiffuRoute.
- Đã kiểm tra repo: `src/SyncSpec` chỉ có design doc; asset DFlash nằm ở
  `externals/dflash` và benchmark adapter hiện hữu.
- Đã ghi kết quả chi tiết vào `findings.md` và `progress.md`.

### Quyết định phạm vi mới

Người dùng đã yêu cầu triển khai toàn bộ pipeline SyncSpec-v1, gồm end-to-end
inference và train process cho drafter. Không thu hẹp thành Milestone A; file
implementation plan đầy đủ nằm tại
`docs/superpowers/plans/2026-09-02-syncspec-pipeline.md`. Tiếp tục theo TDD,
giữ target full-context exact và tách rõ CPU/CUDA/B200 evidence.

### Trạng thái

Pha đọc/đối chiếu hoàn tất; pha triển khai subsystem bắt đầu.

### Next

Tạo contract tests cho config/schema/source evidence trước khi viết production
code; sau mỗi module chạy RED → GREEN, rồi ghép model/engine/training/CLI.

### SyncSpec implementation status — 2026-09-02

Đã hoàn thành code path v1 reference/native: config/schema, source n-gram và
bounded memory, native DFlash2-style drafter, selector q Top-M, survival,
two-level controller, greedy/stochastic exact verifier, Transformers 5
DynamicCache transaction, trajectory cache, train stages diffusion/selector/
survival, measured profile, inference CLI, dispatcher và B200 preflight/smoke.

Fresh evidence: full suite `153 passed, 2 skipped`; `compileall`, `bash -n` và
`git diff --check` pass; Stage 0 → train diffusion/selector/survival → infer,
profile và dispatcher synthetic smoke đều pass trên CPU. CUDA tests được giữ
conditional và skip trên host T4; chưa có quyền truy cập canonical B200 trong
session này nên B200 real-model smoke vẫn là bước external handoff bắt buộc.

P1 cheap predictor: GovReport `FAIL` (recovery 0%), CNN/DailyMail `PASS`
(71.2%), Multi-News `INCONCLUSIVE` vì test chỉ 2 documents. Multi-News P0
confirmatory: H2/H4/P0-5 `FAIL`, O3 headroom `+43.7%`, P0-4 tốt nhất `77.7%`.
EAGLE-3 direct runs 50 documents/dataset: speedup `1.819–1.905x` nhưng
exact-match `0.94–0.98`, không đạt lossless guardrail. Báo cáo đầy đủ tại
`src/analyze/groundsync/final_decision_report_2026-09-02.md`.

Final audit evidence: selector trains all batch lattices with serving Top-M and
candidate-miss masking; real-target survival uses on-policy exact-verifier
rollouts; `--stage joint` orchestrates diffusion/selector/survival; CUDA timing
is synchronized for measured component profiles; B200 preflight resolves local
HF snapshots and checks tokenizer/config/target-drafter widths. Full suite is
now **158 passed, 2 skipped, 17 warnings**; CPU infer/profile/dispatcher pass.
The current T4 host has CUDA unavailable, so strict B200 preflight remains
`BLOCKED` and real B200 smoke is still external handoff work.

### Kiểm chứng sau audit — 2026-09-02

- Đã sửa và kiểm thử override `max_new_tokens=0`, đồng thời chặn giá trị âm.
- Full regression hiện tại: **165 passed, 3 skipped, 21 warnings**; compileall,
  shell syntax và `git diff --check` pass.
- Đã xác nhận thêm real tiny-Transformers train/infer/profile, trajectory
  resume idempotence, ROUGE output, và bfloat16 native drafter/engine.
- B200 real-model smoke chưa chạy vì host hiện tại không có CUDA; canonical
  server vẫn là bước handoff bắt buộc.
- Đã bổ sung `--phase train` cho preflight, wrapper
  `scripts/run_syncspec_b200_train_smoke.sh`, kiểm tra cache writable/compute
  capability, empirical Stage-4 gate calibration và opt-in low-LR joint
  fine-tuning với accumulation/clipping/AMP/seed.
- Full regression cuối sau các thay đổi: **175 passed, 5 skipped, 23 warnings**;
  compileall, shell syntax và `git diff --check` pass. Compact tied-target
  checkpoint và zero-limit contracts cũng đã có test.
- Chưa thể đánh dấu GPU/B200 hoàn tất: host này không có CUDA; cần chạy
  `check_syncspec_b200.py --strict` và `run_syncspec_b200_train_smoke.sh` trên
  canonical server để thu evidence phần cứng thật.
- Wrapper train-smoke đã có thêm infer-phase preflight sau khi tạo
  checkpoint, kiểm tra lại đủ drafter/selector/survival trước infer.
- Đã hoàn tất microbatch decode: `generate_batch` gom draft/verify theo full
  context length, cache stack chỉ là transient và mỗi request commit độc lập;
  CLI/profile dùng đường batch thật. Prefill vẫn tuần tự, nên B200 profile phải
  dùng đúng batch/context trước khi claim throughput.
- Xác minh cuối sau microbatch: toàn bộ repo `180 passed, 5 skipped, 23
  warnings`; `compileall`, `bash -n` và `git diff --check` đều pass. B200 strict
  preflight trên host T4 vẫn `BLOCKED` vì CUDA unavailable/cache read-only.

### Final verification — 2026-09-02

- [x] Context-aware measured profile: tokenized context bin, homogeneous
  microbatch guard, engine context matching.
- [x] Full regression: `192 passed, 5 skipped, 23 warnings`; compileall,
  shell syntax, and diff check pass.
- [ ] External B200 train/infer evidence: pending canonical server because the
  current T4 host has no usable CUDA and cannot provide a valid GPU benchmark.

### Final hardening — 2026-09-02

- [x] Anchor-Offset propagated through Stage 1/2/joint per-row training
  positions and recorded in trajectory metadata.
- [x] Learned `[MASK]` sentinel separated from frozen target embedding with
  legacy checkpoint loading support.
- [x] Explicit CUDA synthetic microbatch smoke launcher with structured
  non-CUDA `BLOCKED` result.

### Verification sau hardening — 2026-09-02

- [x] Full regression: **196 passed, 5 skipped, 23 warnings**.
- [ ] Real B200 GPU train/infer/profile execution remains pending external
  canonical-server evidence.

### Regression sau offset-topology fix — 2026-09-02

- [x] Full repository: **197 passed, 5 skipped, 23 warnings**.
- [x] Selector anchor rebinding uses a safe fallback when stored offset
  metadata no longer matches the current anchor topology.

### Production preflight gate — 2026-09-02

- [x] Require trained selector/survival artifacts for Transformers production
  inference; retain explicit development escape hatch.
- [x] Validate target/drafter weights and measured-profile structure offline.
- [x] Full regression: **200 passed, 5 skipped, 23 warnings**.
- [ ] Execute actual B200 GPU train/infer/profile on canonical server.

### Budget/profile consistency — 2026-09-02

- [x] Add explicit `--kd/--kv` serving contract and shared master-config values.
- [x] Pass the same budget from profiler to B200 smoke inference; document the
  profile-match requirement and preserve synthetic defaults.
- [x] Full regression: **201 passed, 5 skipped, 23 warnings**; static checks
  pass and CPU wrapper budget propagation is verified.
- [ ] Execute actual B200 GPU train/infer/profile on canonical server.

### Implementation audit hoàn tất — 2026-09-02

- [x] Profile round cost cộng draft/selector/survival/verify/scheduler, có
  fallback tương thích profile legacy chỉ chứa verify/e2e.
- [x] Controller/pre-gate không chọn `K_v` vượt số token còn lại; profile
  measured không có đúng `K_v` cũng không được bật speculation.
- [x] AR fallback dừng tại EOS; scalar và batch speculative path đều không
  commit token/bonus sau EOS.
- [x] Full repository: **213 passed, 5 skipped, 23 warnings**; compileall,
  shell syntax và diff check pass. CUDA smoke vẫn trả BLOCKED trên host T4.
- [x] Drafter config fail-fast với `heads/groups/max_positions` không hợp lệ;
  B200 preflight chặn drafter có positional capacity thấp hơn target.
- [ ] Chạy train/profile/infer thật trên canonical B200 với model, data,
  checkpoint và cache local; đây là evidence phần cứng còn thiếu.

### Evidence handoff — 2026-09-03

- [x] Fresh CPU chain chạy Stage 0 → joint (diffusion/selector/survival và
  optional refinement) → batch inference → profile.

## Target-KV execution — 2026-09-03

- [x] Đọc proposal Target-KV và cập nhật design/implementation plan.
- [x] Kiểm tra cache Qwen3-4B + Qwen3-4B-DFlash-b16 và xác nhận T4 runtime
  ngoài `.venv`.
- [x] Triển khai E0 failure-map runner/analyzer/report, selective hidden capture,
  chunked prefill và document bootstrap.
- [x] Chạy E0 FP16 pilot GovReport/Multi-News/CNN, E0 `max_new_tokens=32`
  confirmation cho GovReport/Multi-News và official DFlash cross-check.
- [x] Triển khai E1 extraction/cache/matched probe với hidden/KV controls,
  document-disjoint split và equal parameter budget.
- [x] Chạy E1 GovReport cap4K và Multi-News cap8K, train probes và sinh report.
- [x] Viết `target_kv_decision_report_2026-09-03.md`.
- [ ] E0 natural long-context FP16 và E2/E3 vẫn chưa đủ điều kiện trên T4;
  chỉ thực hiện trên GPU lớn hơn nếu cần đóng claim 8–40K.
- [x] Full repository: **212 passed, 5 skipped, 23 warnings**; static checks
  pass.
- [x] Preflight selector/drafter compatibility đã bao gồm vocab, hidden width
  positional capacity và tokenizer artifact thực.
- [ ] GPU smoke thật và B200 train/profile/infer chưa thể chạy: host hiện là
  Tesla T4, `.venv` CUDA unavailable, canonical `/workspace` chưa mount.

### Training memory hardening — 2026-09-03

- [x] Mini-batch training cho diffusion/selector/survival/joint và chunked
  selector-stage drafter forward.
- [x] Expose `--train-batch-size`/`SYNCSPEC_TRAIN_BATCH_SIZE`, test wrapper và
  cập nhật docs.
- [x] Full regression `214 passed, 5 skipped, 24 warnings`; static checks pass.
- [ ] Chạy evidence train/profile/infer trên canonical B200.

### Survival semantics correction — 2026-09-03

- [x] Align survival loss/calibration with hazard-to-cumulative-survival
  contract in the design.
- [x] Full regression `215 passed, 5 skipped, 24 warnings` and fresh CPU
  Stage 0 → joint → infer → profile chain.
- [ ] Chạy evidence train/profile/infer thật trên canonical B200.

### Final contract hardening — 2026-09-03

- [x] Guard `top_m` dương trong drafter config và candidate helper.
- [x] Full suite `215 passed, 5 skipped, 24 warnings`; compileall, shell
  syntax và diff check pass.
- [ ] Chạy evidence train/profile/infer thật trên canonical B200.

### Long-context position hardening — 2026-09-03

- [x] Fail-fast khi `position_offset + K_d` vượt drafter positional capacity;
  test scalar/per-row offset.
- [x] Full suite `216 passed, 5 skipped, 24 warnings`.
- [ ] Chạy evidence train/profile/infer thật trên canonical B200.

### Post-hardening smoke evidence — 2026-09-03

- [x] Fresh CPU synthetic inference và measured batch profile.
- [x] Fresh CUDA launcher guard: structured `BLOCKED` trên host không có CUDA.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Context-safe serving and staged-train contracts — 2026-09-03

- [x] Cap generation theo target context headroom cho scalar/batch.
- [x] Fail-fast dependency giữa diffusion → selector → survival.
- [x] Full suite `220 passed, 5 skipped, 24 warnings`.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Context-safe trajectory/profile completion — 2026-09-03

- [x] Cap Stage 0 trajectory và vanilla-AR profile reference theo context.
- [x] Full suite `222 passed, 5 skipped, 24 warnings`; static checks pass.
- [x] Fresh CPU infer/profile; GPU guard ghi nhận môi trường hiện tại thiếu
  CUDA.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Latest verification handoff — 2026-09-03

- [x] Full suite cuối: `222 passed, 5 skipped, 24 warnings`, exit `0`.
- [x] Fresh CPU infer/profile và static checks sau toàn bộ hardening.
- [ ] GPU smoke thật và train/profile/infer trên canonical B200.

### Anchor-only trajectory cache — 2026-09-03

- [x] Lưu target features tại anchor, có metadata mapping và legacy fallback.
- [x] Full suite `223 passed, 5 skipped, 24 warnings`.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Binary trajectory cache — 2026-09-03

- [x] Torch cache `.pt` writer/reader, fingerprint/schema, atomic write và
  resume theo sample ID.
- [x] Stage 0 `.pt` smoke và train CLI integration; full suite `225 passed,
  5 skipped, 24 warnings`.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### B200 wrapper binary-cache default — 2026-09-03

- [x] Train-smoke mặc định dùng trajectory `.pt`, có JSONL override.
- [x] Launcher contract và static checks pass; full suite `225 passed, 5
  skipped, 24 warnings`.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Full-block survival-label rollout — 2026-09-03

- [x] Thu survival labels đủ `K_d` bằng exact verifier, không phụ thuộc head
  random; test full-block coverage.
- [x] Fresh CPU `.pt` → joint + optional refinement pass.
- [x] Full suite `226 passed, 5 skipped, 24 warnings`.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Runtime exactness guard — 2026-09-03

- [x] CLI guard so sánh greedy SyncSpec với vanilla target-AR trên cùng request.
- [x] B200 inference/train-smoke wrapper tự bật guard; stochastic mode được
  loại khỏi token-equality check.
- [x] TDD contract/CLI tests xanh; tiếp tục cần evidence thực trên B200.
- [x] Full repository sau guard: `229 passed, 5 skipped, 24 warnings`; static
  checks pass; CUDA guard trả `BLOCKED/cuda_unavailable` đúng contract.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### B200 master-cache default audit — 2026-09-03

- [x] Đồng bộ master example với wrapper để default trajectory là `.pt`.
- [x] Giữ override `SYNCSPEC_TRAIN_TRAJECTORY` cho JSONL và thêm contract test.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Runtime feedback loop — 2026-09-03

- [x] Cập nhật EMA acceptance/component latency theo request và serialize state.
- [x] Điều chỉnh pre-draft gain theo feedback; zero-acceptance fallback test
  pass.
- [x] Full repository sau patch: `231 passed, 5 skipped, 24 warnings`.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Stochastic exactness distribution test — 2026-09-03

- [x] Empirical rejection/residual test với 12.000 samples khớp target
  distribution trong `±0.025`.
- [x] Full repository: `232 passed, 5 skipped, 24 warnings`.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Runtime profile provenance/schema guard — 2026-09-03

- [x] Preflight và engine yêu cầu `schema_version=1` và
  `source="measured"` cho profile dùng để gate CUDA.
- [x] Thêm regression tests cho profile synthetic/unknown-source và schema lạ.
- [x] Full regression sau hardening: `236 passed, 5 skipped, 24 warnings`.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Post-draft measured AR utility gate — 2026-09-03

- [x] Đo target AR một token và ghi `target_ar_tokens=1` trong runtime profile.
- [x] So sánh utility speculative với AR opportunity cost trong scalar/batch
  controller path; fallback ghi budget `{kd: ..., kv: 0}` và acceptance zero.
- [x] Full regression sau patch và batch regression fix:
  `241 passed, 5 skipped, 25 warnings`.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Stage-0 artifact provenance hardening — 2026-09-03

- [x] Fingerprint bounded cho local model artifact và đưa vào trajectory cache
  key/metadata.
- [x] Stage 0 hỗ trợ `--seed` có provenance trong cache key.
- [x] B200 train wrapper truyền cùng `SYNCSPEC_TRAIN_SEED` vào Stage 0 và train.
- [x] Prefill/verify dùng final-hidden hook để tránh all-layer hidden trên
  Transformers Llama/Qwen long-context.
- [x] Full regression cuối: `243 passed, 5 skipped, 25 warnings`; compileall,
  shell syntax và `git diff --check` pass.
- [ ] Chạy train/profile/infer thật trên canonical B200.

### Target-derived source-memory cache + profiler boundary — 2026-09-03

- [x] Stage 0 cache source-chunk descriptors từ target final hidden, có JSONL/
  `.pt` serialization và metadata offsets.
- [x] Stage 1/2/joint ưu tiên descriptor cache để khớp distribution với
  serving; cache legacy giữ embedding fallback.
- [x] B200 train wrapper bật `--include-source-memory` và expose chunk size.
- [x] Đồng bộ CUDA sau prefill trước khi đo target AR opportunity cost.
- [x] Targeted tests/static checks pass; full regression và B200 real run vẫn
  là bước xác nhận tiếp theo.
- [x] Full regression sau source-memory/profiler patch: `247 passed, 5 skipped,
  25 warnings`; tiny Transformers joint/profile/exactness pass.
- [ ] CUDA smoke thật và train/profile/infer thật trên canonical B200.

### Profile-aware pre-draft budget selection — 2026-09-03

- [x] Giữ trục `K_d` trong empirical gate calibration khi trace có profile.
- [x] Pre-gate chọn profile-specific gain cao nhất vượt safety margin và
  không promote `K_d` chưa được đo.
- [x] Engine truyền prior riêng theo `K_d` cho scalar/batch scheduling.
- [x] Profile batch cost được chuẩn hóa theo batch; `target_ar_tokens` ghi
  đúng số request được đo trong batch.
- [x] Full repository `263 passed, 5 skipped, 25 warnings`; CPU adaptive
  exactness smoke và static checks pass.
- [ ] Chạy CUDA smoke thật và train/profile/infer thật trên canonical B200.

### Final adaptive-profile/batch-cost audit — 2026-09-03

- [x] Strict calibration filter không cho phép profile chưa đo dùng prior mặc
  định để bật speculation; fallback thử profile measured khác nếu có.
- [x] Regression toàn repo: `263 passed, 5 skipped, 25 warnings`.
- [x] CPU wrapper adaptive + vanilla-AR exactness: `status=ok`, không có
  exactness failure.
- [x] `compileall`, `bash -n`, `git diff --check` pass.
- [x] Chạy trực tiếp hai wrapper B200; strict preflight block đúng trước khi
  chạy train/infer khi host thiếu CUDA/asset.
- [x] B200 infer wrapper dùng budget override đã normalize (`KD/KV`), tránh
  truyền đồng thời fixed profile và `--budget-profiles`.
- [x] CPU full-chain runner chạy Stage 0 → joint train → profile → batch infer;
  fresh run tạo đủ drafter/selector/survival và exactness không lỗi.
- [x] Transformers adapter ưu tiên keyword `dtype` hiện hành và có fallback
  legacy; full regression sau hardening đạt `265 passed, 5 skipped, 25
  warnings`.
- [x] B200 wrapper fail-fast khi `MAX_SAMPLES < BATCH_SIZE`, không để profile
  batch-N bị dùng cho inference batch nhỏ hơn; direct/contract test pass.
- [x] Full regression sau batch/sample guard: `266 passed, 5 skipped, 25
  warnings`.
- [x] Preflight strict phân biệt exit `PASS=0`, `BLOCKED=2`, `FAIL=1`; test
  không-CUDA xác nhận contract.
- [ ] External handoff: CUDA smoke và train/profile/infer thật trên B200.

### Real-model batch exactness và GPU integrity audit — 2026-09-03

- [x] Batch inference Qwen3-0.6B offline với drafter/selector/survival thật;
  hai request đều exact với vanilla AR.
- [x] Inference, trajectory và training CLI từ chối `--device cuda` khi CUDA
  không khả dụng; thêm regression tests, không cho phép GPU smoke giả.
- [ ] External handoff: CUDA smoke và train/profile/infer thật trên canonical
  B200.

### Final local verification — 2026-09-03

- [x] Full regression `270 passed, 5 skipped, 25 warnings`.
- [x] Compileall, shell syntax và diff check pass.
- [x] CUDA smoke/preflight trên host dev không giả lập GPU: structured
  `BLOCKED`, exit `2`.
- [ ] External handoff: chạy CUDA smoke và train/profile/infer thật trên
  canonical B200.

### CPU full-chain recheck — 2026-09-03

- [x] Fresh Stage 0 → joint train → multi-profile profile → batch infer pass;
  `records=2`, `exactness_failures=0`.
- [ ] External handoff: chạy cùng pipeline với model/checkpoint thật trên
  canonical B200.

### B200 profile provenance guard — 2026-09-03

- [x] Preflight kiểm tra profile measured đúng model/checkpoint/GPU/precision/
  batch; wrapper truyền đủ runtime axes.
- [x] Contract regression: `22 passed`.
- [ ] External handoff: chạy profile và inference thật trên B200 canonical.

### Final regression sau profile guard — 2026-09-03

- [x] Full regression `271 passed, 5 skipped, 25 warnings`.
- [x] Preflight đầy đủ artifact nhưng profile CPU bị từ chối đúng.
- [ ] External handoff: chạy CUDA smoke và train/profile/infer thật trên
  canonical B200.

### Production profiler component guard — 2026-09-03

- [x] Profiler target thật yêu cầu selector/survival checkpoint; random
  components chỉ còn ở diagnostic flag.
- [x] Targeted CLI/profile regression `26 passed`.
- [ ] External handoff: profile thật và infer thật trên canonical B200.

### Final local verification after profiler guard — 2026-09-03

- [x] Full regression `272 passed, 5 skipped, 25 warnings`.
- [x] Compileall, shell syntax và `git diff --check` pass.
- [ ] External handoff: CUDA smoke và train/profile/infer thật trên canonical
  B200.

### Profile hardening và handoff audit — 2026-09-03

- [x] Profile provenance bind selector/survival checkpoint và reject mismatch.
- [x] Diagnostic profile không thể được dùng làm measured production profile.
- [x] Profile đo đúng một round với `K_v` cố định.
- [x] CUDA smoke chọn đúng interpreter khi dùng `FAST_INFER_VENV`.
- [x] Full regression `277 passed, 5 skipped, 25 warnings`.
- [ ] External handoff: chạy GPU smoke và train/profile/infer thật trên
  canonical B200.

### Fresh local smoke verification — 2026-09-03

- [x] Full regression fresh trên worktree hiện tại: `277 passed, 5 skipped,
  25 warnings`.
- [x] CPU full-chain fresh: Stage 0 → joint train → measured multi-profile →
  batch inference; `status=ok`, `records=2`, `exactness_failures=0`.
- [x] CUDA smoke trên host dev không giả lập GPU: `BLOCKED`, exit `2`,
  `reason=cuda_unavailable`.
- [x] Strict B200 preflight trên host dev: `BLOCKED`, exit `2`; thiếu CUDA và
  canonical target/checkpoint/data/profile đúng như thiết kế fail-fast.
- [ ] External handoff: chạy GPU smoke và train/profile/infer thật trên
  canonical B200.
