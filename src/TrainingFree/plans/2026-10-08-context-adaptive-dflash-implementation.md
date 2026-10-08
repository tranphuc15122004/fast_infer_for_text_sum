# Kế hoạch triển khai Context-Adaptive DFlash

> Người triển khai: đọc đặc tả và thực hiện từng task theo workflow `executing-plans`. Checkbox chỉ đánh dấu khi có code và evidence tương ứng. Đây là plan inference training-free; validation dùng CPU deterministic checks và GPU inference, không có training/backward/loss/MFU gate.

**Mục tiêu:** tạo executor, controller và evaluator để kiểm chứng joint draft-context/block-length adaptation trên pretrained DFlash.

**Thư mục implementation:** `src/TrainingFree/context_adaptive/` (cần tạo).

**Giả thuyết:** budget/context selection làm thay đổi gamma tối ưu; policy phối hợp giảm E2E latency so strong fixed và independent controls.

**Đặc tả:** [design](../docs/context_adaptive_dflash/design.md), [integration](../docs/context_adaptive_dflash/integration.md), [protocol](../docs/context_adaptive_dflash/experiment_protocol.md), [schema/timing](../docs/context_adaptive_dflash/measurement_contract.md), [runbook](../docs/context_adaptive_dflash/runbook.md).

**Kiến trúc:** full target KV + full target-derived draft bank; selected draft views và actual block shapes; causal signals/online statistics; evaluator chung cho live generation và offline artifact report.

**Phạm vi validation:** CPU cho custom policy/cache math/schema; GPU parity/correctness sau adapter và G0–G6 theo protocol. Chỉ chạy các checks phục vụ task đã triển khai; không gọi training validation pyramid cho pipeline inference.

## Ràng buộc toàn cục

- Server: Python 3.12 từ PATH, `FI_OFFLINE=1`, local snapshots, dependency hiện có; không installer online.
- Local: CPU-only dev/debug; không sửa CUDA T4, không tạo venv riêng baseline.
- Frozen model weights; target full context; absolute position/RoPE; exact verifier.
- Core không import probes/tests. Không sửa behavior baseline vendored để làm optimized control trông tốt hơn.
- Master ngoài repo qua `config/master.path`; precedence CLI > caller env > master > defaults.
- JSONL qua `io_util.JsonlWriter`, kết thúc summary; reference dùng ROUGE helper chung.
- Artifacts dưới `outputs/context_adaptive_dflash/`, không commit kết quả/model/cache.
- Các file đã thay đổi bởi workstream khác không nằm trong task này.

## Scaffold và file map

Hiện có: `externals/dflash/dflash/model.py`, `scripts/infer_dflash.py`, shared loader/writer/metrics, canonical data và attention probes. `TrainingFree.policy/run/schema` hiện là RECAP-KV; giữ nguyên.

Core files cần tạo được liệt kê đầy đủ ở integration contract. Tests tương lai đặt `src/TrainingFree/tests/context_adaptive/`, tách khỏi core. Launcher theo convention: `scripts/infer_context_adaptive_dflash.py`, `scripts/runners/run_context_adaptive_dflash.sh`, case trong `scripts/run.sh` và cập nhật [baseline guide](../../../docs/baselines/context_adaptive_dflash.md) từ proposed sang implemented sau validation.

## Task 1: Config, prompt layout và source-group split

**Files:** tạo `context_adaptive/{__init__,config,types,prompt}.py`, tests `test_config.py`, `test_prompt.py`, `test_splits.py`.

**Interface:** dataclasses trong integration; `prepare_prompt(record,tokenizer)`; `build_split_manifest(records,exposures,seed)` tạo IDs/hash/group mapping theo protocol, đặt implementation function trong `prompt.py`.

- [ ] Đọc normalized `raw` và renderer hiện có; copy behavior `_format_prompt`, không thêm chat template canonical.
- [ ] Tạo dataclasses đúng field names/tensor contracts; validate mọi enum/value/signature, không thông qua invalid gamma silently.
- [ ] Source span mapping trên final string với tokenizer offsets; ambiguous/missing span xử lý full/B-only hoặc explicit failure.
- [ ] Build split theo source fingerprint + provenance union, không tách QMSum queries cùng conversation; exposure registry đọc IDs/source metadata.
- [ ] CPU fixtures: cùng context hai query phải cùng split; token boundary thuộc source/global protection; repeated source string báo ambiguous; `B<protected` infeasible; gamma vượt checkpoint bị chặn.

**Evidence hoàn tất:** schema config + deterministic split manifest trên fixture; input IDs/hash bằng baseline formatter. Không chạy model để kiểm tra grouping.

## Task 2: Full draft bank và adapter parity

**Files:** tạo `context_adaptive/{cache,attention,generation}.py`, tests `test_bank.py`, `test_attention.py`.

**Consumes:** types/config/layout Task 1. **Produces:** `DraftContextBank.append/gather`, `draft_block`, full-mode `generate_adaptive`.

- [ ] Chiếu target concat features qua `hidden_norm(fc(...))`; context K/V theo mỗi layer, K norm/RoPE original position, bank không chứa noise keys.
- [ ] Preallocate bank capacity prompt+output; append delta ở logical positions liên tục, full gather giữ thứ tự source/processed output.
- [ ] Attention implementation theo reference equation trong integration; giữ current block bidirectional, normalizations, GQA, output projection và MLP weights.
- [ ] CPU fixture positions `{0,100,3000,9000}`; rotate/gather bằng rotate trước rồi index, khác rotate tại `{0,1,2,3}`. Full selection matches dense reference tensor math.
- [ ] GPU G1 parity với vendored trên gamma 3/7/11/15, compare logits/candidates/round outputs trước rollout dài.

**Evidence hoàn tất:** full adapter parity report và bank positions/capacity invariants, checkpoint unchanged. Chưa bật selection/controller để cô lập lỗi adapter.

## Task 3: Exact verifier, parent alignment và boundary

**Files:** hoàn thiện `generation.py`, tạo `signals.py`, tests `test_verifier.py`, `test_entropy.py`, `test_boundaries.py`.

**Consumes:** bank/draft Task 2. **Produces:** `VerificationOutcome`, `GenerationResult`, scalar parent signal có causal origin.

- [ ] Draft argmax; target greedy hoặc sample posterior, prefix match liên tiếp; correction/bonus pending token, crop full KV và append retained target features đúng `[0:L+1]`.
- [ ] Parent logits/attention row **L**, kể cả L=0 hoặc all-accepted; entropy raw logits với signal temperature 1.0.
- [ ] Implement EOS/cap/output counters, AR boundary/emergency và target hidden updates khi trở lại draft.
- [ ] Fixtures: gamma3 proposals `[a,b,c]`, posterior predictions `[a,x,z,w]` → L=1, processed commits2, pending x; full accept → L=3, pending row3; EOS rejected suffix không stop.
- [ ] Toy sampling: q delta tại a, target `{a:0.2,b:0.3,c:0.5}`; verify accept mass0.2 và residual conditional `{b:0.375,c:0.625}`, unconditional output khôi phục target. Thêm two-step conditional-prefix fixture.
- [ ] GPU greedy compare target-only và adapter full tới EOS/cap; lưu first mismatch nếu có. Debug RNG theo absolute token positions nếu cần sampling trajectory parity.

**Evidence hoàn tất:** G1 boundary/greedy/sampler report; không dùng quality metrics để bỏ qua mismatch.

## Task 4: Target-parent selector và sparse execution

**Files:** hoàn thiện `signals.py`, tạo `selection.py`, cập nhật `attention.py`, tests `test_selection.py`, `test_target_signal.py`.

**Consumes:** prompt layout, bank, verifier parent alignment. **Produces:** target-parent score vectors + `Selection`, sparse `generate_adaptive` fixed action.

- [ ] Target capture Q/K không thay output backend; GQA repeat, scale/mask và full-scope FP32 softmax; thu query L sau outcome, prefill dùng last query.
- [ ] Chunk source 128; protection global + first64 source + recent256 output; rank và whole-chunk budget rule trong design.
- [ ] `recent_only` và seeded random controls cùng budget/protection; source score unavailable không tạo zero pseudo-evidence.
- [ ] Gather selected post-RoPE bank + full current block, giữ absolute positions; target forward luôn full cache.
- [ ] CPU fixture source chunks `{0,1}`, `{2,3}`, `{4}` với scores `[0.2,0.9,0.1]`, protected `{0,1}`, B=4 → positions `{0,1,2,3}`; B=1 infeasible. Tie stable theo source position.
- [ ] GPU G2 intervention: cùng prefix dense/sparse; assert target full length, no future-key mass và output parity; đo cả score/gather cost.

**Evidence hoàn tất:** deployable fixed-selection executor và intervention report; selected context đúng schema, không chỉ oracle coverage.

## Task 5: Prefix statistics, cost model và controllers

**Files:** tạo `statistics.py`, `controller.py`, tests `test_statistics.py`, `test_controller.py`.

**Consumes:** state/action/outcome và completed timings. **Produces:** fixed/history/entropy/joint controllers với shared choose/observe contract.

- [ ] Prefix survival update/smoothing/backoff theo design; mọi key chứa selector/B/gamma/runtime/refresh; không share shape outcomes.
- [ ] Entropy quartiles/calibration bins, history EMA và source concentration; n>=8 support rule; reset request state từ prior.
- [ ] Cost query theo context bucket; `frozen_cost` mode rõ ràng, async EMA chỉ dùng event đã ready; không synchronize để đợi action statistics.
- [ ] Feasible actions, min J, tie order, margin2%, full fallback, refresh và output cap override đúng design.
- [ ] Implement independent A+B đúng protocol: B-controller giữ reference gamma; gamma-controller giữ full context; không đọc action của controller kia.
- [ ] Fixture gamma3,L1: survival indicators `[1,0,0]`; L3 không update gamma7; no supported action → full fallback. Cost fixtures `(full,15):T=20,C=5` và `(1024,7):T=9,C=3` → chọn reduced vì 3<4 ms/commit, nếu đủ support/margin.

**Evidence hoàn tất:** deterministic action selection và censor/backoff report; chưa claim adaptive speedup từ toy cost fixtures.

## Task 6: Draft-based refresh và source-signal ablation

**Files:** cập nhật `selection.py`, `signals.py`, `attention.py`, tests `test_refresh.py`.

**Consumes:** sparse executor/controllers. **Produces:** `draft_refresh` selector riêng layer/shared, refresh events và ranking provenance.

- [ ] Bootstrap/periodic dense draft là proposal thật; capture proposal-query mean, block giữ trong denominator, chỉ source keys được rank.
- [ ] Per-layer rankings và shared-layer ablation; recent processed output cập nhật ngay, không reuse noise KV thành output context.
- [ ] Sparse rounds không cập nhật omitted-key global ranking; period4 default, 2/8 dev ablation; record ranking gamma/age.
- [ ] Controller ép full ở refresh, vẫn chọn supported gamma; fallback/wasted work được hạch toán.
- [ ] Fixture round indices0/4/8 refresh full; round1 dùng ranking round0; newly processed outputs xuất hiện trong protection; stale omitted score không bị đổi thành0.

**Evidence hoàn tất:** dense/sparse refresh rollout correctness và cost trace, không thêm unaccounted dense forward.

## Task 7: Evaluator, calibration artifacts và report

**Files:** tạo `schema.py`, `benchmark.py`, `calibration.py`, `report.py`, tests `test_schema.py`, `test_calibration.py`, `test_report.py`.

**Consumes:** executor + state/action grids. **Produces:** artifact contracts và shared evaluation core cho live generation/artifact report.

- [ ] Immutable prefix-state replay cho từng budget/gamma; fork setup ngoài round profiling và ghi setup cost; actual shorter draft shape được forward riêng.
- [ ] Calibration chỉ calibration groups, dev chỉ dev groups; quantiles/prior/cost signatures lưu versioned JSON, reject mismatch/nonfinite/unsupported support.
- [ ] Live request wall timing/sync boundaries, events/components, model/load/warmup scope, full bank/gather bytes và counters theo contract.
- [ ] Request/error/round/summary writer validation, ROUGE reference và aggregate; schema missing/nonfinite fail rõ, không giả zero metrics.
- [ ] Report pair hashes/signatures, đủ coverage, document-cluster bootstrap10k, seeds42 và gate decisions. Sampling RNG trajectory không dùng như greedy exactness gate.
- [ ] Fixtures: QMSum two queries/three repeats chỉ một bootstrap cluster; unmatched/mismatch row invalidates headline; append/resume không duplicate request hoặc summary giữa file; output counters reconcile.

**Evidence hoàn tất:** completed toy live/artifact report từ cùng evaluator; không fabricate measured GPU values trong calibration file.

## Task 8: Launcher và master-config contract

**Files:** tạo `scripts/infer_context_adaptive_dflash.py`, `scripts/runners/run_context_adaptive_dflash.sh`; thêm case dispatcher trong `scripts/run.sh`; cập nhật baseline guide sau code pass.

**Consumes:** config/evaluator. **Produces:** CLI phases/options chính xác như runbook; prepare/report model-free.

- [ ] Load master ngoài repo, snapshot/restore caller CAD overrides và source shared runtime; không tạo config env baseline riêng.
- [ ] Parse CLI rồi merge defaults đúng precedence; `--help`, prepare/report không yêu cầu CUDA/model load.
- [ ] Bootstrap import path như baseline: repo `src`, `scripts`, vendored DFlash; không prepend SGLang source khác vào environment hiện hành.
- [ ] Phase output layout, resume compatibility, preflight status/exit codes, smoke default32 và test locked-config requirements.
- [ ] Fixtures shell/config: CLI thắng caller/master; both SMOKE/FULL fail; `phase=test` thiếu lock fail trước model load; output existing không resume fail trước append.

**Evidence hoàn tất:** CLI contract checks và metadata-only preflight; dispatcher chưa được đưa vào full matrix khi GPU correctness còn pending.

## Task 9: Pipeline inference và thực nghiệm [INTEGRATION]

**Consumes:** toàn bộ Task1–8. **Files:** nối phases trong `benchmark.py`/entrypoint, hoàn thiện `report.py`, docs/runbook trạng thái và tests `test_pipeline.py`.

- [ ] Chạy CPU deterministic/custom-contract suite theo các fixture đã định nghĩa; không broad rerun khi không có thay đổi/failure mới.
- [ ] Chạy server asset/shape preflight rồi GPU smoke6 source groups, request JSONL kết thúc summary, error rows/progress/heartbeat đúng.
- [ ] Thu G0–G2 correctness evidence; có lỗi cache/mismatch thì sửa trước experiments hiệu năng.
- [ ] Thu M1 fixed selection và M2 B-only; report G3/G4 cả kết quả fail, không bỏ control thua.
- [ ] Thu action-grid/counterfactual và real joint/independent dev matrix, xác định G5; chọn locked config chỉ từ calibration/dev.
- [ ] Test heldout toàn cells khóa, 3 repetitions và paired document bootstrap; tạo G6 report với coverage/CI/correctness và natural-output ROUGE.
- [ ] Cập nhật README/baseline guide theo mức đã đạt, chỉ đánh dấu experiment complete khi có artifacts thật; giữ proposal claims nếu gates fail hoặc incomplete.

**Quan sát evaluator:** start/end mỗi phase/cell, progress theo request, heartbeat tối đa60s; lỗi missing/unreadable checkpoint, config/split/prior mismatch, empty data, nonfinite metric và aggregation failure được ghi rõ. Không silence hoặc biến failure thành successful zero metric.

**Evidence hoàn tất:** executor có correctness proof/audit tương ứng, completed G0–G6 report hoặc kết luận gate fail có scope chính xác. Việc hoàn thiện tài liệu hiện tại không đánh dấu task này đã chạy.

## Tự rà soát trước bàn giao implementation

- [ ] Mọi yêu cầu trong design có task/code boundary và fixture tương ứng.
- [ ] Type/field names/variant IDs/config keys khớp giữa design, integration, schema và runbook.
- [ ] Có đúng một task `[INTEGRATION]`; tất cả inference phases dùng shared evaluator.
- [ ] Không runtime API giả, placeholder scientific result, online installer hoặc test leakage.
- [ ] Các files ngoài scope giữ nguyên; commit chỉ khi phiên làm việc yêu cầu, không commit artifacts.

Trạng thái Task1–9: **pending implementation**. Bước bắt đầu là Task1 rồi full-mode parity; không bắt đầu joint heldout khi adapter/signal/correctness chưa có evidence.
