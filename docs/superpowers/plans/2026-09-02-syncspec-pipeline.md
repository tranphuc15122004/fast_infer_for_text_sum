# Kế hoạch triển khai SyncSpec-v1 — pipeline train/infer đầy đủ

> Ngày: 2026-09-02  
> Phạm vi: triển khai phần mềm và quy trình reproducible; không thay đổi
> `src/SyncSpec/SyncSpec_v1_design_complete.md`, vì file đó là source of truth
> cho v1.1.

## Mục tiêu và tiêu chí hoàn thành

Xây một subsystem `src/SyncSpec` có thể:

1. tạo target trajectory/cache offline từ target model thật;
2. train drafter theo các stage 1–4 của thiết kế;
3. chạy end-to-end greedy lossless inference với hai budget `K_d/K_v`,
   source n-gram/memory, selector, survival, pre/post gate và exact target
   verification;
4. chạy stochastic rejection-correction đúng phân phối trên adapter hỗ trợ;
5. xuất JSONL cùng summary/timing để so với AR baseline;
6. chạy deterministic synthetic CPU smoke, CUDA smoke và B200 preflight/full
   command bằng model/checkpoint local, không tải internet.

Không claim speedup hay quality trước khi có benchmark thật. Các claim
SyncSpec-specific chỉ được đánh giá bằng artifact và gate G0–G4 trong design.

## Quyết định kiến trúc đã chốt

- Package Python đặt tại `src/SyncSpec` và được thêm vào `PYTHONPATH` bởi
  launcher; tên thư mục giữ nguyên theo design hiện hữu.
- Tách bốn interface: `TargetAdapter`, `DrafterAdapter`, `Selector`,
  `Verifier`. Engine chỉ phụ thuộc interface nên synthetic/CPU, CUDA toy và
  Transformers offline dùng cùng execution path.
- Target luôn full-context exact. Adapter Transformers dùng full prefill một
  lần, cache transaction clone cho block verify, commit correction rồi discard
  suffix chưa commit. Không dùng semantic pruning hoặc target KV eviction.
- Drafter native là shallow DFlash2-style block model: shared-width embedding/
  LM head, bidirectional self-attention trong block, grouped dynamic causal
  convolution, target-anchor/recent-ring/source-memory cross conditioning,
  Top-M candidate lattice. Checkpoint có config JSON + `state_dict`; có thể
  khởi tạo/tie embedding và LM head từ target local.
- V1 selector là sequential Top-M selector, q được normalize trên candidate
  set; survival là hazard rồi cumulative product; controller chọn từ finite
  profiles. `R` cố định trong config v1, không đưa joint-RK oracle cũ vào
  serving mặc định.
- CLI có `--backend synthetic|transformers`; synthetic là contract test, còn
  B200 bắt buộc `--local-files-only` và đường dẫn từ master config.

## Quy trình TDD và validation chung

Mỗi task bên dưới phải theo thứ tự RED → GREEN → refactor:

```bash
python3 -m pytest tests/test_syncspec_<module>.py -q
python3 -m compileall -q src/SyncSpec scripts
python3 -m pytest tests/test_syncspec_*.py -q
git diff --check
```

Các test CUDA phải `pytest.skip` có lý do khi `torch.cuda.is_available()` là
false; không giả vờ biến T4 dev thành B200. Lệnh B200 thật sẽ được kiểm tra
thêm bằng `python3 scripts/check_syncspec_b200.py` trên canonical server.

## Các task triển khai

### 1. Contract, config và output schema

- Tạo `SyncSpecConfig`, `BudgetProfile`, `RuntimeProfile`, seed/offline/device
  validation; default profile đúng v1.1: AR, `(8,4)`, `(8,8)`, `(16,4)`,
  `(16,8)`, `(16,12)`, `(16,16)`.
- Tạo typed request/state/result và JSON serialization; result phải có
  `status`, `input_id`, `generated_text/token_ids`, `accepted_lengths`,
  `kd/kv`, `fallback`, timing breakdown và summary counters.
- Test invalid profile, `K_v <= K_d`, deterministic seed, schema summary và
  offline path.

### 2. Source evidence

- Implement `SourceNgramIndex` cho n=2..6: longest suffix, count, continuity,
  location; chỉ rerank candidate IDs, tuyệt đối không sinh token mới.
- Implement `SourceMemoryBank`: fixed-size source chunks, descriptors từ
  target-derived features/embedding, top-R retrieval, fallback anchor/recent
  memory và explicit retrieval status.
- Test exact lexical features, boundaries, no-candidate mutation, empty-source
  fallback, deterministic top-R and batch/context metadata.

### 3. Drafter model và checkpoint

- Implement native `SyncSpecDrafter` với block `[B,K_d,d]`, masked future slots,
  block bidirectional attention, grouped dynamic causal conv, target feature
  conditioning qua mỗi layer, cross-attention đến anchor/ring/source memory,
  unary logits/hidden và Top-M extraction.
- Implement `DrafterCheckpoint`/load-save, target embedding/LM-head tying,
  pilot dimensions (2–3 layers) và full config không hard-code kích thước.
- Implement adapter protocol để engine nhận native drafter hoặc a compatible
  local Transformers/DFlash candidate provider; không bắt buộc CUDA trong
  module model.
- Test shape, mask/offset, Top-M recall contract, save/load, CPU backward và
  CUDA forward nếu có.

### 4. Selector, survival, gates và controller

- Implement low-rank predecessor coherence, source n-gram feature vector,
  learned source gate và `q=softmax(scores/tau)` chỉ trên Top-M.
- Implement hazard head với `S_j=prod(1-h_j)` monotonic, hard/soft labels,
  finite-profile pre-draft gate và post-draft `argmax` utility từ measured
  profile.
- Test q normalization/zero outside candidates, monotonic survival, gate 0
  khi evidence miss, AR fallback khi utility dưới margin, tie-breaking và
  batch-aware profile selection.

### 5. Exact target verifier và transactional KV

- Implement pure greedy longest-prefix verification and stochastic
  `min(1,p/q)` plus residual `[p-q]_+`; support EOS and empty candidate set.
- Implement transaction state with snapshot, accepted/correction commit and
  rollback; expose exact accepted prefix and correction token.
- Implement `TransformersTargetAdapter` offline: full prefill, block logits
  from cloned cache, commit only accepted/correction tokens, compatible with
  Transformers 5 `return_dict=False`/cache API.
- Test against a tiny known target distribution: greedy output equals vanilla
  AR; stochastic empirical distribution matches target; rejected suffix never
  leaks into committed state; CPU and CUDA toy adapter.

### 6. End-to-end engine and CLI

- Implement `SyncSpecEngine.generate()` round loop:
  prefill → pre-gate/Kd → draft → selector → survival/Kv → exact verify →
  commit/history/profile update; stop at EOS/max tokens.
- Add synthetic deterministic target/drafter for full CPU smoke and an offline
  Transformers backend for real model paths. Use existing `JsonlWriter` and
  repo master config conventions.
- Add `scripts/infer_syncspec.py`, `scripts/run_syncspec.sh`, `syncspec` case
  to `scripts/run.sh`, and docs `docs/baselines/syncspec.md` with canonical
  B200 environment variables and AR comparison command.
- Test one-record CPU E2E, output schema, exact greedy equality, AR fallback,
  smoke/full argument handling and no model download.

### 7. Stage 0 trajectory/cache pipeline

- Implement target-generated trajectory records using exact prompt/tokenizer,
  target output, source boundaries, random anchor positions, target hidden
  features, optional top-logits and ngram metadata.
- Add JSONL/torch cache writer/reader with schema version, model/checkpoint
  fingerprint, tokenizer fingerprint, dtype, context length and seed.
- Add `scripts/build_syncspec_trajectories.py`; support `--backend synthetic`
  for CPU and `--backend transformers --local-files-only` for B200.
- Test round-trip, resume/idempotency, cache fingerprint mismatch and EOS.

### 8. Stage 1–4 drafter training

- Implement corruption/masked-block builder with random anchors and Kd,
  position-weighted CE/KL and optional Top-M margin loss.
- Implement selector training with true serving Top-M, candidate miss masking,
  target restricted KD and teacher-forcing schedule 100→75→50%.
- Implement on-policy rollout survival training with hard/soft labels and
  ECE/Brier metrics; implement optional low-LR joint finetune only behind flag.
- Add `scripts/train_syncspec.py --stage {diffusion,selector,survival,joint}`
  with checkpoint/resume, AMP/bfloat16 on CUDA, gradient accumulation, seed,
  offline validation and JSON training summary.
- Test every loss on tiny synthetic cache, gradients/frozen target weights,
  stage checkpoint resume and a short CPU one-step training run; add CUDA
  smoke test that uses same code when device exists.

### 9. Profiling, B200 preflight and smoke

- Implement timing profiler for draft/selector/survival/verify/scheduler and
  profile keys model/checkpoint/GPU/precision/kernel/context-bin/batch-bin/Kd/Kv.
- Add `scripts/profile_syncspec.py` to produce finite profile JSON consumed by
  controller; never estimate GPU cost from CPU numbers.
- Add `scripts/check_syncspec_b200.py` and `scripts/run_syncspec_b200_smoke.sh`
  using shared runtime/master config. Preflight must check Python 3.12, CUDA,
  B200 capability/name, offline local model files, tokenizer, target/drafter
  compatibility and writable caches. If hardware is absent, emit structured
  `BLOCKED`, not fabricated PASS.
- Test preflight on local CPU with a synthetic temporary master and assert
  structured blocked report; CUDA path is conditional.

### 10. Integration audit and docs

- Run all SyncSpec unit/integration tests, compileall and diff check.
- Run CPU synthetic trajectory → train stage 1–3 → checkpoint → E2E inference;
  run CUDA synthetic smoke when available.
- On B200 run preflight, one-sample target/drafter smoke, then a short real
  trajectory/train/infer smoke using cached models. Record exact commands and
  artifact locations; do not call full benchmark complete until this external
  run is available.
- Update `progress.md`, `findings.md`, `task_plan.md` with fresh evidence and
  explicitly separate locally verified results from B200-pending results.

## Handoff commands

CPU smoke:

```bash
FAST_INFER_PYTHON="$PWD/.venv/bin/python" \
  SYNCSPEC_CPU_SMOKE_DIR=/tmp/syncspec_cpu_smoke \
  bash scripts/run_syncspec_cpu_smoke.sh docs/fast_infer_master.example.env
```

B200 smoke (sau khi canonical mount đã sẵn sàng):

```bash
python3 scripts/check_syncspec_b200.py --strict
bash scripts/run_syncspec_b200_smoke.sh
```
