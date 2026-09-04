# SyncSpec DFlash Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use spml:ml-subagent-dev to implement this plan task-by-task.

**Goal:** Align SyncSpec's runtime drafting and drafter training with DFlash block semantics while preserving SyncSpec's selector, survival, controller, and exact verifier contracts.

**Experiment directory:** `src/SyncSpec`

**Hypothesis:** Treating the last committed token (including a verification bonus/correction token) as an explicit anchor slot, and training only the following `kd` masked slots with the corresponding target hidden state, will remove the train–serve mismatch and make each SyncSpec block technically DFlash-compatible.

**Validation scope:** TDD unit/integration tests, CPU synthetic/Transformer tiny-model smoke, compileall and diff checks; CUDA/B200 tests remain conditional and must report blocked when hardware or local artifacts are unavailable.

**Evaluation design:** This change is a contract correction rather than a quality experiment. Validate exact greedy output against vanilla AR, anchor/label alignment, proposal length, checkpoint round-trip, and training loss/backward on tiny data. No claim of speedup is made without a real B200 profile.

**Architecture:** Keep `kd` as the number of proposal tokens exposed to selector/verifier. Internally use a physical block of length `kd + 1`: slot 0 is the last committed token, slots 1..`kd` are masks. The drafter output at slot 0 is discarded; Stage 1 loss and candidate extraction operate on slots 1..`kd`. Stage 0 stores explicit anchor token, target anchor hidden, and serving-equivalent recent hidden for each eligible target state.

---

## Shared Scaffold

### Existing infra (do not rewrite unnecessarily)
- Exact verification and transactional target cache: `src/SyncSpec/verifier.py`, `src/SyncSpec/transformers_adapter.py`.
- Round loop, bonus commit, selector, survival and controller: `src/SyncSpec/engine.py`.
- Native drafter: `src/SyncSpec/model.py`.
- Trajectory/cache types and Stage 1 trainer: `src/SyncSpec/trajectory.py`, `src/SyncSpec/training.py`.
- CLI and smoke runners: `scripts/build_syncspec_trajectories.py`, `scripts/train_syncspec.py`, `scripts/run_syncspec_cpu_smoke.sh`.

### Shared invariants
- `kd` remains the number of proposed/verified tokens in all public engine, profile, selector and output-schema APIs.
- The physical draft input is `[anchor_token] + [MASK] * kd`.
- The anchor slot is never selected, verified, or included in the drafter loss.
- For target suffix `y` and state index `a`, `anchor_token` is the prompt's last token when `a == 0`, otherwise `y[a-1]`; labels are `y[a:a+kd]`.
- `target_anchor` and `recent_hidden` represent the same post-anchor target state used by serving.

## Subtask 1: Stage-0 DFlash-compatible trajectory contract

**Role:** Make cached training records represent the exact target state from which a runtime block starts.

**Implementation:** Extend `TrajectoryRecord`/JSON serialization and `TargetTrajectoryBuilder` to store explicit anchor token IDs, target anchor hidden states, and recent hidden windows for selected anchors. Make anchor count configurable and support enough eligible anchors for DFlash-style multi-anchor training. Preserve backward compatibility only through an explicit legacy fallback/warning.

**Unit Tests:** Stage-0 records for `a == 0` and `a > 0`, anchor token derivation, recent-hidden position mapping, JSON round-trip, configurable anchor count, and EOS/short-suffix behavior.

**Expected Conclusion:** A cache row unambiguously identifies the committed anchor token, its target hidden state, its recent hidden context, and the next target labels.

### Step 1: Write unit tests for the new cache contract

Add tests to `tests/test_syncspec_pipeline.py` that assert the exact anchor token and hidden/recent positions for synthetic targets.

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/test_syncspec_pipeline.py -k 'anchor_token or recent_hidden or configurable_anchor' -q`

Expected: FAIL because the fields and builder options do not yet exist.

### Step 3: Implement the trajectory/cache changes

Modify only core trajectory/training record code and the trajectory CLI needed to expose the new options.

### Step 4: Run tests to verify they pass

Run: `python3 -m pytest tests/test_syncspec_pipeline.py -q`

### Step 5: Commit

```bash
git add src/SyncSpec/trajectory.py src/SyncSpec/training.py scripts/build_syncspec_trajectories.py tests/test_syncspec_pipeline.py
git commit -m "syncspec: store DFlash-compatible anchor states"
```

## Subtask 2: Runtime block construction and proposal slicing

**Role:** Make inference consume the verification-produced last committed/bonus token as the explicit DFlash anchor.

**Implementation:** Update `build_masked_block` call sites in `NativeDrafterAdapter` and `draft_batch` to build `kd + 1` slots with `sample_from_anchor=True`. Slice output logits/hidden from slot 1 onward before creating Top-M candidates. Keep `engine.py`'s bonus/correction commit behavior unchanged, but add assertions/tests that the next draft sees the committed token and updated target hidden.

**Unit Tests:** Adapter input shape/content, candidate/hidden shape remains `[kd, ...]`, scalar and batched drafting, bonus-token anchor propagation, and exact greedy output preservation.

**Expected Conclusion:** Inference has DFlash block semantics without changing the external `kd` contract.

### Step 1: Write failing runtime contract tests

Extend `tests/test_syncspec_native_e2e.py` with an instrumented drafter/model assertion for `[anchor] + [MASK] * kd` and slot-1 slicing.

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/test_syncspec_native_e2e.py -q`

Expected: FAIL because the adapter currently sends only masked slots and uses slot 0 as a proposal.

### Step 3: Implement adapter changes

Update `src/SyncSpec/transformers_adapter.py` and only the helper documentation/validation in `src/SyncSpec/model.py` if required.

### Step 4: Run tests to verify they pass

Run: `python3 -m pytest tests/test_syncspec_native_e2e.py -q`

### Step 5: Commit

```bash
git add src/SyncSpec/transformers_adapter.py src/SyncSpec/model.py tests/test_syncspec_native_e2e.py
git commit -m "syncspec: use committed token as draft anchor"
```

## Subtask 3: DFlash-compatible Stage-1 training

**Role:** Train the drafter on the same physical block and conditioning inputs used by inference.

**Implementation:** Change `build_stage1_batch` to return `[B, kd+1]` input/target/mask tensors with the anchor in slot 0 and `valid[:, 0] == False`. Pass cached `target_anchor` and `recent_hidden` into the model. Compute loss on slots 1..`kd`, add DFlash-style exponential position decay with explicit `gamma` (default 7.0 for the DFlash baseline), and retain optional KL/Top-M terms aligned to sliced future logits. Sample many stored anchors per step rather than one fixed anchor per record.

**Unit Tests:** Exact block/label alignment, anchor exclusion from loss, recent-hidden propagation, position-decay weights, multi-anchor batch sampling, optional teacher-logit shape, frozen target embedding/LM-head and CPU backward/checkpoint resume.

**Expected Conclusion:** Stage 1 optimizes exactly the future slots that serving exposes to selector/verifier.

### Step 1: Write failing Stage-1 tests

Update `tests/test_syncspec_pipeline.py` and `tests/test_syncspec_training.py` for block length `kd+1`, slot-0 exclusion, recent-hidden input, and DFlash decay.

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/test_syncspec_pipeline.py tests/test_syncspec_training.py -q`

Expected: FAIL on current tensor shapes and missing recent-hidden/decay behavior.

### Step 3: Implement Stage-1 changes

Modify `src/SyncSpec/training.py`, `scripts/train_syncspec.py`, and the necessary cache-loading paths. Keep the public summary explicit about `kd`, `physical_block_size`, `num_anchors`, and `loss_decay_gamma`.

### Step 4: Run tests to verify they pass

Run: `python3 -m pytest tests/test_syncspec_pipeline.py tests/test_syncspec_training.py -q`

### Step 5: Commit

```bash
git add src/SyncSpec/training.py scripts/train_syncspec.py tests/test_syncspec_pipeline.py tests/test_syncspec_training.py
git commit -m "syncspec: train future slots from DFlash anchors"
```

## Subtask 4: Selector/survival and CLI contract adaptation

**Role:** Ensure later stages consume only proposal slots and do not accidentally learn or score the anchor slot.

**Implementation:** Update selector-stage lattice construction and joint training to use sliced future hidden/logits/targets. Keep survival positions indexed `0..kd-1` over proposals. Update CLI help, trajectory CLI options, cache compatibility messages, checkpoint metadata, and SyncSpec docs to define physical block length separately from public `kd`.

**Unit Tests:** Selector lattice shapes, target alignment, survival feature length, joint training smoke, checkpoint metadata, legacy-cache rejection/warning, and CLI argument contract.

**Expected Conclusion:** Selector, survival, controller, and output schema remain behaviorally unchanged while receiving the correct `kd` proposal rows.

### Step 1: Write failing stage-2/3 contract tests

Add assertions to `tests/test_syncspec_training.py`, `tests/test_syncspec_selector.py`, and `tests/test_syncspec_cli.py` for proposal-only tensors and metadata.

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/test_syncspec_training.py tests/test_syncspec_selector.py tests/test_syncspec_cli.py -q`

Expected: FAIL where existing code assumes the Stage-1 tensor length equals `kd`.

### Step 3: Implement stage-2/3 and CLI changes

Update stage orchestration without introducing EAGLE autoregressive TTT into SyncSpec.

### Step 4: Run tests to verify they pass

Run: `python3 -m pytest tests/test_syncspec_training.py tests/test_syncspec_selector.py tests/test_syncspec_cli.py -q`

### Step 5: Commit

```bash
git add src/SyncSpec scripts/train_syncspec.py tests/test_syncspec_*.py docs/baselines/syncspec.md
git commit -m "syncspec: preserve proposal-only downstream stages"
```

## Subtask 5: SpecForge-style random eligible-anchor sampling

**Role:** Match SpecForge's DFlash training regime instead of cycling a pre-flattened anchor list.
**Implementation:** Build eligible anchors from cached target states and physical position capacity. At every diffusion/joint optimizer forward, sample without replacement up to `num_anchors` from each selected trajectory using the trainer's seeded CPU generator. Randomize selector lattice batches with the same generator while keeping candidate lattices serving-valid. Expose `--num-anchors` and record sampled/eligible counts in summaries.
**Unit Tests:** Verify per-forward samples are different but reproducible, always eligible, respect physical capacity, and preserve `kd+1`/anchor-slot masking.
**Expected Conclusion:** Training sees fresh DFlash-compatible anchor blocks on every forward, with deterministic replay under the same seed.

### Step 1: Write failing random-sampling tests

Add tests for reproducible per-forward random anchor selection and trainer instrumentation.

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/test_syncspec_training.py -k 'random_anchor or sampled_anchor' -q`

Expected: FAIL because training currently cycles the flattened anchor rows.

### Step 3: Implement random eligible-anchor sampling

Update `src/SyncSpec/training.py` and `scripts/train_syncspec.py`; keep legacy cache fallbacks explicit and never sample an ineligible/over-capacity state.

### Step 4: Run tests to verify they pass

Run: `python3 -m pytest tests/test_syncspec_training.py -k 'random_anchor or sampled_anchor' -q`

### Step 5: Commit

```bash
git add src/SyncSpec/training.py scripts/train_syncspec.py tests/test_syncspec_training.py
git commit -m "syncspec: sample eligible anchors per training forward"
```

## Subtask 6: FlashAttention-first SDPA dispatch

**Role:** Prefer the FlashAttention SDPA backend during drafter training while retaining an offline-safe fallback.
**Implementation:** Preserve the existing `MultiheadAttention` checkpoint layout, add an `attention_backend` config (default `flash`), dispatch self/cross attention through `torch.nn.attention.sdpa_kernel` with FlashAttention first, and retry using efficient/math SDPA when FlashAttention is unavailable. Record the selected backend and add a CLI override for diagnostics.
**Unit Tests:** Validate flash-first dispatch selection, SDPA fallback on unsupported devices, checkpoint round-trip, CPU forward/backward, and no change to output shapes.
**Expected Conclusion:** B200 can select FlashAttention without a checkpoint migration; CPU/T4 development falls back to SDPA cleanly.

### Step 1: Write failing attention-dispatch tests

Add tests that observe backend selection and force a FlashAttention failure to exercise fallback.

### Step 2: Run tests to verify they fail

Run: `python3 -m pytest tests/test_syncspec_model.py -k 'attention_backend or flash' -q`

Expected: FAIL because attention backend is not configurable or flash-first.

### Step 3: Implement backend dispatch

Update `src/SyncSpec/model.py`, `scripts/train_syncspec.py`, and checkpoint/config compatibility paths.

### Step 4: Run tests to verify they pass

Run: `python3 -m pytest tests/test_syncspec_model.py -k 'attention_backend or flash' -q`

### Step 5: Commit

```bash
git add src/SyncSpec/model.py scripts/train_syncspec.py tests/test_syncspec_model.py
git commit -m "syncspec: prefer flash attention with SDPA fallback"
```

## Subtask 7: Final SyncSpec training/inference pipeline [INTEGRATION]

**Hypothesis:** The assembled pipeline will use one consistent DFlash-style state contract from target verification through drafting, training, selector, survival, and exact commit.

**Components consumed:** Stage-0 cache contract, runtime adapter, SpecForge-style random anchor sampler, flash-first attention dispatch, Stage-1 trainer, selector/survival orchestration, and existing verifier/engine.

**Implementation:** Run the full trajectory → Stage 1 → Stage 2 → Stage 3 → checkpoint → inference path. Add per-run metadata for `physical_block_size`, anchor source, recent-hidden availability, and cache contract version. Ensure training summary and inference artifacts clearly separate proposal `kd` from physical block size.

**Integration Tests:** Tiny Transformer target: build trajectory with features/recent hidden, train one Stage-1 step, train selector/survival, load checkpoint, run exact greedy inference, and assert output equals vanilla AR while adapter instrumentation confirms the anchor slot.

**Validation Pyramid:** L0 static review of device/precision/optimizer/loss outputs/attention backend/logging; L1 CPU real-data or tiny-model runtime validation. CUDA/B200 validation is conditional on canonical hardware and local artifacts; no fabricated PASS.

**Evaluation contract:** Validate in-memory and checkpoint-loaded inference through the shared engine, with explicit phase logs and structured summaries. Fail on missing cache contract, shape mismatch, non-finite loss, or target/drafter vocabulary mismatch.

**Expected Conclusion:** All local tests and CPU integration smoke pass; B200 result is reported separately as PASS or BLOCKED.

### Step 1: Write integration assertions

Extend the tiny Transformer CLI/e2e test to require the new trajectory fields, Stage-1 physical block metadata, and exact output equality.

### Step 2: Run integration tests to verify they fail

Run: `python3 -m pytest tests/test_syncspec_native_e2e.py tests/test_syncspec_cli.py -q`

Expected: FAIL until every stage agrees on the new block contract.

### Step 3: Assemble and validate the pipeline

Run the complete CPU path, then compile all changed Python files and inspect the diff.

### Step 4: Run integration tests to verify they pass

Run:

```bash
python3 -m pytest tests/test_syncspec_*.py -q
python3 -m compileall -q src/SyncSpec scripts
git diff --check
```

### Step 5: Run the CPU smoke path

Run the repository's SyncSpec CPU smoke command with a temporary output directory and verify structured artifacts.

### Step 6: Run conditional B200 validation

Run `python3 scripts/check_syncspec_b200.py --strict` and the B200 training/inference smoke only when the canonical server exposes CUDA, target model artifacts, writable cache, and compatible local dependencies.

### Step 7: Record conclusion

Record local test counts, CPU smoke status, and any B200 `BLOCKED` reason in the final response; do not modify unrelated user files such as `task_plan.md`.

### Step 8: Commit

```bash
git add src/SyncSpec scripts tests docs/baselines/syncspec.md docs/superpowers/plans/2026-09-04-syncspec-dflash-alignment.md
git commit -m "syncspec: align training and inference with DFlash blocks"
```
