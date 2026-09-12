# MR-DFlash SpecForge Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use spml:subagent-driven-development (recommended) or spml:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay phase cache HF bằng offline SGLang/SpecForge capture có token-budget batching, async I/O và resume an toàn trên 4 GPU B200 180GB.

**Experiment directory:** `docs/superpowers/`

**Hypothesis:** Offline SGLang capture với batch theo độ dài và `max_total_tokens` sẽ tăng throughput cache trên dữ liệu chủ yếu dưới 8K mà vẫn giữ nguyên feature contract MR-DFlash.

**Validation scope:** L0 static/unit checks; L1 B200 smoke một GPU rồi bốn GPU; parity HF-vs-SGLang trên fixture nhỏ; full 90K chỉ sau khi L0/L1 pass.

**Evaluation design:** Không có model evaluation trong task cache. Cache evaluation dùng coverage, schema/parity, samples/s, tokens/s, peak VRAM, engine warm-up, I/O backlog và khả năng resume.

**Architecture:** Một parent giữ topology data-parallel hiện tại và mỗi GPU chạy một worker SGLang offline capture `TP=1`. Worker dùng scheduler bucket/token-budget, chuyển capture result về `ShardedFeatureWriter` riêng, rồi parent merge sau khi kiểm tra coverage.

**Tech Stack:** Python 3.12, PyTorch, SGLang/FlashInfer có sẵn trên server, `tqdm`, pytest, MR-DFlash sharded feature store.

**Spec:** `docs/superpowers/specs/2026-09-12-mr-dflash-specforge-cache-design.md`

## Global Constraints

- Không gọi network; model/tokenizer/SGLang kernels phải có local cache.
- Giữ `feature_layer_ids=[1,9,17,25,33]`, `mr_dflash_feature_sharded_v1` và dtype provenance.
- Mỗi GPU chỉ có một target engine; không tự dùng GPU ngoài `--parallel-gpu-ids`.
- `max_total_tokens` là giới hạn active token; concurrency pending không được biến thành batch VRAM vô hạn.
- Profile khởi đầu aggressive, sử dụng khoảng 98--99% VRAM GPU B200 180GB; OOM phải tạo retry hint và cho phép hạ profile rồi `--resume`.
- Atomic shard/manifest publish; resume không duplicate hoặc mất sample.
- Không sửa `externals/SpecForge`; chỉ gọi public/local capture boundary qua adapter.

## Shared Scaffold

### Existing infra (don't touch unless required by the tasks)

- Tokenized input: `src/MR_DFlash/tokenized_data.py`
- Feature writer/reader: `src/MR_DFlash/offline_features.py`
- Current HF capture: `src/MR_DFlash/capture.py`
- Cache worker: `scripts/mr_dflash/cache_target_features.py`
- Parallel parent/merge: `scripts/mr_dflash/parallel_stage.py`
- Pipeline CLI: `scripts/mr_dflash/run_preprocess_pipeline.py`
- SpecForge reference: `externals/SpecForge/specforge/offline_capture/`

### Needs setup

- Add lazy SGLang dependency/preflight detection; CPU tests must not import SGLang.
- Add cache backend, token-budget scheduler and profile metadata without changing existing HF fallback behavior unless explicitly selected.

## Subtask 1: Token-budget cache scheduler and profile contract

**Role:** Chọn batch an toàn theo độ dài thực tế dưới 8K và ghi provenance để resume không nhầm profile.

**Files:**
- Create: `src/MR_DFlash/cache_throughput.py`
- Test: `src/MR_DFlash/tests/test_cache_throughput.py`
- Modify: `scripts/mr_dflash/cache_target_features.py`

**Interfaces:**
- `CacheThroughputProfile.from_payload(payload) -> CacheThroughputProfile`
- `CacheThroughputProfile.batch_for_length(length: int) -> int`
- `CacheThroughputProfile.token_budget_for_length(length: int) -> int`
- `CacheThroughputProfile.validate(max_length: int, batch_limit: int) -> None`

- [x] Step 1: Write tests for bucket selection, token budget, malformed profile rejection and deterministic profile hash.
- [x] Step 2: Run the focused test and verify the initial missing-module failure.
- [x] Step 3: Implement the profile parser and deterministic scheduler with initial buckets `4096/8192/16384/32768`, candidate batch `64/32/16/4`, and token budgets `262144/262144/262144/131072`.
- [x] Step 4: Wire cache worker to select batch by actual first sample length and record profile/backend metadata.
- [x] Step 5: Run the focused test and existing cache tests; all pass.

## Subtask 2: SpecForge offline capture adapter

**Role:** Reuse SpecForge’s offline SGLang target path and map DFlash hidden states to MR-DFlash’s feature tensor without loading an HF backbone for every worker.

**Files:**
- Create: `scripts/mr_dflash/specforge_capture.py`
- Modify: `scripts/mr_dflash/cache_target_features.py`
- Test: `src/MR_DFlash/tests/test_specforge_capture_adapter.py`

**Interfaces:**
- `SpecForgeTargetCapture.from_pretrained(...) -> SpecForgeTargetCapture`
- `SpecForgeTargetCapture.capture_batch(input_ids, attention_mask, loss_mask) -> list[torch.Tensor]`
- `SpecForgeTargetCapture.close() -> None`
- `build_capture_backend(name, ...) -> capture object`

- [x] Step 1: Write CPU tests with a fake SpecForge capture object proving output samples are split by valid lengths and preserve input alignment.
- [x] Step 2: Run the focused test and verify it fails at the missing adapter boundary, without importing SGLang during collection.
- [x] Step 3: Implement lazy import of `specforge.offline_capture.load_offline_capture`, call `set_capture_layers(..., capture_method="dflash")`, and map `aux_hidden_states` to per-sample `[length,width]` tensors.
- [x] Step 4: Add explicit backend selection (`hf` or `specforge_sglang`) and fail loudly when SGLang or the target DFlash capture hook is unavailable.
- [x] Step 5: Run focused adapter and offline feature tests.

## Subtask 3: High-throughput asynchronous cache worker

**Role:** Run tokenized data through the selected SGLang backend with bounded GPU tokens, larger asynchronous writer queue and tqdm progress.

**Files:**
- Modify: `scripts/mr_dflash/cache_target_features.py`
- Modify: `src/MR_DFlash/offline_features.py`
- Test: `src/MR_DFlash/tests/test_offline_feature_cache.py`

**Interfaces:**
- `cache_dataset(..., capture_backend="hf", cache_profile=None, max_total_tokens=None, io_threads=8, io_queue_size=32) -> dict`
- Existing writer API remains compatible; new metadata fields are additive.

- [x] Step 1: Add tests for backend dispatch, bounded token batches, async shard publication, OOM hint and profile-changing resume.
- [x] Step 2: Run the focused cache tests and verify the new assertions initially fail at the missing wiring.
- [x] Step 3: Implement bounded producer batching, backend capture, per-sample CPU transfer only after one batched capture, and async writer support.
- [x] Step 4: Allow performance-only profile/token-budget changes on resume, preserve correctness metadata, and append profile history to the manifest.
- [x] Step 5: Add preflight size estimate and structured failure for unavailable backend, invalid profile or insufficient storage; write a lower-budget retry hint on OOM/SIGKILL.
- [x] Step 6: Run all cache/feature-store tests and verify schema and sample ID coverage.

## Subtask 4: Four-GPU orchestration and CLI integration [INTEGRATION]

**Role:** Assemble the new cache backend into the existing pipeline while preventing concurrent model-load `SIGKILL` and preserving parent aggregation/resume.

**Files:**
- Modify: `scripts/mr_dflash/parallel_stage.py`
- Modify: `scripts/mr_dflash/run_preprocess_pipeline.py`
- Test: `src/MR_DFlash/tests/test_worker_progress.py`
- Test: `src/MR_DFlash/tests/test_phase1_smoke_pipeline.py`
- Modify: `docs/mr_dflash_cache_auto_batch.md`
- Modify: `docs/mr_dflash_pilot_pipeline.md`

**Interfaces:**
- CLI flags: `--cache-backend {hf,specforge_sglang}`, `--cache-max-total-tokens`, `--cache-concurrency`, `--cache-startup-stagger-seconds`, `--cache-profile`.
- Status fields: backend, active token budget, completed tokens, aggregate samples/s and aggregate tokens/s.

- [x] Step 1: Write integration tests that generate a four-rank dry-run plan and assert backend/profile/token-budget flags are propagated to every worker.
- [x] Step 2: Run the integration tests and verify the initial missing-flag failure.
- [x] Step 3: Implement CLI/config propagation, startup stagger, aggregate tqdm, independent rendezvous ports, and retry hints; preserve HF fallback.
- [x] Step 4: Use the built-in aggressive B200 profile and allow an explicit reusable throughput profile JSON; retry profiles are emitted on OOM.
- [x] Step 5: Run CPU integration tests, compile/import checks and existing pipeline tests.
- [x] Step 6: L0 static validation: verify device/backend/precision/config propagation, no hidden fallback and complete observability.
- [ ] Step 7: L1 runtime validation on B200: 1 GPU × 128 samples, then 4 GPU × 512 samples; record parity, coverage, throughput, peak VRAM, I/O backlog and warm-up time.
- [ ] Step 8: Only after L0/L1 pass, run the requested 90K train cache with `--resume` and record the final manifest/benchmark summary.
