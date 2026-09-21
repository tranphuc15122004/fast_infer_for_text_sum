# MR-DFlash vLLM/SGLang Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thêm vLLM 0.24.0 regenerate worker và hoàn thiện cấu hình multi-GPU trong khi giữ SGLang 0.5.14 adaptive cache.

**Architecture:** Mỗi worker regenerate gọi một vLLM OpenAI-compatible server bằng prompt token IDs; vLLM continuous-batch các request. Cache tiếp tục dùng SpecForge/SGLang offline capture trên một worker/GPU với token-budget adaptive batch và async writer hiện có.

**Tech Stack:** Python 3.12, urllib/OpenAI-compatible HTTP, vLLM 0.24.0, SGLang 0.5.14, SpecForge, PyTorch, pytest.

**Spec:** `docs/superpowers/specs/2026-09-20-mr-dflash-vllm-sglang-acceleration-design.md`

## Global Constraints

- Không tải package/model qua internet; vLLM/SGLang/model phải có sẵn trên server.
- Không sửa `externals/SpecForge`.
- Regenerate mặc định HF vẫn giữ tương thích; vLLM là backend opt-in.
- Cache SGLang chỉ bật khi `--cache-backend specforge_sglang`.
- Một vLLM server hoặc SGLang worker không được dùng chung một GPU với backend khác.
- Giữ output schema, feature schema, layer IDs, resume và coverage contract.
- Mọi production code mới phải có test trước khi implement.

### Task 1: vLLM request client and token-aware concurrency

**Files:**
- Create: `scripts/mr_dflash/vllm_regenerate.py`
- Create: `src/MR_DFlash/tests/test_vllm_regenerate.py`

**Interfaces:**
- `VLLMCompletionClient(base_url, model, timeout_seconds)`
- `VLLMCompletionClient.complete(prompt_ids, max_tokens, temperature, seed) -> str`
- `TokenAdmissionController(start_size, max_size, max_batched_tokens)`
- `run_vllm_regeneration(args) -> None`

- [x] Write tests for exact `/v1/completions` payload, token admission, success growth, server-error backoff and response parsing using a local fake HTTP handler.
- [x] Run the focused test first and confirm the expected missing-module failure.
- [x] Implement the minimal HTTP client with token-list prompt, `max_tokens`, `temperature`, `seed`, `stream=false`, timeout and structured server errors.
- [x] Implement bounded concurrent requests; preserve input item metadata and use existing regeneration helpers for tokenizer/chat-template/budget fitting.
- [x] Implement resume, skipped report, durable output and manifest fields compatible with `mr_dflash_regeneration_v1`.
- [x] Run the focused tests and verify all pass.

### Task 2: Parallel/pipeline vLLM backend propagation

**Files:**
- Modify: `scripts/mr_dflash/parallel_stage.py`
- Modify: `scripts/mr_dflash/run_preprocess_pipeline.py`
- Modify: `src/MR_DFlash/tests/test_cache_auto_batch.py`
- Modify: `src/MR_DFlash/tests/test_pilot_scripts.py`

**Interfaces:**
- CLI `--regenerate-backend {hf,vllm}`
- CLI `--vllm-server-addresses host:port [...]`
- CLI `--vllm-model MODEL_ID`
- CLI `--vllm-request-concurrency N`
- Worker command maps rank `i` to address `addresses[i]`.

- [x] Add plan tests proving vLLM flags are present only for vLLM regenerate and cache flags remain unchanged.
- [x] Run the focused plan tests and verify failure before production edits.
- [x] Add immutable pipeline options and parser validation: vLLM requires at least one address and multi-GPU mode requires enough addresses for workers.
- [x] Update `parallel_stage.py` to launch `vllm_regenerate.py` with the rank-specific address while retaining HF behavior by default.
- [x] Record backend, server addresses, request concurrency and vLLM model in plan/manifest metadata.
- [x] Run pipeline/parallel tests and compile checks.

### Task 3: Lease granularity and server runbook

**Files:**
- Modify: `scripts/mr_dflash/parallel_stage.py`
- Modify: `docs/mr_dflash_pilot_pipeline.md`
- Modify: `docs/mr_dflash_cache_auto_batch.md`
- Modify: `docs/server_environment.md`
- Test: `src/MR_DFlash/tests/test_shared_scheduler.py`

- [x] Add a failing test for bounded shared-lease quantum so one huge lease cannot monopolize a worker round.
- [x] Run the scheduler test and verify failure.
- [x] Add `--queue-quantum-items` with a safe default and pass it to shared-lease claims; keep resume/reclaim semantics unchanged.
- [x] Document vLLM 0.24.0 regenerate launch assumptions, SGLang 0.5.14/SpecForge cache requirements, 180 GiB memory targets and no GPU sharing between phases.
- [x] Run scheduler and documentation-linked CLI tests.

### Task 4: Integrated validation

**Files:**
- Modify: `docs/superpowers/specs/2026-09-20-mr-dflash-vllm-sglang-acceleration-design.md` only if validation findings change the contract.
- Test: `src/MR_DFlash/tests/test_pilot_scripts.py`
- Test: `src/MR_DFlash/tests/test_cache_auto_batch.py`

- [x] Run the complete focused suite for regeneration, cache adaptive batch, shared scheduler and pipeline plan (`101 passed`).
- [x] Run `py_compile` for all changed scripts.
- [ ] Run server preflight when the canonical GPU server is available; verify vLLM 0.24.0, SGLang 0.5.14 and SpecForge import/patch compatibility.
- [ ] Run a 20–100 sample GPU smoke with vLLM regenerate and SGLang cache; compare output/hidden-state parity and record throughput/VRAM.
- [x] Document the full-run command with an explicit caveat that no speedup is claimed before GPU parity/smoke.

GPU validation remains intentionally pending: this workspace is the CPU-only
development host described in `AGENTS.md`, not the CUDA 13 server.
