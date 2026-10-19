# DFlash Phase A Benchmark Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai thuật toán Context Focusing Phase A (Winning Production Recipe: Target Layer 18, Multi-Query Max-Pooling, Chunk 128, Sink 16 + Recent 32, Budget B=2048) cho cả hai backend vLLM và FA4 Native trong `src/Benchmark`, chạy đồng bộ với các baseline hiện có trên server B200.

**Architecture:** Tạo module selector thuần túy `src/Benchmark/common/phase_a_selector.py` tính toán Multi-Query Max-Pooling và phân bổ chunk 128 tokens; tích hợp vào pipeline FA4 native (`native_fa4_benchmark.py`) bằng cách thu gọn KV Cache drafter trước khi gọi FA4 dispatch; tích hợp vào pipeline vLLM (`vllm_all_baselines.py` & `vllm_pilot.py`) bằng cách lọc block tables của PagedAttention trong `DFlashSpeculator`. Cả hai backend xuất metrics theo schema chuẩn (TPOT, DSR, ESR, $E[L]$, Token Parity).

**Tech Stack:** Python 3.12, PyTorch 2.x, vLLM 0.30 Model Runner V2 (PagedAttention), FlashAttention-4 (CUTLASS b15), pytest.

**Spec:** [`docs/superpowers/specs/2026-10-19-dflash-phase-a-benchmark-integration-design.md`](file:///home/tuantb/fast_infer_text_sum/docs/superpowers/specs/2026-10-19-dflash-phase-a-benchmark-integration-design.md)

## Global Constraints (Zero-Overhead Engineering Rules)
- **Zero GPU-CPU Synchronization**: Tuyệt đối không gọi `.cpu()`, `.tolist()` trong quá trình tính toán selector hoặc filtering ở mỗi decoding round.
- **Không dùng `output_attentions=True`**: Chỉ đặt forward hook cục bộ trên `target.model.layers[18].self_attn.q_proj` để thu giữ $Q_{18}$ ($K \le 16$). Tận dụng $K_{18}$ có sẵn trong KV cache để làm 1 phép GEMM duy nhất ($<0.05\,\text{ms}$).
- **Vectorized Chunking**: Tính điểm Chunk 128 bằng `.view(-1, 128).sum(-1)` và `torch.topk` hoàn toàn trên GPU (thời gian $<10\,\mu\text{s}$).
- **Zero Memory Movement trong vLLM**: Lọc `block_table` tensor của PagedAttention trên GPU thay vì sao chép hay tái cấp phát KV cache.
- Target Layer mặc định: 18 ($L_{18}$)
- Toán tử mặc định: Multi-Query Max-Pooling ($M_1$) trên toàn bộ accepted prefix
- Kích thước Chunk: 128 tokens (Spatial Continuity $>98\%$)
- Neo bảo vệ: 16 Sink tokens đầu prompt + 32 Recent tokens cuối
- Ngân sách mặc định: $B = 2048$ tokens (hỗ trợ override qua `DFLASH_PHASE_A_BUDGET`)
- Không làm thay đổi hành vi mặc định của `dflash` (Dense) hay bất kỳ baseline nào khác (`vanilla`, `eagle3`, `domino`, `dspark`)
- Kiểm thử được 100% logic trên CPU local qua `.venv/bin/pytest` mà không cần CUDA GPU

---

### Task 1: Core Context Focusing Selector Module (`phase_a_selector.py`)

**Files:**
- Create: `src/Benchmark/common/phase_a_selector.py`
- Test: `tests/test_phase_a_selector.py`

**Interfaces:**
- Produces:
  - `PhaseAConfig(target_layer=18, selector="m1_multi_query", chunk_size=128, sink_tokens=16, recent_tokens=32, budget=2048, query_pooling="max")`
  - `compute_multi_query_scores(attention_weights_or_matrix: torch.Tensor, accepted_len: int, query_pooling: str = "max") -> torch.Tensor`
  - `select_focused_context_indices(scores: torch.Tensor, total_length: int, config: PhaseAConfig) -> torch.Tensor`
  - `select_focused_paged_blocks(scores: torch.Tensor, total_length: int, block_size: int, config: PhaseAConfig) -> list[int]`

- [ ] **Step 1: Write the failing test for PhaseAConfig and selector math**

```python
# tests/test_phase_a_selector.py
import pytest
import torch
from Benchmark.common.phase_a_selector import (
    PhaseAConfig,
    compute_multi_query_scores,
    select_focused_context_indices,
    select_focused_paged_blocks,
)

def test_phase_a_config_defaults_and_env(monkeypatch):
    cfg = PhaseAConfig()
    assert cfg.target_layer == 18
    assert cfg.chunk_size == 128
    assert cfg.sink_tokens == 16
    assert cfg.recent_tokens == 32
    assert cfg.budget == 2048
    assert cfg.query_pooling == "max"

    monkeypatch.setenv("DFLASH_PHASE_A_BUDGET", "1024")
    monkeypatch.setenv("DFLASH_PHASE_A_LAYER", "20")
    monkeypatch.setenv("DFLASH_PHASE_A_CHUNK_SIZE", "64")
    cfg_env = PhaseAConfig.from_env()
    assert cfg_env.budget == 1024
    assert cfg_env.target_layer == 20
    assert cfg_env.chunk_size == 64

def test_compute_multi_query_scores_max_pooling():
    # 3 accepted queries, 10 context tokens
    # attn shape: [heads, queries, context] or [queries, context]
    attn = torch.tensor([
        [0.1, 0.5, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
        [0.0, 0.2, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
        [0.4, 0.1, 0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.2],
    ])
    scores = compute_multi_query_scores(attn, accepted_len=3, query_pooling="max")
    expected = torch.tensor([0.4, 0.5, 0.8, 0.1, 0.9, 0.1, 0.0, 0.0, 0.3, 0.2])
    assert torch.allclose(scores, expected)

def test_select_focused_context_indices_anchors_and_chunks():
    # Context length 1024 tokens, chunk_size=128 (8 chunks)
    # Sink=16, recent=32, budget=256
    cfg = PhaseAConfig(chunk_size=128, sink_tokens=16, recent_tokens=32, budget=256)
    scores = torch.zeros(1024)
    # Give high score to chunk 2 (tokens 256..383)
    scores[256:384] = 1.0

    indices = select_focused_context_indices(scores, total_length=1024, config=cfg)
    assert indices.ndim == 1
    assert len(indices) <= 256
    # Must preserve sink tokens 0..15
    for i in range(16):
        assert i in indices
    # Must preserve recent tokens 1024-32..1023
    for i in range(1024 - 32, 1024):
        assert i in indices
    # Must contain high-scoring chunk 2
    assert 256 in indices
    assert 383 in indices
    # Must be sorted and unique
    assert torch.equal(indices, torch.unique(indices))
    assert torch.equal(indices, torch.sort(indices).values)

def test_select_focused_paged_blocks():
    # Context 1024 tokens, block_size=16 -> 64 blocks
    # chunk_size=128 -> 8 blocks per chunk
    cfg = PhaseAConfig(chunk_size=128, sink_tokens=16, recent_tokens=32, budget=256)
    scores = torch.zeros(1024)
    scores[256:384] = 10.0 # Chunk 2
    block_indices = select_focused_paged_blocks(scores, total_length=1024, block_size=16, config=cfg)
    # Block 0 (sink) must be in block_indices
    assert 0 in block_indices
    # Blocks for recent (last 32 tokens = 2 blocks = 62, 63)
    assert 62 in block_indices
    assert 63 in block_indices
    # Chunk 2 (tokens 256..383) = blocks 16, 17, ..., 23
    for b in range(16, 24):
        assert b in block_indices
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_phase_a_selector.py -v`  
Expected: FAIL with ModuleNotFoundError or import error.

- [ ] **Step 3: Implement `src/Benchmark/common/phase_a_selector.py`**

Implement `PhaseAConfig`, `compute_multi_query_scores`, `select_focused_context_indices`, `select_focused_paged_blocks` using efficient PyTorch operations with CPU and CUDA device support.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_phase_a_selector.py -v`  
Expected: PASS with 4 tests passed.

- [ ] **Step 5: Commit**

```bash
git add src/Benchmark/common/phase_a_selector.py tests/test_phase_a_selector.py
git commit -m "feat(benchmark): add Phase A context focusing selector module and unit tests"
```

---

### Task 2: Native FA4 Backend Integration (`dflash_phase_a`)

**Files:**
- Modify: `src/Benchmark/common/flashattn_runtime.py:12-16`
- Modify: `src/Benchmark/dflash_compat.py`
- Modify: `src/Benchmark/native_fa4_benchmark.py:90-105, 630-660, 930-990`
- Modify: `externals/dflash/dflash/model.py:64-150` (support context selector hook in `dflash_generate`)
- Test: `tests/test_benchmark_fa4_phase_a.py`

**Interfaces:**
- Consumes: `PhaseAConfig`, `select_focused_context_indices`, `compute_multi_query_scores` from `Benchmark.common.phase_a_selector`
- Produces:
  - `SUPPORTED_METHODS` in `flashattn_runtime.py` includes `"dflash_phase_a"`
  - `MODEL_REPOS["dflash_phase_a"] = os.environ.get("MODEL_DFLASH_DRAFT", "")`
  - `native_fa4_benchmark.py` executes `dflash_phase_a` with focused context via FA4 attention
  - Record Phase A metrics (`phase_a_budget`, `phase_a_layer`, `phase_a_selected_tokens`) in output records

- [ ] **Step 1: Write failing contract test for FA4 runner with `dflash_phase_a`**

```python
# tests/test_benchmark_fa4_phase_a.py
import pytest
from Benchmark.common.flashattn_runtime import SUPPORTED_METHODS
from Benchmark.native_fa4_benchmark import MODEL_REPOS

def test_fa4_supported_methods_includes_dflash_phase_a():
    assert "dflash_phase_a" in SUPPORTED_METHODS
    assert "dflash_phase_a" in MODEL_REPOS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_benchmark_fa4_phase_a.py -v`  
Expected: FAIL (`AssertionError: assert 'dflash_phase_a' in SUPPORTED_METHODS`).

- [ ] **Step 3: Update `flashattn_runtime.py`, `dflash_compat.py`, `dflash_generate`, and `native_fa4_benchmark.py`**

- In `flashattn_runtime.py`: Add `"dflash_phase_a"` to `SUPPORTED_METHODS`.
- In `externals/dflash/dflash/model.py` (`dflash_generate`):
  Add parameter `phase_a_selector: Optional[Any] = None`.
  When `phase_a_selector` is provided, at round $t$:
  - Extract layer 18 attention weights from target verifier.
  - Compute Multi-Query scores on accepted tokens.
  - Select focused indices of size $B=2048$.
  - Gather/slice `past_key_values_draft` (or drafter's context K/V) to focused indices before calling draft model forward with FA4.
- In `native_fa4_benchmark.py`:
  - Register `dflash_phase_a` in `MODEL_REPOS`.
  - In `_load_models_for_method`: if `method == "dflash_phase_a"`, instantiate `PhaseAContextSelector`.
  - In `_generate`: pass `phase_a_selector` to `dflash_generate`.
  - In record summary: record Phase A config and metrics.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_benchmark_fa4_phase_a.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Benchmark/common/flashattn_runtime.py src/Benchmark/dflash_compat.py src/Benchmark/native_fa4_benchmark.py externals/dflash/dflash/model.py tests/test_benchmark_fa4_phase_a.py
git commit -m "feat(fa4): integrate DFlash Phase A context focusing into native FA4 benchmark"
```

---

### Task 3: vLLM Backend Integration (`vllm_all_baselines.py` & `vllm_pilot.py`)

**Files:**
- Modify: `src/Benchmark/common/vllm_pilot.py:18-40, 200-480`
- Modify: `src/Benchmark/vllm_all_baselines.py:43-50, 95-140`
- Modify: `scripts/run_vllm_all.sh:65-75`
- Test: `tests/test_vllm_phase_a.py`

**Interfaces:**
- Consumes: `PhaseAConfig`, `select_focused_paged_blocks` from `Benchmark.common.phase_a_selector`
- Produces:
  - `METHODS` in `vllm_all_baselines.py` includes `"dflash_phase_a"`
  - `DRAFT_ARGUMENTS["dflash_phase_a"] = "dflash_model"`
  - `install_phase_a_vllm_compat()` in `vllm_pilot.py`
  - vLLM PagedAttention block pruning for draft forward pass when method is `dflash_phase_a`

- [ ] **Step 1: Write failing contract test for vLLM runner with `dflash_phase_a`**

```python
# tests/test_vllm_phase_a.py
import pytest
from Benchmark.vllm_all_baselines import METHODS, DRAFT_ARGUMENTS, build_speculative_config
from Benchmark.common.vllm_pilot import select_pilot_methods

def test_vllm_methods_includes_dflash_phase_a():
    assert "dflash_phase_a" in METHODS
    assert DRAFT_ARGUMENTS.get("dflash_phase_a") == "dflash_model"

def test_select_pilot_methods_allows_dflash_phase_a():
    selected = select_pilot_methods(
        "vanilla_vllm,dflash,dflash_phase_a",
        available_methods=METHODS,
        reference="vanilla_vllm",
    )
    assert "dflash_phase_a" in selected

def test_build_speculative_config_for_dflash_phase_a():
    fake_config = {"block_size": 16}
    spec, meta = build_speculative_config(
        "dflash_phase_a",
        "/path/to/dflash",
        fake_config,
        target_num_hidden_layers=36,
    )
    assert spec["method"] == "dflash"
    assert spec["num_speculative_tokens"] == 16
    assert meta["vllm_method"] == "dflash"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_vllm_phase_a.py -v`  
Expected: FAIL (`AssertionError: assert 'dflash_phase_a' in METHODS`).

- [ ] **Step 3: Implement vLLM integration in `vllm_all_baselines.py` and `vllm_pilot.py`**

- In `vllm_all_baselines.py`:
  - Add `"dflash_phase_a"` to `METHODS`.
  - Add `"dflash_phase_a": "dflash_model"` to `DRAFT_ARGUMENTS`.
  - In `build_speculative_config`: map `"dflash_phase_a"` to `vllm_method = "dflash"`.
- In `vllm_pilot.py`:
  - Implement `install_phase_a_vllm_compat()`:
    Hook `DFlashSpeculator._run_model` (or `_generate_draft`):
    When request method is `dflash_phase_a`:
    Filter the active block table entries in `attn_metadata` using `select_focused_paged_blocks` so the draft attention executes over only the focused blocks ($B=2048$ tokens).
- In `scripts/run_vllm_all.sh`:
  - Update `METHODS` default list or documentation to include `dflash_phase_a`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src .venv/bin/pytest tests/test_vllm_phase_a.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/Benchmark/common/vllm_pilot.py src/Benchmark/vllm_all_baselines.py scripts/run_vllm_all.sh tests/test_vllm_phase_a.py
git commit -m "feat(vllm): integrate DFlash Phase A context focusing into vLLM benchmark pipeline"
```

---

### Task 4: Documentation & Full Test Suite Verification

**Files:**
- Create: `docs/baselines/dflash_phase_a.md`
- Modify: `src/Benchmark/README.md`
- Modify: `docs/flashattn4_benchmark.md`

- [ ] **Step 1: Write documentation for running Phase A on B200**

Create `docs/baselines/dflash_phase_a.md` explaining:
- Theory & Winning recipe ($L_{18}$, Multi-Query Max-Pooling, Chunk 128, Sink 16 + Recent 32, $B=2048$)
- How to run via vLLM pipeline: `bash scripts/run.sh vllm_all --methods vanilla_vllm,dflash,dflash_phase_a`
- How to run via FA4 pipeline: `bash scripts/run.sh fa4_native --methods vanilla_hf,dflash,dflash_phase_a`
- Environment variables: `DFLASH_PHASE_A_BUDGET`, `DFLASH_PHASE_A_CHUNK_SIZE`, `DFLASH_PHASE_A_LAYER`.

- [ ] **Step 2: Run all regression unit tests**

Run:
```bash
PYTHONPATH=src .venv/bin/pytest \
  tests/test_phase_a_selector.py \
  tests/test_benchmark_fa4_phase_a.py \
  tests/test_vllm_phase_a.py \
  tests/test_vllm_pilot.py \
  tests/test_benchmark_vietnamese_fa4.py -v
```
Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add docs/baselines/dflash_phase_a.md src/Benchmark/README.md docs/flashattn4_benchmark.md
git commit -m "docs(benchmark): add DFlash Phase A execution guide for B200 server"
```
