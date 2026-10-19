"""Unit tests for Phase B integration across Native FA4 and vLLM research."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
import torch
import torch.nn as nn

from TrainingFree.context_adaptive.cache import DraftContextBank
from TrainingFree.context_adaptive.types import Selection
from TrainingFree.speculation_experiments.b_controller import (
    VerificationPrefixController,
    FixedVerificationPrefixController,
)
from Benchmark.common.flashattn_runtime import (
    SUPPORTED_METHODS,
    resolve_flashattn_methods,
)
from Benchmark.vllm_research.plan import build_jobs
from Benchmark.vllm_research.compat import configure_worker_variant


def test_native_fa4_method_resolution():
    assert "dflash_phase_b" in SUPPORTED_METHODS
    resolved = resolve_flashattn_methods("vanilla_hf,dflash,dflash_phase_b")
    assert "dflash_phase_b" in resolved
    assert "dflash" in resolved
    assert "vanilla_hf" in resolved


def test_draft_context_bank_zero_copy_view():
    class DummyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 64
            self.config = SimpleNamespace(num_key_value_heads=2)

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummyAttention()

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DummyLayer() for _ in range(2)])
            self.param = nn.Parameter(torch.zeros(1))

    model = DummyModel()
    bank = DraftContextBank(model, capacity=32)
    bank.length = 16

    # 1. Full contiguous slice: 0..15 -> must return zero-copy view
    full_selection = Selection(
        positions_by_layer=(tuple(range(16)), tuple(range(16))),
        requested_budget="full",
        selector_id="full",
    )
    gathered_full = bank.gather(full_selection)
    assert len(gathered_full) == 2
    for layer_ctx, storage in zip(gathered_full, bank.layer_storage):
        assert layer_ctx.key.shape == (1, 2, 16, 64)
        assert layer_ctx.value.shape == (1, 2, 16, 64)
        # Verify it's a view into storage without data copying:
        assert layer_ctx.key.data_ptr() == storage.key.data_ptr()
        assert layer_ctx.value.data_ptr() == storage.value.data_ptr()

    # 2. Subset selection: e.g. [0, 2, 4] -> must use index_select (new tensor)
    subset_selection = Selection(
        positions_by_layer=((0, 2, 4), (0, 2, 4)),
        requested_budget=3,
        selector_id="subset",
    )
    gathered_subset = bank.gather(subset_selection)
    for layer_ctx, storage in zip(gathered_subset, bank.layer_storage):
        assert layer_ctx.key.shape == (1, 2, 3, 64)
        assert layer_ctx.key.data_ptr() != storage.key.data_ptr()


def test_vllm_research_b3_plan_and_hazard_variants():
    jobs = build_jobs(["B3"])
    job_variants = [j.variant for j in jobs]
    assert "adaptive_k_R-Hazard-Prefix" in job_variants
    assert "adaptive_k_R-Hazard-Suffix-Margin" in job_variants
    assert "adaptive_k_R-Hazard-Pos" in job_variants

    # Test configure_worker_variant with hazard variant
    hazard_model = {
        "model_type": "hazard",
        "variant": "R-Hazard-Prefix",
        "max_candidates": 15,
        "features": ["position", "draft_present_log_margin", "draft_prefix_tv"],
        "beta": [-0.5, 0.15, -0.8, -0.4],
        "center": [8.0, 1.5, 0.2],
        "scale": [4.32, 1.2, 0.15],
        "training_groups": ["calibration"],
    }
    cost_profiles = {
        "gov_report::context=0": {
            "cost_by_prefix": {"3": 12.0, "7": 18.0, "11": 24.0, "15": 30.0}
        }
    }
    configured = configure_worker_variant(
        "b3",
        {"model": hazard_model, "cost_profiles": cost_profiles, "dataset": "gov_report"},
    )
    assert configured["engine_overrides"]["max_num_seqs"] == 1


def test_dflash_generate_with_phase_b_controller():
    from externals.dflash.dflash.model import dflash_generate

    vocab_size = 100
    hidden_size = 64
    block_size = 16

    class DummyTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device("cpu")
            self.model = SimpleNamespace(
                embed_tokens=nn.Embedding(vocab_size, hidden_size)
            )
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            self.forward_calls = []

        def forward(self, input_ids, position_ids=None, past_key_values=None, use_cache=True, logits_to_keep=None, output_hidden_states=False):
            self.forward_calls.append(input_ids.shape[1])
            batch, seq_len = input_ids.shape
            logits = torch.randn(batch, seq_len, vocab_size)
            if logits_to_keep is not None:
                logits = logits[:, -logits_to_keep:, :]
            hidden_states = (
                torch.randn(batch, seq_len, hidden_size),
                torch.randn(batch, seq_len, hidden_size),
            )
            return SimpleNamespace(logits=logits, hidden_states=hidden_states)

    class DummyDraft(nn.Module):
        def __init__(self):
            super().__init__()
            self.block_size = block_size
            self.mask_token_id = 0
            self.target_layer_ids = [0]

        def forward(self, target_hidden, noise_embedding, position_ids, past_key_values, use_cache, is_causal):
            batch, seq_len, _ = noise_embedding.shape
            return torch.randn(batch, seq_len, hidden_size)

    target = DummyTarget()
    draft = DummyDraft()

    # Controller that always requests fixed prefix k=3
    fixed_controller = FixedVerificationPrefixController(prefix=3)

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    stats = dflash_generate(
        draft,
        target,
        input_ids,
        max_new_tokens=8,
        stop_token_ids=None,
        temperature=0.0,
        block_size=block_size,
        return_stats=True,
        verification_prefix_controller=fixed_controller,
    )

    assert stats is not None
    assert len(stats.output_ids[0]) > 5
    # Verification forward calls (excluding prefill of length 5):
    # Since controller restricts verification prefix to k=3, verification length must be k+1 = 4!
    verify_calls = target.forward_calls[1:]
    assert len(verify_calls) > 0
    for call_len in verify_calls:
        # verification length should be <= 4 (k=3 plus current token)
        assert call_len <= 4, f"Expected verify call <= 4 with k=3 controller, got {call_len}"
