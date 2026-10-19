import pytest
import os
import torch
from Benchmark.common.flashattn_runtime import SUPPORTED_METHODS
from Benchmark.native_fa4_benchmark import MODEL_REPOS, detect_default_datasets


def test_fa4_supported_methods_includes_dflash_phase_a():
    assert "dflash_phase_a" in SUPPORTED_METHODS
    assert "dflash_phase_a" in MODEL_REPOS


def test_phase_a_selector_integration():
    from Benchmark.common.phase_a_selector import PhaseAConfig, PhaseAContextSelector
    import torch.nn as nn

    # Mock target model with layer 18
    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.head_dim = 64
            self.scaling = 0.125
            self.q_proj = nn.Linear(128, 128)
            self.k_proj = nn.Linear(128, 128)
            self.q_norm = nn.LayerNorm(64)
            self.config = type("Config", (), {"num_attention_heads": 2})()

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = MockAttention()

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer() for _ in range(36)])

    class MockTarget(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = MockModel()

    target = MockTarget()
    cfg = PhaseAConfig(target_layer=18, budget=256, chunk_size=128)
    selector = PhaseAContextSelector(target, cfg)

    # Test begin/end round hooks
    selector.begin_verification_round()
    dummy_input = torch.randn(1, 4, 128)
    # Simulate forward through q_proj
    _ = selector.capture.module.q_proj(dummy_input)

    class MockCache:
        def __init__(self):
            self.key_cache = [torch.randn(1, 2, 1024, 64) for _ in range(36)]

    cache = MockCache()
    indices = selector.end_verification_round(cache, accepted_length=4, total_length=1024)
    assert indices.ndim == 1
    assert len(indices) <= 256
    selector.close()


def test_fa4_server_supports_dflash_phase_a():
    from Benchmark.fa4_server import SERVER_MODEL_ENV, SERVER_MODEL_ALIASES, SUPPORTED_METHODS, resolve_server_models

    assert "dflash_phase_a" in SUPPORTED_METHODS
    assert "dflash_phase_a" in SERVER_MODEL_ENV
    assert "dflash_phase_a" in SERVER_MODEL_ALIASES

    env = {"MODEL_TARGET": "/path/target", "MODEL_DFLASH_DRAFT": "/path/draft"}
    resolved = resolve_server_models(env, "vanilla_hf,dflash,dflash_phase_a")
    assert resolved["vanilla_hf"] == "/path/target"
    assert resolved["dflash"] == "/path/draft"
    assert resolved["dflash_phase_a"] == "/path/draft"
