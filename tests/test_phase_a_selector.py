import pytest
import torch
from Benchmark.common.phase_a_selector import (
    PhaseAConfig,
    compute_multi_query_scores,
    compute_multi_query_scores_from_qk,
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
    monkeypatch.setenv("DFLASH_PHASE_A_SINK", "8")
    monkeypatch.setenv("DFLASH_PHASE_A_RECENT", "16")
    cfg_env = PhaseAConfig.from_env()
    assert cfg_env.budget == 1024
    assert cfg_env.target_layer == 20
    assert cfg_env.chunk_size == 64
    assert cfg_env.sink_tokens == 8
    assert cfg_env.recent_tokens == 16


def test_compute_multi_query_scores_max_pooling():
    # 3 accepted queries, 10 context tokens
    attn = torch.tensor([
        [0.1, 0.5, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1],
        [0.0, 0.2, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0],
        [0.4, 0.1, 0.1, 0.0, 0.9, 0.0, 0.0, 0.0, 0.0, 0.2],
    ])
    scores = compute_multi_query_scores(attn, accepted_len=3, query_pooling="max")
    expected = torch.tensor([0.4, 0.5, 0.8, 0.1, 0.9, 0.1, 0.0, 0.0, 0.3, 0.2])
    assert torch.allclose(scores, expected)


def test_compute_multi_query_scores_from_qk():
    # Test lightweight GEMM from Q and K
    batch, heads, num_queries, head_dim = 1, 2, 2, 4
    context_len = 8
    q = torch.randn(batch, heads, num_queries, head_dim)
    k = torch.randn(batch, heads, context_len, head_dim)
    scaling = 0.5
    scores = compute_multi_query_scores_from_qk(q, k, scaling=scaling, query_pooling="max")
    assert scores.shape == (context_len,)
    assert scores.ndim == 1
    # Check that scores are nonnegative (after softmax or normalization)
    assert (scores >= 0).all()


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
        assert i in indices.tolist()
    # Must preserve recent tokens 1024-32..1023
    for i in range(1024 - 32, 1024):
        assert i in indices.tolist()
    # Must contain high-scoring chunk 2
    assert 256 in indices.tolist()
    assert 383 in indices.tolist()
    # Must be sorted and unique
    assert torch.equal(indices, torch.unique(indices))
    assert torch.equal(indices, torch.sort(indices).values)


def test_select_focused_context_indices_full_budget_fallback():
    # When budget >= total_length, return all indices
    cfg = PhaseAConfig(chunk_size=128, sink_tokens=16, recent_tokens=32, budget=2048)
    scores = torch.rand(512)
    indices = select_focused_context_indices(scores, total_length=512, config=cfg)
    assert len(indices) == 512
    assert indices.tolist() == list(range(512))


def test_select_focused_paged_blocks():
    # Context 1024 tokens, block_size=16 -> 64 blocks
    # chunk_size=128 -> 8 blocks per chunk
    cfg = PhaseAConfig(chunk_size=128, sink_tokens=16, recent_tokens=32, budget=256)
    scores = torch.zeros(1024)
    scores[256:384] = 10.0  # Chunk 2
    block_indices = select_focused_paged_blocks(scores, total_length=1024, block_size=16, config=cfg)
    block_list = block_indices.tolist()
    # Block 0 (sink) must be in block_indices
    assert 0 in block_list
    # Blocks for recent (last 32 tokens = 2 blocks = 62, 63)
    assert 62 in block_list
    assert 63 in block_list
    # Chunk 2 (tokens 256..383) = blocks 16, 17, ..., 23
    for b in range(16, 24):
        assert b in block_list
