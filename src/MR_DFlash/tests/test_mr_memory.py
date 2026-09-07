"""Contract tests CPU cho memory HCA/CSA của MR-DFlash."""

from __future__ import annotations

import torch

from MR_DFlash.memory import CSAIndexer, MRMemoryState, MRTargetMemory


def test_hca_csa_compress_and_index_cpu() -> None:
    torch.manual_seed(0)
    memory = MRTargetMemory(
        input_dim=12,
        hidden_size=8,
        hca_compression_ratio=4,
        csa_compression_ratio=2,
        local_window=3,
        csa_top_k=5,
        indexer_dim=4,
    )
    features = torch.randn(2, 9, 12, requires_grad=True)

    state = memory.build(features)

    assert isinstance(state, MRMemoryState)
    assert state.hca.shape == (2, 2, 8)
    assert state.csa.shape == (2, 4, 8)
    assert state.pending_hca.shape == (2, 1, 8)
    assert state.pending_csa.shape == (2, 1, 8)
    assert state.local_hca.shape == (2, 3, 8)
    assert state.local_csa.shape == (2, 3, 8)
    assert state.hca_positions.shape == (2, 2)
    assert state.csa_positions.shape == (2, 4)
    assert torch.isfinite(state.hca).all()
    assert torch.isfinite(state.csa).all()

    indices, scores = memory.indexer.select(
        query=torch.randn(2, 2, 8),
        csa_memory=state.csa,
        top_k=5,
    )
    assert indices.shape == (2, 2, 4)
    assert scores.shape == (2, 2, 4)
    assert indices.max().item() < state.csa.shape[1]
    assert torch.isfinite(scores).all()

    loss = state.hca.square().mean() + state.csa.square().mean()
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_topk_is_clamped_for_short_memory() -> None:
    indexer = CSAIndexer(hidden_size=8, indexer_dim=4)
    indices, scores = indexer.select(
        query=torch.randn(1, 1, 8),
        csa_memory=torch.randn(1, 2, 8),
        top_k=64,
    )
    assert indices.shape == (1, 1, 2)
    assert scores.shape == (1, 1, 2)


def test_append_only_updates_accepted_tokens() -> None:
    torch.manual_seed(1)
    memory = MRTargetMemory(
        input_dim=12,
        hidden_size=8,
        hca_compression_ratio=4,
        csa_compression_ratio=2,
        local_window=3,
        csa_top_k=4,
    )
    initial = memory.build(torch.randn(1, 5, 12))
    accepted = torch.randn(1, 2, 12)

    updated = memory.append(
        initial,
        accepted,
        positions=torch.tensor([[5, 6]]),
    )
    assert updated.total_tokens == 7
    assert updated.local_positions.tolist() == [[4, 5, 6]]
    assert updated.pending_hca_positions.shape[1] < 4
    assert updated.pending_csa_positions.shape[1] < 2

    rejected = memory.append(
        initial,
        accepted[:, :0],
        positions=torch.empty(1, 0, dtype=torch.long),
    )
    assert rejected.total_tokens == initial.total_tokens
    assert torch.equal(rejected.local_positions, initial.local_positions)


def test_build_then_append_matches_full_build_with_partial_tails() -> None:
    torch.manual_seed(12)
    memory = MRTargetMemory(
        input_dim=6,
        hidden_size=4,
        hca_compression_ratio=4,
        csa_compression_ratio=2,
        local_window=3,
    )
    features = torch.randn(1, 5, 6)

    incremental = memory.build(features[:, :3])
    incremental = memory.append(
        incremental,
        features[:, 3:],
        positions=torch.tensor([[3, 4]]),
    )
    full = memory.build(features)

    assert torch.allclose(incremental.hca, full.hca)
    assert torch.allclose(incremental.csa, full.csa)
    assert torch.equal(incremental.hca_positions, full.hca_positions)
    assert torch.equal(incremental.csa_positions, full.csa_positions)
    assert torch.allclose(incremental.pending_hca, full.pending_hca)
    assert torch.allclose(incremental.pending_csa, full.pending_csa)
    assert torch.equal(incremental.pending_hca_positions, full.pending_hca_positions)
    assert torch.equal(incremental.pending_csa_positions, full.pending_csa_positions)


def test_hca_and_csa_pending_positions_are_independent() -> None:
    torch.manual_seed(13)
    memory = MRTargetMemory(
        input_dim=6,
        hidden_size=4,
        hca_compression_ratio=4,
        csa_compression_ratio=2,
        local_window=3,
    )
    state = memory.build(torch.randn(1, 5, 6))
    updated = memory.append(
        state,
        torch.randn(1, 1, 6),
        positions=torch.tensor([[5]]),
    )

    assert updated.pending_hca_positions.tolist() == [[4, 5]]
    assert updated.pending_csa_positions.numel() == 0

    # A subsequent append must not feed HCA pending positions into CSA.
    next_state = memory.append(
        updated,
        torch.randn(1, 1, 6),
        positions=torch.tensor([[6]]),
    )
    assert next_state.pending_csa_positions.tolist() == [[6]]


def test_build_exposes_anchor_relative_local_windows() -> None:
    torch.manual_seed(14)
    memory = MRTargetMemory(
        input_dim=6,
        hidden_size=4,
        hca_compression_ratio=4,
        csa_compression_ratio=2,
        local_window=2,
    )
    features = torch.randn(1, 10, 6)
    anchors = torch.tensor([[3, 8]])
    state = memory.build(features, query_positions=anchors)

    assert state.local_hca.shape == (1, 2, 2, 4)
    assert state.local_csa.shape == (1, 2, 2, 4)
    assert state.local_positions.tolist() == [[ [1, 2], [6, 7] ]]
