"""Behavioral contracts for AMR-DFlash memory and training primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from AMR_DFlash.core import (  # noqa: E402
    AcceptanceSelector,
    ComplementaryCompressor,
    alignment_kl_loss,
    build_sparse_attention_mask,
    greedy_acceptance,
    preference_loss,
)


def test_selector_keeps_recent_guard_inside_raw_budget_and_returns_unique_positions():
    selector = AcceptanceSelector(hidden_size=4, index_dim=3)
    with torch.no_grad():
        selector.key_projection.weight.zero_()
        selector.key_projection.weight[0, 3] = 1.0
        selector.key_projection.weight[1, 1] = 1.0
        selector.key_projection.weight[2, 2] = 1.0
        selector.query_projection.weight.zero_()
        selector.query_projection.weight[0, 3] = 1.0
        selector.position_bias.zero_()
    features = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
          [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]]
    )
    positions = torch.arange(6).unsqueeze(0)
    anchor = torch.tensor([[0.0, 0.0, 0.0, 1.0]])

    selected = selector.select(features, positions, anchor, raw_budget=3, local_window=2)

    assert selected.tolist() == [[3, 4, 5]]
    assert selected.shape[1] == 3
    assert len(set(selected[0].tolist())) == 3


def test_sparse_attention_mask_preserves_selected_raw_and_live_keys_and_gates_slots():
    mask = build_sparse_attention_mask(
        raw_positions=torch.tensor([[1, 7]]),
        slot_positions=torch.tensor([[4]]),
        live_positions=torch.tensor([[8, 9, 10]]),
        query_length=3,
        slot_gate=-2.0,
        dtype=torch.float32,
    )

    assert mask.shape == (1, 1, 3, 6)
    assert torch.equal(mask[0, 0, 0, :2], torch.zeros(2))
    assert mask[0, 0, 0, 2].item() == pytest.approx(-2.0)
    assert torch.equal(mask[0, 0, 0, 3:], torch.zeros(3))


def test_streaming_compressor_matches_single_pass_and_slot_positions_are_observed():
    torch.manual_seed(3)
    compressor = ComplementaryCompressor(hidden_size=5, num_slots=3)
    features = torch.randn(1, 7, 5)
    positions = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])

    whole = compressor(features, positions)
    state = compressor.empty_state(batch_size=1, device=features.device, dtype=features.dtype)
    compressor.update(state, features[:, :3], positions[:, :3])
    compressor.update(state, features[:, 3:], positions[:, 3:])
    streamed, streamed_positions = compressor.finalize(state)

    assert torch.allclose(whole.values, streamed, atol=1e-6, rtol=1e-6)
    assert torch.equal(whole.positions, streamed_positions)
    assert int(streamed_positions.max()) <= 6


def test_preference_loss_rewards_correct_pair_and_sends_gradient_to_scores():
    positive = torch.tensor([2.0], requires_grad=True)
    negative = torch.tensor([0.0], requires_grad=True)

    loss = preference_loss(positive, negative, torch.ones(1))
    loss.backward()

    assert loss.item() < 0.2
    assert positive.grad.item() < 0
    assert negative.grad.item() > 0


def test_alignment_loss_ignores_invalid_horizon_and_backpropagates_only_valid_rows():
    draft = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [20.0, -20.0]]], requires_grad=True)
    target = torch.tensor([[[3.0, 0.0], [0.0, 3.0], [-20.0, 20.0]]])

    loss = alignment_kl_loss(draft, target, torch.tensor([[True, True, False]]))
    loss.backward()

    assert loss.item() < 0.02
    assert torch.count_nonzero(draft.grad[0, :2]).item() > 0
    assert torch.count_nonzero(draft.grad[0, 2]).item() == 0
    assert target.grad is None


def test_greedy_acceptance_reports_first_mismatch_and_full_acceptance():
    draft = torch.tensor([[10, 11, 12, 13]])
    target = torch.tensor([[10, 99, 12, 13]])

    assert greedy_acceptance(draft, target).accepted == 1
    assert greedy_acceptance(draft, target).first_mismatch == 1
    full = greedy_acceptance(draft, draft)
    assert full.accepted == 4
    assert full.first_mismatch is None
