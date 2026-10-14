from __future__ import annotations

import pytest
import torch

from TrainingFree.context_adaptive.types import PromptLayout
from TrainingFree.context_interventions.masks import (
    build_region_positions,
    exact_budget_positions,
    build_online_selection_policy,
    matched_deletion_positions,
    remove_attention_positions,
)
from TrainingFree.context_interventions.attention import full_key_attention_scores
from TrainingFree.context_interventions.matrix import a1_matrix, a2_matrix, a3_matrix
from TrainingFree.context_interventions.replay import choose_reference_prefixes


def _layout() -> PromptLayout:
    # Prompt: [instruction, source x4, query], then four processed outputs.
    source = (1, 2, 3, 4)
    return PromptLayout(
        prompt_tokens=6,
        source_positions=source,
        global_positions=(0, 5),
        source_chunks=((1, 2, 3, 4),),
        prompt_hash="prompt",
        source_available=True,
    )


def test_region_masks_separate_source_recent_and_old_output() -> None:
    layout = _layout()
    kwargs = dict(
        layout=layout,
        bank_length=10,
        processed_output_start=6,
        source_anchors=2,
        recent_output=2,
    )
    assert build_region_positions("source_only", **kwargs) == tuple(range(6))
    assert build_region_positions("recent_only", **kwargs) == (0, 1, 2, 5, 8, 9)
    assert build_region_positions("source_recent", **kwargs) == (0, 1, 2, 3, 4, 5, 8, 9)
    assert build_region_positions("full", **kwargs) == tuple(range(10))


def test_exact_budget_keeps_global_and_protected_positions_and_is_exact() -> None:
    result = exact_budget_positions(
        bank_length=12,
        budget=7,
        protected_positions=(0, 1, 2),
        scores={position: float(position) for position in range(12)},
        strategy="score_high",
    )
    assert len(result) == 7
    assert set((0, 1, 2)).issubset(result)
    assert result == (0, 1, 2, 8, 9, 10, 11)


def test_exact_budget_fails_when_protected_set_exceeds_budget() -> None:
    with pytest.raises(ValueError, match="protected"):
        exact_budget_positions(
            bank_length=8,
            budget=2,
            protected_positions=(0, 1, 2),
            strategy="recency",
        )


def test_attention_deletion_removes_exact_count_with_stable_ties() -> None:
    positions = remove_attention_positions(
        bank_length=8,
        protected_positions=(0, 1),
        scores={i: 0.5 for i in range(8)},
        count=3,
        side="low",
    )
    assert positions == (0, 1, 5, 6, 7)


def test_matched_deletion_is_deterministic_and_does_not_remove_protected() -> None:
    kwargs = dict(
        bank_length=20,
        region_positions=(4, 5, 6, 7, 8, 9),
        protected_positions=(0, 1, 4),
        count=3,
        seed=42,
    )
    first = matched_deletion_positions(**kwargs)
    assert first == matched_deletion_positions(**kwargs)
    assert len(first) == 17
    assert {0, 1, 4}.issubset(first)


def test_full_key_attention_scores_are_causal_and_handle_gqa() -> None:
    # One query head, one KV head; future key 2 must never enter normalization.
    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor([[[[0.0, 0.0], [1.0, 0.0], [100.0, 0.0]]]])
    scores = full_key_attention_scores(
        query,
        key,
        query_index=0,
        query_position=1,
        head_groups=1,
        scaling=1.0,
    )
    assert set(scores) == {0, 1}
    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores[1] > scores[0]


def test_attention_finite_check_does_not_sync_once_per_context_token(monkeypatch):
    # Tensor.item() on CUDA synchronizes the host with a device scalar. The
    # number of such operations must remain constant as context grows.
    original_item = torch.Tensor.item
    scalar_reads = []

    def count_item(tensor, *args):
        scalar_reads.append(tensor.numel())
        return original_item(tensor, *args)

    monkeypatch.setattr(torch.Tensor, "item", count_item)
    scores = full_key_attention_scores(
        torch.ones(1, 1, 1, 2), torch.zeros(1, 1, 128, 2),
        query_index=0, query_position=127, head_groups=1, scaling=1.0,
    )
    assert len(scores) == 128
    assert sum(scores.values()) == pytest.approx(1.0)
    assert len(scalar_reads) <= 2


def test_b200_matrix_contains_registered_primary_cells() -> None:
    assert len(a1_matrix("pilot")["attention_scoring"]) == 7
    assert a1_matrix("dev")["attention_deletions"] == (1024, 2048, 4096)
    assert a2_matrix("confirm")["budgets"] == (2048,)
    assert a3_matrix("dev")["budgets"] == (512, 1024, 2048, 4096, 8192, "full")


def test_online_previous_selector_uses_per_layer_exact_masks_and_full_refresh() -> None:
    policy = build_online_selection_policy("previous_dflash", budget=6, source_anchors=1, recent_output=1)
    state = type("State", (), {"ranking_age": 1})()
    common = dict(
        state=state, layout=_layout(), bank_length=10, processed_output_start=6,
        requested_budget=6, round_index=2, refresh_required=False,
        target_parent_scores=None, num_draft_layers=2,
    )
    result = policy(previous_dflash_scores={0: {i: float(i) for i in range(10)}, 1: {i: float(-i) for i in range(10)}}, **common)
    assert [len(layer) for layer in result.positions_by_layer] == [6, 6]
    assert result.positions_by_layer[0] != result.positions_by_layer[1]
    refreshed = policy(previous_dflash_scores={}, **{**common, "requested_budget": "full", "refresh_required": True})
    assert refreshed.positions_by_layer == (tuple(range(10)), tuple(range(10)))


def test_reference_prefix_selection_uses_regular_rounds_and_causal_previous_rounds() -> None:
    output = list(range(80))
    rounds = [
        {"type": "round", "round_index": index, "gamma_executed": 15,
         "processed_output_before": processed, "logical_length_before": 100 + processed}
        for index, processed in enumerate((0, 16, 32, 48, 64))
    ]
    selected = choose_reference_prefixes(
        rounds, output, prompt_length=100, max_new_tokens=80, eos_ids=set(), max_prefixes=6,
    )
    assert selected
    assert selected[0].processed_output_tokens == 0
    assert selected[0].previous_dense_round_index is None
    assert all(prefix.remaining_output_tokens >= 16 for prefix in selected)
    for prefix in selected[1:]:
        assert prefix.previous_dense_round_index is not None
        assert prefix.previous_dense_round_index < prefix.round_index


def test_reference_prefix_selection_stops_before_eos_and_short_tail() -> None:
    output = list(range(40))
    output[25] = 99
    rounds = [
        {"type": "round", "round_index": index, "gamma_executed": 15,
         "processed_output_before": processed, "logical_length_before": 10 + processed}
        for index, processed in enumerate((0, 16, 24))
    ]
    selected = choose_reference_prefixes(
        rounds, output, prompt_length=10, max_new_tokens=40, eos_ids={99}, max_prefixes=6,
    )
    assert all(prefix.processed_output_tokens < 25 for prefix in selected)
    assert all(25 - prefix.processed_output_tokens >= 16 for prefix in selected)
