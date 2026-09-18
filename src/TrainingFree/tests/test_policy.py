from __future__ import annotations

import pytest

from src.TrainingFree.policy import (
    RecapConfig,
    RecapState,
    adaptive_budget,
    compute_consumption,
    compute_current_relevance,
    residual_utility,
    select_topk,
    update_residual,
)


def test_current_relevance_is_normalized_and_uses_best_prototype() -> None:
    relevance = compute_current_relevance(
        [[1.0, 0.0]],
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0]],
        ],
        temperature=0.1,
    )

    assert sum(relevance) == pytest.approx(1.0)
    assert relevance[0] > relevance[1]


def test_consumption_and_residual_update_respect_floor() -> None:
    consumption = compute_consumption(
        [0.8, 0.2],
        [1.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    residual = update_residual(
        [1.0, 1.0], consumption, eta=10.0, floor=0.4
    )

    assert consumption[0] > consumption[1]
    assert residual[0] == pytest.approx(0.4)
    assert residual[1] == pytest.approx(1.0)


def test_recap_state_can_reactivate_a_demoted_block() -> None:
    state = RecapState(
        RecapConfig(k_min=1, k_max=1, beta=0.8, eta=4.0, residual_floor=0.1),
        residual=[1.0, 1.0],
    )
    first = state.step(
        [[1.0, 0.0]],
        [1.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [[[1.0, 0.0]], [[0.0, 1.0]]],
    )
    second = state.step(
        [[0.0, 1.0]],
        [0.0, 1.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [[[1.0, 0.0]], [[0.0, 1.0]]],
    )

    assert first["active"] == [0]
    assert second["active"] == [1]
    assert state.residual[0] < 1.0


def test_recap_state_can_advance_from_an_observed_relevance_distribution() -> None:
    state = RecapState(RecapConfig(k_min=1, k_max=2), residual=[1.0, 1.0])

    result = state.step_with_relevance(
        [0.9, 0.1],
        [1.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
    )

    assert result["active"][0] == 0
    assert state.residual[0] < 1.0


def test_budget_and_topk_are_bounded_and_deterministic() -> None:
    assert adaptive_budget([0.25, 0.25, 0.25, 0.25], 1, 3) == 3
    assert adaptive_budget([1.0, 0.0, 0.0, 0.0], 1, 3) == 1
    assert select_topk([0.5, 0.5, 0.1], 2) == [0, 1]


def test_policy_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="temperature"):
        compute_current_relevance([[1.0]], [[[1.0]]], temperature=0.0)
    with pytest.raises(ValueError, match="beta"):
        residual_utility([0.5], [1.0], beta=1.0)
    with pytest.raises(ValueError, match="floor"):
        update_residual([1.0], [0.1], eta=1.0, floor=0.0)
