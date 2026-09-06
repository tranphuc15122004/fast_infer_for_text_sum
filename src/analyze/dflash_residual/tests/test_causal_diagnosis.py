from __future__ import annotations

from src.analyze.dflash_residual.causal_diagnosis import (
    compare_training_to_utility,
    theoretical_decay_weights,
)


def test_theoretical_decay_weights_matches_dflash_gamma():
    weights = theoretical_decay_weights(5, 7.0)
    assert weights[0] == 1.0
    assert weights[1] < weights[0]
    assert weights[-1] == __import__("math").exp(-3.0 / 7.0)


def test_utility_compare_normalizes_and_reports_critical_mass():
    result = compare_training_to_utility(
        {"1": 2.0, "3": 1.0, "8": 1.0},
        {"1": 1.0, "3": 2.0, "8": 1.0},
    )
    assert abs(sum(result["training_normalized"].values()) - 1.0) < 1e-9
    assert abs(sum(result["utility_normalized"].values()) - 1.0) < 1e-9
    assert result["critical_training_mass_3_8"] > 0.0
