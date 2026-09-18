from __future__ import annotations

import math

import pytest

from src.TrainingFree.lease import (
    LeaseState,
    actual_mass,
    certificate_from_anchor,
    select_hot_cold,
)


def test_select_hot_cold_keeps_anchor_mass_above_target() -> None:
    hot, cold = select_hot_cold([0.6, 0.25, 0.1, 0.05], delta_anchor=0.1)

    assert hot == [0, 1, 2]
    assert cold == [3]


def test_certificate_upper_bounds_bruteforce_cold_mass() -> None:
    anchor_logits = [2.0, 0.5, -1.0, 1.0]
    current_logits = [1.4, 1.0, -0.5, 0.8]
    hot, cold = [0, 1], [2, 3]
    log_z0 = [value for value in anchor_logits]
    kappa = [1.0, 1.0, 1.0, 1.0]
    drift = math.sqrt(sum((a - b) ** 2 for a, b in zip(anchor_logits, current_logits)))

    bound = certificate_from_anchor(
        log_z0,
        kappa,
        hot,
        cold,
        live_log_z=None,
        drift=drift,
    )
    actual = actual_mass(current_logits, cold)

    assert 0.0 <= actual <= bound + 1e-5


def test_certificate_stays_finite_for_large_logits() -> None:
    bound = certificate_from_anchor(
        [1000.0, 999.0],
        [2.0, 3.0],
        [0],
        [1],
        live_log_z=998.0,
        drift=10.0,
    )

    assert math.isfinite(bound)
    assert 0.0 <= bound <= 1.0


def test_empty_cold_set_has_zero_bound() -> None:
    assert certificate_from_anchor([0.0], [1.0], [0], [], None, 100.0) == 0.0


def test_lease_expires_at_first_invalid_step() -> None:
    state = LeaseState(
        anchor_query=[0.0, 0.0],
        log_z0=[0.0, -3.0],
        kappa=[1.0, 1.0],
        hot=[0],
        cold=[1],
        delta_cert=0.2,
        anchor_step=4,
    )

    valid = state.step([0.0, 0.0], None, actual_cold_mass=0.01, step_index=5)
    expired = state.step([2.0, 0.0], None, actual_cold_mass=0.2, step_index=6)

    assert valid["valid"] is True
    assert valid["expired"] is False
    assert expired["valid"] is False
    assert expired["expired"] is True
    assert expired["anchor_step"] == 4
    assert expired["step"] == 6


def test_rejects_invalid_certificate_inputs() -> None:
    with pytest.raises(ValueError, match="delta_anchor"):
        select_hot_cold([1.0], delta_anchor=1.0)
    with pytest.raises(ValueError, match="aligned"):
        certificate_from_anchor([0.0], [1.0, 2.0], [0], [], None, 0.0)
