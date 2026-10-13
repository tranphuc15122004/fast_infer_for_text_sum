from __future__ import annotations

import pytest

from TrainingFree.speculation_experiments.replay import expected_prefix_acceptance


@pytest.mark.parametrize(
    ("full_accepted", "k", "expected"),
    [(0, 3, 0), (2, 3, 2), (3, 3, 3), (8, 7, 7), (15, 11, 11), (15, 15, 15)],
)
def test_prefix_verification_acceptance_identity(full_accepted: int, k: int, expected: int) -> None:
    assert expected_prefix_acceptance(full_accepted, k) == expected


def test_prefix_verification_rejects_unsupported_grid() -> None:
    with pytest.raises(ValueError, match="verify_prefix"):
        expected_prefix_acceptance(3, 8)
