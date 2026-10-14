from __future__ import annotations

from TrainingFree.context_adaptive.config import AdaptiveConfig
from TrainingFree.context_adaptive.controller import AdaptiveController
from TrainingFree.context_adaptive.types import Action, RoundState


def test_verification_prefix_history_uses_executed_prefix_length() -> None:
    controller = AdaptiveController(
        AdaptiveConfig(budgets=("full",), gammas=(3, 7, 11, 15)),
        statistics=None,
        variant="dflash_full_fixed",
    )
    state = RoundState(
        round_index=0,
        logical_length=4096,
        processed_output_tokens=0,
        remaining_output_tokens=128,
        parent_entropy=0.3,
        source_concentration=None,
        history_acceptance=None,
        ranking_age=0,
        refresh_required=False,
    )

    controller.observe(
        state,
        Action("full", 15),
        accepted_prefix=2,
        history_denominator=3,
    )

    assert controller.history_acceptance == 2 / 3
