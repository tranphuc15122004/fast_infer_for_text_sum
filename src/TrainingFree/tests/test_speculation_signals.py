from __future__ import annotations

import math

import pytest
import torch

from TrainingFree.speculation_experiments.signals import summarize_draft_logits


def test_uniform_logits_have_max_entropy_and_low_confidence() -> None:
    values = summarize_draft_logits(torch.zeros((1, 2, 4), dtype=torch.float16))

    assert values["normalized_entropy"] == pytest.approx([1.0, 1.0])
    assert values["confidence"] == pytest.approx([0.25, 0.25])
    assert values["probability_margin"] == pytest.approx([0.0, 0.0])
    assert values["logit_margin"] == pytest.approx([0.0, 0.0])
    assert values["raw_entropy"] == pytest.approx([math.log(4), math.log(4)])


def test_peaked_logits_have_low_entropy_and_expected_top_two_margins() -> None:
    logits = torch.tensor([[[12.0, 0.0, -1.0], [0.0, 0.0, 0.0]]])

    values = summarize_draft_logits(logits)

    assert values["normalized_entropy"][0] < 0.01
    assert values["confidence"][0] > 0.999
    assert values["probability_margin"][0] > 0.999
    assert values["logit_margin"][0] == pytest.approx(12.0)
    assert values["normalized_entropy"][1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "logits",
    [torch.empty((1, 0, 4)), torch.tensor([[[float("nan"), 0.0]]])],
)
def test_invalid_draft_logits_are_rejected(logits: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="logits"):
        summarize_draft_logits(logits)
