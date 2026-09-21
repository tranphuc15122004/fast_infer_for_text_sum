import pytest
import torch

from src.TrainingFree.concentration import (
    head_source_concentration,
    head_source_oracle_metrics,
)


def test_head_source_concentration_reports_quantiles_per_head() -> None:
    attention = torch.tensor(
        [[
            [[0.1, 0.72, 0.09, 0.045, 0.045]],
            [[0.1, 0.225, 0.225, 0.225, 0.225]],
        ]],
        dtype=torch.float32,
    )

    rows = head_source_concentration(
        attention, source_start=1, source_end=5
    )

    assert len(rows) == 2
    assert rows[0]["head"] == 0
    assert rows[0]["source_mass"] == pytest.approx(0.9)
    assert rows[0]["k90"] == 2
    assert rows[0]["k95"] == 3
    assert rows[0]["k99"] == 4
    assert rows[0]["k95_fraction"] == pytest.approx(0.75)
    assert rows[1]["k95"] == 4
    assert rows[1]["k99_fraction"] == pytest.approx(1.0)


def test_head_source_concentration_rejects_invalid_source_span() -> None:
    attention = torch.ones(1, 1, 1, 4)

    with pytest.raises(ValueError, match="source span"):
        head_source_concentration(attention, source_start=2, source_end=2)


def test_head_source_oracle_metrics_reports_mass_and_output_error() -> None:
    attention = torch.tensor(
        [[[[0.1, 0.7, 0.1, 0.05, 0.05]]]], dtype=torch.float32
    )
    values = torch.tensor(
        [[[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]]],
        dtype=torch.float32,
    )

    rows = head_source_oracle_metrics(
        attention,
        values,
        source_start=1,
        source_end=5,
        routed_fractions=(0.5,),
    )

    assert rows[0]["source_mass"] == pytest.approx(0.9)
    assert rows[0]["routed_fraction_0.5"] == pytest.approx(0.5)
    assert rows[0]["routed_missed_mass_0.5"] == pytest.approx(0.1)
    assert rows[0]["routed_output_error_0.5"] >= 0.0
