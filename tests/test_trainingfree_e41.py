from __future__ import annotations

import pytest

from scripts.analyze_trainingfree_e41 import summarize_e41


def _record(sample_id: str, dataset: str, concentrated: bool) -> dict:
    if concentrated:
        rows = [
            {
                "layer": 27,
                "head": 0,
                "source_tokens": 10,
                "source_mass": 0.9,
                "k90": 1,
                "k90_fraction": 0.1,
                "k95": 2,
                "k95_fraction": 0.2,
                "k99": 3,
                "k99_fraction": 0.3,
            }
        ]
    else:
        rows = [
            {
                "layer": 27,
                "head": 0,
                "source_tokens": 10,
                "source_mass": 0.9,
                "k90": 10,
                "k90_fraction": 1.0,
                "k95": 10,
                "k95_fraction": 1.0,
                "k99": 10,
                "k99_fraction": 1.0,
            }
        ]
    return {
        "status": "ok",
        "sample_id": sample_id,
        "dataset": dataset,
        "steps": [{"head_concentration": rows}],
    }


def test_summarize_e41_reports_head_fraction_and_heatmap() -> None:
    result = summarize_e41(
        [_record("a", "gov_report", True), _record("b", "gov_report", False)]
    )

    assert result["status"] == "ok"
    assert result["record_count"] == 2
    assert result["datasets"]["gov_report"]["global_head_fraction"]["0.5"] == pytest.approx(0.5)
    cell = result["datasets"]["gov_report"]["heatmap"][0]
    assert cell["layer"] == 27
    assert cell["head"] == 0
    assert cell["mean_k95_fraction"] == pytest.approx(0.6)


def test_summarize_e41_rejects_traces_without_concentration() -> None:
    result = summarize_e41([{"status": "ok", "dataset": "gov_report", "steps": []}])

    assert result["status"] == "inconclusive"
    assert result["reason"] == "no_head_concentration_rows"
