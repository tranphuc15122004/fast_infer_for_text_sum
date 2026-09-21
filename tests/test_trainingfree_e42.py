from __future__ import annotations

import pytest

from scripts.analyze_trainingfree_e42 import summarize_e42


def test_summarize_e42_selects_diffuse_heads_as_global() -> None:
    rows = []
    for head, k95, missed, error in (
        (0, 0.8, 0.20, 0.10),
        (1, 0.1, 0.05, 0.02),
    ):
        rows.append(
            {
                "layer": 27,
                "head": head,
                "k95_fraction": k95,
                "routed_fraction_0.5": 0.5,
                "routed_missed_mass_0.5": missed,
                "routed_output_error_0.5": error,
            }
        )
    record = {
        "status": "ok",
        "sample_id": "x",
        "dataset": "gov_report",
        "steps": [{"head_oracle_metrics": rows}],
    }

    result = summarize_e42(
        [record], global_fractions=(0.5,), routed_fractions=(0.5,)
    )
    sweep = result["datasets"]["gov_report"]["sweeps"]["global_0.5_routed_0.5"]

    assert sweep["selected_global_heads"] == [[27, 0]]
    assert sweep["exact_expansion_fraction"] == pytest.approx(0.75)
    assert sweep["missed_attention_mass"]["mean"] == pytest.approx(0.025)
    assert sweep["attention_output_error"]["mean"] == pytest.approx(0.01)


def test_summarize_e42_is_inconclusive_without_oracle_rows() -> None:
    result = summarize_e42(
        [{"status": "ok", "dataset": "gov_report", "steps": []}]
    )

    assert result["status"] == "inconclusive"
    assert result["reason"] == "no_head_oracle_rows"
