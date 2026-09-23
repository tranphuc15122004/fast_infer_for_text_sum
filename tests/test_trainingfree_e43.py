from __future__ import annotations

from scripts.analyze_trainingfree_e43 import summarize_e43


def _record(sample_id: str, dataset: str, cost: float, missed: float) -> dict:
    return {
        "status": "ok",
        "sample_id": sample_id,
        "dataset": dataset,
        "steps": [
            {
                "temporal": {
                    "gqa_oracle": {
                        "B0.2": {
                            "gqa_expansion_fraction": 0.3,
                            "mean_missed_mass": missed,
                            "p99_missed_mass": missed,
                        }
                    },
                    "lag": {
                        "L1_B0.2": {
                            "gqa_expansion_fraction": 0.3,
                            "mean_missed_mass": missed,
                            "p99_missed_mass": missed,
                        }
                    },
                    "recurrent": {
                        "R4_B32_A1": {
                            "refresh": False,
                            "source_cost_ratio": cost,
                            "working_source_cost_ratio": 0.2,
                            "mean_missed_mass": missed,
                            "p99_missed_mass": missed,
                        }
                    },
                }
            }
        ],
    }


def test_summarize_e43_promotes_candidate_only_when_all_guardrails_pass() -> None:
    records = [
        _record("g1", "gov_report", 0.4, 0.005),
        _record("g2", "gov_report", 0.4, 0.005),
        _record("m1", "multi_news", 0.4, 0.005),
        _record("m2", "multi_news", 0.4, 0.005),
    ]

    result = summarize_e43(records, min_documents_per_dataset=2)

    assert result["status"] == "GO_MASKED_GENERATION"
    assert result["best_candidate"] == "R4_B32_A1"
    assert result["recurrent_candidates"]["R4_B32_A1"]["gate"] == {
        "cost_lt_50pct": True,
        "mean_missed_mass_le_1pct": True,
        "p99_missed_mass_le_5pct": True,
    }


def test_summarize_e43_stops_when_candidate_misses_tail_guardrail() -> None:
    records = [
        _record("g1", "gov_report", 0.4, 0.005),
        _record("g2", "gov_report", 0.4, 0.060),
        _record("m1", "multi_news", 0.4, 0.005),
        _record("m2", "multi_news", 0.4, 0.005),
    ]

    result = summarize_e43(records, min_documents_per_dataset=2)

    assert result["status"] == "STOP_SOURCE_SELECTION"
    assert result["best_candidate"] is None


def test_summarize_e43_is_inconclusive_without_temporal_rows() -> None:
    result = summarize_e43([
        {"status": "ok", "sample_id": "x", "dataset": "gov_report", "steps": []}
    ])

    assert result["status"] == "INCONCLUSIVE"
