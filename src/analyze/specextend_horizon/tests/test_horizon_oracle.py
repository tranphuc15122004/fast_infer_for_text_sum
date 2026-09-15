from analyze.specextend_horizon.horizon_oracle import (
    acceptance_curve,
    analyze_records,
    set_overlap,
    weighted_top_ids,
)


def test_weighted_top_ids_is_deterministic():
    assert weighted_top_ids({"b": 0.9, "a": 0.9, "c": 0.1}, 2) == ["a", "b"]


def test_set_overlap_reports_recall_and_precision():
    result = set_overlap([1, 2, 3], [2, 3, 4])
    assert result["intersection_count"] == 2
    assert result["recall_current_in_horizon"] == 2 / 3
    assert result["precision_current_vs_horizon"] == 2 / 3


def test_acceptance_curve_uses_accept_length_plus_target_token():
    rows = [
        {"type": "cycle", "dataset": "gov", "accept_length": 2},
        {"type": "cycle", "dataset": "gov", "accept_length": 0},
    ]
    result = acceptance_curve(rows)["gov"]
    assert result["mean_accepted_tokens_per_cycle"] == 2
    assert result["survival"]["2"] == 0.5


def test_analysis_exposes_projected_only_status():
    rows = [
        {
            "type": "cycle",
            "dataset": "gov",
            "cycle": 1,
            "accepted_tokens": 3,
            "current_chunk_ids": [1, 2],
            "horizon_chunk_ids": [2, 3],
            "current_context_tokens": 64,
            "horizon_context_tokens": 64,
            "cycle_time_s": 1.0,
        }
    ]
    result = analyze_records(rows)
    assert result["overlap"]["mean_recall_current_in_horizon"] == 0.5
    assert result["projected_cost"]["interpretation"].startswith("projected only")
