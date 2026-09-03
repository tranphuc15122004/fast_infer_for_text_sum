from src.analyze.groundsync.p1_strong_drafter import (
    acceptance_rows_from_generations,
    analyze_acceptance_rows,
    summarize_e2e,
)


def test_eagle_acceptance_length_removes_target_fallback():
    rows = acceptance_rows_from_generations(
        [{"document_id": "d0", "new_tokens": 5, "acceptance_lengths": [3, 2]}],
        max_k=4,
    )
    assert [row["accepted_len"] for row in rows] == [2, 1]
    assert [row["start_position"] for row in rows] == [0, 1]


def test_eagle_analysis_has_within_and_persistence_outputs():
    generations = [
        {"document_id": f"d{i}", "new_tokens": 8, "acceptance_lengths": [3, 3, 1, 2]}
        for i in range(8)
    ]
    result = analyze_acceptance_rows(generations)
    assert result["status"] == "ok"
    assert result["coverage"]["round_count"] == 32
    assert "h1_to_later_hazard_ratio" in result["within_block_burstiness"]
    assert "1" in result["across_round_persistence"]["by_delta"]


def test_e2e_summary_requires_paired_timing():
    assert summarize_e2e([{"status": "ok", "new_tokens": 4, "eagle_time_s": 1.0}])["status"] == "UNAVAILABLE"
    result = summarize_e2e([{
        "status": "ok", "new_tokens": 4, "eagle_time_s": 1.0,
        "naive_time_s": 2.0, "eagle_tokens_per_s": 4.0,
        "naive_tokens_per_s": 2.0, "exact_match_to_naive": True,
    }])
    assert result["aggregate_speedup"] == 2.0
    assert result["exact_match_rate"] == 1.0
