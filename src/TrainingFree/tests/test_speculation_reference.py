from __future__ import annotations

from TrainingFree.speculation_experiments.reference import reference_history, select_reference_prefixes


def test_prefix_selection_covers_beginning_middle_end_and_respects_regular_horizon() -> None:
    output = list(range(200))
    rounds = [
        {"type": "round", "round_index": index, "gamma_executed": 15, "processed_output_before": index * 4,
         "accepted_candidates": 3}
        for index in range(40)
    ]

    selected = select_reference_prefixes(
        rounds,
        output,
        prompt_length=100,
        max_new_tokens=200,
        eos_ids=set(),
        max_prefixes=15,
    )

    assert 1 <= len(selected) <= 15
    assert [row.round_index for row in selected] == sorted(row.round_index for row in selected)
    assert all(row.remaining_output_tokens >= 16 for row in selected)
    thirds = [sum(1 for row in selected if row.processed_output_tokens < 67),
              sum(1 for row in selected if 67 <= row.processed_output_tokens < 134),
              sum(1 for row in selected if row.processed_output_tokens >= 134)]
    assert all(count >= 1 for count in thirds)


def test_reference_history_only_uses_prior_rounds_and_marks_missing_first() -> None:
    rounds = [
        {"type": "round", "round_index": 0, "accepted_candidates": 4, "gamma_executed": 15},
        {"type": "round", "round_index": 1, "accepted_candidates": 10, "gamma_executed": 15},
        {"type": "round", "round_index": 2, "accepted_candidates": 1, "gamma_executed": 15},
    ]

    first = reference_history(rounds, current_round=0)
    third = reference_history(rounds, current_round=2)

    assert first["missing"] is True
    assert third["previous_accepted_candidates"] == 10
    assert third["ema_acceptance"] == 0.1 * (10 / 15) + 0.9 * (4 / 15)
