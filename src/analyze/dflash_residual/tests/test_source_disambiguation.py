from __future__ import annotations

from collections import Counter

import pytest

from src.analyze.dflash_residual.source_disambiguation import (
    analyze_source_ladder,
    analyze_target_near_ties,
    annotate_source_rows,
    build_source_index,
    select_diagnostic_candidate,
)


def _row(position: int, target: int, candidates: list[int], logits: list[float], *, source: bool) -> dict:
    return {
        "status": "ok",
        "run_id": "r1",
        "sample_id": "s1",
        "document_id": "d1",
        "dataset": "cnn_dailymail",
        "task_regime": "cnn_dm",
        "context_length": 1024,
        "context_bin": "0-2k",
        "round_index": 0,
        "draft_position": position,
        "max_depth": 3,
        "target_token_id": target,
        "target_token_source": "verifier_posterior",
        "candidate_token_ids": candidates,
        "candidate_logits": logits,
        "dflash_selected_token_id": candidates[0],
        "accepted_draft_len": 0,
        "block_size": 4,
        "native_block_size": 4,
        "_source_expected": source,
    }


def test_build_and_annotate_source_index_marks_exact_token_support() -> None:
    index = build_source_index(
        [{"id": "s1", "dataset": "cnn_dailymail", "document": "unused"}],
        lambda text: [10, 10, 20],
    )
    rows = [_row(1, 10, [11, 10], [2.0, 1.0], source=True), _row(2, 30, [30, 10], [2.0, 1.0], source=False)]
    annotated = annotate_source_rows(rows, index)
    assert annotated[0]["source_token_present"] is True
    assert annotated[0]["source_token_frequency"] == 2
    assert annotated[0]["source_stratum"] == "copyable"
    assert annotated[1]["source_token_present"] is False
    assert annotated[1]["source_stratum"] == "source_novel"
    assert annotated[0]["candidate_source_frequencies"] == [0, 2]


def test_source_selector_never_selects_outside_recorded_candidates() -> None:
    row = _row(1, 20, [10, 20, 30], [3.0, 2.0, 1.0], source=False)
    row.update({"candidate_source_frequencies": [0, 4, 1], "candidate_source_present": [False, True, True]})
    selected = select_diagnostic_candidate(row, mode="u_plus_source", source_weight=1.0)
    assert selected in row["candidate_token_ids"]
    assert selected == 20


def test_semantic_selector_lambda_zero_reproduces_dflash_choice() -> None:
    row = _row(1, 20, [10, 20, 30], [3.0, 2.0, 1.0], source=False)
    row["candidate_source_semantic_scores"] = [0.0, 1.0, 0.2]
    assert select_diagnostic_candidate(
        row,
        mode="u_plus_source_semantic",
        source_weight=0.0,
    ) == row["dflash_selected_token_id"]


def test_source_phrase_annotation_records_ngram_support() -> None:
    from src.analyze.dflash_residual.source_disambiguation import annotate_source_phrase_rows

    rows = [
        _row(1, 10, [10, 11], [2.0, 1.0], source=True),
        _row(2, 11, [11, 12], [2.0, 1.0], source=True),
    ]
    index = {"s1": {"ngram_counts": {2: Counter({(10, 11): 2}), 3: Counter()}}}
    annotated = annotate_source_phrase_rows(rows, index)
    assert annotated[1]["candidate_source_phrase_scores"][0] > 0.0


def test_source_ladder_reports_oracle_and_recovery() -> None:
    rows = [
        _row(1, 20, [10, 20, 30], [3.0, 2.0, 1.0], source=True),
        _row(2, 21, [21, 11, 31], [3.0, 2.0, 1.0], source=False),
        _row(3, 22, [12, 22, 32], [3.0, 2.0, 1.0], source=True),
    ]
    index = {"s1": {"token_counts": Counter({10: 1, 20: 1, 21: 1, 22: 1}), "document_count": 4}}
    annotated = annotate_source_rows(rows, index)
    result = analyze_source_ladder(annotated, lambda_values=(0.0, 1.0))
    assert result["status"] == "ok"
    assert result["datasets"]["cnn_dm"]["lambda_results"]["1.0"]["mat_selected"] >= 0.0
    assert result["datasets"]["cnn_dm"]["mat_oracle"] >= result["datasets"]["cnn_dm"]["mat_d"]


def test_target_near_tie_compares_target_against_dflash_selected_token() -> None:
    row = _row(1, 20, [10, 20, 30], [3.0, 2.0, 1.0], source=False)
    row.update({
        "target_candidate_logits": [10.0, 10.4, 2.0],
        "source_stratum": "source_novel",
    })
    result = analyze_target_near_ties([row], near_tie_margin=0.5)
    dataset = result["datasets"]["cnn_dm"]
    assert dataset["mean_target_margin"] == pytest.approx(0.4)
    assert dataset["mismatch_rows"] == 1
    assert dataset["mismatch_near_tie_rate"] == 1.0
