from __future__ import annotations

import pytest

from src.analyze.dflash_residual.metrics import (
    fit_context_depth_interaction,
    oracle_prefix_length,
    prefix_match_length,
    recall_at_k,
    summarize_headroom,
    summarize_p1,
    survival,
)
from src.analyze.dflash_residual.schema import normalize_trace_row, validate_trace_row


def _row(
    *,
    sample_id: str = "s1",
    document_id: str = "d1",
    round_index: int = 0,
    draft_position: int = 1,
    target: int = 10,
    candidates: list[int] | None = None,
    dflash2: int | None = None,
    context_length: int = 1024,
    accepted: int | None = None,
) -> dict:
    return normalize_trace_row(
        {
            "schema_version": "dflash_residual.trace.v1",
            "status": "ok",
            "run_id": "r1",
            "sample_id": sample_id,
            "document_id": document_id,
            "dataset": "cnn_dailymail",
            "task_regime": "cnn_dm",
            "context_length": context_length,
            "round_index": round_index,
            "draft_position": draft_position,
            "max_depth": 2,
            "target_token_id": target,
            "candidate_token_ids": candidates or [target, 99, 98, 97],
            "dflash_selected_token_id": candidates[0] if candidates else target,
            "dflash2_selected_token_id": dflash2,
            "accepted_draft_len": accepted,
            "block_size": 3,
            "native_block_size": 3,
            "target_token_source": "verifier_posterior",
        }
    )


def test_schema_reports_missing_fields_and_normalizes_regime() -> None:
    problems = validate_trace_row({"status": "ok"})
    assert "sample_id" in " ".join(problems)
    row = normalize_trace_row(
        {
            "status": "ok",
            "run_id": "r",
            "sample_id": 7,
            "document_id": 7,
            "dataset": "gov_report",
            "context_length": 4096,
            "round_index": 0,
            "draft_position": 1,
            "max_depth": 1,
            "target_token_id": 4,
            "candidate_token_ids": [4, 5],
            "dflash_selected_token_id": 4,
            "target_token_source": "verifier_posterior",
        }
    )
    assert row["sample_id"] == "7"
    assert row["task_regime"] == "govreport"
    assert row["context_bin"] == "4-8k"


def test_recall_at_k_does_not_inject_target_outside_candidate_set() -> None:
    rows = [_row(target=10, candidates=[11, 12, 13, 14])]
    assert recall_at_k(rows, 1) == 0.0
    assert recall_at_k(rows, 4) == 0.0
    assert recall_at_k(rows, 16) == 0.0


def test_prefix_and_oracle_lengths_are_longest_consecutive_prefixes() -> None:
    assert prefix_match_length([1, 2, 7], [1, 2, 3]) == 2
    rows = [
        _row(round_index=0, draft_position=1, target=10, candidates=[10, 1]),
        _row(round_index=0, draft_position=2, target=11, candidates=[11, 2]),
        _row(round_index=0, draft_position=3, target=12, candidates=[99, 98]),
    ]
    assert oracle_prefix_length(rows, 2) == 2
    assert oracle_prefix_length(rows, 1) == 2


def test_p1_reports_mat_survival_and_recall_by_regime() -> None:
    rows = [
        _row(round_index=0, draft_position=1, target=10, accepted=1),
        _row(round_index=0, draft_position=2, target=11, candidates=[99, 11], accepted=1),
        _row(sample_id="s2", document_id="d2", round_index=0, draft_position=1, target=10, accepted=0),
    ]
    result = summarize_p1(rows)
    assert result["regimes"]["cnn_dm"]["blocks"] == 2
    assert result["regimes"]["cnn_dm"]["mat"] == pytest.approx(0.5)
    assert result["regimes"]["cnn_dm"]["survival"]["1"] == pytest.approx(0.5)
    assert result["regimes"]["cnn_dm"]["recall"]["1"] == pytest.approx(2.0 / 3.0)


def test_p1_keeps_same_round_separate_across_context_conditions() -> None:
    rows = [
        _row(context_length=1024, accepted=1),
        _row(context_length=8192, accepted=0),
    ]

    result = summarize_p1(rows)

    assert result["regimes"]["cnn_dm"]["blocks"] == 2
    assert result["regimes"]["cnn_dm"]["mat"] == pytest.approx(0.5)


def test_headroom_separates_candidate_miss_and_selection_error() -> None:
    rows = [
        _row(round_index=0, draft_position=1, target=10, candidates=[1, 10], dflash2=10),
        _row(round_index=1, draft_position=1, target=11, candidates=[11, 2], dflash2=2),
        _row(sample_id="s2", document_id="d2", round_index=0, draft_position=1, target=10, candidates=[99, 98], dflash2=99),
    ]
    result = summarize_headroom(rows, oracle_k=2)
    assert result["status"] == "ok"
    assert result["candidate_miss_rows"] == 1
    assert result["selection_error_rows"] == 1
    assert result["mat_d"] == pytest.approx(1.0 / 3.0)
    assert result["mat_d2"] == pytest.approx(1.0 / 3.0)
    assert result["mat_o16"] == pytest.approx(2.0 / 3.0)
    assert result["rho_d2"] == pytest.approx(0.0)


def test_headroom_is_unavailable_without_dflash2_selection() -> None:
    result = summarize_headroom([_row()], oracle_k=16)
    assert result["status"] == "unavailable"
    assert result["reason"] == "missing_dflash2_selection"


def test_survival_is_empty_safe_and_interaction_uses_document_bootstrap() -> None:
    assert survival([]) == {}
    rows = []
    for document_id, context_length, hit in (("d1", 1000, 1), ("d2", 8000, 0), ("d3", 16000, 0)):
        for position in (1, 2):
            rows.append(
                _row(
                    document_id=document_id,
                    context_length=context_length,
                    draft_position=position,
                    target=10,
                    candidates=[10, 1] if hit else [99, 98],
                )
            )
    result = fit_context_depth_interaction(rows, bootstrap_samples=30, seed=4, min_documents=3)
    assert result["status"] == "ok"
    assert result["documents"] == 3
    assert "beta_log_context_x_position" in result
    assert len(result["bootstrap_ci"]) == 2
