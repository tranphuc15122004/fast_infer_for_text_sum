from __future__ import annotations

import pytest

from src.analyze.dflash_residual.alignment import compare_acceptance
from src.analyze.dflash_residual.p1_task_regime import analyze_task_regimes
from src.analyze.dflash_residual.p2_coverage import analyze_coverage
from src.analyze.dflash_residual.p3_headroom import analyze_headroom
from src.analyze.dflash_residual.p4_interaction import analyze_interaction
from src.analyze.dflash_residual.schema import normalize_trace_row


def _row(
    sample_id: str,
    document_id: str,
    position: int,
    *,
    context_length: int = 1024,
    target: int = 10,
    candidates: list[int] | None = None,
    accepted: int | None = None,
) -> dict:
    return normalize_trace_row({
        "status": "ok",
        "run_id": "r",
        "sample_id": sample_id,
        "document_id": document_id,
        "dataset": "cnn_dailymail",
        "context_length": context_length,
        "round_index": 0,
        "draft_position": position,
        "max_depth": 2,
        "target_token_id": target,
        "candidate_token_ids": candidates or [target, 99],
        "dflash_selected_token_id": target,
        "accepted_draft_len": accepted,
        "target_token_source": "verifier_posterior",
    })


def test_p0_passes_only_when_both_runners_have_positive_and_close_acceptance() -> None:
    official = [{"sample_id": "s1", "round_index": i, "accepted_draft_len": 1} for i in range(5)]
    custom = [{"sample_id": "s1", "round_index": i, "accepted_draft_len": 1} for i in range(5)]
    result = compare_acceptance(official, custom, min_blocks=5)
    assert result["status"] == "PASS"
    assert result["mat_official"] == pytest.approx(1.0)
    assert result["mat_custom"] == pytest.approx(1.0)


def test_p0_rejects_runner_mismatch_after_positive_acceptance() -> None:
    official = [{"sample_id": "s1", "round_index": i, "accepted_draft_len": 2} for i in range(5)]
    custom = [{"sample_id": "s1", "round_index": i, "accepted_draft_len": 0} for i in range(5)]
    result = compare_acceptance(official, custom, min_blocks=5, mat_tolerance=0.1)
    assert result["status"] == "FAIL"
    assert result["reason"] == "non_positive_acceptance"


def test_p0_rejects_explicit_protocol_mismatch_before_scientific_comparison() -> None:
    rows = [{"sample_id": "s1", "round_index": i, "accepted_draft_len": 1} for i in range(5)]
    result = compare_acceptance(
        rows,
        rows,
        min_blocks=5,
        official_metadata={"block_size": 16, "thinking_mode": False},
        custom_metadata={"block_size": 32, "thinking_mode": False},
    )
    assert result["status"] == "FAIL"
    assert result["reason"] == "protocol_mismatch"
    assert result["protocol_check"]["mismatches"]["block_size"]["official"] == 16


def test_p1_groups_workloads_and_p2_builds_context_depth_coverage_table() -> None:
    rows = [
        _row("s1", "d1", 1, context_length=1024, candidates=[10, 11], accepted=2),
        _row("s1", "d1", 2, context_length=1024, target=12, candidates=[99, 12], accepted=2),
        _row("s2", "d2", 1, context_length=9000, candidates=[99, 10], accepted=0),
    ]
    p1 = analyze_task_regimes(rows)
    assert p1["status"] == "ok"
    assert p1["regimes"]["cnn_dm"]["blocks"] == 2
    p2 = analyze_coverage(rows, recall_k=16)
    assert p2["status"] == "ok"
    assert p2["rows"] == 3
    assert p2["table"][0]["draft_position"] == 1
    assert {item["context_bin"] for item in p2["table"]} == {"0-2k", "8-16k"}


def test_p2_gate_uses_context_bins_for_document_counts() -> None:
    rows = []
    for index, context_length in enumerate((900, 1000, 1100, 1200, 1300)):
        rows.append(_row(
            f"short-{index}", f"short-{index}", 1,
            context_length=context_length,
            candidates=[10, 99],
        ))
    for index, context_length in enumerate((2500, 2600, 2700, 2800, 2900)):
        rows.append(_row(
            f"long-{index}", f"long-{index}", 1,
            context_length=context_length,
            candidates=[99, 98],
        ))

    result = analyze_coverage(rows, recall_k=16, min_documents=5)

    assert result["h2_gate"]["decision"] == "PASS"
    assert result["h2_gate"]["short_documents"] == 5
    assert result["h2_gate"]["long_documents"] == 5


def test_p2_is_unavailable_without_candidate_rows() -> None:
    result = analyze_coverage([], recall_k=16)
    assert result == {"status": "unavailable", "reason": "no_valid_trace_rows", "rows": 0}


def test_p3_reports_context_headroom_and_p4_rejects_sparse_design() -> None:
    rows = [
        _row("s1", "d1", 1, context_length=1024, candidates=[1, 10]),
        _row("s2", "d2", 1, context_length=9000, candidates=[99, 10]),
    ]
    for row, selected in zip(rows, (10, 99)):
        row["dflash2_selected_token_id"] = selected
    p3 = analyze_headroom(rows, oracle_k=16, min_blocks=2)
    assert p3["status"] == "ok"
    assert set(p3["by_context_bin"]) == {"0-2k", "8-16k"}
    p4 = analyze_interaction(rows, bootstrap_samples=10, min_documents=5)
    assert p4["status"] == "inconclusive"
    assert p4["reason"] == "insufficient_documents"
