from __future__ import annotations

import json

import pytest

from src.analyze.dflash_residual.io import (
    join_selection_trace,
    read_acceptance_records,
    read_selection_jsonl,
    read_trace_jsonl,
    write_metrics_bundle,
)
from src.analyze.dflash_residual.report import render_markdown_report


def _trace(sample_id: str = "s1", draft_position: int = 1) -> dict:
    return {
        "schema_version": "dflash_residual.trace.v1",
        "status": "ok",
        "run_id": "r1",
        "sample_id": sample_id,
        "document_id": sample_id,
        "dataset": "gov_report",
        "context_length": 2048,
        "round_index": 0,
        "draft_position": draft_position,
        "max_depth": 2,
        "target_token_id": 10,
        "candidate_token_ids": [10, 11],
        "dflash_selected_token_id": 10,
        "target_token_source": "verifier_posterior",
    }


def test_read_trace_jsonl_skips_blank_lines_and_keeps_error_rows(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(_trace()) + "\n\n" + json.dumps({"status": "error", "error": "OOM"}) + "\n", encoding="utf-8")
    rows = read_trace_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["sample_id"] == "s1"
    assert rows[1]["status"] == "error"


def test_read_acceptance_records_expands_legacy_acceptance_lengths(tmp_path) -> None:
    path = tmp_path / "official.jsonl"
    path.write_text(
        json.dumps({"id": "s1", "dataset": "gsm8k", "acceptance_lengths": [2, 1]}) + "\n",
        encoding="utf-8",
    )
    rows = read_acceptance_records(path, run_id="official")
    assert [row["accepted_draft_len"] for row in rows] == [1, 0]
    assert rows[0]["round_index"] == 0
    assert rows[0]["run_id"] == "official"


def test_read_acceptance_records_accepts_flat_accepted_draft_rows(tmp_path) -> None:
    path = tmp_path / "flat.jsonl"
    path.write_text(json.dumps({"sample_id": "s1", "accepted_draft_len": 3}) + "\n", encoding="utf-8")
    rows = read_acceptance_records(path, run_id="custom")
    assert rows[0]["accepted_draft_len"] == 3


def test_join_selection_trace_requires_exact_unique_keys() -> None:
    base = [_trace()]
    selected = [{**_trace(), "dflash2_selected_token_id": 11}]
    joined = join_selection_trace(base, selected)
    assert joined[0]["dflash2_selected_token_id"] == 11
    with pytest.raises(ValueError, match="duplicate"):
        join_selection_trace(base, selected + selected)
    with pytest.raises(ValueError, match="missing"):
        join_selection_trace(base, [])


def test_read_selection_jsonl_accepts_minimal_dflash2_rows(tmp_path) -> None:
    path = tmp_path / "selection.jsonl"
    path.write_text(json.dumps({
        "run_id": "r1", "sample_id": "s1", "round_index": 0,
        "draft_position": 1, "selected_token_id": 11,
    }) + "\n", encoding="utf-8")
    rows = read_selection_jsonl(path)
    assert rows[0]["selected_token_id"] == 11


def test_report_marks_missing_selection_unavailable_without_positive_claim() -> None:
    report = render_markdown_report(
        "p3",
        {"status": "unavailable", "reason": "missing_dflash2_selection"},
    )
    assert "UNAVAILABLE" in report
    assert "không đủ" in report
    assert "PASS" not in report


def test_write_metrics_bundle_is_deterministic_and_writes_csv(tmp_path) -> None:
    paths = write_metrics_bundle(
        tmp_path,
        {"status": "ok", "value": 1.0},
        tables={"rows": [{"b": 2, "a": 1}]},
        report="# report\n",
    )
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "rows.csv").read_text(encoding="utf-8").splitlines() == ["a,b", "1,2"]
    assert paths["report"].endswith("report.md")
