from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_metric_audit_reports_missing_full_e2e_fields_and_quality():
    from common.metric_audit import audit_record

    audit = audit_record(
        {
            "sample_id": "s1",
            "status": "success",
            "measurement_scope": "full_e2e",
            "input_tokens": 100,
            "output_tokens": 10,
            "decode_ms": 20.0,
            "e2e_ms": None,
            "text": None,
            "reference_output": "reference",
            "max_new_tokens": 32,
        }
    )

    assert audit["timing"]["status"] == "partial"
    assert "e2e_ms" in audit["timing"]["missing"]
    assert audit["quality"]["status"] == "missing_text"
    assert "missing_e2e_ms" in audit["issues"]
    assert "missing_text" in audit["issues"]
    assert audit["tokens"]["budget_valid"] is True


def test_metric_audit_summary_counts_fields_scopes_and_issues():
    from common.metric_audit import summarize_audits

    audits = [
        {
            "sample_id": "s1",
            "status": "success",
            "measurement_scope": "decode_only",
            "timing": {"present": ["decode_ms"], "missing": []},
            "quality": {"status": "not_available"},
            "issues": [],
        },
        {
            "sample_id": "s2",
            "status": "failed",
            "measurement_scope": "full_e2e",
            "timing": {"present": [], "missing": ["e2e_ms"]},
            "quality": {"status": "missing_text"},
            "issues": ["failed_record", "missing_e2e_ms"],
        },
    ]

    summary = summarize_audits(audits)

    assert summary["num_records"] == 2
    assert summary["status_counts"] == {"success": 1, "failed": 1}
    assert summary["scope_counts"] == {"decode_only": 1, "full_e2e": 1}
    assert summary["issue_counts"]["missing_e2e_ms"] == 1
    assert summary["quality_status_counts"]["missing_text"] == 1


def test_metric_audit_file_is_machine_readable(tmp_path):
    from common.metric_audit import audit_output_file

    output = tmp_path / "toy.jsonl"
    output.write_text(
        json.dumps(
            {
                "sample_id": "s1",
                "status": "success",
                "measurement_scope": "e2e_only",
                "input_tokens": 5,
                "output_tokens": 2,
                "e2e_ms": 10.0,
                "throughput_tok_s": 200.0,
                "text": "answer",
                "reference_output": "answer",
                "rouge1": 1.0,
            }
        )
        + "\n"
        + json.dumps({"type": "summary", "status": "success"})
        + "\n",
        encoding="utf-8",
    )
    audit_path = tmp_path / "toy.metrics.json"

    summary = audit_output_file(
        output,
        baseline="toy",
        dataset="toy",
        audit_path=audit_path,
        expected_output_tokens=2,
    )

    assert summary["num_records"] == 1
    assert audit_path.is_file()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["baseline"] == "toy"
    assert payload["summary"]["quality_records"] == 1
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["metric_audit"]["quality"]["status"] == "complete"
    assert rows[1]["metric_audit_summary"]["num_records"] == 1
