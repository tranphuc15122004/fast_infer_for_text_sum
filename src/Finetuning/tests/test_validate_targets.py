from __future__ import annotations

import json

import pytest

from Finetuning.validate_targets import validate_teacher_jsonl


def test_validation_writes_quality_report_and_keeps_clean_trajectory(tmp_path) -> None:
    source = tmp_path / "teacher.jsonl"
    report_path = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "id": "vi-1",
                "document": "Chính phủ công bố kế hoạch hỗ trợ doanh nghiệp đổi mới công nghệ trong năm nay.",
                "summary": "Chính phủ công bố kế hoạch hỗ trợ doanh nghiệp đổi mới công nghệ.",
                "reference_summary": "Kế hoạch hỗ trợ doanh nghiệp đổi mới công nghệ được công bố.",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_teacher_jsonl(
        source,
        report_path=report_path,
        max_anomaly_rate=0.0,
        min_rouge1=0.1,
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.passed_gate is True
    assert payload["total_records"] == 1
    assert payload["anomalous_records"] == 0


def test_validation_gate_rejects_empty_or_degenerated_teacher_output(tmp_path) -> None:
    source = tmp_path / "bad_teacher.jsonl"
    source.write_text(
        json.dumps(
            {"id": "bad", "document": "Nguồn văn bản đủ dài để đánh giá.", "summary": ""},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="validation failed gate"):
        validate_teacher_jsonl(source, max_anomaly_rate=0.0)
