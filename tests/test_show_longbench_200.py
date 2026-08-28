import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.benchmark_data import DATASETS  # noqa: E402
from show_longbench_200 import format_record, render_report  # noqa: E402


def test_format_record_shows_the_fields_a_human_needs_to_understand_a_sample():
    row = {
        "id": "lcc_demo",
        "task_type": "code_completion",
        "context": "class Example:\n    return value\n" * 20,
        "input": "next line",
        "reference_output": "    return result",
        "input_tokens": 1234,
        "length_bin": 3,
    }

    rendered = format_record(row, sample_number=1, context_chars=40, field_chars=30)

    assert "Sample 1" in rendered
    assert "id=lcc_demo" in rendered
    assert "task_type=code_completion" in rendered
    assert "context (" in rendered
    assert "class Example:" in rendered
    assert "input (9 chars): next line" in rendered
    assert "reference_output (17 chars):     return result" in rendered
    assert "input_tokens=1234" in rendered
    assert "length_bin=3" in rendered
    assert "..." in rendered


def test_render_report_summarizes_selected_datasets_and_limits_samples(tmp_path):
    for dataset in DATASETS:
        rows = [
            {
                "id": f"{dataset}_{index}",
                "task_type": "code_completion"
                if dataset in {"lcc", "repobench-p"}
                else "summarization",
                "context": f"{dataset} context {index}",
                "input": f"query {index}",
                "reference_output": f"reference {index}",
                "input_tokens": 100 + index,
                "length_bin": index if dataset in {"lcc", "repobench-p"} else None,
            }
            for index in range(2)
        ]
        (tmp_path / f"{dataset}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    rendered = render_report(
        tmp_path,
        datasets=["gov_report", "lcc"],
        samples=1,
        context_chars=80,
        field_chars=40,
    )

    assert "LongBench canonical dataset preview" in rendered
    assert "gov_report" in rendered
    assert "lcc" in rendered
    assert "total displayed datasets: 2" in rendered
    assert "gov_report_0" in rendered
    assert "gov_report_1" not in rendered
    assert "lcc_0" in rendered
    assert "qmsum" not in rendered
