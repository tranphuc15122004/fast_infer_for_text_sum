import json
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_longbench_200 import (  # noqa: E402
    _bundle_longbench_records,
    _materialize_bundle_outputs,
    _split_bundle_sample_rows,
)


def test_bundle_preserves_dataset_order_and_builds_unique_sample_mapping():
    payloads = [
        (
            "gov_report",
            [{"id": "gov-0", "dataset": "gov_report"}],
            [{"id": "gov-0", "prompt": "g"}],
        ),
        (
            "qmsum",
            [{"id": "q-0", "dataset": "qmsum"}],
            [{"id": "q-0", "prompt": "q"}],
        ),
    ]

    source_rows, normalized_rows, sample_to_dataset = _bundle_longbench_records(
        payloads
    )

    assert [row["id"] for row in source_rows] == ["gov-0", "q-0"]
    assert [row["id"] for row in normalized_rows] == ["gov-0", "q-0"]
    assert sample_to_dataset == {"gov-0": "gov_report", "q-0": "qmsum"}


def test_bundle_rejects_duplicate_sample_ids_across_datasets():
    payloads = [
        ("gov_report", [{"id": "same"}], [{"id": "same"}]),
        ("qmsum", [{"id": "same"}], [{"id": "same"}]),
    ]

    with pytest.raises(ValueError, match="duplicate sample_id"):
        _bundle_longbench_records(payloads)


def test_split_bundle_rows_restores_per_dataset_membership_and_rejects_unknown_ids():
    rows = [
        {"sample_id": "q-0", "dataset": "bundle", "scope": "sample"},
        {"sample_id": "gov-0", "dataset": "bundle", "scope": "sample"},
        {"type": "summary", "dataset": "bundle"},
    ]

    split = _split_bundle_sample_rows(
        rows,
        sample_to_dataset={"gov-0": "gov_report", "q-0": "qmsum"},
    )

    assert [row["sample_id"] for row in split["gov_report"]] == ["gov-0"]
    assert split["gov_report"][0]["dataset"] == "gov_report"
    assert [row["sample_id"] for row in split["qmsum"]] == ["q-0"]

    with pytest.raises(ValueError, match="unknown sample_id"):
        _split_bundle_sample_rows(
            [{"sample_id": "missing", "scope": "sample"}],
            sample_to_dataset={"gov-0": "gov_report"},
        )


def test_materialize_bundle_outputs_writes_collector_compatible_cells(tmp_path):
    bundle = tmp_path / "vanilla_hf" / "_bundle.jsonl"
    bundle.parent.mkdir()
    bundle.write_text(
        "\n".join(
            [
                '{"sample_id":"gov-0","dataset":"bundle","scope":"sample","status":"success","e2e_ms":10}',
                '{"sample_id":"q-0","dataset":"bundle","scope":"sample","status":"success","e2e_ms":20}',
                '{"type":"summary","dataset":"bundle","num_samples":2}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths = {
        "gov_report": tmp_path / "vanilla_hf" / "gov_report.jsonl",
        "qmsum": tmp_path / "vanilla_hf" / "qmsum.jsonl",
    }

    counts = _materialize_bundle_outputs(
        bundle,
        output_paths=paths,
        sample_to_dataset={"gov-0": "gov_report", "q-0": "qmsum"},
        sample_order={"gov-0": 0, "q-0": 1},
        baseline="vanilla_hf",
        run_id="test-run",
        data_parallel=True,
        processes_per_gpu=6,
    )

    assert counts == {"gov_report": 1, "qmsum": 1}
    gov_rows = [
        json.loads(line)
        for line in paths["gov_report"].read_text(encoding="utf-8").splitlines()
    ]
    assert gov_rows[0]["dataset"] == "gov_report"
    assert gov_rows[-1]["type"] == "summary"
    assert gov_rows[-1]["processes_per_gpu"] == 6
