import json
import hashlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common.benchmark_data import (  # noqa: E402
    REQUIRED_FIELDS,
    canonicalize_record,
    metric_family,
    render_prompt,
    stratified_sample,
    validate_output_dir,
)
from common.data_loader import normalize  # noqa: E402
from common import metrics  # noqa: E402
import collect_metrics  # noqa: E402


def fake_tokenizer(text, add_special_tokens=False):
    del add_special_tokens
    return {"input_ids": text.split()}


def test_canonicalize_record_has_shared_schema_and_stable_id():
    source = {"context": "class A:\n    pass", "input": "", "answers": ["\n"]}
    first = canonicalize_record("lcc", 7, source, fake_tokenizer)
    second = canonicalize_record("lcc", 7, source, fake_tokenizer)

    assert REQUIRED_FIELDS <= set(first)
    assert first == second
    assert first["task_type"] == "code_completion"
    assert first["reference_output"] == "\n"
    assert first["input_tokens"] == len(render_prompt(first).split())


def test_stratified_sample_is_reproducible_and_balances_five_bins():
    rows = [
        {"id": str(i), "input_tokens": i, "source_index": i}
        for i in range(500)
    ]
    selected = stratified_sample(rows, n=200, n_bins=5, seed=42)

    bins = [row["length_bin"] for row in selected]
    assert bins.count(0) == 40
    assert bins.count(4) == 40
    assert [row["id"] for row in selected] == [
        row["id"] for row in stratified_sample(rows, n=200, n_bins=5, seed=42)
    ]


def test_validate_output_dir_rejects_duplicate_ids(tmp_path):
    output = tmp_path / "longbench_200"
    output.mkdir()
    row = canonicalize_record(
        "gov_report",
        0,
        {"context": "report", "input": "", "answers": ["summary"]},
        fake_tokenizer,
    )
    (output / "gov_report.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8"
    )
    checksum = hashlib.sha256(
        (output / "gov_report.jsonl").read_bytes()
    ).hexdigest()
    (output / "manifest.json").write_text(
        json.dumps({"file_sha256": {"gov_report": checksum}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_output_dir(output, expected_count=2)


def test_builder_reads_local_jsonl_and_writes_manifest(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "output"
    source_dir.mkdir()
    row = {"context": "report text", "input": "", "answers": ["reference"]}
    (source_dir / "gov_report.jsonl").write_text(
        "\n".join(json.dumps(row) for _ in range(2)) + "\n", encoding="utf-8"
    )

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_longbench_200 import build_one_dataset

    build_one_dataset(
        "gov_report",
        source_dir,
        output_dir,
        fake_tokenizer,
        2,
        42,
    )
    assert len((output_dir / "gov_report.jsonl").read_text().splitlines()) == 2
    assert (output_dir / "manifest.json").exists()


def _write_valid_output_dir(output: Path, count: int = 5) -> dict:
    output.mkdir()
    manifest = {"selected_counts": {}, "file_sha256": {}}
    for dataset in ("gov_report", "qmsum", "multi_news", "lcc", "repobench-p"):
        rows = []
        for index in range(count):
            row = canonicalize_record(
                dataset,
                index,
                {"context": f"context {index}", "input": "", "answers": ["reference"]},
                fake_tokenizer,
            )
            if dataset in {"lcc", "repobench-p"}:
                row["length_bin"] = index
            rows.append(row)
        path = output / f"{dataset}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest["selected_counts"][dataset] = count
        manifest["file_sha256"][dataset] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_validate_output_dir_rejects_checksum_mismatch(tmp_path):
    output = tmp_path / "longbench_200"
    _write_valid_output_dir(output)
    path = output / "gov_report.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        validate_output_dir(output, expected_count=5)


def test_task_type_selects_metric_family():
    assert metric_family("summarization") == "rouge"
    assert metric_family("code_completion") == "code_completion"


def test_common_loader_renders_prompt_from_canonical_context_and_input():
    row = canonicalize_record(
        "repobench-p",
        4,
        {"context": "class Example:\n", "input": "return", "answers": [" value"]},
        fake_tokenizer,
    )
    loaded = normalize(row, 0)

    assert loaded["id"] == row["id"]
    assert "Please complete the code" in loaded["prompt"]
    assert "return" in loaded["prompt"]
    assert loaded["reference"] == " value"


def test_collector_loads_canonical_dataset_filenames(tmp_path):
    for dataset in ("gov_report", "qmsum", "multi_news", "lcc", "repobench-p"):
        row = {
            "id": f"{dataset}_1",
            "reference": "reference",
            "context": "context",
        }
        (tmp_path / f"{dataset}.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )

    datasets, index = collect_metrics.load_data_index(tmp_path)

    assert datasets == ["gov_report", "lcc", "multi_news", "qmsum", "repobench-p"]
    assert index["lcc"]["lcc_1"]["reference"] == "reference"


def test_code_completion_metrics_do_not_use_rouge():
    scores = metrics.code_completion_scores("  return value  ", "return value")

    assert scores["code_exact_match"] == 1.0
    assert scores["code_edit_similarity"] == 1.0
    assert "rouge1_f" not in scores


def test_collector_routes_code_records_to_code_metrics():
    group = collect_metrics.compute_group(
        [{"id": "lcc_1", "text": "return value"}],
        {"lcc_1": {"reference": "return value", "task_type": "code_completion"}},
    )

    assert group["code_completion"]["code_exact_match"] == 1.0
    assert "semantic" not in group
