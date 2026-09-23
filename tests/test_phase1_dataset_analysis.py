from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mr_dflash"))


def _write_fixture_sources(root: Path) -> tuple[Path, Path]:
    sharegpt = root / "sharegpt.json"
    sharegpt.write_text(
        json.dumps(
            [
                {
                    "id": "s1",
                    "conversations": [
                        {"from": "human", "value": "short question"},
                        {"from": "gpt", "value": "short answer"},
                        {"from": "human", "value": "follow-up"},
                    ],
                },
                {
                    "id": "s2",
                    "conversations": [{"from": "human", "value": "second"}],
                },
                {"id": "bad", "conversations": [{"from": "gpt", "value": "no user"}]},
                {
                    "id": "s1",
                    "conversations": [{"from": "human", "value": "duplicate"}],
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    arxiv = root / "arxiv.jsonl"
    arxiv.write_text(
        "\n".join(
            [
                json.dumps({"id": "a1", "text": "a" * 20, "summary": "sum"}),
                "{malformed",
                json.dumps({"id": "a2", "document": "b" * 50}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return sharegpt, arxiv


def test_analysis_scans_in_input_order_without_sampling(tmp_path: Path) -> None:
    from analyze_phase1_dataset import analyze_dataset

    sharegpt, arxiv = _write_fixture_sources(tmp_path)
    output = tmp_path / "analysis"

    summary = analyze_dataset(
        sharegpt_source=sharegpt,
        arxiv_source=arxiv,
        output_root=output,
    )

    records = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [row["id"] for row in records] == ["s1", "s2", "bad", "s1", "a1", "a2"]
    assert summary["scanned_rows"] == 6
    assert summary["valid_rows"] == 4
    assert summary["invalid_rows"] == 2
    assert summary["duplicate_ids"] == {"s1": 2}
    assert summary["source_counts"] == {"arxiv": 2, "sharegpt": 4}
    assert not (output / "train_prompts.jsonl").exists()
    assert not (output / "val_prompts.jsonl").exists()
    assert not (output / "test_prompts.jsonl").exists()


def test_analysis_reports_distribution_and_issues(tmp_path: Path) -> None:
    from analyze_phase1_dataset import analyze_dataset

    sharegpt, arxiv = _write_fixture_sources(tmp_path)
    output = tmp_path / "analysis"
    summary = analyze_dataset(
        sharegpt_source=sharegpt,
        arxiv_source=arxiv,
        output_root=output,
    )

    assert summary["length_stats"]["count"] == 4
    assert summary["length_stats"]["quantiles"]["p50"] is not None
    issues = [
        json.loads(line)
        for line in (output / "issues.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert any(issue["kind"] == "malformed_json" for issue in issues)
    assert any(issue["kind"] == "invalid_record" for issue in issues)


def test_analysis_writes_reproducible_figures(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    from analyze_phase1_dataset import analyze_dataset

    sharegpt, arxiv = _write_fixture_sources(tmp_path)
    figures = tmp_path / "analysis" / "figures"
    analyze_dataset(
        sharegpt_source=sharegpt,
        arxiv_source=arxiv,
        output_root=figures.parent,
    )

    expected = {
        "token_length_distribution.png",
        "token_length_ecdf.png",
        "source_composition.png",
        "conversation_structure.png",
    }
    assert {path.name for path in figures.iterdir()} == expected
    assert all(path.stat().st_size > 0 for path in figures.iterdir())
