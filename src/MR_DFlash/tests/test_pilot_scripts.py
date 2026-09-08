"""Smoke tests không cần model thật cho các bước prepare/split/regenerate."""

from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_prepare_split_regenerate_validate_smoke(tmp_path: Path) -> None:
    from build_pilot_dataset import main as build_main
    from prepare_arxiv import main as arxiv_main
    from prepare_sharegpt import main as sharegpt_main
    from regenerate_pilot import main as regenerate_main
    from validate_pilot_dataset import main as validate_main

    share_raw = tmp_path / "share.jsonl"
    arxiv_raw = tmp_path / "arxiv.jsonl"
    _write_jsonl(
        share_raw,
        [
            {"id": "1", "conversations": [{"from": "human", "value": "Question one"}, {"from": "gpt", "value": "old"}]},
            {"id": "2", "conversations": [{"from": "human", "value": "Question two"}]},
        ],
    )
    _write_jsonl(arxiv_raw, [{"id": "p1", "text": "A paper document."}, {"id": "p2", "article": "Another document."}, {"id": "p3", "body": "Third document."}])
    normalized = tmp_path / "normalized"
    share_path = normalized / "sharegpt_prompts.jsonl"
    arxiv_path = normalized / "arxiv_prompts.jsonl"
    sharegpt_main(["--input", str(share_raw), "--output", str(share_path)])
    arxiv_main(["--input", str(arxiv_raw), "--output", str(arxiv_path)])
    build_main([
        "--sharegpt", str(share_path),
        "--arxiv", str(arxiv_path),
        "--output-root", str(tmp_path),
        "--sharegpt-count", "2",
        "--arxiv-count", "3",
        "--allow-short",
    ])
    responses = [{"id": row["id"], "assistant": f"Generated response for {row['id']}"} for row in map(json.loads, (tmp_path / "normalized" / "train_prompts.jsonl").read_text().splitlines())]
    response_path = tmp_path / "responses.jsonl"
    _write_jsonl(response_path, responses)
    regenerated = tmp_path / "regenerated" / "train.jsonl"
    regenerate_main([
        "--input", str(tmp_path / "normalized" / "train_prompts.jsonl"),
        "--output", str(regenerated),
        "--responses-jsonl", str(response_path),
    ])
    validate_main(["--input", str(regenerated), "--require-generated"])
    rows = list(regenerated.open("r", encoding="utf-8"))
    assert rows


def test_prepare_sharegpt_accepts_json_array(tmp_path: Path) -> None:
    import json

    from prepare_sharegpt import main as sharegpt_main

    source = tmp_path / "ShareGPT_V3.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "conversation-1",
                    "conversations": [
                        {"from": "human", "value": "First question"},
                        {"from": "gpt", "value": "Old answer"},
                        {"from": "human", "value": "Final question"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "sharegpt_prompts.jsonl"
    sharegpt_main(["--input", str(source), "--output", str(output)])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["id"] == "sharegpt_conversation-1"
    assert rows[0]["conversations"][-1] == {
        "role": "user",
        "content": "Final question",
    }
    assert rows[0]["metadata"]["original_turn_count"] == 3


def test_prepare_arxiv_joins_paragraphs_and_preserves_reference(tmp_path: Path) -> None:
    import json

    from prepare_arxiv import main as arxiv_main

    source = tmp_path / "train.label.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "paper-1",
                "text": ["Paragraph one.", "Paragraph two."],
                "summary": ["Summary one.", "Summary two."],
                "label": [1, 3],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "arxiv_prompts.jsonl"
    arxiv_main(["--input", str(source), "--output", str(output)])
    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["conversations"][0]["content"].endswith(
        "Paragraph one.\n\nParagraph two."
    )
    assert row["metadata"]["reference_summary"] == "Summary one.\n\nSummary two."
    assert row["metadata"]["label"] == [1, 3]


def test_prepare_server_sources_builds_current_pilot_layout(tmp_path: Path) -> None:
    import json

    from prepare_server_data import main as prepare_server_main

    share_source = tmp_path / "sharegpt.json"
    share_source.write_text(
        json.dumps(
            [
                {
                    "id": "s1",
                    "conversations": [
                        {"from": "human", "value": "Share question"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    arxiv_source = tmp_path / "arxiv.jsonl"
    arxiv_source.write_text(
        json.dumps({"id": "a1", "text": ["Document"], "summary": ["Reference"]})
        + "\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "pilot"
    prepare_server_main(
        [
            "--sharegpt-source",
            str(share_source),
            "--arxiv-source",
            str(arxiv_source),
            "--output-root",
            str(output_root),
            "--sharegpt-count",
            "1",
            "--arxiv-count",
            "1",
            "--allow-short",
        ]
    )
    assert (output_root / "normalized" / "train_prompts.jsonl").exists()
    manifest = json.loads(
        (output_root / "manifests" / "source_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sharegpt_input"] == str(share_source)
    assert manifest["arxiv_input"] == str(arxiv_source)
    assert manifest["sharegpt_count"] == 1
    assert manifest["arxiv_count"] == 1


def test_analyze_pilot_data_reports_small_sample(tmp_path: Path) -> None:
    import json

    from analyze_pilot_data import main as analyze_main

    source = tmp_path / "prompts.jsonl"
    rows = [
        {
            "id": "sharegpt_s1",
            "source": "sharegpt",
            "conversations": [{"role": "user", "content": "Question"}],
            "metadata": {"source_length": 8},
        },
        {
            "id": "arxiv_a1",
            "source": "arxiv",
            "conversations": [{"role": "user", "content": "Document"}],
            "metadata": {"source_length": 8, "source_token_length": 2, "reference_summary": "Summary"},
        },
    ]
    _write_jsonl(source, rows)
    report_path = tmp_path / "analysis.json"
    analyze_main(["--input", str(source), "--output", str(report_path), "--limit", "2"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["rows"] == 2
    assert report["source_counts"] == {"arxiv": 1, "sharegpt": 1}
    assert report["duplicate_ids"] == []
    assert report["reference_rows"] == 1


def test_server_data_defaults_balance_sharegpt_and_arxiv() -> None:
    from build_pilot_dataset import DEFAULT_ARXIV_COUNT, DEFAULT_SHAREGPT_COUNT
    from prepare_server_data import DEFAULT_ARXIV_COUNT as WRAPPER_ARXIV_COUNT
    from prepare_server_data import DEFAULT_SHAREGPT_COUNT as WRAPPER_SHAREGPT_COUNT

    assert DEFAULT_SHAREGPT_COUNT == 50000
    assert DEFAULT_ARXIV_COUNT == 50000
    assert WRAPPER_SHAREGPT_COUNT == 50000
    assert WRAPPER_ARXIV_COUNT == 50000
