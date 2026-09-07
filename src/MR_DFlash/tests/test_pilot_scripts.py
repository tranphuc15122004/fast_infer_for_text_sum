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
