"""Unified plug-and-play data loader for every baseline script.

Drop a jsonl file (one JSON record per line) into ``data/`` and point the
baseline's ``DATA_FILE`` config at it. All accepted record shapes are
normalized to ``{id, prompt, answer?, keyword?, text?}``.

Accepted fields (first match wins):
    {"id": 0, "prompt": "..."}                 # preferred
    {"id": 0, "question": "...", "answer": "..."}
    {"id": 0, "text": "...", "answer": "..."}  # summarization-style docs
    {"id": 0, "turns": ["user prompt", ...]}   # EAGLE-style chat turns
    {"id": 0, "instruction": "..."}

Optional extra hints used by verification:
    "keyword"  -> a distinctive entity expected to survive compression
    "answer"   -> reference answer (for QA-style checks)
"""

from __future__ import annotations

import json
from pathlib import Path


def _get(record: dict, *keys) -> str | None:
    for k in keys:
        v = record.get(k)
        if v is None:
            continue
        if k == "turns" and isinstance(v, list):
            if v:
                return str(v[0])
            continue
        if isinstance(v, str) and v.strip():
            return v
    return None


def normalize(record: dict, idx: int) -> dict:
    prompt = _get(record, "prompt", "question", "instruction", "text", "turns")
    return {
        "id": record.get("id", idx),
        "prompt": str(prompt) if prompt is not None else "",
        "answer": record.get("answer"),
        "keyword": record.get("keyword"),
        "text": record.get("text"),
        "raw": record,
    }


def load_records(path: Path, max_samples: int | None = None) -> list[dict]:
    """Load and normalize a jsonl file of records."""
    path = Path(path)
    if not path.is_absolute():
        from common.paths import ROOT
        path = ROOT / path
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"No records in {path}")
    return [normalize(r, i) for i, r in enumerate(rows)]


def load_prompts(path: Path, max_samples: int | None = None) -> list[dict]:
    """Like load_records but drops records without a prompt."""
    records = load_records(path, max_samples)
    with_prompt = [r for r in records if r["prompt"]]
    if not with_prompt:
        raise ValueError(f"No record with a usable 'prompt'/'question'/'text' field in {path}")
    return with_prompt
