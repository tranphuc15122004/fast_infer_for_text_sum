#!/usr/bin/env python3
"""Fit the empirical Stage-4 SyncSpec pre-draft gate from paired traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.controller import fit_empirical_gate_table  # noqa: E402


def _read_rows(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_no}: {exc}") from exc
        if isinstance(value, dict) and "summary" not in value:
            rows.append(value)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSONL paired serving traces")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    table = fit_empirical_gate_table(_read_rows(args.input))
    payload = {
        "schema_version": 1,
        "method": "syncspec_gate",
        "source": "empirical",
        "gate_table": table,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
