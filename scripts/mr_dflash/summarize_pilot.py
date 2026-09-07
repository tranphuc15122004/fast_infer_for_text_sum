"""Tổng hợp metrics train và JSONL speculative evaluation thành report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Summarize MR-DFlash pilot runs")
    parser.add_argument("--run", nargs="+", help="name=output_dir pairs")
    parser.add_argument("--eval", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report: Dict[str, Any] = {"runs": {}, "evaluations": {}}
    for item in args.run or []:
        if "=" not in item:
            raise ValueError("--run dùng dạng name=output_dir")
        name, directory = item.split("=", 1)
        root = Path(directory)
        metrics = _jsonl(root / "metrics.jsonl") if (root / "metrics.jsonl").exists() else []
        report["runs"][name] = {
            "output_dir": str(root),
            "steps": len(metrics),
            "last": metrics[-1] if metrics else None,
            "best_loss": min((float(row["loss"]) for row in metrics if "loss" in row), default=None),
            "best_accept_ge_1": max((float(row["accept_ge_1"]) for row in metrics if "accept_ge_1" in row), default=None),
        }
    for item in args.eval:
        path = Path(item)
        rows = _jsonl(path)
        summary = next((row for row in reversed(rows) if row.get("type") == "summary"), None)
        report["evaluations"][path.stem] = summary or {"samples": len(rows)}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[summarize_pilot] output={target}")


if __name__ == "__main__":
    main()
