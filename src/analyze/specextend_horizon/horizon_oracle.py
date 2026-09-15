"""Hindsight Horizon-CMR analysis for chunk-level SpecExtend traces.

The module is deliberately independent from torch and the SpecExtend model.
It consumes JSONL trace records written by the optional runtime hook and
produces reproducible overlap, acceptance and projected-cost statistics.
No target-future signal is used by the runtime policy; future relevance is
used only in this offline oracle.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _chunk_id(chunk: Any) -> int | str:
    if isinstance(chunk, dict):
        return chunk.get("id", chunk.get("chunk_id"))
    if isinstance(chunk, (list, tuple)):
        return chunk[0]
    return chunk


def _chunk_span(chunk: Any) -> tuple[int, int] | None:
    if isinstance(chunk, dict):
        start, end = chunk.get("start"), chunk.get("end")
    elif isinstance(chunk, (list, tuple)) and len(chunk) >= 3:
        _, start, end = chunk[:3]
    else:
        return None
    try:
        return int(start), int(end)
    except (TypeError, ValueError):
        return None


def weighted_top_ids(scores: dict[Any, Any] | None, k: int) -> list[Any]:
    """Return deterministic top-k IDs from a score mapping."""
    if not scores:
        return []
    valid = [(key, _finite(value)) for key, value in scores.items()]
    valid = [(key, value) for key, value in valid if value is not None]
    valid.sort(key=lambda item: (-item[1], str(item[0])))
    return [key for key, _ in valid[: max(0, int(k))]]


def set_overlap(current: Iterable[Any], horizon: Iterable[Any]) -> dict[str, float | int]:
    cur = set(current)
    hor = set(horizon)
    inter = len(cur & hor)
    union = len(cur | hor)
    return {
        "current_count": len(cur),
        "horizon_count": len(hor),
        "intersection_count": inter,
        "recall_current_in_horizon": inter / len(hor) if hor else None,
        "precision_current_vs_horizon": inter / len(cur) if cur else None,
        "jaccard": inter / union if union else None,
    }


def acceptance_curve(records: list[dict[str, Any]], max_position: int | None = None) -> dict[str, Any]:
    """Compute P(accepted tokens >= j) from per-cycle acceptance lengths."""
    by_dataset: dict[str, list[int]] = defaultdict(list)
    for record in records:
        if record.get("type") != "cycle":
            continue
        accepted = record.get("accepted_tokens")
        if accepted is None:
            raw = record.get("accept_length")
            try:
                accepted = int(raw) + 1
            except (TypeError, ValueError):
                continue
        try:
            accepted = max(0, int(accepted))
        except (TypeError, ValueError):
            continue
        by_dataset[str(record.get("dataset", "unknown"))].append(accepted)

    result: dict[str, Any] = {}
    for dataset, values in sorted(by_dataset.items()):
        if not values:
            continue
        upper = max_position or max(values)
        survival = {
            str(position): sum(value >= position for value in values) / len(values)
            for position in range(1, upper + 1)
        }
        result[dataset] = {
            "cycles": len(values),
            "mean_accepted_tokens_per_cycle": sum(values) / len(values),
            "max_accepted_tokens": max(values),
            "survival": survival,
        }
    return result


def _projected_cost(record: dict[str, Any]) -> dict[str, float | None]:
    """Project cycle cost by retained-token ratio, without claiming a run.

    Horizon and current policies normally use the same chunk budget, so this
    projection will often be exactly 1.0. That is an important negative result:
    a horizon policy can change candidate quality without reducing attention
    work unless it also changes retained context size.
    """
    current = record.get("current_context_tokens")
    horizon = record.get("horizon_context_tokens")
    cycle = _finite(record.get("cycle_time_s"))
    current = _finite(current)
    horizon = _finite(horizon)
    if not current or current <= 0 or horizon is None:
        return {
            "projected_time_s": None,
            "horizon_to_current_context_ratio": None,
            "projected_speedup": None,
        }
    ratio = horizon / current
    return {
        "projected_time_s": cycle * ratio if cycle is not None else None,
        "horizon_to_current_context_ratio": ratio,
        "projected_speedup": 1.0 / ratio if ratio > 0 else None,
    }


def analyze_records(records: list[dict[str, Any]], top_k: int = 32) -> dict[str, Any]:
    cycle_records = [record for record in records if record.get("type") == "cycle"]
    overlaps: list[dict[str, Any]] = []
    projected: list[dict[str, Any]] = []
    accepted = []
    attention_available = 0
    for record in cycle_records:
        current_ids = record.get("current_chunk_ids") or []
        horizon_ids = record.get("horizon_chunk_ids") or weighted_top_ids(
            record.get("horizon_chunk_scores"), top_k
        )
        if record.get("horizon_chunk_scores") is not None:
            attention_available += 1
        overlap = set_overlap(current_ids, horizon_ids)
        overlap.update({"dataset": record.get("dataset", "unknown"), "cycle": record.get("cycle")})
        overlaps.append(overlap)
        projected.append(_projected_cost(record))
        accepted.append(record)

    def mean(key: str) -> float | None:
        values = [_finite(row.get(key)) for row in overlaps]
        values = [value for value in values if value is not None]
        return sum(values) / len(values) if values else None

    ratios = [
        _finite(row.get("horizon_to_current_context_ratio"))
        for row in projected
    ]
    ratios = [value for value in ratios if value is not None]
    return {
        "trace_records": len(records),
        "cycle_records": len(cycle_records),
        "attention_available_cycles": attention_available,
        "overlap": {
            "mean_recall_current_in_horizon": mean("recall_current_in_horizon"),
            "mean_precision_current_vs_horizon": mean("precision_current_vs_horizon"),
            "mean_jaccard": mean("jaccard"),
            "per_cycle": overlaps,
        },
        "projected_cost": {
            "mean_horizon_to_current_context_ratio": sum(ratios) / len(ratios) if ratios else None,
            "mean_projected_speedup": (
                sum(1.0 / ratio for ratio in ratios if ratio > 0) / len(ratios)
                if ratios else None
            ),
            "per_cycle": projected,
            "interpretation": "projected only; no oracle-context runtime was executed",
        },
        "acceptance": acceptance_curve(accepted),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=32)
    args = parser.parse_args()
    summary = analyze_records(load_jsonl(args.trace), top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "overlap"}, indent=2))


if __name__ == "__main__":
    main()
