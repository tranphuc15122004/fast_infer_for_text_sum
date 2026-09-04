"""P2 candidate-coverage tables over context bins and draft positions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .metrics import recall_at_k


def analyze_coverage(
    rows: Iterable[Mapping[str, Any]],
    *,
    recall_k: int = 16,
    recall_ks: tuple[int, ...] = (1, 4, 8, 16),
    min_relative_drop: float = 0.15,
    min_documents: int = 5,
) -> dict[str, Any]:
    """Aggregate Recall@M for every ``(regime, context_bin, j)`` cell."""

    usable = [row for row in rows if row.get("status", "ok") == "ok"]
    if not usable:
        return {"status": "unavailable", "reason": "no_valid_trace_rows", "rows": 0}
    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        grouped[(str(row.get("task_regime", "other")), str(row["context_bin"]), int(row["draft_position"]))].append(row)
    table: list[dict[str, Any]] = []
    for (regime, context, position), cell in sorted(grouped.items(), key=lambda item: item[0]):
        table.append({
            "task_regime": regime,
            "context_bin": context,
            "draft_position": position,
            "rows": len(cell),
            **{f"recall_at_{k}": recall_at_k(cell, k) for k in recall_ks},
        })
    def context_sort_key(value: str) -> tuple[int, str]:
        first = value.split("-", 1)[0].replace("k", "").replace(">", "")
        try:
            return (int(first), value)
        except ValueError:
            return (10**9, value)

    contexts = sorted({str(row["context_bin"]) for row in usable}, key=context_sort_key)
    positions = sorted({int(row["draft_position"]) for row in usable})
    heatmap_grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        heatmap_grouped[(str(row["context_bin"]), int(row["draft_position"]))].append(row)
    heatmap = {
        context: {
            str(position): recall_at_k(heatmap_grouped[(context, position)], recall_k)
            for position in positions
        }
        for context in contexts
    }
    short_context = contexts[0] if len(contexts) >= 2 else None
    long_context = contexts[-1] if len(contexts) >= 2 else None
    short_rows = [row for row in usable if str(row["context_bin"]) == short_context]
    long_rows = [row for row in usable if str(row["context_bin"]) == long_context]
    short_recall = recall_at_k(short_rows, recall_k) if short_context is not None else None
    long_recall = recall_at_k(long_rows, recall_k) if long_context is not None else None
    relative_drop = None
    if short_recall is not None and long_recall is not None and short_recall > 0:
        relative_drop = (short_recall - long_recall) / short_recall
    h2_gate = {
        "decision": "INCONCLUSIVE",
        "reason": "missing_short_or_long_recall",
        "short_recall": short_recall,
        "long_recall": long_recall,
        "short_context_bin": short_context,
        "long_context_bin": long_context,
        "relative_drop": relative_drop,
        "threshold": min_relative_drop,
        "min_documents": min_documents,
    }
    short_documents = len({str(row["document_id"]) for row in short_rows})
    long_documents = len({str(row["document_id"]) for row in long_rows})
    h2_gate["short_documents"] = short_documents
    h2_gate["long_documents"] = long_documents
    if relative_drop is not None and short_documents >= min_documents and long_documents >= min_documents:
        h2_gate["decision"] = "PASS" if relative_drop >= min_relative_drop else "FAIL"
        h2_gate["reason"] = "long_context_recall_drop" if h2_gate["decision"] == "PASS" else "drop_below_gate"
    elif relative_drop is not None:
        h2_gate["reason"] = "insufficient_short_or_long_documents"
    return {
        "status": "ok",
        "rows": len(usable),
        "recall_k": recall_k,
        "contexts": contexts,
        "positions": positions,
        "table": table,
        "heatmap": heatmap,
        "h2_gate": h2_gate,
    }
