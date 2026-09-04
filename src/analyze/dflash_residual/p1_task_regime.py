"""P1 canonical-to-summarization task-regime comparison."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .metrics import summarize_p1


def analyze_task_regimes(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_relative_drop: float = 0.15,
    min_documents: int = 5,
) -> dict[str, Any]:
    """Return regime metrics and the H1 canonical-to-summarization gate."""

    usable = [row for row in rows if row.get("status", "ok") == "ok"]
    result = summarize_p1(usable)
    regimes = result.get("regimes", {})
    canonical = regimes.get("canonical")
    summary_rows = [row for row in usable if str(row.get("task_regime")) != "canonical"]
    summary = summarize_p1(summary_rows).get("regimes", {})
    summary_mat_values = [
        (float(value["mat"]), int(value.get("blocks", 0)))
        for value in summary.values()
        if value.get("mat") is not None and value.get("blocks", 0)
    ]
    summary_mat = (
        sum(mat * blocks for mat, blocks in summary_mat_values) / sum(blocks for _, blocks in summary_mat_values)
        if summary_mat_values else None
    )
    comparison: dict[str, Any] = {
        "status": "INCONCLUSIVE",
        "decision": "INCONCLUSIVE",
        "reason": None,
        "canonical_mat": canonical.get("mat") if canonical else None,
        "summarization_mat": summary_mat,
        "relative_drop": None,
        "summarization_blocks": sum(int(value.get("blocks", 0)) for value in summary.values()),
        "min_documents": min_documents,
    }
    canonical_documents = int(canonical.get("documents", 0)) if canonical else 0
    summary_documents = len({str(row["document_id"]) for row in summary_rows})
    comparison["canonical_documents"] = canonical_documents
    comparison["summarization_documents"] = summary_documents
    if (
        not canonical or canonical.get("mat") is None or summary_mat is None
        or canonical_documents < min_documents or summary_documents < min_documents
    ):
        comparison["reason"] = "insufficient_canonical_or_summarization_documents"
    elif float(canonical["mat"]) <= 0:
        comparison["reason"] = "canonical_acceptance_not_positive"
    else:
        relative_drop = (float(canonical["mat"]) - summary_mat) / float(canonical["mat"])
        comparison["relative_drop"] = relative_drop
        comparison["status"] = "ok"
        if relative_drop >= min_relative_drop:
            comparison["decision"] = "PASS"
            comparison["reason"] = "summarization_drop_exceeds_gate"
        else:
            comparison["decision"] = "FAIL"
            comparison["reason"] = "summarization_drop_below_gate"
    result["h1_gate"] = comparison
    return result
