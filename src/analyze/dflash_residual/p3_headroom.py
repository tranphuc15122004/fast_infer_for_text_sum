"""P3 DFlash/DFlash2/oracle headroom decomposition."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from .metrics import summarize_headroom


def analyze_headroom(
    rows: Iterable[Mapping[str, Any]],
    *,
    oracle_k: int = 16,
    min_blocks: int = 5,
) -> dict[str, Any]:
    """Compute global and context-bin decomposition on a fixed candidate lattice."""

    usable = [row for row in rows if row.get("status", "ok") == "ok"]
    result = summarize_headroom(usable, oracle_k=oracle_k)
    if result.get("status") != "ok":
        return result
    if int(result.get("blocks", 0)) < min_blocks:
        return {
            **result,
            "status": "inconclusive",
            "reason": "insufficient_blocks",
            "min_blocks": min_blocks,
        }
    by_context: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        by_context[str(row["context_bin"])].append(row)
    result["by_context_bin"] = {
        context: summarize_headroom(context_rows, oracle_k=oracle_k)
        for context, context_rows in sorted(by_context.items())
    }
    context_rho = {
        context: values.get("rho_d2")
        for context, values in result["by_context_bin"].items()
        if values.get("status") == "ok" and values.get("rho_d2") is not None
    }
    result["rho_by_context_bin"] = context_rho
    def context_sort_key(value: str) -> tuple[int, str]:
        first = value.split("-", 1)[0].replace("k", "").replace(">", "")
        try:
            return (int(first), value)
        except ValueError:
            return (10**9, value)

    ordered_rho = [context_rho[context] for context in sorted(context_rho, key=context_sort_key)]
    if len(ordered_rho) >= 2:
        short_rho, long_rho = ordered_rho[0], ordered_rho[-1]
        result["h3_gate"] = {
            "decision": "PASS" if long_rho < short_rho else "FAIL",
            "reason": "rho_decreases_with_context" if long_rho < short_rho else "rho_does_not_decrease_with_context",
            "short_rho": short_rho,
            "long_rho": long_rho,
        }
    else:
        result["h3_gate"] = {"decision": "INCONCLUSIVE", "reason": "insufficient_context_bins_with_rho"}
    rho = result.get("rho_d2")
    if rho is None:
        result["selection_gate"] = {"decision": "INCONCLUSIVE", "reason": result.get("rho_status")}
    elif rho < 0.5:
        result["selection_gate"] = {"decision": "PASS", "reason": "low_dflash2_recovery"}
    elif rho >= 0.7:
        result["selection_gate"] = {"decision": "STOP", "reason": "dflash2_recovers_most_oracle_headroom"}
    else:
        result["selection_gate"] = {"decision": "INCONCLUSIVE", "reason": "rho_between_gates"}
    return result
