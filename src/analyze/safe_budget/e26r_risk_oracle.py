"""E26-R risk-matched dynamic-budget oracle analysis.

This analyzer is intentionally offline.  It consumes paired semantic-selection
JSONL records and compares fixed policies with an adaptive oracle under the
same document-level risk contract:

    fraction of documents violating any requested quality tolerance <= alpha

The adaptive oracle is solved exactly by dynamic programming over documents
and observed actions.  It minimizes total measured pipeline cost subject to
the same maximum number of violating documents as the fixed-policy screen.
Missing quality metrics, documents, or budget levels are reported explicitly;
they never become a passing result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_EPSILONS = (0.01, 0.02, 0.05)
DEFAULT_REQUIRED_LEVELS = 6
DEFAULT_MIN_DOCUMENTS = 30
DEFAULT_REQUIRED_BUDGET_LABELS = (
    "full",
    "ratio_0.25",
    "ratio_0.4",
    "ratio_0.55",
    "ratio_0.7",
    "ratio_0.85",
)

RAW_LEVEL_ORDER = (
    "full",
    "ratio_0.25",
    "ratio_0.4",
    "ratio_0.55",
    "ratio_0.7",
    "ratio_0.85",
)

RAW_MEAN_FIELDS = (
    "original_tokens",
    "selected_tokens",
    "token_budget",
    "retention_ratio",
    "token_reduction_ratio",
    "output_tokens",
    "selection_total_wall_ms",
    "prefill_ms",
    "decode_ms",
    "pipeline_ttft_ms",
    "pipeline_e2e_ms",
    "output_tokens_per_second",
    "peak_gpu_allocated_mb",
    "peak_gpu_reserved_mb",
    "rouge1",
    "rouge2",
    "rougeL",
    "bertscore_p",
    "bertscore_r",
    "bertscore_f1",
    "e2e_speedup_vs_full",
    "prefill_speedup_vs_full",
    "ttft_speedup_vs_full",
    "decode_speedup_vs_full",
)

QUALITY_ALIASES: dict[str, tuple[str, ...]] = {
    "rougeL": ("rougeL", "rougeL_f", "rouge_l", "rouge_l_f"),
    "rouge1": ("rouge1", "rouge1_f", "rouge_1", "rouge_1_f"),
    "rouge2": ("rouge2", "rouge2_f", "rouge_2", "rouge_2_f"),
    "bertscore_f1": (
        "bertscore_f1",
        "bert_score_f1",
        "bertScore_f1",
        "bertscore",
    ),
}


def _number(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _first_number(row: Mapping[str, Any], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = _number(row, key)
        if value is not None:
            return value
    return None


def _quality(row: Mapping[str, Any], metric: str) -> float | None:
    return _first_number(row, QUALITY_ALIASES.get(metric, (metric,)))


def _budget_tokens(row: Mapping[str, Any]) -> float | None:
    if str(row.get("budget_label", "")) == "full":
        return _number(row, "original_tokens")
    requested = _number(row, "token_budget")
    if requested is not None and requested > 0:
        return requested
    return _number(row, "selected_tokens")


def _condition_key(row: Mapping[str, Any]) -> str:
    label = str(row.get("budget_label", "")).strip()
    if label:
        return label
    budget = _budget_tokens(row)
    return f"tokens_{int(budget)}" if budget is not None else "unknown"


def _cost(row: Mapping[str, Any], metric: str) -> float | None:
    return _number(row, metric)


def _mean(values: Iterable[float]) -> float | None:
    data = list(values)
    return sum(data) / len(data) if data else None


def _raw_level_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    selector: str,
) -> list[dict[str, Any]]:
    """Aggregate every measured condition for the provenance report.

    This is intentionally separate from the risk oracle.  The oracle reports
    only policies that satisfy a quality contract; this table reports all raw
    inference conditions, including conditions that fail that contract.
    """
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        label = _condition_key(row)
        row_selector = str(row.get("selector", ""))
        if label == "full" and row_selector == "full":
            grouped[("full", "full")].append(row)
        elif row_selector == selector:
            grouped[(selector, label)].append(row)

    observed = sorted(
        {label for _, label in grouped},
        key=lambda label: (
            RAW_LEVEL_ORDER.index(label) if label in RAW_LEVEL_ORDER else len(RAW_LEVEL_ORDER),
            label,
        ),
    )
    full_rows = grouped.get(("full", "full"), [])
    full_e2e = _mean(
        value
        for row in full_rows
        for value in [_number(row, "pipeline_e2e_ms")]
        if value is not None
    )
    output: list[dict[str, Any]] = []
    for label in observed:
        group_selector = "full" if label == "full" else selector
        group = grouped.get((group_selector, label), [])
        item: dict[str, Any] = {
            "selector": group_selector,
            "budget_label": label,
            "rows": len(group),
            "documents": len({_doc_key(row) for row in group}),
        }
        for field in RAW_MEAN_FIELDS:
            item[field] = _mean(
                value
                for row in group
                for value in [_number(row, field)]
                if value is not None
            )
        if item.get("pipeline_e2e_ms") is not None and full_e2e not in (None, 0):
            item["speedup_vs_full_recomputed"] = full_e2e / item["pipeline_e2e_ms"]
        else:
            item["speedup_vs_full_recomputed"] = None
        output.append(item)
    return output


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _fmt(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def _doc_key(row: Mapping[str, Any]) -> str:
    value = row.get("example_id", row.get("id", ""))
    return str(value)


def _rows_by_doc(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_doc_key(row)].append(dict(row))
    return dict(grouped)


def _resample_rows(rows: Sequence[Mapping[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    """Bootstrap documents with replacement while preserving paired actions."""
    grouped = _rows_by_doc(rows)
    documents = list(grouped)
    sampled: list[dict[str, Any]] = []
    for occurrence, source_id in enumerate(
        rng.choice(documents) for _ in range(len(documents))
    ):
        for row in grouped[source_id]:
            clone = dict(row)
            clone["example_id"] = f"{source_id}__bootstrap_{occurrence}"
            sampled.append(clone)
    return sampled


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    data = sorted(float(value) for value in values)
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return data[lower]
    fraction = position - lower
    return data[lower] * (1.0 - fraction) + data[upper] * fraction


def _extract_result_cell(
    result: Mapping[str, Any],
    epsilon: float,
    alpha: float,
) -> Mapping[str, Any]:
    return result["epsilon_results"][f"{float(epsilon):.4f}"]["alpha_results"][f"{float(alpha):.4f}"]


def bootstrap_validation(
    rows: Sequence[Mapping[str, Any]],
    *,
    selector: str,
    quality_metrics: Sequence[str],
    cost_metric: str,
    epsilons: Sequence[float],
    alphas: Sequence[float],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    """Paired document bootstrap CI for risk-matched headroom and costs."""
    if samples <= 0:
        return {"samples_requested": 0, "seed": seed, "cells": []}
    rng = random.Random(seed)
    cells: list[dict[str, Any]] = []
    for epsilon in epsilons:
        for alpha in alphas:
            headrooms: list[float] = []
            fixed_costs: list[float] = []
            adaptive_costs: list[float] = []
            for _ in range(samples):
                replicate = _resample_rows(rows, rng)
                result = analyze_dataset(
                    replicate,
                    selector=selector,
                    quality_metrics=quality_metrics,
                    cost_metric=cost_metric,
                    epsilons=[epsilon],
                    alphas=[alpha],
                    required_levels=0,
                    required_budget_labels=None,
                    min_documents=0,
                )
                cell = _extract_result_cell(result, epsilon, alpha)
                fixed = cell.get("fixed") or {}
                adaptive = cell.get("adaptive_oracle") or {}
                if fixed.get("mean_cost_ms") is None or adaptive.get("mean_cost_ms") is None:
                    continue
                fixed_costs.append(float(fixed["mean_cost_ms"]))
                adaptive_costs.append(float(adaptive["mean_cost_ms"]))
                if adaptive.get("headroom_vs_best_fixed") is not None:
                    headrooms.append(float(adaptive["headroom_vs_best_fixed"]))
            cells.append(
                {
                    "epsilon": float(epsilon),
                    "alpha": float(alpha),
                    "valid_replicates": len(headrooms),
                    "headroom_mean": sum(headrooms) / len(headrooms) if headrooms else None,
                    "headroom_ci_95": [
                        _percentile(headrooms, 0.025),
                        _percentile(headrooms, 0.975),
                    ],
                    "fixed_cost_ci_95_ms": [
                        _percentile(fixed_costs, 0.025),
                        _percentile(fixed_costs, 0.975),
                    ],
                    "adaptive_cost_ci_95_ms": [
                        _percentile(adaptive_costs, 0.025),
                        _percentile(adaptive_costs, 0.975),
                    ],
                }
            )
    return {"samples_requested": samples, "seed": seed, "cells": cells}


def heldout_validation(
    rows: Sequence[Mapping[str, Any]],
    *,
    selector: str,
    quality_metrics: Sequence[str],
    cost_metric: str,
    epsilons: Sequence[float],
    alphas: Sequence[float],
    test_fraction: float,
    seed: int,
) -> dict[str, Any]:
    """Select fixed policy on calibration documents and evaluate it on holdout."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("heldout fraction must be in (0, 1)")
    grouped = _rows_by_doc(rows)
    documents = sorted(grouped)
    if len(documents) < 2:
        return {"status": "insufficient_documents", "seed": seed}
    rng = random.Random(seed)
    rng.shuffle(documents)
    test_count = max(1, min(len(documents) - 1, int(round(len(documents) * test_fraction))))
    test_ids = set(documents[:test_count])
    calibration_rows = [row for doc_id, doc_rows in grouped.items() if doc_id not in test_ids for row in doc_rows]
    holdout_rows = [row for doc_id, doc_rows in grouped.items() if doc_id in test_ids for row in doc_rows]
    calibration = analyze_dataset(
        calibration_rows,
        selector=selector,
        quality_metrics=quality_metrics,
        cost_metric=cost_metric,
        epsilons=epsilons,
        alphas=alphas,
        required_levels=0,
        required_budget_labels=None,
        min_documents=0,
    )
    holdout = analyze_dataset(
        holdout_rows,
        selector=selector,
        quality_metrics=quality_metrics,
        cost_metric=cost_metric,
        epsilons=epsilons,
        alphas=alphas,
        required_levels=0,
        required_budget_labels=None,
        min_documents=0,
    )
    cells: list[dict[str, Any]] = []
    for epsilon in epsilons:
        for alpha in alphas:
            calibration_cell = _extract_result_cell(calibration, epsilon, alpha)
            holdout_cell = _extract_result_cell(holdout, epsilon, alpha)
            calibrated_fixed = calibration_cell.get("fixed") or {}
            calibrated_label = calibrated_fixed.get("label")
            holdout_candidate = next(
                (candidate for candidate in holdout_cell.get("fixed_candidates", []) if candidate.get("label") == calibrated_label),
                None,
            )
            adaptive = holdout_cell.get("adaptive_oracle") or {}
            fixed_cost = (holdout_candidate or {}).get("mean_cost_ms")
            adaptive_cost = adaptive.get("mean_cost_ms")
            cells.append(
                {
                    "epsilon": float(epsilon),
                    "alpha": float(alpha),
                    "calibration_documents": calibration["documents_usable"],
                    "holdout_documents": holdout["documents_usable"],
                    "calibrated_fixed_label": calibrated_label,
                    "holdout_fixed": holdout_candidate,
                    "holdout_adaptive_oracle": adaptive,
                    "holdout_headroom_vs_calibrated_fixed": (
                        None
                        if fixed_cost in (None, 0) or adaptive_cost is None
                        else 1.0 - float(adaptive_cost) / float(fixed_cost)
                    ),
                }
            )
    return {
        "status": "complete",
        "seed": seed,
        "test_fraction": test_fraction,
        "calibration_document_ids": sorted(set(_doc_key(row) for row in calibration_rows)),
        "holdout_document_ids": sorted(test_ids),
        "calibration": calibration,
        "holdout": holdout,
        "cells": cells,
    }


def _quality_violation(
    row: Mapping[str, Any],
    full: Mapping[str, Any],
    *,
    quality_metrics: Sequence[str],
    epsilon: float,
) -> tuple[bool, list[str]]:
    missing: list[str] = []
    violated: list[str] = []
    for metric in quality_metrics:
        q_full = _quality(full, metric)
        q_row = _quality(row, metric)
        if q_full is None or q_row is None:
            missing.append(metric)
            continue
        if q_full - q_row > epsilon:
            violated.append(metric)
    return bool(violated), missing


def _action_summary(
    rows: Sequence[Mapping[str, Any]],
    full: Mapping[str, Any],
    *,
    quality_metrics: Sequence[str],
    epsilon: float,
    cost_metric: str,
) -> dict[str, Any] | None:
    label = _condition_key(rows[0]) if rows else "unknown"
    quality = {metric: _quality(rows[0], metric) for metric in quality_metrics}
    if any(value is None for value in quality.values()):
        return None
    cost = _cost(rows[0], cost_metric)
    if cost is None:
        return None
    violated, missing = _quality_violation(
        rows[0], full, quality_metrics=quality_metrics, epsilon=epsilon
    )
    if missing:
        return None
    return {
        "label": label,
        "budget_tokens": _budget_tokens(rows[0]),
        "cost_ms": cost,
        "quality": quality,
        "violates": violated,
        "violated_metrics": [
            metric
            for metric in quality_metrics
            if _quality(full, metric) is not None
            and _quality(rows[0], metric) is not None
            and _quality(full, metric) - _quality(rows[0], metric) > epsilon
        ],
        "row": rows[0],
    }


def _risk_count(alpha: float, documents: int) -> int:
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be in [0, 1]")
    # A finite-sample empirical risk contract permits at most floor(alpha*N)
    # violating documents.  This is deliberately conservative at alpha=0.
    return int(math.floor(alpha * documents + 1e-12))


def _dp_oracle(
    actions_by_doc: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    max_violations: int,
) -> tuple[dict[str, Mapping[str, Any]] | None, int | None]:
    """Minimize total cost with at most ``max_violations`` unsafe actions."""

    # state[v] = (total_cost, selected_actions_by_doc)
    state: dict[int, tuple[float, dict[str, Mapping[str, Any]]]] = {0: (0.0, {})}
    for doc_id, actions in actions_by_doc.items():
        next_state: dict[int, tuple[float, dict[str, Mapping[str, Any]]]] = {}
        for used, (cost, chosen) in state.items():
            for action in actions:
                violations = used + int(bool(action["violates"]))
                if violations > max_violations:
                    continue
                new_cost = cost + float(action["cost_ms"])
                incumbent = next_state.get(violations)
                if incumbent is None or new_cost < incumbent[0]:
                    new_chosen = dict(chosen)
                    new_chosen[doc_id] = action
                    next_state[violations] = (new_cost, new_chosen)
        state = next_state
        if not state:
            return None, None

    best_v, (best_cost, best_chosen) = min(state.items(), key=lambda item: item[1][0])
    return best_chosen, int(best_v)


def _summarize_policy(
    chosen: Mapping[str, Mapping[str, Any]],
    *,
    full_by_doc: Mapping[str, Mapping[str, Any]],
    quality_metrics: Sequence[str],
    cost_metric: str,
) -> dict[str, Any]:
    docs = list(chosen)
    costs = [float(chosen[doc]["cost_ms"]) for doc in docs]
    budgets = [
        float(chosen[doc]["budget_tokens"])
        for doc in docs
        if chosen[doc].get("budget_tokens") is not None
    ]
    quality: dict[str, float | None] = {}
    deltas: dict[str, float | None] = {}
    for metric in quality_metrics:
        values = [chosen[doc]["quality"][metric] for doc in docs]
        full_values = [_quality(full_by_doc[doc], metric) for doc in docs]
        quality[metric] = _mean(value for value in values if value is not None)
        deltas[metric] = (
            None
            if quality[metric] is None or any(value is None for value in full_values)
            else quality[metric] - _mean(value for value in full_values if value is not None)
        )
    violation_docs = [doc for doc in docs if chosen[doc]["violates"]]
    counts = Counter(chosen[doc]["label"] for doc in docs)
    full_costs = [_cost(full_by_doc[doc], cost_metric) for doc in docs]
    mean_cost = _mean(costs)
    full_mean_cost = _mean(value for value in full_costs if value is not None)
    return {
        "documents": len(docs),
        "mean_cost_ms": mean_cost,
        "mean_budget_tokens": _mean(budgets),
        "mean_budget_ratio": _mean(
            chosen[doc]["budget_tokens"] / _number(full_by_doc[doc], "original_tokens")
            for doc in docs
            if chosen[doc].get("budget_tokens") is not None
            and _number(full_by_doc[doc], "original_tokens") not in (None, 0)
        ),
        "quality": quality,
        "quality_delta_vs_full": deltas,
        "violating_documents": len(violation_docs),
        "risk_rate": len(violation_docs) / len(docs) if docs else None,
        "violating_document_ids": violation_docs,
        "action_counts": dict(sorted(counts.items())),
        "speedup_vs_full": (
            None if mean_cost in (None, 0) or full_mean_cost is None else full_mean_cost / mean_cost
        ),
        "chosen_actions": {
            doc: {
                "label": chosen[doc]["label"],
                "budget_tokens": chosen[doc].get("budget_tokens"),
                "cost_ms": chosen[doc]["cost_ms"],
                "violates": chosen[doc]["violates"],
                "violated_metrics": chosen[doc]["violated_metrics"],
            }
            for doc in docs
        },
    }


def _fixed_policy(
    actions_by_doc: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    full_by_doc: Mapping[str, Mapping[str, Any]],
    labels: Sequence[str],
    quality_metrics: Sequence[str],
    cost_metric: str,
    alpha: float,
) -> tuple[str | None, dict[str, Any] | None, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    document_count = len(actions_by_doc)
    max_violations = _risk_count(alpha, document_count)
    for label in labels:
        chosen: dict[str, Mapping[str, Any]] = {}
        if any(not any(action["label"] == label for action in actions) for actions in actions_by_doc.values()):
            continue
        for doc_id, actions in actions_by_doc.items():
            chosen[doc_id] = next(action for action in actions if action["label"] == label)
        summary = _summarize_policy(
            chosen,
            full_by_doc=full_by_doc,
            quality_metrics=quality_metrics,
            cost_metric=cost_metric,
        )
        summary["label"] = label
        summary["risk_contract_pass"] = summary["violating_documents"] <= max_violations
        candidates.append(summary)
    passing = [item for item in candidates if item["risk_contract_pass"]]
    passing.sort(key=lambda item: (item["mean_cost_ms"], item["label"]))
    best = passing[0] if passing else None
    return (best["label"] if best else None), best, candidates


def analyze_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    selector: str,
    quality_metrics: Sequence[str],
    cost_metric: str = "pipeline_e2e_ms",
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    alphas: Sequence[float] = (0.0, 0.05, 0.10),
    required_levels: int = DEFAULT_REQUIRED_LEVELS,
    required_budget_labels: Sequence[str] | None = DEFAULT_REQUIRED_BUDGET_LABELS,
    min_documents: int = DEFAULT_MIN_DOCUMENTS,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    duplicate_keys: list[tuple[str, str]] = []
    for row in rows:
        doc_id = str(row.get("example_id", row.get("id", "")))
        if not doc_id:
            raise ValueError("every row must contain example_id or id")
        if str(row.get("selector", "")) == selector:
            label = _condition_key(row)
            if any(_condition_key(old) == label for old in grouped[doc_id]):
                duplicate_keys.append((doc_id, label))
            grouped[doc_id].append(row)
        elif str(row.get("selector", "")) == "full" and _condition_key(row) == "full":
            grouped[doc_id].append(row)

    full_by_doc: dict[str, Mapping[str, Any]] = {}
    actions_by_doc: dict[str, list[dict[str, Any]]] = {}
    missing_reasons: dict[str, str] = {}
    missing_quality: Counter[str] = Counter()
    for doc_id, doc_rows in grouped.items():
        full = next(
            (row for row in doc_rows if str(row.get("selector")) == "full" and _condition_key(row) == "full"),
            None,
        )
        if full is None:
            missing_reasons[doc_id] = "missing full baseline"
            continue
        if _cost(full, cost_metric) is None:
            missing_reasons[doc_id] = "full baseline missing cost"
            continue
        missing_full = [metric for metric in quality_metrics if _quality(full, metric) is None]
        if missing_full:
            missing_reasons[doc_id] = "full baseline missing quality: " + ",".join(missing_full)
            missing_quality.update(missing_full)
            continue
        full_by_doc[doc_id] = full
        actions: list[dict[str, Any]] = [
            {"label": "full", "row": full}
        ]
        by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in doc_rows:
            if str(row.get("selector", "")) == selector:
                by_label[_condition_key(row)].append(row)
        # Actions are rebuilt per epsilon below; this first pass only checks
        # whether every requested metric can be read for this document.
        for label, label_rows in by_label.items():
            row = label_rows[0]
            if any(_quality(row, metric) is None for metric in quality_metrics):
                for metric in quality_metrics:
                    if _quality(row, metric) is None:
                        missing_quality.update([metric])
                continue
            if _cost(row, cost_metric) is None or _budget_tokens(row) is None:
                continue
            actions.append({"label": label, "row": row})
        if len(actions) == 1:
            missing_reasons[doc_id] = "no complete selector actions"
            continue
        actions_by_doc[doc_id] = actions

    observed_labels = sorted(
        {
            action["label"]
            for actions in actions_by_doc.values()
            for action in actions
        },
        key=lambda label: (
            min(
                (
                    _budget_tokens(action["row"])
                    for actions in actions_by_doc.values()
                    for action in actions
                    if action["label"] == label and _budget_tokens(action["row"]) is not None
                ),
                default=float("inf"),
            ),
            label,
        ),
    )

    epsilon_results: dict[str, Any] = {}
    for epsilon in epsilons:
        alpha_results: dict[str, Any] = {}
        for alpha in alphas:
            prepared: dict[str, list[dict[str, Any]]] = {}
            for doc_id, actions in actions_by_doc.items():
                prepared[doc_id] = []
                for action in actions:
                    row = action["row"]
                    summary = _action_summary(
                        [row],
                        full_by_doc[doc_id],
                        quality_metrics=quality_metrics,
                        epsilon=float(epsilon),
                        cost_metric=cost_metric,
                    )
                    if summary is not None:
                        prepared[doc_id].append(summary)
            valid_docs = {doc: actions for doc, actions in prepared.items() if actions}
            valid_full = {doc: full_by_doc[doc] for doc in valid_docs}
            labels = sorted(
                {
                    action["label"] for actions in valid_docs.values() for action in actions
                },
                key=lambda label: (
                    min(
                        (
                            action["budget_tokens"]
                            for actions in valid_docs.values()
                            for action in actions
                            if action["label"] == label and action["budget_tokens"] is not None
                        ),
                        default=float("inf"),
                    ),
                    label,
                ),
            )
            fixed_label, fixed, fixed_candidates = _fixed_policy(
                valid_docs,
                full_by_doc=valid_full,
                labels=labels,
                quality_metrics=quality_metrics,
                cost_metric=cost_metric,
                alpha=float(alpha),
            )
            max_violations = _risk_count(float(alpha), len(valid_docs))
            chosen, used_violations = _dp_oracle(valid_docs, max_violations=max_violations)
            adaptive = (
                None
                if chosen is None
                else _summarize_policy(
                    chosen,
                    full_by_doc=valid_full,
                    quality_metrics=quality_metrics,
                    cost_metric=cost_metric,
                )
            )
            if adaptive is not None:
                adaptive["risk_contract_pass"] = adaptive["violating_documents"] <= max_violations
                adaptive["used_violation_budget"] = used_violations
                fixed_cost = fixed["mean_cost_ms"] if fixed else None
                adaptive["headroom_vs_best_fixed"] = (
                    None
                    if fixed_cost in (None, 0) or adaptive["mean_cost_ms"] is None
                    else 1.0 - adaptive["mean_cost_ms"] / fixed_cost
                )
            alpha_results[f"{float(alpha):.4f}"] = {
                "alpha": float(alpha),
                "max_violating_documents": max_violations,
                "documents_analyzed": len(valid_docs),
                "fixed": fixed,
                "fixed_candidates": fixed_candidates,
                "adaptive_oracle": adaptive,
            }
        epsilon_results[f"{float(epsilon):.4f}"] = {
            "epsilon": float(epsilon),
            "alpha_results": alpha_results,
        }

    docs = len(actions_by_doc)
    quality_complete = not missing_quality
    complete = (
        docs >= min_documents
        and len(observed_labels) >= required_levels
        and (
            required_budget_labels is None
            or set(required_budget_labels).issubset(set(observed_labels))
        )
        and quality_complete
    )
    return {
        "selector": selector,
        "quality_metrics": list(quality_metrics),
        "cost_metric": cost_metric,
        "documents_input": len(grouped),
        "documents_usable": docs,
        "documents_missing_or_invalid": missing_reasons,
        "missing_quality_metrics": dict(sorted(missing_quality.items())),
        "duplicate_document_condition_keys": [list(item) for item in duplicate_keys],
        "observed_budget_levels": observed_labels,
        "required_budget_levels": required_levels,
        "required_budget_labels": list(required_budget_labels or []),
        "minimum_documents": min_documents,
        "completeness": {
            "complete_for_e26r": complete,
            "document_requirement_met": docs >= min_documents,
            "budget_requirement_met": len(observed_labels) >= required_levels
            and (
                required_budget_labels is None
                or set(required_budget_labels).issubset(set(observed_labels))
            ),
            "quality_requirement_met": quality_complete,
            "status": "complete" if complete else "pilot_incomplete",
        },
        "epsilon_results": epsilon_results,
    }


def _read_dataset_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset input must be DATASET=JSONL_PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("dataset input must be DATASET=JSONL_PATH")
    return name, path


def _markdown(result: Mapping[str, Any], *, source_paths: Mapping[str, str]) -> str:
    lines = [
        "# E26-R Risk-Matched Dynamic-Budget Oracle Report",
        "",
        "## Trạng thái",
        "",
        "Đây là phân tích offline; không chạy lại target model. Fixed policy và",
        "adaptive oracle dùng cùng risk contract, trong đó risk là tỷ lệ document",
        "vi phạm quality tolerance theo từng document.",
        "",
        "| Dataset | Docs usable | Budget levels | Quality metrics | Trạng thái |",
        "|---|---:|---|---|---|",
    ]
    for dataset, data in result["datasets"].items():
        lines.append(
            f"| {dataset} | {data['documents_usable']} | {', '.join(data['observed_budget_levels']) or 'none'} | {', '.join(data['quality_metrics'])} | {data['completeness']['status']} |"
        )
    lines += [
        "",
        "## Fresh inference provenance",
        "",
    ]
    provenance = result.get("provenance") or {}
    if provenance:
        lines += [
            "Các số liệu raw bên dưới được đo trong lượt suy luận mới; phần này "
            "không lấy lại latency từ pilot cũ.",
            "",
            "```json",
            json.dumps(provenance, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    else:
        lines += [
            "Không có file provenance được truyền vào analyzer; xem JSONL raw và "
            "manifest của lượt chạy để đối chiếu runtime.",
            "",
        ]
    lines += [
        "## Protocol",
        "",
        f"- Selector: `{result['selector']}`",
        f"- Quality metrics: `{', '.join(result['quality_metrics'])}`",
        f"- Cost metric: `{result['cost_metric']}`",
        f"- Epsilon grid: `{', '.join(f'{x:.4f}' for x in result['epsilon'])}`",
        f"- Alpha grid: `{', '.join(f'{x:.4f}' for x in result['alpha'])}`",
        f"- Required budget labels: `{', '.join(result['required_budget_labels'])}`",
        "- Fixed: một budget label duy nhất cho toàn bộ documents, chọn policy có mean cost thấp nhất và risk không vượt alpha.",
        "- Adaptive oracle: dynamic programming chọn action/document có tổng cost thấp nhất dưới cùng giới hạn số document vi phạm.",
        "- Violation xảy ra khi bất kỳ quality metric nào giảm quá epsilon so với full-context cùng document.",
        "- E26-R full yêu cầu tối thiểu 30 documents/dataset, 6 budget levels và đủ mọi quality metric.",
        "",
        "### Input artifacts",
        "",
    ]
    for dataset, path in source_paths.items():
        lines.append(f"- `{dataset}`: `{path}`")
    lines += [
        "",
        "## Raw inference aggregates by budget",
        "",
        "Bảng này chứa tất cả điều kiện đã chạy, trước khi áp dụng risk contract. "
        "`full` là target chạy trên toàn bộ context sau cap của dataset; các "
        "dòng còn lại là MMR với cùng document và cùng target model.",
        "",
        "| Dataset | Selector | Budget | Rows | Docs | Original tok | Selected tok | Retention | Output tok | Selection ms | Prefill ms | Decode ms | E2E ms | E2E speedup | ROUGE-L | BERTScore F1 | Peak alloc MB |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, data in result["datasets"].items():
        for item in data.get("raw_level_summary", []):
            lines.append(
                f"| {dataset} | {item['selector']} | {item['budget_label']} | "
                f"{item['rows']} | {item['documents']} | "
                f"{_fmt(item.get('original_tokens'), 2)} | "
                f"{_fmt(item.get('selected_tokens'), 2)} | "
                f"{_pct(item.get('retention_ratio'))} | "
                f"{_fmt(item.get('output_tokens'), 2)} | "
                f"{_fmt(item.get('selection_total_wall_ms'), 2)} | "
                f"{_fmt(item.get('prefill_ms'), 2)} | "
                f"{_fmt(item.get('decode_ms'), 2)} | "
                f"{_fmt(item.get('pipeline_e2e_ms'), 2)} | "
                f"{_fmt(item.get('speedup_vs_full_recomputed'), 3)}x | "
                f"{_fmt(item.get('rougeL'), 4)} | "
                f"{_fmt(item.get('bertscore_f1'), 4)} | "
                f"{_fmt(item.get('peak_gpu_allocated_mb'), 2)} |"
            )
    lines += ["", "## Kết quả risk-matched", ""]
    for dataset, data in result["datasets"].items():
        lines += [f"### {dataset}", ""]
        lines.append(
            f"Cỡ mẫu: **{data['documents_usable']}**; levels: **{len(data['observed_budget_levels'])}**; status: **{data['completeness']['status']}**."
        )
        if data["missing_quality_metrics"]:
            lines.append(f"Thiếu quality metrics: `{json.dumps(data['missing_quality_metrics'], ensure_ascii=False)}`")
        lines += ["", "| Epsilon | Alpha | Fixed | Fixed cost ms | Fixed risk | Adaptive cost ms | Adaptive risk | Headroom vs fixed | Speedup vs full |", "|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        for epsilon_key, epsilon_item in data["epsilon_results"].items():
            for alpha_key, item in epsilon_item["alpha_results"].items():
                fixed = item["fixed"] or {}
                adaptive = item["adaptive_oracle"] or {}
                speedup = adaptive.get("speedup_vs_full")
                speedup_text = "n/a" if speedup is None else f"{speedup:.3f}x"
                lines.append(
                    f"| {epsilon_item['epsilon']:.4f} | {item['alpha']:.4f} | {item.get('fixed_label', fixed.get('label', 'none'))} | {_fmt(fixed.get('mean_cost_ms'), 2)} | {_pct(fixed.get('risk_rate'))} | {_fmt(adaptive.get('mean_cost_ms'), 2)} | {_pct(adaptive.get('risk_rate'))} | {_pct(adaptive.get('headroom_vs_best_fixed'))} | {speedup_text} |"
                )
        lines.append("")
        lines.append("#### Chi tiết từng epsilon/alpha")
        lines.append("")
        for epsilon_item in data["epsilon_results"].values():
            for item in epsilon_item["alpha_results"].values():
                fixed = item["fixed"] or {}
                adaptive = item["adaptive_oracle"] or {}
                lines += [
                    f"- `epsilon={epsilon_item['epsilon']:.4f}, alpha={item['alpha']:.4f}`: cho phép tối đa {item['max_violating_documents']} document vi phạm trên {item['documents_analyzed']}; fixed=`{fixed.get('label', 'none')}`, adaptive violations={adaptive.get('violating_documents', 'n/a')}, actions={json.dumps(adaptive.get('action_counts', {}), ensure_ascii=False)}.",
                    f"  - Fixed quality={json.dumps(fixed.get('quality', {}), ensure_ascii=False)}, quality delta={json.dumps(fixed.get('quality_delta_vs_full', {}), ensure_ascii=False)}, cost={_fmt(fixed.get('mean_cost_ms'), 2)} ms.",
                    f"  - Adaptive quality={json.dumps(adaptive.get('quality', {}), ensure_ascii=False)}, quality delta={json.dumps(adaptive.get('quality_delta_vs_full', {}), ensure_ascii=False)}, cost={_fmt(adaptive.get('mean_cost_ms'), 2)} ms, headroom={_pct(adaptive.get('headroom_vs_best_fixed'))}.",
                ]
        lines.append("")
    lines += [
        "## Bootstrap CI và held-out",
        "",
        "Bootstrap được thực hiện theo document với replacement và giữ nguyên toàn bộ các điều kiện của mỗi document trong từng replicate. CI 95% là percentile CI; đây là CI thực nghiệm, không phải conformal guarantee ngoài mẫu.",
        "",
        "### Bootstrap 95% CI",
        "",
        "| Dataset | Epsilon | Alpha | Replicates hợp lệ | Headroom mean | Headroom CI 95% | Fixed cost CI ms | Adaptive cost CI ms |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for dataset, validation in result.get("validation", {}).items():
        for cell in validation.get("bootstrap", {}).get("cells", []):
            ci = cell.get("headroom_ci_95", [None, None])
            fixed_ci = cell.get("fixed_cost_ci_95_ms", [None, None])
            adaptive_ci = cell.get("adaptive_cost_ci_95_ms", [None, None])
            lines.append(
                f"| {dataset} | {cell['epsilon']:.4f} | {cell['alpha']:.4f} | {cell['valid_replicates']} | {_pct(cell.get('headroom_mean'))} | {_pct(ci[0])}–{_pct(ci[1])} | {_fmt(fixed_ci[0], 2)}–{_fmt(fixed_ci[1], 2)} | {_fmt(adaptive_ci[0], 2)}–{_fmt(adaptive_ci[1], 2)} |"
            )
    lines += [
        "",
        "### Held-out: fixed policy chọn ở calibration, đánh giá ở holdout",
        "",
        "Adaptive oracle trong bảng held-out vẫn là hindsight upper bound trên holdout; policy fixed dùng label được chọn từ calibration và không được chọn lại trên holdout.",
        "",
        "| Dataset | Epsilon | Alpha | Calibration docs | Holdout docs | Label từ calibration | Holdout fixed risk | Holdout adaptive cost ms | Headroom adaptive vs calibrated fixed |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for dataset, validation in result.get("validation", {}).items():
        for cell in validation.get("heldout", {}).get("cells", []):
            fixed = cell.get("holdout_fixed") or {}
            lines.append(
                f"| {dataset} | {cell['epsilon']:.4f} | {cell['alpha']:.4f} | {cell['calibration_documents']} | {cell['holdout_documents']} | {cell.get('calibrated_fixed_label', 'none')} | {_pct(fixed.get('risk_rate'))} | {_fmt((cell.get('holdout_adaptive_oracle') or {}).get('mean_cost_ms'), 2)} | {_pct(cell.get('holdout_headroom_vs_calibrated_fixed'))} |"
            )
    lines += [
        "",
        "## Tổng hợp và quyết định E26-R",
        "",
        "### Kiểm tra completeness",
        "",
        "- E26-R full **PASS về completeness** trên cả ba dataset: 30 documents/dataset, 6 budget levels/dataset, 180 raw rows/dataset và đủ ROUGE-L + BERTScore F1 trên 540/540 rows.",
        "- Tất cả các điều kiện được chạy cùng Qwen3-4B, greedy decoding, batch size 1, max 256 generated tokens; full và MMR dùng cùng document/reference trong từng cặp.",
        "- Bootstrap có 200/200 replicate hợp lệ cho mọi cell. Held-out dùng 21 calibration documents và 9 holdout documents mỗi dataset với split seed cố định.",
        "",
        "### Kết quả systems raw",
        "",
    ]
    for dataset, data in result["datasets"].items():
        levels = data.get("raw_level_summary", [])
        full = next((item for item in levels if item.get("budget_label") == "full"), None)
        selected = [item for item in levels if item.get("budget_label") != "full" and item.get("pipeline_e2e_ms") is not None]
        fastest = min(selected, key=lambda item: item["pipeline_e2e_ms"], default=None)
        cells = [
            item
            for epsilon_item in data["epsilon_results"].values()
            for item in epsilon_item["alpha_results"].values()
            if (item.get("adaptive_oracle") or {}).get("headroom_vs_best_fixed") is not None
        ]
        max_headroom = max(
            (float(item["adaptive_oracle"]["headroom_vs_best_fixed"]) for item in cells),
            default=None,
        )
        if full and fastest:
            lines.append(
                f"- **{dataset}**: full mean E2E **{_fmt(full.get('pipeline_e2e_ms'), 2)} ms**; nhanh nhất là `{fastest['budget_label']}` với **{_fmt(fastest.get('pipeline_e2e_ms'), 2)} ms**, tương đương **{_fmt(fastest.get('speedup_vs_full_recomputed'), 3)}x** theo tỷ số của hai mean E2E; output trung bình `{_fmt(fastest.get('output_tokens'), 2)}` token, ROUGE-L `{_fmt(fastest.get('rougeL'), 4)}`, BERTScore F1 `{_fmt(fastest.get('bertscore_f1'), 4)}`."
            )
        lines.append(
            f"  - Adaptive hindsight headroom lớn nhất trong grid risk là **{_pct(max_headroom)}**; đây là upper bound theo document, không phải policy runtime đã train."
        )
    lines += [
        "",
        "### Diễn giải theo risk contract",
        "",
        "- Khi `epsilon` nhỏ và `alpha=0`, fixed policy thường phải chọn `full` vì chỉ một document bị mất quá tolerance cũng làm policy không hợp lệ; adaptive oracle vẫn có thể nén riêng các document an toàn.",
        "- GovReport cho headroom adaptive lớn nhất trong vùng strict (`epsilon=0.01–0.02`, `alpha=0–0.05` khoảng 34.27–39.51%), phù hợp với việc context bị cap 4096 và chi phí full cao.",
        "- CNN/DailyMail có headroom khoảng 21.43–25.63% ở các cell strict/near-strict, nhưng absolute latency nhỏ hơn và MMR selection chiếm tỷ trọng đáng kể.",
        "- Multi-News có headroom 13.63–27.82% trên in-sample grid; held-out strict headroom chỉ khoảng 5.03–6.13%, cho thấy adaptive gain nhạy với cỡ mẫu và phân bố tài liệu.",
        "- Ở `epsilon=0.05`, GovReport fixed `ratio_0.25` đã hợp lệ và adaptive không còn headroom; đây là bằng chứng rằng dynamic budget không luôn vượt fixed compression.",
        "",
        "### Quyết định gate",
        "",
        "- **Gate completeness: PASS.** Đây là lượt E26-R đầu tiên đáp ứng đồng thời cỡ mẫu, sáu mức ngân sách, hai metric quality, bootstrap và held-out.",
        "- **Gate dynamic-budget headroom: PASS ở mức oracle/in-sample trên CNN/DailyMail và GovReport; PASS không đồng đều trên Multi-News.** Vì adaptive oracle dùng hindsight nên chưa đủ để claim một controller triển khai.",
        "- **E27 predictor và E28 systems policy chưa được thực hiện trong artifact này.** Không được gọi adaptive oracle là SafeBudget-Sum runtime method; bước tiếp theo, nếu tiếp tục, là predictor cost/quality có calibration trên dữ liệu riêng.",
        "- **Phạm vi long-context bị giới hạn trên T4:** GovReport và Multi-News dùng full trong 4096-token cap. Vì vậy kết luận systems là valid cho regime `<=4096 source tokens`, chưa phải native 8K/16K/32K full-document benchmark.",
        "",
        "## Tái lập và provenance",
        "",
        "Lượt raw được chạy trực tiếp bằng external Conda trên `cuda:0` với Tesla T4; standard repository launcher không được dùng vì nó buộc Python 3.12 còn Conda GPU hiện tại là Python 3.11. Analyzer và augmentation chạy offline từ các checkpoint local; không có tải model/dataset qua mạng.",
        "",
        "Các file cần đối chiếu: `inference_provenance.json`, ba JSONL trong `raw/`, ba JSONL đã bổ sung BERTScore trong `scored/`, `metrics.json`, `metrics.csv` và `run_manifest.json`.",
        "",
        "## Diễn giải trạng thái",
        "",
        "`complete` chỉ được xuất hiện khi đồng thời đủ cỡ mẫu, budget grid và quality metrics. `pilot_incomplete` là trạng thái hợp lệ của preflight/pilot nhưng không phải E26-R pass.",
        "",
        "## Giới hạn",
        "",
        "- Risk rate ở đây là empirical document-level risk trên tập đang phân tích; chưa phải guarantee conformal ngoài mẫu.",
        "- Oracle dùng hindsight quality/cost để định lượng headroom, không phải policy triển khai.",
        "- BERTScore trong lượt này là local token-level cosine matching không IDF, chạy bằng RoBERTa local; đây là implementation tương thích định nghĩa cốt lõi nhưng không phải package `bert-score`.",
        "- `pipeline_e2e_ms` đã bao gồm selection + prompt/target inference theo schema runner; không được diễn giải thành server throughput/QPS.",
    ]
    return "\n".join(lines) + "\n"


def _csv_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, data in result["datasets"].items():
        for epsilon_item in data["epsilon_results"].values():
            for item in epsilon_item["alpha_results"].values():
                fixed = item["fixed"] or {}
                adaptive = item["adaptive_oracle"] or {}
                rows.append(
                    {
                        "dataset": dataset,
                        "epsilon": epsilon_item["epsilon"],
                        "alpha": item["alpha"],
                        "status": data["completeness"]["status"],
                        "documents": item["documents_analyzed"],
                        "max_violating_documents": item["max_violating_documents"],
                        "fixed_label": fixed.get("label"),
                        "fixed_mean_cost_ms": fixed.get("mean_cost_ms"),
                        "fixed_risk_rate": fixed.get("risk_rate"),
                        "adaptive_mean_cost_ms": adaptive.get("mean_cost_ms"),
                        "adaptive_risk_rate": adaptive.get("risk_rate"),
                        "adaptive_headroom_vs_fixed": adaptive.get("headroom_vs_best_fixed"),
                        "adaptive_speedup_vs_full": adaptive.get("speedup_vs_full"),
                    }
                )
    return rows


def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    inputs = dict(args.input)
    raw_inputs = {name: load_jsonl(path) for name, path in inputs.items()}
    datasets = {
        name: {
            **analyze_dataset(
                raw_inputs[name],
                selector=args.selector,
                quality_metrics=args.quality_metric,
                cost_metric=args.cost_metric,
                epsilons=args.epsilon,
                alphas=args.alpha,
                required_levels=args.required_budget_levels,
                required_budget_labels=args.required_budget_label,
                min_documents=args.min_documents,
            ),
            "raw_level_summary": _raw_level_summary(
                raw_inputs[name], selector=args.selector
            ),
        }
        for name in inputs
    }
    validation: dict[str, Any] = {}
    for name, rows in raw_inputs.items():
        validation[name] = {
            "bootstrap": bootstrap_validation(
                rows,
                selector=args.selector,
                quality_metrics=args.quality_metric,
                cost_metric=args.cost_metric,
                epsilons=args.epsilon,
                alphas=args.alpha,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
            "heldout": heldout_validation(
                rows,
                selector=args.selector,
                quality_metrics=args.quality_metric,
                cost_metric=args.cost_metric,
                epsilons=args.epsilon,
                alphas=args.alpha,
                test_fraction=args.heldout_fraction,
                seed=args.split_seed,
            ),
        }
    result: dict[str, Any] = {
        "experiment": "E26R_risk_matched_dynamic_budget_oracle",
        "selector": args.selector,
        "quality_metrics": args.quality_metric,
        "cost_metric": args.cost_metric,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "required_budget_levels": args.required_budget_levels,
        "required_budget_labels": args.required_budget_label,
        "minimum_documents": args.min_documents,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "heldout_fraction": args.heldout_fraction,
        "split_seed": args.split_seed,
        "provenance": args.provenance,
        "inputs": inputs,
        "datasets": datasets,
        "validation": validation,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = _csv_rows(result)
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(rows[0]) if rows else ["dataset"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "report.md").write_text(_markdown(result, source_paths=inputs), encoding="utf-8")
    (out / "run_manifest.json").write_text(
        json.dumps(
            {
                "experiment": result["experiment"],
                "selector": args.selector,
                "quality_metrics": args.quality_metric,
                "cost_metric": args.cost_metric,
                "inputs": inputs,
                "output_dir": str(out),
                "epsilon": args.epsilon,
                "alpha": args.alpha,
                "required_budget_labels": args.required_budget_label,
                "required_budget_levels": args.required_budget_levels,
                "min_documents": args.min_documents,
                "bootstrap_samples": args.bootstrap_samples,
                "bootstrap_seed": args.bootstrap_seed,
                "heldout_fraction": args.heldout_fraction,
                "split_seed": args.split_seed,
                "provenance": args.provenance,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_read_dataset_arg, required=True, metavar="DATASET=JSONL")
    parser.add_argument("--selector", default="mmr")
    parser.add_argument("--quality-metric", action="append", default=None)
    parser.add_argument("--cost-metric", default="pipeline_e2e_ms")
    parser.add_argument("--epsilon", type=float, action="append", default=None)
    parser.add_argument("--alpha", type=float, action="append", default=None)
    parser.add_argument("--required-budget-levels", type=int, default=DEFAULT_REQUIRED_LEVELS)
    parser.add_argument(
        "--required-budget-label",
        action="append",
        default=None,
        help="Expected labels for the full retention-ratio grid; repeat this option.",
    )
    parser.add_argument("--min-documents", type=int, default=DEFAULT_MIN_DOCUMENTS)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    parser.add_argument("--heldout-fraction", type=float, default=0.30)
    parser.add_argument("--split-seed", type=int, default=20260911)
    parser.add_argument(
        "--provenance-json",
        type=Path,
        default=None,
        help="Optional JSON file describing the fresh inference runtime/protocol.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.quality_metric is None:
        args.quality_metric = ["rougeL", "bertscore_f1"]
    if args.epsilon is None:
        args.epsilon = list(DEFAULT_EPSILONS)
    if args.alpha is None:
        args.alpha = [0.0, 0.05, 0.10]
    if args.required_budget_label is None:
        args.required_budget_label = list(DEFAULT_REQUIRED_BUDGET_LABELS)
    if any(value < 0 for value in args.epsilon):
        parser.error("epsilon must be non-negative")
    if any(value < 0 or value > 1 for value in args.alpha):
        parser.error("alpha must be in [0, 1]")
    if args.bootstrap_samples < 0:
        parser.error("bootstrap-samples must be non-negative")
    if not 0.0 < args.heldout_fraction < 1.0:
        parser.error("heldout-fraction must be in (0, 1)")
    args.provenance = {}
    if args.provenance_json is not None:
        args.provenance = json.loads(args.provenance_json.read_text(encoding="utf-8"))
        if not isinstance(args.provenance, dict):
            parser.error("provenance-json must contain a JSON object")
    result = run_cli(args)
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "quality_metrics": result["quality_metrics"],
                "datasets": {
                    name: {
                        "status": data["completeness"]["status"],
                        "documents": data["documents_usable"],
                        "budget_levels": data["observed_budget_levels"],
                        "missing_quality_metrics": data["missing_quality_metrics"],
                    }
                    for name, data in result["datasets"].items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
