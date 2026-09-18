"""Offline RECAP-KV ranking metrics and scientific gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from .policy import RecapConfig, RecapState
from .schema import validate_trace_record


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _row(values: Sequence[float], width: int | None = None) -> list[float]:
    result = [_finite(value, "attention") for value in values]
    if not result or (width is not None and len(result) != width):
        raise ValueError("attention rows must be non-empty and equally wide")
    if any(value < 0.0 for value in result) or sum(result) <= 0.0:
        raise ValueError("attention rows must have positive non-negative mass")
    total = sum(result)
    return [value / total for value in result]


def future_use_matrix(attention_by_step: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return mean source use in the strict future after every current step."""

    if not attention_by_step:
        raise ValueError("attention_by_step must not be empty")
    width = len(attention_by_step[0])
    rows = [_row(row, width) for row in attention_by_step]
    result: list[list[float]] = []
    for start in range(len(rows)):
        future = rows[start + 1 :]
        if not future:
            result.append([0.0] * width)
            continue
        result.append([
            sum(row[index] for row in future) / len(future)
            for index in range(width)
        ])
    return result


def _ranking(scores: Sequence[float]) -> list[int]:
    values = [_finite(value, "scores") for value in scores]
    if not values:
        raise ValueError("scores must not be empty")
    return [index for index, _ in sorted(enumerate(values), key=lambda item: (-item[1], item[0]))]


def ranking_metrics(
    scores: Sequence[float], oracle: Sequence[float], *, k: int
) -> dict[str, float]:
    """Compute Recall@K and continuous-gain NDCG@K against future use."""

    if len(scores) != len(oracle) or not scores:
        raise ValueError("scores and oracle must be non-empty and aligned")
    if k <= 0:
        raise ValueError("k must be positive")
    oracle_values = [_finite(value, "oracle") for value in oracle]
    if any(value < 0.0 for value in oracle_values):
        raise ValueError("oracle must be non-negative")
    width = len(scores)
    k = min(int(k), width)
    predicted = _ranking(scores)[:k]
    ideal = _ranking(oracle_values)[:k]
    oracle_mass = sum(oracle_values[index] for index in ideal)
    recall = (
        sum(oracle_values[index] for index in predicted) / oracle_mass
        if oracle_mass > 0.0
        else 0.0
    )

    def discounted(indices: Sequence[int]) -> float:
        return sum(
            oracle_values[index] / math.log2(position + 2)
            for position, index in enumerate(indices)
        )

    ideal_dcg = discounted(ideal)
    ndcg = discounted(predicted) / ideal_dcg if ideal_dcg > 0.0 else 0.0
    return {
        "recall_at_k": float(recall),
        "ndcg_at_k": float(ndcg),
        "predicted_top": int(predicted[0]),
        "oracle_top": int(ideal[0]),
    }


@dataclass(frozen=True)
class EvaluationConfig:
    policy: RecapConfig = RecapConfig()
    budgets: tuple[int, ...] = (1, 2, 4, 8)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_trace(record: Mapping[str, Any], config: EvaluationConfig) -> dict[str, Any]:
    """Evaluate all online policies on one trace without oracle leakage."""

    trace = validate_trace_record(record)
    if trace["status"] != "ok":
        return {"status": "error", "sample_id": trace["sample_id"], "error": trace["error"]}
    blocks = trace["source_blocks"]
    segments = trace["segments"]
    attention = [segment["attention"] for segment in segments]
    oracle_rows = future_use_matrix(attention)
    state = RecapState(config.policy, [1.0] * len(blocks))
    attention_state = RecapState(config.policy, [1.0] * len(blocks))
    accumulators: dict[str, dict[str, dict[str, list[float]]]] = {
        str(k): {
            policy: {"recall": [], "ndcg": []}
            for policy in ("current", "historical", "recap", "recap_attention")
        }
        for k in config.budgets
    }
    historical: list[float] = [0.0] * len(blocks)
    evaluated_steps = 0
    for index, segment in enumerate(segments):
        current = [float(value) for value in segment["attention"]]
        historical = [old + new for old, new in zip(historical, current)]
        recap = state.step(
            segment["query_vectors"],
            segment["segment_vector"],
            [block["embedding"] for block in blocks],
            [block["prototypes"] for block in blocks],
        )
        recap_attention = attention_state.step_with_relevance(
            current,
            segment["segment_vector"],
            [block["embedding"] for block in blocks],
        )
        oracle = oracle_rows[index]
        if index == len(segments) - 1 or sum(oracle) <= 0.0:
            continue
        evaluated_steps += 1
        for budget in config.budgets:
            for policy, scores in (
                ("current", current),
                ("historical", historical),
                ("recap", recap["utility"]),
                ("recap_attention", recap_attention["utility"]),
            ):
                metrics = ranking_metrics(scores, oracle, k=budget)
                accumulators[str(budget)][policy]["recall"].append(metrics["recall_at_k"])
                accumulators[str(budget)][policy]["ndcg"].append(metrics["ndcg_at_k"])

    if evaluated_steps == 0:
        return {
            "status": "inconclusive",
            "sample_id": trace["sample_id"],
            "dataset": trace.get("dataset", "unknown"),
            "steps": 0,
            "metrics": {},
            "reason": "trace_has_no_strict_future_segment",
        }
    metrics: dict[str, Any] = {}
    for budget, policies in accumulators.items():
        metrics[budget] = {}
        for policy, values in policies.items():
            metrics[budget][policy] = {
                "count": len(values["recall"]),
                "recall_at_k": _mean(values["recall"]),
                "ndcg_at_k": _mean(values["ndcg"]),
            }
    return {
        "status": "ok",
        "sample_id": trace["sample_id"],
        "dataset": trace.get("dataset", "unknown"),
        "steps": evaluated_steps,
        "metrics": metrics,
    }


def aggregate_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate documents and apply the preregistered minimum-data gate."""

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    by_dataset: dict[str, int] = {}
    for row in ok_rows:
        dataset = str(row.get("dataset", "unknown"))
        by_dataset[dataset] = by_dataset.get(dataset, 0) + 1
    if len(by_dataset) < 2 or min(by_dataset.values(), default=0) < 10:
        return {
            "status": "INCONCLUSIVE",
            "reason": "minimum_dataset_or_document_gate_not_met",
            "datasets": by_dataset,
            "ok_documents": len(ok_rows),
        }

    budgets = sorted({budget for row in ok_rows for budget in row.get("metrics", {})}, key=int)
    comparison: dict[str, Any] = {}
    improvements: list[bool] = []
    for budget in budgets:
        comparison[budget] = {}
        for metric in ("recall_at_k", "ndcg_at_k"):
            current = _mean([
                float(row["metrics"][budget]["current"][metric])
                for row in ok_rows
                if budget in row.get("metrics", {})
            ])
            recap = _mean([
                float(row["metrics"][budget]["recap"][metric])
                for row in ok_rows
                if budget in row.get("metrics", {})
            ])
            ablations = {
                policy: _mean([
                    float(row["metrics"][budget][policy][metric])
                    for row in ok_rows
                    if budget in row.get("metrics", {}) and policy in row["metrics"][budget]
                ])
                for policy in ("historical", "recap_attention")
            }
            comparison[budget][metric] = {
                "current": current,
                "recap": recap,
                "delta": recap - current,
                "ablations": {
                    policy: {"value": value, "delta": value - current}
                    for policy, value in ablations.items()
                },
            }
            improvements.append(recap > current)
    supported = bool(comparison) and all(
        values[metric]["delta"] >= 0.0
        for values in comparison.values()
        for metric in ("recall_at_k", "ndcg_at_k")
    ) and any(improvements)
    return {
        "status": "RECAP_SUPPORTED" if supported else "GATE_FAIL",
        "datasets": by_dataset,
        "ok_documents": len(ok_rows),
        "comparison": comparison,
    }
