"""Shared, dependency-light utilities for Target-KV E0/E1 experiments.

The model runners live in the CLI modules next to this file.  This module is
kept free of Transformers and CUDA imports so that data contracts, acceptance
semantics, and statistical aggregation can be tested on CPU.
"""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence


CONTEXT_BUCKETS = ("0-2k", "2-4k", "4-8k", "8-16k", "16-32k", "32-40k", ">40k")


def context_bucket(input_tokens: int, *, max_position_embeddings: int = 40960) -> str:
    """Return the natural context-length bucket for an input token count."""

    tokens = int(input_tokens)
    if tokens < 0:
        raise ValueError("input_tokens must be non-negative")
    if tokens > max_position_embeddings:
        return ">40k" if max_position_embeddings == 40960 else f">{max_position_embeddings}"
    if tokens <= 2048:
        return "0-2k"
    if tokens <= 4096:
        return "2-4k"
    if tokens <= 8192:
        return "4-8k"
    if tokens <= 16384:
        return "8-16k"
    if tokens <= 32768:
        return "16-32k"
    return "32-40k"


def prepare_record_metadata(
    record: Mapping[str, Any],
    *,
    max_position_embeddings: int = 40960,
) -> dict[str, Any]:
    """Validate a data row without truncating it silently."""

    raw_length = record.get("input_tokens")
    if raw_length is None:
        raise ValueError("record must contain input_tokens")
    try:
        input_tokens = int(raw_length)
    except (TypeError, ValueError) as exc:
        raise ValueError("input_tokens must be an integer") from exc
    if input_tokens < 0:
        raise ValueError("input_tokens must be non-negative")
    sample_id = str(record.get("id", record.get("sample_id", "")))
    dataset = str(record.get("dataset", "unknown"))
    result = {
        "sample_id": sample_id,
        "document_id": sample_id,
        "dataset": dataset,
        "input_tokens": input_tokens,
        "context_bucket": context_bucket(
            input_tokens, max_position_embeddings=max_position_embeddings
        ),
    }
    if input_tokens > max_position_embeddings:
        result.update(
            status="excluded",
            exclusion_reason="input_exceeds_model_limit",
        )
    else:
        result["status"] = "ok"
    return result


def apply_input_length_limit(metadata: Mapping[str, Any], limit: int) -> dict[str, Any]:
    """Mark rows over an experimental input cap; never truncate their label."""

    result = dict(metadata)
    if limit <= 0:
        return result
    if int(result.get("input_tokens", 0)) > limit:
        result.update(status="excluded", exclusion_reason="t4_input_cap")
    return result


def dflash_acceptance_to_draft_tokens(raw_length: int, *, block_size: int) -> int:
    """Convert DFlash's fallback-inclusive length to accepted draft tokens.

    DFlash records ``accepted_draft_tokens + 1`` because the target fallback
    token is always committed.  The E0 survival variable A counts draft
    tokens only, so the fallback must be removed before analysis.
    """

    length = int(raw_length)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 1 <= length <= block_size + 1:
        raise ValueError("DFlash acceptance length is outside its valid range")
    return length - 1


def first_rejection_position(raw_length: int, *, block_size: int) -> int | None:
    """Return the one-indexed first rejected draft position, or censoring None."""

    accepted = dflash_acceptance_to_draft_tokens(raw_length, block_size=block_size)
    return None if accepted >= block_size else accepted + 1


def flatten_dflash_rounds(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand one DFlash output record into one row per speculative round."""

    if record.get("status", "ok") != "ok":
        return []
    raw_lengths = record.get("acceptance_lengths")
    if not isinstance(raw_lengths, list):
        raise ValueError("record must contain acceptance_lengths list")
    block_size = int(record.get("block_size", 0))
    input_tokens = int(record["input_tokens"])
    metadata = {
        "sample_id": str(record.get("sample_id", record.get("document_id", ""))),
        "document_id": str(record.get("document_id", record.get("sample_id", ""))),
        "dataset": str(record.get("dataset", "unknown")),
        "input_tokens": input_tokens,
        "context_bucket": record.get("context_bucket") or context_bucket(input_tokens),
        "block_size": block_size,
        "exact_match_target_ar": bool(record.get("exact_match_target_ar", False)),
    }
    rows: list[dict[str, Any]] = []
    for round_index, raw_length in enumerate(raw_lengths):
        accepted = dflash_acceptance_to_draft_tokens(raw_length, block_size=block_size)
        rows.append(
            {
                **metadata,
                "round_index": round_index,
                "raw_acceptance_length": int(raw_length),
                "accepted_draft_tokens": accepted,
                "first_rejection_rel": first_rejection_position(
                    raw_length, block_size=block_size
                ),
                "fully_accepted": accepted >= block_size,
            }
        )
    return rows


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_document_mean_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    samples: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap a mean after giving every document equal weight."""

    if not rows:
        raise ValueError("rows must not be empty")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if "document_id" not in row or value_key not in row:
            raise ValueError("row must contain document_id and requested value")
        value = float(row[value_key])
        if not math.isfinite(value):
            raise ValueError("bootstrap values must be finite")
        grouped[str(row["document_id"])].append(value)
    document_values = [statistics.fmean(values) for values in grouped.values()]
    observed = statistics.fmean(document_values)
    rng = random.Random(seed)
    bootstrap_means = [
        statistics.fmean(rng.choice(document_values) for _ in document_values)
        for _ in range(samples)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": observed,
        "ci_low": _percentile(bootstrap_means, alpha),
        "ci_high": _percentile(bootstrap_means, 1.0 - alpha),
        "document_count": len(document_values),
        "row_count": len(rows),
        "bootstrap_samples": samples,
        "confidence": confidence,
    }


def _bucket_rows(rows: Sequence[Mapping[str, Any]], bucket: str, k: int) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if str(row.get("context_bucket")) == bucket
        and int(row.get("block_size", 0)) == k
    ]


def _survival_rows(rows: Sequence[Mapping[str, Any]], position: int) -> list[dict[str, Any]]:
    return [
        {**row, "survived": float(int(row["accepted_draft_tokens"]) >= position)}
        for row in rows
    ]


def _context_side(rows: Sequence[Mapping[str, Any]], side: str) -> list[dict[str, Any]]:
    short = {"0-2k", "2-4k"}
    long = {"8-16k", "16-32k", "32-40k"}
    allowed = short if side == "short" else long
    return [dict(row) for row in rows if str(row.get("context_bucket")) in allowed]


def aggregate_e0_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_ks: Sequence[int] = (4, 8, 16),
    bootstrap_samples: int = 2000,
    min_documents_for_gate: int = 4,
) -> dict[str, Any]:
    """Aggregate E0 survival and a conservative long-context gate."""

    if not rows:
        raise ValueError("E0 rows must not be empty")
    by_k: dict[str, Any] = {}
    context_drop: dict[str, Any] = {}
    for raw_k in candidate_ks:
        k = int(raw_k)
        if k <= 0:
            raise ValueError("candidate K must be positive")
        k_rows = [dict(row) for row in rows if int(row.get("block_size", 0)) == k]
        by_bucket: dict[str, Any] = {}
        for bucket in CONTEXT_BUCKETS:
            bucket_rows = _bucket_rows(k_rows, bucket, k)
            if not bucket_rows:
                continue
            survival: dict[str, float] = {}
            survival_ci: dict[str, Any] = {}
            for position in range(1, k + 1):
                survival_rows = _survival_rows(bucket_rows, position)
                ci = bootstrap_document_mean_ci(
                    survival_rows,
                    value_key="survived",
                    samples=bootstrap_samples,
                )
                survival[str(position)] = ci["mean"]
                survival_ci[str(position)] = ci
            mat_ci = bootstrap_document_mean_ci(
                bucket_rows,
                value_key="accepted_draft_tokens",
                samples=bootstrap_samples,
            )
            by_bucket[bucket] = {
                "row_count": len(bucket_rows),
                "document_count": len({str(row["document_id"]) for row in bucket_rows}),
                "mat": mat_ci["mean"],
                "mat_ci": mat_ci,
                "survival": survival,
                "survival_ci": survival_ci,
            }
        by_k[str(k)] = {"row_count": len(k_rows), "by_bucket": by_bucket}

        position = min(8, k)
        short_rows = _survival_rows(_context_side(k_rows, "short"), position)
        long_rows = _survival_rows(_context_side(k_rows, "long"), position)
        short_ci = (
            bootstrap_document_mean_ci(
                short_rows, value_key="survived", samples=bootstrap_samples
            )
            if short_rows
            else None
        )
        long_ci = (
            bootstrap_document_mean_ci(
                long_rows, value_key="survived", samples=bootstrap_samples
            )
            if long_rows
            else None
        )
        if short_ci is None or long_ci is None or short_ci["mean"] <= 0.0:
            context_drop[str(k)] = {
                "status": "INCONCLUSIVE",
                "position": position,
                "short": short_ci,
                "long": long_ci,
                "relative_drop": None,
            }
        else:
            relative_drop = (short_ci["mean"] - long_ci["mean"]) / short_ci["mean"]
            enough = min(short_ci["document_count"], long_ci["document_count"]) >= min_documents_for_gate
            context_drop[str(k)] = {
                "status": "PASS" if enough and relative_drop >= 0.15 else "FAIL" if enough else "INCONCLUSIVE",
                "position": position,
                "short": short_ci,
                "long": long_ci,
                "relative_drop": relative_drop,
            }

    gate_candidates = list(context_drop.values())
    if any(value.get("status") == "PASS" for value in gate_candidates):
        decision = {"status": "PASS", "reason": "long_context_survival_drop_ge_15pct"}
    elif any(value.get("status") == "INCONCLUSIVE" for value in gate_candidates):
        decision = {"status": "INCONCLUSIVE", "reason": "insufficient_natural_bucket_coverage"}
    else:
        decision = {"status": "FAIL", "reason": "no_long_context_survival_drop"}
    return {
        "row_count": len(rows),
        "document_count": len({str(row["document_id"]) for row in rows}),
        "by_k": by_k,
        "context_drop": context_drop,
        "decision": decision,
    }


def representation_parameter_audit(parameters: Mapping[str, int]) -> dict[str, Any]:
    """Require exactly matched trainable budgets across E1 representations."""

    if not parameters:
        raise ValueError("parameter budget must not be empty")
    values = {str(name): int(value) for name, value in parameters.items()}
    if any(value < 0 for value in values.values()):
        raise ValueError("parameter budget must be non-negative")
    if len(set(values.values())) != 1:
        raise ValueError("representation parameter budget must be equal")
    only = next(iter(values.values()))
    return {"equal": True, "min": only, "max": only, "by_representation": values}
