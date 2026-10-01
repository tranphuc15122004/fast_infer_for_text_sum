"""ROUGE metric scoring."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _ngram_counts(tokens: list[str], n: int) -> dict[tuple[str, ...], int]:
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _rouge_n(pred_tokens: list[str], ref_tokens: list[str], n: int) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = _ngram_counts(pred_tokens, n)
    ref_counts = _ngram_counts(ref_tokens, n)
    if not ref_counts:
        return 0.0

    overlap = sum(
        min(count, ref_counts.get(ng, 0)) for ng, count in pred_counts.items()
    )
    total_ref = sum(ref_counts.values())
    total_pred = sum(pred_counts.values())

    recall = overlap / total_ref if total_ref > 0 else 0.0
    precision = overlap / total_pred if total_pred > 0 else 0.0

    if recall + precision == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def _lcs_length(x: list[str], y: list[str]) -> int:
    m, n = len(x), len(y)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _rouge_l(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    recall = lcs / len(ref_tokens)
    precision = lcs / len(pred_tokens)
    if recall + precision == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def add_rouge(record: dict[str, Any], text: str, reference: str | None) -> dict[str, Any]:
    if not reference or not str(reference).strip():
        record["rouge1"] = None
        record["rouge2"] = None
        record["rougeL"] = None
        return record

    pred_tokens = str(text).split()
    ref_tokens = str(reference).split()

    record["rouge1"] = round(_rouge_n(pred_tokens, ref_tokens, 1), 4)
    record["rouge2"] = round(_rouge_n(pred_tokens, ref_tokens, 2), 4)
    record["rougeL"] = round(_rouge_l(pred_tokens, ref_tokens), 4)
    return record


def aggregate_rouge(records: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    r1 = [float(r["rouge1"]) for r in records if r.get("rouge1") is not None]
    r2 = [float(r["rouge2"]) for r in records if r.get("rouge2") is not None]
    rL = [float(r["rougeL"]) for r in records if r.get("rougeL") is not None]
    return {
        "rouge1": round(sum(r1) / len(r1), 4) if r1 else None,
        "rouge2": round(sum(r2) / len(r2), 4) if r2 else None,
        "rougeL": round(sum(rL) / len(rL), 4) if rL else None,
    }
