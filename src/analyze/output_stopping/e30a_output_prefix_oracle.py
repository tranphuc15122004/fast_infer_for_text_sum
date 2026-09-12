"""E30-A: offline output-prefix utility oracle for adaptive summary stopping.

This module consumes the full-context generations already measured by E26-R.
It does not run target inference.  A summary is cut only at sentence
boundaries.  The primary risk contract is document-level: a document violates
an action if either ROUGE-L or local BERTScore F1 drops by more than epsilon
relative to that document's full output.  Fixed policies use one action for
all documents; the adaptive oracle selects the earliest safe boundary per
document.  The oracle is a hindsight upper bound, not a deployable policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.rouge import rouge_scores  # noqa: E402


DEFAULT_EPSILONS = (0.01, 0.02, 0.05)
DEFAULT_ALPHAS = (0.0, 0.05, 0.10)
DEFAULT_FIXED_BUDGETS = (64, 96, 128, 160, 192, 224)
QUALITY_METRICS = ("rougeL", "bertscore_f1")
DATASETS = ("cnn_dailymail", "govreport", "multi_news")

_BOUNDARY_RE = re.compile(r"[.!?。！？]+(?:[\"'”’»»\)\]]+)?(?:\s+|$)|\n+")


@dataclass(frozen=True)
class PrefixRecord:
    index: int
    text: str
    token_count: int
    rouge_l: float
    bertscore_f1: float
    projected_cost_ms: float
    is_full: bool
    boundary_over_budget: bool = False


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_json(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _safe_json(value.item())
        except (TypeError, ValueError):
            pass
    return value


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            rows.append(item)
    return rows


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return non-empty sentence-like spans without dropping source text.

    The E30-A input is English generated summaries.  Punctuation and newline
    boundaries are handled deterministically; a final unterminated fragment is
    retained.  This is an analysis boundary detector, not a linguistic parser.
    """

    text = str(text or "")
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY_RE.finditer(text):
        end = match.end()
        candidate = text[start:end].strip()
        if candidate:
            left = start + len(text[start:end]) - len(text[start:end].lstrip())
            segment = text[start:end]
            right = end - (len(segment) - len(segment.rstrip()))
            if right > left:
                spans.append((left, right))
        start = end
    if start < len(text):
        candidate = text[start:].strip()
        if candidate:
            left = start + len(text[start:]) - len(text[start:].lstrip())
            segment = text[start:]
            right = len(text) - (len(segment) - len(segment.rstrip()))
            if right > left:
                spans.append((left, right))
    if not spans and text.strip():
        left = len(text) - len(text.lstrip())
        right = len(text.rstrip())
        spans.append((left, right))
    return spans


def prefix_texts(text: str) -> list[str]:
    value = str(text)
    return [value[:end].strip() for _, end in sentence_spans(value)]


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    return len(ids)


def projected_cost(prefill_ms: float, decode_ms: float, prefix_tokens: int, full_tokens: int) -> float:
    if full_tokens <= 0:
        return float(prefill_ms)
    return float(prefill_ms) + float(decode_ms) * float(prefix_tokens) / float(full_tokens)


def is_safe_quality(quality: Mapping[str, float], full_quality: Mapping[str, float], epsilon: float) -> bool:
    return all(float(full_quality[key]) - float(quality[key]) <= float(epsilon) for key in QUALITY_METRICS)


def fixed_prefix_index(prefixes: Sequence[PrefixRecord], budget: int) -> int:
    """Choose the largest complete prefix not exceeding budget.

    If the first sentence itself exceeds the requested budget, the first
    sentence is retained as a non-empty sentence-boundary fallback and marked
    ``boundary_over_budget`` in its copied record by the caller.
    """

    eligible = [i for i, prefix in enumerate(prefixes) if prefix.token_count <= budget]
    return eligible[-1] if eligible else 0


def adaptive_prefix_index(prefixes: Sequence[PrefixRecord], epsilon: float, full_quality: Mapping[str, float]) -> int:
    for index, prefix in enumerate(prefixes):
        quality = {"rougeL": prefix.rouge_l, "bertscore_f1": prefix.bertscore_f1}
        if is_safe_quality(quality, full_quality, epsilon):
            return index
    return len(prefixes) - 1


def percentile(values: Sequence[float], probability: float) -> float | None:
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


def _doc_quality(row: Mapping[str, Any]) -> dict[str, float]:
    rouge = rouge_scores(str(row.get("summary", "")), str(row.get("reference", "")))
    bert = _finite(row.get("bertscore_f1"))
    if bert is None:
        raise ValueError("full E26-R row is missing bertscore_f1")
    return {"rougeL": float(rouge["rougeL"]), "bertscore_f1": bert}


def _make_full_docs(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if str(row.get("selector", "")) != "full" or str(row.get("budget_label", "")) != "full":
            continue
        doc_id = str(row.get("example_id", row.get("id", "")))
        if not doc_id or doc_id in seen:
            continue
        summary = str(row.get("summary", "")).strip()
        reference = str(row.get("reference", "")).strip()
        prefill = _finite(row.get("prefill_ms"))
        decode = _finite(row.get("decode_ms"))
        full_tokens = _finite(row.get("output_tokens"))
        if not summary or not reference or prefill is None or decode is None or full_tokens is None or full_tokens <= 0:
            continue
        docs.append({
            "example_id": doc_id,
            "summary": summary,
            "reference": reference,
            "full_prefill_ms": prefill,
            "full_decode_ms": decode,
            "full_output_tokens_measured": int(full_tokens),
            "existing_full_rougeL": _finite(row.get("rougeL")),
            "existing_full_bertscore_f1": _finite(row.get("bertscore_f1")),
            "source_row": dict(row),
        })
        seen.add(doc_id)
    return docs


def _score_bert_sorted(scorer: Any, tokenizer: Any, candidates: Sequence[str], references: Sequence[str]) -> list[dict[str, float]]:
    """Score pairs while minimizing padding and duplicate reference encodes.

    The scorer's mathematical operation is unchanged.  Sorting only changes
    batching order; reference embeddings are cached by exact reference text,
    which is safe because the local token-cosine scorer is deterministic.
    """

    candidate_order = sorted(range(len(candidates)), key=lambda i: len(tokenizer(candidates[i], add_special_tokens=False)["input_ids"]))
    candidate_embeddings_sorted = scorer._encode([candidates[i] for i in candidate_order])
    candidate_embeddings: list[Any] = [None] * len(candidates)
    for index, embedding in zip(candidate_order, candidate_embeddings_sorted):
        candidate_embeddings[index] = embedding

    reference_indices: dict[str, int] = {}
    unique_references: list[str] = []
    for reference in references:
        if reference not in reference_indices:
            reference_indices[reference] = len(unique_references)
            unique_references.append(reference)
    reference_order = sorted(range(len(unique_references)), key=lambda i: len(tokenizer(unique_references[i], add_special_tokens=False)["input_ids"]))
    reference_embeddings_sorted = scorer._encode([unique_references[i] for i in reference_order])
    reference_embeddings: list[Any] = [None] * len(unique_references)
    for index, embedding in zip(reference_order, reference_embeddings_sorted):
        reference_embeddings[index] = embedding

    scores: list[dict[str, float]] = []
    for index, candidate_embedding in enumerate(candidate_embeddings):
        precision, recall, f1 = scorer._pair(candidate_embedding, reference_embeddings[reference_indices[references[index]]])
        scores.append({"bertscore_p": precision, "bertscore_r": recall, "bertscore_f1": f1})
    return scores


def _build_records(docs: Sequence[Mapping[str, Any]], tokenizer: Any, scorer: Any) -> list[dict[str, Any]]:
    """Build per-document prefix records, scoring BERTScore with cached refs."""

    pending: list[tuple[int, int, str]] = []
    prefix_data: list[list[dict[str, Any]]] = []
    for doc_index, doc in enumerate(docs):
        texts = prefix_texts(str(doc["summary"]))
        if not texts:
            raise ValueError(f"document {doc['example_id']} has no sentence boundary")
        items: list[dict[str, Any]] = []
        for prefix_index, text in enumerate(texts):
            items.append({
                "index": prefix_index,
                "text": text,
                "token_count": _token_count(tokenizer, text),
            })
            pending.append((doc_index, prefix_index, text))
        prefix_data.append(items)

    candidates = [item[2] for item in pending]
    references = [str(docs[item[0]]["reference"]) for item in pending]
    bert_scores = _score_bert_sorted(scorer, tokenizer, candidates, references)

    records: list[dict[str, Any]] = []
    for doc_index, doc in enumerate(docs):
        full_quality = _doc_quality(doc["source_row"])
        items = prefix_data[doc_index]
        for item in items:
            score = rouge_scores(item["text"], str(doc["reference"]))
            bert = float(bert_scores[len(records)]["bertscore_f1"])
            is_full = item["index"] == len(items) - 1
            cost = (
                float(doc["full_prefill_ms"]) + float(doc["full_decode_ms"])
                if is_full
                else projected_cost(
                    float(doc["full_prefill_ms"]),
                    float(doc["full_decode_ms"]),
                    int(item["token_count"]),
                    int(doc["full_output_tokens_measured"]),
                )
            )
            records.append({
                "example_id": str(doc["example_id"]),
                "prefix_index": int(item["index"]),
                "prefix_text": item["text"],
                "prefix_token_count": int(item["token_count"]),
                "sentence_count_total": len(items),
                "rougeL": float(score["rougeL"]),
                "bertscore_f1": bert,
                "full_rougeL": float(full_quality["rougeL"]),
                "full_bertscore_f1": float(full_quality["bertscore_f1"]),
                "delta_rougeL": float(full_quality["rougeL"] - float(score["rougeL"])),
                "delta_bertscore_f1": float(full_quality["bertscore_f1"] - bert),
                "projected_cost_ms": cost,
                "full_prefill_ms": float(doc["full_prefill_ms"]),
                "full_decode_ms": float(doc["full_decode_ms"]),
                "full_output_tokens_measured": int(doc["full_output_tokens_measured"]),
                "is_full": is_full,
                "existing_full_rougeL": doc["existing_full_rougeL"],
                "existing_full_bertscore_f1": doc["existing_full_bertscore_f1"],
            })
    # Use the recomputed full prefix as the reference point for every prefix
    # in the document.  This avoids a tiny implementation/rounding mismatch
    # between the stored E26-R full BERTScore and the prefix scorer when the
    # full summary itself is evaluated.
    by_document = _group_records(records)
    for rows in by_document.values():
        full_row = rows[-1]
        full_rouge = float(full_row["rougeL"])
        full_bert = float(full_row["bertscore_f1"])
        for row in rows:
            row["full_rougeL"] = full_rouge
            row["full_bertscore_f1"] = full_bert
    return records


def _group_records(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["example_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["prefix_index"]))
    return dict(grouped)


def _prefix_as_record(row: Mapping[str, Any], *, over_budget: bool = False) -> PrefixRecord:
    return PrefixRecord(
        index=int(row["prefix_index"]),
        text=str(row["prefix_text"]),
        token_count=int(row["prefix_token_count"]),
        rouge_l=float(row["rougeL"]),
        bertscore_f1=float(row["bertscore_f1"]),
        projected_cost_ms=float(row["projected_cost_ms"]),
        is_full=bool(row["is_full"]),
        boundary_over_budget=over_budget,
    )


def _selected_action(
    doc_rows: Sequence[Mapping[str, Any]],
    label: str,
    epsilon: float,
    full_quality: Mapping[str, float],
) -> dict[str, Any]:
    prefixes = [_prefix_as_record(row) for row in doc_rows]
    if label == "full":
        index = len(prefixes) - 1
        requested_budget = None
    else:
        requested_budget = int(label)
        index = fixed_prefix_index(prefixes, requested_budget)
    selected = prefixes[index]
    quality = {"rougeL": selected.rouge_l, "bertscore_f1": selected.bertscore_f1}
    violated_metrics = [
        metric for metric in QUALITY_METRICS
        if float(full_quality[metric]) - float(quality[metric]) > float(epsilon)
    ]
    return {
        "label": label,
        "requested_budget_tokens": requested_budget,
        "prefix_index": selected.index,
        "prefix_token_count": selected.token_count,
        "projected_cost_ms": selected.projected_cost_ms,
        "quality": quality,
        "quality_delta_vs_full": {metric: float(full_quality[metric] - quality[metric]) for metric in QUALITY_METRICS},
        "violates": bool(violated_metrics),
        "violated_metrics": violated_metrics,
        "boundary_over_budget": bool(requested_budget is not None and selected.token_count > requested_budget),
        "is_full": selected.is_full,
    }


def summarize_fixed(
    docs: Mapping[str, Mapping[str, PrefixRecord]],
    full_quality: Mapping[str, Mapping[str, float]],
    *,
    label: str,
    epsilon: float,
) -> dict[str, Any]:
    actions: dict[str, dict[str, Any]] = {}
    for doc_id, choices in docs.items():
        prefix = choices[label]
        quality = {"rougeL": prefix.rouge_l, "bertscore_f1": prefix.bertscore_f1}
        violated_metrics = [m for m in QUALITY_METRICS if full_quality[doc_id][m] - quality[m] > epsilon]
        actions[doc_id] = {
            "label": label,
            "prefix_token_count": prefix.token_count,
            "projected_cost_ms": prefix.projected_cost_ms,
            "quality": quality,
            "violates": bool(violated_metrics),
            "violated_metrics": violated_metrics,
            "boundary_over_budget": prefix.boundary_over_budget,
        }
    return _summarize_actions(actions, full_quality)


def _summarize_actions(actions: Mapping[str, Mapping[str, Any]], full_quality: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    ids = list(actions)
    costs = [float(actions[doc]["projected_cost_ms"]) for doc in ids]
    quality = {
        metric: statistics.fmean(float(actions[doc]["quality"][metric]) for doc in ids)
        for metric in QUALITY_METRICS
    }
    full_means = {
        metric: statistics.fmean(float(full_quality[doc][metric]) for doc in ids)
        for metric in QUALITY_METRICS
    }
    violations = [doc for doc in ids if actions[doc]["violates"]]
    return {
        "documents": len(ids),
        "mean_projected_cost_ms": statistics.fmean(costs) if costs else None,
        "quality": quality,
        "full_quality": full_means,
        "quality_delta_vs_full": {m: quality[m] - full_means[m] for m in QUALITY_METRICS},
        "violating_documents": len(violations),
        "risk_rate": len(violations) / len(ids) if ids else None,
        "violating_document_ids": violations,
        "mean_prefix_tokens": statistics.fmean(float(actions[d]["prefix_token_count"]) for d in ids) if ids else None,
        "over_budget_fallback_documents": sum(bool(actions[d].get("boundary_over_budget")) for d in ids),
        "action_counts": dict(Counter(str(actions[d]["label"]) for d in ids)),
        "chosen_actions": {doc: dict(actions[doc]) for doc in ids},
    }


def risk_limit(alpha: float, documents: int) -> int:
    return int(math.floor(float(alpha) * int(documents) + 1e-12))


def evaluate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    fixed_labels: Sequence[str],
    epsilons: Sequence[float],
    alphas: Sequence[float],
) -> dict[str, Any]:
    grouped = _group_records(records)
    full_quality = {
        doc: {"rougeL": float(rows[-1]["full_rougeL"]), "bertscore_f1": float(rows[-1]["full_bertscore_f1"])}
        for doc, rows in grouped.items()
    }
    choices: dict[str, dict[str, PrefixRecord]] = {}
    for doc, rows in grouped.items():
        prefixes = [_prefix_as_record(row) for row in rows]
        choices[doc] = {"full": prefixes[-1]}
        for label in fixed_labels:
            if label == "full":
                continue
            index = fixed_prefix_index(prefixes, int(label))
            selected = prefixes[index]
            choices[doc][label] = PrefixRecord(**{**asdict(selected), "boundary_over_budget": selected.token_count > int(label)})

    fixed_by_contract: dict[str, dict[str, Any]] = {}
    adaptive_by_contract: dict[str, dict[str, Any]] = {}
    for epsilon in epsilons:
        for alpha in alphas:
            key = f"epsilon={epsilon:.4f},alpha={alpha:.4f}"
            candidates: list[dict[str, Any]] = []
            max_violations = risk_limit(alpha, len(choices))
            for label in fixed_labels:
                candidate = summarize_fixed(choices, full_quality, label=label, epsilon=epsilon)
                candidate = {**candidate, "label": label, "risk_contract_pass": candidate["violating_documents"] <= max_violations}
                candidates.append(candidate)
            passing = [candidate for candidate in candidates if candidate["risk_contract_pass"]]
            best_fixed = min(passing, key=lambda item: (item["mean_projected_cost_ms"], item["label"])) if passing else None

            adaptive_actions: dict[str, dict[str, Any]] = {}
            for doc, rows in grouped.items():
                prefixes = [_prefix_as_record(row) for row in rows]
                index = adaptive_prefix_index(prefixes, epsilon, full_quality[doc])
                prefix = prefixes[index]
                adaptive_actions[doc] = {
                    "label": "oracle_adaptive",
                    "prefix_index": prefix.index,
                    "prefix_token_count": prefix.token_count,
                    "projected_cost_ms": prefix.projected_cost_ms,
                    "quality": {"rougeL": prefix.rouge_l, "bertscore_f1": prefix.bertscore_f1},
                    "quality_delta_vs_full": {m: full_quality[doc][m] - getattr(prefix, "rouge_l" if m == "rougeL" else "bertscore_f1") for m in QUALITY_METRICS},
                    "violates": False,
                    "violated_metrics": [],
                    "boundary_over_budget": False,
                    "is_full": prefix.is_full,
                }
            adaptive = _summarize_actions(adaptive_actions, full_quality)
            adaptive["label"] = "oracle_adaptive"
            adaptive["risk_contract_pass"] = adaptive["violating_documents"] <= max_violations
            adaptive["max_violating_documents"] = max_violations
            if best_fixed and adaptive["mean_projected_cost_ms"] is not None:
                adaptive["headroom_vs_best_fixed"] = 1.0 - adaptive["mean_projected_cost_ms"] / best_fixed["mean_projected_cost_ms"]
            else:
                adaptive["headroom_vs_best_fixed"] = None
            if adaptive["mean_projected_cost_ms"] is not None:
                full_cost = statistics.fmean(float(rows[-1]["projected_cost_ms"]) for rows in grouped.values())
                adaptive["mean_prefix_token_ratio"] = statistics.fmean(
                    float(adaptive_actions[doc]["prefix_token_count"]) / float(grouped[doc][-1]["prefix_token_count"])
                    for doc in grouped
                )
                adaptive["mean_output_token_saving"] = statistics.fmean(
                    1.0 - float(adaptive_actions[doc]["prefix_token_count"]) / float(grouped[doc][-1]["prefix_token_count"])
                    for doc in grouped
                )
                adaptive["non_full_fraction"] = statistics.fmean(
                    0.0 if adaptive_actions[doc].get("is_full") else 1.0
                    for doc in grouped
                )
                adaptive["oracle_savings_vs_full"] = 1.0 - adaptive["mean_projected_cost_ms"] / full_cost
                adaptive["speedup_vs_full"] = full_cost / adaptive["mean_projected_cost_ms"]
            else:
                adaptive["oracle_savings_vs_full"] = None
                adaptive["speedup_vs_full"] = None
            fixed_by_contract[key] = {"epsilon": epsilon, "alpha": alpha, "max_violating_documents": max_violations, "best": best_fixed, "candidates": candidates}
            adaptive_by_contract[key] = adaptive

    doc_summary: list[dict[str, Any]] = []
    for doc, rows in grouped.items():
        full = rows[-1]
        doc_summary.append({
            "example_id": doc,
            "sentence_count": len(rows),
            "full_output_tokens": int(full["prefix_token_count"]),
            "measured_full_output_tokens": int(full["full_output_tokens_measured"]),
            "full_projected_cost_ms": float(full["projected_cost_ms"]),
            "full_rougeL": float(full["full_rougeL"]),
            "full_bertscore_f1": float(full["full_bertscore_f1"]),
            "existing_full_rougeL": full.get("existing_full_rougeL"),
            "existing_full_bertscore_f1": full.get("existing_full_bertscore_f1"),
            "prefixes": [dict(row) for row in rows],
        })
    return {
        "documents": len(grouped),
        "fixed_labels": list(fixed_labels),
        "epsilons": list(epsilons),
        "alphas": list(alphas),
        "fixed_by_contract": fixed_by_contract,
        "adaptive_by_contract": adaptive_by_contract,
        "documents_detail": doc_summary,
    }


def _bootstrap_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    fixed_labels: Sequence[str],
    epsilon: float,
    alpha: float,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped = _group_records(records)
    ids = list(grouped)
    rng = random.Random(seed)
    headrooms: list[float] = []
    oracle_savings: list[float] = []
    fixed_costs: list[float] = []
    adaptive_costs: list[float] = []
    for _ in range(samples):
        sampled_ids = [rng.choice(ids) for _ in ids]
        cloned: list[dict[str, Any]] = []
        for occurrence, source_id in enumerate(sampled_ids):
            for row in grouped[source_id]:
                copy = dict(row)
                copy["example_id"] = f"{source_id}__bootstrap_{occurrence}"
                cloned.append(copy)
        result = evaluate_dataset(cloned, fixed_labels=fixed_labels, epsilons=[epsilon], alphas=[alpha])
        key = f"epsilon={epsilon:.4f},alpha={alpha:.4f}"
        fixed = result["fixed_by_contract"][key]["best"]
        adaptive = result["adaptive_by_contract"][key]
        if fixed and adaptive.get("mean_projected_cost_ms") is not None:
            fixed_costs.append(float(fixed["mean_projected_cost_ms"]))
            adaptive_costs.append(float(adaptive["mean_projected_cost_ms"]))
            headrooms.append(float(adaptive["headroom_vs_best_fixed"]))
            oracle_savings.append(float(adaptive["oracle_savings_vs_full"]))
    return {
        "samples_requested": samples,
        "valid_replicates": len(headrooms),
        "seed": seed,
        "headroom_mean": statistics.fmean(headrooms) if headrooms else None,
        "headroom_ci_95": [percentile(headrooms, .025), percentile(headrooms, .975)],
        "oracle_savings_mean": statistics.fmean(oracle_savings) if oracle_savings else None,
        "oracle_savings_ci_95": [percentile(oracle_savings, .025), percentile(oracle_savings, .975)],
        "fixed_cost_ci_95_ms": [percentile(fixed_costs, .025), percentile(fixed_costs, .975)],
        "adaptive_cost_ci_95_ms": [percentile(adaptive_costs, .025), percentile(adaptive_costs, .975)],
    }


def _holdout_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    fixed_labels: Sequence[str],
    epsilons: Sequence[float],
    alphas: Sequence[float],
    fraction: float,
    seed: int,
) -> dict[str, Any]:
    grouped = _group_records(records)
    ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = max(1, min(len(ids) - 1, int(round(len(ids) * fraction))))
    test_ids = set(ids[:n_test])
    calibration_ids = [doc for doc in ids if doc not in test_ids]
    cal_rows = [row for doc in calibration_ids for row in grouped[doc]]
    test_rows = [row for doc in sorted(test_ids) for row in grouped[doc]]
    calibration = evaluate_dataset(cal_rows, fixed_labels=fixed_labels, epsilons=epsilons, alphas=alphas)
    holdout = evaluate_dataset(test_rows, fixed_labels=fixed_labels, epsilons=epsilons, alphas=alphas)
    cells: list[dict[str, Any]] = []
    for epsilon in epsilons:
        for alpha in alphas:
            key = f"epsilon={epsilon:.4f},alpha={alpha:.4f}"
            label = (calibration["fixed_by_contract"][key]["best"] or {}).get("label")
            candidate = next((item for item in holdout["fixed_by_contract"][key]["candidates"] if item["label"] == label), None) if label else None
            adaptive = holdout["adaptive_by_contract"][key]
            headroom = None
            if candidate and candidate.get("mean_projected_cost_ms"):
                headroom = 1.0 - adaptive["mean_projected_cost_ms"] / candidate["mean_projected_cost_ms"]
            cells.append({
                "epsilon": epsilon,
                "alpha": alpha,
                "calibrated_fixed_label": label,
                "calibration_documents": len(calibration_ids),
                "holdout_documents": len(test_ids),
                "holdout_fixed": candidate,
                "holdout_fixed_risk_contract_pass": bool(candidate and candidate["risk_contract_pass"]),
                "holdout_adaptive": adaptive,
                "holdout_headroom_vs_calibrated_fixed": headroom,
            })
    return {
        "seed": seed,
        "fraction": fraction,
        "calibration_document_ids": calibration_ids,
        "holdout_document_ids": sorted(test_ids),
        "cells": cells,
    }


def _dataset_records(input_path: str | Path, tokenizer: Any, scorer: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    docs = _make_full_docs(load_jsonl(input_path))
    records = _build_records(docs, tokenizer, scorer)
    return docs, records


def _raw_prefix_csv_rows(dataset_results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, result in dataset_results.items():
        for doc in result["documents_detail"]:
            for prefix in doc["prefixes"]:
                rows.append({"dataset": dataset, **prefix})
    return rows


def _contract_csv_rows(dataset_results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, result in dataset_results.items():
        for key, fixed in result["fixed_by_contract"].items():
            adaptive = result["adaptive_by_contract"][key]
            best = fixed["best"] or {}
            rows.append({
                "dataset": dataset,
                "epsilon": fixed["epsilon"],
                "alpha": fixed["alpha"],
                "documents": result["documents"],
                "max_violating_documents": fixed["max_violating_documents"],
                "best_fixed_label": best.get("label"),
                "best_fixed_cost_ms": best.get("mean_projected_cost_ms"),
                "best_fixed_risk_rate": best.get("risk_rate"),
                "adaptive_oracle_cost_ms": adaptive.get("mean_projected_cost_ms"),
                "adaptive_oracle_risk_rate": adaptive.get("risk_rate"),
                "oracle_savings_vs_full": adaptive.get("oracle_savings_vs_full"),
                "adaptive_headroom_vs_best_fixed": adaptive.get("headroom_vs_best_fixed"),
                "adaptive_speedup_vs_full": adaptive.get("speedup_vs_full"),
            })
    return rows


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.{digits}f}%"


def _markdown(
    *,
    result: Mapping[str, Any],
    source_paths: Mapping[str, str],
    output_dir: Path,
) -> str:
    primary_key = "epsilon=0.0200,alpha=0.0500"
    lines = [
        "# E30-A Output-Prefix Utility Oracle — Báo cáo đầy đủ",
        "",
        "> **Phạm vi:** phân tích offline trên full generations của E26-R; không chạy E30-B/C và không khẳng định có controller triển khai.",
        "",
        "## 1. Kết luận điều hành",
        "",
    ]
    primary_cells = []
    for dataset, item in result["datasets"].items():
        fixed = item["fixed_by_contract"][primary_key]["best"] or {}
        adaptive = item["adaptive_by_contract"][primary_key]
        primary_cells.append((dataset, fixed, adaptive))
        lines.append(
            f"- **{dataset}**: full projected cost `{_fmt(item['full_mean_cost_ms'], 2)} ms`; oracle `{_fmt(adaptive.get('mean_projected_cost_ms'), 2)} ms`, savings vs full `{_pct(adaptive.get('oracle_savings_vs_full'))}`; best fixed `{fixed.get('label', 'none')}` at `{_fmt(fixed.get('mean_projected_cost_ms'), 2)} ms`; adaptive headroom vs best fixed `{_pct(adaptive.get('headroom_vs_best_fixed'))}`."
        )
    qualifying_oracle = sum(bool(adaptive.get("oracle_savings_vs_full") is not None and adaptive["oracle_savings_vs_full"] >= .20) for _, _, adaptive in primary_cells)
    qualifying_headroom = sum(bool(adaptive.get("headroom_vs_best_fixed") is not None and adaptive["headroom_vs_best_fixed"] >= .10) for _, _, adaptive in primary_cells)
    lines += [
        "",
        f"**E30-A gate sơ bộ:** oracle savings ≥20% ở `{qualifying_oracle}/3` dataset; adaptive headroom ≥10% so với best fixed ở `{qualifying_headroom}/3` dataset. Gate này chỉ là feasibility/oracle screen; adaptive oracle dùng hindsight.",
        "",
        "## 2. Câu hỏi và giả thuyết",
        "",
        "E30-A kiểm tra liệu summary prefix ngắn, kết thúc ở sentence boundary, có đạt chất lượng gần full output hay không; và liệu sự khác biệt giữa các tài liệu tạo ra headroom cho adaptive stopping vượt best fixed output budget hay không.",
        "",
        "Hypothesis: `Q(prefix) ≈ Q(full)` tại một boundary sớm `k*` trên một phần đáng kể documents, nhưng `k*` thay đổi theo document.",
        "",
        "Không được diễn giải oracle này thành một thuật toán online: oracle biết trước quality của mọi prefix và reference.",
        "",
        "## 3. Dữ liệu và provenance",
        "",
        "| Dataset | Full rows usable | Prefix rows | Sentence boundaries | Source artifact |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset, item in result["datasets"].items():
        lines.append(f"| {dataset} | {item['documents']} | {item['prefix_rows']} | {item['prefix_count_total']} | `{source_paths[dataset]}` |")
    lines += [
        "",
        "Các row được giữ là `selector=full`, `budget_label=full`, có summary/reference, prefill/decode/output_tokens và BERTScore F1. E26-R có 30 document/dataset; nếu một dataset không đủ 30 thì trạng thái bị đánh dấu incomplete và không được gọi là full.",
        "",
        "### Runtime/provenance kế thừa từ E26-R",
        "",
        "- Target generation đã có sẵn từ lượt fresh T4 E26-R: Qwen3-4B, greedy, batch size 1, max-new-tokens 256.",
        "- E30-A không chạy lại target model; chỉ dùng measured `prefill_ms`, `decode_ms`, `output_tokens` của full rows.",
        "- ROUGE-L được tính lại bằng `scripts/common/rouge.py` trên từng prefix.",
        "- BERTScore F1 được tính lại bằng local RoBERTa token-cosine, no-IDF, max length 512, cùng implementation E26-R; đây không phải package `bert-score`.",
        "- Prefix token count dùng local Qwen3-4B tokenizer, `add_special_tokens=False`.",
        "",
        "## 4. Protocol chính xác",
        "",
        "### 4.1 Prefix construction",
        "",
        "Summary full được tách theo dấu câu `. ! ?`/Unicode tương đương và newline. Mỗi prefix là phần tích lũy `s1…sk`; không cắt giữa câu. Nếu câu đầu dài hơn fixed budget, policy dùng câu đầu như fallback không rỗng và ghi `boundary_over_budget=true`; số token thực tế vẫn được báo, không giả vờ rằng nó nằm trong budget.",
        "",
        "### 4.2 Quality và violation",
        "",
        "Với document `i`, chất lượng full là `Q_full,i`; prefix vi phạm khi ít nhất một điều kiện sau đúng:",
        "",
        "```text",
        "full_rougeL_i - prefix_rougeL_i > epsilon",
        "full_bertscore_f1_i - prefix_bertscore_f1_i > epsilon",
        "```",
        "",
        "Risk = số document vi phạm / tổng document. Contract cho phép tối đa `floor(alpha*N)` document vi phạm.",
        "",
        "### 4.3 Fixed và adaptive oracle",
        "",
        "- Fixed actions: `64, 96, 128, 160, 192, 224, full`; một action chung cho toàn dataset; tại sentence boundary chọn prefix dài nhất không vượt budget, hoặc câu đầu fallback như trên.",
        "- Adaptive oracle: với mỗi document chọn sentence boundary sớm nhất thỏa cả ROUGE-L và BERTScore F1 trong epsilon; full boundary luôn là fallback an toàn.",
        "- Best fixed: trong các fixed actions có empirical risk ≤ alpha, chọn action có mean projected cost thấp nhất.",
        "- Cost prefix: `full_prefill_ms + full_decode_ms * prefix_tokens / full_output_tokens_measured`; đây là projected cost, không phải fresh online latency.",
        "- `S_oracle = 1 - C_oracle/C_full`; `H_adaptive = 1 - C_oracle/C_best_fixed`.",
        "",
        "### 4.4 Grids và validation",
        "",
        f"- Epsilon: `{', '.join(f'{x:.2f}' for x in result['epsilon'])}`; alpha: `{', '.join(f'{x:.2f}' for x in result['alpha'])}`.",
        "- Primary contract: `epsilon=0.02, alpha=0.05`.",
        "- Bootstrap: document resampling with replacement, 200 replicates, fixed seeds per dataset; percentile 95% CI.",
        "- Holdout: deterministic 70/30 document split, fixed seed 20260912; fixed label selected on calibration and evaluated unchanged on holdout. Holdout adaptive remains hindsight upper bound.",
        "",
        "## 5. Prefix-level descriptive results",
        "",
        "| Dataset | Docs | Mean sentence count | Mean full output tokens | Mean full ROUGE-L | Mean full BERTScore F1 | Mean oracle prefix tokens | Mean oracle token saving | Oracle non-full fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in result["datasets"].items():
        detail = item["documents_detail"]
        sentence_counts = [d["sentence_count"] for d in detail]
        full_tokens = [d["full_output_tokens"] for d in detail]
        full_rouge = [d["full_rougeL"] for d in detail]
        full_bert = [d["full_bertscore_f1"] for d in detail]
        primary_adaptive = item["adaptive_by_contract"][primary_key]
        chosen = primary_adaptive.get("chosen_actions", {})
        adaptive = item["adaptive_by_contract"][primary_key]
        lines.append(f"| {dataset} | {len(detail)} | {_fmt(statistics.fmean(sentence_counts), 2)} | {_fmt(statistics.fmean(full_tokens), 2)} | {_fmt(statistics.fmean(full_rouge), 4)} | {_fmt(statistics.fmean(full_bert), 4)} | {_fmt(adaptive.get('mean_prefix_tokens'), 2)} | {_pct(adaptive.get('mean_output_token_saving'))} | {_pct(adaptive.get('non_full_fraction'))} |")
    lines += [
        "",
        "Tất cả prefix-level metrics được lưu đầy đủ trong `metrics.json`; `prefix_metrics.csv` chứa một dòng cho mỗi document × sentence boundary, không chỉ các dòng aggregate.",
        "",
        "## 6. Tất cả risk-contract cells",
        "",
        "Bảng sau chứa mọi epsilon/alpha cell. `best_fixed` là policy fixed tốt nhất dưới đúng contract; `oracle_headroom` là khoảng cách với oracle adaptive.",
        "",
        "| Dataset | Epsilon | Alpha | Max violations | Best fixed | Fixed cost ms | Fixed risk | Oracle cost ms | Oracle risk | Savings vs full | Headroom vs fixed |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in _contract_csv_rows(result["datasets"]):
        lines.append(f"| {row['dataset']} | {row['epsilon']:.2f} | {row['alpha']:.2f} | {row['max_violating_documents']} | {row['best_fixed_label'] or 'none'} | {_fmt(row['best_fixed_cost_ms'], 2)} | {_pct(row['best_fixed_risk_rate'])} | {_fmt(row['adaptive_oracle_cost_ms'], 2)} | {_pct(row['adaptive_oracle_risk_rate'])} | {_pct(row['oracle_savings_vs_full'])} | {_pct(row['adaptive_headroom_vs_best_fixed'])} |")
    lines += ["", "### Fixed action candidates cho primary contract", "", "| Dataset | Action | Mean prefix tokens | Projected cost ms | Risk | Pass alpha=0.05 | Fallback docs |", "|---|---|---:|---:|---:|---|---:|"]
    for dataset, item in result["datasets"].items():
        key = primary_key
        fixed_cell = item["fixed_by_contract"][key]
        for candidate in fixed_cell["candidates"]:
            lines.append(f"| {dataset} | {candidate['label']} | {_fmt(candidate.get('mean_prefix_tokens'), 2)} | {_fmt(candidate.get('mean_projected_cost_ms'), 2)} | {_pct(candidate.get('risk_rate'))} | {'yes' if candidate['risk_contract_pass'] else 'no'} | {candidate.get('over_budget_fallback_documents', 0)} |")
    lines += [
        "",
        "## 7. Bootstrap CI 95%",
        "",
        "CI là document-level empirical percentile intervals; chúng không phải conformal guarantee và không sửa vấn đề cỡ mẫu nhỏ.",
        "",
        "| Dataset | Epsilon | Alpha | Valid reps | Oracle savings mean | CI savings | Headroom mean | CI headroom | Fixed cost CI ms | Oracle cost CI ms |",
        "|---|---:|---:|---:|---:|---|---:|---|---|---|",
    ]
    for dataset, validation in result["validation"].items():
        for cell in validation["bootstrap"]:
            lines.append(f"| {dataset} | {cell['epsilon']:.2f} | {cell['alpha']:.2f} | {cell['valid_replicates']} | {_pct(cell.get('oracle_savings_mean'))} | {_pct(cell['oracle_savings_ci_95'][0])}–{_pct(cell['oracle_savings_ci_95'][1])} | {_pct(cell.get('headroom_mean'))} | {_pct(cell['headroom_ci_95'][0])}–{_pct(cell['headroom_ci_95'][1])} | {_fmt(cell['fixed_cost_ci_95_ms'][0], 2)}–{_fmt(cell['fixed_cost_ci_95_ms'][1], 2)} | {_fmt(cell['adaptive_cost_ci_95_ms'][0], 2)}–{_fmt(cell['adaptive_cost_ci_95_ms'][1], 2)} |")
    lines += [
        "",
        "## 8. Holdout evaluation",
        "",
        "Holdout không train predictor; nó chỉ kiểm tra mức độ ổn định của fixed label được chọn ở calibration. Adaptive holdout vẫn là hindsight upper bound nên không phải generalization proof.",
        "",
        "| Dataset | Epsilon | Alpha | Cal docs | Holdout docs | Calibrated fixed | Holdout fixed risk | Holdout fixed pass | Holdout adaptive cost ms | Headroom vs calibrated fixed |",
        "|---|---:|---:|---:|---:|---|---:|---|---:|---:|",
    ]
    for dataset, validation in result["validation"].items():
        for cell in validation["holdout"]:
            lines.append(f"| {dataset} | {cell['epsilon']:.2f} | {cell['alpha']:.2f} | {cell['calibration_documents']} | {cell['holdout_documents']} | {cell['calibrated_fixed_label'] or 'none'} | {_pct((cell.get('holdout_fixed') or {}).get('risk_rate'))} | {'yes' if cell['holdout_fixed_risk_contract_pass'] else 'no'} | {_fmt((cell.get('holdout_adaptive') or {}).get('mean_projected_cost_ms'), 2)} | {_pct(cell.get('holdout_headroom_vs_calibrated_fixed'))} |")
    lines += [
        "",
        "## 9. Gate và diễn giải",
        "",
        "### Gate E30-A",
        "",
        "- Oracle E2E projected savings ≥20% trên ≥2/3 datasets.",
        "- Oracle adaptive headroom ≥10% so với best risk-matched fixed output budget trên ≥2/3 datasets; ngưỡng 15% được ghi riêng là strong screen.",
        "- Đây là gate feasibility, không phải bằng chứng method online.",
        "",
        "### Quy tắc kết luận",
        "",
        "- Nếu oracle savings thấp: không có đủ output-side ceiling.",
        "- Nếu oracle savings cao nhưng headroom so với fixed thấp: fixed length control đã lấy gần hết ceiling; không nên build adaptive controller.",
        "- Nếu in-sample headroom cao nhưng holdout fixed risk fail hoặc CI rộng: evidence chỉ exploratory; cần dataset lớn hơn trước khi claim.",
        "- Prefix quality không đồng nghĩa factuality/faithfulness; E30-A chưa có factuality metric và chưa chạy real early-stop inference.",
        "",
        "## 10. Artifact manifest",
        "",
        f"- Output directory: `{output_dir}`",
        "- `metrics.json`: toàn bộ document/prefix rows, fixed candidates, adaptive choices, bootstrap và holdout.",
        "- `prefix_metrics.csv`: toàn bộ prefix rows.",
        "- `contract_metrics.csv`: tất cả 27 risk cells (3 epsilon × 3 alpha × 3 dataset).",
        "- `run_manifest.json`: command, paths, tokenizer/scorer, protocol, counts và timing.",
        "- `report.md`: báo cáo này.",
        "",
        "## CONFIRMED",
        "",
        "- Chỉ các claim về việc E30-A đã được chạy đúng trên các full rows đủ điều kiện và các số liệu oracle/fixed trong artifact là confirmed.",
        "- Gate oracle/headroom được đánh giá trực tiếp theo số liệu primary cell; câu kết luận cụ thể nằm ở phần 11 bên dưới.",
        "",
        "## EXPLORATORY",
        "",
        "- Mọi oracle adaptive saving là hindsight upper bound; không phải policy runtime.",
        "- Holdout adaptive và bootstrap CI là diagnostic với 30 docs/dataset, không phải guarantee ngoài mẫu.",
        "",
        "## FAILED / INCOMPLETE",
        "",
        "- E30-B (online stopping signal) và E30-C (fresh inference) chưa chạy theo thiết kế tuần tự; không được ghi là đã hoàn thành.",
        "- Chưa có factuality/faithfulness evaluation ở E30-A.",
        "",
        "## HIGHEST VERIFIED RUNG",
        "",
        "- E30-A offline oracle screen; không có model training và không có runtime intervention. Artifact chứng minh: `metrics.json`, `contract_metrics.csv`, `run_manifest.json`.",
        "",
        "## EVIDENCE GAPS",
        "",
        "- Chưa biết signal online nào dự đoán boundary an toàn.",
        "- Chưa đo latency thực khi generation dừng tại sentence boundary.",
        "- Chưa kiểm tra trên native 8K/16K hoặc dataset ngoài ba dataset E26-R.",
        "- BERTScore là local no-IDF; quality contract không bao quát factuality.",
        "",
        "## RECOMMENDED NEXT",
        "",
        "- Chỉ nếu E30-A đạt gate với CI/holdout hợp lý: chạy E30-B so sánh EOS, redundancy, semantic drift và marginal source utility trên cùng prefix records; nếu không đạt gate, đóng output-stopping branch.",
        "",
        "## 11. Quyết định E30-A từ số liệu",
        "",
    ]
    for dataset, fixed, adaptive in primary_cells:
        oracle_pass = adaptive.get("oracle_savings_vs_full") is not None and adaptive["oracle_savings_vs_full"] >= .20
        headroom_pass = adaptive.get("headroom_vs_best_fixed") is not None and adaptive["headroom_vs_best_fixed"] >= .10
        lines.append(f"- **{dataset}**: oracle gate={'PASS' if oracle_pass else 'FAIL'}; adaptive-vs-fixed gate={'PASS' if headroom_pass else 'FAIL'}; strong 15% headroom={'PASS' if adaptive.get('headroom_vs_best_fixed') is not None and adaptive['headroom_vs_best_fixed'] >= .15 else 'FAIL'}. ")
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_safe_json(row) for row in rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer  # local import keeps pure tests light
    from src.analyze.safe_budget.augment_bertscore import LocalBERTScorer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    scorer = LocalBERTScorer(
        args.bertscore_model,
        device=args.device,
        dtype=args.dtype,
        max_length=args.bertscore_max_length,
        batch_size=args.bertscore_batch_size,
    )
    datasets: dict[str, dict[str, Any]] = {}
    source_paths: dict[str, str] = {}
    start = __import__("time").perf_counter()
    for dataset, input_path in args.input:
        source_paths[dataset] = str(input_path)
        docs, records = _dataset_records(input_path, tokenizer, scorer)
        print(f"[E30-A] {dataset}: scored {len(docs)} documents / {len(records)} sentence prefixes", flush=True)
        if len(docs) < args.min_documents:
            raise ValueError(f"{dataset}: only {len(docs)} usable full rows, expected at least {args.min_documents}")
        evaluation = evaluate_dataset(
            records,
            fixed_labels=[str(value) for value in args.fixed_budget] + ["full"],
            epsilons=args.epsilon,
            alphas=args.alpha,
        )
        full_cost = statistics.fmean(float(doc["full_prefill_ms"]) + float(doc["full_decode_ms"]) for doc in docs)
        evaluation["full_mean_cost_ms"] = full_cost
        evaluation["prefix_rows"] = len(records)
        evaluation["prefix_count_total"] = len(records)
        evaluation["source_path"] = str(input_path)
        datasets[dataset] = evaluation

    validation: dict[str, Any] = {}
    for dataset, item in datasets.items():
        records = [prefix for doc in item["documents_detail"] for prefix in doc["prefixes"]]
        bootstrap_cells = []
        for epsilon in args.epsilon:
            for alpha in args.alpha:
                bootstrap_cells.append({
                    "epsilon": epsilon,
                    "alpha": alpha,
                    **_bootstrap_dataset(records, fixed_labels=[str(value) for value in args.fixed_budget] + ["full"], epsilon=epsilon, alpha=alpha, samples=args.bootstrap_samples, seed=args.seed + len(bootstrap_cells)),
                })
        holdout = _holdout_dataset(
            records,
            fixed_labels=[str(value) for value in args.fixed_budget] + ["full"],
            epsilons=args.epsilon,
            alphas=args.alpha,
            fraction=args.holdout_fraction,
            seed=args.seed,
        )
        validation[dataset] = {"bootstrap": bootstrap_cells, "holdout": holdout["cells"], "holdout_meta": holdout}

    result: dict[str, Any] = {
        "experiment": "E30-A output-prefix utility oracle",
        "status": "complete",
        "datasets": datasets,
        "validation": validation,
        "epsilon": list(args.epsilon),
        "alpha": list(args.alpha),
        "fixed_budget": list(args.fixed_budget),
        "quality_metrics": list(QUALITY_METRICS),
        "primary_contract": {"epsilon": 0.02, "alpha": 0.05},
        "protocol": {
            "sentence_boundary_only": True,
            "first_sentence_over_budget_fallback": True,
            "cost_formula": "full_prefill_ms + full_decode_ms * prefix_token_count / full_output_tokens_measured",
            "risk_formula": "fraction of documents with any quality metric drop > epsilon",
            "oracle": "earliest safe sentence boundary per document using reference/full hindsight",
            "fixed": "one action for all documents, lowest projected cost among empirical-risk-valid actions",
        },
        "provenance": {
            "tokenizer": str(args.tokenizer),
            "bertscore_model": str(args.bertscore_model),
            "bertscore_device": args.device,
            "bertscore_dtype": args.dtype,
            "bertscore_implementation": "local_token_cosine_no_idf",
            "bertscore_max_length": args.bertscore_max_length,
            "bertscore_batch_size": args.bertscore_batch_size,
            "torch_threads": args.torch_threads,
            "target_inference_rerun": False,
        },
        "runtime": {"elapsed_seconds": __import__("time").perf_counter() - start},
    }
    output_dir.joinpath("metrics.json").write_text(json.dumps(_safe_json(result), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(output_dir / "prefix_metrics.csv", _raw_prefix_csv_rows(datasets))
    _write_csv(output_dir / "contract_metrics.csv", _contract_csv_rows(datasets))
    manifest = {
        "experiment": "E30-A",
        "command": " ".join(sys.argv),
        "input": source_paths,
        "output_dir": str(output_dir),
        "datasets": {name: {"documents": value["documents"], "prefix_rows": value["prefix_rows"]} for name, value in datasets.items()},
        "protocol": result["protocol"],
        "provenance": result["provenance"],
        "runtime": result["runtime"],
    }
    output_dir.joinpath("run_manifest.json").write_text(json.dumps(_safe_json(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    report = _markdown(result=result, source_paths=source_paths, output_dir=output_dir)
    output_dir.joinpath("report.md").write_text(report, encoding="utf-8")
    return result


def _input_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be DATASET=JSONL_PATH")
    dataset, path = value.split("=", 1)
    if not dataset or not path:
        raise argparse.ArgumentTypeError("input must be DATASET=JSONL_PATH")
    return dataset, path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_input_arg, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--bertscore-model", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float16", "float32", "bfloat16"), default="float32")
    parser.add_argument("--bertscore-max-length", type=int, default=512)
    parser.add_argument("--bertscore-batch-size", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--fixed-budget", action="append", type=int, default=None)
    parser.add_argument("--epsilon", action="append", type=float, default=None)
    parser.add_argument("--alpha", action="append", type=float, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--holdout-fraction", type=float, default=.30)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--min-documents", type=int, default=30)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.fixed_budget is None:
        args.fixed_budget = list(DEFAULT_FIXED_BUDGETS)
    if args.epsilon is None:
        args.epsilon = list(DEFAULT_EPSILONS)
    if args.alpha is None:
        args.alpha = list(DEFAULT_ALPHAS)
    result = run(args)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir), "datasets": {k: v["documents"] for k, v in result["datasets"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
