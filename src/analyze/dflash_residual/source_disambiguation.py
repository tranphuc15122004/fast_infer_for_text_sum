"""Offline source-conditioned diagnostics for the DFlash candidate lattice.

This module deliberately implements diagnostic, frozen-lattice selectors.  It
does not emulate DFlash2 and it never evaluates candidates outside the recorded
Top-M lattice.  Source support is exact tokenizer-token support; semantic
retrieval is a separate experiment and is not silently substituted here.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping, Sequence

from .metrics import _blocks, _observed_acceptance, prefix_match_length
from .prefix_gap import _candidate_hit, prefix_oracle_length


def build_source_index(
    records: Iterable[Mapping[str, Any]],
    encode: Callable[[str], Sequence[int]],
    token_filter_by_sample: Mapping[str, set[int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build an exact token-frequency index keyed by representative sample ID."""

    result: dict[str, dict[str, Any]] = {}
    for record in records:
        sample_id = record.get("id", record.get("sample_id", record.get("document_id")))
        if sample_id is None:
            continue
        text = record.get("document", record.get("context", record.get("text", "")))
        token_filter = token_filter_by_sample.get(str(sample_id)) if token_filter_by_sample else None
        raw_token_ids = [int(token) for token in encode(str(text or ""))]
        token_ids = list(raw_token_ids)
        if token_filter is not None:
            token_ids = [token for token in token_ids if token in token_filter]
        ngram_counts: dict[int, Counter[tuple[int, ...]]] = {}
        if token_filter is not None:
            for n in (2, 3):
                counts: Counter[tuple[int, ...]] = Counter()
                for start in range(0, max(0, len(raw_token_ids) - n + 1)):
                    gram = tuple(raw_token_ids[start:start + n])
                    if all(token in token_filter for token in gram):
                        counts[gram] += 1
                ngram_counts[n] = counts
        result[str(sample_id)] = {
            "dataset": str(record.get("dataset", "other")),
            "token_counts": Counter(token_ids),
            "token_count": len(token_ids),
            "document_count": len(set(token_ids)),
            "ngram_counts": ngram_counts,
        }
    return result


def annotate_source_rows(
    rows: Iterable[Mapping[str, Any]],
    source_index: Mapping[str, Mapping[str, Any]],
    *,
    copy_rows: bool = True,
) -> list[dict[str, Any]]:
    """Attach exact source-token support to every trace row.

    ``source_novel`` means the target token ID is absent from the source token
    sequence.  It is intentionally a token-level proxy; it is not claimed to be
    a semantic abstraction label.
    """

    annotated: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw) if copy_rows else raw  # type: ignore[assignment]
        source = source_index.get(str(row.get("sample_id")))
        if source is None:
            row.update({
                "source_metadata_status": "missing",
                "source_token_present": None,
                "source_token_frequency": None,
                "source_stratum": "unknown",
                "candidate_source_frequencies": None,
                "candidate_source_present": None,
            })
            annotated.append(row)
            continue
        counts = source.get("token_counts", {})
        target = int(row["target_token_id"])
        frequencies = [int(counts.get(int(token), 0)) for token in row["candidate_token_ids"]]
        row.update({
            "source_metadata_status": "ok",
            "source_token_present": bool(counts.get(target, 0)),
            "source_token_frequency": int(counts.get(target, 0)),
            "source_stratum": "copyable" if counts.get(target, 0) else "source_novel",
            "candidate_source_frequencies": frequencies,
            "candidate_source_present": [value > 0 for value in frequencies],
        })
        annotated.append(row)
    return annotated


def annotate_source_phrase_rows(
    rows: Iterable[Mapping[str, Any]],
    source_index: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach exact source support for DFlash-prefix-plus-candidate phrases."""

    annotated = list(rows)
    for block in _blocks(annotated):
        history: list[int] = []
        for raw in block:
            row = raw if isinstance(raw, dict) else dict(raw)
            source = source_index.get(str(row.get("sample_id")), {})
            ngram_counts = source.get("ngram_counts", {})
            supports: list[float] = []
            for candidate in [int(token) for token in row["candidate_token_ids"]]:
                support = 0.0
                if history:
                    support = max(support, float(ngram_counts.get(2, {}).get(tuple(history[-1:] + [candidate]), 0)))
                if len(history) >= 2:
                    support = max(support, float(ngram_counts.get(3, {}).get(tuple(history[-2:] + [candidate]), 0)))
                supports.append(math.log1p(support) if support > 0 else 0.0)
            row["candidate_source_phrase_scores"] = supports
            history.append(int(row["dflash_selected_token_id"]))
    return annotated


def _ok(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


def _rank(row: Mapping[str, Any], token: int) -> int | None:
    try:
        return [int(value) for value in row["candidate_token_ids"]].index(int(token)) + 1
    except ValueError:
        return None


def _entropy(values: Sequence[float]) -> float | None:
    if not values:
        return None
    maximum = max(float(value) for value in values)
    weights = [math.exp(float(value) - maximum) for value in values]
    total = sum(weights)
    if total <= 0.0:
        return None
    return -sum((weight / total) * math.log(weight / total) for weight in weights if weight > 0.0)


def _document_count(rows: Iterable[Mapping[str, Any]]) -> int:
    return len({str(row["document_id"]) for row in rows})


def _row_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    usable = _ok(rows)
    if not usable:
        return {"rows": 0, "documents": 0, "status": "empty"}
    top16 = [row for row in usable if _rank(row, int(row["target_token_id"])) is not None and _rank(row, int(row["target_token_id"])) <= 16]
    ranks = [_rank(row, int(row["target_token_id"])) for row in top16]
    ranks = [int(rank) for rank in ranks if rank is not None]
    recall1 = sum(int(_rank(row, int(row["target_token_id"])) == 1) for row in usable) / len(usable)
    recall16 = len(top16) / len(usable)
    return {
        "status": "ok",
        "rows": len(usable),
        "documents": _document_count(usable),
        "recall_at_1": recall1,
        "recall_at_16": recall16,
        "mrr_top16": sum(1.0 / rank for rank in ranks) / len(ranks) if ranks else None,
        "mean_target_rank_top16": sum(ranks) / len(ranks) if ranks else None,
        "source_support_rate": sum(bool(row.get("source_token_present")) for row in usable) / len(usable),
    }


def _block_stratum(block: Sequence[Mapping[str, Any]]) -> str:
    known = [row for row in block if row.get("source_stratum") in {"copyable", "source_novel"}]
    if not known:
        return "unknown"
    novel_fraction = sum(row.get("source_stratum") == "source_novel" for row in known) / len(known)
    return "source_novel_dominant" if novel_fraction >= 0.5 else "copyable_dominant"


def _block_group_metrics(blocks: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    if not blocks:
        return {"status": "empty", "blocks": 0, "documents": 0}
    observed = [_observed_acceptance(block) for block in blocks]
    oracle = [prefix_oracle_length(block, 16) for block in blocks]
    mat_d = sum(observed) / len(observed)
    mat_o16 = sum(oracle) / len(oracle)
    return {
        "status": "ok",
        "blocks": len(blocks),
        "documents": len({str(row["document_id"]) for block in blocks for row in block}),
        "mat_d": mat_d,
        "mat_o16": mat_o16,
        "oracle_headroom": mat_o16 - mat_d,
        "oracle_ratio": mat_o16 / mat_d if mat_d > 0 else None,
    }


def analyze_source_strata(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Run E6 token-level and block-stratified copyable/novel analysis."""

    usable = _ok(rows)
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        by_dataset[str(row.get("task_regime", row.get("dataset", "other")))].append(row)
    output: dict[str, Any] = {
        "status": "ok" if usable else "unavailable",
        "experiment": "E6",
        "rows": len(usable),
        "source_label": "exact_token_id_support",
        "datasets": {},
    }
    if not usable:
        output["reason"] = "no_valid_trace_rows"
        return output
    for dataset, dataset_rows in sorted(by_dataset.items()):
        strata: dict[str, Any] = {}
        for stratum in ("copyable", "source_novel", "unknown"):
            selected = [row for row in dataset_rows if row.get("source_stratum") == stratum]
            strata[stratum] = _row_summary(selected)
        blocks = _blocks(dataset_rows)
        block_groups: dict[str, list[Sequence[Mapping[str, Any]]]] = defaultdict(list)
        for block in blocks:
            block_groups[_block_stratum(block)].append(block)
        output["datasets"][dataset] = {
            "rows": len(dataset_rows),
            "documents": _document_count(dataset_rows),
            "strata": strata,
            "block_strata": {name: _block_group_metrics(group) for name, group in sorted(block_groups.items())},
        }
    return output


def _source_score(row: Mapping[str, Any], index: int) -> float:
    values = row.get("candidate_source_frequencies")
    if not isinstance(values, list) or index >= len(values):
        return 0.0
    maximum = max((math.log1p(max(0, int(value))) for value in values), default=0.0)
    if maximum <= 0.0:
        return 0.0
    return math.log1p(max(0, int(values[index]))) / maximum


def _source_phrase_score(row: Mapping[str, Any], index: int) -> float:
    values = row.get("candidate_source_phrase_scores")
    if not isinstance(values, list) or index >= len(values):
        return 0.0
    maximum = max((float(value) for value in values), default=0.0)
    return float(values[index]) / maximum if maximum > 0.0 else 0.0


def _source_semantic_score(row: Mapping[str, Any], index: int) -> float:
    values = row.get("candidate_source_semantic_scores")
    if not isinstance(values, list) or index >= len(values):
        return 0.0
    numeric = [float(value) for value in values]
    minimum = min(numeric, default=0.0)
    maximum = max(numeric, default=0.0)
    if maximum <= minimum:
        return 0.0
    return (float(values[index]) - minimum) / (maximum - minimum)


def select_diagnostic_candidate(
    row: Mapping[str, Any],
    *,
    mode: str = "unary",
    source_weight: float = 0.0,
) -> int:
    """Select only among recorded candidates using a deterministic diagnostic score."""

    candidates = [int(token) for token in row["candidate_token_ids"]]
    if not candidates:
        raise ValueError("candidate list must not be empty")
    dflash_selected = row.get("dflash_selected_token_id")
    if mode == "unary" or (
        mode in {"u_plus_source", "u_plus_source_phrase", "u_plus_source_semantic"}
        and float(source_weight) == 0.0
    ):
        if dflash_selected is not None and int(dflash_selected) in candidates:
            return int(dflash_selected)
    logits = row.get("candidate_logits")
    if not isinstance(logits, list) or len(logits) != len(candidates):
        logits = [float(len(candidates) - index) for index in range(len(candidates))]
    maximum = max(float(value) for value in logits)
    minimum = min(float(value) for value in logits)
    scale = maximum - minimum
    selected_index = 0
    selected_score = float("-inf")
    for index, value in enumerate(logits):
        unary = (float(value) - minimum) / scale if scale > 0 else float(len(candidates) - index)
        if mode == "u_plus_source":
            source = _source_score(row, index)
        elif mode == "u_plus_source_phrase":
            source = _source_phrase_score(row, index)
        elif mode == "u_plus_source_semantic":
            source = _source_semantic_score(row, index)
        else:
            source = 0.0
        score = unary + float(source_weight) * source
        key = (score, unary, -index)
        selected_key = (selected_score, 0.0, 0)
        if key > selected_key:
            selected_index = index
            selected_score = score
    return candidates[selected_index]


def _selected_prefix(block: Sequence[Mapping[str, Any]], *, mode: str, source_weight: float) -> int:
    selected = [select_diagnostic_candidate(row, mode=mode, source_weight=source_weight) for row in block]
    target = [int(row["target_token_id"]) for row in block]
    return prefix_match_length(selected, target)


def _ladder_dataset(
    rows: Sequence[Mapping[str, Any]],
    lambda_values: Sequence[float],
    *,
    mode: str = "u_plus_source",
) -> dict[str, Any]:
    blocks = _blocks(rows)
    if not blocks:
        return {"status": "empty"}
    mat_d = sum(_observed_acceptance(block) for block in blocks) / len(blocks)
    mat_o16 = sum(prefix_oracle_length(block, 16) for block in blocks) / len(blocks)
    result: dict[str, Any] = {
        "status": "ok",
        "rows": len(rows),
        "documents": _document_count(rows),
        "blocks": len(blocks),
        "mat_d": mat_d,
        "mat_oracle": mat_o16,
        "oracle_headroom": mat_o16 - mat_d,
        "row_summary": _row_summary(rows),
        "lambda_results": {},
    }
    for value in lambda_values:
        weight = float(value)
        selected_prefix = [_selected_prefix(block, mode=mode, source_weight=weight) for block in blocks]
        mat_selected = sum(selected_prefix) / len(selected_prefix)
        result["lambda_results"][str(weight)] = {
            "source_weight": weight,
            "mat_selected": mat_selected,
            "recovery_rho": (
                (mat_selected - mat_d) / (mat_o16 - mat_d)
                if mat_o16 > mat_d else None
            ),
            "selected_target_rate": sum(
                int(select_diagnostic_candidate(row, mode=mode, source_weight=weight) == int(row["target_token_id"]))
                for row in rows
            ) / len(rows) if rows else None,
        }
    return result


def analyze_source_ladder(
    rows: Iterable[Mapping[str, Any]],
    *,
    lambda_values: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """Run E7/E8 unary-vs-source lexical diagnostics on frozen lattice states."""

    usable = _ok(rows)
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        by_dataset[str(row.get("task_regime", row.get("dataset", "other")))].append(row)
    output: dict[str, Any] = {
        "status": "ok" if usable else "unavailable",
        "experiment": "E7_E8",
        "selector_scope": "frozen_recorded_top16_lattice",
        "lambda_values": [float(value) for value in lambda_values],
        "datasets": {},
    }
    if not usable:
        output["reason"] = "no_valid_trace_rows"
        return output
    for dataset, dataset_rows in sorted(by_dataset.items()):
        result = _ladder_dataset(dataset_rows, lambda_values)
        phrase_result = _ladder_dataset(dataset_rows, lambda_values, mode="u_plus_source_phrase")
        result["phrase_lambda_results"] = phrase_result.get("lambda_results", {})
        result["phrase_score_available"] = any(
            isinstance(row.get("candidate_source_phrase_scores"), list)
            for row in dataset_rows
        )
        if any(isinstance(row.get("candidate_source_semantic_scores"), list) for row in dataset_rows):
            semantic_result = _ladder_dataset(
                dataset_rows,
                lambda_values,
                mode="u_plus_source_semantic",
            )
            result["semantic_lambda_results"] = semantic_result.get("lambda_results", {})
            result["semantic_score_available"] = True
        output["datasets"][dataset] = result
    return output


def _best_lambda(rows: Sequence[Mapping[str, Any]], lambda_values: Sequence[float]) -> float:
    result = _ladder_dataset(rows, lambda_values)
    candidates = result.get("lambda_results", {})
    if not candidates:
        return float(lambda_values[0])
    return max(
        (float(key) for key, value in candidates.items()),
        key=lambda key: (candidates[str(key)].get("recovery_rho") is not None, candidates[str(key)].get("recovery_rho") or float("-inf"), -key),
    )


def analyze_leave_one_dataset_out(
    rows: Iterable[Mapping[str, Any]],
    *,
    lambda_values: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """Tune a fixed source-weight on two datasets and evaluate on the third."""

    usable = _ok(rows)
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        dataset = str(row.get("task_regime", row.get("dataset", "other")))
        if dataset in {"cnn_dm", "govreport", "multi_news"}:
            by_dataset[dataset].append(row)
    output: dict[str, Any] = {"status": "ok", "experiment": "E10", "protocol": "leave_one_dataset_out", "folds": {}}
    for held_out in sorted(by_dataset):
        train = [row for name, group in by_dataset.items() if name != held_out for row in group]
        test = by_dataset[held_out]
        weight = _best_lambda(train, lambda_values)
        train_result = _ladder_dataset(train, (weight,))
        test_result = _ladder_dataset(test, (weight,))
        output["folds"][held_out] = {
            "train_datasets": [name for name in sorted(by_dataset) if name != held_out],
            "held_out_dataset": held_out,
            "selected_source_weight": weight,
            "train": train_result,
            "test": test_result,
        }
    if not output["folds"]:
        output.update({"status": "unavailable", "reason": "insufficient_summarization_datasets"})
    return output


def analyze_leave_one_dataset_out_from_ladder_metrics(
    ladder_metrics: Mapping[str, Mapping[str, Any]],
    *,
    lambda_values: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0),
) -> dict[str, Any]:
    """Run E10 from compact per-dataset E7 artifacts.

    This avoids loading the 110K-row trace a second time.  MAT values are
    combined by block count, preserving the weighting used by the main MAT
    analysis.
    """

    names = sorted(set(ladder_metrics) & {"cnn_dm", "govreport", "multi_news"})
    if len(names) < 3:
        return {
            "status": "unavailable",
            "experiment": "E10",
            "reason": "three_dataset_ladder_metrics_required",
        }
    output: dict[str, Any] = {
        "status": "ok",
        "experiment": "E10",
        "protocol": "leave_one_dataset_out_from_compact_ladder_metrics",
        "folds": {},
    }

    def build_folds(result_key: str) -> dict[str, Any]:
        if any(result_key not in ladder_metrics[name] for name in names):
            return {}
        folds: dict[str, Any] = {}
        for held_out in names:
            train_names = [name for name in names if name != held_out]
            train_metrics = [ladder_metrics[name] for name in train_names]
            train_blocks = sum(int(item.get("blocks", 0)) for item in train_metrics)

            def pooled_lambda(weight: float) -> dict[str, Any]:
                selected_key = str(float(weight))
                selected_total = sum(
                    float(item[result_key][selected_key]["mat_selected"]) * int(item.get("blocks", 0))
                    for item in train_metrics
                )
                d_total = sum(float(item["mat_d"]) * int(item.get("blocks", 0)) for item in train_metrics)
                oracle_total = sum(float(item["mat_oracle"]) * int(item.get("blocks", 0)) for item in train_metrics)
                mat_selected = selected_total / train_blocks
                mat_d = d_total / train_blocks
                mat_oracle = oracle_total / train_blocks
                return {
                    "mat_selected": mat_selected,
                    "mat_d": mat_d,
                    "mat_oracle": mat_oracle,
                    "recovery_rho": (mat_selected - mat_d) / (mat_oracle - mat_d) if mat_oracle > mat_d else None,
                }

            candidates = {str(float(weight)): pooled_lambda(float(weight)) for weight in lambda_values}
            best_weight = max(
                (float(key) for key in candidates),
                key=lambda weight: (
                    candidates[str(weight)]["recovery_rho"] is not None,
                    candidates[str(weight)]["recovery_rho"] or float("-inf"),
                    -weight,
                ),
            )
            held_out_metrics = ladder_metrics[held_out]
            held_out_result = held_out_metrics[result_key][str(float(best_weight))]
            folds[held_out] = {
                "train_datasets": train_names,
                "held_out_dataset": held_out,
                "selected_source_weight": best_weight,
                "train_candidates": candidates,
                "test": {
                    "mat_d": held_out_metrics["mat_d"],
                    "mat_oracle": held_out_metrics["mat_oracle"],
                    "mat_selected": held_out_result["mat_selected"],
                    "recovery_rho": held_out_result["recovery_rho"],
                    "blocks": held_out_metrics["blocks"],
                },
            }
        return folds

    output["folds"] = build_folds("lambda_results")
    phrase_folds = build_folds("phrase_lambda_results")
    if phrase_folds:
        output["phrase_folds"] = phrase_folds
    semantic_folds = build_folds("semantic_lambda_results")
    if semantic_folds:
        output["semantic_folds"] = semantic_folds
    return output


def _softmax(values: Sequence[float]) -> list[float]:
    maximum = max(float(value) for value in values)
    weights = [math.exp(float(value) - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights] if total > 0 else [1.0 / len(values)] * len(values)


def _js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    midpoint = [(a + b) / 2.0 for a, b in zip(left, right)]
    def kl(values: Sequence[float]) -> float:
        return sum(value * math.log(value / mid) for value, mid in zip(values, midpoint) if value > 0 and mid > 0)
    return 0.5 * (kl(left) + kl(right))


def analyze_target_near_ties(rows: Iterable[Mapping[str, Any]], *, near_tie_margin: float = 0.5) -> dict[str, Any]:
    """Analyze target-vs-DFlash candidate logits when E9 fields are available."""

    usable = [row for row in rows if row.get("status", "ok") == "ok" and isinstance(row.get("target_candidate_logits"), list)]
    by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        by_dataset[str(row.get("task_regime", row.get("dataset", "other")))].append(row)
    output: dict[str, Any] = {
        "status": "ok" if usable else "unavailable",
        "experiment": "E9",
        "near_tie_margin": float(near_tie_margin),
        "datasets": {},
    }
    if not usable:
        output["reason"] = "missing_target_candidate_logits"
        return output
    for dataset, dataset_rows in sorted(by_dataset.items()):
        measured: list[dict[str, Any]] = []
        for row in dataset_rows:
            draft_logits = row.get("candidate_logits")
            target_logits = row.get("target_candidate_logits")
            candidates = row.get("candidate_token_ids")
            if not isinstance(draft_logits, list) or not isinstance(target_logits, list) or len(draft_logits) != len(target_logits):
                continue
            target = int(row["target_token_id"])
            if target not in [int(value) for value in candidates]:
                continue
            target_index = [int(value) for value in candidates].index(target)
            dflash_selected = int(row.get("dflash_selected_token_id", candidates[0]))
            dflash_index = (
                [int(value) for value in candidates].index(dflash_selected)
                if dflash_selected in [int(value) for value in candidates]
                else 0
            )
            target_probs = _softmax([float(value) for value in target_logits])
            draft_probs = _softmax([float(value) for value in draft_logits])
            target_logit = float(target_logits[target_index])
            dflash_logit = float(target_logits[dflash_index])
            measured.append({
                "source_stratum": row.get("source_stratum", "unknown"),
                "target_margin": target_logit - dflash_logit,
                "target_probability": target_probs[target_index],
                "dflash_selected_is_target": dflash_selected == target,
                "target_entropy": _entropy([float(value) for value in target_logits]),
                "js_divergence": _js_divergence(draft_probs, target_probs),
            })
        if not measured:
            output["datasets"][dataset] = {"status": "empty"}
            continue
        output["datasets"][dataset] = {
            "status": "ok",
            "rows": len(measured),
            "near_tie_rate": sum(item["target_margin"] <= near_tie_margin for item in measured) / len(measured),
            "mismatch_rows": sum(not item["dflash_selected_is_target"] for item in measured),
            "mean_mismatch_target_margin": (
                sum(item["target_margin"] for item in measured if not item["dflash_selected_is_target"])
                / max(1, sum(not item["dflash_selected_is_target"] for item in measured))
            ),
            "mismatch_near_tie_rate": (
                sum(
                    not item["dflash_selected_is_target"] and item["target_margin"] <= near_tie_margin
                    for item in measured
                ) / max(1, sum(not item["dflash_selected_is_target"] for item in measured))
            ),
            "mean_target_margin": sum(item["target_margin"] for item in measured) / len(measured),
            "mean_target_probability": sum(item["target_probability"] for item in measured) / len(measured),
            "mean_target_entropy": sum(item["target_entropy"] for item in measured if item["target_entropy"] is not None) / len(measured),
            "mean_js_divergence": sum(item["js_divergence"] for item in measured) / len(measured),
            "strata": {
                stratum: {
                    "rows": sum(item["source_stratum"] == stratum for item in measured),
                    "mismatch_rows": sum(
                        item["source_stratum"] == stratum and not item["dflash_selected_is_target"]
                        for item in measured
                    ),
                    "near_tie_rate": (
                        sum(item["source_stratum"] == stratum and item["target_margin"] <= near_tie_margin for item in measured)
                        / max(1, sum(item["source_stratum"] == stratum for item in measured))
                    ),
                    "mismatch_near_tie_rate": (
                        sum(
                            item["source_stratum"] == stratum
                            and not item["dflash_selected_is_target"]
                            and item["target_margin"] <= near_tie_margin
                            for item in measured
                        )
                        / max(
                            1,
                            sum(
                                item["source_stratum"] == stratum
                                and not item["dflash_selected_is_target"]
                                for item in measured
                            ),
                        )
                    ),
                }
                for stratum in ("copyable", "source_novel")
            },
        }
    return output
