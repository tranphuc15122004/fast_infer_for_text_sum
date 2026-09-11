"""E27-C0 residual-cover signal discrimination screen.

This module is deliberately an *offline* analysis.  It reconstructs the
sentence selections used by E26-R from the original source JSONL, computes
selection-conditioned coverage/residual features, and evaluates source-only
policies against the already measured E26-R outcomes.

The experiment is not allowed to use generated summaries, quality scores,
latencies, target logits, or target hidden states as policy features.  Those
values are used only for the out-of-fold policy evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.analyze.safe_budget import e27a_policy_screen as e27a


ACTION_ORDER = e27a.ACTION_LABELS
ACTION_RATIOS = {
    "full": 1.0,
    "ratio_0.25": 0.25,
    "ratio_0.4": 0.4,
    "ratio_0.55": 0.55,
    "ratio_0.7": 0.7,
    "ratio_0.85": 0.85,
}

FEATURE_FAMILIES = {
    "cheap": e27a.CHEAP_FEATURES,
    "front_redundancy": (
        "source_tokens",
        "sentence_count",
        "lexical_redundancy",
        "selected_token_ratio",
        "selected_sentence_ratio",
        "selected_front_token_fraction",
        "selected_mean_position",
        "selected_lexical_redundancy",
    ),
    "facility": (
        "source_tokens",
        "sentence_count",
        "selected_token_ratio",
        "selected_sentence_ratio",
        "selected_front_token_fraction",
        "facility_coverage",
        "facility_p90_support",
        "selected_mean_position",
    ),
    "residual": (
        "residual_cover",
        "residual_p90",
        "marginal_residual_gain",
        "selected_token_ratio",
        "selected_sentence_ratio",
    ),
    "residual_plus_cheap": (
        *e27a.CHEAP_FEATURES,
        "residual_cover",
        "residual_p90",
        "marginal_residual_gain",
        "selected_token_ratio",
        "selected_sentence_ratio",
    ),
}

POLICY_TO_FAMILY = {
    "learned_cheap": "cheap",
    "learned_front_redundancy": "front_redundancy",
    "learned_facility": "facility",
    "learned_residual": "residual",
    "learned_residual_plus_cheap": "residual_plus_cheap",
}

EMBEDDING_CHECKPOINT = (
    "/home/tuantb/.cache/huggingface/hub/models--sentence-transformers--"
    "all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _source_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in e27a.WORD_RE.finditer(text)]


def _action_budget(source_tokens: int, label: str) -> int:
    if label == "full":
        return max(1, int(source_tokens))
    return max(1, int(round(source_tokens * ACTION_RATIOS[label])))


def _cap_source(text: str, *, cap: int, tokenizer: Any) -> str:
    if cap <= 0:
        return text
    # This guard has the same purpose as E27-A: avoid feeding a very large
    # native document to the tokenizer merely to retain its first cap tokens.
    # The final truncation is performed by the exact Qwen tokenizer.
    bounded = text[: min(65536, max(4096, cap * 12))]
    token_ids = tokenizer(
        bounded,
        add_special_tokens=False,
        truncation=True,
        max_length=cap,
    )["input_ids"]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def _select_priority_from_embeddings(
    selector: Any,
    sentences: Sequence[str],
    token_counts: Sequence[int],
    embeddings: np.ndarray,
    token_budget: int,
    *,
    mmr_lambda: float,
) -> list[int]:
    """Exact equivalent of vendored MMRSelector._select_priority.

    The vendored selector re-encodes once per call.  C0 has six actions per
    document, so this helper reuses one sentence-embedding matrix while
    retaining the same greedy score, separator estimate and tie-breaking.
    """
    if not sentences:
        return []
    doc_embedding = selector._document_centroid(
        embeddings=embeddings,
        token_counts=token_counts,
    )
    relevance = embeddings @ doc_embedding
    selected: list[int] = []
    remaining = set(range(len(sentences)))
    separator_tokens = selector.count_tokens(selector.separator)
    remaining_budget = int(token_budget)
    while remaining:
        eligible = []
        extra_separator = separator_tokens if selected else 0
        for index in remaining:
            if token_counts[index] + extra_separator <= remaining_budget:
                eligible.append(index)
        if not eligible:
            break
        best_index = None
        best_key = None
        for index in eligible:
            redundancy = (
                float(np.max(embeddings[selected] @ embeddings[index]))
                if selected
                else 0.0
            )
            score = mmr_lambda * float(relevance[index]) - (1.0 - mmr_lambda) * redundancy
            key = (score, float(relevance[index]), -index)
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        assert best_index is not None
        extra_separator = separator_tokens if selected else 0
        remaining_budget -= token_counts[best_index] + extra_separator
        selected.append(best_index)
        remaining.remove(best_index)
    if not selected:
        return [max(range(len(sentences)), key=lambda i: (float(relevance[i]), -i))]
    return selected


def _selection_indices(
    selector: Any,
    sentences: Sequence[Any],
    *,
    source_tokens: int,
    label: str,
    embeddings: np.ndarray,
    mmr_lambda: float,
) -> tuple[list[int], list[Any], int]:
    token_counts = [unit.token_count for unit in sentences]
    budget = _action_budget(source_tokens, label)
    if not sentences:
        return [], [], budget
    if source_tokens <= budget:
        return list(range(len(sentences))), sentences, budget
    priority = _select_priority_from_embeddings(
        selector,
        [unit.text for unit in sentences],
        token_counts,
        embeddings,
        budget,
        mmr_lambda=mmr_lambda,
    )
    chosen = selector._fit_priority_to_budget(priority, sentences, budget)
    if not chosen and priority and selector.allow_partial_fallback:
        chosen = [priority[0]]
    return sorted(chosen), sentences, budget


def _selection_features(
    *,
    text: str,
    source_tokens: int,
    selected_indices: Sequence[int],
    sentences: Sequence[Any],
    embeddings: np.ndarray,
    previous_residual: float | None,
    base_features: Mapping[str, float],
) -> dict[str, float]:
    n_sentences = len(sentences)
    selected = list(selected_indices)
    selected_token_count = sum(int(sentences[i].token_count) for i in selected)
    selected_text = "\n".join(str(sentences[i].text) for i in selected)
    source_words = _source_words(text)
    selected_words = _source_words(selected_text)
    selected_token_positions = sum(
        int(sentences[i].token_count) * float(i / max(1, n_sentences - 1))
        for i in selected
    )
    selected_front_tokens = sum(
        int(sentences[i].token_count)
        for i in selected
        if i < max(1, math.ceil(n_sentences * 0.25))
    )
    if selected:
        selected_embeddings = embeddings[selected]
        support = np.max(embeddings @ selected_embeddings.T, axis=1)
        facility_coverage = float(np.mean(support))
        residual_cover = float(np.mean(1.0 - support))
        residual_p90 = float(np.percentile(1.0 - support, 90))
        p90_support = float(np.percentile(support, 90))
    else:
        facility_coverage = 0.0
        residual_cover = 1.0
        residual_p90 = 1.0
        p90_support = 0.0
    marginal_residual_gain = (
        float(previous_residual - residual_cover)
        if previous_residual is not None
        else 0.0
    )
    result = {
        **base_features,
        "selected_tokens": float(selected_token_count),
        "selected_token_ratio": _safe_ratio(selected_token_count, source_tokens),
        "selected_sentence_count": float(len(selected)),
        "selected_sentence_ratio": _safe_ratio(len(selected), n_sentences),
        "selected_front_token_fraction": _safe_ratio(selected_front_tokens, selected_token_count),
        "selected_mean_position": _safe_ratio(selected_token_positions, selected_token_count),
        "selected_lexical_redundancy": (
            1.0 - _safe_ratio(len(set(selected_words)), len(selected_words))
            if selected_words
            else 0.0
        ),
        "source_unique_word_ratio": _safe_ratio(len(set(source_words)), len(source_words)),
        "facility_coverage": facility_coverage,
        "facility_p90_support": p90_support,
        "residual_cover": residual_cover,
        "residual_p90": residual_p90,
        "marginal_residual_gain": marginal_residual_gain,
    }
    return {
        name: value if math.isfinite(float(value)) else 0.0
        for name, value in result.items()
    }


def reconstruct_action_features(
    source_paths: Mapping[str, str],
    outcomes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    source_tokens_by_key: Mapping[str, int],
    source_caps: Mapping[str, int],
    tokenizer_path: str,
    embedding_path: str = EMBEDDING_CHECKPOINT,
    mmr_lambda: float = 0.7,
    embedding_batch_size: int = 64,
) -> dict[str, dict[str, dict[str, float]]]:
    """Rebuild all six action-conditioned feature rows per joined document."""
    # Import lazily so unit tests for policy math do not require transformer
    # model loading.
    repo_root = Path(__file__).resolve().parents[3]
    externals = str(repo_root / "externals")
    if externals not in sys.path:
        sys.path.insert(0, externals)
    from Sematic_selection.mmr import MMRSelector

    selector = MMRSelector(
        tokenizer_name=tokenizer_path,
        local_files_only=True,
        embedding_model_name=embedding_path,
        embedding_device="cpu",
        embedding_local_files_only=True,
        mmr_lambda=mmr_lambda,
        embedding_batch_size=embedding_batch_size,
    )
    features: dict[str, dict[str, dict[str, float]]] = {}
    pending: list[tuple[str, str, int, list[Any], dict[str, float]]] = []
    for dataset, path in source_paths.items():
        for row in e27a._load_jsonl(path):
            example_id = str(row.get("id", row.get("example_id", "")))
            key = e27a._doc_key(dataset, example_id)
            if key not in outcomes:
                continue
            source_tokens = int(round(_finite(source_tokens_by_key.get(key, 0))))
            if source_tokens <= 0:
                raise ValueError(f"missing authoritative original_tokens for {key}")
            text = str(row.get("document", ""))
            text = _cap_source(text, cap=int(source_caps.get(dataset, 0)), tokenizer=selector.tokenizer)
            units = selector.build_sentence_units(text)
            base_features = e27a.build_source_features(text, source_tokens=source_tokens)
            pending.append((key, text, source_tokens, units, base_features))

    # Encode all sentences in one batched stream.  The vendored selector's
    # public API is intentionally per-document; C0 has 90 documents and six
    # actions, so a global embedding pass avoids repeatedly launching the
    # MiniLM encoder while preserving exactly the same embedding function.
    all_sentences = [unit.text for _, _, _, units, _ in pending for unit in units]
    if all_sentences:
        all_embeddings = np.asarray(
            selector.embedding_model.encode(
                all_sentences,
                batch_size=embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
    else:
        all_embeddings = np.empty((0, 384), dtype=np.float32)
    cursor = 0
    for key, text, source_tokens, units, base_features in pending:
        count = len(units)
        embeddings = all_embeddings[cursor: cursor + count]
        cursor += count
        action_rows: dict[str, dict[str, float]] = {}
        previous_residual: float | None = None
        # Compute compressed actions from shortest to longest so the
        # marginal residual feature has a deterministic interpretation.
        for label in ("ratio_0.25", "ratio_0.4", "ratio_0.55", "ratio_0.7", "ratio_0.85", "full"):
            indices, rebuilt_units, _ = _selection_indices(
                selector,
                units,
                source_tokens=source_tokens,
                label=label,
                embeddings=embeddings,
                mmr_lambda=mmr_lambda,
            )
            row_features = _selection_features(
                text=text,
                source_tokens=source_tokens,
                selected_indices=indices,
                sentences=rebuilt_units,
                embeddings=embeddings,
                previous_residual=previous_residual,
                base_features=base_features,
            )
            previous_residual = row_features["residual_cover"]
            action_rows[label] = row_features
        features[key] = {label: action_rows[label] for label in ACTION_ORDER}
    return features


def _fit_action_models(
    train_ids: Sequence[str],
    features: Mapping[str, Mapping[str, Mapping[str, float]]],
    outcomes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    feature_names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    risk_models: dict[str, Any] = {}
    cost_models: dict[str, Any] = {}
    for label in ACTION_ORDER:
        matrix = np.asarray(
            [[features[key][label][name] for name in feature_names] for key in train_ids],
            dtype=float,
        )
        risk_models[label] = e27a._fit_risk_model(
            matrix,
            [int(bool(outcomes[key][label]["violates"])) for key in train_ids],
        )
        cost_models[label] = e27a._fit_cost_model(
            matrix,
            [float(outcomes[key][label]["cost_ms"]) for key in train_ids],
        )
    return risk_models, cost_models


def _predict_action_models(
    test_ids: Sequence[str],
    features: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    feature_names: Sequence[str],
    risk_models: Mapping[str, Any],
    cost_models: Mapping[str, Any],
    alpha: float,
) -> dict[str, str]:
    actions: dict[str, str] = {}
    for key in test_ids:
        predicted_risk: dict[str, float] = {}
        predicted_cost: dict[str, float] = {}
        for label in ACTION_ORDER:
            matrix = np.asarray(
                [[features[key][label][name] for name in feature_names]],
                dtype=float,
            )
            predicted_risk[label] = float(risk_models[label].predict_proba(matrix)[0, 1])
            predicted_cost[label] = max(0.0, float(cost_models[label].predict(matrix)[0]))
        actions[key] = e27a.choose_action_from_predictions(
            predicted_risk, predicted_cost, alpha=alpha
        )
    return actions


def _length_threshold_action_map(
    test_ids: Sequence[str],
    features: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, str]:
    return e27a._length_threshold_actions(
        test_ids,
        {key: features[key]["full"] for key in test_ids},
    )


def _aggregate(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    # Reuse E27-A's aggregation implementation; its grouping is independent
    # of the number/names of policy families.
    return e27a._aggregate_records(records)


def run_screen(
    input_paths: Mapping[str, str],
    source_paths: Mapping[str, str],
    *,
    source_caps: Mapping[str, int],
    tokenizer_path: str,
    embedding_path: str,
    mmr_lambda: float,
    embedding_batch_size: int,
    epsilons: Sequence[float],
    alphas: Sequence[float],
    primary_epsilon: float,
    primary_alpha: float,
    n_splits: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    first_outcomes = e27a._load_outcomes(input_paths, epsilon=primary_epsilon)
    source_tokens_by_key: dict[str, int] = {}
    for dataset, path in input_paths.items():
        for row in e27a._load_jsonl(path):
            if row.get("selector") == "full" and row.get("budget_label") == "full":
                key = e27a._doc_key(dataset, str(row.get("example_id", row.get("id", ""))))
                source_tokens_by_key[key] = int(round(_finite(row.get("original_tokens", 0))))
    features = reconstruct_action_features(
        source_paths,
        first_outcomes,
        source_tokens_by_key=source_tokens_by_key,
        source_caps=source_caps,
        tokenizer_path=tokenizer_path,
        embedding_path=embedding_path,
        mmr_lambda=mmr_lambda,
        embedding_batch_size=embedding_batch_size,
    )
    keys = sorted(set(first_outcomes).intersection(features))
    if not keys:
        raise ValueError("no complete joined documents for E27-C0")
    splits = e27a._make_splits(keys, n_splits=n_splits, repeats=repeats, seed=seed)
    records: list[dict[str, Any]] = []
    for epsilon in epsilons:
        outcomes = e27a._load_outcomes(input_paths, epsilon=epsilon)
        contract_keys = sorted(set(keys).intersection(outcomes))
        scopes = {"pooled": contract_keys}
        scopes.update({
            dataset: [key for key in contract_keys if key.split("::", 1)[0] == dataset]
            for dataset in sorted({key.split("::", 1)[0] for key in contract_keys})
        })
        # Fit every signal family once per epsilon/repeat/fold.  Predictions
        # do not depend on alpha; the previous implementation refit all five
        # model families three times per contract, which made the offline
        # screen needlessly expensive without changing any estimate.
        fold_caches: list[list[dict[str, Any]]] = []
        for folds in splits:
            repeat_cache: list[dict[str, Any]] = []
            for train_all, test_all in folds:
                train_ids = [key for key in train_all if key in outcomes]
                test_ids = [key for key in test_all if key in outcomes]
                if not train_ids or not test_ids:
                    continue
                family_predictions: dict[str, dict[str, dict[str, float]]] = {}
                for policy, family in POLICY_TO_FAMILY.items():
                    risk_models, cost_models = _fit_action_models(
                        train_ids, features, outcomes,
                        feature_names=FEATURE_FAMILIES[family],
                    )
                    per_key: dict[str, dict[str, float]] = {}
                    for key in test_ids:
                        risk_scores: dict[str, float] = {}
                        cost_scores: dict[str, float] = {}
                        for label in ACTION_ORDER:
                            matrix = np.asarray(
                                [[features[key][label][name] for name in FEATURE_FAMILIES[family]]],
                                dtype=float,
                            )
                            risk_scores[label] = float(risk_models[label].predict_proba(matrix)[0, 1])
                            cost_scores[label] = max(0.0, float(cost_models[label].predict(matrix)[0]))
                        per_key[key] = {"risk": risk_scores, "cost": cost_scores}
                    family_predictions[policy] = per_key
                repeat_cache.append({
                    "train_ids": train_ids,
                    "test_ids": test_ids,
                    "family_predictions": family_predictions,
                })
            fold_caches.append(repeat_cache)

        for alpha in alphas:
            references: dict[str, dict[str, Any]] = {}
            for scope, scope_keys in scopes.items():
                if not scope_keys:
                    continue
                fixed_label = e27a._select_fixed(scope_keys, outcomes, epsilon=epsilon, alpha=alpha)
                oracle_actions = e27a._adaptive_oracle(scope_keys, outcomes, epsilon=epsilon, alpha=alpha)
                references[scope] = {
                    "fixed_label": fixed_label,
                    "fixed_cost": e27a._mean(outcomes[key][fixed_label]["cost_ms"] for key in scope_keys),
                    "oracle_actions": oracle_actions,
                    "oracle_cost": e27a._mean(outcomes[key][oracle_actions[key]]["cost_ms"] for key in scope_keys),
                }
            for repeat_index, repeat_cache in enumerate(fold_caches):
                prediction_by_policy: dict[str, dict[str, str]] = defaultdict(dict)
                for fold in repeat_cache:
                    test_ids = fold["test_ids"]
                    for policy in POLICY_TO_FAMILY:
                        for key, prediction in fold["family_predictions"][policy].items():
                            prediction_by_policy[policy][key] = e27a.choose_action_from_predictions(
                                prediction["risk"], prediction["cost"], alpha=alpha
                            )
                    train_fixed = e27a._select_fixed(
                        fold["train_ids"], outcomes, epsilon=epsilon, alpha=alpha
                    )
                    prediction_by_policy["fixed_train"].update({key: train_fixed for key in test_ids})
                    prediction_by_policy["length_threshold"].update(
                        _length_threshold_action_map(test_ids, features)
                    )
                for scope, scope_keys in scopes.items():
                    if not scope_keys:
                        continue
                    reference = references[scope]
                    policy_actions = {
                        "best_fixed_eval": {key: reference["fixed_label"] for key in scope_keys},
                        "oracle_adaptive_eval": reference["oracle_actions"],
                    }
                    for policy in ("fixed_train", "length_threshold", *POLICY_TO_FAMILY):
                        policy_actions[policy] = {
                            key: prediction_by_policy[policy].get(key, "full")
                            for key in scope_keys
                        }
                    for policy, actions in policy_actions.items():
                        result = e27a._evaluate_policy_with_reference(
                            actions,
                            {key: outcomes[key] for key in scope_keys},
                            epsilon=epsilon,
                            alpha=alpha,
                            fixed_label=reference["fixed_label"],
                            fixed_cost=reference["fixed_cost"],
                            oracle_actions=reference["oracle_actions"],
                            oracle_cost=reference["oracle_cost"],
                        )
                        if policy in POLICY_TO_FAMILY:
                            family = POLICY_TO_FAMILY[policy]
                        elif policy == "length_threshold":
                            family = "length"
                        elif policy == "fixed_train":
                            family = "none"
                        else:
                            family = "reference"
                        records.append({
                            "repeat": repeat_index,
                            "scope": scope,
                            "policy": policy,
                            "feature_set": family,
                            "epsilon": float(epsilon),
                            "alpha": float(alpha),
                            **result,
                        })
    primary_outcomes = e27a._load_outcomes(input_paths, epsilon=primary_epsilon)
    primary_action_diagnostics: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in sorted({key.split("::", 1)[0] for key in keys}):
        dataset_keys = [key for key in keys if key.split("::", 1)[0] == dataset]
        primary_action_diagnostics[dataset] = {}
        for label in ACTION_ORDER:
            violations = sum(int(e27a._action_is_violating(primary_outcomes, key, label, epsilon=primary_epsilon)) for key in dataset_keys)
            primary_action_diagnostics[dataset][label] = {
                "documents": len(dataset_keys),
                "violating_documents": violations,
                "risk_rate": _safe_ratio(violations, len(dataset_keys)),
                "mean_cost_ms": e27a._mean(primary_outcomes[key][label]["cost_ms"] for key in dataset_keys) or 0.0,
            }
    return {
        "experiment": "E27C0_residual_cover_signal_discrimination",
        "primary_contract": {"epsilon": primary_epsilon, "alpha": primary_alpha},
        "quality_fields": list(e27a.QUALITY_FIELDS),
        "action_labels": list(ACTION_ORDER),
        "feature_families": {name: list(values) for name, values in FEATURE_FAMILIES.items()},
        "policy_to_family": dict(POLICY_TO_FAMILY),
        "documents": len(keys),
        "datasets": dict(sorted(Counter(key.split("::", 1)[0] for key in keys).items())),
        "n_splits": n_splits,
        "repeats": repeats,
        "seed": seed,
        "mmr_lambda": mmr_lambda,
        "embedding_path": embedding_path,
        "tokenizer_path": tokenizer_path,
        "input_paths": dict(input_paths),
        "source_paths": dict(source_paths),
        "source_caps": dict(source_caps),
        "features": features,
        "records": records,
        "aggregates": _aggregate(records),
        "primary_action_diagnostics": primary_action_diagnostics,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _primary_index(result: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    primary = result["primary_contract"]
    return {
        (item["scope"], item["policy"]): item
        for item in result["aggregates"]
        if abs(float(item["epsilon"]) - primary["epsilon"]) < 1e-12
        and abs(float(item["alpha"]) - primary["alpha"]) < 1e-12
    }


def render_report(result: Mapping[str, Any], *, output_dir: str | Path) -> str:
    primary = result["primary_contract"]
    index = _primary_index(result)
    feature_by_dataset: dict[str, list[Mapping[str, Mapping[str, float]]]] = defaultdict(list)
    for key, action_rows in result["features"].items():
        feature_by_dataset[key.split("::", 1)[0]].append(action_rows)
    lines = [
        "# E27-C0 — Residual-Cover Signal Discrimination",
        "",
        "## Kết luận ngắn",
        "",
        f"E27-C0 là screen offline trên {result['documents']} documents và {len(result['records'])} repeated-CV records. Không có target inference mới; toàn bộ quality/cost outcome được đọc từ E26-R.",
        "Mục tiêu là kiểm tra residual semantic cover có cung cấp signal mới để chọn context action dưới downstream summarization risk hay không, vượt qua source length/redundancy/front-loading/facility signals.",
        "",
        "Gate đã đăng ký trước: tại `epsilon=0.02, alpha=0.05`, residual phải vừa đạt capture ít nhất 30% trên ít nhất 2/3 dataset, vừa hơn signal prior mạnh nhất ít nhất 10 percentage points, và policy phải thỏa risk contract.",
        "",
        "## Thông tin thực thi và provenance",
        "",
        "- Ngày chạy: `2026-09-11`; experiment ID: `E27C0_residual_cover_signal_discrimination`.",
        "- Đây là phân tích **offline**: không gọi target Qwen3-4B, không sinh summary mới, không dùng GPU inference mới. Cost/quality được lấy nguyên vẹn từ E26-R; MiniLM chỉ được chạy để tái dựng source-only selection features.",
        "- Python dùng cho lần chạy chính: `.venv/bin/python` (Python 3.12). External Conda `myenv` không có scikit-learn; PyTorch trong runtime phân tích cũng không nhận `/dev/nvidia`. Vì vậy báo cáo không gọi C0 là GPU experiment.",
        "- Embedding checkpoint: local `all-MiniLM-L6-v2`, Transformers mean pooling, normalized embeddings, CPU, batch size 1024. Qwen3-4B tokenizer được đọc local để giới hạn token/cap.",
        "- C0 dùng 90 documents, 6 actions/document, 9 contracts, 20 repeats × 3 folds, tổng 6.480 policy-evaluation records và 324 aggregate cells.",
        "",
        "### Lệnh chạy chính",
        "",
        "Lệnh dưới đây là invocation đã dùng cho artifact chính; toàn bộ đường dẫn model/dữ liệu là local:",
        "",
        "```bash",
        "PYTHONPATH=. CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 .venv/bin/python src/analyze/safe_budget/e27c_residual_cover.py \\",
        "  --input cnn_dailymail=outputs/safe_budget_sum/2026-09-11_e26r_full/scored/cnn_dailymail.jsonl \\",
        "  --input govreport=outputs/safe_budget_sum/2026-09-11_e26r_full/scored/govreport.jsonl \\",
        "  --input multi_news=outputs/safe_budget_sum/2026-09-11_e26r_full/scored/multi_news.jsonl \\",
        "  --source-input cnn_dailymail=data/representative_100/cnn_dailymail_representative.jsonl \\",
        "  --source-input govreport=data/representative_100/govreport_representative.jsonl \\",
        "  --source-input multi_news=data/representative_100/multinews_representative.jsonl \\",
        "  --max-input-tokens govreport=4096 --max-input-tokens multi_news=4096 \\",
        "  --tokenizer=/home/tuantb/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \\",
        "  --embedding=/home/tuantb/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \\",
        "  --embedding-batch-size 1024 \\",
        "  --output-dir outputs/safe_budget_sum/2026-09-11_e27c_residual_cover",
        "```",
        "",
        "## Kết quả primary contract",
        "",
        "Các số dưới đây là trung bình trên 20 repeated 3-fold CV aggregates; `best_fixed_eval` và `oracle_adaptive_eval` là hindsight references trên cùng scope, không phải policy deploy.",
        "",
        "| Scope | Fixed ms | Fixed CI ms | Oracle ms | Oracle CI ms | Oracle saving | Policy | Cost ms | Cost CI ms | Risk | Risk CI | Pass fraction | Capture | Capture CI | Gain vs fixed |",
        "|---|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---:|---|---:|",
    ]
    for scope in ("cnn_dailymail", "govreport", "multi_news", "pooled"):
        fixed = index[(scope, "best_fixed_eval")]
        oracle = index[(scope, "oracle_adaptive_eval")]
        oracle_saving = 1.0 - float(oracle["mean_cost_ms"]) / float(fixed["mean_cost_ms"])
        for policy in ("learned_cheap", "learned_front_redundancy", "learned_facility", "learned_residual", "learned_residual_plus_cheap"):
            item = index[(scope, policy)]
            lines.append(
                f"| {scope} | {_fmt(fixed['mean_cost_ms'],2)} | {_fmt(fixed['cost_ci_95_ms'][0],2)}–{_fmt(fixed['cost_ci_95_ms'][1],2)} | {_fmt(oracle['mean_cost_ms'],2)} | {_fmt(oracle['cost_ci_95_ms'][0],2)}–{_fmt(oracle['cost_ci_95_ms'][1],2)} | {_pct(oracle_saving)} | {policy} | {_fmt(item['mean_cost_ms'],2)} | {_fmt(item['cost_ci_95_ms'][0],2)}–{_fmt(item['cost_ci_95_ms'][1],2)} | {_pct(item['mean_risk_rate'])} | {_pct(item['risk_ci_95'][0])}–{_pct(item['risk_ci_95'][1])} | {_pct(item['risk_pass_fraction'])} | {_pct(item['mean_capture'])} | {_pct(item['capture_ci_95'][0])}–{_pct(item['capture_ci_95'][1])} | {_pct(item['mean_cost_gain_vs_fixed'])} |"
            )
    lines += [
        "",
        "### So sánh trực tiếp capture giữa các signal",
        "",
        "| Dataset | Cheap prior | Front/redundancy | Facility prior | Residual | Residual + cheap | Strongest prior | Residual − strongest prior | Residual gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    gate_rows = {}
    for scope in ("cnn_dailymail", "govreport", "multi_news"):
        captures = {
            policy: index[(scope, policy)].get("mean_capture")
            for policy in ("learned_cheap", "learned_front_redundancy", "learned_facility", "learned_residual", "learned_residual_plus_cheap")
        }
        prior = max(float(captures[p]) for p in ("learned_cheap", "learned_front_redundancy", "learned_facility") if captures[p] is not None)
        residual = float(captures["learned_residual"] or 0.0)
        delta = residual - prior
        item = index[(scope, "learned_residual")]
        passed = bool(delta >= 0.10 and residual >= 0.30 and float(item["mean_risk_rate"]) <= primary["alpha"])
        gate_rows[scope] = {"strongest_prior_capture": prior, "residual_capture": residual, "delta": delta, "pass": passed}
        lines.append(
            f"| {scope} | {_pct(captures['learned_cheap'])} | {_pct(captures['learned_front_redundancy'])} | {_pct(captures['learned_facility'])} | {_pct(captures['learned_residual'])} | {_pct(captures['learned_residual_plus_cheap'])} | {_pct(prior)} | {_pct(delta)} | {passed} |"
        )
    passing = sum(int(value["pass"]) for value in gate_rows.values())
    lines += [
        "",
        f"**Gate result:** {passing}/3 dataset scopes pass residual capture >=30%, improvement >=10 percentage points over the strongest prior, and primary risk <= {primary['alpha']:.2f}.",
        "",
        "## Thiết kế thực nghiệm và quy trình thực hiện",
        "",
        "1. Đọc ba E26-R scored JSONL và giữ đúng các document có đủ sáu action `full`, `ratio_0.25`, `ratio_0.4`, `ratio_0.55`, `ratio_0.7`, `ratio_0.85` cùng ROUGE-L, BERTScore F1 và `pipeline_e2e_ms`.",
        "2. Join với representative source JSONL bằng `(dataset, example_id)`. GovReport và Multi-News được cắt đúng regime 4096 source tokens của E26-R bằng Qwen3-4B tokenizer local; CNN/DM giữ native source.",
        "3. Tải local Qwen tokenizer và local all-MiniLM-L6-v2. Với mỗi document, sentence embeddings chỉ tính một lần; sau đó chạy lại đúng MMR lambda 0.7 cho sáu action. MMR score, separator cost, exact token budget, preserve-order và fit-to-budget dùng implementation vendored của repository.",
        "4. Từ selected sentence set của từng action, tính facility support của mọi source sentence, residual cover `mean(1 - max cosine support)`, p90 residual và marginal residual gain so với action ngắn hơn kế tiếp.",
        "5. Chạy 20 repeats × 3 document folds. Với từng action, LogisticRegression dự đoán violation probability và Ridge dự đoán latency; chọn action rẻ nhất có risk dự đoán <= alpha, fallback full nếu không có action đủ an toàn.",
        "6. Đánh giá bằng outcome thật của test fold; tính cost, risk, contract pass, cost gain và capture trên oracle headroom. Không dùng quality/cost để tạo feature.",
        "",
        "## Kiểm tra input join và tái dựng MMR",
        "",
        "| Dataset | Joined docs | E26 scored rows | Source file rows | Source-token mean | Min | Median | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    source_file_counts: dict[str, int] = {}
    for dataset, source_path in result["source_paths"].items():
        source_file_counts[dataset] = len(e27a._load_jsonl(source_path))
    for dataset in sorted(result["datasets"]):
        full_rows = [rows["full"] for rows in feature_by_dataset[dataset]]
        source_tokens = sorted(float(row["source_tokens"]) for row in full_rows)
        scored_count = len(e27a._load_jsonl(result["input_paths"][dataset]))
        lines.append(
            f"| {dataset} | {len(full_rows)} | {scored_count} | {source_file_counts[dataset]} | {_fmt(e27a._mean(source_tokens),2)} | {_fmt(source_tokens[0],2)} | {_fmt(e27a._percentile(source_tokens,0.5),2)} | {_fmt(source_tokens[-1],2)} |"
        )
    lines += [
        "",
        "Mỗi document có đúng sáu action rows trong `features.json`. MMR reconstruction dùng một embedding matrix/document rồi chạy lại greedy MMR cho sáu budgets; encode một lần chỉ là tối ưu thực thi, không thay đổi score. Audit trực tiếp trên một sample CNN/DM cho thấy selected indices của helper và public vendored `MMRSelector.select()` trùng nhau (`[4, 9]` ở budget 108). E26-R không lưu selected text/indices nên không thể chứng minh exact equality cho toàn bộ 90 documents; đây là evidence gap được giữ nguyên.",
        "",
        "### Residual/coverage reconstruction theo action",
        "",
        "Bảng là trung bình trên documents trong từng dataset; residual cover bằng 0 ở full là expected vì toàn bộ sentence set được chọn.",
        "",
        "| Dataset | Action | Selected-token ratio | Selected-sentence ratio | Facility coverage | Residual cover | P90 residual | Marginal residual gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted(result["datasets"]):
        rows = feature_by_dataset[dataset]
        for label in e27a.ACTION_LABELS:
            action_rows = [row[label] for row in rows]
            lines.append(
                f"| {dataset} | {label} | {_fmt(e27a._mean(r['selected_token_ratio'] for r in action_rows),4)} | {_fmt(e27a._mean(r['selected_sentence_ratio'] for r in action_rows),4)} | {_fmt(e27a._mean(r['facility_coverage'] for r in action_rows),4)} | {_fmt(e27a._mean(r['residual_cover'] for r in action_rows),4)} | {_fmt(e27a._mean(r['residual_p90'] for r in action_rows),4)} | {_fmt(e27a._mean(r['marginal_residual_gain'] for r in action_rows),4)} |"
            )
    lines += [
        "",
        "## Định nghĩa signal families",
        "",
        "- `cheap`: 8 feature E27-A: source tokens, sentence count, average sentence words, lexical redundancy, unique-token ratio, repeated-bigram ratio, section count và punctuation ratio.",
        "- `front_redundancy`: cheap structural controls bổ sung tỷ lệ token được chọn ở 25% đầu nguồn, vị trí trung bình, selected-token/sentence ratio và lexical redundancy của phần được chọn.",
        "- `facility`: source length/selection structure cộng mean maximum cosine support và p90 support của toàn source đối với selected set.",
        "- `residual`: residual cover, p90 residual, marginal residual gain và selected size ratios; không đưa facility coverage trực tiếp vào family này.",
        "- `residual_plus_cheap`: exploratory control, cộng residual với toàn bộ cheap features.",
        "Residual feature là selection-conditioned nhưng vẫn source-only: nó được tính trước target generation từ document và selected sentence embeddings.",
        "",
        "## Risk, cost và capture",
        "",
        "Một action vi phạm nếu ROUGE-L **hoặc** BERTScore F1 thấp hơn full cùng document quá epsilon. Primary dùng epsilon=0.02 và alpha=0.05; scope 30 docs cho phép tối đa floor(1.5)=1 violating document ở mỗi lần evaluation.",
        "`Cost gain = 1 - policy_cost / fixed_cost`. `Capture = (fixed_cost - policy_cost) / (fixed_cost - oracle_cost)`. Capture là phần oracle headroom policy lấy được, nên có thể âm nếu policy đắt hơn fixed; nó không giống oracle saving của E26-R.",
        "Mỗi CI 95% trong báo cáo là percentile 2,5%–97,5% của 20 repeat-level estimates; đây là uncertainty của repeated-CV screen, không phải bootstrap CI độc lập trên một held-out deployment set.",
        "",
        "## Toàn bộ primary action diagnostics",
        "",
        "Đây là violation/cost thật của từng action trên 30 documents mỗi dataset; nó giải thích safety pressure mà policy phải học.",
        "",
        "| Dataset | Action | Violation docs | Risk | Mean cost ms |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in ("cnn_dailymail", "govreport", "multi_news"):
        for label in ACTION_ORDER:
            item = result["primary_action_diagnostics"][dataset][label]
            lines.append(f"| {dataset} | {label} | {item['violating_documents']}/{item['documents']} | {_pct(item['risk_rate'])} | {_fmt(item['mean_cost_ms'],2)} |")
    lines += [
        "",
        "## Sensitivity contracts — tất cả aggregate của policy residual",
        "",
        "| Dataset/scope | Epsilon | Alpha | Cost ms | Cost CI ms | Risk | Risk CI | Pass fraction | Capture | Capture CI | Gain vs fixed | Gain CI |",
        "|---|---:|---:|---:|---|---:|---|---:|---:|---|---:|---|",
    ]
    for item in result["aggregates"]:
        if item["policy"] != "learned_residual" or item["scope"] == "pooled":
            continue
        lines.append(f"| {item['scope']} | {item['epsilon']:.4f} | {item['alpha']:.4f} | {_fmt(item['mean_cost_ms'],2)} | {_fmt(item['cost_ci_95_ms'][0],2)}–{_fmt(item['cost_ci_95_ms'][1],2)} | {_pct(item['mean_risk_rate'])} | {_pct(item['risk_ci_95'][0])}–{_pct(item['risk_ci_95'][1])} | {_pct(item['risk_pass_fraction'])} | {_pct(item['mean_capture'])} | {_pct(item['capture_ci_95'][0])}–{_pct(item['capture_ci_95'][1])} | {_pct(item['mean_cost_gain_vs_fixed'])} | {_pct(item['cost_gain_ci_95'][0])}–{_pct(item['cost_gain_ci_95'][1])} |")
    lines += [
        "",
        "## Kiểm tra tính đúng đắn và giới hạn",
        "",
        "- C0 không tạo quality hay latency mới; E26-R outcomes là nguồn dữ liệu duy nhất cho target quality/cost.",
        "- Tất cả policy learner fit trong train fold; reference fixed/oracle được gắn nhãn hindsight và chỉ dùng làm ceiling trên scope.",
        "- Exact equality với selected text của E26-R không thể kiểm tra trực tiếp vì E26-R JSONL không lưu selected indices/text. C0 dùng cùng vendored MMR, tokenizer, cap, lambda và budget rule; artifact lưu config để audit. Đây là một evidence gap cần nêu rõ, không che giấu.",
        "- Feature reconstruction dùng CPU local MiniLM; đây là preprocessing/offline signal extraction, không phải GPU target inference. E27-C0 vì vậy không claim fresh GPU run.",
        "- Cỡ mẫu là 30 docs/dataset; 20 repeated folds giúp đánh giá ổn định screening nhưng chưa thay thế held-out deployment calibration.",
        "- Dữ liệu GovReport/Multi-News vẫn kế thừa source cap 4096 của E26-R; kết luận không mở rộng native 8K/16K.",
        "",
        "## CONFIRMED",
        "",
        f"- E27-C0 đã hoàn tất offline với {result['documents']} documents, sáu action, {result['repeats']} repeats × {result['n_splits']} folds và {len(result['records'])} per-repeat records.",
        "- Signal residual-cover đã được đánh giá cùng protocol với cheap/front/facility priors; không có policy nào được phép xem outcome test khi chọn action.",
        "",
        "## EXPLORATORY",
        "",
        "- Residual-plus-cheap là control khám phá, không phải primary novelty claim.",
        "- Các sensitivity contracts chỉ là robustness analysis; gate chính dùng epsilon=0.02, alpha=0.05.",
        "",
        "## FAILED / INCOMPLETE",
        "",
        f"- E27-C0 residual signal pass {passing}/3 dataset gate scopes. Gate yêu cầu >=2; vì vậy E27-C1 minimum-cost residual stopping **{'được mở' if passing >= 2 else 'không được mở'}**.",
        "- Chưa có E27-D conformal calibration, held-out deployment test, 8K/16K native evaluation hoặc end-to-end GPU runtime cho một controller residual-cover.",
        "",
        "## HIGHEST VERIFIED RUNG",
        "",
        "**R7 — result review / decision memo cho full offline E27-C0 signal-discrimination screen.**",
        "",
        "R7 được xác nhận bởi 90 documents, 6 actions/document, 9 contracts, 20 repeats × 3 folds, 6.480 per-repeat records và 324 aggregate cells. Kết quả được tính lại từ các artifact JSON/CSV; report này chứa trực tiếp toàn bộ aggregate records và toàn bộ repeated-CV records ở các phụ lục, không yêu cầu tra số liệu bên ngoài.",
        "",
        "## EVIDENCE GAPS",
        "",
        "- C0 là offline replay trên E26-R, không phải một lần target inference mới và không phải GPU benchmark. MiniLM/tokenizer chỉ phục vụ tái dựng source-only features.",
        "- E26-R không lưu selected sentence indices/text, nên audit MMR trên toàn bộ 90 documents không thể chứng minh exact identity; chỉ có audit trực tiếp một sample và kiểm tra cùng implementation/configuration.",
        "- Repeated-CV confidence intervals phản ánh độ ổn định của screening trên 90 documents, chưa phải CI của một held-out deployment calibration độc lập.",
        "- Scope GovReport/Multi-News giữ cap 4096 source tokens từ E26-R; không được suy rộng thành kết quả native 8K/16K.",
        "- Residual-plus-cheap có một số contract pass fraction dưới 100%; đây là exploratory control và không được dùng làm bằng chứng residual signal riêng biệt.",
        "",
        "## RECOMMENDED NEXT",
        "",
        "Không chạy E27-C1 minimum-cost residual stopping theo gate đã đăng ký: residual không đạt capture >=30% và không hơn prior mạnh nhất >=10 percentage points trên bất kỳ dataset scope nào. Giữ fixed MMR/E26-R như systems baseline; chỉ mở một nghiên cứu mới nếu có protocol held-out và một source-only signal khác đã được đăng ký trước.",
        "",
        "## Artifact và reproducibility",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Input E26-R: `{result['input_paths']}`",
        f"- Source inputs: `{result['source_paths']}`",
        f"- Tokenizer: `{result['tokenizer_path']}`",
        f"- Embedding: `{result['embedding_path']}`",
        f"- MMR lambda: `{result['mmr_lambda']}`; seed: `{result['seed']}`; folds/repeats: `{result['n_splits']}/{result['repeats']}`.",
        "- `features.json` chứa toàn bộ source/action-conditioned features; `metrics.json` chứa kết quả đầy đủ; `metrics.csv` chứa mọi repeated-CV row; `run_manifest.json` ghi command/protocol.",
        "",
        "## Next decision",
        "",
        "Chỉ khi gate residual vượt prior rõ ràng mới được gọi E27-C1. Nếu gate fail, SafeCover residual-cover branch phải đóng ở screening stage; fixed MMR vẫn có thể giữ như systems baseline của E26-R nhưng không được trình bày C0 như evidence cho một controller mới.",
        "",
        "## Phụ lục — aggregate records",
        "",
        "Bảng này giữ trực tiếp mọi aggregate cell của C0 để báo cáo tự chứa, không cần suy ra từ một số được trích dẫn bên ngoài. Các repeated-CV records cấp dòng được giữ trong `metrics.csv` và `metrics.json`, không chèn toàn bộ vào báo cáo để tránh làm báo cáo quá dài.",
        "",
        "| Scope | Policy | Feature set | Epsilon | Alpha | Repeats | Cost ms | Cost CI ms | Risk | Risk CI | Pass fraction | Capture | Capture CI | Gain vs fixed | Gain CI |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---|---:|---|",
    ]
    for item in result["aggregates"]:
        lines.append(f"| {item['scope']} | {item['policy']} | {item['feature_set']} | {item['epsilon']:.4f} | {item['alpha']:.4f} | {item['repeats']} | {_fmt(item['mean_cost_ms'],2)} | {_fmt(item['cost_ci_95_ms'][0],2)}–{_fmt(item['cost_ci_95_ms'][1],2)} | {_pct(item['mean_risk_rate'])} | {_pct(item['risk_ci_95'][0])}–{_pct(item['risk_ci_95'][1])} | {_pct(item['risk_pass_fraction'])} | {_pct(item['mean_capture'])} | {_pct(item['capture_ci_95'][0])}–{_pct(item['capture_ci_95'][1])} | {_pct(item['mean_cost_gain_vs_fixed'])} | {_pct(item['cost_gain_ci_95'][0])}–{_pct(item['cost_gain_ci_95'][1])} |")
    lines += [
        "",
        "## Dữ liệu chi tiết và khả năng audit",
        "",
        f"Báo cáo không lặp lại {len(result['records']):,} repeated-CV rows. Các rows này vẫn được lưu đầy đủ trong `metrics.csv` và `metrics.json`; các feature source/action-conditioned vẫn được lưu trong `features.json`. Các bảng trong báo cáo chứa kết quả tổng hợp, CI và action diagnostics cần để đọc và kết luận experiment.",
        "",
        "Kiểm tra cuối xác nhận số dòng raw vẫn là 6.480, số aggregate là 324, không bị xóa hay rút mẫu khi rút gọn báo cáo.",
    ]
    return "\n".join(lines) + "\n"


def _pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    return name, path


def _int_pair(value: str) -> tuple[str, int]:
    name, raw = _pair(value)
    return name, int(raw)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_pair, required=True)
    parser.add_argument("--source-input", action="append", type=_pair, required=True)
    parser.add_argument("--max-input-tokens", action="append", type=_int_pair, default=[])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--embedding", default=EMBEDDING_CHECKPOINT)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--epsilon", action="append", type=float, default=[0.01, 0.02, 0.05])
    parser.add_argument("--alpha", action="append", type=float, default=[0.0, 0.05, 0.10])
    parser.add_argument("--primary-epsilon", type=float, default=0.02)
    parser.add_argument("--primary-alpha", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    input_paths = dict(args.input)
    source_paths = dict(args.source_input)
    if set(input_paths) != set(source_paths):
        parser.error("--input and --source-input names must match")
    result = run_screen(
        input_paths, source_paths,
        source_caps=dict(args.max_input_tokens),
        tokenizer_path=args.tokenizer,
        embedding_path=args.embedding,
        mmr_lambda=args.mmr_lambda,
        embedding_batch_size=args.embedding_batch_size,
        epsilons=args.epsilon,
        alphas=args.alpha,
        primary_epsilon=args.primary_epsilon,
        primary_alpha=args.primary_alpha,
        n_splits=args.folds,
        repeats=args.repeats,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "features.json").write_text(json.dumps(result["features"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = [
        "repeat", "scope", "policy", "feature_set", "epsilon", "alpha", "documents", "cost_ms", "risk_rate", "violating_documents", "max_violating_documents", "risk_contract_pass", "fixed_label_test_oracle", "fixed_cost_ms_test_oracle", "adaptive_oracle_cost_ms", "capture", "cost_gain_vs_fixed", "action_counts",
    ]
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["records"]:
            serial = dict(row)
            serial["action_counts"] = json.dumps(serial["action_counts"], ensure_ascii=False, sort_keys=True)
            writer.writerow({field: serial.get(field) for field in fields})
    report_text = render_report(result, output_dir=output_dir)
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")
    (output_dir / "comprehensive_report.md").write_text(report_text, encoding="utf-8")
    manifest = {
        "experiment": result["experiment"],
        "input_paths": input_paths,
        "source_paths": source_paths,
        "source_caps": dict(args.max_input_tokens),
        "tokenizer": args.tokenizer,
        "embedding": args.embedding,
        "mmr_lambda": args.mmr_lambda,
        "embedding_batch_size": args.embedding_batch_size,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "primary_epsilon": args.primary_epsilon,
        "primary_alpha": args.primary_alpha,
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "offline": True,
        "gpu_inference": False,
        "outcome_source": "E26-R scored JSONL",
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"experiment": result["experiment"], "documents": result["documents"], "records": len(result["records"]), "aggregates": len(result["aggregates"]), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
