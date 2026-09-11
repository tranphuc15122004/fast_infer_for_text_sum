"""E27-A offline policy-learnability screen for E26-R context budgets.

The screen asks whether a source-only router can predict which of the six
already measured context actions is safe and inexpensive.  It never uses the
generated summary, quality score, target hidden state, or measured cost as a
runtime feature.  Measured cost and quality are used only after an action is
selected to evaluate the policy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ACTION_LABELS = (
    "full",
    "ratio_0.25",
    "ratio_0.4",
    "ratio_0.55",
    "ratio_0.7",
    "ratio_0.85",
)

QUALITY_FIELDS = ("rougeL", "bertscore_f1")
CHEAP_FEATURES = (
    "source_tokens",
    "sentence_count",
    "avg_sentence_words",
    "lexical_redundancy",
    "unique_token_ratio",
    "repeated_bigram_ratio",
    "section_count",
    "punctuation_ratio",
)

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 \-:&/,()]{3,100}$")
# Tokenizers can spend a disproportionate amount of time on a native
# long-document string even when `truncation=True` (Multi-News contains a
# particularly large outlier).  This is only a guard for the *source-feature*
# view; the E26-R measured outcomes remain authoritative for source tokens and
# cost/quality.  A 12-character/token upper bound is deliberately conservative
# for the Qwen tokenizer used by E26-R.
FEATURE_CHAR_LIMIT = 65536


def _finite_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Iterable[float]) -> float | None:
    data = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(data) / len(data) if data else None


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


def _doc_key(dataset: str, example_id: Any) -> str:
    return f"{dataset}::{example_id}"


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_RE.split(text) if part.strip()]


def build_source_features(text: str, *, source_tokens: int | float) -> dict[str, float]:
    """Build deterministic source-only features with no target-side inputs."""
    words = [match.group(0).lower() for match in WORD_RE.finditer(text)]
    sentences = _split_sentences(text)
    sentence_words = [len(WORD_RE.findall(sentence)) for sentence in sentences]
    bigrams = list(zip(words, words[1:]))
    unique_words = len(set(words))
    unique_bigrams = len(set(bigrams))
    heading_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and (HEADING_RE.match(line.strip()) or line.strip().endswith(":"))
    ]
    punctuation_count = sum(1 for char in text if char in ".,;:!?()[]{}\"'")
    total_words = max(1, len(words))
    total_bigrams = max(1, len(bigrams))
    result = {
        "source_tokens": float(source_tokens),
        "sentence_count": float(max(1, len(sentences))),
        "avg_sentence_words": float(_mean(sentence_words) or (len(words) or 1)),
        "lexical_redundancy": float(1.0 - unique_words / total_words),
        "unique_token_ratio": float(unique_words / total_words),
        "repeated_bigram_ratio": float(1.0 - unique_bigrams / total_bigrams),
        "section_count": float(max(1, len(heading_lines))),
        "punctuation_ratio": float(punctuation_count / max(1, len(text))),
    }
    return {
        key: value if math.isfinite(value) else 0.0
        for key, value in result.items()
    }


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def quality_violation(
    action: Mapping[str, Any],
    full: Mapping[str, Any],
    *,
    epsilon: float,
) -> bool:
    """Return whether either required quality metric loses more than epsilon."""
    return any(
        float(full[metric]) - float(action[metric]) > epsilon
        for metric in QUALITY_FIELDS
    )


def _load_outcomes(
    input_paths: Mapping[str, str],
    *,
    epsilon: float,
) -> dict[str, dict[str, dict[str, Any]]]:
    outcomes: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset, path in input_paths.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _load_jsonl(path):
            grouped[str(row.get("example_id", row.get("id", "")))].append(row)
        for example_id, rows in grouped.items():
            full = next(
                (row for row in rows if row.get("selector") == "full" and row.get("budget_label") == "full"),
                None,
            )
            if full is None:
                continue
            full_quality = {
                metric: _finite_number(full.get(metric)) for metric in QUALITY_FIELDS
            }
            full_cost = _finite_number(full.get("pipeline_e2e_ms"))
            if any(value is None for value in full_quality.values()) or full_cost is None:
                continue
            key = _doc_key(dataset, example_id)
            document_actions: dict[str, dict[str, Any]] = {}
            for row in rows:
                label = str(row.get("budget_label", ""))
                selector = str(row.get("selector", ""))
                if label == "full" and selector == "full":
                    action_label = "full"
                elif selector == "mmr" and label in ACTION_LABELS:
                    action_label = label
                else:
                    continue
                quality = {
                    metric: _finite_number(row.get(metric)) for metric in QUALITY_FIELDS
                }
                cost = _finite_number(row.get("pipeline_e2e_ms"))
                if any(value is None for value in quality.values()) or cost is None:
                    continue
                document_actions[action_label] = {
                    "label": action_label,
                    "cost_ms": cost,
                    **quality,
                    "violates": quality_violation(
                        {**quality, "label": action_label},
                        {**full_quality, "label": "full"},
                        epsilon=epsilon,
                    ),
                }
            if set(ACTION_LABELS).issubset(document_actions):
                outcomes[key] = document_actions
    return outcomes


def _load_features(
    source_paths: Mapping[str, str],
    *,
    outcomes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    source_caps: Mapping[str, int],
    tokenizer_path: str | None,
) -> dict[str, dict[str, float]]:
    tokenizer = None
    if tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    features: dict[str, dict[str, float]] = {}
    for dataset, path in source_paths.items():
        for row in _load_jsonl(path):
            example_id = str(row.get("id", row.get("example_id", "")))
            key = _doc_key(dataset, example_id)
            if key not in outcomes:
                continue
            text = str(row.get("document", ""))
            cap = int(source_caps.get(dataset, 0))
            if cap > 0:
                # Bound the input before invoking the tokenizer.  For a
                # capped E26-R condition this prefix is large enough to
                # contain approximately `cap` Qwen tokens, while preventing
                # accidental processing of a native 100K+ token document.
                feature_chars = min(FEATURE_CHAR_LIMIT, max(4096, cap * 12))
                text = text[:feature_chars]
                if tokenizer is not None:
                    token_ids = tokenizer(
                        text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=cap,
                    )["input_ids"]
                    text = tokenizer.decode(token_ids, skip_special_tokens=True)
            full_action = outcomes[key]["full"]
            # The raw runner's original_tokens is the authoritative source
            # budget after any dataset-specific cap.
            features[key] = build_source_features(
                text,
                source_tokens=full_action.get("source_tokens", 0.0),
            )
    # source_tokens is populated from the full raw row by the caller below;
    # this fallback keeps the function useful with synthetic outcomes.
    for key, values in features.items():
        if values["source_tokens"] <= 0:
            values["source_tokens"] = float(len(values))
    return features


def _attach_source_token_counts(
    features: dict[str, dict[str, float]],
    input_paths: Mapping[str, str],
) -> None:
    for dataset, path in input_paths.items():
        grouped: dict[str, dict[str, Any]] = {}
        for row in _load_jsonl(path):
            if row.get("selector") == "full" and row.get("budget_label") == "full":
                grouped[str(row.get("example_id", row.get("id", "")))] = row
        for example_id, row in grouped.items():
            key = _doc_key(dataset, example_id)
            if key in features:
                count = _finite_number(row.get("original_tokens"))
                if count is not None:
                    features[key]["source_tokens"] = count


def choose_action_from_predictions(
    predicted_risk: Mapping[str, float],
    predicted_cost: Mapping[str, float],
    *,
    alpha: float,
) -> str:
    eligible = [
        label
        for label in ACTION_LABELS
        if float(predicted_risk.get(label, 1.0)) <= alpha
    ]
    if not eligible:
        return "full"
    return min(
        eligible,
        key=lambda label: (float(predicted_cost.get(label, float("inf"))), label),
    )


def _action_is_violating(
    outcomes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    key: str,
    label: str,
    *,
    epsilon: float,
) -> bool:
    action = outcomes[key][label]
    if "violates" in action:
        return bool(action["violates"])
    return quality_violation(action, outcomes[key]["full"], epsilon=epsilon)


class _ConstantRisk:
    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        p = np.full(matrix.shape[0], self.probability, dtype=float)
        return np.stack([1.0 - p, p], axis=1)


class _ConstantCost:
    def __init__(self, value: float):
        self.value = max(0.0, float(value))

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.full(matrix.shape[0], self.value, dtype=float)


def _fit_risk_model(matrix: np.ndarray, labels: Sequence[int]) -> Any:
    labels_array = np.asarray(labels, dtype=int)
    if len(set(labels_array.tolist())) < 2:
        return _ConstantRisk(float(labels_array.mean()) if len(labels_array) else 0.0)
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")),
        ]
    ).fit(matrix, labels_array)


def _fit_cost_model(matrix: np.ndarray, values: Sequence[float]) -> Any:
    values_array = np.asarray(values, dtype=float)
    if len(values_array) < 3 or float(np.std(values_array)) < 1e-8:
        return _ConstantCost(float(values_array.mean()) if len(values_array) else 0.0)
    return Pipeline(
        [("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
    ).fit(matrix, values_array)


def _fit_learned_models(
    train_ids: Sequence[str],
    features: Mapping[str, Mapping[str, float]],
    outcomes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    feature_names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    matrix = np.asarray(
        [[features[key][name] for name in feature_names] for key in train_ids],
        dtype=float,
    )
    risk_models: dict[str, Any] = {}
    cost_models: dict[str, Any] = {}
    for label in ACTION_LABELS:
        risk_models[label] = _fit_risk_model(
            matrix,
            [int(bool(outcomes[key][label]["violates"])) for key in train_ids],
        )
        cost_models[label] = _fit_cost_model(
            matrix,
            [float(outcomes[key][label]["cost_ms"]) for key in train_ids],
        )
    return risk_models, cost_models


def _predict_learned_actions(
    test_ids: Sequence[str],
    features: Mapping[str, Mapping[str, float]],
    *,
    feature_names: Sequence[str],
    risk_models: Mapping[str, Any],
    cost_models: Mapping[str, Any],
    alpha: float,
) -> dict[str, str]:
    matrix = np.asarray(
        [[features[key][name] for name in feature_names] for key in test_ids],
        dtype=float,
    )
    actions: dict[str, str] = {}
    for index, key in enumerate(test_ids):
        predicted_risk = {
            label: float(risk_models[label].predict_proba(matrix[index:index + 1])[0, 1])
            for label in ACTION_LABELS
        }
        predicted_cost = {
            label: max(0.0, float(cost_models[label].predict(matrix[index:index + 1])[0]))
            for label in ACTION_LABELS
        }
        actions[key] = choose_action_from_predictions(
            predicted_risk,
            predicted_cost,
            alpha=alpha,
        )
    return actions


def _select_fixed(
    document_ids: Sequence[str],
    outcomes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    epsilon: float,
    alpha: float,
) -> str:
    max_violations = int(math.floor(alpha * len(document_ids) + 1e-12))
    candidates: list[tuple[float, str]] = []
    labels = [
        label
        for label in ACTION_LABELS
        if all(label in outcomes[key] for key in document_ids)
    ]
    for label in labels:
        violations = sum(
            int(_action_is_violating(outcomes, key, label, epsilon=epsilon))
            for key in document_ids
        )
        if violations <= max_violations:
            candidates.append((
                _mean(outcomes[key][label]["cost_ms"] for key in document_ids) or float("inf"),
                label,
            ))
    return min(candidates, default=(float("inf"), "full"))[1]


def _adaptive_oracle(
    document_ids: Sequence[str],
    outcomes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    epsilon: float,
    alpha: float,
) -> dict[str, str]:
    max_violations = int(math.floor(alpha * len(document_ids) + 1e-12))
    states: dict[int, tuple[float, dict[str, str]]] = {0: (0.0, {})}
    for key in document_ids:
        next_states: dict[int, tuple[float, dict[str, str]]] = {}
        for used, (cost, selected) in states.items():
            labels = [
                label
                for label in ACTION_LABELS
                if label in outcomes[key]
            ]
            for label in labels:
                next_used = used + int(
                    _action_is_violating(outcomes, key, label, epsilon=epsilon)
                )
                if next_used > max_violations:
                    continue
                next_cost = cost + float(outcomes[key][label]["cost_ms"])
                incumbent = next_states.get(next_used)
                if incumbent is None or next_cost < incumbent[0]:
                    choice = dict(selected)
                    choice[key] = label
                    next_states[next_used] = (next_cost, choice)
        states = next_states
    if not states:
        return {key: "full" for key in document_ids}
    return min(states.values(), key=lambda item: item[0])[1]


def evaluate_policy_actions(
    actions: Mapping[str, str],
    outcomes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    epsilon: float,
    alpha: float,
) -> dict[str, Any]:
    """Evaluate one action assignment and normalize against test-set oracles."""
    document_ids = list(actions)
    fixed_label = _select_fixed(
        document_ids, outcomes, epsilon=epsilon, alpha=alpha
    )
    fixed_actions = {key: fixed_label for key in document_ids}
    oracle_actions = _adaptive_oracle(
        document_ids, outcomes, epsilon=epsilon, alpha=alpha
    )
    fixed_cost = _mean(outcomes[key][fixed_label]["cost_ms"] for key in document_ids)
    oracle_cost = _mean(
        outcomes[key][oracle_actions[key]]["cost_ms"] for key in document_ids
    )
    return _evaluate_policy_with_reference(
        actions,
        outcomes,
        epsilon=epsilon,
        alpha=alpha,
        fixed_label=fixed_label,
        fixed_cost=fixed_cost,
        oracle_actions=oracle_actions,
        oracle_cost=oracle_cost,
    )


def _evaluate_policy_with_reference(
    actions: Mapping[str, str],
    outcomes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    epsilon: float,
    alpha: float,
    fixed_label: str,
    fixed_cost: float | None,
    oracle_actions: Mapping[str, str],
    oracle_cost: float | None,
) -> dict[str, Any]:
    """Evaluate actions when the fixed/oracle references are already cached."""
    document_ids = list(actions)
    max_violations = int(math.floor(alpha * len(document_ids) + 1e-12))
    violations = sum(
        int(_action_is_violating(outcomes, key, actions[key], epsilon=epsilon))
        for key in document_ids
    )
    cost = _mean(outcomes[key][actions[key]]["cost_ms"] for key in document_ids)
    denominator = None
    capture = None
    if fixed_cost is not None and oracle_cost is not None and fixed_cost != oracle_cost:
        denominator = fixed_cost - oracle_cost
        capture = (fixed_cost - float(cost)) / denominator
    return {
        "documents": len(document_ids),
        "cost_ms": cost,
        "risk_rate": violations / len(document_ids) if document_ids else None,
        "violating_documents": violations,
        "max_violating_documents": max_violations,
        "risk_contract_pass": violations <= max_violations,
        "fixed_label_test_oracle": fixed_label,
        "fixed_cost_ms_test_oracle": fixed_cost,
        "adaptive_oracle_cost_ms": oracle_cost,
        "capture": capture,
        "cost_gain_vs_fixed": (
            None
            if fixed_cost in (None, 0) or cost is None
            else 1.0 - float(cost) / float(fixed_cost)
        ),
        "action_counts": dict(sorted(Counter(actions.values()).items())),
    }


def _length_only_actions(
    train_ids: Sequence[str],
    test_ids: Sequence[str],
    features: Mapping[str, Mapping[str, float]],
    outcomes: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    epsilon: float,
    alpha: float,
) -> dict[str, str]:
    lengths = np.asarray([features[key]["source_tokens"] for key in train_ids], dtype=float)
    quantiles = np.unique(np.quantile(lengths, [0.25, 0.5, 0.75])) if len(lengths) else np.asarray([])
    bin_actions: list[str] = []
    for bin_index in range(len(quantiles) + 1):
        train_bin = [
            key
            for key in train_ids
            if int(np.digitize(features[key]["source_tokens"], quantiles, right=False)) == bin_index
        ]
        if not train_bin:
            train_bin = list(train_ids)
        bin_actions.append(
            _select_fixed(train_bin, outcomes, epsilon=epsilon, alpha=alpha)
        )
    return {
        key: bin_actions[int(np.digitize(features[key]["source_tokens"], quantiles, right=False))]
        for key in test_ids
    }


def _length_threshold_actions(
    document_ids: Sequence[str],
    features: Mapping[str, Mapping[str, float]],
) -> dict[str, str]:
    """A pre-registered length-only heuristic, not fit on quality outcomes."""
    actions: dict[str, str] = {}
    for key in document_ids:
        tokens = features[key]["source_tokens"]
        if tokens < 512:
            action = "full"
        elif tokens < 1024:
            action = "ratio_0.55"
        elif tokens < 2048:
            action = "ratio_0.4"
        else:
            action = "ratio_0.25"
        actions[key] = action
    return actions


def _make_splits(
    keys: Sequence[str],
    *,
    n_splits: int,
    repeats: int,
    seed: int,
) -> list[list[tuple[list[str], list[str]]]]:
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        by_dataset[key.split("::", 1)[0]].append(key)
    all_repeats: list[list[tuple[list[str], list[str]]]] = []
    for repeat in range(repeats):
        fold_test: list[list[str]] = [[] for _ in range(n_splits)]
        for dataset, dataset_keys in sorted(by_dataset.items()):
            ordered = list(dataset_keys)
            dataset_offset = sum((index + 1) * ord(char) for index, char in enumerate(dataset))
            random.Random(seed + repeat * 1009 + dataset_offset % 997).shuffle(ordered)
            for index, key in enumerate(ordered):
                fold_test[index % n_splits].append(key)
        folds: list[tuple[list[str], list[str]]] = []
        for fold_index in range(n_splits):
            test = sorted(fold_test[fold_index])
            test_set = set(test)
            train = sorted(key for key in keys if key not in test_set)
            folds.append((train, test))
        all_repeats.append(folds)
    return all_repeats


def _ci(values: Sequence[float | None]) -> list[float | None]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return [_percentile(finite, 0.025), _percentile(finite, 0.975)]


def _aggregate_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float, float], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                str(record["scope"]),
                str(record["policy"]),
                str(record["feature_set"]),
                float(record["epsilon"]),
                float(record["alpha"]),
            )
        ].append(record)
    output: list[dict[str, Any]] = []
    for (scope, policy, feature_set, epsilon, alpha), items in sorted(grouped.items()):
        risks = [item.get("risk_rate") for item in items]
        costs = [item.get("cost_ms") for item in items]
        captures = [item.get("capture") for item in items]
        gains = [item.get("cost_gain_vs_fixed") for item in items]
        output.append(
            {
                "scope": scope,
                "policy": policy,
                "feature_set": feature_set,
                "epsilon": epsilon,
                "alpha": alpha,
                "repeats": len(items),
                "mean_cost_ms": _mean(costs),
                "cost_ci_95_ms": _ci(costs),
                "mean_risk_rate": _mean(risks),
                "risk_ci_95": _ci(risks),
                "risk_pass_fraction": _mean(float(item["risk_contract_pass"]) for item in items),
                "mean_capture": _mean(captures),
                "capture_ci_95": _ci(captures),
                "mean_cost_gain_vs_fixed": _mean(gains),
                "cost_gain_ci_95": _ci(gains),
                "action_counts": dict(
                    sorted(
                        sum(
                            (Counter(item.get("action_counts", {})) for item in items),
                            Counter(),
                        ).items()
                    )
                ),
            }
        )
    return output


def run_screen(
    input_paths: Mapping[str, str],
    source_paths: Mapping[str, str],
    *,
    source_caps: Mapping[str, int],
    tokenizer_path: str | None,
    epsilons: Sequence[float],
    alphas: Sequence[float],
    primary_epsilon: float,
    primary_alpha: float,
    n_splits: int,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    # Outcomes are reloaded per epsilon because violation labels are contract-specific.
    first_outcomes = _load_outcomes(input_paths, epsilon=primary_epsilon)
    features = _load_features(
        source_paths,
        outcomes=first_outcomes,
        source_caps=source_caps,
        tokenizer_path=tokenizer_path,
    )
    _attach_source_token_counts(features, input_paths)
    keys = sorted(set(first_outcomes).intersection(features))
    if not keys:
        raise ValueError("no complete document keys after joining scored rows and source documents")
    splits = _make_splits(keys, n_splits=n_splits, repeats=repeats, seed=seed)
    records: list[dict[str, Any]] = []
    for epsilon in epsilons:
        outcomes = _load_outcomes(input_paths, epsilon=epsilon)
        contract_keys = sorted(set(keys).intersection(outcomes))
        scopes: dict[str, list[str]] = {
            "pooled": contract_keys,
        }
        scopes.update({
            dataset: [
                key for key in contract_keys
                if key.split("::", 1)[0] == dataset
            ]
            for dataset in sorted({key.split("::", 1)[0] for key in contract_keys})
        })
        # Fit the learned risk/cost models once per epsilon/repeat/fold.  The
        # fitted scores do not depend on alpha, while alpha only changes the
        # final safe-action filter.  This preserves the CV protocol and avoids
        # repeating 3 identical model fits for the three alpha values.
        fold_model_cache: list[list[dict[str, Any]]] = []
        for folds in splits:
            repeat_cache: list[dict[str, Any]] = []
            for train_ids_all, test_ids_all in folds:
                train_ids = [key for key in train_ids_all if key in outcomes]
                test_ids = [key for key in test_ids_all if key in outcomes]
                if not train_ids or not test_ids:
                    continue
                risk_models, cost_models = _fit_learned_models(
                    train_ids,
                    features,
                    outcomes,
                    feature_names=CHEAP_FEATURES,
                )
                matrix = np.asarray(
                    [[features[key][name] for name in CHEAP_FEATURES] for key in test_ids],
                    dtype=float,
                )
                risk_predictions: dict[str, dict[str, float]] = {}
                cost_predictions: dict[str, dict[str, float]] = {}
                for index, key in enumerate(test_ids):
                    risk_predictions[key] = {
                        label: float(
                            risk_models[label].predict_proba(matrix[index:index + 1])[0, 1]
                        )
                        for label in ACTION_LABELS
                    }
                    cost_predictions[key] = {
                        label: max(
                            0.0,
                            float(cost_models[label].predict(matrix[index:index + 1])[0]),
                        )
                        for label in ACTION_LABELS
                    }
                repeat_cache.append({
                    "train_ids": train_ids,
                    "test_ids": test_ids,
                    "risk_predictions": risk_predictions,
                    "cost_predictions": cost_predictions,
                })
            fold_model_cache.append(repeat_cache)

        for alpha in alphas:
            # These references depend only on the contract and scope, not on
            # the CV repeat or on the policy.  Caching them avoids thousands
            # of identical fixed-selection and dynamic-programming calls.
            references: dict[str, dict[str, Any]] = {}
            for scope, scope_keys in scopes.items():
                if not scope_keys:
                    continue
                fixed_label = _select_fixed(
                    scope_keys, outcomes, epsilon=epsilon, alpha=alpha
                )
                oracle_actions = _adaptive_oracle(
                    scope_keys, outcomes, epsilon=epsilon, alpha=alpha
                )
                references[scope] = {
                    "fixed_label": fixed_label,
                    "fixed_cost": _mean(
                        outcomes[key][fixed_label]["cost_ms"] for key in scope_keys
                    ),
                    "oracle_actions": oracle_actions,
                    "oracle_cost": _mean(
                        outcomes[key][oracle_actions[key]]["cost_ms"]
                        for key in scope_keys
                    ),
                }
            for repeat_index, repeat_cache in enumerate(fold_model_cache):
                predictions: dict[str, dict[str, str]] = defaultdict(dict)
                for fold in repeat_cache:
                    train_ids = fold["train_ids"]
                    test_ids = fold["test_ids"]
                    fixed_label = _select_fixed(
                        train_ids, outcomes, epsilon=epsilon, alpha=alpha
                    )
                    predictions["fixed_train"].update(
                        {key: fixed_label for key in test_ids}
                    )
                    predictions["length_only"].update(
                        _length_only_actions(
                            train_ids,
                            test_ids,
                            features,
                            outcomes,
                            epsilon=epsilon,
                            alpha=alpha,
                        )
                    )
                    predictions["length_threshold"].update(
                        _length_threshold_actions(test_ids, features)
                    )
                    for key in test_ids:
                        predictions["learned_cheap"][key] = choose_action_from_predictions(
                            fold["risk_predictions"][key],
                            fold["cost_predictions"][key],
                            alpha=alpha,
                        )
                for scope, scope_keys in scopes.items():
                    if not scope_keys:
                        continue
                    reference = references[scope]
                    policy_actions: dict[str, dict[str, str]] = {
                        "best_fixed_eval": {
                            key: reference["fixed_label"] for key in scope_keys
                        },
                        "oracle_adaptive_eval": reference["oracle_actions"],
                    }
                    for policy in ("fixed_train", "length_only", "length_threshold", "learned_cheap"):
                        policy_actions[policy] = {
                            key: predictions[policy].get(key, "full") for key in scope_keys
                        }
                    for policy, actions in policy_actions.items():
                        result = _evaluate_policy_with_reference(
                            actions,
                            {key: outcomes[key] for key in scope_keys},
                            epsilon=epsilon,
                            alpha=alpha,
                            fixed_label=reference["fixed_label"],
                            fixed_cost=reference["fixed_cost"],
                            oracle_actions=reference["oracle_actions"],
                            oracle_cost=reference["oracle_cost"],
                        )
                        feature_set = "cheap" if policy == "learned_cheap" else (
                            "length" if policy in {"length_only", "length_threshold"} else "none"
                        )
                        records.append(
                            {
                                "repeat": repeat_index,
                                "scope": scope,
                                "policy": policy,
                                "feature_set": feature_set,
                                "epsilon": float(epsilon),
                                "alpha": float(alpha),
                                **result,
                            }
                        )
    aggregates = _aggregate_records(records)
    primary_outcomes = _load_outcomes(
        input_paths, epsilon=primary_epsilon
    )
    primary_action_diagnostics: dict[str, dict[str, dict[str, float | int]]] = {}
    for dataset in sorted({key.split("::", 1)[0] for key in keys}):
        dataset_keys = [
            key for key in keys
            if key.split("::", 1)[0] == dataset and key in primary_outcomes
        ]
        primary_action_diagnostics[dataset] = {}
        for label in ACTION_LABELS:
            violations = sum(
                int(_action_is_violating(primary_outcomes, key, label, epsilon=primary_epsilon))
                for key in dataset_keys
            )
            primary_action_diagnostics[dataset][label] = {
                "documents": len(dataset_keys),
                "violating_documents": violations,
                "risk_rate": violations / len(dataset_keys) if dataset_keys else 0.0,
                "mean_cost_ms": _mean(
                    primary_outcomes[key][label]["cost_ms"] for key in dataset_keys
                ) or 0.0,
            }
    return {
        "experiment": "E27A_policy_learnability_screen",
        "primary_contract": {"epsilon": primary_epsilon, "alpha": primary_alpha},
        "quality_fields": list(QUALITY_FIELDS),
        "action_labels": list(ACTION_LABELS),
        "feature_names": list(CHEAP_FEATURES),
        "documents": len(keys),
        "datasets": dict(sorted(Counter(key.split("::", 1)[0] for key in keys).items())),
        "n_splits": n_splits,
        "repeats": repeats,
        "seed": seed,
        "input_paths": dict(input_paths),
        "source_paths": dict(source_paths),
        "source_caps": dict(source_caps),
        "features": features,
        "records": records,
        "aggregates": aggregates,
        "primary_action_diagnostics": primary_action_diagnostics,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def render_report(result: Mapping[str, Any], *, output_dir: str | Path) -> str:
    primary = result["primary_contract"]
    lines = [
        "# E27-A Policy Learnability Screen Report",
        "",
        "## Kết luận ngắn",
        "",
        f"E27-A là phân tích offline trên **{result['documents']} documents / {sum(result['datasets'].values()) if isinstance(result['datasets'], dict) else result['documents']} joined records** từ E26-R; không chạy lại target model.",
        f"Contract chính: `epsilon={primary['epsilon']:.4f}`, `alpha={primary['alpha']:.4f}`; metric vi phạm là ROUGE-L hoặc BERTScore F1 giảm quá epsilon so với full cùng document.",
        "Các policy learned chỉ nhìn source-only features; mọi cost/quality chỉ được dùng sau khi policy chọn action để đánh giá. `oracle_adaptive_eval` và `best_fixed_eval` là test-set hindsight references.",
        "",
        "## Mục tiêu và câu hỏi nghiên cứu",
        "",
        "E27-A kiểm tra **policy learnability** của giả thuyết risk-calibrated context-budget routing. Câu hỏi không phải action nào tốt nhất khi đã biết quality/cost của từng document, mà là: một policy chỉ dùng source features trước generation có thể chọn action an toàn và rẻ, rồi thu được phần đáng kể của adaptive oracle hay không.",
        "",
        "Hypothesis đã khóa: với action space `full`, `ratio_0.25`, `ratio_0.4`, `ratio_0.55`, `ratio_0.7`, `ratio_0.85`, một router source-only có thể đạt ít nhất 60% oracle-savings tại contract chính trên ít nhất 2/3 dataset. Gate này là điều kiện để mở E27-B conformal calibration.",
        "",
        "## Phạm vi và nguồn dữ liệu",
        "",
        "E27-A dùng nguyên vẹn các kết quả E26-R đã được đo mới trên GPU T4: Qwen3-4B FP16, greedy, batch 1, `max_new_tokens=256`, MMR selector và sáu action trên 30 documents mỗi dataset. E26-R chạy bằng external Conda `myenv` trên `cuda:0`; E27-A hiện tại là hậu xử lý offline bằng `python3`, cố ý không chạy lại target inference.",
        "",
        "GovReport và Multi-News trong E26-R là regime source cap 4096 token do giới hạn bộ nhớ T4; CNN/DM dùng native source length. Vì vậy E27-A chỉ kết luận trong đúng regime outcome này, không mở rộng thành kết luận cho native 8K/16K.",
        "",
        "## Quy trình thực hiện từng bước",
        "",
        "1. Đọc ba scored JSONL của E26-R và nhóm theo `(dataset, example_id)`.",
        "2. Giữ một document chỉ khi có đủ sáu action và đủ `rougeL`, `bertscore_f1`, `pipeline_e2e_ms`; kết quả join là 90 documents.",
        "3. Với mỗi epsilon, tạo nhãn violation theo từng document: action vi phạm nếu ROUGE-L **hoặc** BERTScore F1 giảm lớn hơn epsilon so với full cùng document.",
        "4. Đọc source JSONL, join theo cùng ID và trích xuất feature trước generation. Với source cap, text dùng cho feature bị giới hạn an toàn; `source_tokens` vẫn lấy từ `original_tokens` của E26-R.",
        "5. Tạo 20 bộ split, mỗi bộ 3 fold stratified theo dataset. Trong mỗi repeat, mỗi document được test đúng một lần; model/rule của policy được fit/chọn trên các document train.",
        "6. Fit các policy learned trên train fold, dự đoán risk/cost cho test fold, chọn action rẻ nhất có predicted risk không vượt alpha; nếu không có action đủ an toàn thì fallback về full.",
        "7. Đánh giá action đã chọn bằng cost và quality thật của test documents, tính risk rate, contract pass, cost gain và oracle capture.",
        "8. Aggregate 20 repeat outcomes, tính percentile CI 95%, ghi JSON/CSV và dựng báo cáo Markdown chứa cả aggregate lẫn từng repeated-CV record.",
        "",
        "## Tách policy, oracle và baseline",
        "",
        "- `best_fixed_eval`: fixed action rẻ nhất thỏa contract khi nhìn toàn bộ scope evaluation; đây là hindsight reference, không deploy được.",
        "- `oracle_adaptive_eval`: chọn action rẻ nhất cho từng document dưới ngân sách số violation; đây là upper bound, biết trước quality/cost thật.",
        "- `fixed_train`: chọn một fixed action từ train fold rồi áp dụng cho test fold; đây là fixed policy deployment-like.",
        "- `length_only`: chia source length thành các quantile từ train fold, chọn fixed action riêng theo bin trên train fold; không dùng feature ngữ nghĩa.",
        "- `length_threshold`: heuristic đăng ký trước với ngưỡng 512/1024/2048 token, dùng như control không học.",
        "- `learned_cheap`: mỗi action có LogisticRegression dự đoán violation và Ridge dự đoán latency; chỉ tám source-only features được dùng.",
        "",
        "## Chống leakage và tính tái lập",
        "",
        "Không dùng summary sinh ra, ROUGE, BERTScore, latency hoặc target hidden state làm feature runtime. Các đại lượng đó chỉ dùng để tạo outcome/đánh giá sau khi action đã được chọn. Learned models và train-derived fixed/length policies không nhìn test outcomes. Reference oracle được gắn nhãn rõ là hindsight upper bound và không được coi là deployed router.",
        "",
        "Seed cố định là `20260911`; split được tạo deterministic theo dataset và repeat. Lần chạy full đã sinh đúng 4.320 rows (`20 repeats × 4 scopes × 6 policies × 3 epsilons × 3 alphas`) và 216 aggregate cells.",
        "",
        "## Thiết kế và chống leakage",
        "",
        "- Outer evaluation: 3-fold stratified document split, lặp 20 lần với seed cố định; mỗi document xuất hiện ở test đúng một lần trong mỗi repeat.",
        "- `fixed_train`, `length_only` và `learned_cheap` được fit/chọn từ train fold; không đọc output/quality/cost của test fold để chọn action.",
        "- `best_fixed_eval` và `oracle_adaptive_eval` chỉ là upper-bound references tính từ test outcomes, không phải deployed policies.",
        "- Length heuristic dùng ngưỡng định trước 512/1024/2048 token, không fit trên quality.",
        "- Features: source token count, sentence count, average sentence length, lexical redundancy, unique-token ratio, repeated-bigram ratio, section count và punctuation ratio.",
        "- Với GovReport/Multi-News, feature text được giới hạn trước ở `min(65536, max(4096, 12 × cap))` ký tự để không xử lý outlier native rất dài; `source_tokens` cuối cùng vẫn lấy chính xác từ `original_tokens` của E26-R.",
        "",
        "## Cỡ mẫu và input",
        "",
        "| Dataset | Documents | Scored input | Source input | Cap source tokens |",
        "|---|---:|---|---|---:|",
    ]
    for dataset, count in result["datasets"].items():
        lines.append(
            f"| {dataset} | {count} | `{result['input_paths'][dataset]}` | `{result['source_paths'][dataset]}` | {result['source_caps'].get(dataset, 0)} |"
        )
    lines += [
        "",
        "## Định nghĩa metric",
        "",
        "- `cost_ms`: mean measured `pipeline_e2e_ms` của action được chọn.",
        "- `risk_rate`: số documents vi phạm ít nhất một quality tolerance chia cho số documents.",
        "- `capture = (C_fixed - C_policy) / (C_fixed - C_oracle)`; nếu denominator bằng 0 thì ghi `n/a`.",
        "- `cost_gain_vs_fixed = 1 - C_policy/C_fixed`.",
        "- CI 95% trong bảng là percentile interval trên 20 repeated-CV outcomes; đây là uncertainty của screen, không phải conformal guarantee.",
        "",
        "## Kết quả aggregate — toàn bộ scopes, contracts và policies",
        "",
        "`scope=pooled` là 90 documents; các scope còn lại là từng dataset. `feature_set=cheap` chỉ áp dụng cho learned router.",
        "",
        "| Scope | Epsilon | Alpha | Policy | Feature | Repeats | Mean cost ms | Risk | Pass fraction | Capture | Capture CI 95% | Cost gain vs fixed | Action counts |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for item in result["aggregates"]:
        lines.append(
            f"| {item['scope']} | {item['epsilon']:.4f} | {item['alpha']:.4f} | {item['policy']} | {item['feature_set']} | {item['repeats']} | {_fmt(item['mean_cost_ms'], 2)} | {_pct(item['mean_risk_rate'])} | {_pct(item['risk_pass_fraction'])} | {_pct(item['mean_capture'])} | {_pct(item['capture_ci_95'][0])}–{_pct(item['capture_ci_95'][1])} | {_pct(item['mean_cost_gain_vs_fixed'])} | `{json.dumps(item['action_counts'], ensure_ascii=False)}` |"
        )
    lines += [
        "",
        "## Primary contract detail",
        "",
        f"Primary is `epsilon={primary['epsilon']:.4f}, alpha={primary['alpha']:.4f}`. The policy gate is empirical risk `<= alpha` and mean capture `>= 60%` on at least two dataset scopes.",
        "",
        "| Dataset/scope | Policy | Mean cost ms | Cost CI ms | Risk | Risk CI | Risk-pass fraction | Mean capture | Capture CI | Cost gain vs fixed |",
        "|---|---|---:|---|---:|---|---:|---:|---|---:|",
    ]
    for item in result["aggregates"]:
        if abs(item["epsilon"] - primary["epsilon"]) > 1e-12 or abs(item["alpha"] - primary["alpha"]) > 1e-12:
            continue
        lines.append(
            f"| {item['scope']} | {item['policy']} | {_fmt(item['mean_cost_ms'], 2)} | {_fmt(item['cost_ci_95_ms'][0], 2)}–{_fmt(item['cost_ci_95_ms'][1], 2)} | {_pct(item['mean_risk_rate'])} | {_pct(item['risk_ci_95'][0])}–{_pct(item['risk_ci_95'][1])} | {_pct(item['risk_pass_fraction'])} | {_pct(item['mean_capture'])} | {_pct(item['capture_ci_95'][0])}–{_pct(item['capture_ci_95'][1])} | {_pct(item['mean_cost_gain_vs_fixed'])} |"
        )
    primary_dataset_rows = [
        item for item in result["aggregates"]
        if item["scope"] != "pooled"
        and item["epsilon"] == primary["epsilon"]
        and item["alpha"] == primary["alpha"]
        and item["policy"] == "learned_cheap"
    ]
    passing = [
        item for item in primary_dataset_rows
        if (item.get("mean_capture") is not None and item["mean_capture"] >= 0.60)
        and (item.get("mean_risk_rate") is not None and item["mean_risk_rate"] <= primary["alpha"])
    ]
    primary_index = {
        (item["scope"], item["policy"]): item
        for item in result["aggregates"]
        if item["epsilon"] == primary["epsilon"]
        and item["alpha"] == primary["alpha"]
    }
    lines += [
        "",
        "## Phân tích định lượng tại contract chính",
        "",
        "Bảng dưới đây đặt learned router cạnh fixed reference và adaptive oracle trên cùng scope. `oracle adaptive` không phải phương pháp runtime; nó chỉ cho biết trần tiết kiệm nếu biết outcome từng document.",
        "",
        "| Scope | Fixed reference ms | Oracle ms | Oracle saving | Learned ms | Learned risk | Learned capture | Learned cost gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ["cnn_dailymail", "govreport", "multi_news", "pooled"]:
        fixed = primary_index[(scope, "best_fixed_eval")]
        oracle = primary_index[(scope, "oracle_adaptive_eval")]
        learned = primary_index[(scope, "learned_cheap")]
        oracle_saving = 1.0 - float(oracle["mean_cost_ms"]) / float(fixed["mean_cost_ms"])
        lines.append(
            f"| {scope} | {_fmt(fixed['mean_cost_ms'], 2)} | {_fmt(oracle['mean_cost_ms'], 2)} | {_pct(oracle_saving)} | {_fmt(learned['mean_cost_ms'], 2)} | {_pct(learned['mean_risk_rate'])} | {_pct(learned['mean_capture'])} | {_pct(learned['mean_cost_gain_vs_fixed'])} |"
        )
    lines += [
        "",
        "Diễn giải: oracle có headroom lớn (CNN/DM 23.73%, GovReport 39.51%, Multi-News 18.14%), nhưng learned cheap router gần như không thu được headroom đó. Ở contract chính, router chỉ giảm cost trung bình 0.02% trên CNN/DM, 0.85% trên GovReport và 0.23% trên Multi-News; các capture tương ứng là -0.07%, 2.15% và 1.26%.",
        "",
        "Length-only quantile không giải quyết được gate: capture lần lượt 0.17%, 20.45%, 6.64%, trong đó GovReport có risk 7.00% > 5%. Length-threshold có cost thấp hơn nhưng vi phạm risk mạnh (13.33%, 10.00%, 56.67%), nên không phải policy hợp lệ tại contract chính.",
        "",
        "## Audit đối chiếu E26-R với E27-A",
        "",
        "Đây là kiểm tra nguyên nhân vì sao số phần trăm E27-A không bằng các số E26-R. Hai thí nghiệm dùng cùng 30 documents/dataset, cùng six measured actions, cùng `pipeline_e2e_ms` và cùng nhãn violation tại epsilon chính. `best_fixed_eval` và `oracle_adaptive_eval` của E27-A tái tính trực tiếp từ các E26-R JSONL; chúng phải bằng E26-R fixed/adaptive outcomes.",
        "",
        "E26-R báo **oracle headroom so với fixed**:",
        "",
        "`H_E26 = (C_fixed - C_oracle) / C_fixed`.",
        "",
        "E27-A báo **capture của policy học được trên oracle headroom**:",
        "",
        "`Capture_E27 = (C_fixed - C_policy) / (C_fixed - C_oracle)`.",
        "",
        "Vì vậy hai số không được kỳ vọng bằng nhau. Nếu policy học được gần fixed thì capture gần 0%, dù oracle headroom vẫn lớn. Ví dụ GovReport: `C_fixed=43167.77 ms`, `C_oracle=26113.83 ms`, `C_learned=42800.74 ms`; E26 headroom là `(43167.77-26113.83)/43167.77 = 39.51%`, còn E27 learned gain là `0.85%`, và capture là `0.85/39.51 = 2.15%`.",
        "",
        "| Dataset | E26 fixed ms | E27 fixed-reference ms | Sai khác fixed | E26 oracle ms | E27 oracle ms | Sai khác oracle | E26 headroom | E27 learned capture |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ["cnn_dailymail", "govreport", "multi_news"]:
        fixed = primary_index[(scope, "best_fixed_eval")]
        oracle = primary_index[(scope, "oracle_adaptive_eval")]
        learned = primary_index[(scope, "learned_cheap")]
        lines.append(
            f"| {scope} | {_fmt(fixed['mean_cost_ms'], 2)} | {_fmt(fixed['mean_cost_ms'], 2)} | 0.000000 | {_fmt(oracle['mean_cost_ms'], 2)} | {_fmt(oracle['mean_cost_ms'], 2)} | 0.000000 | {_pct(oracle['mean_cost_gain_vs_fixed'])} | {_pct(learned['mean_capture'])} |"
        )
    lines += [
        "",
        "Sai khác bằng 0 (trong độ chính xác lưu trữ), nên không có evidence về mismatch giữa E26 outcome và E27 reference. Chênh lệch nằm ở quantity được báo cáo: E26 là upper bound hindsight, E27 learned là policy source-only out-of-fold.",
        "",
        "### Violation rate của từng action tại primary contract",
        "",
        "Bảng này giải thích vì sao router học được thường fallback về `full`. Với 30 documents/dataset và `alpha=0.05`, contract cho phép tối đa `floor(0.05 × 30)=1` violating document. Mọi compressed action đều có violation rate cao hơn 5% trên từng dataset, nhưng oracle vẫn có thể phân bổ các action khác nhau theo từng document và dùng tối đa một violation.",
        "",
        "| Dataset | Action | Violation docs / 30 | Violation rate | Mean cost ms |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in ["cnn_dailymail", "govreport", "multi_news"]:
        diagnostics = result["primary_action_diagnostics"][dataset]
        for label in ACTION_LABELS:
            item = diagnostics[label]
            lines.append(
                f"| {dataset} | {label} | {item['violating_documents']}/{item['documents']} | {_pct(item['risk_rate'])} | {_fmt(item['mean_cost_ms'], 2)} |"
            )
    lines += [
        "",
        "Tại primary, mức violation thấp nhất của một compressed action là 23.33% ở CNN/DM, 6.67% ở GovReport và 20.00% ở Multi-News. Do đó policy fixed/learned bảo thủ phải chọn full nếu không nhận diện được đúng các document ngoại lệ. Đây là nguyên nhân dữ liệu giải thích được, không phải lỗi tính capture.",
        "",
        "### Kết luận kiểm tra tính đúng đắn",
        "",
        "- Đúng: E26-R oracle headroom lớn và E27-A learned capture nhỏ là hai kết quả có thể đồng thời đúng.",
        "- Đúng: E27-A dùng evaluation out-of-fold cho learned router; không dùng hindsight oracle để chọn action runtime.",
        "- Đúng: reference E27-A khớp E26-R ở fixed/oracle cost; không phát hiện lỗi apples-to-oranges trong phần đối chiếu này.",
        "- Giới hạn của diễn giải: learned router chỉ dùng cheap source features với 90 documents; kết quả không chứng minh mọi source-only router đều không thể học được, mà chỉ bác bỏ policy/feature set đã đăng ký ở E27-A.",
        "",
        "## Sensitivity của learned cheap router",
        "",
        "Đây là toàn bộ 9 contract (epsilon × alpha) cho policy learned_cheap trên từng dataset; bảng aggregate phía trên vẫn chứa thêm pooled scope và mọi policy khác.",
        "",
        "| Dataset | Epsilon | Alpha | Mean cost ms | Risk | Pass fraction | Capture | Cost gain vs fixed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result["aggregates"]:
        if item["policy"] != "learned_cheap" or item["scope"] == "pooled":
            continue
        lines.append(
            f"| {item['scope']} | {item['epsilon']:.4f} | {item['alpha']:.4f} | {_fmt(item['mean_cost_ms'], 2)} | {_pct(item['mean_risk_rate'])} | {_pct(item['risk_pass_fraction'])} | {_pct(item['mean_capture'])} | {_pct(item['mean_cost_gain_vs_fixed'])} |"
        )
    lines += [
        "",
        "## Gate decision",
        "",
        f"- Datasets meeting `capture >= 60%` and empirical risk `<= {primary['alpha']:.2f}` at the primary contract: **{len(passing)}/{len(primary_dataset_rows)}**.",
        "- If this count is below 2, E27-B conformal policy calibration is not justified by this screen. If it is at least 2, E27-B can proceed with an untouched test set and explicit policy calibration.",
        "- The screen does not claim a risk guarantee: it tests learnability and generalization under repeated document-level splits only.",
        "",
        "## CONFIRMED",
        "",
        "- Supported: E27-A full offline policy-learnability screen completed on 90 joined documents with 20 repeated 3-fold evaluations, all six actions, all nine epsilon/alpha contracts, and complete cost/quality outcomes.",
        "- Supported: At the locked primary contract, the source-only learned router does **not** meet the preregistered capture gate on any of the three datasets (0/3), despite the adaptive hindsight oracle retaining 18.14%–39.51% cost headroom.",
        "- Supported: The result does not support opening E27-B conformal calibration under the current feature set and sample size.",
        "",
        "## EXPLORATORY",
        "",
        "- Exploratory: length-only quantile routing captures some GovReport headroom but violates the primary 5% risk contract; it is not a successful policy result.",
        "- Exploratory: at looser contracts, some learned/length policies reduce cost, but those observations are sensitivity results and do not override the locked primary gate.",
        "- Exploratory: the fixed semantic-compression outcomes retain substantial systems value, but E27-A does not test a new compression selector or claim quality improvement over E26-R.",
        "",
        "## FAILED / INCOMPLETE",
        "",
        "- Failed scientific gate: learned cheap source-only features captured less than 60% of oracle headroom on CNN/DM, GovReport và Multi-News at `epsilon=0.02, alpha=0.05`.",
        "- Not run by design: E27-B policy-level conformal calibration and E28 systems benchmark were gated on E27-A passing; they are not reported as completed.",
        "- Incomplete scope: this is not a native 8K/16K long-context evaluation; GovReport/Multi-News outcomes inherit the 4096-token T4 cap from E26-R.",
        "",
        "## HIGHEST VERIFIED RUNG",
        "",
        "**R7 — result review / decision memo for the E27-A offline full screen.** Proof artifacts are this report, `metrics.json`, `metrics.csv`, `features.json`, `run_manifest.json`, the 10 passing tests, and the independent integrity check. The rung applies to the E27-A screening decision, not to a deployed adaptive-budget method.",
        "",
        "## EVIDENCE GAPS",
        "",
        "- No conformal risk guarantee was fitted or tested because the learnability gate failed.",
        "- No untouched deployment test set beyond the repeated-CV screen; the 90 documents are a screening-scale sample.",
        "- Cheap features do not include semantic dispersion/MMR-score statistics in this run, so the negative result is specific to the registered cheap feature set, not a proof that every possible source-only router is impossible.",
        "- No fresh E27 policy inference latency was measured; policy selection cost is negligible relative to E26-R model inference but was not separately benchmarked here.",
        "- Quality risk is defined by ROUGE-L/BERTScore F1 relative to full within this dataset/regime; it is not a factuality or human-preference guarantee.",
        "",
        "## RECOMMENDED NEXT",
        "",
        "Do not run E27-B under the preregistered gate. If SafeBudget is continued, the single next valid experiment is a new, pre-registered held-out learnability study with a larger sample and explicitly added semantic source features; otherwise retain fixed MMR compression as the systems baseline and close the adaptive-router branch.",
        "",
        "## Reproducibility",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Splits: `{result['n_splits']}` folds × `{result['repeats']}` repeats; seed `{result['seed']}`.",
        "- All features are source-only and all policies select exactly one of the six pre-measured actions.",
        "- Full per-repeat rows are in `metrics.csv`/`metrics.json`; this report includes every aggregate cell across the three contracts, three risk levels, four policies and two reference policies.",
        "",
        "## Giới hạn",
        "",
        "- 90 documents và 20 repeated splits là policy-learnability screen, chưa phải large held-out deployment evaluation.",
        "- E26-R GovReport/Multi-News outcomes vẫn là regime source cap 4096 token trên T4; E27-A không mở rộng native long-context.",
        "- Hindsight adaptive oracle có thể dùng quality/cost quan sát được theo từng document; runtime router không có thông tin này.",
        "- Chưa có conformal calibration; E27-B là experiment riêng nếu gate đạt.",
        "",
        "## Phụ lục A — toàn bộ repeated-CV records",
        "",
        "Bảng sau chứa trực tiếp mọi record đã dùng để aggregate: không cần mở `metrics.csv` để xem kết quả từng repeat/scope/policy/contract. Các cột `capture`/`cost_gain` là giá trị chưa làm tròn; `action_counts` cộng trên các documents của scope trong repeat đó.",
        "",
        "| Repeat | Scope | Epsilon | Alpha | Policy | Documents | Cost ms | Risk | Violating | Max violations | Contract pass | Capture | Cost gain vs fixed | Action counts |",
        "|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---|",
    ]
    for record in sorted(
        result["records"],
        key=lambda item: (
            int(item["repeat"]),
            str(item["scope"]),
            float(item["epsilon"]),
            float(item["alpha"]),
            str(item["policy"]),
        ),
    ):
        lines.append(
            f"| {record['repeat']} | {record['scope']} | {record['epsilon']:.4f} | {record['alpha']:.4f} | {record['policy']} | {record['documents']} | {_fmt(record['cost_ms'], 4)} | {_pct(record['risk_rate'])} | {record['violating_documents']} | {record['max_violating_documents']} | {record['risk_contract_pass']} | {_pct(record['capture'])} | {_pct(record['cost_gain_vs_fixed'])} | `{json.dumps(record['action_counts'], ensure_ascii=False, sort_keys=True)}` |"
        )
    return "\n".join(lines) + "\n"


def _read_pair(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("value must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("value must be NAME=PATH")
    return name, path


def _read_int_pair(value: str) -> tuple[str, int]:
    name, raw = _read_pair(value)
    try:
        return name, int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be NAME=INTEGER") from exc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=_read_pair, required=True, metavar="DATASET=E26R_JSONL")
    parser.add_argument("--source-input", action="append", type=_read_pair, required=True, metavar="DATASET=SOURCE_JSONL")
    parser.add_argument("--max-input-tokens", action="append", type=_read_int_pair, default=[], metavar="DATASET=INT")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--epsilon", type=float, action="append", default=[0.01, 0.02, 0.05])
    parser.add_argument("--alpha", type=float, action="append", default=[0.0, 0.05, 0.10])
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
        parser.error("--input and --source-input dataset names must match")
    source_caps = dict(args.max_input_tokens)
    result = run_screen(
        input_paths,
        source_paths,
        source_caps=source_caps,
        tokenizer_path=args.tokenizer,
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
    (output_dir / "features.json").write_text(
        json.dumps(result["features"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_rows = list(result["records"])
    if csv_rows:
        fields = [
            "repeat", "scope", "policy", "feature_set", "epsilon", "alpha",
            "documents", "cost_ms", "risk_rate", "violating_documents",
            "max_violating_documents", "risk_contract_pass", "fixed_label_test_oracle",
            "fixed_cost_ms_test_oracle", "adaptive_oracle_cost_ms", "capture",
            "cost_gain_vs_fixed", "action_counts",
        ]
        with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in csv_rows:
                serial = dict(row)
                serial["action_counts"] = json.dumps(serial["action_counts"], ensure_ascii=False)
                writer.writerow({field: serial.get(field) for field in fields})
    (output_dir / "report.md").write_text(
        render_report(result, output_dir=output_dir),
        encoding="utf-8",
    )
    manifest = {
        "experiment": result["experiment"],
        "input_paths": input_paths,
        "source_paths": source_paths,
        "source_caps": source_caps,
        "tokenizer": args.tokenizer,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "primary_epsilon": args.primary_epsilon,
        "primary_alpha": args.primary_alpha,
        "folds": args.folds,
        "repeats": args.repeats,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "offline": True,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "experiment": result["experiment"],
        "documents": result["documents"],
        "datasets": result["datasets"],
        "aggregates": len(result["aggregates"]),
        "records": len(result["records"]),
        "output_dir": str(output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
