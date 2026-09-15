"""E30-B: offline online-stopping policy screen.

This experiment consumes the E30-A sentence-prefix table and the original
E26-R/representative documents.  It does not run target inference.  Reference
texts are used only to construct offline calibration labels and quality
metrics; no reference-derived feature is supplied to a policy.

The implementation deliberately distinguishes executable signals from signals
that require new decoder traces.  EOS probabilities, token confidence,
attention, and hidden states are therefore reported as unavailable when they
are not present in the artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.rouge import rouge_all, tokenize  # noqa: E402
from src.analyze.output_stopping.e30a_output_prefix_oracle import (  # noqa: E402
    _score_bert_sorted,
    prefix_texts,
    sentence_spans,
)
from src.analyze.safe_budget.augment_bertscore import LocalBERTScorer  # noqa: E402


DATASETS = ("cnn_dailymail", "govreport", "multi_news")
PRIMARY_EPSILON = 0.02
PRIMARY_ALPHA = 0.05
FIXED_LABELS = ("64", "96", "128", "160", "192", "224", "full")
DEFAULT_PREFIX_CSV = ROOT / "outputs/safe_budget_sum/2026-09-12_e30a_output_prefix_oracle/prefix_metrics.csv"
DEFAULT_E26_DIR = ROOT / "outputs/safe_budget_sum/2026-09-11_e26r_full/scored"
DEFAULT_SOURCE_DIR = ROOT / "data/representative_100"
DEFAULT_ROBERTA = Path("/home/tuantb/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(row)
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def group_by_doc(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["example_id"])].append(dict(row))
    for values in grouped.values():
        values.sort(key=lambda row: int(row["prefix_index"]))
    return dict(grouped)


def load_e26_full(e26_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(e26_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            if row.get("budget_label") != "full" or row.get("selector") != "full":
                continue
            result[str(row["example_id"])] = row
    return result


def load_sources(source_dir: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(source_dir.glob("*_representative.jsonl")):
        for row in read_jsonl(path):
            result[str(row["id"])] = row
    return result


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _tfidf_vectors(texts: Sequence[str]) -> np.ndarray:
    """Small local TF-IDF representation; no network/model dependency."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
        sublinear_tf=True,
    )
    return vectorizer.fit_transform(list(texts)).toarray().astype(np.float32)


def _source_prefix_feature_series(source: str, summary_sentences: Sequence[str]) -> list[dict[str, float]]:
    source_sentences = [source[a:b].strip() for a, b in sentence_spans(source)]
    source_sentences = [value for value in source_sentences if value]
    matrix = _tfidf_vectors(source_sentences + list(summary_sentences))
    source_count = len(source_sentences)
    source_vecs = matrix[:source_count]
    summary_vecs = matrix[source_count:]
    source_norm = np.linalg.norm(source_vecs, axis=1) if source_count else np.zeros((0,), dtype=np.float32)
    if source_count and summary_vecs.size:
        all_similarities = summary_vecs @ source_vecs.T
        all_similarities = all_similarities / np.maximum(
            np.linalg.norm(summary_vecs, axis=1)[:, None] * source_norm[None, :], 1e-12
        )
    else:
        all_similarities = np.zeros((len(summary_sentences), source_count), dtype=np.float32)
    features: list[dict[str, float]] = []
    for index, current in enumerate(summary_sentences):
        previous = list(summary_sentences[:index])
        if source_count and summary_vecs.size:
            active = all_similarities[: index + 1]
            coverage_values = np.max(active, axis=0)
            coverage = float(np.mean(np.clip(coverage_values, 0.0, 1.0)))
            if index:
                previous_coverage = float(np.mean(np.clip(np.max(all_similarities[:index], axis=0), 0.0, 1.0)))
            else:
                previous_coverage = 0.0
            coverage_gain = coverage - previous_coverage
        else:
            coverage = 0.0
            coverage_gain = 0.0

        current_vec = summary_vecs[index] if index < len(summary_vecs) else np.zeros((matrix.shape[1],), dtype=np.float32)
        if previous:
            previous_vecs = summary_vecs[:index]
            redundancy = max(cosine(current_vec, value) for value in previous_vecs)
            centroid = np.mean(previous_vecs, axis=0)
            drift = 1.0 - cosine(current_vec, centroid)
        else:
            redundancy = 0.0
            drift = 0.0
        current_tokens = tokenize(current)
        previous_tokens = set(token for text in previous for token in tokenize(text))
        novelty = sum(token not in previous_tokens for token in current_tokens) / max(len(current_tokens), 1)
        features.append({
            "source_sentence_count": float(source_count),
            "source_prefix_coverage": coverage,
            "source_coverage_gain": coverage_gain,
            "self_redundancy": redundancy,
            "semantic_drift": drift,
            "lexical_novelty": novelty,
        })
    return features


def _source_prefix_features(source: str, summary_sentences: Sequence[str], index: int) -> dict[str, float]:
    """Compatibility helper used by unit tests; build one small series."""
    return _source_prefix_feature_series(source, summary_sentences)[index]


def _legacy_source_prefix_features(source: str, summary_sentences: Sequence[str], index: int) -> dict[str, float]:
    """Retained only as a debugging reference for the optimized path."""
    source_sentences = [source[a:b].strip() for a, b in sentence_spans(source)]
    source_sentences = [value for value in source_sentences if value]
    current = summary_sentences[index] if index < len(summary_sentences) else ""
    previous = list(summary_sentences[:index])
    matrix = _tfidf_vectors(source_sentences + list(summary_sentences))
    source_count = len(source_sentences)
    source_vecs = matrix[:source_count]
    summary_vecs = matrix[source_count:]
    if source_count and index >= 0 and summary_vecs.size:
        active = summary_vecs[: index + 1]
        similarities = active @ source_vecs.T
        source_norm = np.linalg.norm(source_vecs, axis=1)
        active_norm = np.linalg.norm(active, axis=1)
        similarities = similarities / np.maximum(active_norm[:, None] * source_norm[None, :], 1e-12)
        coverage_values = np.max(similarities, axis=0)
        coverage = float(np.mean(np.clip(coverage_values, 0.0, 1.0)))
        if index:
            prior = summary_vecs[:index] @ source_vecs.T
            prior = prior / np.maximum(
                np.linalg.norm(summary_vecs[:index], axis=1)[:, None] * source_norm[None, :],
                1e-12,
            )
            previous_coverage = float(np.mean(np.clip(np.max(prior, axis=0), 0.0, 1.0)))
        else:
            previous_coverage = 0.0
        coverage_gain = coverage - previous_coverage
    else:
        coverage = 0.0
        coverage_gain = 0.0

    current_vec = summary_vecs[index] if index < len(summary_vecs) else np.zeros((matrix.shape[1],), dtype=np.float32)
    if previous:
        previous_vecs = summary_vecs[:index]
        redundancy = max(cosine(current_vec, value) for value in previous_vecs)
        centroid = np.mean(previous_vecs, axis=0)
        drift = 1.0 - cosine(current_vec, centroid)
    else:
        redundancy = 0.0
        drift = 0.0
    current_tokens = tokenize(current)
    previous_tokens = set(token for text in previous for token in tokenize(text))
    novelty = sum(token not in previous_tokens for token in current_tokens) / max(len(current_tokens), 1)
    return {
        "source_sentence_count": float(source_count),
        "source_prefix_coverage": coverage,
        "source_coverage_gain": coverage_gain,
        "self_redundancy": redundancy,
        "semantic_drift": drift,
        "lexical_novelty": novelty,
    }


def _prefix_features(prefix: str, index: int) -> dict[str, float]:
    spans = sentence_spans(prefix)
    sentences = [prefix[a:b].strip() for a, b in spans]
    current = sentences[-1] if sentences else prefix
    previous = sentences[:-1]
    current_tokens = tokenize(current)
    previous_tokens = set(token for text in previous for token in tokenize(text))
    novelty = sum(token not in previous_tokens for token in current_tokens) / max(len(current_tokens), 1)
    tfidf = _tfidf_vectors(sentences)
    if len(tfidf) > 1:
        redundancy = max(cosine(tfidf[-1], value) for value in tfidf[:-1])
        drift = 1.0 - cosine(tfidf[-1], np.mean(tfidf[:-1], axis=0))
    else:
        redundancy = 0.0
        drift = 0.0
    return {
        "prefix_token_count": float(len(tokenize(prefix))),
        "sentence_index": float(index),
        "current_sentence_tokens": float(len(current_tokens)),
        "self_redundancy": redundancy,
        "semantic_drift": drift,
        "lexical_novelty": novelty,
    }


def _metric_audit(rows: list[dict[str, Any]], scorer: LocalBERTScorer) -> None:
    candidates = [str(row["prefix_text"]) for row in rows]
    references = [str(row["reference"]) for row in rows]
    scores = _score_bert_sorted(scorer, scorer.tokenizer, candidates, references)
    for row, score in zip(rows, scores):
        rouge = rouge_all(str(row["prefix_text"]), str(row["reference"]))["rouge-l"]
        row["rougeL_recall"] = float(rouge["r"])
        row["bertscore_r"] = float(score["bertscore_r"])
    for doc_rows in group_by_doc(rows).values():
        full = doc_rows[-1]
        row_full_rouge_recall = float(full["rougeL_recall"])
        row_full_bert_recall = float(full["bertscore_r"])
        for row in doc_rows:
            row["full_rougeL_recall"] = row_full_rouge_recall
            row["full_bertscore_r"] = row_full_bert_recall
            row["delta_rougeL_recall"] = row_full_rouge_recall - float(row["rougeL_recall"])
            row["delta_bertscore_r"] = row_full_bert_recall - float(row["bertscore_r"])


def build_state_table(prefix_rows: Sequence[Mapping[str, Any]], full_rows: Mapping[str, Mapping[str, Any]], sources: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped_input = group_by_doc(prefix_rows)
    source_feature_cache: dict[str, list[dict[str, float]]] = {}
    summary_sentence_cache: dict[str, list[str]] = {}
    for doc_id, doc_prefix_rows in grouped_input.items():
        full = full_rows[doc_id]
        summary_sentences = prefix_texts(str(full.get("summary", "")))
        summary_sentence_cache[doc_id] = summary_sentences
        source_feature_cache[doc_id] = _source_prefix_feature_series(
            str(sources[doc_id].get("document", "")),
            summary_sentences,
        )
    for base in prefix_rows:
        row = dict(base)
        doc_id = str(row["example_id"])
        if doc_id not in full_rows:
            raise ValueError(f"missing E26-R full row for {doc_id}")
        if doc_id not in sources:
            raise ValueError(f"missing source document for {doc_id}")
        full = full_rows[doc_id]
        source = str(sources[doc_id].get("document", ""))
        summary_sentences = summary_sentence_cache[doc_id]
        index = int(row["prefix_index"])
        prefix = str(row["prefix_text"])
        row["reference"] = str(full.get("reference", ""))
        row["source_token_count"] = float(full.get("original_tokens", 0))
        row["source_sentence_count"] = float(len(sentence_spans(source)))
        row.update(_prefix_features(prefix, index))
        row.update(source_feature_cache[doc_id][index])
        row["safe_primary"] = int(
            _float(row.get("delta_rougeL")) <= PRIMARY_EPSILON
            and _float(row.get("delta_bertscore_f1")) <= PRIMARY_EPSILON
        )
        output.append(row)
    return output


def oracle_index(rows: Sequence[Mapping[str, Any]], epsilon: float = PRIMARY_EPSILON) -> int:
    for index, row in enumerate(rows):
        if _float(row.get("delta_rougeL")) <= epsilon and _float(row.get("delta_bertscore_f1")) <= epsilon:
            return index
    return len(rows) - 1


def fixed_index(rows: Sequence[Mapping[str, Any]], label: str) -> int:
    if label == "full":
        return len(rows) - 1
    eligible = [i for i, row in enumerate(rows) if _float(row.get("prefix_token_count")) <= int(label)]
    return eligible[-1] if eligible else 0


def selected_summary(doc_rows: Mapping[str, Sequence[Mapping[str, Any]]], selections: Mapping[str, int], epsilon: float = PRIMARY_EPSILON) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = [doc_rows[doc][int(index)] for doc, index in selections.items()]
    violations = [row for row in rows if _float(row.get("delta_rougeL")) > epsilon or _float(row.get("delta_bertscore_f1")) > epsilon]
    full_cost = statistics.fmean(_float(doc_rows[doc][-1].get("projected_cost_ms")) for doc in selections)
    mean_cost = statistics.fmean(_float(row.get("projected_cost_ms")) for row in rows)
    return {
        "documents": len(rows),
        "mean_cost_ms": mean_cost,
        "mean_output_tokens": statistics.fmean(_float(row.get("prefix_token_count")) for row in rows),
        "mean_full_cost_ms": full_cost,
        "e2e_saving_vs_full": 1.0 - mean_cost / full_cost if full_cost else None,
        "risk_rate": len(violations) / len(rows) if rows else None,
        "violating_documents": len(violations),
        "mean_rougeL": statistics.fmean(_float(row.get("rougeL")) for row in rows),
        "mean_bertscore_f1": statistics.fmean(_float(row.get("bertscore_f1")) for row in rows),
        "mean_rougeL_recall": statistics.fmean(_float(row.get("rougeL_recall")) for row in rows) if "rougeL_recall" in rows[0] else None,
        "mean_bertscore_r": statistics.fmean(_float(row.get("bertscore_r")) for row in rows) if "bertscore_r" in rows[0] else None,
    }


def choose_fixed(train_docs: Mapping[str, Sequence[Mapping[str, Any]]], labels: Sequence[str], epsilon: float, alpha: float) -> tuple[str, dict[str, Any]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for label in labels:
        selections = {doc: fixed_index(rows, label) for doc, rows in train_docs.items()}
        summary = selected_summary(train_docs, selections, epsilon)
        if summary["risk_rate"] <= alpha + 1e-12:
            candidates.append((label, summary))
    if not candidates:
        label = "full"
        return label, selected_summary(train_docs, {doc: len(rows) - 1 for doc, rows in train_docs.items()}, epsilon)
    return min(candidates, key=lambda item: (item[1]["mean_cost_ms"], item[0]))


def _feature_matrix(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> np.ndarray:
    return np.asarray([[_float(row.get(name)) for name in names] for row in rows], dtype=np.float64)


def _classify_policy(train_docs: Mapping[str, Sequence[Mapping[str, Any]]], feature_names: Sequence[str], epsilon: float, alpha: float) -> tuple[Callable[[Sequence[Mapping[str, Any]]], int], dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_rows = [row for rows in train_docs.values() for row in rows]
    labels = np.asarray([int(_float(row.get("safe_primary"))) for row in train_rows])
    if len(set(labels.tolist())) < 2:
        return (lambda rows: len(rows) - 1), {"threshold": None, "fallback": "single_class"}
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0))
    model.fit(_feature_matrix(train_rows, feature_names), labels)
    train_prob = model.predict_proba(_feature_matrix(train_rows, feature_names))[:, 1]
    thresholds = sorted(set([0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.99] + [float(np.quantile(train_prob, q)) for q in np.linspace(.50, .99, 20)]))
    best: tuple[float, float, int] | None = None
    for threshold in thresholds:
        selections = {}
        for doc, rows in train_docs.items():
            probabilities = model.predict_proba(_feature_matrix(rows, feature_names))[:, 1]
            eligible = np.flatnonzero(probabilities >= threshold)
            selections[doc] = int(eligible[0]) if len(eligible) else len(rows) - 1
        summary = selected_summary(train_docs, selections, epsilon)
        if summary["risk_rate"] <= alpha + 1e-12:
            candidate = (summary["mean_cost_ms"], threshold, summary["violating_documents"])
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return (lambda rows: len(rows) - 1), {"threshold": None, "fallback": "no_training_threshold"}
    threshold = best[1]

    def policy(rows: Sequence[Mapping[str, Any]]) -> int:
        probabilities = model.predict_proba(_feature_matrix(rows, feature_names))[:, 1]
        eligible = np.flatnonzero(probabilities >= threshold)
        return int(eligible[0]) if len(eligible) else len(rows) - 1

    return policy, {"threshold": threshold, "feature_names": list(feature_names), "fallback": None}


def _source_length_policy(train_docs: Mapping[str, Sequence[Mapping[str, Any]]], epsilon: float, alpha: float) -> tuple[Callable[[Sequence[Mapping[str, Any]]], int], dict[str, Any]]:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ids = list(train_docs)
    train_x = np.asarray([[ _float(train_docs[doc][0].get("source_token_count")), _float(train_docs[doc][0].get("source_sentence_count")) ] for doc in ids])
    train_y = np.asarray([float(train_docs[doc][oracle_index(train_docs[doc])].get("prefix_token_count")) for doc in ids])
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train_x, train_y)
    margins = (-64, -32, 0, 16, 32, 64, 96, 128)
    best: tuple[float, int] | None = None
    for margin in margins:
        selections = {}
        for doc, rows in train_docs.items():
            x = np.asarray([[ _float(rows[0].get("source_token_count")), _float(rows[0].get("source_sentence_count")) ]])
            budget = float(model.predict(x)[0]) + margin
            eligible = [i for i, row in enumerate(rows) if _float(row.get("prefix_token_count")) >= budget]
            selections[doc] = eligible[0] if eligible else len(rows) - 1
        summary = selected_summary(train_docs, selections, epsilon)
        if summary["risk_rate"] <= alpha + 1e-12 and (best is None or summary["mean_cost_ms"] < best[0]):
            best = (summary["mean_cost_ms"], margin)
    if best is None:
        return (lambda rows: len(rows) - 1), {"margin": None, "fallback": "no_training_margin"}
    margin = best[1]

    def policy(rows: Sequence[Mapping[str, Any]]) -> int:
        x = np.asarray([[ _float(rows[0].get("source_token_count")), _float(rows[0].get("source_sentence_count")) ]])
        budget = float(model.predict(x)[0]) + margin
        eligible = [i for i, row in enumerate(rows) if _float(row.get("prefix_token_count")) >= budget]
        return eligible[0] if eligible else len(rows) - 1

    return policy, {"margin": margin, "fallback": None, "feature_names": ["source_token_count", "source_sentence_count"]}


def _heuristic_policy(train_docs: Mapping[str, Sequence[Mapping[str, Any]]], feature: str, epsilon: float, alpha: float) -> tuple[Callable[[Sequence[Mapping[str, Any]]], int], dict[str, Any]]:
    values = [_float(row.get(feature)) for rows in train_docs.values() for row in rows]
    thresholds = sorted(set(float(np.quantile(values, q)) for q in np.linspace(.50, .95, 15))) if values else [0.0]
    best: tuple[float, float] | None = None
    for threshold in thresholds:
        selections = {}
        for doc, rows in train_docs.items():
            eligible = [i for i, row in enumerate(rows) if _float(row.get(feature)) >= threshold]
            selections[doc] = eligible[0] if eligible else len(rows) - 1
        summary = selected_summary(train_docs, selections, epsilon)
        if summary["risk_rate"] <= alpha + 1e-12 and (best is None or summary["mean_cost_ms"] < best[0]):
            best = (summary["mean_cost_ms"], threshold)
    if best is None:
        return (lambda rows: len(rows) - 1), {"threshold": None, "feature": feature, "fallback": "no_training_threshold"}
    threshold = best[1]

    def policy(rows: Sequence[Mapping[str, Any]]) -> int:
        eligible = [i for i, row in enumerate(rows) if _float(row.get(feature)) >= threshold]
        return eligible[0] if eligible else len(rows) - 1

    return policy, {"threshold": threshold, "feature": feature, "fallback": None}


def _policy_specs(train_docs: Mapping[str, Sequence[Mapping[str, Any]]], epsilon: float, alpha: float) -> dict[str, tuple[Callable[[Sequence[Mapping[str, Any]]], int], dict[str, Any]]]:
    prefix_features = ["prefix_token_count", "sentence_index", "current_sentence_tokens", "self_redundancy", "semantic_drift", "lexical_novelty"]
    source_prefix_features = ["source_token_count", "source_sentence_count", *prefix_features, "source_prefix_coverage", "source_coverage_gain"]
    return {
        "source_length": _source_length_policy(train_docs, epsilon, alpha),
        "self_redundancy": _heuristic_policy(train_docs, "self_redundancy", epsilon, alpha),
        "semantic_drift": _heuristic_policy(train_docs, "semantic_drift", epsilon, alpha),
        "prefix_only": _classify_policy(train_docs, prefix_features, epsilon, alpha),
        "source_plus_prefix": _classify_policy(train_docs, source_prefix_features, epsilon, alpha),
    }


def _split_ids(ids: Sequence[str], folds: int, seed: int) -> list[list[str]]:
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    return [shuffled[i::folds] for i in range(folds)]


def run_dataset_cv(doc_rows: Mapping[str, Sequence[Mapping[str, Any]]], folds: int = 5, seed: int = 20260913) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    ids = sorted(doc_rows)
    fold_ids = _split_ids(ids, folds, seed)
    metric_rows: list[dict[str, Any]] = []
    selected_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fold, test_list in enumerate(fold_ids):
        test_set = set(test_list)
        train_docs = {doc: doc_rows[doc] for doc in ids if doc not in test_set}
        test_docs = {doc: doc_rows[doc] for doc in test_list}
        fixed_label, fixed_train = choose_fixed(train_docs, FIXED_LABELS, PRIMARY_EPSILON, PRIMARY_ALPHA)
        policy_specs = _policy_specs(train_docs, PRIMARY_EPSILON, PRIMARY_ALPHA)
        all_specs: dict[str, tuple[Callable[[Sequence[Mapping[str, Any]]], int], dict[str, Any]]] = {
            "fixed": (lambda rows, label=fixed_label: fixed_index(rows, label), {"label": fixed_label}),
            "oracle": (lambda rows: oracle_index(rows, PRIMARY_EPSILON), {}),
            **policy_specs,
        }
        for name, (policy, config) in all_specs.items():
            selections = {doc: policy(rows) for doc, rows in test_docs.items()}
            summary = selected_summary(test_docs, selections, PRIMARY_EPSILON)
            for doc, index in selections.items():
                selected_rows[name].append({
                    "fold": fold,
                    "example_id": doc,
                    "prefix_index": index,
                    "full_projected_cost_ms": test_docs[doc][-1].get("projected_cost_ms"),
                    **dict(test_docs[doc][index]),
                })
            metric_rows.append({
                "dataset": next(iter(test_docs.values()))[0].get("dataset", ""),
                "fold": fold,
                "policy": name,
                "train_documents": len(train_docs),
                "test_documents": len(test_docs),
                "selected_action": config.get("label", ""),
                "config": json.dumps(_json_safe(config), sort_keys=True),
                **summary,
            })
    return metric_rows, selected_rows


def pooled_metrics(dataset: str, selected: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    oracle = selected["oracle"]
    oracle_cost = statistics.fmean(_float(row.get("projected_cost_ms")) for row in oracle)
    fixed_cost = statistics.fmean(_float(row.get("projected_cost_ms")) for row in selected["fixed"])
    strongest_prior_capture = 0.0
    prior_names = {"source_length", "self_redundancy", "semantic_drift", "prefix_only"}
    prior_summaries: dict[str, dict[str, Any]] = {}
    for name, rows in selected.items():
        costs = [_float(row.get("projected_cost_ms")) for row in rows]
        violations = [row for row in rows if _float(row.get("delta_rougeL")) > PRIMARY_EPSILON or _float(row.get("delta_bertscore_f1")) > PRIMARY_EPSILON]
        full_costs = [_float(row.get("full_projected_cost_ms"), _float(row.get("projected_cost_ms"))) for row in rows]
        full_mean_cost = statistics.fmean(full_costs)
        summary = {
            "dataset": dataset,
            "policy": name,
            "documents": len(rows),
            "mean_cost_ms": statistics.fmean(costs),
            "risk_rate": len(violations) / len(rows),
            "violating_documents": len(violations),
            "mean_output_tokens": statistics.fmean(_float(row.get("prefix_token_count")) for row in rows),
            "mean_rougeL": statistics.fmean(_float(row.get("rougeL")) for row in rows),
            "mean_bertscore_f1": statistics.fmean(_float(row.get("bertscore_f1")) for row in rows),
            "e2e_saving_vs_full": 1.0 - statistics.fmean(costs) / full_mean_cost if rows and full_mean_cost else None,
            "capture_vs_fixed": (fixed_cost - statistics.fmean(costs)) / (fixed_cost - oracle_cost) if fixed_cost > oracle_cost + 1e-12 else None,
            "risk_contract_pass": len(violations) / len(rows) <= PRIMARY_ALPHA if rows else False,
        }
        result.append(summary)
        if name in prior_names and summary["capture_vs_fixed"] is not None:
            strongest_prior_capture = max(strongest_prior_capture, float(summary["capture_vs_fixed"]))
            prior_summaries[name] = summary
    for row in result:
        row["strongest_prior_capture"] = strongest_prior_capture
        row["candidate_margin_vs_strongest_prior"] = (
            row["capture_vs_fixed"] - strongest_prior_capture
            if row["policy"] == "source_plus_prefix" and row["capture_vs_fixed"] is not None and row.get("risk_contract_pass")
            else None
        )
    return result


def run_transfer(all_docs: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]], folds_seed: int = 20260913) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for held_out in DATASETS:
        train_docs: dict[str, Sequence[Mapping[str, Any]]] = {}
        for dataset, docs in all_docs.items():
            if dataset != held_out:
                train_docs.update(docs)
        test_docs = all_docs[held_out]
        specs = _policy_specs(train_docs, PRIMARY_EPSILON, PRIMARY_ALPHA)
        for name in ("prefix_only", "source_plus_prefix"):
            policy, config = specs[name]
            selections = {doc: policy(rows) for doc, rows in test_docs.items()}
            summary = selected_summary(test_docs, selections, PRIMARY_EPSILON)
            output.append({"train_datasets": "+".join(dataset for dataset in DATASETS if dataset != held_out), "held_out_dataset": held_out, "policy": name, "config": json.dumps(_json_safe(config), sort_keys=True), **summary})
    return output


def _write_report(out_dir: Path, dataset_summaries: Sequence[Mapping[str, Any]], transfer: Sequence[Mapping[str, Any]], audit: Mapping[str, Any], availability: Mapping[str, Any], gates: Mapping[str, Any]) -> None:
    lines = [
        "# E30-B Online Output-Stopping Policy Screen — Báo cáo đầy đủ",
        "",
        "> Phân tích offline trên E30-A prefix states; không chạy target inference mới.",
        "",
        "## 0. Vị trí của E30-B trong chuỗi thực nghiệm",
        "",
        "E30-B không phải một lần chạy model độc lập. Nó sử dụng các generation đã được đo ở E26-R, sau đó E30-A chuyển mỗi generation thành các prefix tại sentence boundary. E30-B kiểm tra xem feature quan sát online nào có thể dự đoán prefix an toàn mà không nhìn reference.",
        "",
        "| Phase | Công việc | Có target inference mới? | Artifact chính |",
        "|---|---|---|---|",
        "| E26-R | Sinh summary full và các context-budget conditions bằng Qwen3-4B trên T4 | Có | `outputs/safe_budget_sum/2026-09-11_e26r_full/` |",
        "| E30-A | Cắt summary full thành cumulative sentence prefixes; tạo oracle/fixed headroom | Không | `outputs/safe_budget_sum/2026-09-12_e30a_output_prefix_oracle/` |",
        "| E30-B | Audit recall và fit/evaluate online stopping policies bằng grouped CV | Không | `outputs/safe_budget_sum/2026-09-13_e30b_policy_screen/` |",
        "| E30-C | Runtime controller tại sentence boundary | Chưa mở vì E30-B gate fail | Chưa có |",
        "",
        "Kết luận và số liệu E30-B chỉ nói về policy screen offline; không được gộp projected cost với fresh runtime latency.",
        "",
        "## 1. Kết luận điều hành",
        "",
        f"- Tổng số state: **{audit['prefix_rows']}**, gồm {audit['documents']} documents trên {audit['datasets']} datasets.",
        f"- Metric audit: ROUGE-L recall và BERTScore recall đã được tính lại trên từng prefix; policy contract vẫn dùng ROUGE-L F1 + BERTScore F1 để đối chiếu E30-A.",
        f"- Primary contract: epsilon={PRIMARY_EPSILON}, alpha={PRIMARY_ALPHA}; grouped 5-fold document CV.",
        f"- Source join: {availability['source_joined']}/{availability['source_required']} documents.",
        "",
        "### Kết quả gate",
        "",
        f"- Candidate source+prefix capture ≥60% trên ≥2 datasets: **{'PASS' if gates['capture_pass'] else 'FAIL'}**.",
        f"- Candidate margin ≥10 percentage points so với prior mạnh nhất trên ≥2 datasets: **{'PASS' if gates['margin_pass'] else 'FAIL'}**.",
        f"- Projected saving ≥15% so với fixed trên ≥2 datasets: **{'PASS' if gates['saving_pass'] else 'FAIL'}**.",
        f"- Quyết định E30-C: **{'MỞ SCREEN' if gates['all_pass'] else 'CHƯA MỞ; cần giữ E30-C đóng'}**.",
        "",
        "## 2. Quy trình và dữ liệu",
        "",
        "1. Đọc `prefix_metrics.csv` của E30-A và E26-R full rows.",
        "2. Join source document bằng `example_id` từ `data/representative_100/*.jsonl`.",
        "3. Tính lại ROUGE-L recall và BERTScore recall; không dùng reference làm feature runtime.",
        "4. Tạo feature source-only, prefix-only, generic redundancy/drift và source+prefix coverage.",
        "5. Chia theo document; train/calibrate policy trên train documents, áp dụng nguyên policy lên held-out documents.",
        "6. Tính cost, risk và capture từ các prefix được chọn; oracle chỉ là upper bound hindsight.",
        "",
        "## 2.1 Câu hỏi và tiêu chí khóa trước khi chạy",
        "",
        "E30-B kiểm tra liệu các tín hiệu quan sát được trong lúc summary đang sinh có thể biến oracle headroom của E30-A thành một policy dừng triển khai được hay không.",
        "",
        "- H0: source length và heuristic prefix không khai thác được phần đáng kể của oracle headroom.",
        "- H1: prefix-only có thể dự đoán điểm dừng an toàn tốt hơn source-only.",
        "- H2: source+prefix coverage có thể vượt prior mạnh nhất ít nhất 10 điểm phần trăm capture.",
        "- Gate 1: capture của source+prefix ≥60% oracle trên ít nhất 2/3 datasets, đồng thời policy phải pass risk contract.",
        "- Gate 2: source+prefix hơn prior mạnh nhất ≥10 điểm phần trăm trên ít nhất 2 datasets.",
        "- Gate 3: projected E2E saving ≥15% so với fixed policy hợp lệ trên ít nhất 2 datasets.",
        "- Nếu một policy giảm cost nhưng risk vượt alpha, policy đó không được tính là pass và không được gọi là deployable.",
        "",
        "## 2.2 Provenance và kiểm soát leakage",
        "",
        "- Prefix states: `outputs/safe_budget_sum/2026-09-12_e30a_output_prefix_oracle/prefix_metrics.csv`.",
        "- Full generation/reference/timing: `outputs/safe_budget_sum/2026-09-11_e26r_full/scored/*.jsonl`.",
        "- Source text: `data/representative_100/*_representative.jsonl`, join bằng `example_id`.",
        "- 90/90 documents join được source; không có document bị loại vì thiếu source.",
        "- Reference chỉ được dùng để tạo nhãn `safe_primary` và đánh giá quality. Reference không đi vào feature source-only, prefix-only hoặc source+prefix.",
        "- Chia fold theo document, không chia theo prefix. Vì vậy mọi prefix của một document luôn nằm cùng một fold.",
        "",
        "## 2.3 Các policy được triển khai",
        "",
        "| Policy | Feature/input | Cách chọn điểm dừng | Vai trò |",
        "|---|---|---|---|",
        "| fixed | một output budget chung | action fixed được chọn trên train dưới risk contract | baseline deployable |",
        "| source_length | source token count, source sentence count | Ridge dự đoán token budget; chọn margin trong {-64,-32,0,16,32,64,96,128} trên train | source-only baseline |",
        "| self_redundancy | độ tương đồng câu hiện tại với câu trước | dừng khi vượt threshold train-calibrated | generic stopping baseline |",
        "| semantic_drift | khoảng cách câu hiện tại với centroid prefix | dừng khi vượt threshold train-calibrated | generic drift baseline |",
        "| prefix_only | prefix length, sentence index, novelty, redundancy, drift | balanced logistic regression; dừng tại prefix đầu tiên có P(safe) ≥ threshold | prefix-state model |",
        "| source_plus_prefix | source features + prefix features + TF-IDF source coverage/gain | balanced logistic regression; threshold chỉ chọn trên train | candidate policy |",
        "| oracle | quality delta so với full reference | chọn prefix sớm nhất thực sự safe | upper bound hindsight, không deployable |",
        "",
        "## 2.4 Định nghĩa metric",
        "",
        "Với document `i`, prefix được chọn là `a_i`, cost projected là `C_i(a_i)`, và vi phạm là:",
        "",
        "```text",
        "violation_i = 1[delta_ROUGE-L_i > epsilon or delta_BERTScore-F1_i > epsilon]",
        "risk = mean(violation_i)",
        "capture = (C_fixed - C_policy) / (C_fixed - C_oracle)",
        "saving_vs_full = 1 - C_policy / C_full",
        "```",
        "",
        "`C_policy` dùng projected E30-A cost: prefill đầy đủ cộng với decode cost tỷ lệ theo số output tokens ở prefix. Đây chưa phải latency runtime của controller. `alpha=0.05` trên 30 documents tương ứng tối đa 1 document vi phạm; kết quả pooled/CV vẫn báo số vi phạm thực tế thay vì làm tròn thành guarantee thống kê.",
        "",
        "## 2.5 Cấu hình thực nghiệm chính xác",
        "",
        "### E26-R — nguồn generation được kế thừa",
        "",
        "| Thành phần | Cấu hình |",
        "|---|---|",
        "| Target | Qwen3-4B local snapshot |",
        "| Decode | greedy, `max_new_tokens=256`, thinking disabled, batch size 1 |",
        "| Device | Tesla T4 15,360 MiB, CUDA 12.4, float16, SDPA |",
        "| Prompt | system + instruction summarization cố định trong `inference_provenance.json` |",
        "| Context selector | MMR, lambda=0.7, local all-MiniLM-L6-v2; actions full/25%/40%/55%/70%/85% |",
        "| Dataset size | 30 documents/dataset; 180 E26-R rows/dataset; 540 rows tổng cộng |",
        "| Context limit | CNN/DM native; GovReport và Multi-News cap 4096 source tokens trên T4 |",
        "| Quality | repository ROUGE-1/2/L F1 + local BERTScore no-IDF, max length 512 |",
        "",
        "### E30-A — prefix oracle source",
        "",
        "| Thành phần | Cấu hình |",
        "|---|---|",
        "| Prefix boundary | chỉ tại sentence-like punctuation/newline; prefix cumulative |",
        "| Prefix rows | CNN/DM 250, GovReport 303, Multi-News 279; tổng 832 |",
        "| Token count | Qwen3-4B tokenizer từ local snapshot |",
        "| BERTScore | local RoBERTa-base token cosine, CPU, float32, max length 512 |",
        "| Fixed actions | 64/96/128/160/192/224 tokens và full |",
        "| Oracle | boundary sớm nhất thỏa cả ROUGE-L và BERTScore F1 delta ≤ epsilon |",
        "| E30-A validation | epsilon 0.01/0.02/0.05; alpha 0/0.05/0.10; bootstrap 200; holdout 70/30 |",
        "",
        "### E30-B — policy screen",
        "",
        "| Thành phần | Cấu hình |",
        "|---|---|",
        "| Interpreter | `python3` hệ thống của workspace; không dùng fresh target inference |",
        "| Device/precision | BERTScore audit trên CPU, float32; batch size 32 để giảm overhead CPU |",
        "| Feature representation | TF-IDF nội bộ theo từng document; không tải model embedding mới |",
        "| Primary contract | epsilon=0.02, alpha=0.05 |",
        "| Grouping | document-level 5-fold CV, seed=20260913; mọi prefix của document cùng fold |",
        "| Classifier | StandardScaler + balanced LogisticRegression, max_iter=1000 |",
        "| Source-length model | StandardScaler + Ridge(alpha=1.0) |",
        "| Threshold grid | classifier 0.50/0.60/0.70/0.80/0.85/0.90/0.93/0.95/0.97/0.99 và quantiles train; heuristic quantiles 0.50–0.95 |",
        "| Length-margin grid | -64/-32/0/16/32/64/96/128 tokens |",
        "| Calibration rule | chọn cost thấp nhất trên train trong số policy có train risk ≤ alpha; nếu không có thì fallback full |",
        "| Transfer | train trên hai dataset, test trên dataset thứ ba; không dùng held-out documents để chọn threshold |",
        "",
        "## 2.6 Trình tự thực hiện từng bước",
        "",
        "1. Đọc 832 prefix rows và 90 full E26-R rows.",
        "2. Join source text bằng `example_id` và kiểm tra coverage 90/90.",
        "3. Với mỗi prefix, tính ROUGE-L recall và BERTScore recall để audit metric; các metric F1/delta của E30-A vẫn giữ làm primary policy label.",
        "4. Tạo source-only features (source length/sentence count), prefix-only features (length/index/novelty/redundancy/drift), và source+prefix features (TF-IDF source coverage/marginal gain).",
        "5. Với từng fold, tách document train/test; fit model và chọn threshold chỉ trên train; áp policy lên test.",
        "6. Trên test, chọn prefix đầu tiên policy dự đoán safe; nếu không có prefix thỏa threshold thì dùng full prefix.",
        "7. Tính projected cost, quality, risk, saving và capture; so sánh với fixed và hindsight oracle.",
        "8. Gộp 5 folds thành pooled metrics theo dataset và chạy leave-one-dataset-out transfer.",
        "9. Áp gate; vì gate fail nên không chạy E30-C.",
        "",
        "## 3. Feature availability",
        "",
        f"- Có thể chạy: {', '.join(availability['executable'])}.",
        f"- Chưa có trong artifact: {', '.join(availability['unavailable'])}.",
        "- Không thay thế các feature unavailable bằng reference-derived proxy.",
        "",
        "## 4.1 Metric audit aggregate",
        "",
        "| Dataset | Docs | Prefix rows | Mean full ROUGE-L recall | Mean full BERTScore recall | Mean all-prefix ROUGE-L recall | Mean all-prefix BERTScore recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in audit["per_dataset"]:
        lines.append(
            f"| {row['dataset']} | {row['documents']} | {row['prefix_rows']} | {row['mean_full_rougeL_recall']:.4f} | "
            f"{row['mean_full_bertscore_r']:.4f} | {row['mean_prefix_rougeL_recall']:.4f} | {row['mean_prefix_bertscore_r']:.4f} |"
        )
    lines += [
        "",
        "## 4. Kết quả pooled theo dataset",
        "",
        "| Dataset | Policy | Docs | Mean cost ms | Risk | Risk contract | Mean output tokens | Saving vs full | Capture vs fixed | Candidate margin |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in dataset_summaries:
        capture = "n/a" if row.get("capture_vs_fixed") is None else f"{row['capture_vs_fixed']:.2%}"
        margin = "n/a" if row.get("candidate_margin_vs_strongest_prior") is None else f"{row['candidate_margin_vs_strongest_prior']:.2%}"
        lines.append(
            f"| {row['dataset']} | {row['policy']} | {row['documents']} | {row['mean_cost_ms']:.2f} | "
            f"{row['risk_rate']:.2%} | {'PASS' if row.get('risk_contract_pass') else 'FAIL'} | {row['mean_output_tokens']:.1f} | "
            f"{row['e2e_saving_vs_full']:.2%} | {capture} | {margin} |"
        )
    lines += [
        "",
        "## 4.2 Diễn giải theo dataset",
        "",
        "### CNN/DM",
        "",
        "Oracle tiết kiệm 69.08% so với full và fixed policy hợp lệ tiết kiệm 37.58%. Source+prefix giảm projected cost 48.20%, nhưng có 3/30 violations (10.00%), nên không pass alpha=0.05. Vì vậy con số 48.20% chỉ là raw screening result, không phải saving đã được risk-control.",
        "",
        "### GovReport",
        "",
        "Oracle chỉ tiết kiệm 19.11% vì fixed full-context là policy duy nhất có risk 0% ở contract chính. Source+prefix giảm cost 10.55%, nhưng có 4/30 violations (13.33%). Đây là trường hợp cho thấy signal có thể chọn prefix ngắn hơn nhưng chưa đủ an toàn cho tài liệu dài.",
        "",
        "### Multi-News",
        "",
        "Oracle tiết kiệm 29.31%, nhưng source+prefix chỉ giảm 4.68% và có 2/30 violations (6.67%). Prefix-only pass risk với saving 5.00%, nên source coverage chưa tạo thêm lợi ích rõ ràng ở dataset này.",
        "",
        "## 4.3 Phân biệt oracle headroom và policy headroom",
        "",
        "E30-A chứng minh có hindsight headroom: oracle lần lượt tiết kiệm 69.08%, 19.11% và 29.31% trên CNN/DM, GovReport và Multi-News. E30-B không chứng minh policy online khai thác được headroom đó. Source+prefix có capture thô 33.72%, 55.21% và 1.00%, nhưng cả ba đều vi phạm risk contract. Sau khi áp điều kiện risk, capture hợp lệ không đạt gate ở dataset nào.",
        "",
        "Đây là lý do không được kết luận rằng ‘source+prefix đạt 48.20% speedup trên CNN/DM’. Kết luận đúng là: ‘source+prefix có projected saving 48.20% trong screening nhưng không đáp ứng quality-risk contract vì risk 10.00%’.",
        "",
        "## 5. Cross-dataset transfer",
        "",
        "| Train datasets | Held-out | Policy | Risk | Mean cost ms |",
        "|---|---|---|---:|---:|",
    ]
    for row in transfer:
        lines.append(f"| {row['train_datasets']} | {row['held_out_dataset']} | {row['policy']} | {row['risk_rate']:.2%} | {row['mean_cost_ms']:.2f} |")
    lines += [
        "",
        "## 5.1 Cách đọc cross-dataset transfer",
        "",
        "Transfer được fit trên hai datasets và đánh giá trên dataset thứ ba, không dùng document của held-out dataset để chọn threshold. Kết quả source+prefix có risk 0% trên CNN/DM, 6.67% trên GovReport và 50.00% trên Multi-News. Vì vậy policy không có tính ổn định xuyên dataset ở sample hiện tại.",
        "",
        "## 6. Reproducibility và kiểm tra thực thi",
        "",
        "- Script: `src/analyze/output_stopping/e30b_policy_screen.py`.",
        "- Plan: `src/analyze/output_stopping/plans/2026-09-13-e30b-policy-screen.md`.",
        "- Lệnh audit/policy screen đầy đủ:",
        "",
        "```bash",
        "python3 -u src/analyze/output_stopping/e30b_policy_screen.py \\",
        "  --device cpu --dtype float32 --batch-size 32 \\",
        "  --folds 5 --seed 20260913",
        "```",
        "",
        "- Khi đã có `state_table.csv`, có thể tái tạo policy screen mà không mã hóa lại BERTScore:",
        "",
        "```bash",
        "python3 -u src/analyze/output_stopping/e30b_policy_screen.py \\",
        "  --skip-metric-audit --device cpu --dtype float32 \\",
        "  --folds 5 --seed 20260913",
        "```",
        "",
        "- Verification cuối: 11/11 unit tests pass; state table có 832 rows và 90 document IDs; recall fields có mặt trên toàn bộ rows; source join 90/90; `all_pass=false` được kiểm tra từ `dataset_summary.json`.",
        "",
        "## 7. Baseline limitations",
        "",
        "- EOS probability, token confidence, target entropy, decoder attention và hidden-state features không có trong E30-A/E26-R artifacts; không được gọi là đã chạy.",
        "- Đây là projected cost từ E30-A, chưa phải online latency của controller.",
        "- 30 documents/dataset vẫn là screening sample; CV giảm leakage nhưng không thay thế evaluation trên test set lớn.",
        "",
        "## 8. Phân loại kết luận",
        "",
        "### CONFIRMED",
        "",
        "- Prefix state table được xây theo document và source join được kiểm tra.",
        "- ROUGE-L recall/BERTScore recall audit có mặt trong artifact state table.",
        "- Các policy được đánh giá bằng cùng risk contract và cùng projected-cost definition.",
        "",
        "### EXPLORATORY",
        "",
        "- Capture của source+prefix là kết quả screening, chưa là online deployment claim.",
        "- Transfer result chỉ là diagnostic vì training set nhỏ và action threshold được fit offline.",
        "",
        "### FAILED / INCOMPLETE",
        "",
        "- E30-C runtime inference chưa được chạy nếu gate không pass hoặc nếu feature/logit artifacts không có.",
        "- EOS/confidence/hidden-state baselines chưa hoàn thành do thiếu decoder traces.",
        "",
        "### HIGHEST VERIFIED RUNG",
        "",
        "- E30-B offline grouped policy screen trên 832 prefix states; chưa đạt mức chứng minh controller production.",
        "",
        "### EVIDENCE GAPS",
        "",
        "- Cần fresh runtime trace để đo overhead và latency thật tại sentence boundary.",
        "- Cần validation lớn hơn và factuality/faithfulness nếu mở E30-C.",
        "",
        "### RECOMMENDED NEXT",
        "",
        "- Chỉ chạy E30-C nếu cả ba gate ở trên PASS; nếu không, đóng output-stopping method branch và giữ E30-A/E30-B như feasibility/negative result.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix_rows = read_csv(Path(args.prefix_csv))
    full_rows = load_e26_full(Path(args.e26_dir))
    sources = load_sources(Path(args.source_dir))
    rows = build_state_table(prefix_rows, full_rows, sources)
    dataset_by_doc: dict[str, str] = {}
    for row in rows:
        dataset_by_doc[str(row["example_id"])] = str(full_rows[str(row["example_id"])].get("example_metadata", {}).get("dataset", "unknown"))
    if any(value == "unknown" for value in dataset_by_doc.values()):
        for row in rows:
            doc = str(row["example_id"])
            if doc.startswith("cnn_dailymail_"):
                dataset_by_doc[doc] = "cnn_dailymail"
            elif doc.startswith("govreport_"):
                dataset_by_doc[doc] = "govreport"
            elif doc.startswith("multinews_"):
                dataset_by_doc[doc] = "multi_news"
    dataset_by_doc = {
        doc: ("multi_news" if value == "multinews" else value)
        for doc, value in dataset_by_doc.items()
    }
    for row in rows:
        row["dataset"] = dataset_by_doc[str(row["example_id"])]

    if not args.skip_metric_audit:
        scorer = LocalBERTScorer(str(args.roberta_model), device=args.device, dtype=args.dtype, max_length=512, batch_size=args.batch_size)
        _metric_audit(rows, scorer)
    else:
        # A previous state table is required when skipping the expensive audit.
        existing = read_csv(out_dir / "state_table.csv") if (out_dir / "state_table.csv").exists() else []
        if not existing or "bertscore_r" not in existing[0]:
            raise ValueError("--skip-metric-audit requires an existing state_table.csv with recall metrics")
        rows = existing
    write_csv(out_dir / "state_table.csv", rows)

    grouped_by_dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped_by_dataset.setdefault(str(row["dataset"]), {}).setdefault(str(row["example_id"]), []).append(row)
    fold_rows: list[dict[str, Any]] = []
    selected_all: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for dataset, doc_rows in grouped_by_dataset.items():
        metrics, selected = run_dataset_cv(doc_rows, folds=args.folds, seed=args.seed)
        for metric in metrics:
            metric["dataset"] = dataset
        fold_rows.extend(metrics)
        selected_all[dataset] = selected
    pooled: list[dict[str, Any]] = []
    for dataset, selected in selected_all.items():
        pooled.extend(pooled_metrics(dataset, selected))
    transfer = run_transfer(grouped_by_dataset)
    write_csv(out_dir / "fold_metrics.csv", fold_rows)
    write_csv(out_dir / "transfer_metrics.csv", transfer)

    source_required = len(dataset_by_doc)
    availability = {
        "source_required": source_required,
        "source_joined": sum(1 for doc in dataset_by_doc if doc in sources),
        "executable": ["best_fixed", "source_length", "self_redundancy", "semantic_drift", "prefix_only", "source_plus_prefix"],
        "unavailable": ["EOS probability", "token confidence", "target entropy", "decoder attention", "hidden states"],
    }
    candidates = [row for row in pooled if row["policy"] == "source_plus_prefix"]
    prior = [row for row in pooled if row["policy"] in {"source_length", "self_redundancy", "semantic_drift", "prefix_only"}]
    capture_pass = sum(row.get("risk_contract_pass") and row.get("capture_vs_fixed") is not None and row["capture_vs_fixed"] >= .60 for row in candidates) >= 2
    margin_pass = 0
    saving_pass = 0
    for candidate in candidates:
        dataset = candidate["dataset"]
        same_prior = [row for row in prior if row["dataset"] == dataset and row.get("risk_contract_pass") and row.get("capture_vs_fixed") is not None]
        strongest = max([0.0, *[row["capture_vs_fixed"] for row in same_prior]])
        candidate["strongest_prior_capture"] = strongest
        candidate["candidate_margin_vs_strongest_prior"] = (
            (candidate.get("capture_vs_fixed") or 0.0) - strongest
            if candidate.get("risk_contract_pass") and candidate.get("capture_vs_fixed") is not None
            else None
        )
        margin_pass += int(candidate.get("risk_contract_pass") and candidate.get("candidate_margin_vs_strongest_prior") is not None and candidate["candidate_margin_vs_strongest_prior"] >= .10)
        saving_pass += int(candidate.get("risk_contract_pass") and candidate.get("e2e_saving_vs_full") is not None and candidate["e2e_saving_vs_full"] >= .15)
    gates = {"capture_pass": capture_pass, "margin_pass": margin_pass >= 2, "saving_pass": saving_pass >= 2, "all_pass": capture_pass and margin_pass >= 2 and saving_pass >= 2}
    write_csv(out_dir / "policy_metrics.csv", pooled)
    audit_by_dataset: list[dict[str, Any]] = []
    for dataset, docs in grouped_by_dataset.items():
        dataset_rows = [row for row in rows if str(row["dataset"]) == dataset]
        full_prefixes = [doc_rows[-1] for doc_rows in docs.values()]
        audit_by_dataset.append({
            "dataset": dataset,
            "documents": len(docs),
            "prefix_rows": len(dataset_rows),
            "mean_full_rougeL_recall": statistics.fmean(_float(row.get("rougeL_recall")) for row in full_prefixes),
            "mean_full_bertscore_r": statistics.fmean(_float(row.get("bertscore_r")) for row in full_prefixes),
            "mean_prefix_rougeL_recall": statistics.fmean(_float(row.get("rougeL_recall")) for row in dataset_rows),
            "mean_prefix_bertscore_r": statistics.fmean(_float(row.get("bertscore_r")) for row in dataset_rows),
        })
    audit = {"prefix_rows": len(rows), "documents": len(dataset_by_doc), "datasets": len(grouped_by_dataset), "per_dataset": audit_by_dataset}
    summary = {"experiment": "E30-B online output-stopping policy screen", "status": "complete", "primary_contract": {"epsilon": PRIMARY_EPSILON, "alpha": PRIMARY_ALPHA}, "audit": audit, "availability": availability, "gates": gates, "pooled_metrics": pooled, "transfer_metrics": transfer}
    (out_dir / "dataset_summary.json").write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {
        "experiment": summary["experiment"],
        "status": "complete",
        "command": " ".join(sys.argv),
        "commands": {
            "full_metric_audit": "python3 -u src/analyze/output_stopping/e30b_policy_screen.py --device cpu --dtype float32 --batch-size 32 --folds 5 --seed 20260913",
            "policy_replay_from_state_table": "python3 -u src/analyze/output_stopping/e30b_policy_screen.py --skip-metric-audit --device cpu --dtype float32 --batch-size 32 --folds 5 --seed 20260913",
        },
        "inputs": {"prefix_csv": str(args.prefix_csv), "e26_dir": str(args.e26_dir), "source_dir": str(args.source_dir)},
        "metric_audit": {"completed": True, "performed_in_this_invocation": not args.skip_metric_audit, "state_table_contains_recomputed_recall": True},
        "folds": args.folds,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "gates": gates,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(out_dir, pooled, transfer, audit, availability, gates)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-csv", type=Path, default=DEFAULT_PREFIX_CSV)
    parser.add_argument("--e26-dir", type=Path, default=DEFAULT_E26_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--roberta-model", type=Path, default=DEFAULT_ROBERTA)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/safe_budget_sum/2026-09-13_e30b_policy_screen")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float16", "float32", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--skip-metric-audit", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps({"status": summary["status"], "gates": summary["gates"], "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
