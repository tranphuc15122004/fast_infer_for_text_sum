"""P1 cheap admission predictor for the GroundSync/BurstSpec decision.

The model in this module is intentionally small and causal.  It predicts only
the binary event ``accepted_len > 0`` from signals available at proposal entry;
it never reads grounding, future attention, accepted length, or any other
post-verification value while making a policy decision.  Evaluation is split
by document and utility is computed from the measured common timing rows.

This is an analysis runner, not a serving implementation.  Its output is
therefore explicit about calibration/test coverage and about whether an
observed utility gain recovers the first-token admission-oracle gap.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .p0_decision import _committed, _finite, _timing_cost


FEATURE_NAMES = (
    "target_entropy_at_start",
    "draft_confidence_first",
    "recent_acceptance",
    "output_position_log1p",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSONL records without silently dropping malformed rows."""

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected JSON object")
        rows.append(value)
    return rows


def _ok(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status") == "ok"]


def _doc_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    train_fraction: float = 0.6,
    dev_fraction: float = 0.2,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, list[str]]]:
    """Create deterministic train/dev/test partitions at document granularity."""

    if not 0.0 < train_fraction < 1.0 or not 0.0 <= dev_fraction < 1.0:
        raise ValueError("invalid split fractions")
    if train_fraction + dev_fraction >= 1.0:
        raise ValueError("train + dev fractions must be below one")
    documents = sorted({str(row.get("document_id")) for row in rows})
    train_count = max(1, int(len(documents) * train_fraction)) if documents else 0
    dev_count = max(1, int(len(documents) * dev_fraction)) if len(documents) >= 3 else 0
    if train_count + dev_count >= len(documents) and len(documents) >= 3:
        dev_count = max(1, len(documents) - train_count - 1)
    train_ids = documents[:train_count]
    dev_ids = documents[train_count : train_count + dev_count]
    test_ids = documents[train_count + dev_count :]
    sets = {
        "train": train_ids,
        "dev": dev_ids,
        "test": test_ids,
    }
    lookup = {name: set(values) for name, values in sets.items()}
    partitions = tuple(
        [row for row in rows if str(row.get("document_id")) in lookup[name]]
        for name in ("train", "dev", "test")
    )
    return partitions[0], partitions[1], partitions[2], sets


def _feature_value(row: Mapping[str, Any], name: str) -> float | None:
    if name == "target_entropy_at_start":
        return _finite(row.get(name))
    if name == "draft_confidence_first":
        values = row.get("draft_confidence")
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
            return None
        return _finite(values[0])
    if name == "recent_acceptance":
        return _finite(row.get(name))
    if name == "output_position_log1p":
        value = _finite(row.get("start_position"))
        return math.log1p(max(0.0, value)) if value is not None else None
    raise KeyError(name)


def feature_matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[Any, Any, list[Mapping[str, Any]]]:
    """Return finite feature matrix and admission labels for usable rows."""

    import numpy as np

    usable: list[Mapping[str, Any]] = []
    features: list[list[float]] = []
    labels: list[float] = []
    for row in _ok(rows):
        values = [_feature_value(row, name) for name in FEATURE_NAMES]
        if any(value is None or not math.isfinite(float(value)) for value in values):
            continue
        usable.append(row)
        features.append([float(value) for value in values])
        labels.append(float(int(int(row.get("accepted_len", 0) or 0) > 0)))
    if not features:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,)), usable
    return np.asarray(features, dtype=float), np.asarray(labels, dtype=float), usable


class StandardizedLogistic:
    """Dependency-light ridge logistic regression used for reproducibility."""

    def __init__(self, *, l2: float = 1.0, max_iter: int = 120) -> None:
        if l2 <= 0.0 or max_iter <= 0:
            raise ValueError("l2 and max_iter must be positive")
        self.l2 = float(l2)
        self.max_iter = int(max_iter)
        self.mean_: Any = None
        self.scale_: Any = None
        self.weights_: Any = None

    def fit(self, features: Any, labels: Any) -> "StandardizedLogistic":
        import numpy as np

        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=float)
        if x.ndim != 2 or len(x) == 0 or len(np.unique(y)) < 2:
            raise ValueError("fit requires non-empty features and both label classes")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        z = (x - self.mean_) / self.scale_
        design = np.column_stack((np.ones(len(z)), z))
        weights = np.zeros(design.shape[1], dtype=float)
        penalty = np.diag([0.0] + [self.l2] * (design.shape[1] - 1))
        for _ in range(self.max_iter):
            logits = np.clip(design @ weights, -35.0, 35.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            curvature = np.maximum(probabilities * (1.0 - probabilities), 1e-8)
            hessian = design.T @ (curvature[:, None] * design) + penalty
            gradient = design.T @ (y - probabilities) - penalty @ weights
            try:
                step = np.linalg.solve(hessian, gradient)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(hessian) @ gradient
            weights += step
            if float(np.linalg.norm(step)) < 1e-9:
                break
        self.weights_ = weights
        return self

    def predict_proba(self, features: Any) -> Any:
        import numpy as np

        if self.weights_ is None:
            raise RuntimeError("model is not fitted")
        x = np.asarray(features, dtype=float)
        z = (x - self.mean_) / self.scale_
        logits = np.clip(np.column_stack((np.ones(len(z)), z)) @ self.weights_, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-logits))


def _classification_metrics(labels: Sequence[float], probabilities: Sequence[float]) -> dict[str, Any]:
    """Compute finite test metrics, returning INCONCLUSIVE for one-class test sets."""

    import numpy as np
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0 - 1e-7)
    result: dict[str, Any] = {
        "count": int(len(y)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "status": "ok" if len(np.unique(y)) == 2 else "INCONCLUSIVE",
        "auroc": None,
        "auprc": None,
        "log_loss": None,
        "brier": None,
        "ece_10_bin": None,
    }
    if not len(y):
        result["status"] = "UNAVAILABLE"
        return result
    result["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    result["brier"] = float(np.mean((p - y) ** 2))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (p >= left) & ((p < right) if right < 1.0 else (p <= right))
        if mask.any():
            ece += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    result["ece_10_bin"] = float(ece)
    if result["status"] == "ok":
        result["auroc"] = float(roc_auc_score(y, p))
        result["auprc"] = float(average_precision_score(y, p))
    return result


def _policy_utility(rows: Sequence[Mapping[str, Any]], selected_k: Sequence[int]) -> dict[str, Any]:
    committed: list[float] = []
    costs: list[float] = []
    for row, k in zip(rows, selected_k):
        cost = _timing_cost(row, int(k))
        if cost is None:
            continue
        committed.append(_committed(row, int(k)))
        costs.append(float(cost))
    result = {
        "status": "ok" if committed and len(committed) == len(rows) else "UNAVAILABLE",
        "count": len(committed),
        "mean_committed_tokens": statistics.fmean(committed) if committed else None,
        "mean_cost_ms": statistics.fmean(costs) if costs else None,
        "tokens_per_ms": sum(committed) / sum(costs) if costs and sum(costs) > 0.0 else None,
        "selected_k_counts": dict(sorted(Counter(int(k) for k in selected_k).items())),
    }
    return result


def _best_fixed(rows: Sequence[Mapping[str, Any]], candidates: Sequence[int]) -> tuple[int | None, dict[str, Any]]:
    scored = [(int(k), _policy_utility(rows, [int(k)] * len(rows))) for k in candidates if int(k) > 0]
    scored = [(k, value) for k, value in scored if value.get("tokens_per_ms") is not None]
    if not scored:
        return None, {"status": "UNAVAILABLE"}
    return max(scored, key=lambda item: float(item[1]["tokens_per_ms"]))


def _admission_oracle(rows: Sequence[Mapping[str, Any]], candidate_k: int) -> dict[str, Any]:
    selected = [candidate_k if int(int(row.get("accepted_len", 0) or 0) > 0) else 0 for row in rows]
    result = _policy_utility(rows, selected)
    result["candidate_k"] = int(candidate_k)
    return result


def _threshold_grid() -> list[float]:
    return [i / 100.0 for i in range(0, 101, 2)]


def _select_threshold(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
    candidate_k: int,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for threshold in _threshold_grid():
        selected = [candidate_k if float(probability) >= threshold else 0 for probability in probabilities]
        utility = _policy_utility(rows, selected)
        scored.append({"threshold": threshold, "utility": utility.get("tokens_per_ms")})
    valid = [item for item in scored if item["utility"] is not None]
    selected = max(valid, key=lambda item: float(item["utility"])) if valid else None
    return {
        "selected": selected["threshold"] if selected else None,
        "selected_utility": selected["utility"] if selected else None,
        "grid": scored,
    }


def analyze_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_ks: Sequence[int] = (2, 4, 8, 16),
    l2: float = 1.0,
) -> dict[str, Any]:
    """Fit and evaluate one dataset using document-disjoint partitions."""

    finite_rows = [
        row for row in _ok(rows)
        if all(_feature_value(row, name) is not None for name in FEATURE_NAMES)
        and all(_timing_cost(row, int(k)) is not None for k in (0, *tuple(int(k) for k in candidate_ks)))
    ]
    train_rows, dev_rows, test_rows, split = _doc_split(finite_rows)
    train_x, train_y, train_used = feature_matrix(train_rows)
    dev_x, dev_y, dev_used = feature_matrix(dev_rows)
    test_x, test_y, test_used = feature_matrix(test_rows)
    base: dict[str, Any] = {
        "status": "ok",
        "feature_names": list(FEATURE_NAMES),
        "features_are_causal": True,
        "excluded_features": ["grounding", "future_attention", "accepted_len", "first_reject_rel"],
        "coverage": {
            "input_ok_rows": len(_ok(rows)),
            "usable_rows": len(finite_rows),
            "documents": len({str(row.get("document_id")) for row in finite_rows}),
            "train_rows": len(train_used),
            "dev_rows": len(dev_used),
            "test_rows": len(test_used),
            "train_documents": len(split["train"]),
            "dev_documents": len(split["dev"]),
            "test_documents": len(split["test"]),
        },
        "split_documents": split,
        "model": None,
        "classification_test": None,
        "candidate_evaluations": {},
    }
    if len(train_used) == 0 or len(dev_used) == 0 or len(test_used) == 0:
        base["status"] = "INCONCLUSIVE"
        base["reason"] = "empty document-disjoint partition"
        return base
    if len(set(train_y.tolist())) < 2:
        base["status"] = "INCONCLUSIVE"
        base["reason"] = "training partition has one admission class"
        return base

    model = StandardizedLogistic(l2=l2).fit(train_x, train_y)
    train_p = model.predict_proba(train_x)
    dev_p = model.predict_proba(dev_x)
    test_p = model.predict_proba(test_x)
    base["model"] = {
        "type": "standardized_ridge_logistic",
        "l2": l2,
        "intercept": float(model.weights_[0]),
        "standardized_coefficients": {
            name: float(value) for name, value in zip(FEATURE_NAMES, model.weights_[1:])
        },
        "train_metrics": _classification_metrics(train_y, train_p),
        "dev_metrics": _classification_metrics(dev_y, dev_p),
    }
    base["classification_test"] = _classification_metrics(test_y, test_p)

    train_k, train_fixed = _best_fixed(train_rows, candidate_ks)
    calibration_rows = train_rows + dev_rows
    calibration_k, calibration_fixed = _best_fixed(calibration_rows, candidate_ks)
    if calibration_k is None:
        base["status"] = "UNAVAILABLE"
        base["reason"] = "no candidate k has complete measured timing"
        return base
    threshold = _select_threshold(dev_rows, dev_p, calibration_k)
    selected_threshold = threshold["selected"]
    if selected_threshold is None:
        base["status"] = "UNAVAILABLE"
        base["reason"] = "no valid threshold on dev rows"
        return base
    predicted_selected = [calibration_k if float(value) >= selected_threshold else 0 for value in test_p]
    predicted = _policy_utility(test_rows, predicted_selected)
    fixed_test = _policy_utility(test_rows, [calibration_k] * len(test_rows))
    admission_test = _admission_oracle(test_rows, calibration_k)
    # A hindsight oracle over the same test rows, used only as a descriptive
    # ceiling. It is not used for model fitting or threshold selection.
    oracle_selected: list[int] = []
    for row in test_rows:
        scores = [(int(k), _committed(row, int(k)) / float(_timing_cost(row, int(k))))
                  for k in (0, *tuple(int(k) for k in candidate_ks))
                  if _timing_cost(row, int(k)) is not None]
        oracle_selected.append(max(scores, key=lambda item: (item[1], -item[0]))[0] if scores else 0)
    oracle_test = _policy_utility(test_rows, oracle_selected)
    baseline_speed = fixed_test.get("tokens_per_ms")
    entry_speed = admission_test.get("tokens_per_ms")
    predicted_speed = predicted.get("tokens_per_ms")
    recovery = None
    if baseline_speed is not None and entry_speed is not None and predicted_speed is not None and entry_speed != baseline_speed:
        recovery = (predicted_speed - baseline_speed) / (entry_speed - baseline_speed)
    base["policy_selection"] = {
        "candidate_ks": [int(k) for k in candidate_ks],
        "train_best_fixed_k": train_k,
        "calibration_best_fixed_k": calibration_k,
        "train_best_fixed": train_fixed,
        "calibration_best_fixed": calibration_fixed,
        "threshold_selection_on_dev": threshold,
        "selected_threshold": selected_threshold,
    }
    base["test_policies"] = {
        "predicted_admission": predicted,
        "fixed_selected_k": fixed_test,
        "first_token_admission_oracle": admission_test,
        "true_cost_oracle_descriptive": oracle_test,
        "oracle_selected_k_counts": dict(sorted(Counter(oracle_selected).items())),
    }
    base["utility"] = {
        "predicted_tokens_per_ms": predicted_speed,
        "fixed_tokens_per_ms": baseline_speed,
        "entry_oracle_tokens_per_ms": entry_speed,
        "true_cost_oracle_tokens_per_ms": oracle_test.get("tokens_per_ms"),
        "recovery_of_entry_oracle_gap": recovery,
        "decision_gate": "at least 10 test documents, recovery >= 0.50, and predicted utility > selected fixed",
        "decision": (
            "PASS" if len(split["test"]) >= 10 and len(set(test_y.tolist())) == 2
            and recovery is not None and recovery >= 0.50 and predicted_speed > baseline_speed
            else "FAIL" if len(split["test"]) >= 10 and len(set(test_y.tolist())) == 2
            and recovery is not None else "INCONCLUSIVE"
        ),
    }
    return base


def _csv_rows(dataset_results: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, result in dataset_results.items():
        coverage = result.get("coverage", {})
        classification = result.get("classification_test") or {}
        utility = result.get("utility", {})
        selection = result.get("policy_selection", {})
        rows.append({
            "dataset": dataset,
            "status": result.get("status"),
            "train_documents": coverage.get("train_documents"),
            "dev_documents": coverage.get("dev_documents"),
            "test_documents": coverage.get("test_documents"),
            "test_rows": coverage.get("test_rows"),
            "test_auroc": classification.get("auroc"),
            "test_auprc": classification.get("auprc"),
            "test_log_loss": classification.get("log_loss"),
            "selected_k": selection.get("calibration_best_fixed_k"),
            "threshold": selection.get("selected_threshold"),
            "predicted_tokens_per_ms": utility.get("predicted_tokens_per_ms"),
            "fixed_tokens_per_ms": utility.get("fixed_tokens_per_ms"),
            "entry_oracle_tokens_per_ms": utility.get("entry_oracle_tokens_per_ms"),
            "recovery": utility.get("recovery_of_entry_oracle_gap"),
            "decision": utility.get("decision"),
        })
    return rows


def render_report(result: Mapping[str, Any]) -> str:
    lines = [
        "# P1 — Cheap online admission predictor",
        "",
        "Đây là đánh giá model nhỏ dự đoán duy nhất bit admission `accepted_len > 0`.",
        "Mọi split đều theo document; ngưỡng và `k` được chọn trước test. Không dùng",
        "grounding/future attention/accepted length khi suy luận.",
        "",
        "## Kết quả",
        "",
        "| Dataset | Docs train/dev/test | Test rows | AUROC | AUPRC | k | threshold | Predictor tok/ms | Fixed tok/ms | Entry-oracle tok/ms | Recovery | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in _csv_rows(result.get("datasets", {})):
        coverage = result["datasets"][row["dataset"]].get("coverage", {})
        docs = f"{coverage.get('train_documents', 0)}/{coverage.get('dev_documents', 0)}/{coverage.get('test_documents', 0)}"
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.4f}" if isinstance(value, (float, int)) else str(value)
        lines.append(
            f"| {row['dataset']} | {docs} | {row['test_rows']} | {fmt(row['test_auroc'])} | "
            f"{fmt(row['test_auprc'])} | {row['selected_k']} | {fmt(row['threshold'])} | "
            f"{fmt(row['predicted_tokens_per_ms'])} | {fmt(row['fixed_tokens_per_ms'])} | "
            f"{fmt(row['entry_oracle_tokens_per_ms'])} | {fmt(row['recovery'])} | {row['decision']} |"
        )
    cross = result.get("cross_regime", {})
    lines.extend([
        "",
        "## Diễn giải và gate",
        "",
        f"- Quyết định cross-regime: **{cross.get('decision', 'UNAVAILABLE')}**.",
        "- Gate deploy: recovery ít nhất 50% gap của first-token admission oracle và utility predictor vượt fixed đã chọn.",
        "- `tokens/ms` là measured block timing của controlled trace, không phải throughput server E2E.",
        "- Với test nhỏ hoặc test chỉ có một class, AUROC được ghi `INCONCLUSIVE`, không thay bằng số giả.",
        "",
        "## Hạn chế",
        "",
        "- P1 chỉ có timing subset 55 GovReport và 50 CNN/DailyMail; test document count còn nhỏ.",
        "- Các tín hiệu top1-top2 logit gap/sentence boundary không có trong trace hiện tại nên không được tái tạo giả; model dùng bốn tín hiệu thực sự có trong schema.",
        "- Kết quả chỉ chứng minh khả năng của predictor trên measured controlled timing, chưa chứng minh serving production.",
    ])
    return "\n".join(lines) + "\n"


def _load_paths(paths: Path | Sequence[Path]) -> list[dict[str, Any]]:
    if isinstance(paths, Path):
        paths = [paths]
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(load_jsonl(path))
    return rows


def analyze_files(
    input_paths: Mapping[str, Path | Sequence[Path]],
    output_dir: Path,
    *,
    candidate_ks: Sequence[int] = (2, 4, 8, 16),
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {name: analyze_dataset(_load_paths(path), candidate_ks=candidate_ks) for name, path in input_paths.items()}
    decisions = [str(value.get("utility", {}).get("decision", "UNAVAILABLE")) for value in datasets.values()]
    if decisions and all(value == "PASS" for value in decisions):
        cross_decision = "PASS"
    elif any(value == "PASS" for value in decisions):
        cross_decision = "MIXED"
    elif decisions and all(value == "FAIL" for value in decisions):
        cross_decision = "FAIL"
    else:
        cross_decision = "INCONCLUSIVE"
    result: dict[str, Any] = {
        "schema_version": "groundsync.p1.predictor.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
        },
        "protocol": {
            "candidate_ks": list(candidate_ks),
            "split": "sorted document IDs, 60/20/20, fit train, threshold dev, report test",
            "features": list(FEATURE_NAMES),
        },
        "datasets": datasets,
        "cross_regime": {"decision": cross_decision, "dataset_decisions": dict(zip(datasets, decisions))},
    }
    (output_dir / "p1_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = _csv_rows(datasets)
    if rows:
        with (output_dir / "p1_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "p1_report.md").write_text(render_report(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gov-timing", type=Path, action="append", required=True,
                        help="one or more JSONL timing files; repeated files are concatenated")
    parser.add_argument("--cnn-timing", type=Path, required=True)
    parser.add_argument("--multinews-timing", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-ks", type=int, nargs="+", default=[2, 4, 8, 16])
    args = parser.parse_args()
    input_paths: dict[str, Path | Sequence[Path]] = {
        "govreport": args.gov_timing,
        "cnn_dailymail": args.cnn_timing,
    }
    if args.multinews_timing is not None:
        input_paths["multinews"] = args.multinews_timing
    result = analyze_files(
        input_paths,
        args.output_dir,
        candidate_ks=args.candidate_ks,
    )
    print(json.dumps(result["cross_regime"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
