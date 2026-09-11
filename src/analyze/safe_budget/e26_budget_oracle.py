"""E26 dynamic-budget oracle analysis for semantic-selection outputs.

This module is deliberately offline.  It consumes JSONL records emitted by the
existing semantic-selection runner and never treats missing budget conditions
as successful evidence.  The full E26 protocol needs a six-point budget grid;
when only a subset is present the generated report is marked as a pilot and
the gate is not evaluated as a complete study.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_EPSILONS = (0.0, 0.01, 0.02, 0.05)
DEFAULT_REQUIRED_LEVELS = 6
DEFAULT_MIN_DOCS = 30


def _number(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _budget_tokens(row: Mapping[str, Any]) -> float | None:
    label = str(row.get("budget_label", ""))
    if label == "full":
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


def _quality(row: Mapping[str, Any], metric: str) -> float | None:
    return _number(row, metric)


def _cost(row: Mapping[str, Any], metric: str) -> float | None:
    return _number(row, metric)


def _mean(values: Iterable[float]) -> float | None:
    data = list(values)
    return sum(data) / len(data) if data else None


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


def _select_one(rows: Sequence[Mapping[str, Any]], *, label: str) -> Mapping[str, Any] | None:
    matching = [row for row in rows if _condition_key(row) == label]
    if not matching:
        return None
    # Duplicate records are not silently averaged.  The runner should emit
    # one record per document/selector/budget; use the first only after the
    # caller has made duplicates explicit in the diagnostics.
    return matching[0]


def conservative_safe_budget(
    conditions: Sequence[Mapping[str, Any]],
    *,
    full_quality: float,
    epsilon: float,
    quality_metric: str,
) -> Mapping[str, Any] | None:
    """Return the smallest observed condition safe under conservative closure.

    Every observed condition at an equal-or-larger requested budget must meet
    the quality threshold.  The caller must include the full-context record in
    ``conditions`` so the closure has a verified upper endpoint.
    """

    threshold = full_quality - epsilon
    usable = [
        row
        for row in conditions
        if _budget_tokens(row) is not None and _quality(row, quality_metric) is not None
    ]
    usable.sort(key=lambda row: (_budget_tokens(row) or float("inf"), _condition_key(row)))
    for candidate in usable:
        candidate_budget = _budget_tokens(candidate)
        assert candidate_budget is not None
        higher = [
            row
            for row in usable
            if (_budget_tokens(row) or float("inf")) >= candidate_budget
        ]
        if higher and all((_quality(row, quality_metric) or -float("inf")) >= threshold for row in higher):
            return candidate
    return None


def _fixed_candidate_summary(
    rows_by_doc: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    label: str,
    full_quality_by_doc: Mapping[str, float],
    quality_metric: str,
    cost_metric: str,
    epsilon: float,
) -> dict[str, Any] | None:
    selected: list[Mapping[str, Any]] = []
    for doc_id, rows in rows_by_doc.items():
        row = _select_one(rows, label=label)
        if row is None or _quality(row, quality_metric) is None or _cost(row, cost_metric) is None:
            return None
        selected.append(row)
    if not selected:
        return None
    qualities = [_quality(row, quality_metric) for row in selected]
    costs = [_cost(row, cost_metric) for row in selected]
    assert all(value is not None for value in qualities)
    assert all(value is not None for value in costs)
    full_mean = _mean(full_quality_by_doc.values())
    selected_mean = _mean(value for value in qualities if value is not None)
    threshold = (full_mean or 0.0) - epsilon
    return {
        "label": label,
        "documents": len(selected),
        "mean_quality": selected_mean,
        "mean_quality_delta_vs_full": None if selected_mean is None or full_mean is None else selected_mean - full_mean,
        "quality_threshold": threshold,
        "quality_pass": selected_mean is not None and selected_mean >= threshold,
        "mean_cost_ms": _mean(value for value in costs if value is not None),
        "mean_budget_tokens": _mean(value for value in (_budget_tokens(row) for row in selected) if value is not None),
    }


def analyze_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    selector: str,
    quality_metric: str = "rougeL",
    cost_metric: str = "pipeline_e2e_ms",
    epsilons: Sequence[float] = DEFAULT_EPSILONS,
    required_levels: int = DEFAULT_REQUIRED_LEVELS,
    min_documents: int = DEFAULT_MIN_DOCS,
) -> dict[str, Any]:
    """Analyze one JSONL file and return a serializable result."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    duplicate_keys: list[tuple[str, str]] = []
    for row in rows:
        doc_id = str(row.get("example_id", row.get("id", "")))
        if not doc_id:
            raise ValueError("every row must contain example_id or id")
        if str(row.get("selector", "")) == selector:
            key = (doc_id, _condition_key(row))
            if any(_condition_key(old) == key[1] for old in grouped[doc_id]):
                duplicate_keys.append(key)
            grouped[doc_id].append(row)
        elif str(row.get("selector", "")) == "full" and str(row.get("budget_label", "")) == "full":
            grouped[doc_id].append(row)

    full_quality: dict[str, float] = {}
    full_cost: dict[str, float] = {}
    usable_docs: dict[str, list[Mapping[str, Any]]] = {}
    missing_reasons: dict[str, str] = {}
    for doc_id, doc_rows in grouped.items():
        full = next(
            (row for row in doc_rows if str(row.get("selector")) == "full" and _condition_key(row) == "full"),
            None,
        )
        if full is None:
            missing_reasons[doc_id] = "missing full baseline"
            continue
        q_full = _quality(full, quality_metric)
        c_full = _cost(full, cost_metric)
        if q_full is None or c_full is None:
            missing_reasons[doc_id] = "full baseline missing quality or cost"
            continue
        selected_rows = [row for row in doc_rows if str(row.get("selector")) == selector]
        if not selected_rows:
            missing_reasons[doc_id] = f"missing selector={selector} condition"
            continue
        full_quality[doc_id] = q_full
        full_cost[doc_id] = c_full
        usable_docs[doc_id] = [*selected_rows, full]

    observed_labels = sorted(
        {
            _condition_key(row)
            for rows_for_doc in usable_docs.values()
            for row in rows_for_doc
            if _budget_tokens(row) is not None
        },
        key=lambda label: (
            min(
                (_budget_tokens(row) for rows_for_doc in usable_docs.values() for row in rows_for_doc if _condition_key(row) == label and _budget_tokens(row) is not None),
                default=float("inf"),
            ),
            label,
        ),
    )

    epsilon_results: dict[str, Any] = {}
    for epsilon in epsilons:
        fixed = []
        for label in observed_labels:
            summary = _fixed_candidate_summary(
                usable_docs,
                label=label,
                full_quality_by_doc=full_quality,
                quality_metric=quality_metric,
                cost_metric=cost_metric,
                epsilon=float(epsilon),
            )
            if summary is not None and summary["quality_pass"]:
                fixed.append(summary)
        fixed.sort(key=lambda item: item["mean_budget_tokens"])
        best_fixed = fixed[0] if fixed else None

        oracle_rows: list[Mapping[str, Any]] = []
        unsafe_docs: list[str] = []
        for doc_id, conditions in usable_docs.items():
            safe = conservative_safe_budget(
                conditions,
                full_quality=full_quality[doc_id],
                epsilon=float(epsilon),
                quality_metric=quality_metric,
            )
            if safe is None:
                unsafe_docs.append(doc_id)
                continue
            oracle_rows.append(safe)

        oracle_quality = [_quality(row, quality_metric) for row in oracle_rows]
        oracle_cost = [_cost(row, cost_metric) for row in oracle_rows]
        oracle_budget = [_budget_tokens(row) for row in oracle_rows]
        full_mean_quality = _mean(full_quality.values())
        full_mean_cost = _mean(full_cost.values())
        oracle_mean_quality = _mean(value for value in oracle_quality if value is not None)
        oracle_mean_cost = _mean(value for value in oracle_cost if value is not None)
        oracle_mean_budget = _mean(value for value in oracle_budget if value is not None)
        headroom = None
        speedup = None
        if best_fixed is not None and best_fixed["mean_cost_ms"] and oracle_mean_cost is not None:
            headroom = 1.0 - oracle_mean_cost / float(best_fixed["mean_cost_ms"])
        if full_mean_cost and oracle_mean_cost:
            speedup = full_mean_cost / oracle_mean_cost
        epsilon_results[f"{float(epsilon):.4f}"] = {
            "epsilon": float(epsilon),
            "full_mean_quality": full_mean_quality,
            "full_mean_cost_ms": full_mean_cost,
            "best_fixed": best_fixed,
            "oracle_adaptive": {
                "documents_assigned": len(oracle_rows),
                "documents_without_safe_observed_budget": unsafe_docs,
                "mean_quality": oracle_mean_quality,
                "mean_quality_delta_vs_full": None if oracle_mean_quality is None or full_mean_quality is None else oracle_mean_quality - full_mean_quality,
                "mean_cost_ms": oracle_mean_cost,
                "mean_budget_tokens": oracle_mean_budget,
                "mean_budget_ratio": _mean(
                    budget / _number(row, "original_tokens")
                    for row, budget in zip(oracle_rows, oracle_budget)
                    if budget is not None and _number(row, "original_tokens") not in (None, 0)
                ),
                "headroom_vs_best_fixed": headroom,
                "speedup_vs_full": speedup,
            },
        }

    docs = len(usable_docs)
    complete = docs >= min_documents and len(observed_labels) >= required_levels
    return {
        "selector": selector,
        "quality_metric": quality_metric,
        "cost_metric": cost_metric,
        "documents_input": len(grouped),
        "documents_usable": docs,
        "documents_missing_or_invalid": missing_reasons,
        "duplicate_document_condition_keys": [list(item) for item in duplicate_keys],
        "observed_budget_levels": observed_labels,
        "required_budget_levels": required_levels,
        "minimum_documents": min_documents,
        "completeness": {
            "complete_for_full_e26": complete,
            "document_requirement_met": docs >= min_documents,
            "budget_requirement_met": len(observed_labels) >= required_levels,
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
        "# E26 Dynamic-Budget Oracle Report",
        "",
        "## Trạng thái",
        "",
        "Báo cáo này là phân tích offline từ JSONL output của semantic-selection.",
        "Nó không chạy lại target model. E26 full chỉ được coi là hoàn chỉnh khi",
        f"có ít nhất {result['minimum_documents']} documents/dataset và {result['required_budget_levels']} budget levels.",
        "",
        "| Dataset | Documents dùng được | Budget levels quan sát | Trạng thái |",
        "|---|---:|---|---|",
    ]
    for dataset, data in result["datasets"].items():
        lines.append(
            f"| {dataset} | {data['documents_usable']} | {', '.join(data['observed_budget_levels']) or 'none'} | {data['completeness']['status']} |"
        )
    lines += [
        "",
        "## Protocol và dữ liệu",
        "",
        f"- Selector: `{result['selector']}`",
        f"- Quality metric: `{result['quality_metric']}`",
        f"- Cost metric: `{result['cost_metric']}`",
        "- Default epsilon sensitivity: 0.00, 0.01, 0.02, 0.05 ROUGE-L",
        "- Safe budget dùng conservative closure trên mọi budget quan sát được ở mức lớn hơn hoặc bằng budget ứng viên.",
        "- Best fixed là budget nhỏ nhất có mean quality không thấp hơn mean full-context quality trừ epsilon.",
        "",
        "### Input artifacts",
        "",
    ]
    for dataset, path in source_paths.items():
        lines.append(f"- `{dataset}`: `{path}`")
    lines += ["", "## Kết quả theo dataset", ""]
    for dataset, data in result["datasets"].items():
        lines += [f"### {dataset}", ""]
        lines.append(
            f"Documents usable: **{data['documents_usable']}**; full E26 complete: **{data['completeness']['complete_for_full_e26']}**."
        )
        if data["documents_missing_or_invalid"]:
            lines.append(f"Rows/documents bị loại: `{json.dumps(data['documents_missing_or_invalid'], ensure_ascii=False)}`")
        lines += ["", "| Epsilon | Best fixed | Fixed cost ms | Oracle cost ms | Oracle headroom | Oracle speedup vs full | Oracle quality delta |", "|---:|---|---:|---:|---:|---:|---:|"]
        for key, item in data["epsilon_results"].items():
            fixed = item["best_fixed"] or {}
            oracle = item["oracle_adaptive"]
            lines.append(
                f"| {item['epsilon']:.2f} | {fixed.get('label', 'none')} | {_fmt(fixed.get('mean_cost_ms'), 2)} | {_fmt(oracle.get('mean_cost_ms'), 2)} | {_pct(oracle.get('headroom_vs_best_fixed'))} | {_fmt(oracle.get('speedup_vs_full'), 3)}x | {_fmt(oracle.get('mean_quality_delta_vs_full'), 4)} |"
            )
        lines += ["", "#### Chi tiết budget và quality", ""]
        for key, item in data["epsilon_results"].items():
            fixed = item["best_fixed"] or {}
            oracle = item["oracle_adaptive"]
            lines += [
                f"- `epsilon={item['epsilon']:.2f}`: full mean quality={_fmt(item['full_mean_quality'])}, full mean cost={_fmt(item['full_mean_cost_ms'], 2)} ms.",
                f"  - Best fixed: `{fixed.get('label', 'none')}`, quality={_fmt(fixed.get('mean_quality'))}, quality delta={_fmt(fixed.get('mean_quality_delta_vs_full'))}, cost={_fmt(fixed.get('mean_cost_ms'), 2)} ms.",
                f"  - Oracle adaptive: assigned={oracle['documents_assigned']}, mean budget={_fmt(oracle.get('mean_budget_tokens'), 2)} tokens, ratio={_pct(oracle.get('mean_budget_ratio'))}, quality={_fmt(oracle.get('mean_quality'))}, cost={_fmt(oracle.get('mean_cost_ms'), 2)} ms.",
                f"  - Headroom vs best fixed={_pct(oracle.get('headroom_vs_best_fixed'))}; documents without a safe observed budget={len(oracle['documents_without_safe_observed_budget'])}.",
            ]
        lines.append("")
    lines += [
        "## Kết luận và gate",
        "",
        "E26 chỉ được mở gate nếu mọi dataset đủ budget grid và cỡ mẫu tối thiểu.",
        "Nếu output là `pilot_incomplete`, các headroom numbers chỉ là mô tả pilot",
        "và không được dùng để claim E26 full pass.",
        "",
    ]
    for dataset, data in result["datasets"].items():
        item = data["epsilon_results"].get("0.0200", {})
        oracle = item.get("oracle_adaptive", {})
        lines.append(
            f"- {dataset}: status=`{data['completeness']['status']}`, epsilon 0.02 headroom={_pct(oracle.get('headroom_vs_best_fixed'))}."
        )
    lines += [
        "",
        "## Giới hạn",
        "",
        "- Không có budget 25/40/55/70/85/100% đầy đủ trong input hiện tại.",
        "- Không có calibration predictor E27.",
        "- Không có Domino checkpoint hoặc E2E frontier benchmark E28.",
        "- Mọi kết luận về speedup chỉ dùng cost đã ghi trong input; không suy ra throughput mới.",
    ]
    return "\n".join(lines) + "\n"


def _csv_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, data in result["datasets"].items():
        for item in data["epsilon_results"].values():
            fixed = item["best_fixed"] or {}
            oracle = item["oracle_adaptive"]
            rows.append(
                {
                    "dataset": dataset,
                    "epsilon": item["epsilon"],
                    "status": data["completeness"]["status"],
                    "documents": data["documents_usable"],
                    "observed_budget_levels": ",".join(data["observed_budget_levels"]),
                    "best_fixed_label": fixed.get("label"),
                    "best_fixed_mean_quality": fixed.get("mean_quality"),
                    "best_fixed_mean_cost_ms": fixed.get("mean_cost_ms"),
                    "oracle_documents_assigned": oracle.get("documents_assigned"),
                    "oracle_mean_quality": oracle.get("mean_quality"),
                    "oracle_mean_cost_ms": oracle.get("mean_cost_ms"),
                    "oracle_mean_budget_tokens": oracle.get("mean_budget_tokens"),
                    "oracle_mean_budget_ratio": oracle.get("mean_budget_ratio"),
                    "oracle_headroom_vs_best_fixed": oracle.get("headroom_vs_best_fixed"),
                    "oracle_speedup_vs_full": oracle.get("speedup_vs_full"),
                }
            )
    return rows


def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    inputs = dict(args.input)
    datasets = {
        name: analyze_dataset(
            load_jsonl(path),
            selector=args.selector,
            quality_metric=args.quality_metric,
            cost_metric=args.cost_metric,
            epsilons=args.epsilon,
            required_levels=args.required_budget_levels,
            min_documents=args.min_documents,
        )
        for name, path in inputs.items()
    }
    result: dict[str, Any] = {
        "experiment": "E26_dynamic_budget_oracle",
        "selector": args.selector,
        "quality_metric": args.quality_metric,
        "cost_metric": args.cost_metric,
        "required_budget_levels": args.required_budget_levels,
        "minimum_documents": args.min_documents,
        "inputs": inputs,
        "datasets": datasets,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        rows = _csv_rows(result)
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
                "inputs": inputs,
                "output_dir": str(out),
                "epsilon": args.epsilon,
                "required_budget_levels": args.required_budget_levels,
                "min_documents": args.min_documents,
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
    parser.add_argument("--quality-metric", default="rougeL")
    parser.add_argument("--cost-metric", default="pipeline_e2e_ms")
    parser.add_argument("--epsilon", type=float, action="append", default=None)
    parser.add_argument("--required-budget-levels", type=int, default=DEFAULT_REQUIRED_LEVELS)
    parser.add_argument("--min-documents", type=int, default=DEFAULT_MIN_DOCS)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.epsilon is None:
        args.epsilon = list(DEFAULT_EPSILONS)
    result = run_cli(args)
    print(json.dumps({
        "experiment": result["experiment"],
        "selector": result["selector"],
        "datasets": {
            name: {
                "status": data["completeness"]["status"],
                "documents": data["documents_usable"],
                "budget_levels": data["observed_budget_levels"],
            }
            for name, data in result["datasets"].items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()

