"""Analyze E15 task-matched DFlash adaptation on paired held-out traces."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from .io import read_trace_jsonl, write_metrics_bundle
from .joint_lattice import lattice_stats
from .prefix_gap import prefix_oracle_length, _candidate_hit
from .metrics import _blocks, _observed_acceptance, recall_at_k


def _ok(rows: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


def _mat_d(rows: list[Mapping[str, Any]]) -> float:
    blocks = _blocks(rows)
    return sum(_observed_acceptance(block) for block in blocks) / len(blocks)


def _metric(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    usable = _ok(rows)
    stats = lattice_stats(usable, k=16, max_position=15)
    return {
        "rows": len(usable),
        "blocks": stats.get("blocks"),
        "documents": stats.get("documents"),
        "mat_d": _mat_d(usable),
        "recall_at_1": recall_at_k(usable, 1),
        "recall_at_16": recall_at_k(usable, 16),
        "mat_o16": stats.get("mat_o16"),
        "marginal_recall": stats.get("marginal_recall"),
        "joint_survival": stats.get("joint_survival"),
        "independent_survival": stats.get("independent_survival"),
        "coherence": stats.get("coherence"),
    }


def _paired_rows(
    rows: list[Mapping[str, Any]],
    document_ids: set[str],
) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row.get("document_id")) in document_ids]


def _document_summary(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    """Sufficient statistics for document-level paired bootstrap."""
    blocks = _blocks(rows)
    oracle_lengths = [prefix_oracle_length(block, 16) for block in blocks]
    return {
        "rows": float(len(rows)),
        "blocks": float(len(blocks)),
        "mat_d_sum": float(sum(_observed_acceptance(block) for block in blocks)),
        "mat_o16_sum": float(sum(oracle_lengths)),
        "hit1_sum": float(sum(_candidate_hit(row, 1) for row in rows)),
        "hit16_sum": float(sum(_candidate_hit(row, 16) for row in rows)),
    }


def _ci(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]]


def _bootstrap(
    base: list[Mapping[str, Any]],
    adapted: list[Mapping[str, Any]],
    canonical_mat: float,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    base_by_doc: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    adapted_by_doc: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in base:
        base_by_doc[str(row["document_id"])].append(row)
    for row in adapted:
        adapted_by_doc[str(row["document_id"])].append(row)
    documents = sorted(set(base_by_doc) & set(adapted_by_doc))
    if len(documents) < 2:
        return {"status": "inconclusive", "documents": len(documents)}
    base_summary = {key: _document_summary(value) for key, value in base_by_doc.items()}
    adapted_summary = {key: _document_summary(value) for key, value in adapted_by_doc.items()}
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for replicate in range(samples):
        sampled = [rng.choice(documents) for _ in documents]
        base_totals = {key: sum(base_summary[doc][key] for doc in sampled) for key in base_summary[documents[0]]}
        adapted_totals = {key: sum(adapted_summary[doc][key] for doc in sampled) for key in adapted_summary[documents[0]]}
        def from_totals(totals: dict[str, float]) -> dict[str, float]:
            return {
                "mat_d": totals["mat_d_sum"] / totals["blocks"],
                "recall_at_1": totals["hit1_sum"] / totals["rows"],
                "recall_at_16": totals["hit16_sum"] / totals["rows"],
                "mat_o16": totals["mat_o16_sum"] / totals["blocks"],
            }
        base_metric = from_totals(base_totals)
        adapted_metric = from_totals(adapted_totals)
        for name in ("mat_d", "recall_at_16", "mat_o16"):
            values[f"base_{name}"].append(float(base_metric[name]))
            values[f"adapted_{name}"].append(float(adapted_metric[name]))
            values[f"delta_{name}"].append(float(adapted_metric[name]) - float(base_metric[name]))
        adapted_recovery = (float(adapted_metric["mat_o16"]) - float(base_metric["mat_o16"])) / max(canonical_mat - float(base_metric["mat_o16"]), 1e-9)
        values["adapted_oracle_recovery"].append(adapted_recovery)
    return {
        "status": "ok",
        "samples": samples,
        "seed": seed,
        "documents": len(documents),
        "ci95": {name: {"mean": sum(vals) / len(vals), "ci": _ci(vals)} for name, vals in values.items()},
    }


def _report(metrics: dict[str, Any]) -> str:
    base = metrics["baseline"]
    adapted = metrics["adapted"]
    lines = [
        "# E15 — Minimal summarization adaptation",
        "",
        f"E15 giữ nguyên kiến trúc DFlash 5-layer, block size 16, target Qwen3-4B, warm-start checkpoint gốc và loss `dflash`; chỉ thay dữ liệu teacher-forced bằng 50 CNN/DM + 50 GovReport. Đánh giá trên **{metrics.get('evaluation_label', 'paired evaluation set')}**.",
        "",
        "## Paired held-out result",
        "",
        "| Metric | Original DFlash | Adapted DFlash | Delta adapted - original |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (
        ("mat_d", "MAT_D"),
        ("recall_at_1", "Recall@1"),
        ("recall_at_16", "Recall@16"),
        ("mat_o16", "MAT_O16"),
    ):
        lines.append(f"| {label} | {base[name]:.6f} | {adapted[name]:.6f} | {adapted[name] - base[name]:+.6f} |")
    denominator = metrics["canonical_mat_o16"] - base["mat_o16"]
    recovery = (adapted["mat_o16"] - base["mat_o16"]) / denominator if denominator else None
    lines.extend([
        "",
        f"Canonical MAT_O16 reference: `{metrics['canonical_mat_o16']:.6f}`; held-out oracle headroom before adaptation: `{denominator:.6f}`.",
        f"E15 oracle recovery: `{recovery:.6%}` of the canonical-vs-held-out gap." if recovery is not None else "E15 oracle recovery: unavailable.",
        "",
        "## Prefix survival and normalized coherence",
        "",
        "| Position | J16 original | J16 adapted | C16 original | C16 adapted |",
        "|---:|---:|---:|---:|---:|",
    ])
    for position in (1, 2, 4, 8, 15):
        key = str(position)
        lines.append(
            f"| {position} | {base['joint_survival'].get(key)} | {adapted['joint_survival'].get(key)} "
            f"| {base['coherence'].get(key)} | {adapted['coherence'].get(key)} |"
        )
    boot = metrics.get("bootstrap", {})
    if boot.get("status") == "ok":
        lines.extend(["", "## Paired document bootstrap", "", "| Quantity | Mean | 95% CI |", "|---|---:|---:|"])
        for key in ("delta_mat_d", "delta_recall_at_16", "delta_mat_o16", "adapted_oracle_recovery"):
            item = boot["ci95"].get(key)
            lines.append(f"| {key} | {item['mean']:.6f} | {item['ci']} |")
    lines.extend([
        "",
        "## Gate interpretation",
        "",
        "E15 là causal diagnostic cho training-distribution mismatch, không phải proposal method. Chỉ khi `MAT_O16` tăng trên held-out dataset mới có bằng chứng adaptation sửa candidate generation; nếu chỉ MAT_D thay đổi mà MAT_O16 không tăng thì adaptation không sửa lattice coverage.",
        "",
        "## Reproducibility",
        "",
        f"Baseline trace: `{metrics['baseline_trace']}`; adapted trace: `{metrics['adapted_trace']}`; canonical trace: `{metrics['canonical_trace']}`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-trace", required=True)
    parser.add_argument("--adapted-trace", required=True)
    parser.add_argument("--canonical-trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--evaluation-label", default="paired evaluation set")
    args = parser.parse_args()
    baseline_all = _ok(read_trace_jsonl(args.baseline_trace))
    adapted = _ok(read_trace_jsonl(args.adapted_trace))
    adapted_documents = {str(row["document_id"]) for row in adapted}
    baseline = _paired_rows(baseline_all, adapted_documents)
    canonical = _ok(read_trace_jsonl(args.canonical_trace))
    canonical_mat = float(lattice_stats(canonical, k=16, max_position=15)["mat_o16"])
    metrics = {
        "status": "ok",
        "experiment": "E15",
        "baseline_trace": args.baseline_trace,
        "adapted_trace": args.adapted_trace,
        "canonical_trace": args.canonical_trace,
        "canonical_mat_o16": canonical_mat,
        "evaluation_label": args.evaluation_label,
        "baseline": _metric(baseline),
        "adapted": _metric(adapted),
        "bootstrap": _bootstrap(
            baseline,
            adapted,
            canonical_mat,
            samples=args.bootstrap_samples,
            seed=42,
        ) if args.bootstrap_samples > 0 else {"status": "skipped"},
    }
    output = Path(args.output)
    write_metrics_bundle(output, metrics, report=_report(metrics))
    print(f"E15: {metrics['status']}; output={output}")


if __name__ == "__main__":
    main()
