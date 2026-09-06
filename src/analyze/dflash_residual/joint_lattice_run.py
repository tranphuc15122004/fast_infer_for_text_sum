"""CLI for E14 marginal/joint lattice decomposition and E14b entropy control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_trace_jsonl, write_metrics_bundle
from .joint_lattice import (
    analyze_decomposition,
    bootstrap_decomposition,
    entropy_standardized_stats,
    lattice_stats,
)


def _report(e14: dict, e14b: dict | None) -> str:
    lines = [
        "# E14/E14b — Joint candidate-lattice degradation",
        "",
        f"**E14:** `{str(e14.get('status', 'unavailable')).upper()}`; candidate K=`{e14.get('candidate_k')}`.",
        "",
        "The trace has 15 draft positions inside the native 16-token DFlash block; all prefix sums therefore use positions 1–15.",
        "",
        "## E14 marginal/joint decomposition",
        "",
        "| Dataset | MAT_O16 | Marginal counterfactual | Marginal component | Joint component | Marginal fraction | Joint fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    groups = e14.get("groups", {})
    decomposition = e14.get("decomposition", {})
    for dataset, item in groups.items():
        if dataset == "canonical":
            continue
        d = decomposition.get(dataset, {})
        lines.append(
            f"| {dataset} | {item.get('mat_o16')} | {d.get('mat_o16_marginal_counterfactual')} "
            f"| {d.get('marginal_component')} | {d.get('joint_component')} "
            f"| {d.get('marginal_fraction')} | {d.get('joint_fraction')} |"
        )
    bootstrap = e14.get("bootstrap", {})
    if bootstrap.get("status") == "ok" and bootstrap.get("samples", 0) >= 100:
        lines.extend(["", "## E14 bootstrap", "", "| Dataset | Marginal fraction mean (95% CI) | Joint fraction mean (95% CI) |", "|---|---:|---:|"])
        for dataset, item in bootstrap.get("regimes", {}).items():
            m = item.get("components", {}).get("marginal_fraction", {})
            j = item.get("components", {}).get("joint_fraction", {})
            lines.append(f"| {dataset} | {m.get('mean')} {m.get('ci95')} | {j.get('mean')} {j.get('ci95')} |")
    else:
        lines.extend(["", "## E14 uncertainty", "", "Bootstrap CI không được dùng trong kết quả chính: canonical chỉ có 8 documents và bootstrap nhỏ không ổn định cho counterfactual coherence. Các số E14 dưới đây là point estimates; cần mở rộng canonical trước khi báo cáo CI."])
    if e14b:
        lines.extend(["", "## E14b entropy-standardized control", "", "| Dataset | Actual MAT_O16 | Entropy-standardized MAT | Canonical standardized MAT | Standardized gap |", "|---|---:|---:|---:|---:|"])
        for dataset, item in e14b.get("datasets", {}).items():
            lines.append(
                f"| {dataset} | {item.get('actual_mat_o16')} | {item.get('mat_entropy_standardized')} "
                f"| {item.get('mat_reference_entropy_standardized')} | {item.get('entropy_standardized_gap')} |"
            )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The decomposition is descriptive/counterfactual: canonical coherence is applied to summarization marginal recalls. It is not a causal intervention. E14b uses equal-weight shared entropy bins and should be read as a control, not as a replacement for a matched-pairs causal estimate.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-trace", required=True)
    parser.add_argument("--summary-trace", action="append", required=True)
    parser.add_argument("--entropy-trace", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-position", type=int, default=15)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()
    canonical = read_trace_jsonl(args.canonical_trace)
    summary = []
    for path in args.summary_trace:
        summary.extend(read_trace_jsonl(path))
    rows = canonical + summary
    e14 = analyze_decomposition(rows, k=16, max_position=args.max_position)
    if args.bootstrap_samples > 0:
        e14["bootstrap"] = bootstrap_decomposition(rows, samples=args.bootstrap_samples, seed=42, k=16)
    else:
        e14["bootstrap"] = {"status": "skipped", "reason": "bootstrap_samples=0"}
    e14b = None
    if args.entropy_trace:
        entropy_rows = []
        for path in args.entropy_trace:
            entropy_rows.extend(read_trace_jsonl(path))
        entropy_groups = {}
        for row in entropy_rows:
            entropy_groups.setdefault(str(row.get("task_regime", row.get("dataset", "other"))), []).append(row)
        entropy_canonical = entropy_groups.get("canonical", [])
        e14b = {"status": "ok", "experiment": "E14b", "datasets": {}, "traces": args.entropy_trace}
        e14b["canonical"] = entropy_standardized_stats(entropy_canonical, entropy_canonical, max_position=args.max_position)
        e14b["datasets"]["canonical"] = {
            "actual_mat_o16": lattice_stats(entropy_canonical, k=16, max_position=args.max_position).get("mat_o16"),
            **e14b["canonical"],
        }
        for dataset, dataset_rows in sorted(entropy_groups.items()):
            if dataset == "canonical":
                continue
            stats = entropy_standardized_stats(dataset_rows, entropy_canonical, max_position=args.max_position)
            e14b["datasets"][dataset] = {
                "actual_mat_o16": lattice_stats(dataset_rows, k=16, max_position=args.max_position).get("mat_o16"),
                **stats,
            }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_metrics_bundle(output / "e14", e14, report=_report(e14, e14b))
    if e14b is not None:
        write_metrics_bundle(output / "e14b", e14b, report=_report(e14, e14b))
    (output / "run_manifest.json").write_text(json.dumps({
        "experiment": "E14_E14b",
        "canonical_trace": args.canonical_trace,
        "summary_traces": args.summary_trace,
        "entropy_traces": args.entropy_trace,
        "max_position": args.max_position,
        "bootstrap_samples": args.bootstrap_samples,
        "status": e14.get("status"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"E14: {e14.get('status')}; E14b: {e14b.get('status') if e14b else 'not-run'}")


if __name__ == "__main__":
    main()
