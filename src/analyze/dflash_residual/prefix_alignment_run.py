"""CLI runner for E11 target–draft alignment and E12 prefix utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_trace_jsonl, write_metrics_bundle
from .prefix_alignment import analyze_alignment, analyze_alignment_utility


def _report(e11: dict, e12: dict) -> str:
    lines = [
        "# Prefix utility alignment — E11/E12",
        "",
        f"**E11:** `{str(e11.get('status', 'unavailable')).upper()}`; "
        f"rows with target logits: `{e11.get('rows_with_target_logits', 0)}`.",
        f"**E12:** `{str(e12.get('status', 'unavailable')).upper()}`.",
        "",
        "All statistics are computed on the recorded Top-16 candidate lattice. "
        "Target logits are analysis labels and are not test-time selector features.",
        "",
        "## E11 summary",
        "",
        "| Dataset | Rows | Documents | Kendall tau | Spearman rho | Inversion rate | JS divergence | Target in lattice |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, values in e11.get("datasets", {}).items():
        lines.append(
            f"| {dataset} | {values.get('rows')} | {values.get('documents')} "
            f"| {values.get('mean_kendall_tau')} | {values.get('mean_spearman_rho')} "
            f"| {values.get('mean_pairwise_inversion_rate')} | {values.get('mean_js_divergence')} "
            f"| {values.get('target_in_lattice_rate')} |"
        )
    bootstrap = e11.get("document_bootstrap", {})
    if bootstrap.get("regimes"):
        lines.extend([
            "",
            "## E11 document-bootstrap differences (summary − canonical)",
            "",
            "| Dataset | Kendall Δ (95% CI) | Spearman Δ (95% CI) | Inversion Δ (95% CI) | JS Δ (95% CI) |",
            "|---|---:|---:|---:|---:|",
        ])
        for dataset, values in bootstrap["regimes"].items():
            metrics = values["metrics"]
            formatted = []
            for metric in ("kendall_tau", "spearman_rho", "pairwise_inversion_rate", "js_divergence"):
                item = metrics[metric]
                formatted.append(
                    f"{item['mean_difference_summary_minus_canonical']:.4f} "
                    f"[{item['ci95'][0]:.4f}, {item['ci95'][1]:.4f}]"
                )
            lines.append(f"| {dataset} | " + " | ".join(formatted) + " |")
    lines.extend([
        "",
        "## E12 summary",
        "",
        "| Dataset | Blocks | MAT_D | MAT_O16 | Oracle gap | Align→MAT_D Spearman | Align→gap Spearman |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, values in e12.get("datasets", {}).items():
        lines.append(
            f"| {dataset} | {values.get('blocks')} | {values.get('mat_d')} | {values.get('mat_o16')} "
            f"| {values.get('oracle_gap')} | {values.get('alignment_vs_mat_d', {}).get('spearman_rho')} "
            f"| {values.get('alignment_vs_oracle_gap', {}).get('spearman_rho')} |"
        )
    lines.extend([
        "",
        "## Caveat",
        "",
        "E11/E12 do not establish causality. They measure association between target–draft "
        "ordering and observed prefix utility on frozen states; E13 is required before an "
        "objective or method proposal is opened.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-prefix", type=int, default=16)
    args = parser.parse_args()
    rows = []
    for trace in args.trace:
        rows.extend(read_trace_jsonl(trace))
    e11 = analyze_alignment(rows)
    e12 = analyze_alignment_utility(rows, max_prefix=args.max_prefix)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_metrics_bundle(output / "e11", e11, report=_report(e11, e12))
    write_metrics_bundle(output / "e12", e12, report=_report(e11, e12))
    (output / "run_manifest.json").write_text(json.dumps({
        "experiment": "E11_E12",
        "traces": args.trace,
        "rows_read": len(rows),
        "max_prefix": args.max_prefix,
        "status": "ok" if e11.get("status") == "ok" and e12.get("status") == "ok" else "unavailable",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"E11: {e11.get('status')}; E12: {e12.get('status')}")


if __name__ == "__main__":
    main()
