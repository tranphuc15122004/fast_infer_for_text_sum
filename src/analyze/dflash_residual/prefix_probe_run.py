"""CLI for E13 tiny probe recoverability experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import read_trace_jsonl
from .prefix_probe import OBJECTIVES, run_probe_suite


def _pct(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _report(result: dict) -> str:
    lines = [
        "# E13 — Tiny probe recoverability",
        "",
        "The probes are trained on frozen DFlash candidate lattices. Target candidate logits are supervision only; they are not model features at test time.",
        "",
        "## Test-set results",
        "",
        "| Regime | Objective | Test documents | MAT_D | MAT_probe | MAT_O16 | Recovery |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for regime, regime_result in result.get("regimes", {}).items():
        for objective, metrics in regime_result.get("objectives", {}).items():
            test = metrics.get("test", {})
            lines.append(
                f"| {regime} | {objective} | {regime_result.get('test_documents')} "
                f"| {test.get('mat_d')} | {test.get('mat_probe')} | {test.get('mat_o16')} "
                f"| {_pct(test.get('oracle_recovery'))} |"
            )
    if result.get("pooled_summarization"):
        pooled = result["pooled_summarization"]
        lines.extend(["", "## Pooled summarization", ""])
        for objective, metrics in pooled.get("objectives", {}).items():
            test = metrics.get("test", {})
            lines.append(
                f"- `{objective}`: MAT_D={test.get('mat_d')}, MAT_probe={test.get('mat_probe')}, "
                f"MAT_O16={test.get('mat_o16')}, recovery={_pct(test.get('oracle_recovery'))}."
            )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "Recovery is `(MAT_probe − MAT_D) / (MAT_O16 − MAT_D)`. It is a bounded diagnostic, not a claim that a trained selector is a deployable method. The document-disjoint test split is the primary result.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--objective", action="append", choices=OBJECTIVES)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-prefix", type=int, default=16)
    args = parser.parse_args()
    rows = []
    for trace in args.trace:
        rows.extend(read_trace_jsonl(trace))
    result = run_probe_suite(
        rows,
        objectives=tuple(args.objective or OBJECTIVES),
        test_fraction=args.test_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        max_prefix=args.max_prefix,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    (output / "run_manifest.json").write_text(json.dumps({
        "experiment": "E13",
        "traces": args.trace,
        "rows_read": len(rows),
        "objectives": list(args.objective or OBJECTIVES),
        "test_fraction": args.test_fraction,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": args.device,
        "status": result.get("status"),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"E13: {result.get('status')}")


if __name__ == "__main__":
    main()
