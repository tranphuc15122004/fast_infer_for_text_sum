#!/usr/bin/env python3
"""Create the paper-style Table 1 from an existing LongBench run.

The table is intentionally conservative: failed cells and metrics outside the
recorded measurement scope are rendered as an em dash instead of being
reconstructed from unrelated telemetry.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

from common import metrics


DATASETS = [
    ("gov_report", "GovReport"),
    ("qmsum", "QMSum"),
    ("multi_news", "Multi-News"),
    ("lcc", "LCC"),
    ("repobench-p", "RepoBench-P"),
]

# Keep the paper/table order. LongSpec and SSSD are displayed explicitly even
# though they are not part of the current 7-baseline run.
METHODS = [
    ("Vanilla HF", "vanilla_hf"),
    ("Vanilla FA", "vanilla_fa"),
    ("MagicDec", "magicdec"),
    ("LongSpec", None),
    ("EAGLE3", "eagle3"),
    ("Dflash", "dflash"),
    ("SpecExtend", "specextend"),
    ("SSSD", None),
    ("FAFO", "fafo_stream-llm"),
]

MISSING = "—"
NOT_APPLICABLE = "N/A"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _summary_row(summary: dict[str, Any], dataset: str, method: str) -> dict[str, Any] | None:
    row = summary.get("metrics", {}).get(dataset, {}).get(method)
    return row if isinstance(row, dict) else None


def _valid_count(row: dict[str, Any] | None, expected_samples: int) -> int:
    if not row:
        return 0
    try:
        return int(row.get("num_records", 0))
    except (TypeError, ValueError):
        return 0


def _mean_speed(row: dict[str, Any] | None, metric: str, expected_samples: int) -> float | None:
    if _valid_count(row, expected_samples) != expected_samples:
        return None
    value = row.get("speed", {}).get(metric, {}).get("mean")
    return float(value) if isinstance(value, (int, float)) else None


def _speedup(row: dict[str, Any] | None, metric: str, expected_samples: int) -> float | None:
    if not row or _valid_count(row, expected_samples) != expected_samples:
        return None
    value = row.get("speedup", {}).get(metric)
    return float(value) if isinstance(value, (int, float)) else None


def _raw_rows(run_dir: Path, method: str, dataset: str) -> list[dict[str, Any]]:
    path = run_dir / method / f"{dataset}.jsonl"
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") == "success" and row.get("sample_id"):
            rows.append(row)
    return rows


def _speedup_pair_count(run_dir: Path, method: str, dataset: str) -> int | None:
    """Return valid paired samples when the raw cell exposes that audit."""
    rows = _raw_rows(run_dir, method, dataset)
    if not rows:
        return None
    flags = [row.get("speedup_valid") for row in rows]
    if not any(flag is not None for flag in flags):
        return None
    return sum(flag is True for flag in flags)


def _tau(run_dir: Path, method: str | None, dataset: str, expected_samples: int) -> float | None:
    if method is None:
        return None
    rows = _raw_rows(run_dir, method, dataset)
    if len(rows) != expected_samples:
        return None
    return metrics.mean_acceptance_length(rows)


def _fmt(value: float | None, digits: int = 2) -> str:
    return MISSING if value is None else f"{value:.{digits}f}"


def _cell(summary: dict[str, Any], run_dir: Path, dataset: str, method: str | None,
          expected_samples: int, metric: str) -> tuple[str, str]:
    if method is None:
        return MISSING, "not-run"
    row = _summary_row(summary, dataset, method)
    if metric == "tau":
        # A vanilla autoregressive decoder accepts one target token per
        # iteration by convention.  This is the same table convention used
        # by LongSpec; it is not a measured draft/verification trace.
        if method in {"vanilla_hf", "vanilla_fa"}:
            return "1.00†", "convention"
        if method == "magicdec":
            value = _tau(run_dir, method, dataset, expected_samples)
            # Preserve N/A for historical target-only MagicDec artifacts, but
            # show the measured value automatically after a self-spec rerun.
            return (
                _fmt(value, 2), "measured"
            ) if value is not None else (NOT_APPLICABLE, "not-applicable")
        value = _tau(run_dir, method, dataset, expected_samples)
        return _fmt(value, 2), "measured" if value is not None else "missing"
    if metric == "tokens_s":
        value = _mean_speed(row, "throughput_tok_s", expected_samples)
        return _fmt(value, 2), "measured" if value is not None else "incomplete"
    if metric in {"esr", "dsr"}:
        if method == "vanilla_fa":
            return "1.00", "reference"
        pair_count = _speedup_pair_count(run_dir, method, dataset)
        if pair_count is not None and pair_count < int(expected_samples * 0.95):
            return MISSING, f"only-{pair_count}-pairs"
        value = _speedup(row, metric, expected_samples)
        if value is None:
            return MISSING, "not-valid"
        # DFlash/MagicDec have a small number of invalid pairs in the current
        # artifact; mark them without dropping their aggregate value.
        suffix = "*" if pair_count is not None and pair_count < expected_samples else ""
        return f"{value:.2f}{suffix}", "partial" if suffix else "measured"
    raise ValueError(metric)


def _build(summary: dict[str, Any], run_dir: Path, expected_samples: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    statuses: dict[str, Any] = {}
    for label, method in METHODS:
        output: dict[str, Any] = {"model": "L-8B", "setting": label}
        method_status: dict[str, Any] = {}
        for dataset, _ in DATASETS:
            for metric in ("tau", "tokens_s", "esr", "dsr"):
                value, status = _cell(summary, run_dir, dataset, method, expected_samples, metric)
                output[f"{dataset}_{metric}"] = value
                method_status[f"{dataset}_{metric}"] = status
        rows.append(output)
        statuses[label] = method_status
    return rows, statuses


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "setting"]
    for dataset, _ in DATASETS:
        fields.extend(f"{dataset}_{metric}" for metric in ("tau", "tokens_s", "esr", "dsr"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], statuses: dict[str, Any],
                    run_dir: Path, expected_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Model", "Setting"]
    for _, title in DATASETS:
        fields.extend([f"{title} τ", f"{title} Tokens/s", f"{title} ESR", f"{title} DSR"])
    lines = [
        "# Table 1 — Current LongBench benchmark (provisional)",
        "",
        "> Vanilla HF uses the PyTorch/eager attention path; Vanilla FA uses FlashAttention-2. "
        "Speedups are relative to Vanilla FA and are shown only when the current artifact "
        "contains a valid aggregate. `τ` is mean acceptance length.",
        "",
        f"- Run: `{run_dir}`",
        f"- Profile: `{expected_samples}` samples/dataset; current artifact is partial, not a locked final table.",
        "- `*` = speedup with fewer than 100% but at least 95% valid sample pairs.",
        "- `†` = autoregressive baseline convention (`τ=1`), not a draft/verification trace.",
        "- `—` = not run, failed, missing scope, or not valid for this metric.",
        "",
        "| " + " | ".join(fields) + " |",
        "|" + "|".join(["---"] * len(fields)) + "|",
    ]
    for row in rows:
        values = [row["model"], row["setting"]]
        for dataset, _ in DATASETS:
            values.extend(row[f"{dataset}_{metric}"] for metric in ("tau", "tokens_s", "esr", "dsr"))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "## Validity notes",
        "",
        "- FAFO is excluded because all five cells failed and emitted an aggregate `output_tokens=202752`, not valid sample records.",
        "- EAGLE3 currently has decode-only timing and no canonical E2E speedup; its `τ` and Tokens/s are shown as measured diagnostics.",
        "- `MagicDec` is `N/A` only for historical target-only artifacts; a self-spec rerun with per-sample acceptance instrumentation is rendered as measured `τ`.",
        "- SpecExtend currently has decode-only timing; DSR is shown, while ESR and `τ` remain unavailable.",
        "- LongSpec and SSSD are included to preserve the requested paper layout, but were not part of this seven-baseline run.",
        "- Before publication, rerun after fixing the Vanilla FA quality degeneration and FAFO budget/sidecar failure.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_html(path: Path, rows: list[dict[str, Any]], run_dir: Path, expected_samples: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head_groups = "".join(f'<th colspan="4">{html.escape(title)}</th>' for _, title in DATASETS)
    subheads = "".join("<th>τ</th><th>Tokens/s</th><th>ESR</th><th>DSR</th>" for _ in DATASETS)
    body = []
    for row in rows:
        cells = [f"<td>{html.escape(row['model'])}</td>", f"<td>{html.escape(row['setting'])}</td>"]
        for dataset, _ in DATASETS:
            for metric in ("tau", "tokens_s", "esr", "dsr"):
                value = row[f"{dataset}_{metric}"]
                cells.append(f"<td>{html.escape(value)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Table 1 — LongBench</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
h2 {{ margin-bottom: 4px; }}
.caption {{ color: #444; max-width: 1200px; font-size: 13px; }}
table {{ border-collapse: collapse; margin-top: 18px; font-size: 13px; }}
th, td {{ border: 1px solid #777; padding: 7px 10px; text-align: center; white-space: nowrap; }}
thead tr:first-child th {{ background: #1f527d; color: white; font-weight: 700; }}
thead tr:nth-child(2) th {{ background: #dcebf5; color: #111; }}
td:first-child, td:nth-child(2) {{ text-align: left; font-weight: 600; }}
tr:nth-child(even) td {{ background: #f5f8fa; }}
.notes {{ font-size: 13px; max-width: 1200px; line-height: 1.5; }}
</style></head><body>
<h2>Table 1: Average acceptance length τ, decoding speed (tokens/s), and speedups</h2>
<p class="caption">Vanilla HF uses the PyTorch/eager attention path, while Vanilla FA uses FlashAttention-2. Speedups are relative to Vanilla FA. Current artifact: {html.escape(str(run_dir))}; {expected_samples} samples per dataset. This is a provisional table.</p>
<table><thead><tr><th rowspan="2">Model</th><th rowspan="2">Setting</th>{head_groups}</tr><tr>{subheads}</tr></thead><tbody>{''.join(body)}</tbody></table>
<div class="notes"><p><b>Notes:</b> `*` marks aggregate speedups with a small number of invalid pairs. `N/A` means the metric is not defined for that setting; `—` means not run, failed, missing scope, or not valid. Historical target-only MagicDec artifacts remain `N/A`; self-spec artifacts are rendered from their per-sample acceptance trace. FAFO failed all five cells; EAGLE3 has decode-only timing; SpecExtend has decode-only timing; LongSpec and SSSD were not run. The table must be regenerated after fixing Vanilla FA quality degeneration and the FAFO budget/sidecar failure.</p></div>
</body></html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=100)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    args = parser.parse_args()

    summary_path = args.run_dir / "metrics_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"missing metrics summary: {summary_path}")
    summary = _load_json(summary_path)
    rows, statuses = _build(summary, args.run_dir, args.expected_samples)
    out_md = args.out_md or args.run_dir / "table1_current.md"
    out_csv = args.out_csv or args.run_dir / "table1_current.csv"
    out_html = args.out_html or args.run_dir / "table1_current.html"
    _write_markdown(out_md, rows, statuses, args.run_dir, args.expected_samples)
    _write_csv(out_csv, rows)
    _write_html(out_html, rows, args.run_dir, args.expected_samples)
    print(f"wrote {out_md}")
    print(f"wrote {out_csv}")
    print(f"wrote {out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
