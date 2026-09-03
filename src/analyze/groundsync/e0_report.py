"""Markdown/CSV rendering for E0 Target-KV runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_e0_report(manifest: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    """Render a self-contained E0 report without hiding sparse buckets."""

    lines = [
        "# E0 — Target-KV DFlash Failure Map",
        "",
        f"- Status: **{manifest.get('status', 'UNKNOWN')}**",
        f"- Target: `{manifest.get('target_model', '—')}`",
        f"- Drafter: `{manifest.get('draft_model', '—')}`",
        f"- Quantized target: `{manifest.get('target_load_in_8bit', False)}`",
        f"- `records_selected`: `{manifest.get('records_selected', '—')}`; `records_excluded`: `{manifest.get('records_excluded', '—')}`",
        f"- Successful rows: `{manifest.get('successful_generation_rows', '—')}`; round rows: `{manifest.get('round_rows', '—')}`",
        f"- Candidate K: `{manifest.get('candidate_ks', '—')}`; max new tokens: `{manifest.get('max_new_tokens', '—')}`",
        "",
        "## Decision",
        "",
        f"- E0 gate: **{metrics.get('decision', {}).get('status', metrics.get('status', 'UNKNOWN'))}**",
        f"- Reason: `{metrics.get('decision', {}).get('reason', '—')}`",
        "",
        "## Survival and MAT by natural context bucket",
        "",
        "| K | Bucket | Documents | Rounds | MAT | S(1) | S(4) | S(8) | S(16) |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for raw_k, k_metrics in sorted(metrics.get("by_k", {}).items(), key=lambda item: int(item[0])):
        for bucket, values in sorted(k_metrics.get("by_bucket", {}).items()):
            survival = values.get("survival", {})
            lines.append(
                "| "
                + " | ".join(
                    _fmt(value)
                    for value in (
                        raw_k,
                        bucket,
                        values.get("document_count"),
                        values.get("row_count"),
                        values.get("mat"),
                        survival.get("1"),
                        survival.get("4"),
                        survival.get("8"),
                        survival.get("16"),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Context-drop gate",
            "",
            "| K | Status | Relative drop | Short documents | Long documents |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for raw_k, values in sorted(metrics.get("context_drop", {}).items(), key=lambda item: int(item[0])):
        lines.append(
            "| "
            + " | ".join(
                _fmt(value)
                for value in (
                    raw_k,
                    values.get("status"),
                    values.get("relative_drop"),
                    (values.get("short") or {}).get("document_count"),
                    (values.get("long") or {}).get("document_count"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `accepted_draft_tokens` loại target fallback token khỏi DFlash `acceptance_lengths`; survival không đếm fallback là draft acceptance.",
            "- Bootstrap được thực hiện ở document level; các bucket ít document được ghi `INCONCLUSIVE`.",
            "- Run 8-bit chỉ là feasibility evidence và không được gộp với canonical FP16.",
            "- Prompt vượt model/T4 cap không bị truncate im lặng; chúng xuất hiện trong `exclusions.jsonl`.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_e0_report(run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    path = run_dir / "e0_report.md"
    path.write_text(render_e0_report(manifest, metrics), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(write_e0_report(args.run_dir))


if __name__ == "__main__":
    main()
