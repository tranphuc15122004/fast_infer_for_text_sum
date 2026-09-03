"""Markdown renderer for E1 representation-probe artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_e1_report(manifest: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    lines = [
        "# E1 — Target-KV Representation Sufficiency Probe",
        "",
        f"- Extraction status: **{manifest.get('status', 'UNKNOWN')}**",
        f"- Device: `{manifest.get('hardware', {}).get('device', result.get('device', '—'))}`",
        f"- Feature rows: `{manifest.get('feature_rows', '—')}`; excluded: `{manifest.get('excluded_rows', '—')}`",
        f"- Horizon: `{manifest.get('horizon', '—')}`; memory interface: `{manifest.get('max_memory_tokens', '—')}×{manifest.get('interface_dim', '—')}`",
        f"- Partitions: `{result.get('partitions', '—')}`",
        "",
        "## Representation comparison",
        "",
        "| Representation | Status | Rows train/test | CE mean | Acc@1 mean | Acc@5 mean | Prefix exact @8 | Parameters |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in result.get("representations", {}).items():
        lines.append(
            "| "
            + " | ".join(
                _fmt(value)
                for value in (
                    name,
                    metrics.get("status"),
                    f"{metrics.get('train_rows', '—')}/{metrics.get('test_rows', '—')}",
                    metrics.get("ce_mean"),
                    sum(metrics.get("acc1_by_position", [])) / max(len(metrics.get("acc1_by_position", [])), 1)
                    if metrics.get("acc1_by_position")
                    else None,
                    sum(metrics.get("acc5_by_position", [])) / max(len(metrics.get("acc5_by_position", [])), 1)
                    if metrics.get("acc5_by_position")
                    else None,
                    (metrics.get("prefix_exact_by_position") or [None] * 8)[7]
                    if len(metrics.get("prefix_exact_by_position", [])) >= 8
                    else None,
                    metrics.get("parameter_count"),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `hidden` là lower control; `hidden_sequence` là control bắt buộc để không nhầm KV với việc có thêm token-wise memory.",
            "- `kv_shuffled`, `kv_recent` và `kv_wrong_document` là negative controls; kết quả phải được đọc cùng document split.",
            "- CE/accuracy của một pilot ít document chỉ là directional evidence; không được gọi là xác nhận tổng quát nếu CI hoặc coverage không đủ.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_e1_report(run_dir: Path) -> Path:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((run_dir / "probe_metrics.json").read_text(encoding="utf-8"))
    path = run_dir / "e1_report.md"
    path.write_text(render_e1_report(manifest, result), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(write_e1_report(args.run_dir))


if __name__ == "__main__":
    main()

