"""Conservative Markdown rendering for residual-headroom phase outputs."""

from __future__ import annotations

import json
from typing import Any, Mapping


def render_markdown_report(phase: str, metrics: Mapping[str, Any]) -> str:
    status = str(metrics.get("status", "UNAVAILABLE")).upper()
    reason = metrics.get("reason")
    lines = [f"# DFlash residual — {phase}", "", f"**Trạng thái:** `{status}`", ""]
    if reason:
        lines.extend([f"**Lý do:** {reason}", ""])
    if status == "UNAVAILABLE":
        lines.extend([
            "Dữ liệu không đủ để kết luận phase này; không gán các metric thiếu bằng 0.",
            "",
        ])
        return "\n".join(lines)
    lines.extend(["## Metrics", "", "```json", json.dumps(dict(metrics), ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    if status in {"PASS", "FAIL", "INCONCLUSIVE"}:
        lines.extend(["## Diễn giải", "", "Decision chỉ áp dụng cho scope/config của trace này.", ""])
    return "\n".join(lines)
