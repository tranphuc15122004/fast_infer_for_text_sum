"""Conservative Markdown rendering for residual-headroom phase outputs."""

from __future__ import annotations

import json
from typing import Any, Mapping


def _render_prefix_gap_section(phase: str, metrics: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    if phase == "e1":
        gate = metrics.get("h1_gate", {})
        lines.extend([
            "## Matched-context task comparison",
            "",
            f"Context cap: `{metrics.get('context_cap')}`; gate: `{gate.get('decision', 'INCONCLUSIVE')}`.",
            "",
            "| Comparison | MAT canonical | MAT summarization | Relative drop | Bootstrap CI | Decision |",
            "|---|---:|---:|---:|---|---|",
        ])
        for name, comparison in metrics.get("pairwise", {}).items():
            lines.append(
                f"| {name} | {comparison.get('canonical_mat')} | {comparison.get('summarization_mat')} "
                f"| {comparison.get('relative_drop')} | {comparison.get('bootstrap_ci')} | {comparison.get('decision')} |"
            )
        lines.append("")
    elif phase == "e2":
        lines.extend([
            "## Top-K prefix oracle",
            "",
            "`MAT_Ok` (and `MAT_O16` when K=16) is the sum of joint Top-K prefix survival "
            "`S_k(j)`; it is not the marginal Recall@K.",
            "",
            "Top-K membership is tie-aware at the recorded logit boundary so BF16 `topk` "
            "ordering ties cannot make the oracle smaller than the greedy path.",
            "",
            "| Group | K | Documents | Blocks | MAT_D | MAT_Ok | Oracle headroom | MAT_O16/MAT_D | E4 gate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for group, group_metrics in metrics.get("groups", {}).items():
            for k, values in group_metrics.get("k_values", {}).items():
                lines.append(
                    f"| {group} | {k} | {group_metrics.get('documents')} | {group_metrics.get('blocks')} "
                    f"| {group_metrics.get('mat_d')} | {values.get('mat_oracle')} "
                    f"| {values.get('oracle_headroom_over_dflash')} | {group_metrics.get('oracle_ratio_k16') if int(k) == 16 else ''} "
                    f"| {group_metrics.get('e4_gate_k16') if int(k) == 16 else ''} |"
                )
        lines.extend([
            "",
            "The JSON/CSV artifacts also contain `joint_survival`, `independent_survival`, "
            "`conditional_survival`, and marginal per-position recall.",
            "",
        ])
    elif phase == "e3":
        lines.extend([
            "## Candidate rank ambiguity",
            "",
            "Rank-conditioned statistics include only rows where the target is already in Top-16.",
            "",
            "| Regime | Rows | R@16 | MRR | Mean rank | Logit deficit | Top-16 entropy | Top1–Top2 margin |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for regime, values in metrics.get("regimes", {}).items():
            lines.append(
                f"| {regime} | {values.get('rows')} | {values.get('recall_at_16')} "
                f"| {values.get('mrr')} | {values.get('mean_target_rank')} "
                f"| {values.get('mean_target_logit_deficit')} | {values.get('mean_top16_entropy')} "
                f"| {values.get('mean_top1_top2_margin')} |"
            )
        lines.append("")
    return lines


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
    lines.extend(_render_prefix_gap_section(phase, metrics))
    lines.extend(["## Metrics", "", "```json", json.dumps(dict(metrics), ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    if status in {"PASS", "FAIL", "INCONCLUSIVE"}:
        lines.extend(["## Diễn giải", "", "Decision chỉ áp dụng cho scope/config của trace này.", ""])
    return "\n".join(lines)
