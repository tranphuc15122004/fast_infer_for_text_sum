"""Offline analysis for the self-conditioned DFlash bridge."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import read_trace_jsonl
from .joint_lattice import lattice_stats
from .metrics import _blocks


def _ok(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status", "ok") == "ok"]


def _mean_acceptance(rows: Sequence[Mapping[str, Any]]) -> float | None:
    blocks = _blocks(rows)
    if not blocks:
        return None
    values = []
    for block in blocks:
        accepted = [row.get("accepted_draft_len") for row in block if row.get("accepted_draft_len") is not None]
        if accepted:
            values.append(float(accepted[0]))
    return sum(values) / len(values) if values else None


def _mean_critical(values: Mapping[str, float | None]) -> float | None:
    selected = [float(values[str(position)]) for position in range(3, 9) if values.get(str(position)) is not None]
    return sum(selected) / len(selected) if selected else None


def _timing(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    blocks = _blocks(rows)
    values: dict[str, list[float]] = {"draft_s": [], "verify_s": [], "total_s": []}
    for block in blocks:
        row = block[0]
        for key, field in (("draft_s", "draft_elapsed_s"), ("verify_s", "verify_elapsed_s"), ("total_s", "cycle_elapsed_s")):
            values[key].append(float(row.get(field, 0.0)))
    return {
        "mean_draft_s": sum(values["draft_s"]) / len(values["draft_s"]) if values["draft_s"] else None,
        "mean_verify_s": sum(values["verify_s"]) / len(values["verify_s"]) if values["verify_s"] else None,
        "mean_cycle_s": sum(values["total_s"]) / len(values["total_s"]) if values["total_s"] else None,
        "total_draft_s": sum(values["draft_s"]),
        "total_verify_s": sum(values["verify_s"]),
        "total_cycle_s": sum(values["total_s"]),
        "blocks": len(values["total_s"]),
    }


def _condition_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stats = lattice_stats(rows, k=16, max_position=15)
    timing = _timing(rows)
    mat_d = _mean_acceptance(rows)
    return {
        "rows": len(rows),
        "blocks": len(_blocks(rows)),
        "mat_d": mat_d,
        "mat_o16": stats.get("mat_o16"),
        "mean_r16_3_8": _mean_critical(stats.get("marginal_recall", {})),
        "mean_j16_3_8": _mean_critical(stats.get("joint_survival", {})),
        "timing": timing,
    }


def analyze(path: str | Path) -> dict[str, Any]:
    rows = _ok(read_trace_jsonl(path))
    by_stage: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_r: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        stage = str(row.get("self_condition_stage", "unknown"))
        by_stage[stage].append(row)
        if stage == "redraft":
            by_r[str(int(row.get("self_condition_r", 0)))].append(row)
    if "baseline" not in by_stage:
        return {"status": "inconclusive", "reason": "missing_baseline", "trace": str(path)}
    baseline = _condition_stats(by_stage["baseline"])
    conditions: dict[str, Any] = {}
    for reveal, redraft_rows in sorted(by_r.items(), key=lambda item: int(item[0])):
        redraft = _condition_stats(redraft_rows)
        base_mat = baseline["mat_d"]
        base_oracle = baseline["mat_o16"]
        base_speed = (
            base_mat / baseline["timing"]["mean_cycle_s"]
            if base_mat is not None and baseline["timing"]["mean_cycle_s"]
            else None
        )
        total_method_time = baseline["timing"]["mean_cycle_s"] + redraft["timing"]["mean_cycle_s"]
        self_speed = (
            redraft["mat_d"] / total_method_time
            if redraft["mat_d"] is not None and total_method_time
            else None
        )
        redraft.update({
            "reveal_count": int(reveal),
            "delta_mat_d": redraft["mat_d"] - base_mat if redraft["mat_d"] is not None and base_mat is not None else None,
            "relative_mat_d_gain": (redraft["mat_d"] - base_mat) / base_mat if redraft["mat_d"] is not None and base_mat else None,
            "delta_mat_o16": redraft["mat_o16"] - base_oracle if redraft["mat_o16"] is not None and base_oracle is not None else None,
            "relative_mat_o16_gain": (redraft["mat_o16"] - base_oracle) / base_oracle if redraft["mat_o16"] is not None and base_oracle else None,
            "delta_r16_3_8": redraft["mean_r16_3_8"] - baseline["mean_r16_3_8"] if redraft["mean_r16_3_8"] is not None else None,
            "method_mean_cycle_s": total_method_time,
            "baseline_accepted_tokens_per_cycle_s": base_speed,
            "self_conditioned_accepted_tokens_per_cycle_s": self_speed,
            "cost_adjusted_utility_relative_gain": (self_speed - base_speed) / base_speed if self_speed is not None and base_speed else None,
        })
        conditions[reveal] = redraft
    return {
        "status": "ok",
        "experiment": "E21_SELF_CONDITIONED",
        "trace": str(path),
        "rows": len(rows),
        "baseline": baseline,
        "conditions": conditions,
    }


def report(result: Mapping[str, Any]) -> str:
    lines = [
        "# E21 — Self-conditioned DFlash redraft bridge",
        "",
        "Đây là oracle-to-method bridge: pass 1 sinh prefix bằng DFlash, pass 2 dùng chính prefix đó để redraft phần suffix song song. Target chỉ dùng cho verification/measurement.",
        "",
        "## Kết quả",
        "",
        "| Dataset | Baseline MAT_D | Baseline MAT_O16 | Base R16(3:8) | r | Redraft MAT_D | Redraft MAT_O16 | Redraft R16(3:8) | ΔMAT_D | ΔMAT_O16 | Cost-adjusted utility gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in result.get("datasets", {}).items():
        if item.get("status") != "ok":
            lines.append(f"| {dataset} | n/a | n/a | n/a | inconclusive | n/a | n/a | n/a | n/a |")
            continue
        for reveal, condition in item.get("conditions", {}).items():
            def f(value: Any, digits: int = 4) -> str:
                return "n/a" if value is None else f"{float(value):.{digits}f}"
            lines.append(
                f"| {dataset} | {f(item['baseline']['mat_d'])} | {f(item['baseline']['mat_o16'])} | {f(item['baseline']['mean_r16_3_8'])} | {reveal} | "
                f"{f(condition['mat_d'])} | {f(condition['mat_o16'])} | {f(condition['mean_r16_3_8'])} | {f(condition['delta_mat_d'])} | "
                f"{f(condition['delta_mat_o16'])} | {f(condition['cost_adjusted_utility_relative_gain'])} |"
            )
    lines.extend([
        "",
        "## Timing",
        "",
        "Timing gồm DFlash draft forward và target verification forward; shared fixed-state target prefill không tính trong cycle timing. Self-conditioned utility tính cả pass 1 + pass 2.",
        "",
        "| Dataset | Baseline draft ms | Baseline verify ms | Baseline cycle ms | r | Redraft draft ms | Redraft verify ms | Total method cycle ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset, item in result.get("datasets", {}).items():
        if item.get("status") != "ok":
            continue
        b = item["baseline"]["timing"]
        for reveal, c in item["conditions"].items():
            t = c["timing"]
            lines.append(
                f"| {dataset} | {1000*b['mean_draft_s']:.2f} | {1000*b['mean_verify_s']:.2f} | {1000*b['mean_cycle_s']:.2f} | {reveal} | "
                f"{1000*t['mean_draft_s']:.2f} | {1000*t['mean_verify_s']:.2f} | {1000*c['method_mean_cycle_s']:.2f} |"
            )
    lines.extend([
        "",
        "## Gate",
        "",
        "Gate đề xuất: ΔMAT_D ≥15–20% và cost-adjusted utility dương. Không gọi method promising nếu chỉ tăng oracle nhưng giảm accepted tokens/second.",
        "",
        "## Giới hạn",
        "",
        "Đây là prototype screening trên fixed on-policy states và Qwen3-4B DFlash. Prefix tự điều kiện hóa được lấy từ raw greedy draft của pass 1; chưa có training, chưa có long rollout, chưa có throughput benchmark server và chưa so sánh DFlash2.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", action="append", required=True, help="DATASET=TRACE_PATH")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    datasets: dict[str, Any] = {}
    for spec in args.trace:
        dataset, sep, path = spec.partition("=")
        if not sep:
            raise ValueError(f"trace must be DATASET=PATH, got {spec!r}")
        datasets[dataset] = analyze(path)
    result = {"status": "ok", "experiment": "E21_SELF_CONDITIONED", "datasets": datasets}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "report.md").write_text(report(result), encoding="utf-8")
    print(f"self-conditioned report: {output}")


if __name__ == "__main__":
    main()
