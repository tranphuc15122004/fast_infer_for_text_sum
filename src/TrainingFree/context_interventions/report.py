"""Coverage summaries, paired cluster intervals, and conservative decisions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from TrainingFree.context_adaptive.schema import atomic_json, read_jsonl
from TrainingFree.context_interventions.evaluation import paired_cluster_bootstrap


def _pair_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    left_label: str,
    right_label: str,
    left_field: str,
    right_field: str,
    family_fields: Sequence[str],
) -> list[dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = tuple(row.get(field) for field in family_fields)
        label = str(row.get("selector_id", ""))
        if label in {left_label, right_label}:
            indexed[key][label] = row
    result = []
    for key, pair in indexed.items():
        if left_label not in pair or right_label not in pair:
            continue
        left, right = pair[left_label], pair[right_label]
        if left.get(left_field) is None or right.get(right_field) is None:
            continue
        result.append({
            "dataset": left.get("dataset"),
            "source_group_id": left.get("source_group_id"),
            "baseline": float(left[left_field]),
            "variant": float(right[right_field]),
        })
    return result


def _metric(rows: Sequence[Mapping[str, Any]], *, confidence: float = 0.95, metric: str = "difference") -> dict[str, Any] | None:
    if not rows:
        return None
    return paired_cluster_bootstrap(
        rows,
        baseline_field="baseline",
        variant_field="variant",
        metric=metric,
        resamples=10_000,
        seed=42,
        confidence=confidence,
    )


def _family_report(rows: Sequence[Mapping[str, Any]], phase: str) -> dict[str, Any]:
    a1_region = [row for row in rows if row.get("experiment") == "A1_region" and row.get("phase") == phase]
    a1_matched = [row for row in rows if row.get("experiment") == "A1_matched_deletion" and row.get("phase") == phase]
    a1 = [row for row in rows if row.get("experiment") == "A1_attention" and row.get("phase") == phase]
    a2 = [row for row in rows if row.get("experiment") == "A2_selector" and row.get("phase") == phase]
    a3 = [row for row in rows if row.get("experiment") == "A3_budget_replay" and row.get("phase") == phase]
    result: dict[str, Any] = {"A1": {}, "A2": {}, "A3": {}}
    region_comparisons = {}
    for name, baseline, variant in (
        ("recent_added_given_source", "source_only", "source_recent"),
        ("source_added_given_recent", "recent_only", "source_recent"),
        ("old_output_added_given_source_recent", "source_recent", "full"),
    ):
        paired = _pair_labels(a1_region, baseline, variant, key_fields=("dataset", "source_group_id", "prefix_round_index"))
        region_comparisons[name] = _metric(paired, confidence=0.9875)
    matched = _pair_labels(
        a1_matched, "delete_recent_D128", "delete_source_D128",
        key_fields=("dataset", "source_group_id", "prefix_round_index"),
    )
    result["A1"]["region_contrasts"] = region_comparisons
    result["A1"]["matched_source_minus_recent_D128"] = _metric(matched, confidence=0.9875)
    high_low = _pair_custom(
        a1,
        left=lambda row: row.get("deletion_side") == "high" and row.get("attention_scoring") == "target_mean5" and row.get("deletion_count") == 2048,
        right=lambda row: row.get("deletion_side") == "low" and row.get("attention_scoring") == "target_mean5" and row.get("deletion_count") == 2048,
        left_field="L_action",
        right_field="L_action",
    )
    attention_ci = _metric(high_low)
    result["A1"]["target_mean5_D2048_low_minus_high"] = attention_ci
    if attention_ci is None:
        result["A1"]["decision"] = "inconclusive"
    elif attention_ci["ci_low"] > 0 and attention_ci["estimate"] >= 0.25:
        result["A1"]["decision"] = "supported"
    elif attention_ci["ci_high"] < 0:
        result["A1"]["decision"] = "opposite_effect"
    else:
        result["A1"]["decision"] = "inconclusive_or_small_effect"

    comparisons: dict[str, Any] = {}
    for baseline in ("random", "recency_B2048", "target_parent_B2048"):
        paired = _pair_a2_selector(a2, baseline)
        comparisons[f"previous_vs_{baseline}"] = _metric(paired, confidence=1.0 - 0.05 / 3.0)
    available = [value for value in comparisons.values() if value is not None]
    if len(available) == 3 and all(value["ci_low"] > 0 and value["estimate"] >= 0.25 for value in available):
        decision = "supported_vs_all_registered_controls"
    elif available and any(value["ci_high"] < 0 for value in available):
        decision = "not_supported_vs_at_least_one_control"
    elif not available:
        decision = "inconclusive_missing_pairs"
    else:
        decision = "inconclusive_or_not_superior_to_all_controls"
    result["A2"] = {"comparisons": comparisons, "decision": decision}
    selector_descriptives = {}
    for selector in ("random_B2048_seed17", "random_B2048_seed42", "random_B2048_seed123", "recency_B2048",
                     "target_parent_B2048", "previous_dflash_B2048", "current_dflash_oracle_B2048"):
        selected_rows = [row for row in a2 if row.get("selector_id") == selector and row.get("status") == "ok"]
        group_deltas: dict[tuple[str, str], list[float]] = defaultdict(list)
        group_survival: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in selected_rows:
            group_key = (str(row.get("dataset")), str(row.get("source_group_id")))
            if row.get("delta_L") is not None:
                group_deltas[group_key].append(float(row["delta_L"]))
            if row.get("L_action") is not None:
                for candidate in range(1, 16):
                    group_survival[group_key][candidate].append(float(int(row["L_action"]) >= candidate))
        deltas = [sum(values) / len(values) for values in group_deltas.values() if values]
        deltas.sort()
        if deltas:
            survival = {}
            for candidate in range(1, 16):
                group_rates = [sum(group_survival[key][candidate]) / len(group_survival[key][candidate])
                               for key in group_survival if group_survival[key].get(candidate)]
                survival[str(candidate)] = sum(group_rates) / len(group_rates) if group_rates else None
            selector_descriptives[selector] = {
                "n_source_groups": len(deltas),
                "P_delta_L_ge_0": sum(value >= 0 for value in deltas) / len(deltas),
                "P_delta_L_gt_0": sum(value > 0 for value in deltas) / len(deltas),
                "delta_L_quantiles": {
                    "q10": deltas[int(0.10 * (len(deltas) - 1))],
                    "q50": deltas[int(0.50 * (len(deltas) - 1))],
                    "q90": deltas[int(0.90 * (len(deltas) - 1))],
                },
                "prefix_survival_P_L_ge_i_group_macro": survival,
            }
    result["A2"]["selector_descriptives"] = selector_descriptives

    for selector in sorted({str(row.get("selector_id")) for row in a3}):
        selector_rows = [row for row in a3 if row.get("selector_id") == selector]
        by_budget = {}
        for budget in sorted({row.get("nominal_budget") for row in selector_rows}, key=lambda item: (item == "full", int(item) if item != "full" else 0)):
            budget_rows = [row for row in selector_rows if row.get("nominal_budget") == budget and row.get("status") == "ok"]
            by_budget[str(budget)] = _metric([
                {"dataset": row.get("dataset"), "source_group_id": row.get("source_group_id"),
                 "baseline": row.get("L_full"), "variant": row.get("L_action")}
                for row in budget_rows
            ])
        result["A3"][selector] = {"acceptance_delta_by_budget": by_budget}
    return result


def _pair_custom(
    rows: Sequence[Mapping[str, Any]],
    *,
    left,
    right,
    left_field: str,
    right_field: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("experiment") == "A1_attention":
            key = (row.get("dataset"), row.get("source_group_id"), row.get("prefix_round_index"), row.get("attention_scoring"), row.get("deletion_count"))
        elif row.get("experiment") == "A1_region":
            key = (row.get("dataset"), row.get("source_group_id"), row.get("prefix_round_index"))
        elif row.get("experiment") == "A2_selector":
            key = (row.get("dataset"), row.get("source_group_id"), row.get("prefix_round_index"), row.get("nominal_budget"), row.get("random_seed") if row.get("selector_id", "").startswith("random") else None)
        else:
            key = (row.get("dataset"), row.get("source_group_id"), row.get("prefix_round_index"), row.get("nominal_budget"))
        if left(row):
            grouped[key]["left"] = row
        if right(row):
            grouped[key]["right"] = row
    paired = []
    for cells in grouped.values():
        if "left" not in cells or "right" not in cells:
            continue
        left_row, right_row = cells["left"], cells["right"]
        if left_row.get(left_field) is None or right_row.get(right_field) is None:
            continue
        paired.append({
            "dataset": left_row.get("dataset"),
            "source_group_id": left_row.get("source_group_id"),
            "baseline": float(left_row[left_field]),
            "variant": float(right_row[right_field]),
        })
    return paired


def _pair_labels(
    rows: Sequence[Mapping[str, Any]],
    baseline_label: str,
    variant_label: str,
    *,
    key_fields: Sequence[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "ok" or row.get("L_action") is None:
            continue
        key = tuple(row.get(field) for field in key_fields)
        label = str(row.get("selector_id", ""))
        if label == baseline_label:
            grouped[key]["baseline"] = row
        elif label == variant_label:
            grouped[key]["variant"] = row
    result = []
    for cells in grouped.values():
        if "baseline" not in cells or "variant" not in cells:
            continue
        base, variant = cells["baseline"], cells["variant"]
        result.append({
            "dataset": base.get("dataset"),
            "source_group_id": base.get("source_group_id"),
            "baseline": float(base["L_action"]),
            "variant": float(variant["L_action"]),
        })
    return result


def _pair_a2_selector(rows: Sequence[Mapping[str, Any]], baseline_name: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("status") != "ok" or row.get("nominal_budget") != 2048:
            continue
        label = str(row.get("selector_id", ""))
        if label.startswith("previous_dflash_B2048"):
            variant = "previous"
        elif baseline_name == "random" and label.startswith("random_B2048_seed"):
            variant = "baseline"
        elif label == baseline_name:
            variant = "baseline"
        else:
            continue
        key = (row.get("dataset"), row.get("source_group_id"), row.get("prefix_round_index"))
        grouped[key][variant].append(row)
    paired = []
    for key, cells in grouped.items():
        if not cells.get("baseline") or not cells.get("previous"):
            continue
        baseline_values = [float(row["L_action"]) for row in cells["baseline"] if row.get("L_action") is not None]
        previous_values = [float(row["L_action"]) for row in cells["previous"] if row.get("L_action") is not None]
        if not baseline_values or not previous_values:
            continue
        paired.append({
            "dataset": key[0], "source_group_id": key[1],
            "baseline": sum(baseline_values) / len(baseline_values),
            "variant": sum(previous_values) / len(previous_values),
        })
    return paired


def summarize_phase(rows: Sequence[Mapping[str, Any]], *, phase: str) -> dict[str, Any]:
    statuses = Counter(str(row.get("status", "missing")) for row in rows if row.get("type") != "summary")
    by_experiment: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("type") == "summary":
            continue
        by_experiment[str(row.get("experiment", row.get("type", "unknown")))][str(row.get("status", "missing"))] += 1
    return {
        "schema_version": "dflash-context-interventions.coverage.v1",
        "phase": phase,
        "planned_or_recorded_rows": sum(statuses.values()),
        "status_counts": dict(statuses),
        "by_experiment": {name: dict(counter) for name, counter in by_experiment.items()},
        "unique_source_groups": len({str(row.get("source_group_id")) for row in rows if row.get("source_group_id")}),
        "family_metrics": _family_report(rows, phase),
    }


def build_report(run_root: Path, phase: str) -> dict[str, Any]:
    run_root = Path(run_root)
    phase_rows: list[dict[str, Any]] = []
    for filename in ("a1.jsonl", "a2.jsonl", "a3_replay.jsonl", "a3_rollout.jsonl"):
        path = run_root / phase / filename
        if path.is_file():
            phase_rows.extend(read_jsonl(path))
    report = summarize_phase(phase_rows, phase=phase)
    report["rollout_metrics"] = _rollout_metrics(phase_rows)
    output_dir = run_root / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / f"{phase}_coverage.json", report)
    (output_dir / f"{phase}_decision.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _rollout_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rollout = [row for row in rows if row.get("type") == "rollout"]
    by_pair: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rollout:
        if row.get("status") != "ok":
            continue
        key = (row.get("dataset"), row.get("source_group_id"), row.get("selector"), row.get("budget"), row.get("refresh_period"), row.get("repetition"))
        by_pair[key][str(row.get("variant"))] = row
    configurations = sorted({
        (str(row.get("selector")), int(row.get("budget", -1)), int(row.get("refresh_period", -1)))
        for row in rollout if row.get("selector")
    })
    output: dict[str, dict[str, Any]] = defaultdict(dict)
    for selector, budget, refresh_period in configurations:
        paired = []
        adapter_pairs = []
        accepted_pairs = []
        mismatch_count = 0
        groups: set[tuple[str, str]] = set()
        for key, variants in by_pair.items():
            if (key[2], int(key[3]), int(key[4])) != (selector, budget, refresh_period):
                continue
            full = variants.get("native_full_dflash")
            adapter = variants.get("adapter_full")
            reduced = variants.get(selector)
            if full is None or adapter is None or reduced is None:
                continue
            groups.add((str(key[0]), str(key[1])))
            mismatch_count += int(not bool(reduced.get("matches_ar")))
            paired.append({
                "dataset": key[0], "source_group_id": key[1],
                "baseline": float(full["wall_e2e_ms"]), "variant": float(reduced["wall_e2e_ms"]),
            })
            adapter_pairs.append({
                "dataset": key[0], "source_group_id": key[1],
                "baseline": float(adapter["wall_e2e_ms"]), "variant": float(reduced["wall_e2e_ms"]),
            })
            full_accept = (full.get("counters") or {}).get("mean_accepted_candidates")
            reduced_accept = (reduced.get("counters") or {}).get("mean_accepted_candidates")
            if full_accept is not None and reduced_accept is not None:
                accepted_pairs.append({
                    "dataset": key[0], "source_group_id": key[1],
                    "baseline": float(full_accept), "variant": float(reduced_accept),
                })
        speedup = _metric(paired, metric="speedup")
        adapter_speedup = _metric(adapter_pairs, metric="speedup")
        acceptance = _metric(accepted_pairs)
        rouge_rows = [row for row in rollout if row.get("selector") == selector
                      and int(row.get("budget", -1)) == budget
                      and int(row.get("refresh_period", -1)) == refresh_period
                      and row.get("status") == "ok"]
        rouge = {}
        for variant in ("native_full_dflash", "adapter_full", selector):
            selected = [row for row in rouge_rows if row.get("variant") == variant]
            for key in ("rouge1", "rouge2", "rougeL"):
                values = [float(row[key]) for row in selected if row.get(key) is not None]
                if values:
                    rouge[f"{variant}_{key}"] = sum(values) / len(values)
        component_sums: dict[str, list[float]] = defaultdict(list)
        refresh_costs: list[float] = []
        stopping = Counter()
        for row in rouge_rows:
            if row.get("variant") != selector:
                continue
            stopping[str(row.get("stopped_by", "unknown"))] += 1
            timings = row.get("reported_timings") or {}
            for field, output_name in (
                ("prefill_ms", "prefill_ms"),
                ("draft_latency_ms", "draft_ms"),
                ("gather_latency_ms", "gather_ms"),
                ("verification_latency_ms", "verify_ms"),
                ("selection_latency_ms", "selection_ms"),
                ("bank_update_latency_ms", "bank_update_ms"),
            ):
                value = timings.get(field)
                if value is not None:
                    component_sums[output_name].append(float(value))
            round_rows = row.get("rounds") or []
            refresh_cost = sum(float(item.get("round_host_ms", 0.0)) for item in round_rows
                               if item.get("refresh_required") and item.get("status") == "ok")
            refresh_costs.append(refresh_cost)
        output[selector][f"B{budget}_period{refresh_period}"] = {
            "selector": selector,
            "budget": budget,
            "refresh_period": refresh_period,
            "speedup_vs_native_full_dflash": speedup,
            "speedup_vs_adapter_full": adapter_speedup,
            "acceptance_delta_vs_native_full": acceptance,
            "rouge_macro": rouge,
            "selected_component_ms_mean": {key: sum(values) / len(values) for key, values in component_sums.items() if values},
            "dense_refresh_wall_ms_mean": sum(refresh_costs) / len(refresh_costs) if refresh_costs else None,
            "selected_stopping_counts": dict(stopping),
            "paired_repetitions": len(paired),
            "source_groups": len(groups),
            "greedy_output_mismatch_rows": mismatch_count,
            "decision": (
                "preservation_and_speedup_gate_pass"
                if speedup and acceptance
                and speedup["estimate"] >= 1.05 and speedup["ci_low"] > 1.0
                and acceptance["ci_low"] > -0.25 and mismatch_count == 0
                else "gate_not_passed_or_inconclusive"
            ),
        }
    return dict(output)


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        f"# Báo cáo thực nghiệm DFlash {report.get('phase')}",
        "",
        "## Coverage",
        "",
        f"- Số rows: {report.get('planned_or_recorded_rows')}",
        f"- Source groups độc lập: {report.get('unique_source_groups')}",
        f"- Trạng thái: `{json.dumps(report.get('status_counts', {}), ensure_ascii=False)}`",
        "",
        "## A1 — Attention và vùng context",
        "",
        f"- Region contrasts: `{json.dumps(report['family_metrics']['A1'].get('region_contrasts', {}), ensure_ascii=False)}`",
        f"- Matched source − recent D=128: `{json.dumps(report['family_metrics']['A1'].get('matched_source_minus_recent_D128'), ensure_ascii=False)}`",
        f"- Primary target_mean5, D=2048: `{json.dumps(report['family_metrics']['A1'].get('target_mean5_D2048_low_minus_high'), ensure_ascii=False)}`",
        f"- Kết luận: **{report['family_metrics']['A1'].get('decision')}**",
        "",
        "## A2 — Selector",
        "",
        f"- Kết luận previous DFlash: **{report['family_metrics']['A2'].get('decision')}**",
        f"- Paired CI: `{json.dumps(report['family_metrics']['A2'].get('comparisons', {}), ensure_ascii=False)}`",
        "",
        "## A3 — Budget, acceptance và latency",
        "",
        f"- Fixed-prefix acceptance: `{json.dumps(report['family_metrics']['A3'], ensure_ascii=False)}`",
        f"- Paired E2E: `{json.dumps(report.get('rollout_metrics', {}), ensure_ascii=False)}`",
        "",
        "Errors, infeasible cells và thiếu coverage được giữ riêng; các hàng thiếu không được tính thành gain bằng không.",
        "",
    ]
    return "\n".join(lines)
