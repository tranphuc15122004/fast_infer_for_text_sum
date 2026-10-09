"""Paired, source-document-clustered summaries for benchmark artifacts."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import atomic_json, read_jsonl


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dataset", "unknown")),
        str(row.get("sample_id", "")),
        str(row.get("repetition", 0)),
        str(row.get("output_scope", "natural_eos")),
    )


def _valid_speed_rows(rows: Sequence[Mapping[str, Any]], variant: str) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("type") != "sample" or row.get("status") != "ok" or row.get("variant") != variant:
            continue
        key = _pair_key(row)
        if key in indexed:
            raise ValueError(f"duplicate request pair key for {variant}: {key}")
        latency = row.get("e2e_ms")
        if latency is None or not math.isfinite(float(latency)) or float(latency) <= 0:
            continue
        indexed[key] = row
    return indexed


def _macro_log_speedup(pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> float | None:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for baseline, candidate in pairs:
        base_time, candidate_time = float(baseline["e2e_ms"]), float(candidate["e2e_ms"])
        if base_time > 0 and candidate_time > 0:
            by_dataset[str(candidate.get("dataset", "unknown"))].append(math.log(base_time / candidate_time))
    if not by_dataset:
        return None
    return sum(statistics.mean(values) for values in by_dataset.values()) / len(by_dataset)


def paired_comparison(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_variant: str,
    candidate_variant: str,
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    expected_requests: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    baseline = _valid_speed_rows(rows, baseline_variant)
    candidate = _valid_speed_rows(rows, candidate_variant)
    common = sorted(set(baseline) & set(candidate))
    pairs = [(baseline[key], candidate[key]) for key in common]
    baseline_attempted = {
        _pair_key(row)
        for row in rows
        if row.get("type") in {"sample", "error"} and row.get("variant") == baseline_variant
    }
    candidate_attempted = {
        _pair_key(row)
        for row in rows
        if row.get("type") in {"sample", "error"} and row.get("variant") == candidate_variant
    }
    baseline_errors = sum(
        row.get("type") == "error" and row.get("variant") == baseline_variant for row in rows
    )
    candidate_errors = sum(
        row.get("type") == "error" and row.get("variant") == candidate_variant for row in rows
    )
    pair_integrity: list[dict[str, Any]] = []
    for key in common:
        left, right = baseline[key], candidate[key]
        fields = (
            "prompt_hash", "model_signature", "runtime_signature", "attention_backend",
            "dtype", "run_id", "source_group_id", "output_cap", "generation_temperature",
            "split_manifest_sha256", "dataset_manifest_sha256", "calibration_sha256",
            "implementation_sha256", "statistics_update_mode", "cost_update_mode", "timing_mode",
            "fixed_budget", "fixed_gamma", "gamma_reference",
        )
        differences = [field for field in fields if str(left.get(field)) != str(right.get(field))]
        if differences:
            pair_integrity.append({"key": list(key), "fields": differences})
    greedy_scope = all(float(right.get("generation_temperature", 0.0)) == 0.0 for _, right in pairs)
    mismatch = [
        {
            "key": list(key),
            "baseline_match": baseline[key].get("greedy_exact_match"),
            "candidate_match": candidate[key].get("greedy_exact_match"),
            "baseline_hash": baseline[key].get("output_ids_hash"),
            "candidate_hash": candidate[key].get("output_ids_hash"),
        }
        for key in common
        if greedy_scope and baseline[key].get("output_ids_hash") != candidate[key].get("output_ids_hash")
    ]
    log_speedup = _macro_log_speedup(pairs)
    if log_speedup is None:
        speedup = None
    else:
        speedup = math.exp(log_speedup)

    clusters: dict[tuple[str, str], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for left, right in pairs:
        group_id = str(right.get("source_group_id") or f"{right.get('dataset')}::{right.get('sample_id')}")
        stratum = str(right.get("allocation_stratum") or right.get("dataset", "unknown"))
        clusters[(stratum, group_id)].append((left, right))
    strata: dict[str, list[str]] = defaultdict(list)
    for stratum, group_id in clusters:
        strata[stratum].append(group_id)
    rng = random.Random(seed)
    boot: list[float] = []
    if bootstrap_samples > 0 and clusters:
        for _ in range(bootstrap_samples):
            sampled_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
            for stratum, group_ids in strata.items():
                for group_id in (rng.choice(group_ids) for _ in range(len(group_ids))):
                    sampled_pairs.extend(clusters[(stratum, group_id)])
            value = _macro_log_speedup(sampled_pairs)
            if value is not None and math.isfinite(value):
                boot.append(math.exp(value))
    boot.sort()
    ci = None
    if boot:
        ci = {
            "lower_95": boot[max(0, math.ceil(0.025 * len(boot)) - 1)],
            "upper_95": boot[max(0, math.ceil(0.975 * len(boot)) - 1)],
            "samples": len(boot),
            "seed": seed,
        }
    all_hashes_present = all(
        bool(left.get("output_ids_hash")) and bool(right.get("output_ids_hash"))
        for left, right in pairs
    )
    all_greedy_exact = bool(pairs) and greedy_scope and all_hashes_present and not mismatch
    complete_coverage = (
        bool(baseline_attempted)
        and baseline_attempted == candidate_attempted
        and len(baseline) == len(baseline_attempted)
        and len(candidate) == len(candidate_attempted)
        and baseline_errors == 0
        and candidate_errors == 0
        and (expected_requests is None or len(baseline_attempted) == expected_requests.get(baseline_variant))
        and (expected_requests is None or len(candidate_attempted) == expected_requests.get(candidate_variant))
    )
    return {
        "baseline_variant": baseline_variant,
        "candidate_variant": candidate_variant,
        "matched_requests": len(pairs),
        "candidate_unmatched_requests": len(set(candidate) - set(baseline)),
        "baseline_unmatched_requests": len(set(baseline) - set(candidate)),
        "baseline_attempted_requests": len(baseline_attempted),
        "candidate_attempted_requests": len(candidate_attempted),
        "baseline_expected_requests": (expected_requests or {}).get(baseline_variant),
        "candidate_expected_requests": (expected_requests or {}).get(candidate_variant),
        "baseline_errors": baseline_errors,
        "candidate_errors": candidate_errors,
        "complete_coverage": complete_coverage,
        "source_document_clusters": len(clusters),
        "geometric_macro_speedup": speedup,
        "paired_bootstrap_ci": ci,
        "output_mismatches": mismatch,
        "pair_integrity_errors": pair_integrity,
        "all_greedy_exact": all_greedy_exact,
        "sampling_distribution_audit": "not_applicable_greedy" if greedy_scope else "pending",
        "headline_valid": bool(pairs) and complete_coverage and not mismatch and not pair_integrity and all_greedy_exact,
        "correctness_scope": "token-id hash equality on paired requests; sampling runs are not exactness-gated",
    }


def build_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    phase: str = "dev",
    statistics_mode: str = "online",
    bootstrap_samples: int = 10_000,
    expected_requests: Mapping[str, int] | None = None,
    repetitions_by_variant: Mapping[str, int] | None = None,
    warmups_by_variant: Mapping[str, int] | None = None,
    fixed_actions_by_variant: Mapping[str, Mapping[str, Any]] | None = None,
    configs_by_variant: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = [
        row for row in rows
        if row.get("phase", phase) == phase
        and row.get("statistics_update_mode", statistics_mode) == statistics_mode
    ]
    variants = sorted({
        str(row["variant"])
        for row in rows
        if row.get("type") in {"sample", "error"} and row.get("variant")
    })
    baseline = "ar" if "ar" in variants else None
    comparison_pairs: list[tuple[str, str]] = []
    if baseline:
        comparison_pairs.extend((baseline, variant) for variant in variants if variant != baseline)
    # Primary ablations required to attribute gains to context selection,
    # block-length adaptation, and joint interaction (experiment protocol G3-G5).
    protocol_pairs = (
        ("dflash_full_fixed", "a_only"),
        ("dflash_full_fixed", "b_only_history"),
        ("dflash_full_fixed", "b_only_entropy"),
        ("best_fixed_pair", "joint"),
        ("independent_ab", "joint"),
    )
    for baseline_variant, candidate_variant in protocol_pairs:
        if baseline_variant in variants and candidate_variant in variants:
            pair = (baseline_variant, candidate_variant)
            if pair not in comparison_pairs:
                comparison_pairs.append(pair)
    comparisons = [
        paired_comparison(
            rows,
            baseline_variant=baseline_variant,
            candidate_variant=candidate_variant,
            bootstrap_samples=bootstrap_samples,
            expected_requests=expected_requests,
        )
        for baseline_variant, candidate_variant in comparison_pairs
    ]
    summaries: dict[str, Any] = {}
    for variant in variants:
        success = [row for row in rows if row.get("type") == "sample" and row.get("variant") == variant and row.get("status") == "ok"]
        latencies = [float(row["e2e_ms"]) for row in success if row.get("e2e_ms") is not None]
        summaries[variant] = {
            "requests": len(success),
            "errors": sum(1 for row in rows if row.get("type") == "error" and row.get("variant") == variant),
            "mean_e2e_ms": statistics.mean(latencies) if latencies else None,
            "median_e2e_ms": statistics.median(latencies) if latencies else None,
            "p95_e2e_ms": sorted(latencies)[max(0, math.ceil(0.95 * len(latencies)) - 1)] if latencies else None,
        }
    return {
        "schema_version": "cadflash.report.v1",
        "phase": phase,
        "statistics_update_mode": statistics_mode,
        "run_id": _single_value(rows, "run_id"),
        "split_manifest_sha256": _single_value(rows, "split_manifest_sha256"),
        "dataset_manifest_sha256": _single_value(rows, "dataset_manifest_sha256"),
        "implementation_sha256": _single_value(rows, "implementation_sha256"),
        "provenance_mixed": any(
            len({str(row.get(field)) for row in rows if row.get(field) is not None}) > 1
            for field in ("split_manifest_sha256", "dataset_manifest_sha256", "implementation_sha256", "run_id")
        ),
        "expected_requests": dict(expected_requests or {}),
        "repetitions_by_variant": dict(repetitions_by_variant or {}),
        "warmups_by_variant": dict(warmups_by_variant or {}),
        "fixed_actions_by_variant": {
            key: dict(value) for key, value in (fixed_actions_by_variant or {}).items()
        },
        "configs_by_variant": {
            key: dict(value) for key, value in (configs_by_variant or {}).items()
        },
        "request_rows": sum(1 for row in rows if row.get("type") in {"sample", "error"}),
        "variants": summaries,
        "comparisons": comparisons,
        "gate_status": "pending_gpu_correctness_and_complete_coverage",
    }


def _single_value(rows: Sequence[Mapping[str, Any]], field: str) -> str | None:
    values = {str(row[field]) for row in rows if row.get(field) is not None}
    return next(iter(values)) if len(values) == 1 else None


def report_artifacts(
    root: Path,
    *,
    phase: str = "dev",
    statistics_mode: str = "online",
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    root = Path(root)
    if phase not in {"smoke", "dev", "test"} or statistics_mode not in {"online", "frozen"}:
        raise ValueError("report phase/statistics mode is unsupported")
    files = sorted(root.glob(f"{phase}/*/{statistics_mode}/rep_*/requests.jsonl"))
    if not files:
        raise FileNotFoundError(f"no {phase}/{statistics_mode} requests.jsonl artifacts under {root}")
    rows = [row for path in files for row in read_jsonl(path)]
    expected: dict[str, int] = {}
    repetition_counts: dict[str, int] = {}
    warmups: dict[str, int] = {}
    fixed_actions: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    for path in files:
        manifest_path = path.parents[1] / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variant = str(manifest.get("variant", ""))
        repetition_count = int(manifest.get("repetitions", 0))
        sample_count = int(manifest.get("selected_sample_count", 0))
        if variant:
            expected[variant] = max(expected.get(variant, 0), repetition_count * sample_count)
            repetition_counts[variant] = max(repetition_counts.get(variant, 0), repetition_count)
            warmups[variant] = max(warmups.get(variant, 0), int(manifest.get("warmup_runs", 0)))
            fixed_action = manifest.get("fixed_action")
            if isinstance(fixed_action, dict):
                previous = fixed_actions.get(variant)
                if previous is not None and previous != fixed_action:
                    fixed_actions[variant] = {"inconsistent": True}
                else:
                    fixed_actions[variant] = fixed_action
            config = manifest.get("config")
            if isinstance(config, dict):
                previous_config = configs.get(variant)
                if previous_config is not None and previous_config != config:
                    configs[variant] = {"inconsistent": True}
                else:
                    configs[variant] = config
    report = build_report(
        rows,
        phase=phase,
        statistics_mode=statistics_mode,
        bootstrap_samples=bootstrap_samples,
        expected_requests=expected or None,
        repetitions_by_variant=repetition_counts,
        warmups_by_variant=warmups,
        fixed_actions_by_variant=fixed_actions,
        configs_by_variant=configs,
    )
    report_root = root / phase / statistics_mode
    atomic_json(report_root / "comparison.json", report)
    lines = ["# Context-Adaptive DFlash report", "", f"Request rows: {report['request_rows']}", "", "## Variant summaries", "", "| Variant | Requests | Errors | Mean E2E (ms) | Median E2E (ms) | P95 E2E (ms) |", "|---|---:|---:|---:|---:|---:|"]
    for variant, summary in report["variants"].items():
        lines.append(
            f"| {variant} | {summary['requests']} | {summary['errors']} | "
            f"{summary['mean_e2e_ms'] if summary['mean_e2e_ms'] is not None else 'n/a'} | "
            f"{summary['median_e2e_ms'] if summary['median_e2e_ms'] is not None else 'n/a'} | "
            f"{summary['p95_e2e_ms'] if summary['p95_e2e_ms'] is not None else 'n/a'} |"
        )
    lines.extend(["", "## Paired comparisons", ""])
    if not report["comparisons"]:
        lines.append("Chưa có đủ cặp AR và candidate để tính comparison.")
    for item in report["comparisons"]:
        ci = item.get("paired_bootstrap_ci") or {}
        lines.append(
            f"- `{item['candidate_variant']}` so với `{item['baseline_variant']}`: "
            f"n={item['matched_requests']}, geometric macro speedup="
            f"{item['geometric_macro_speedup']}, CI95={ci.get('lower_95')}–{ci.get('upper_95')}, "
            f"exact={item['all_greedy_exact']}, valid={item['headline_valid']}."
        )
    lines.extend(["", "Gate status: **pending** cho đến khi GPU parity, coverage và protocol gates được xác nhận.", ""])
    (report_root / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report
