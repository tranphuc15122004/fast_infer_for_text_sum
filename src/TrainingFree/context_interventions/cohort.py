"""Exposure-aware source-group cohorts for the three summarization tasks."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from TrainingFree.context_adaptive.benchmark import split_records
from TrainingFree.context_adaptive.splits import build_split_manifest, source_fingerprint


DATASETS = ("gov_report", "multi_news", "qmsum")
QUOTAS = {
    "pilot": {"gov_report": 2, "multi_news": 2, "qmsum": 2},
    "dev": {"gov_report": 10, "multi_news": 10, "qmsum": 4},
    "confirm": {"gov_report": 18, "multi_news": 18, "qmsum": 12},
}
SPLIT_FOR_PHASE = {"pilot": "dev", "dev": "dev", "confirm": "test"}


def _order(value: str, seed: int = 42) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def build_campaign_cohort(
    samples: Sequence[Mapping[str, Any]],
    exposures: Sequence[Mapping[str, Any]] = (),
    *,
    phase: str,
    seed: int = 42,
    require_quota: bool = True,
    split_manifest_path: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if phase not in {*QUOTAS, "validate"}:
        raise ValueError(f"unsupported cohort phase: {phase}")
    if split_manifest_path:
        split_manifest = json.loads(Path(split_manifest_path).read_text(encoding="utf-8"))
        keys = {str(sample["_cad_record_key"]) for sample in samples}
        if not keys.issubset(set(split_manifest.get("record_group", {}))):
            raise ValueError("provided split manifest does not cover the current dataset records")
        if exposures:
            # Preserve the preregistered assignments, but independently make
            # sure newly supplied exposure evidence does not place a known
            # exposed source group in heldout test.
            audited = build_split_manifest(split_records(samples), exposures, seed=seed)
            exposed_groups = {
                str(group["source_group_id"])
                for group in audited.get("groups", []) if group.get("exposed")
            }
            heldout_groups = set(split_manifest.get("splits", {}).get("test", []))
            leaked = sorted(exposed_groups & heldout_groups)
            if leaked:
                raise ValueError(
                    "provided split manifest assigns audited exposed source groups to test: "
                    f"{leaked[:5]} (n={len(leaked)})"
                )
    else:
        split_manifest = build_split_manifest(split_records(samples), exposures, seed=seed)
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    group_by_record = split_manifest["record_group"]
    for sample in samples:
        key = str(sample["_cad_record_key"])
        group_id = str(group_by_record[key])
        by_group[group_id].append(sample)
    group_rows = {str(group["source_group_id"]): group for group in split_manifest["groups"]}
    split_name = SPLIT_FOR_PHASE.get(phase, "dev")
    quota = QUOTAS.get(phase, {dataset: 2 for dataset in DATASETS})
    eligible: dict[str, list[str]] = {dataset: [] for dataset in DATASETS}
    for group_id, group in group_rows.items():
        if group.get("split") != split_name:
            continue
        for dataset in DATASETS:
            if dataset in group.get("datasets", []):
                eligible[dataset].append(group_id)
    counts = {dataset: len(groups) for dataset, groups in eligible.items()}
    shortages = {
        dataset: {"requested": count, "available": counts.get(dataset, 0)}
        for dataset, count in quota.items()
        if counts.get(dataset, 0) < count
    }
    if shortages and require_quota:
        raise ValueError(
            f"insufficient unexposed {split_name} source groups for {phase} quota: {shortages}"
        )
    selected_groups: list[str] = []
    for dataset in DATASETS:
        ranked = sorted(eligible[dataset], key=lambda group: (_order(group, seed), group))
        selected_groups.extend(ranked[: int(quota[dataset])])
    selected: list[dict[str, Any]] = []
    group_manifest: list[dict[str, Any]] = []
    for group_id in selected_groups:
        members = by_group[group_id]
        # One query per source document/meeting, selected independently of the
        # output and then kept paired across every experiment cell.
        sample = sorted(
            members,
            key=lambda row: (_order(str(row.get("id", "")), seed), str(row.get("id", ""))),
        )[0]
        row = dict(sample)
        row["_cad_source_group_id"] = group_id
        row["_cad_split"] = split_name
        row["_cad_allocation_stratum"] = group_rows[group_id].get("allocation_stratum", row.get("dataset"))
        selected.append(row)
        raw = row.get("raw") or {}
        group_manifest.append({
            "dataset": str(row.get("dataset", "unknown")),
            "source_group_id": group_id,
            "sample_id": str(row.get("id", "")),
            "record_key": str(row.get("_cad_record_key", "")),
            "source_fingerprint": source_fingerprint(str(raw.get("context") or "")),
            "input_tokens": row.get("input_tokens"),
            "split": split_name,
        })
    cohort = {
        "schema_version": "dflash-context-interventions.cohort.v1",
        "phase": phase,
        "split": split_name,
        "seed": seed,
        "quotas": quota,
        "available_groups": counts,
        "selected_group_count": len(selected_groups),
        "status": "complete" if not shortages else "insufficient_data",
        "shortages": shortages,
        "groups": group_manifest,
    }
    return selected, split_manifest, cohort
