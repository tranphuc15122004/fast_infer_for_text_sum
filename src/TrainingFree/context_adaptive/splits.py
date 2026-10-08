"""Deterministic source-document grouping and exposure-aware data splits."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def source_fingerprint(context: str) -> str:
    canonical = " ".join(unicodedata.normalize("NFKC", context).split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _digest_order(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def read_exposure_manifest(path: Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "payload": payload}


def _exposure_values(manifests: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    fingerprints: set[str] = set()
    raw_hashes: set[str] = set()
    sample_ids: list[dict[str, Any]] = []
    for wrapper in manifests:
        payload = wrapper.get("payload", wrapper)
        if not isinstance(payload, Mapping):
            continue
        sample_ids.extend(payload.get("samples", []) if isinstance(payload.get("samples"), list) else [])
        for name in ("source_fingerprints", "context_fingerprints", "source_group_ids"):
            values = payload.get(name, [])
            if isinstance(values, list):
                fingerprints.update(str(value) for value in values)
    return fingerprints, raw_hashes, sample_ids


def build_split_manifest(
    records: Sequence[Mapping[str, Any]],
    exposures: Sequence[Mapping[str, Any]] = (),
    *,
    seed: int = 42,
) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot split an empty dataset")

    fingerprint_to_rows: dict[str, list[int]] = defaultdict(list)
    provenance_to_rows: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    raw_hash_to_rows: dict[str, list[int]] = defaultdict(list)
    ids_to_rows: dict[str, list[int]] = defaultdict(list)
    rows = list(records)
    for index, row in enumerate(rows):
        context = str(row.get("context") or "")
        dataset = str(row.get("dataset", "unknown"))
        source_split = str(row.get("source_split", "unknown"))
        source_index = str(row.get("source_index", row.get("id", index)))
        if context.strip():
            fingerprint = source_fingerprint(context)
            fingerprint_to_rows[fingerprint].append(index)
            raw_hash_to_rows[hashlib.sha256(context.encode("utf-8")).hexdigest()].append(index)
        else:
            # Without a source span, keep records distinct unless provenance
            # below explicitly joins them (for example QMSum conversation IDs).
            fingerprint = hashlib.sha256(
                f"missing-source:{dataset}:{source_split}:{source_index}".encode("utf-8")
            ).hexdigest()
            fingerprint_to_rows[fingerprint].append(index)
        provenance_to_rows[(dataset, source_split, source_index)].append(index)
        ids_to_rows[str(row.get("id", ""))].append(index)

    parent = list(range(len(rows)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for collection in (fingerprint_to_rows, provenance_to_rows):
        for indexes in collection.values():
            for index in indexes[1:]:
                union(indexes[0], index)

    fingerprints, exposed_raw_hashes, exposed_samples = _exposure_values(exposures)
    exposed_ids: set[str] = set()
    resolved_exposure_ids: set[str] = set()
    for sample in exposed_samples:
        if not isinstance(sample, Mapping):
            continue
        sample_id = sample.get("sample_id", sample.get("id"))
        if sample_id is not None:
            exposed_ids.add(str(sample_id))
            if str(sample_id) in ids_to_rows:
                resolved_exposure_ids.add(str(sample_id))
        source_hash = sample.get("source_sha256") or sample.get("context_sha256")
        if source_hash:
            exposed_raw_hashes.add(str(source_hash))
            if str(source_hash) in raw_hash_to_rows:
                resolved_exposure_ids.add(str(sample_id)) if sample_id is not None else None

    exposed_rows: set[int] = set()
    for fingerprint in fingerprints:
        exposed_rows.update(fingerprint_to_rows.get(fingerprint, ()))
    for raw_hash in exposed_raw_hashes:
        exposed_rows.update(raw_hash_to_rows.get(raw_hash, ()))
    for sample_id in exposed_ids:
        exposed_rows.update(ids_to_rows.get(sample_id, ()))

    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        grouped[find(index)].append(index)
    groups: list[dict[str, Any]] = []
    row_to_group: dict[int, str] = {}
    for indexes in grouped.values():
        member_fingerprints = sorted({
            source_fingerprint(str(rows[i].get("context") or ""))
            if str(rows[i].get("context") or "").strip()
            else hashlib.sha256(
                f"missing-source:{rows[i].get('dataset', 'unknown')}:{rows[i].get('source_split', 'unknown')}:{rows[i].get('source_index', rows[i].get('id', i))}".encode("utf-8")
            ).hexdigest()
            for i in indexes
        })
        group_id = member_fingerprints[0]
        datasets = sorted({str(rows[i].get("dataset", "unknown")) for i in indexes})
        group = {
            "source_group_id": group_id,
            "allocation_stratum": datasets[0],
            "datasets": datasets,
            "record_indices": indexes,
            "record_ids": [str(rows[i].get("id", i)) for i in indexes],
            "exposed": any(i in exposed_rows for i in indexes),
        }
        groups.append(group)
        for i in indexes:
            row_to_group[i] = group_id

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        strata[group["allocation_stratum"]].append(group)
    for stratum, members in sorted(strata.items()):
        count = len(members)
        calibration_target = int(0.2 * count)
        dev_target = max(calibration_target, sum(bool(g["exposed"]) for g in members))
        insufficient = False
        if count >= 3:
            calibration_target = max(1, calibration_target)
            dev_target = max(1, dev_target)
        if calibration_target + dev_target >= count:
            # Preserve one test group where possible, without moving exposed
            # documents out of dev. Mark impossible quotas as unsupported.
            if sum(bool(g["exposed"]) for g in members) > count - calibration_target - 1:
                insufficient = True
            else:
                dev_target = min(dev_target, count - calibration_target - 1)
        exposed = [g for g in members if g["exposed"]]
        exposed.sort(key=lambda g: g["source_group_id"])
        remaining = sorted(
            (g for g in members if not g["exposed"]),
            key=lambda g: _digest_order(g["source_group_id"], seed),
        )
        for group in exposed:
            group["split"] = "dev"
        for group in remaining[:calibration_target]:
            group["split"] = "calibration"
        already_dev = len(exposed)
        for group in remaining[calibration_target : calibration_target + max(0, dev_target - already_dev)]:
            group["split"] = "dev"
        for group in members:
            group.setdefault("split", "test")
        present = {group["split"] for group in members}
        if insufficient or count < 3 or present != {"calibration", "dev", "test"}:
            for group in members:
                group["split"] = "insufficient_data"

    splits: dict[str, list[str]] = {"calibration": [], "dev": [], "test": []}
    record_split: dict[str, str] = {}
    for group in groups:
        if group["split"] in splits:
            splits[group["split"]].append(group["source_group_id"])
        for index in group["record_indices"]:
            record_split[str(rows[index].get("id", index))] = group["split"]
    exposure_sources = [str(item.get("path", "inline")) for item in exposures]
    return {
        "schema_version": "cadflash.split.v1",
        "seed": seed,
        "status": "complete" if all(g["split"] != "insufficient_data" for g in groups) else "insufficient_data",
        "exposure_sources": exposure_sources,
        "groups": groups,
        "splits": splits,
        "record_split": record_split,
        "record_group": {str(rows[i].get("id", i)): row_to_group[i] for i in range(len(rows))},
        "unresolved_exposure_sample_ids": sorted(exposed_ids - resolved_exposure_ids),
    }
