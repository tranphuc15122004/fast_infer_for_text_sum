#!/usr/bin/env python3
"""Stage a source-group-disjoint, length-stratified Context-Adaptive DFlash pilot."""

from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/length_bins"
STAGE = ROOT / "outputs/cadflash_lengthbins_modal_20261009/assets"
BIN_FILES = (
    "bin1_00k_02k.jsonl",
    "bin2_02k_04k.jsonl",
    "bin3_04k_08k.jsonl",
    "bin4_08k_12k.jsonl",
    "bin5_12k_16k.jsonl",
)
BIN_NAMES = tuple(path.removesuffix(".jsonl") for path in BIN_FILES)
ALLOWED_DATASETS = {"gov_report", "multi_news", "qmsum"}
CALIBRATION_PER_BIN = 6
DEV_PER_BIN = 3


def _canonical_context(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _source_fingerprint(row: dict[str, Any], index: int) -> str:
    context = _canonical_context(str(row.get("context") or ""))
    if context:
        return hashlib.sha256(context.encode("utf-8")).hexdigest()
    provenance = "|".join(
        str(row.get(key, fallback))
        for key, fallback in (("dataset", "unknown"), ("source_split", "unknown"), ("source_index", index))
    )
    return hashlib.sha256(f"missing-source:{provenance}".encode("utf-8")).hexdigest()


def _spread(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda row: (int(row.get("input_tokens", 0)), str(row["id"])))
    if len(rows) <= count:
        return rows
    chosen = []
    for index in range(count):
        rank = round((index + 0.5) * len(rows) / count - 0.5)
        candidate = rows[max(0, min(len(rows) - 1, rank))]
        if candidate not in chosen:
            chosen.append(candidate)
    for candidate in rows:
        if len(chosen) >= count:
            break
        if candidate not in chosen:
            chosen.append(candidate)
    return chosen[:count]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    STAGE.mkdir(parents=True, exist_ok=True)
    input_dir = STAGE / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    rows_by_bin: dict[int, list[dict[str, Any]]] = defaultdict(list)
    provenance_groups: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for bin_ordinal, filename in enumerate(BIN_FILES, start=1):
        path = SOURCE / filename
        for local_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_type") != "summarization" or row.get("dataset") not in ALLOWED_DATASETS:
                continue
            row["_pilot_bin_ordinal"] = bin_ordinal
            row["_pilot_source_fingerprint"] = _source_fingerprint(row, local_index)
            rows_by_bin[bin_ordinal].append(row)
            provenance = (
                str(row.get("dataset", "unknown")),
                str(row.get("source_split", "unknown")),
                str(row.get("source_index", row.get("id", local_index))),
            )
            provenance_groups[provenance].append((bin_ordinal, row))

    # Treat identical source text or identical LongBench source provenance as one group.
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for rows in rows_by_bin.values():
        for row in rows:
            find(row["_pilot_source_fingerprint"])
    by_fingerprint: dict[str, list[str]] = defaultdict(list)
    for rows in rows_by_bin.values():
        for row in rows:
            provenance = (
                str(row.get("dataset", "unknown")),
                str(row.get("source_split", "unknown")),
                str(row.get("source_index", row.get("id", ""))),
            )
            by_fingerprint["|".join(provenance)].append(row["_pilot_source_fingerprint"])
    for fingerprints in by_fingerprint.values():
        for fingerprint in fingerprints[1:]:
            union(fingerprints[0], fingerprint)
    for rows in rows_by_bin.values():
        fingerprints: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            fingerprints[row["_pilot_source_fingerprint"]].append(row["_pilot_source_fingerprint"])
        for values in fingerprints.values():
            for fingerprint in values[1:]:
                union(values[0], fingerprint)
    root_for_fingerprint = {value: find(value) for value in parent}

    used_groups: set[str] = set()
    selected: list[tuple[str, dict[str, Any]]] = []
    for bin_ordinal in range(1, 6):
        candidates_by_group: dict[str, dict[str, Any]] = {}
        for row in rows_by_bin[bin_ordinal]:
            fingerprint = row["_pilot_source_fingerprint"]
            group_id = root_for_fingerprint[fingerprint]
            current = candidates_by_group.get(group_id)
            if current is None or (int(row.get("input_tokens", 0)), str(row["id"])) < (
                int(current.get("input_tokens", 0)), str(current["id"])
            ):
                candidates_by_group[group_id] = row
        candidates = sorted(
            (row for group, row in candidates_by_group.items() if group not in used_groups),
            key=lambda row: (int(row.get("input_tokens", 0)), str(row["dataset"]), str(row["id"])),
        )
        calibration = _spread(candidates, CALIBRATION_PER_BIN)
        for row in calibration:
            selected.append(("calibration", row))
            used_groups.add(root_for_fingerprint[row["_pilot_source_fingerprint"]])
        remaining = [
            row for row in candidates
            if root_for_fingerprint[row["_pilot_source_fingerprint"]] not in used_groups
        ]
        dev = _spread(remaining, DEV_PER_BIN)
        for row in dev:
            selected.append(("dev", row))
            used_groups.add(root_for_fingerprint[row["_pilot_source_fingerprint"]])
        print(
            f"{BIN_NAMES[bin_ordinal - 1]}: summarization_groups={len(candidates_by_group)} "
            f"calibration={len(calibration)} dev={len(dev)}",
            flush=True,
        )

    clean_rows: list[dict[str, Any]] = []
    selected_info: list[dict[str, Any]] = []
    for split, row in selected:
        clean = {key: value for key, value in row.items() if not key.startswith("_pilot_")}
        clean_rows.append(clean)
        bin_ordinal = int(row["_pilot_bin_ordinal"])
        selected_info.append({
            "id": str(row["id"]),
            "dataset": str(row["dataset"]),
            "bin_ordinal": bin_ordinal,
            "length_bin_name": BIN_NAMES[bin_ordinal - 1],
            "source_length_bin_zero_based": int(row.get("length_bin", bin_ordinal - 1)),
            "input_tokens_actual_from_source_file": int(row.get("input_tokens", 0)),
            "task_type": str(row.get("task_type")),
            "split": split,
            "source_group_id": _source_fingerprint(row, 0),
        })

    data_path = input_dir / "lengthbin_pilot.jsonl"
    data_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clean_rows),
        encoding="utf-8",
    )

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(clean_rows):
        grouped[_source_fingerprint(row, index)].append(index)
    record_split: dict[str, str] = {}
    record_group: dict[str, str] = {}
    split_groups: dict[str, list[str]] = {"calibration": [], "dev": [], "test": []}
    group_rows = []
    selection_split = {(item["dataset"], item["id"]): item["split"] for item in selected_info}
    for group_id, indexes in sorted(grouped.items()):
        splits = {
            selection_split[(str(clean_rows[index]["dataset"]), str(clean_rows[index]["id"]))]
            for index in indexes
        }
        if len(splits) != 1:
            raise RuntimeError(f"source group crosses pilot splits: {group_id}")
        split = next(iter(splits))
        datasets = sorted({str(clean_rows[index]["dataset"]) for index in indexes})
        keys = [f"{clean_rows[index]['dataset']}::{clean_rows[index]['id']}" for index in indexes]
        split_groups[split].append(group_id)
        group_rows.append({
            "source_group_id": group_id,
            "allocation_stratum": datasets[0],
            "datasets": datasets,
            "record_indices": indexes,
            "record_ids": keys,
            "exposed": False,
            "split": split,
        })
        for key in keys:
            record_split[key] = split
            record_group[key] = group_id

    split_manifest = {
        "schema_version": "cadflash.split.v1",
        "seed": 42,
        "status": "complete",
        "exposure_sources": [],
        "groups": group_rows,
        "splits": split_groups,
        "record_split": record_split,
        "record_group": record_group,
        "unresolved_exposure_sample_ids": [],
        "pilot_note": (
            "Deterministic, source-group-disjoint, length-stratified exploratory split; "
            "six calibration groups and three dev groups per bin where available."
        ),
    }
    split_path = STAGE / "split_manifest.json"
    split_path.write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    per_bin = {
        name: {
            split: sum(1 for item in selected_info if item["length_bin_name"] == name and item["split"] == split)
            for split in ("calibration", "dev")
        }
        for name in BIN_NAMES
    }
    sample_manifest = {
        "schema_version": "cadflash.lengthbin_pilot.v1",
        "seed": 42,
        "source_data_dir": "data/length_bins",
        "allowed_datasets": sorted(ALLOWED_DATASETS),
        "excluded_tasks": ["code_completion"],
        "length_bin_semantics": (
            "bin_ordinal is 1-based; source_length_bin_zero_based preserves the source JSONL field (0-4)."
        ),
        "samples": selected_info,
        "counts": {
            "records": len(clean_rows),
            "calibration": len(split_groups["calibration"]),
            "dev": len(split_groups["dev"]),
            "groups": len(group_rows),
            "split_group_counts": {key: len(value) for key, value in split_groups.items()},
            "per_bin": per_bin,
        },
        "input_data_sha256": _sha256(data_path),
    }
    (STAGE / "sample_manifest.json").write_text(
        json.dumps(sample_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    expected = ROOT / "outputs/modal_cadflash_pilot_20261009/assets/expected_target_signature.json"
    draft_source = ROOT / "outputs/modal_cadflash_pilot_20261009/assets/models/Qwen3-4B-DFlash-b16"
    draft_target = STAGE / "models/Qwen3-4B-DFlash-b16"
    draft_target.parent.mkdir(parents=True, exist_ok=True)
    if not draft_target.exists():
        shutil.copytree(draft_source, draft_target)
    shutil.copy2(expected, STAGE / "expected_target_signature.json")

    data_hash = _sha256(data_path)
    split_hash = _sha256(split_path)
    sample_hash = _sha256(STAGE / "sample_manifest.json")
    preregistration = f"""# Preregistration — Context-Adaptive DFlash length-bin pilot

## Mục tiêu

Đo mean acceptance length, throughput (tok/s), E2E latency và speedup của `joint` cùng các ablation so với DFlash full-context, fixed-gamma trên cohort nhỏ thuộc đủ 5 khoảng độ dài.

## Cohort và split

- Nguồn: `data/length_bins/`, seed chọn mẫu 42.
- Chỉ giữ `task_type=summarization` thuộc `gov_report`, `multi_news`, `qmsum`; loại code-completion (`lcc`, `repobench-p`).
- Mỗi bin: tối đa 6 source groups calibration và 3 source groups dev, chọn trải đều theo `input_tokens`; thực tế 30 calibration + 15 dev records.
- Tách source groups trước khi chạy; không có cùng context/provenance giữa calibration và dev.
- Hash dữ liệu: `{data_hash}`
- Hash split: `{split_hash}`
- Hash sample manifest: `{sample_hash}`

## So sánh

Variants: `dflash_full_fixed` (đối chứng DFlash gốc: full context, gamma 15), `a_only`, `b_only_history`, `b_only_entropy`, `independent_ab`, `joint_no_entropy`, `joint_no_source_relevance`, `joint`; chạy thêm `ar` để kiểm greedy parity.

- GPU: Modal L40S, target Qwen3-4B và draft Qwen3-4B-DFlash-b16; backend/dtype ghi từ runtime thực tế.
- Greedy decode, temperature 0, output cố định 128 token, 3 repetitions, 1 warmup/request variant.
- Action grid: context budget `1024/full`, gamma `3/7/15`; selector `target_parent`; online statistics, frozen cost, giữ `min_state_support=8`.
- Calibration dùng riêng calibration split, checkpoints output 0 và 64, 3 repetitions. Dev dùng 15 mẫu, 3 repetitions.
- Primary comparisons paired theo sample/repetition. Speedup = geometric mean của `DFlash full fixed E2E / variant E2E` trong từng length bin. Tok/s báo trung bình request-level.
- Báo cả acceptance length và token-output exactness. Nếu greedy output khác AR, latency/speedup chỉ là mô tả, không phải claim inference tương đương.

## Giới hạn đã biết

Đây là pilot nhỏ (3 dev documents/bin, một GPU model/backend, fixed 128-token output); không đủ để kết luận heldout hoặc natural-EOS summarization. Cost/state support và tỷ lệ fallback phải được đọc cùng bảng metrics. Không điều chỉnh policy sau khi xem dev kết quả.
"""
    (STAGE / "preregistration.md").write_text(preregistration, encoding="utf-8")

    print(json.dumps(sample_manifest["counts"], ensure_ascii=False, indent=2))
    print(f"data_sha256={data_hash}")
    print(f"split_sha256={split_hash}")
    print(f"sample_manifest_sha256={sample_hash}")


if __name__ == "__main__":
    main()
