#!/usr/bin/env python3
"""Safely requeue quarantined samples from a full-context regenerate stage.

Run only after the Phase 1 pipeline and its managed vLLM server have stopped.
Default behavior is a read-only plan; pass --apply to back up the affected
artifacts, remove quarantine-only records from worker skip shards, reset the
shared queue, and invalidate the train-regeneration success marker.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("rb") as handle:
        for number, raw in enumerate(handle, 1):
            raw = raw.rstrip(b"\r\n")
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"JSONL lỗi tại {path}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row không phải object tại {path}:{number}")
            rows.append(row)
    return rows


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".requeue.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".requeue.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("id", "")) for row in rows if row.get("id")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="thực hiện thay đổi sau khi đã xác nhận không còn job/vLLM đang chạy",
    )
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    work_root = run_root / "regenerated_full" / "parallel_regenerate_train"
    canonical_output = run_root / "regenerated_full" / "train.jsonl"
    canonical_skipped = run_root / "regenerated_full" / "train.skipped.jsonl"
    canonical_manifest = run_root / "manifests" / "regeneration_full_train.json"
    plan_path = work_root / "parallel_plan.json"
    quarantine_path = work_root / "quarantine.jsonl"

    for path in (canonical_output, canonical_skipped, canonical_manifest, plan_path, quarantine_path):
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu artifact bắt buộc: {path}")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    queue_root = Path(plan["queue_root"]).resolve()
    queue_meta = queue_root / "meta.json"
    if not queue_meta.is_file():
        raise FileNotFoundError(f"Thiếu shared queue metadata: {queue_meta}")
    meta = json.loads(queue_meta.read_text(encoding="utf-8"))
    if not str(meta.get("stage", "")).startswith("regenerate:"):
        raise ValueError(f"Queue không phải regeneration queue: {meta.get('stage')!r}")

    quarantine_records = read_jsonl(quarantine_path)
    quarantine_ids = {
        str(row.get("sample_id", "")) for row in quarantine_records if row.get("sample_id")
    }
    if not quarantine_ids:
        raise ValueError("quarantine.jsonl không có sample_id để requeue")

    canonical_rows = read_jsonl(canonical_output)
    canonical_skips = read_jsonl(canonical_skipped)
    canonical_output_ids = ids(canonical_rows)
    canonical_skip_ids = ids(canonical_skips)
    if canonical_output_ids & canonical_skip_ids:
        raise ValueError("ID xuất hiện đồng thời trong train.jsonl và train.skipped.jsonl")
    canonical_quarantine_ids = {
        str(row.get("id", ""))
        for row in canonical_skips
        if row.get("kind") == "quarantine" and row.get("id")
    }
    if canonical_quarantine_ids != quarantine_ids:
        raise ValueError(
            "Danh sách quarantine trong canonical skipped và work-root không khớp: "
            f"canonical={len(canonical_quarantine_ids)}, queue={len(quarantine_ids)}"
        )
    short_rows = [row for row in canonical_skips if row.get("kind") == "too_short_target"]
    short_ids = ids(short_rows)
    if short_ids & canonical_output_ids:
        raise ValueError("Mẫu too_short_target vẫn còn trong canonical train output")

    rank_roots = sorted(work_root.glob("rank_*"))
    if not rank_roots:
        raise FileNotFoundError(f"Không có rank worker root trong {work_root}")
    if [path.name for path in rank_roots] != ["rank_00"]:
        raise ValueError(
            "Run-root này không chỉ có một worker rank; không tự requeue để tránh "
            f"bỏ sót shard khi merge: {[path.name for path in rank_roots]}"
        )

    loaded: dict[Path, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    rank_quarantine_ids: set[str] = set()
    output_ids: set[str] = set()
    non_quarantine_skip_ids: set[str] = set()
    for rank_root in rank_roots:
        output_path = rank_root / "output.jsonl"
        skipped_path = rank_root / "skipped.jsonl"
        output_rows = read_jsonl(output_path)
        skipped_rows = read_jsonl(skipped_path)
        loaded[rank_root] = (output_rows, skipped_rows)
        output_ids.update(ids(output_rows))
        for row in skipped_rows:
            sample_id = str(row.get("id", ""))
            if row.get("kind") == "quarantine" and sample_id:
                rank_quarantine_ids.add(sample_id)
            elif sample_id:
                non_quarantine_skip_ids.add(sample_id)

    if rank_quarantine_ids != quarantine_ids:
        raise ValueError(
            "Quarantine ID trong worker shard và quarantine.jsonl không khớp: "
            f"shard={len(rank_quarantine_ids)}, queue={len(quarantine_ids)}"
        )
    if short_ids - (output_ids | non_quarantine_skip_ids):
        raise ValueError("Có too_short_target canonical không tìm thấy trong worker shards")

    pending_ids = quarantine_ids - output_ids - non_quarantine_skip_ids
    unexpected_missing = (canonical_output_ids | canonical_skip_ids) - (
        output_ids | non_quarantine_skip_ids | quarantine_ids
    )
    if unexpected_missing:
        raise ValueError(
            "Canonical có ID không được worker shard/quarantine đại diện: "
            f"{sorted(unexpected_missing)[:5]}"
        )

    print(f"run_root: {run_root}")
    print(f"queue_root: {queue_root}")
    print(f"quarantined IDs: {len(quarantine_ids)}")
    print(f"already durable in output/skip: {len(quarantine_ids) - len(pending_ids)}")
    print(f"will be pending on fresh queue: {len(pending_ids)}")
    print(f"short targets preserved as skipped: {len(short_ids)}")
    if not args.apply:
        print("Dry run only. Stop pipeline/vLLM, then rerun with --apply to proceed.")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = run_root / f"recovery_backup_requeue_{stamp}"
    backup.mkdir()

    to_backup = [canonical_output, canonical_skipped, canonical_manifest, plan_path, quarantine_path]
    for rank_root in rank_roots:
        to_backup.extend((rank_root / "output.jsonl", rank_root / "skipped.jsonl"))
    status_path = work_root / "status.json"
    if status_path.is_file():
        to_backup.append(status_path)
    marker_paths = [
        run_root / "pipeline_state" / "regenerate_full_train.success.json",
        run_root / "pipeline_state" / "regenerate_full_train.failed.json",
        run_root / "pipeline_state" / "regenerate_full_train.running.json",
    ]
    to_backup.extend(path for path in marker_paths if path.is_file())
    for path in to_backup:
        if path.is_file():
            destination = backup / path.relative_to(run_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)

    queue_backup = backup / "old_shared_queue"
    shutil.move(str(queue_root), str(queue_backup))

    for rank_root, (output_rows, skipped_rows) in loaded.items():
        output_rows = [row for row in output_rows if str(row.get("id", "")) not in short_ids]
        skipped_rows = [row for row in skipped_rows if row.get("kind") != "quarantine"]
        present_skipped = ids(skipped_rows)
        if rank_root == rank_roots[0]:
            skipped_rows.extend(row for row in short_rows if str(row.get("id", "")) not in present_skipped)
        atomic_jsonl(rank_root / "output.jsonl", output_rows)
        atomic_jsonl(rank_root / "skipped.jsonl", skipped_rows)

    quarantine_path.unlink(missing_ok=True)
    status_path.unlink(missing_ok=True)
    for marker in marker_paths:
        marker.unlink(missing_ok=True)

    print(f"Recovery backup: {backup}")
    print("Quarantine skip records removed from worker shards; old queue backed up.")
    print("Short-target skips preserved in worker shards; train regeneration success marker cleared.")
    print("Run the normal B200 Phase 1 mode-2 launcher with the same RUN_ROOT to retry and merge.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[requeue-quarantine] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
