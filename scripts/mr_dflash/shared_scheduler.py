"""Durable length-aware work queue for MR-DFlash preprocessing.

The queue deliberately uses only regular files.  This is important for the
server setup where different hosts/OSes see the same ``/workspace/storage``
mount but do not share a Python environment or a database service.

``items.jsonl`` is immutable after initialization.  State transitions are
append-only events protected by an atomic directory lock, so a killed worker
can be replaced from another host without rebuilding the input dataset.
Only one scheduler invocation should own a queue at a time; the TTL is for
recovering a lock after a killed process, not for running two coordinators.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Optional


@dataclass(frozen=True)
class WorkItem:
    sample_id: str
    index: int
    length: int

    def __post_init__(self) -> None:
        if not str(self.sample_id):
            raise ValueError("WorkItem cần sample_id")
        if int(self.index) < 0:
            raise ValueError("WorkItem index phải >= 0")
        if int(self.length) < 0:
            raise ValueError("WorkItem length phải >= 0")


@dataclass(frozen=True)
class QueueLease:
    lease_id: str
    worker_id: str
    items: tuple[WorkItem, ...]
    expires_at: float


class SharedLeaseQueue:
    """A resumable queue shared through a filesystem mount.

    ``config_hash`` identifies the data/model/length configuration, but not
    the number or identity of GPUs.  Therefore a resumed run may use a
    different host and a different GPU set without mixing incompatible
    artifacts.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        stage: str,
        config_hash: str,
        lease_ttl_seconds: float = 300.0,
        max_attempts: int = 3,
        lock_ttl_seconds: float = 600.0,
        now_fn: Optional[Callable[[], float]] = None,
        lock_timeout_seconds: float = 30.0,
    ) -> None:
        if float(lease_ttl_seconds) <= 0:
            raise ValueError("lease_ttl_seconds phải > 0")
        if int(max_attempts) < 1:
            raise ValueError("max_attempts phải >= 1")
        if float(lock_ttl_seconds) <= 0:
            raise ValueError("lock_ttl_seconds phải > 0")
        self.root = Path(root)
        self.stage = str(stage)
        self.config_hash = str(config_hash)
        self.lease_ttl_seconds = float(lease_ttl_seconds)
        self.max_attempts = int(max_attempts)
        self.lock_ttl_seconds = float(lock_ttl_seconds)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._now = now_fn or time.time
        self._items_path = self.root / "items.jsonl"
        self._meta_path = self.root / "meta.json"
        self._events_path = self.root / "events.jsonl"
        self._lock_path = self.root / ".scheduler.lock"

    def initialize(self, items: Iterable[WorkItem], *, resume: bool = True) -> None:
        """Create or validate the immutable workload manifest."""
        normalized = sorted(
            (
                item
                if isinstance(item, WorkItem)
                else WorkItem(
                    sample_id=str(item["sample_id"]),
                    index=int(item["index"]),
                    length=int(item.get("length", 0)),
                )
                for item in items
            ),
            key=lambda item: (int(item.length), int(item.index), str(item.sample_id)),
        )
        ids = [item.sample_id for item in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError("queue có sample_id trùng")
        manifest_hash = self._manifest_hash(normalized)
        self.root.mkdir(parents=True, exist_ok=True)
        with self._mutex():
            if self._meta_path.exists():
                meta = self._read_json(self._meta_path)
                self._validate_meta(meta, manifest_hash=manifest_hash)
                if not resume:
                    raise FileExistsError(f"queue đã tồn tại: {self.root}")
                return
            self._write_jsonl_atomic(self._items_path, (asdict(item) for item in normalized))
            self._write_json_atomic(
                self._meta_path,
                {
                    "schema_version": "mr_dflash_shared_queue_v1",
                    "stage": self.stage,
                    "config_hash": self.config_hash,
                    "manifest_hash": manifest_hash,
                    "num_items": len(normalized),
                    "max_attempts": self.max_attempts,
                    "created_at_unix": float(self._now()),
                    "created_by": {
                        "host": socket.gethostname(),
                        "pid": os.getpid(),
                    },
                },
            )
            self._events_path.touch()

    def claim(self, *, worker_id: str, max_items: int) -> QueueLease | None:
        if int(max_items) < 1:
            raise ValueError("max_items phải >= 1")
        with self._mutex():
            items = self._read_items()
            states = self._replay(items)
            now = float(self._now())
            available = [
                item
                for item in items
                if self._is_available(states[item.sample_id], now)
            ]
            if not available:
                return None
            selected = tuple(available[: int(max_items)])
            lease_id = uuid.uuid4().hex
            expires_at = now + self.lease_ttl_seconds
            self._append_event(
                {
                    "event": "claim",
                    "lease_id": lease_id,
                    "worker_id": str(worker_id),
                    "item_ids": [item.sample_id for item in selected],
                    "expires_at": expires_at,
                    "timestamp": now,
                }
            )
            return QueueLease(
                lease_id=lease_id,
                worker_id=str(worker_id),
                items=selected,
                expires_at=expires_at,
            )

    def heartbeat(self, lease_id: str) -> float:
        """Extend a live lease and return its new expiry timestamp."""
        with self._mutex():
            states = self._replay(self._read_items())
            lease_items = [
                state
                for state in states.values()
                if state.get("status") == "leased" and state.get("lease_id") == str(lease_id)
            ]
            if not lease_items or any(float(state["expires_at"]) <= float(self._now()) for state in lease_items):
                raise RuntimeError(f"lease không còn hợp lệ: {lease_id}")
            expires_at = float(self._now()) + self.lease_ttl_seconds
            self._append_event(
                {
                    "event": "heartbeat",
                    "lease_id": str(lease_id),
                    "expires_at": expires_at,
                    "timestamp": float(self._now()),
                }
            )
            return expires_at

    def complete(
        self,
        lease_id: str,
        *,
        sample_ids: Optional[Iterable[str]] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        """Commit all items of a live lease atomically in the event log."""
        with self._mutex():
            states = self._replay(self._read_items())
            leased = [
                (sample_id, state)
                for sample_id, state in states.items()
                if state.get("status") == "leased" and state.get("lease_id") == str(lease_id)
            ]
            if not leased:
                if any(
                    state.get("status") == "completed" and state.get("lease_id") == str(lease_id)
                    for state in states.values()
                ):
                    return
                raise RuntimeError(f"lease không còn hợp lệ: {lease_id}")
            if any(float(state["expires_at"]) <= float(self._now()) for _, state in leased):
                raise RuntimeError(f"lease không còn hợp lệ: {lease_id}")
            artifact_map = artifacts or {}
            selected_ids = (
                [sample_id for sample_id, _ in leased]
                if sample_ids is None
                else [str(value) for value in sample_ids]
            )
            leased_ids = {sample_id for sample_id, _ in leased}
            unknown = sorted(set(selected_ids) - leased_ids)
            if unknown:
                raise RuntimeError(
                    f"sample không thuộc lease {lease_id}: {unknown[:5]}"
                )
            self._append_event(
                {
                    "event": "complete",
                    "lease_id": str(lease_id),
                    "item_ids": selected_ids,
                    "artifacts": {str(key): str(value) for key, value in artifact_map.items()},
                    "timestamp": float(self._now()),
                }
            )

    def seed_completed(
        self,
        sample_ids: Iterable[str],
        *,
        artifact: Optional[str] = None,
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        """Import durable IDs from a legacy/static worker output.

        This is used once when migrating an interrupted static-shard run to
        ``shared_lease``. It never marks an active lease as complete.
        """
        ids = [str(value) for value in sample_ids if str(value)]
        if not ids:
            return
        with self._mutex():
            states = self._replay(self._read_items())
            unknown = sorted(set(ids) - set(states))
            if unknown:
                raise ValueError(f"seed sample_id không có trong queue: {unknown[:5]}")
            eligible = [
                sample_id
                for sample_id in ids
                if states[sample_id].get("status") in {"pending", "completed"}
            ]
            artifact_map = dict(artifacts or {})
            if artifact is not None:
                artifact_map.update({sample_id: str(artifact) for sample_id in eligible})
            if eligible:
                self._append_event(
                    {
                        "event": "seed_complete",
                        "item_ids": sorted(set(eligible)),
                        "artifacts": artifact_map,
                        "timestamp": float(self._now()),
                    }
                )

    def retry(self, lease_id: str, *, error: str) -> str:
        """Requeue a failed lease or quarantine it after max attempts."""
        with self._mutex():
            states = self._replay(self._read_items())
            leased = [
                (sample_id, state)
                for sample_id, state in states.items()
                if state.get("status") == "leased" and state.get("lease_id") == str(lease_id)
            ]
            if not leased:
                raise RuntimeError(f"lease không còn hợp lệ: {lease_id}")
            next_attempt = max(int(state.get("attempts", 0)) + 1 for _, state in leased)
            status = "quarantined" if next_attempt >= self.max_attempts else "pending"
            self._append_event(
                {
                    "event": "retry",
                    "lease_id": str(lease_id),
                    "item_ids": [sample_id for sample_id, _ in leased],
                    "attempts": next_attempt,
                    "status": status,
                    "error": str(error),
                    "timestamp": float(self._now()),
                }
            )
            return "quarantined" if status == "quarantined" else "requeued"

    def snapshot(self) -> Dict[str, int]:
        states = self._replay(self._read_items())
        now = float(self._now())
        counts = {"pending": 0, "leased": 0, "completed": 0, "quarantined": 0}
        for state in states.values():
            status = str(state.get("status", "pending"))
            if status == "leased" and float(state.get("expires_at", 0.0)) <= now:
                status = "pending"
            counts[status] = counts.get(status, 0) + 1
        counts["total"] = len(states)
        return counts

    def quarantine_records(self) -> list[Dict[str, Any]]:
        states = self._replay(self._read_items())
        return [
            {
                "sample_id": sample_id,
                "attempts": int(state.get("attempts", 0)),
                "error": str(state.get("error", "")),
            }
            for sample_id, state in states.items()
            if state.get("status") == "quarantined"
        ]

    def _read_items(self) -> list[WorkItem]:
        if not self._items_path.is_file() or not self._meta_path.is_file():
            raise RuntimeError(f"queue chưa initialize: {self.root}")
        items: list[WorkItem] = []
        with self._items_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    items.append(
                        WorkItem(
                            sample_id=str(value["sample_id"]),
                            index=int(value["index"]),
                            length=int(value.get("length", 0)),
                        )
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"queue items lỗi tại dòng {line_number}") from exc
        return items

    def _replay(self, items: list[WorkItem]) -> Dict[str, Dict[str, Any]]:
        states = {
            item.sample_id: {
                "status": "pending",
                "attempts": 0,
                "lease_id": None,
                "expires_at": 0.0,
                "artifact": None,
                "error": None,
            }
            for item in items
        }
        if not self._events_path.exists():
            return states
        with self._events_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    kind = str(event["event"])
                    ids = [str(value) for value in event.get("item_ids", [])]
                    if kind == "claim":
                        for sample_id in ids:
                            if sample_id in states:
                                states[sample_id].update(
                                    status="leased",
                                    lease_id=str(event["lease_id"]),
                                    expires_at=float(event["expires_at"]),
                                )
                    elif kind == "heartbeat":
                        for state in states.values():
                            if state.get("lease_id") == str(event["lease_id"]):
                                state["expires_at"] = float(event["expires_at"])
                    elif kind == "complete":
                        artifacts = event.get("artifacts") or {}
                        for sample_id in ids:
                            state = states.get(sample_id)
                            if state is not None and state.get("lease_id") == str(event["lease_id"]):
                                state.update(
                                    status="completed",
                                    artifact=artifacts.get(sample_id),
                                )
                    elif kind == "seed_complete":
                        artifacts = event.get("artifacts") or {}
                        for sample_id in ids:
                            state = states.get(sample_id)
                            if state is not None and state.get("status") in {"pending", "completed"}:
                                state.update(
                                    status="completed",
                                    artifact=artifacts.get(sample_id),
                                )
                    elif kind == "retry":
                        for sample_id in ids:
                            state = states.get(sample_id)
                            if state is not None and state.get("lease_id") == str(event["lease_id"]):
                                state.update(
                                    status=str(event["status"]),
                                    attempts=int(event["attempts"]),
                                    error=str(event.get("error", "")),
                                    lease_id=None,
                                    expires_at=0.0,
                                )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"queue events lỗi tại dòng {line_number}") from exc
        return states

    def _is_available(self, state: Dict[str, Any], now: float) -> bool:
        status = str(state.get("status", "pending"))
        return status == "pending" or (
            status == "leased" and float(state.get("expires_at", 0.0)) <= float(now)
        )

    def _validate_meta(self, meta: Dict[str, Any], *, manifest_hash: str) -> None:
        if meta.get("stage") != self.stage:
            raise ValueError(f"queue stage không khớp: {meta.get('stage')} != {self.stage}")
        if meta.get("config_hash") != self.config_hash:
            raise ValueError("queue config_hash không khớp; không được trộn artifact")
        if meta.get("manifest_hash") != manifest_hash:
            raise ValueError("queue manifest_hash không khớp; input đã thay đổi")

    @staticmethod
    def _manifest_hash(items: list[WorkItem]) -> str:
        payload = json.dumps([asdict(item) for item in items], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _append_event(self, event: Dict[str, Any]) -> None:
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _write_jsonl_atomic(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"queue metadata không phải object: {path}")
        return value

    @contextmanager
    def _mutex(self) -> Iterator[None]:
        self.root.mkdir(parents=True, exist_ok=True)
        started = float(self._now())
        while True:
            try:
                self._lock_path.mkdir()
                self._write_json_atomic(
                    self._lock_path / "owner.json",
                    {
                        "host": socket.gethostname(),
                        "pid": os.getpid(),
                        "created_at_unix": float(self._now()),
                    },
                )
                break
            except FileExistsError:
                owner = self._lock_path / "owner.json"
                try:
                    age = float(self._now()) - owner.stat().st_mtime
                except OSError:
                    age = 0.0
                if age > self.lock_ttl_seconds:
                    stale = self.root / f".scheduler.lock.stale.{uuid.uuid4().hex}"
                    try:
                        os.replace(self._lock_path, stale)
                        shutil.rmtree(stale, ignore_errors=True)
                        continue
                    except OSError:
                        pass
                if float(self._now()) - started >= self.lock_timeout_seconds:
                    raise RuntimeError(f"queue đang bị scheduler khác lock: {self.root}")
                time.sleep(0.05)
        try:
            yield
        finally:
            try:
                (self._lock_path / "owner.json").unlink(missing_ok=True)
                self._lock_path.rmdir()
            except OSError:
                # A stale-lock takeover may already have moved this directory.
                pass
