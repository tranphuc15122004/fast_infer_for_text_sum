from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _items():
    from shared_scheduler import WorkItem

    return [
        WorkItem(sample_id="long", index=2, length=30),
        WorkItem(sample_id="short", index=0, length=5),
        WorkItem(sample_id="mid", index=1, length=15),
    ]


def test_shared_queue_claims_in_length_order_and_commits_once(tmp_path: Path) -> None:
    from shared_scheduler import SharedLeaseQueue

    queue = SharedLeaseQueue(
        tmp_path / "queue",
        stage="regenerate_train",
        config_hash="cfg-v1",
        lease_ttl_seconds=60,
    )
    queue.initialize(_items())

    lease = queue.claim(worker_id="host-a/gpu-0", max_items=2)
    assert [item.sample_id for item in lease.items] == ["short", "mid"]
    queue.complete(lease.lease_id, artifacts={"short": "a.jsonl", "mid": "a.jsonl"})

    assert queue.claim(worker_id="host-a/gpu-1", max_items=2).items[0].sample_id == "long"
    assert queue.snapshot()["completed"] == 2


def test_expired_lease_is_reclaimed_by_different_host(tmp_path: Path) -> None:
    from shared_scheduler import SharedLeaseQueue

    now = [100.0]
    queue = SharedLeaseQueue(
        tmp_path / "queue",
        stage="cache_train",
        config_hash="cfg-v1",
        lease_ttl_seconds=10,
        now_fn=lambda: now[0],
    )
    queue.initialize(_items())
    first = queue.claim(worker_id="host-a/gpu-0", max_items=1)
    assert first.items[0].sample_id == "short"

    now[0] = 111.0
    second = queue.claim(worker_id="host-b/gpu-1", max_items=1)
    assert second.items[0].sample_id == "short"
    with pytest.raises(RuntimeError, match="lease không còn hợp lệ"):
        queue.complete(first.lease_id)


def test_retry_quarantines_after_three_attempts_without_crashing_queue(tmp_path: Path) -> None:
    from shared_scheduler import SharedLeaseQueue

    queue = SharedLeaseQueue(
        tmp_path / "queue",
        stage="cache_train",
        config_hash="cfg-v1",
        max_attempts=3,
    )
    queue.initialize([_items()[0]])

    for attempt in range(1, 4):
        lease = queue.claim(worker_id=f"host-a/gpu-{attempt}", max_items=1)
        result = queue.retry(lease.lease_id, error=f"CUDA OOM attempt {attempt}")
        if attempt < 3:
            assert result == "requeued"
        else:
            assert result == "quarantined"

    state = queue.snapshot()
    assert state["pending"] == 0
    assert state["quarantined"] == 1
    assert queue.quarantine_records()[0]["sample_id"] == "long"


def test_resume_accepts_different_gpu_set_and_rejects_config_mix(tmp_path: Path) -> None:
    from shared_scheduler import SharedLeaseQueue

    root = tmp_path / "queue"
    first = SharedLeaseQueue(root, stage="regenerate_train", config_hash="cfg-v1")
    first.initialize(_items())
    first.claim(worker_id="host-a/gpu-0", max_items=1)

    resumed = SharedLeaseQueue(root, stage="regenerate_train", config_hash="cfg-v1")
    assert resumed.snapshot()["leased"] == 1

    with pytest.raises(ValueError, match="config_hash"):
        SharedLeaseQueue(root, stage="regenerate_train", config_hash="cfg-v2").initialize(_items())


def test_seed_completed_imports_durable_artifacts_from_previous_static_run(tmp_path: Path) -> None:
    from shared_scheduler import SharedLeaseQueue

    queue = SharedLeaseQueue(tmp_path / "queue", stage="regenerate", config_hash="cfg")
    queue.initialize(_items())
    queue.seed_completed(["short"], artifact="old/rank_00/output.jsonl")
    assert queue.snapshot()["completed"] == 1
    lease = queue.claim(worker_id="new-host/gpu-0", max_items=10)
    assert lease is not None
    assert [item.sample_id for item in lease.items] == ["mid", "long"]


def test_failed_lease_can_commit_partial_results_before_requeue(tmp_path: Path) -> None:
    from shared_scheduler import SharedLeaseQueue

    queue = SharedLeaseQueue(tmp_path / "queue", stage="regenerate", config_hash="cfg")
    queue.initialize(_items())
    lease = queue.claim(worker_id="host-a/gpu-0", max_items=3)
    assert lease is not None
    queue.complete(lease.lease_id, sample_ids=["short"], artifacts={"short": "partial.jsonl"})
    assert queue.retry(lease.lease_id, error="OOM") == "requeued"
    assert queue.snapshot()["completed"] == 1
    retry = queue.claim(worker_id="host-b/gpu-0", max_items=3)
    assert retry is not None
    assert [item.sample_id for item in retry.items] == ["mid", "long"]


def test_parallel_queue_items_use_token_length_and_ignore_gpu_identity(tmp_path: Path) -> None:
    from parallel_stage import build_shared_queue_items

    input_path = tmp_path / "prompts.jsonl"
    input_path.write_text(
        '{"id":"long","metadata":{"source_token_length":30}}\n'
        '{"id":"short","metadata":{"source_token_length":5}}\n',
        encoding="utf-8",
    )
    items = build_shared_queue_items("regenerate", input_path)
    assert [(item.sample_id, item.length) for item in items] == [("long", 30), ("short", 5)]


def test_shared_lease_assignment_is_bounded_by_queue_quantum(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from parallel_stage import assign_shared_leases
    from shared_scheduler import SharedLeaseQueue, WorkItem

    queue = SharedLeaseQueue(tmp_path / "queue", stage="regenerate", config_hash="cfg")
    queue.initialize([WorkItem(sample_id=f"s{i}", index=i, length=i) for i in range(10)])
    args = SimpleNamespace(
        gpu_ids=[0, 1],
        queue_quantum_items=2,
        mode="regenerate",
        queue_root=None,
        queue_lease_ttl_seconds=300.0,
        queue_lock_ttl_seconds=600.0,
        queue_max_attempts=3,
    )

    leases = assign_shared_leases(queue, args, tmp_path / "work")

    assert {rank: len(lease.items) for rank, lease in leases.items()} == {0: 2, 1: 2}
    assert queue.snapshot()["pending"] == 6


def test_parallel_shared_scheduler_resumes_queued_assignments(tmp_path: Path, monkeypatch) -> None:
    import json

    import parallel_stage
    from _common import write_json, write_jsonl

    input_path = tmp_path / "prompts.jsonl"
    write_jsonl(
        input_path,
        [
            {"id": "s0", "metadata": {"source_token_length": 5}},
            {"id": "s1", "metadata": {"source_token_length": 10}},
        ],
    )
    output_path = tmp_path / "output.jsonl"
    manifest_path = tmp_path / "manifest.json"

    def fake_workers(args, work_root, gpu_ids, *, total_samples):
        del total_samples
        roots = []
        for rank, _gpu_id in enumerate(gpu_ids):
            root = work_root / f"rank_{rank:02d}"
            ids_file = args._shared_sample_ids_files[rank]
            ids = [json.loads(line)["sample_id"] for line in ids_file.read_text().splitlines()]
            write_jsonl(root / "output.jsonl", [{"id": sample_id} for sample_id in ids])
            write_jsonl(root / "skipped.jsonl", [])
            write_json(root / "manifest.json", {"stats": {}})
            roots.append(root)
        return roots

    monkeypatch.setattr(parallel_stage, "_run_workers", fake_workers)
    assert parallel_stage.main(
        [
            "--mode", "regenerate",
            "--scheduler", "shared_lease",
            "--gpu-ids", "0", "1",
            "--input", str(input_path),
            "--output", str(output_path),
            "--manifest", str(manifest_path),
            "--work-root", str(tmp_path / "work"),
            "--target-model-path", "tiny",
            "--max-length", "64",
            "--resume",
        ]
    ) == 0
    assert [json.loads(line)["id"] for line in output_path.read_text().splitlines()] == ["s0", "s1"]


def test_parallel_shared_scheduler_quarantines_final_failed_sample_and_merges(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import json

    import parallel_stage
    from _common import write_jsonl

    input_path = tmp_path / "prompts.jsonl"
    write_jsonl(input_path, [{"id": "bad", "metadata": {"source_token_length": 32}}])
    output_path = tmp_path / "output.jsonl"
    manifest_path = tmp_path / "manifest.json"
    calls = []

    def always_oom(args, work_root, gpu_ids, *, total_samples):
        del gpu_ids, total_samples
        calls.append(1)
        (work_root / "rank_00").mkdir(parents=True, exist_ok=True)
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(parallel_stage, "_run_workers", always_oom)
    assert parallel_stage.main(
        [
            "--mode", "regenerate",
            "--scheduler", "shared_lease",
            "--queue-max-attempts", "1",
            "--gpu-ids", "0",
            "--input", str(input_path),
            "--output", str(output_path),
            "--manifest", str(manifest_path),
            "--work-root", str(tmp_path / "work"),
            "--target-model-path", "tiny",
            "--max-length", "64",
            "--resume",
        ]
    ) == 0
    assert len(calls) == 1
    skipped = [json.loads(line) for line in output_path.with_name("output.skipped.jsonl").read_text().splitlines()]
    assert skipped[0]["id"] == "bad"
    assert skipped[0]["kind"] == "quarantine"
    quarantine = [
        json.loads(line)
        for line in (tmp_path / "work" / "quarantine.jsonl").read_text().splitlines()
    ]
    assert quarantine[0]["sample_id"] == "bad"


def test_parallel_shared_scheduler_requeues_successful_worker_missing_artifact(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import json

    import parallel_stage
    from _common import write_json, write_jsonl

    input_path = tmp_path / "prompts.jsonl"
    write_jsonl(input_path, [{"id": "s0", "metadata": {"source_token_length": 5}}])
    output_path = tmp_path / "output.jsonl"
    manifest_path = tmp_path / "manifest.json"
    calls = []

    def flaky_workers(args, work_root, gpu_ids, *, total_samples):
        del gpu_ids, total_samples
        calls.append(1)
        root = work_root / "rank_00"
        ids = [json.loads(line)["sample_id"] for line in args._shared_sample_ids_files[0].read_text().splitlines()]
        root.mkdir(parents=True, exist_ok=True)
        write_jsonl(root / "skipped.jsonl", [])
        if len(calls) > 1:
            write_jsonl(root / "output.jsonl", [{"id": sample_id} for sample_id in ids])
        else:
            write_jsonl(root / "output.jsonl", [])
        write_json(root / "manifest.json", {"stats": {}})
        return [root]

    monkeypatch.setattr(parallel_stage, "_run_workers", flaky_workers)
    assert parallel_stage.main(
        [
            "--mode", "regenerate",
            "--scheduler", "shared_lease",
            "--gpu-ids", "0",
            "--input", str(input_path),
            "--output", str(output_path),
            "--manifest", str(manifest_path),
            "--work-root", str(tmp_path / "work"),
            "--target-model-path", "tiny",
            "--max-length", "64",
            "--resume",
        ]
    ) == 0
    assert len(calls) == 2
    assert json.loads(output_path.read_text().splitlines()[0])["id"] == "s0"
