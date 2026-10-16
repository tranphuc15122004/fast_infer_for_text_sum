"""Durable job collection; scientific outcomes never gate independent jobs."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


def identity(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_jobs(jobs, root: Path, capabilities: Mapping[str, Any], execute: Callable,
                 *, resume: bool = False) -> list[dict[str, Any]]:
    """Collect all terminal states. SIGINT/SystemExit propagate to the operator."""
    results: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for job in jobs:
        payload = asdict(job)
        signature = identity(payload)
        job_dir = root / "jobs" / job.id
        checkpoint = job_dir / "result.json"
        previous = json.loads(checkpoint.read_text()) if resume and checkpoint.is_file() else None
        if previous and previous.get("job_identity") != signature:
            raise ValueError(f"job identity changed: {job.id}; use a new run ID")
        if previous and previous.get("status") == "succeeded":
            result = previous
        else:
            missing = [dep for dep in job.dependencies if by_id.get(dep, {}).get("status") != "succeeded"]
            capability = capabilities.get(job.capability, {})
            available = capability.get("available", False) if isinstance(capability, dict) else bool(capability)
            if missing:
                result = {"status": "dependency_missing", "missing_dependencies": missing}
            elif not available:
                result = {"status": "unsupported", "reason": capability.get("reason", f"capability unavailable: {job.capability}") if isinstance(capability, dict) else "capability unavailable"}
            else:
                try:
                    job_dir.mkdir(parents=True, exist_ok=True)
                    atomic_json(root / "progress.json", {"job_id": job.id, "experiment": job.experiment, "variant": job.variant, "completed_jobs": len(results), "planned_jobs": len(jobs)})
                    print(f"[vLLM {len(results) + 1}/{len(jobs)}] {job.experiment} {job.variant}", flush=True)
                    result = dict(execute(job, job_dir))
                    if result.get("status") not in {"succeeded", "failed", "unsupported", "dependency_missing"}:
                        raise ValueError(f"executor returned invalid job status: {result.get('status')}")
                except Exception as exc:
                    result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            result.update({"job_id": job.id, "job_identity": signature, "job": payload})
            atomic_json(checkpoint, result)
        results.append(result)
        by_id[job.id] = result
    atomic_json(root / "progress.json", {"status": "finished", "completed_jobs": len(results), "planned_jobs": len(jobs)})
    return results


def summarize(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = Counter(row.get("status", "failed") for row in results)
    return {"campaign_status": "finished", "execution_status": "complete" if results and counts["succeeded"] == len(results) else "partial",
            "planned": len(results), "attempted": sum(counts[k] for k in ("succeeded", "failed")),
            **{k: counts[k] for k in ("succeeded", "failed", "unsupported", "dependency_missing")}}


def batch_totals(records: Sequence[Mapping[str, Any]], wall_ms: float) -> dict[str, Any]:
    """System throughput uses makespan; latency comes from individual requests."""
    def percentile(values, q):
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)
    latencies = [float(row["e2e_ms"]) for row in records if row.get("e2e_ms") is not None and math.isfinite(float(row["e2e_ms"])) and float(row["e2e_ms"]) > 0]
    valid_wall = math.isfinite(wall_ms) and wall_ms > 0
    return {"makespan_ms": wall_ms, "throughput_tok_s": sum(int(row.get("output_tokens", 0)) for row in records) * 1000 / wall_ms if valid_wall else None,
            "qps": len(records) * 1000 / wall_ms if valid_wall else None,
            "latency_p50_ms": percentile(latencies, .5), "latency_p95_ms": percentile(latencies, .95)}
