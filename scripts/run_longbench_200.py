#!/usr/bin/env python3
"""Orchestrate the canonical LongBench × 9-baseline experiment matrix.

This runner owns experiment selection, deterministic input subsets, preflight
statuses, child-process logs and a manifest.  Baseline implementations remain
in their individual scripts; the runner never silently substitutes one method
for another.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from common import io_util  # noqa: E402
from common.benchmark_data import (  # noqa: E402
    DATASETS,
    read_jsonl,
    select_rows,
    validate_output_dir,
)
from common.benchmark_runtime import (  # noqa: E402
    build_status_record,
    runtime_metadata,
)
from common.data_loader import normalize  # noqa: E402
from common.longbench_adapter import (  # noqa: E402
    BASELINES,
    baseline_config_from_env,
    build_adapter_command,
    convert_records_for_baseline,
    preflight_baseline,
)


def _split(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return value.replace(",", " ").split()
    result: list[str] = []
    for item in value:
        result.extend(str(item).replace(",", " ").split())
    return result


def resolve_profile(
    *, mode: str, cuda_available: bool, allow_unsupported: bool = False
) -> dict[str, Any]:
    """Resolve sample limits and enforce the GPU policy for each profile."""

    if mode not in {"smoke", "representative", "full"}:
        raise SystemExit(f"invalid LongBench mode: {mode}")
    if mode in {"representative", "full"} and not cuda_available and not allow_unsupported:
        raise SystemExit(
            f"LongBench {mode} requires CUDA. Use smoke for CPU preflight or "
            "pass --allow-unsupported to record unavailable cells."
        )
    return {
        "mode": mode,
        "samples": {"smoke": 1, "representative": 20, "full": 200}[mode],
        "max_new_tokens": {"smoke": 8, "representative": 64, "full": 64}[mode],
        "cuda_available": bool(cuda_available),
        "allow_unsupported": bool(allow_unsupported),
    }


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number") from exc


def resolve_max_input_tokens(mode: str, cli_value: int | None) -> int:
    """Resolve the input cap, keeping smoke safe for quadratic attention.

    LongBench smoke is a wiring/runtime check, not a full-length quality run.
    The first canonical ``gov_report`` row is about 10k tokens, which is large
    enough to make the eager HF baseline materialize a multi-GiB attention
    matrix.  An explicit CLI value still wins; ``0`` intentionally disables
    the cap for operators who want to override the smoke safety policy.
    """
    if cli_value is not None:
        if cli_value < 0:
            raise SystemExit("--max-input-tokens must be >= 0")
        return cli_value
    if mode == "smoke":
        value = _env_int("LONG_BENCH_SMOKE_MAX_INPUT_TOKENS", 4096)
    else:
        value = _env_int("LONG_BENCH_MAX_INPUT_TOKENS", 0)
    if value < 0:
        raise SystemExit("LONG_BENCH input-token cap must be >= 0")
    return value


def _resolve(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _effective_cuda_available() -> bool:
    """Respect both driver visibility and an explicit ``DEVICE=cpu`` policy."""
    requested = (os.environ.get("LONG_BENCH_DEVICE") or os.environ.get("FI_DEVICE") or "cuda").lower()
    if requested.startswith("cpu"):
        return False
    return _cuda_available()


def _source_manifest_hash(data_dir: Path) -> str | None:
    manifest = data_dir / "manifest.json"
    if not manifest.is_file():
        return None
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _load_selected(
    data_dir: Path,
    dataset: str,
    count: int,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = data_dir / f"{dataset}.jsonl"
    if not path.is_file():
        raise SystemExit(f"LongBench dataset file not found: {path}")
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit(f"LongBench dataset is empty: {path}")
    if count > len(rows):
        raise SystemExit(f"{dataset}: requested {count}, only {len(rows)} rows exist")
    if count == len(rows):
        selected = [dict(row, length_bin=row.get("length_bin")) for row in rows]
    elif count == 1:
        # A one-row smoke test should be cheap and deterministic; balanced
        # stratification is defined for the 5-bin representative/full counts.
        selected = [dict(rows[0])]
    else:
        try:
            selected = select_rows(rows, dataset=dataset, n=count, seed=seed)
        except ValueError as exc:
            raise SystemExit(
                f"{dataset}: profile count {count} cannot be selected from the "
                "canonical 5-bin layout; choose a positive multiple of 5"
            ) from exc
    normalized = [normalize(row, i) for i, row in enumerate(selected)]
    return selected, normalized


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _write_status_file(
    path: Path,
    *,
    baseline: str,
    dataset: str,
    records: Sequence[Mapping[str, Any]],
    status: str,
    reason: str,
    model: str | None,
    config: Mapping[str, Any],
    run_id: str,
) -> int:
    writer = io_util.JsonlWriter(path)
    for sample in records:
        row = build_status_record(
            method=baseline,
            dataset=dataset,
            sample_id=sample["id"],
            status=status,
            reason=reason,
            model=model,
            config=config,
        )
        row.update(
            run_id=run_id,
            task_type=sample.get("raw", {}).get("task_type"),
        )
        writer.add(row)
    writer.finalize(
        {
            "type": "summary",
            "method": baseline,
            "dataset": dataset,
            "run_id": run_id,
            "status": status,
            "reason": reason,
            "preflight_only": status == "preflight_only",
            "num_samples": len(records),
            "successful_samples": 0,
        }
    )
    return len(records)


def _safe_env() -> dict[str, str]:
    """Child environment with the shared Python path and selected GPU IDs."""
    env = dict(os.environ)
    # Baseline output must reach the parent while inference is running.  This
    # applies to Python-based adapters and is harmless for other child tools.
    env["PYTHONUNBUFFERED"] = "1"
    scripts = str(ROOT / "scripts")
    env["PYTHONPATH"] = scripts + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    gpu_ids = env.get("LONG_BENCH_GPU_IDS") or env.get("FI_GPU_IDS")
    if gpu_ids is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    return env


def _run_child(
    command: Sequence[str],
    *,
    output: Path,
    log_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run one baseline while teeing its combined output to log and console.

    ``subprocess.run(capture_output=True)`` delayed the log file until the
    baseline exited, which made long inference runs impossible to monitor.
    A reader thread drains the pipe continuously while the parent keeps the
    existing bounded timeout around ``wait``.  The raw child output is kept in
    the per-cell log; the console copy is prefixed with the log stem so output
    from sequential cells remains attributable to a baseline/dataset.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    timed_out = False
    returncode: int | None = None
    log_tail = ""

    # Create the file before spawning the child so operators can tail it as
    # soon as the cell is launched, even before its first output line.
    log_handle = log_path.open("w", encoding="utf-8", buffering=1)
    try:
        proc = subprocess.Popen(
            list(command),
            cwd=ROOT,
            env=_safe_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except BaseException:
        log_handle.close()
        raise

    def _stream_output() -> None:
        nonlocal log_tail
        assert proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                try:
                    chunk = os.read(proc.stdout.fileno(), 4096)
                except OSError:
                    break
                if not chunk:
                    break
                text = decoder.decode(chunk)
                if not text:
                    continue
                log_tail = (log_tail + text)[-2000:]
                log_handle.write(text)
                log_handle.flush()
                print(f"[{log_path.stem}] {text}", end="", flush=True)
            remainder = decoder.decode(b"", final=True)
            if remainder:
                log_tail = (log_tail + remainder)[-2000:]
                log_handle.write(remainder)
                log_handle.flush()
                print(f"[{log_path.stem}] {remainder}", end="", flush=True)
        finally:
            proc.stdout.close()

    reader = threading.Thread(
        target=_stream_output,
        name=f"longbench-log-{log_path.stem}",
        daemon=True,
    )
    reader.start()
    try:
        try:
            returncode = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.wait()
            returncode = None
    finally:
        # A normal child closes stdout promptly.  On timeout, close the pipe
        # after killing the child so a descendant that inherited stdout cannot
        # keep the logging thread alive indefinitely.
        reader.join(timeout=5 if not timed_out else 1)
        if reader.is_alive() and proc.stdout is not None:
            proc.stdout.close()
            reader.join(timeout=1)
        log_handle.flush()
        log_handle.close()

    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    return {
        "status": "timeout" if timed_out else ("success" if returncode == 0 else "failed"),
        "returncode": returncode,
        "elapsed_ms": elapsed_ms,
        "output_exists": output.is_file(),
        "log": str(log_path),
        "log_tail": log_tail,
        "command": [str(part) for part in command],
    }


def _run_collector(
    run_dir: Path,
    data_dir: Path,
    *,
    baselines: Sequence[str],
    datasets: Sequence[str],
    expected_samples: int,
    strict: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run the metric collector over a finished run directory.

    The collector runs as a child process with the same interpreter and env as
    the orchestrator so aggregation never shares process state with model
    runs, and its output lands in ``run_dir/metrics_summary.{json,csv,md}``
    with a log under ``run_dir/logs/``.  In ``strict`` mode the collector also
    validates that every (baseline, dataset) pair produced the expected number
    of successful samples.  Aggregation is best-effort reporting on top of the
    raw JSONL: a failure here never invalidates the cell outputs already
    written, but it is recorded in ``run_manifest.json`` under ``aggregate``.
    """
    out_path = run_dir / "metrics_summary.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "collect_metrics.py"),
        "--outputs-dir",
        str(run_dir),
        "--data-dir",
        str(data_dir),
        "--out",
        str(out_path),
        "--csv",
        str(run_dir / "metrics_summary.csv"),
        "--md",
        str(run_dir / "metrics_summary.md"),
    ]
    if strict:
        command += [
            "--strict",
            "--expected-baselines",
            " ".join(baselines),
            "--expected-datasets",
            " ".join(datasets),
            "--expected-samples",
            str(expected_samples),
        ]
    result = _run_child(
        command,
        output=out_path,
        log_path=run_dir / "logs" / "collect_metrics.log",
        timeout_seconds=timeout_seconds,
    )
    result["strict"] = bool(strict)
    result["output_files"] = {
        "json": str(out_path),
        "csv": str(run_dir / "metrics_summary.csv"),
        "md": str(run_dir / "metrics_summary.md"),
    }
    return result


_TIMING_FIELDS = (
    "model_load_ms",
    "prefill_ms",
    "ttft_ms",
    "decode_ms",
    "tpot_ms",
    "e2e_ms",
    "throughput_tok_s",
    "decode_throughput_tok_s",
    "qps",
    "peak_memory_gb",
)


def _normalize_child_output(
    path: Path,
    *,
    baseline: str,
    dataset: str,
    source_records: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    run_id: str,
) -> int:
    """Normalize upstream JSONL fields in-place after a successful child run."""
    if not path.is_file():
        return 0
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return 0
    by_id = {str(row["id"]): row for row in source_records if row.get("id") is not None}
    normalized_rows: list[dict[str, Any]] = []
    observations = 0
    for original in rows:
        row = dict(original)
        if row.get("type") == "summary":
            row.update(method=baseline, dataset=dataset, run_id=run_id)
            normalized_rows.append(row)
            continue

        upstream_method = row.get("method")
        if upstream_method and upstream_method != baseline:
            row["upstream_method"] = upstream_method
        row["method"] = baseline
        row["dataset"] = dataset
        row["run_id"] = run_id
        row.setdefault("status", "success")
        row.setdefault("model", config.get("model"))
        row.setdefault("batch_size", 1)
        row.setdefault("device", config.get("device"))
        row.setdefault("dtype", config.get("dtype"))
        row.setdefault("seed", config.get("seed"))
        row.setdefault("temperature", config.get("temperature"))
        row.setdefault("max_new_tokens", config.get("max_new_tokens"))
        row.setdefault("warmup_runs", config.get("warmup_runs"))

        if row.get("sample_id") is None and row.get("question_id") is not None:
            row["sample_id"] = row["question_id"]
        if row.get("output_tokens") is None and row.get("new_tokens") is not None:
            row["output_tokens"] = row["new_tokens"]
        if row.get("text") is None and isinstance(row.get("answer"), str):
            row["text"] = row["answer"]
        if row.get("decode_ms") is None and row.get("eagle_time") is not None:
            row["decode_ms"] = round(float(row["eagle_time"]) * 1000.0, 3)
        if row.get("throughput_tok_s") is None and row.get("eagle_tok_s") is not None:
            row["throughput_tok_s"] = row["eagle_tok_s"]
        if row.get("dense_decode_ms") is None and row.get("naive_time") is not None:
            row["dense_decode_ms"] = round(float(row["naive_time"]) * 1000.0, 3)
        if row.get("eagle_time") is not None:
            # EAGLE's upstream timer explicitly excludes prefill.  Keep that
            # fact visible instead of calling decode-only time "E2E".
            row.setdefault("measurement_scope", "decode_only")

        sample_id = row.get("sample_id")
        source = by_id.get(str(sample_id)) if sample_id is not None else None
        if source:
            row.setdefault("reference_output", source.get("reference_output"))
            row.setdefault("task_type", source.get("task_type"))
        if row.get("task_type") is None and source_records:
            row["task_type"] = source_records[0].get("task_type")
        if row.get("reference_output") is None and source:
            row["reference_output"] = source.get("reference_output")

        aggregate = row.get("scope") == "aggregate" or row.get("sample_id") is None
        row["scope"] = "aggregate" if aggregate else "sample"
        if aggregate and not row.get("sample_ids"):
            row["sample_ids"] = [source["id"] for source in source_records]

        # Code-completion output must not carry summarization metrics from an
        # upstream helper.  The collector computes exact/edit scores from the
        # normalized text/reference pair.
        if row.get("task_type") == "code_completion":
            for key in list(row):
                if key.startswith(("rouge", "bleu")) or key == "length_ratio":
                    row.pop(key, None)
        for field in _TIMING_FIELDS:
            row.setdefault(field, None)
        normalized_rows.append(row)
        observations += 1

    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized_rows),
        encoding="utf-8",
    )
    return observations


# ---------------------------------------------------------------------------
# GPU selection & inventory helpers
# ---------------------------------------------------------------------------

def _gpu_selection_raw() -> str | None:
    """Return the effective GPU-id request from the environment, if any."""
    for name in ("LONG_BENCH_GPU_IDS", "FI_GPU_IDS", "CUDA_VISIBLE_DEVICES"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _parse_gpu_ids(value: str | None) -> list[int] | None:
    """Parse a comma/space separated GPU id list into validated integers."""
    if value is None or not str(value).strip():
        return None
    parts = [part for part in str(value).replace(",", " ").split() if part]
    try:
        ids = [int(part) for part in parts]
    except ValueError as exc:
        raise SystemExit(
            f"invalid GPU ids {value!r}: expected device indices such as "
            "'0', '2' or '0,1'"
        ) from exc
    if any(index < 0 for index in ids):
        raise SystemExit(f"invalid GPU ids {value!r}: indices must be >= 0")
    return ids


def _device_policy() -> str:
    return (
        os.environ.get("LONG_BENCH_DEVICE")
        or os.environ.get("FI_DEVICE")
        or "cuda"
    ).lower()


def _nvidia_smi_gpus() -> list[dict[str, Any]] | None:
    """Physical GPU inventory via nvidia-smi (immune to CUDA_VISIBLE_DEVICES).

    Returns None when nvidia-smi is absent or fails; the caller then falls
    back to what torch reports for the visible subset.
    """
    query = (
        "index,name,memory.total,memory.free,memory.used,"
        "utilization.gpu,compute_cap"
    )
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None

    def _gib(value: str) -> float | None:
        try:
            return round(float(value) / 1024.0, 1)  # MiB -> GiB
        except ValueError:
            return None

    gpus: list[dict[str, Any]] = []
    for line in out.stdout.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 7:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        util_raw = fields[5]
        gpus.append(
            {
                "index": index,
                "name": fields[1],
                "total_memory_gb": _gib(fields[2]),
                "free_memory_gb": _gib(fields[3]),
                "used_memory_gb": _gib(fields[4]),
                "utilization_percent": int(util_raw) if util_raw.isdigit() else None,
                "compute_capability": fields[6] or None,
            }
        )
    return gpus or None


def _torch_visible_gpus() -> dict[str, Any]:
    """Describe the CUDA-visible device subset as seen by torch."""
    result: dict[str, Any] = {"torch_available": False}
    try:
        import torch
    except Exception:
        return result
    result["torch_available"] = True
    if not torch.cuda.is_available():
        result["cuda_available"] = False
        result["visible_gpu_count"] = 0
        result["visible_gpus"] = []
        return result
    result["cuda_available"] = True
    result["cuda_version"] = torch.version.cuda
    visible: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        visible.append(
            {
                "visible_index": index,
                "name": str(props.name),
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / (1024**3), 1),
            }
        )
    result["visible_gpu_count"] = len(visible)
    result["visible_gpus"] = visible
    return result


def describe_gpu_assignment() -> dict[str, Any]:
    """Return a JSON-safe snapshot of host GPUs, requested ids and visibility."""
    requested_raw = _gpu_selection_raw()
    host = _nvidia_smi_gpus() or []
    report: dict[str, Any] = {
        "requested": requested_raw,
        "requested_ids": _parse_gpu_ids(requested_raw),
        "device_policy": _device_policy(),
        "host_gpu_count": len(host) or None,
        "host_gpus": host,
    }
    report.update(_torch_visible_gpus())
    return report


def _missing_requested_gpus(report: Mapping[str, Any]) -> list[int]:
    requested = report.get("requested_ids") or []
    host = report.get("host_gpus") or []
    if not requested or not host:
        return []
    present = {gpu["index"] for gpu in host}
    return [index for index in requested if index not in present]


def gpu_memory_guard_reason(
    report: Mapping[str, Any], *, min_free_gb: float
) -> str | None:
    """Return an actionable reason when the selected GPU is already crowded.

    ``nvidia-smi`` reports physical indices while torch uses the selected
    ``CUDA_VISIBLE_DEVICES`` subset.  The runner has already applied that
    mapping, so compare physical ids here before any model child is spawned.
    If no inventory is available, leave the decision to the child process
    rather than falsely claiming that the GPU is safe or unsafe.
    """
    if min_free_gb <= 0:
        return None
    host_gpus = list(report.get("host_gpus") or [])
    if not host_gpus:
        return None
    requested = list(report.get("requested_ids") or [])
    selected = requested or [int(host_gpus[0]["index"])]
    by_index = {int(gpu["index"]): gpu for gpu in host_gpus}
    failures: list[str] = []
    for index in selected:
        gpu = by_index.get(int(index))
        if gpu is None:
            failures.append(f"GPU {index}: not present on host")
            continue
        free = gpu.get("free_memory_gb")
        if free is None:
            failures.append(f"GPU {index}: free VRAM is unavailable")
        elif float(free) < min_free_gb:
            failures.append(f"GPU {index}: {float(free):.1f} GiB free")
    if not failures:
        return None
    return (
        f"selected GPU(s) do not meet the {min_free_gb:.1f} GiB free-VRAM "
        f"guard ({'; '.join(failures)}). Stop other GPU processes or select a "
        "different GPU; override with --min-free-gb 0 only if intentional."
    )


def print_gpu_summary(report: Mapping[str, Any], *, effective_cuda: bool) -> None:
    """Print a one-line GPU assignment banner for a normal run."""
    requested = report.get("requested_ids") or []
    host_count = report.get("host_gpu_count")
    parts: list[str] = []
    if host_count:
        parts.append(f"host has {host_count} GPU(s)")
    parts.append(
        "selected GPU " + (", ".join(map(str, requested)) if requested else "(auto)")
    )
    if effective_cuda:
        visible = report.get("visible_gpus") or []
        names = ", ".join(gpu["name"] for gpu in visible)
        parts.append(f"torch sees {report.get('visible_gpu_count', 0)} device(s) {names}")
    else:
        policy = report.get("device_policy") or "cuda"
        if policy.startswith("cpu"):
            parts.append(f"device policy {policy!r} -> CPU compute")
        else:
            parts.append("torch CUDA unavailable -> CPU compute")
    print("[gpu] " + " | ".join(parts))
    missing = _missing_requested_gpus(report)
    if missing:
        print(
            f"[gpu] WARNING requested GPU id(s) {missing} not found on the host; "
            "they will be invisible to CUDA",
            file=sys.stderr,
        )


def print_gpu_inventory(report: Mapping[str, Any]) -> None:
    """Print a human-readable GPU inventory and the effective mapping."""
    print("\nHost GPU inventory (physical indices, nvidia-smi):")
    host = report.get("host_gpus") or []
    if not host:
        print("  (no nvidia-smi data available)")
    for gpu in host:
        fields = [
            f"GPU {gpu['index']}: {gpu['name']}",
            f"{gpu.get('total_memory_gb')} GB total",
            f"{gpu.get('free_memory_gb')} GB free",
        ]
        util = gpu.get("utilization_percent")
        if util is not None:
            fields.append(f"{util}% util")
        if gpu.get("compute_capability"):
            fields.append(f"cap {gpu['compute_capability']}")
        print("  " + " | ".join(fields))

    requested = report.get("requested_ids") or []
    if requested:
        print(f"\nRequested GPU ids: {', '.join(map(str, requested))}")
    else:
        print("\nRequested GPU ids: (unset - torch default visibility)")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    print("\nTorch visibility:")
    if not report.get("torch_available"):
        print("  torch is not importable in this interpreter")
    elif not report.get("cuda_available"):
        print("  torch.cuda.is_available() = False (no visible CUDA device)")
    else:
        for gpu in report.get("visible_gpus") or []:
            print(
                "  "
                f"visible {gpu['visible_index']} -> {gpu['name']} "
                f"({gpu['total_memory_gb']} GB, cap {gpu['compute_capability']})"
            )
    missing = _missing_requested_gpus(report)
    if missing:
        print(
            f"\nWARNING: requested GPU id(s) {missing} are not present on the host "
            "and will be invisible to CUDA."
        )
    policy = report.get("device_policy") or "cuda"
    if policy.startswith("cpu"):
        print(f"\nNote: device policy is {policy!r} -> inference will run on CPU.")
    elif not report.get("cuda_available"):
        print("\nNote: no CUDA device is visible to torch -> inference will run on CPU.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "representative", "full"], default=os.environ.get("LONG_BENCH_MODE", "smoke"))
    parser.add_argument("--baselines", default=os.environ.get("LONG_BENCH_BASELINES", " ".join(BASELINES)))
    parser.add_argument("--datasets", default=os.environ.get("LONG_BENCH_DATASETS", " ".join(DATASETS)))
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("LONG_BENCH_DATA_DIR", "data/longbench_200"))
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("LONG_BENCH_OUTPUT_DIR", "outputs/longbench_200"))
    parser.add_argument("--model", default=os.environ.get("LONG_BENCH_MODEL") or os.environ.get("MODEL_TARGET"))
    parser.add_argument("--max-samples", "--samples-per-dataset", dest="max_samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="minimum free VRAM before launching children (default: 32; 0 disables)",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-unsupported", action="store_true")
    parser.add_argument(
        "--gpu-ids",
        dest="gpu_ids",
        default=None,
        help="physical GPU id(s) to run on, e.g. '0', '2' or '0,1'; "
        "overrides LONG_BENCH_GPU_IDS / FI_GPU_IDS / CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--list-gpus",
        action="store_true",
        help="list host GPUs, the current selection and torch visibility, then exit",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=os.environ.get("LONG_BENCH_STRICT", "1") == "1")
    parser.add_argument(
        "--collect",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LONG_BENCH_COLLECT", "1") == "1",
        help="run the metric collector over the finished run and write "
        "metrics_summary.{json,csv,md} into the run directory (default: on; "
        "always skipped for --preflight-only runs)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    # GPU selection: --gpu-ids is authoritative; otherwise honour the existing
    # LONG_BENCH_GPU_IDS / FI_GPU_IDS / CUDA_VISIBLE_DEVICES chain. Applied
    # before torch is imported so enumeration and every child process observe
    # the same physical device(s). An explicit empty CUDA_VISIBLE_DEVICES
    # (CPU smoke) is left untouched.
    if args.gpu_ids is not None and args.gpu_ids.strip():
        _parse_gpu_ids(args.gpu_ids)  # fail fast on malformed values
        os.environ["LONG_BENCH_GPU_IDS"] = args.gpu_ids
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        selection = _gpu_selection_raw()
        if selection:
            os.environ["CUDA_VISIBLE_DEVICES"] = selection

    gpu_report = describe_gpu_assignment()
    if args.list_gpus:
        print_gpu_inventory(gpu_report)
        return 0

    cuda_available = _effective_cuda_available()
    print_gpu_summary(gpu_report, effective_cuda=cuda_available)
    profile = resolve_profile(
        mode=args.mode,
        cuda_available=cuda_available,
        allow_unsupported=args.allow_unsupported,
    )

    data_dir = _resolve(args.data_dir)
    output_root = _resolve(args.output_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"LongBench data directory not found: {data_dir}")
    try:
        # Validate the source before launching any model.  The canonical set is
        # always 200 rows/dataset; this also verifies checksums and task types.
        validate_output_dir(data_dir, expected_count=200)
    except (ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid canonical LongBench data: {exc}") from exc

    baselines = _split(args.baselines)
    datasets = _split(args.datasets)
    unknown_baselines = sorted(set(baselines) - set(BASELINES))
    unknown_datasets = sorted(set(datasets) - set(DATASETS))
    if unknown_baselines:
        raise SystemExit(f"Unknown baseline(s): {', '.join(unknown_baselines)}")
    if unknown_datasets:
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown_datasets)}")
    if not baselines or not datasets:
        raise SystemExit("At least one baseline and dataset are required")

    if args.mode == "representative" and args.datasets == " ".join(DATASETS):
        configured = _split(os.environ.get("LONG_BENCH_REPRESENTATIVE_DATASETS", "gov_report lcc"))
        if configured:
            datasets = configured
    sample_count = args.max_samples or {
        "smoke": _env_int("LONG_BENCH_SMOKE_SAMPLES", 1),
        "representative": _env_int("LONG_BENCH_REPRESENTATIVE_SAMPLES", 20),
        "full": _env_int("LONG_BENCH_FULL_SAMPLES", 200),
    }[args.mode]
    max_new_tokens = args.max_new_tokens or {
        "smoke": _env_int("LONG_BENCH_SMOKE_MAX_NEW_TOKENS", 8),
        "representative": _env_int("LONG_BENCH_MAX_NEW_TOKENS", 64),
        "full": _env_int("LONG_BENCH_MAX_NEW_TOKENS", 64),
    }[args.mode]
    seed = args.seed if args.seed is not None else _env_int("LONG_BENCH_SEED", 42)
    temperature = args.temperature if args.temperature is not None else float(os.environ.get("LONG_BENCH_TEMPERATURE", "0"))
    warmup_runs = args.warmup_runs if args.warmup_runs is not None else _env_int("LONG_BENCH_WARMUP_RUNS", 3)
    max_input_tokens = resolve_max_input_tokens(args.mode, args.max_input_tokens)
    min_free_gb = (
        args.min_free_gb
        if args.min_free_gb is not None
        else _env_float("LONG_BENCH_MIN_FREE_GB", 32.0)
    )
    if min_free_gb < 0:
        raise SystemExit("--min-free-gb/LONG_BENCH_MIN_FREE_GB must be >= 0")
    gpu_guard_reason = gpu_memory_guard_reason(
        gpu_report, min_free_gb=min_free_gb
    ) if cuda_available else None
    if gpu_guard_reason and not args.preflight_only:
        print(f"[gpu] VRAM guard: {gpu_guard_reason}", file=sys.stderr)
        return 2
    timeout_seconds = _env_int("LONG_BENCH_TIMEOUT_SECONDS", 900)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    manifest: dict[str, Any] = {
        "schema_version": "longbench-run-v1",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "preflight_only": bool(args.preflight_only),
        "data_dir": str(data_dir),
        "output_dir": str(run_dir),
        "source_manifest_sha256": _source_manifest_hash(data_dir),
        "model": args.model,
        "baselines": baselines,
        "datasets": datasets,
        "sample_count": sample_count,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "warmup_runs": warmup_runs,
        "max_input_tokens": max_input_tokens,
        "min_free_gb": min_free_gb,
        "gpu_guard_reason": gpu_guard_reason,
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "strict": bool(args.strict),
        "allow_unsupported": bool(args.allow_unsupported),
        "gpu_ids": gpu_report.get("requested"),
        "gpu": gpu_report,
        "runtime": runtime_metadata(),
        "cells": [],
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    print(f"Run directory: {run_dir}", flush=True)
    print(
        f"Run manifest (live): {run_dir / 'run_manifest.json'}",
        flush=True,
    )

    failures = 0
    for dataset in datasets:
        source_rows, normalized = _load_selected(data_dir, dataset, sample_count, seed=seed)
        subset_path = run_dir / "inputs" / f"{dataset}.jsonl"
        _write_jsonl(subset_path, source_rows)
        for baseline in baselines:
            output_path = run_dir / baseline / f"{dataset}.jsonl"
            cfg = baseline_config_from_env(baseline)
            cfg.update(
                model=args.model or cfg.get("model"),
                device=os.environ.get("LONG_BENCH_DEVICE", cfg.get("device", "cuda")),
                temperature=temperature,
                warmup_runs=warmup_runs,
                max_input_tokens=max_input_tokens,
                seed=seed,
                smoke=args.mode == "smoke",
                max_new_tokens=max_new_tokens,
            )
            check = preflight_baseline(
                baseline,
                config=cfg,
                cuda_available=cuda_available,
            )
            cell: dict[str, Any] = {
                "baseline": baseline,
                "dataset": dataset,
                "sample_count": len(normalized),
                "preflight": check,
                "output": str(output_path),
            }
            if args.preflight_only:
                status = check["status"] if check["status"] != "ready" else "preflight_only"
                reason = check["reason"] or "preflight completed; inference was not requested"
                _write_status_file(
                    output_path,
                    baseline=baseline,
                    dataset=dataset,
                    records=normalized,
                    status=status,
                    reason=reason,
                    model=cfg.get("model"),
                    config=cfg,
                    run_id=run_id,
                )
                cell.update(status=status, reason=reason, returncode=0)
                manifest["cells"].append(cell)
                print(f"[{baseline}/{dataset}] {status}: {reason}", flush=True)
                continue

            if check["status"] not in {"ready", "aggregate_only"}:
                if args.strict and not args.allow_unsupported and args.mode != "smoke":
                    failures += 1
                _write_status_file(
                    output_path,
                    baseline=baseline,
                    dataset=dataset,
                    records=normalized,
                    status=check["status"],
                    reason=check["reason"] or "baseline preflight did not pass",
                    model=cfg.get("model"),
                    config=cfg,
                    run_id=run_id,
                )
                cell.update(status=check["status"], reason=check["reason"])
                manifest["cells"].append(cell)
                print(
                    f"[{baseline}/{dataset}] {check['status']}: {check['reason']}",
                    flush=True,
                )
                continue

            converted = run_dir / "inputs" / f"{baseline}_{dataset}.jsonl"
            converted_input = convert_records_for_baseline(baseline, normalized, converted)
            command = build_adapter_command(
                baseline,
                data_file=subset_path,
                converted_input=converted_input,
                output=output_path,
                max_samples=len(normalized),
                max_new_tokens=max_new_tokens,
                config=cfg,
            )
            if command is None:
                reason = "adapter did not produce a command for this dataset"
                _write_status_file(
                    output_path,
                    baseline=baseline,
                    dataset=dataset,
                    records=normalized,
                    status="unsupported_dataset",
                    reason=reason,
                    model=cfg.get("model"),
                    config=cfg,
                    run_id=run_id,
                )
                cell.update(status="unsupported_dataset", reason=reason)
                manifest["cells"].append(cell)
                continue

            live_log = run_dir / "logs" / f"{baseline}_{dataset}.log"
            print(
                f"[{baseline}/{dataset}] launching {len(normalized)} sample(s)\n"
                f"[{baseline}/{dataset}] live log: {live_log}",
                flush=True,
            )
            child = _run_child(
                command,
                output=output_path,
                log_path=live_log,
                timeout_seconds=timeout_seconds,
            )
            if child["status"] == "success":
                normalized_count = _normalize_child_output(
                    output_path,
                    baseline=baseline,
                    dataset=dataset,
                    source_records=normalized,
                    config=cfg,
                    run_id=run_id,
                )
                child["normalized_records"] = normalized_count
                if normalized_count == 0:
                    child["status"] = "failed"
                    child["reason"] = "child exited successfully but wrote no result records"
            cell.update(child)
            if child["status"] != "success":
                failures += 1
                if not output_path.is_file():
                    _write_status_file(
                        output_path,
                        baseline=baseline,
                        dataset=dataset,
                        records=normalized,
                        status=child["status"],
                        reason=f"child process failed; see {child['log']}",
                        model=cfg.get("model"),
                        config=cfg,
                        run_id=run_id,
                    )
            manifest["cells"].append(cell)
            print(
                f"[{baseline}/{dataset}] {child['status']} in "
                f"{child['elapsed_ms']} ms",
                flush=True,
            )

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failure_count"] = failures
    manifest["cell_count"] = len(manifest["cells"])

    # Aggregate metrics as part of the run so every finished run ships its own
    # metrics_summary.{json,csv,md}.  Preflight-only runs write status rows
    # instead of inference records, so aggregation is skipped for them.  Strict
    # completeness is only meaningful when every cell actually succeeded;
    # ``failures == 0`` is not enough because smoke and --allow-unsupported
    # runs may record blocked cells without counting them as failures.
    clean_cells = bool(manifest["cells"]) and all(
        cell.get("status") == "success" for cell in manifest["cells"]
    )
    if args.collect and not args.preflight_only:
        aggregate = _run_collector(
            run_dir,
            data_dir,
            baselines=baselines,
            datasets=datasets,
            expected_samples=sample_count,
            strict=bool(args.strict) and clean_cells,
            timeout_seconds=timeout_seconds,
        )
        if aggregate["status"] == "success":
            print(
                "[aggregate] metrics_summary.{json,csv,md} written to "
                f"{run_dir}",
                flush=True,
            )
        else:
            print(
                "[aggregate] collector failed "
                f"(exit {aggregate.get('returncode')}); see {aggregate['log']}",
                file=sys.stderr,
                flush=True,
            )
    elif args.preflight_only:
        aggregate = {
            "status": "skipped",
            "reason": "preflight-only run has no inference records to aggregate",
        }
    else:
        aggregate = {
            "status": "skipped",
            "reason": "metric collection disabled with --no-collect",
        }
    manifest["aggregate"] = aggregate
    _write_json(run_dir / "run_manifest.json", manifest)
    print(f"Run manifest: {run_dir / 'run_manifest.json'}", flush=True)
    strict_aggregate_failed = (
        aggregate.get("status") == "failed" and bool(aggregate.get("strict"))
    )
    return 1 if (failures or strict_aggregate_failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
