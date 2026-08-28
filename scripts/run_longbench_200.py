#!/usr/bin/env python3
"""Orchestrate the canonical LongBench × 9-baseline experiment matrix.

This runner owns experiment selection, deterministic input subsets, preflight
statuses, child-process logs and a manifest.  Baseline implementations remain
in their individual scripts; the runner never silently substitutes one method
for another.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    timed_out = False
    returncode: int | None = None
    log = ""
    try:
        proc = subprocess.run(
            list(command),
            cwd=ROOT,
            env=_safe_env(),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = proc.returncode
        log = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        log = (exc.stdout or "") + (exc.stderr or "")
        returncode = None
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    log_path.write_text(log, encoding="utf-8", errors="replace")
    return {
        "status": "timeout" if timed_out else ("success" if returncode == 0 else "failed"),
        "returncode": returncode,
        "elapsed_ms": elapsed_ms,
        "output_exists": output.is_file(),
        "log": str(log_path),
        "log_tail": log[-2000:],
        "command": [str(part) for part in command],
    }


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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-unsupported", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=os.environ.get("LONG_BENCH_STRICT", "1") == "1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gpu_ids = os.environ.get("LONG_BENCH_GPU_IDS") or os.environ.get("FI_GPU_IDS")
    if gpu_ids is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    cuda_available = _effective_cuda_available()
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
    max_input_tokens = args.max_input_tokens if args.max_input_tokens is not None else _env_int("LONG_BENCH_MAX_INPUT_TOKENS", 0)
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
        "seed": seed,
        "timeout_seconds": timeout_seconds,
        "strict": bool(args.strict),
        "allow_unsupported": bool(args.allow_unsupported),
        "runtime": runtime_metadata(),
        "cells": [],
    }
    _write_json(run_dir / "run_manifest.json", manifest)

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
                print(f"[{baseline}/{dataset}] {status}: {reason}")
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
                print(f"[{baseline}/{dataset}] {check['status']}: {check['reason']}")
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

            print(f"[{baseline}/{dataset}] launching {len(normalized)} sample(s)")
            child = _run_child(
                command,
                output=output_path,
                log_path=run_dir / "logs" / f"{baseline}_{dataset}.log",
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
            print(f"[{baseline}/{dataset}] {child['status']} in {child['elapsed_ms']} ms")

    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["failure_count"] = failures
    manifest["cell_count"] = len(manifest["cells"])
    _write_json(run_dir / "run_manifest.json", manifest)
    print(f"Run manifest: {run_dir / 'run_manifest.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
