#!/usr/bin/env python3
"""Run the preregistered DFlash context-intervention campaign on B200.

This entrypoint is offline by design. It never installs packages or downloads
models/data. The FA4 primary path, model snapshots and LongBench files must be
present on the server before a GPU phase starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT / "src", ROOT / "scripts", ROOT / "externals" / "dflash"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from TrainingFree.context_interventions.resilience import CampaignIntegrityError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("FAST_INFER_MASTER_CONFIG"))
    parser.add_argument(
        "--mode",
        choices=("prepare", "preflight", "validate", "pilot", "dev", "lock", "confirm", "report"),
        default="preflight",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default="outputs/dflash_context_interventions")
    parser.add_argument("--target-model", default=os.environ.get("MODEL_TARGET") or os.environ.get("TARGET_MODEL"))
    parser.add_argument("--draft-model", default=os.environ.get("MODEL_DFLASH_DRAFT") or os.environ.get("DRAFT_MODEL"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--exposure-manifest", action="append", default=[])
    parser.add_argument("--gpu-ids", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--attention-backend", choices=("flash_attention_4", "sdpa"), default="flash_attention_4")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-partial-confirm",
        action="store_true",
        help=(
            "run confirm on every available untouched group when exposure leaves "
            "fewer groups than the preregistered quota; report the reduced cohort"
        ),
    )
    parser.add_argument("--report-phase", choices=("pilot", "dev", "confirm"), default="dev")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--max-prefixes", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--source-anchors", type=int, default=64)
    parser.add_argument("--recent-output", type=int, default=256)
    parser.add_argument("--a3-selector", choices=("recency", "target_parent", "previous_dflash"), default="previous_dflash")
    parser.add_argument("--a3-budget", type=int, default=2048)
    parser.add_argument("--refresh-period", type=int, default=4)
    parser.add_argument("--locked-config", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("b200_%Y%m%dT%H%M%SZ")


def _resolve(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    data_dir_value = args.data_dir or os.environ.get("DATA_INPUT") or os.environ.get("LONG_BENCH_DATA_DIR")
    data_file_value = args.data_file or os.environ.get("DATA_FILE") or os.environ.get("LONG_BENCH_DATA_FILE")
    if data_dir_value and Path(data_dir_value).is_file() and not data_file_value:
        data_file_value, data_dir_value = data_dir_value, None
    data_dir = _resolve(data_dir_value) if data_dir_value else None
    data_file = _resolve(data_file_value) if data_file_value else None
    if data_dir is None and data_file is None:
        # This is the canonical shared directory, not the local development copy.
        data_dir = Path("/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/longbench_100_14k")
    return data_dir, data_file


def _load_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[Path]]:
    from TrainingFree.context_adaptive.benchmark import load_corpus

    data_dir, data_file = _paths(args)
    if bool(data_dir) == bool(data_file):
        raise ValueError("provide exactly one of --data-dir/--data-file or DATA_INPUT")
    if data_file:
        if not data_file.is_file():
            raise FileNotFoundError(data_file)
        source_paths = [data_file]
    else:
        if not data_dir.is_dir():
            raise FileNotFoundError(data_dir)
        source_paths = sorted(data_dir.glob("*.jsonl"))
    samples = load_corpus(data_dir=data_dir, data_file=data_file)
    datasets = {str(sample.get("dataset")) for sample in samples}
    missing = {"gov_report", "multi_news", "qmsum"} - datasets
    if missing:
        raise ValueError(f"campaign corpus missing required datasets: {sorted(missing)}")
    _sample_index(samples)  # Reject ambiguous identities before any GPU work.
    return samples, source_paths


def _exposures(args: argparse.Namespace) -> list[dict[str, Any]]:
    from TrainingFree.context_adaptive.splits import read_exposure_manifest

    return [read_exposure_manifest(_resolve(path) or Path(path)) for path in args.exposure_manifest]


def _run_root(args: argparse.Namespace) -> Path:
    output_root = _resolve(args.output_root) or (ROOT / "outputs/dflash_context_interventions")
    run_id = args.run_id or _utc_run_id()
    args.run_id = run_id
    path = output_root / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _code_signature() -> dict[str, Any]:
    from TrainingFree.context_adaptive.schema import sha256_file

    candidates = [
        *sorted((ROOT / "src/TrainingFree/context_interventions").glob("*.py")),
        *sorted((ROOT / "src/TrainingFree/context_adaptive").glob("*.py")),
        ROOT / "scripts/run_dflash_context_experiments.py",
        ROOT / "scripts/run_dflash_context_experiments.sh",
        ROOT / "scripts/run_dflash_context_A_b200.sh",
        ROOT / "data/dflash_context_A_audit/combined_exposure_manifest.json",
        ROOT / "data/dflash_context_A_audit/exposure_audit.json",
        ROOT / "scripts/common/runtime.sh",
        ROOT / "scripts/common/config.sh",
        ROOT / "scripts/common/vanilla_inference.py",
        *sorted((ROOT / "externals/dflash/dflash").glob("*.py")),
    ]
    files = [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)} for path in candidates if path.is_file()]
    return {"files": files, "sha256": _hash_json(files)}


def _cohort(args: argparse.Namespace, samples: Sequence[Mapping[str, Any]], phase: str, *, require_quota: bool = True):
    from TrainingFree.context_interventions.cohort import build_campaign_cohort

    split_path = str(_resolve(args.split_manifest)) if args.split_manifest else None
    exposures = _exposures(args)
    if not exposures:
        raise ValueError(
            "cohort preparation requires at least one reviewed --exposure-manifest covering "
            "the earlier probe and pilot runs; refusing to create/use an unaudited cohort"
        )
    selected, split_manifest, cohort = build_campaign_cohort(
        samples,
        exposures,
        phase=phase,
        seed=args.seed,
        require_quota=require_quota,
        split_manifest_path=split_path,
    )
    if split_manifest.get("status") != "complete":
        raise ValueError(
            f"source-group split is not complete ({split_manifest.get('status')}); "
            "do not proceed to pilot/dev/confirm"
        )
    return selected, split_manifest, cohort


def _save_cohort(run_root: Path, cohort: Mapping[str, Any], split: Mapping[str, Any]) -> None:
    from TrainingFree.context_adaptive.schema import atomic_json

    cohort_dir = run_root / "cohorts"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    phase = str(cohort["phase"])
    atomic_json(cohort_dir / f"{phase}.json", cohort)
    split_path = cohort_dir / "split_manifest.json"
    if not split_path.exists():
        atomic_json(split_path, split)
    elif _hash_json(json.loads(split_path.read_text(encoding="utf-8"))) != _hash_json(split):
        raise ValueError("run root already has a different split manifest; use a new run id")


def _preflight(args: argparse.Namespace, run_root: Path) -> int:
    from TrainingFree.context_interventions.environment import run_preflight
    from TrainingFree.context_adaptive.schema import atomic_json

    data_dir, data_file = _paths(args)
    result = run_preflight(
        target_model=args.target_model,
        draft_model=args.draft_model,
        data_dir=str(data_dir) if data_dir else None,
        data_file=str(data_file) if data_file else None,
        attention_backend=args.attention_backend,
    )
    result["run_id"] = args.run_id
    result["config_path"] = args.config
    atomic_json(run_root / "preflight.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["status"] == "ready" else 2


def _prepare(args: argparse.Namespace, run_root: Path) -> int:
    from TrainingFree.context_adaptive.schema import atomic_json

    samples, source_paths = _load_samples(args)
    _, split, preview = _cohort(args, samples, "dev", require_quota=False)
    _save_cohort(run_root, preview, split)
    for phase in ("pilot", "dev", "confirm"):
        _, _, cohort = _cohort(args, samples, phase, require_quota=False)
        atomic_json(run_root / "cohorts" / f"{phase}.json", cohort)
    manifest = {
        "schema_version": "dflash-context-interventions.prepare.v1",
        "run_id": args.run_id,
        "source_files": [
            {"path": str(path), "bytes": path.stat().st_size,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in source_paths
        ],
        "record_count": len(samples),
        "split_status": split.get("status"),
        "cohort_status": {phase: json.loads((run_root / "cohorts" / f"{phase}.json").read_text()).get("status")
                          for phase in ("pilot", "dev", "confirm")},
    }
    atomic_json(run_root / "prepare_manifest.json", manifest)
    print(f"Prepared {len(samples)} records; manifests: {run_root / 'cohorts'}", flush=True)
    return 0


def _plan(args: argparse.Namespace) -> int:
    from TrainingFree.context_interventions.matrix import a1_matrix, a2_matrix, a3_matrix
    payload = {
        "mode": args.mode,
        "run_id": args.run_id,
        "runtime": "system Python 3.12, one B200, BF16, batch=1, gamma=15, block=16",
        "attention_backend": args.attention_backend,
        "modal_dependency": False,
        "data_dir": str(_paths(args)[0]) if _paths(args)[0] else None,
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "pilot": {"source_groups": 6, "max_new_tokens": 1024, "a1": a1_matrix("pilot"), "a2": a2_matrix("pilot"), "a3": a3_matrix("pilot")},
        "dev": {"source_groups": 24, "max_new_tokens": 2048, "a1": a1_matrix("dev"), "a2": a2_matrix("dev"), "a3": a3_matrix("dev")},
        "confirm": {"source_groups": 48, "max_new_tokens": 2048, "a1": a1_matrix("confirm"), "a2": a2_matrix("confirm"), "a3": a3_matrix("confirm")},
        "gpu_will_be_used": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


def _probe_gpu_driver(args: argparse.Namespace, gpu_uuid: str) -> tuple[str, str]:
    """A temporary telemetry timeout must not change an established run signature."""
    normalize = lambda value: str(value).removeprefix("GPU-").strip()
    requested = str(os.environ.get("CUDA_VISIBLE_DEVICES", "0")).split(",")[0].strip()
    last_error = "no matching GPU driver record"
    for _attempt in range(2):
        try:
            lines = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True, timeout=10,
            ).stdout.strip().splitlines()
            parsed = [line.split(",", 1) for line in lines]
            selected = next((row for row in parsed if normalize(row[0]) == normalize(gpu_uuid)), None)
            if selected is None and gpu_uuid == "unknown":
                if requested.isdigit() and int(requested) < len(parsed):
                    selected = parsed[int(requested)]
                else:
                    selected = next((row for row in parsed if normalize(row[0]) == normalize(requested)), None)
            if selected is not None and len(selected) == 2 and selected[1].strip():
                return selected[0].strip(), selected[1].strip()
        except (OSError, subprocess.SubprocessError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
    run_root = _run_root(args)
    records = []
    preflight_path = run_root / "preflight.json"
    if preflight_path.is_file():
        for line in json.loads(preflight_path.read_text(encoding="utf-8")).get("nvidia_smi", []):
            fields = str(line).split(",")
            if len(fields) >= 3:
                records.append((fields[1].strip(), fields[2].strip()))
    for saved_uuid, driver in records:
        if (gpu_uuid != "unknown" and normalize(saved_uuid) == normalize(gpu_uuid) and driver):
            print(
                f"[provenance] nvidia-smi telemetry unavailable ({last_error}); "
                f"reuse recorded driver={driver} for the same physical GPU {saved_uuid}",
                flush=True,
            )
            return str(saved_uuid), str(driver)
    raise CampaignIntegrityError(f"cannot establish GPU/driver provenance: {last_error}")


def _runtime(args: argparse.Namespace) -> dict[str, Any]:
    if not args.target_model or not args.draft_model:
        raise ValueError("target/draft model paths must be set by master config or CLI")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CAD_ATTENTION_BACKEND"] = args.attention_backend
    from TrainingFree.context_adaptive.benchmark import load_runtime

    runtime = load_runtime(args.target_model, args.draft_model, load_draft=True)
    draft_block_size = int(runtime["draft"].block_size)
    if draft_block_size != 16:
        raise RuntimeError(f"campaign requires the DFlash b16 checkpoint (block_size=16), got {draft_block_size}")
    if int(runtime["target"].config.num_hidden_layers) != 36:
        raise RuntimeError("campaign target scoring layers [1,9,17,25,33] require the 36-layer Qwen3-4B target")
    if tuple(int(value) for value in runtime["draft"].target_layer_ids) != (1, 9, 17, 25, 33):
        raise RuntimeError(
            "DFlash target_layer_ids differ from the preregistered mapping [1,9,17,25,33]: "
            f"{runtime['draft'].target_layer_ids}"
        )
    signature = dict(runtime["runtime_signature"])
    signature["cuda_build"] = runtime["torch"].version.cuda
    signature["visible_gpu_ids"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_properties = runtime["torch"].cuda.get_device_properties(0)
    signature["gpu_name"] = str(device_properties.name)
    signature["gpu_capability"] = list(runtime["torch"].cuda.get_device_capability(0))
    physical_uuid = str(getattr(device_properties, "uuid", "unknown") or "unknown")
    signature["gpu_uuid"], signature["driver_version"] = _probe_gpu_driver(args, physical_uuid)
    signature.pop("signature", None)
    signature["signature"] = _hash_json(signature)
    runtime["runtime_signature"] = signature
    return runtime


def _reference_trace_path(run_root: Path, sample_id: str, *, phase: str) -> Path:
    key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
    return run_root / "traces" / phase / f"{key}.json"


def _replay_reference_ids(trace: Mapping[str, Any]) -> list[int]:
    """Return the full-DFlash trajectory paired with the saved adapter rounds.

    The adapter trace owns round indices and accepted-candidate counts. AR is a
    diagnostic trajectory and may diverge from native/full-adapter DFlash.
    """
    native_ids = trace.get("native_ids")
    adapter_ids = trace.get("adapter_ids")
    if native_ids is None or adapter_ids is None:
        raise ValueError("reference trace is missing native or adapter output IDs")
    native_ids = [int(token) for token in native_ids]
    adapter_ids = [int(token) for token in adapter_ids]
    if native_ids != adapter_ids:
        raise ValueError("native full DFlash and adapter IDs differ")
    parity = trace.get("parity")
    if not isinstance(parity, Mapping) or parity.get("status") != "pass" or not parity.get("native_equals_adapter"):
        raise ValueError("reference trace does not contain passing native/full-adapter parity evidence")
    if trace.get("replay_reference_trajectory", "adapter_ids") != "adapter_ids":
        raise ValueError("reference trace names an unsupported replay trajectory")
    return adapter_ids


def _reference_result_error(measured) -> str | None:
    from TrainingFree.context_adaptive.schema import _check_finite

    invalid = _check_finite(measured["parity"])
    return f"nonfinite reference values: {invalid}" if invalid else None


def _run_parity_cohort(
    args: argparse.Namespace,
    run_root: Path,
    runtime: Mapping[str, Any],
    samples: Sequence[dict[str, Any]],
    cohort: Mapping[str, Any],
    *,
    max_new_tokens: int,
) -> tuple[bool, list[dict[str, Any]]]:
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.context_interventions.execution import run_reference_pair
    from TrainingFree.context_interventions.resilience import FailureStreak, run_compute_unit

    traces: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    code_sig = _code_signature()
    runtime_sig = runtime["runtime_signature"].get("signature")
    failures = FailureStreak()
    for index, sample in enumerate(samples, start=1):
        sample_id = str(sample["id"])
        trace_path = _reference_trace_path(run_root, sample_id, phase=str(cohort["phase"]))
        trace_signature = _hash_json({
            "sample_id": sample_id, "run_id": args.run_id, "max_new_tokens": max_new_tokens,
            "runtime": runtime_sig, "code": code_sig["sha256"], "attention_backend": args.attention_backend,
            "source_anchors": args.source_anchors, "recent_output": args.recent_output,
        })
        if str(cohort["phase"]) == "pilot" and not trace_path.is_file():
            validated_path = _reference_trace_path(run_root, sample_id, phase="validate")
            if validated_path.is_file():
                validated_trace = json.loads(validated_path.read_text(encoding="utf-8"))
                if (validated_trace.get("trace_signature") == trace_signature
                        and validated_trace.get("parity", {}).get("status") == "pass"):
                    atomic_json(trace_path, validated_trace)
                    print(f"[parity] reuse validated trace for {sample['dataset']}:{sample_id}", flush=True)
        if (args.resume or str(cohort["phase"]) == "pilot") and trace_path.is_file():
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            if trace.get("trace_signature") != trace_signature:
                raise ValueError(f"resume reference trace signature mismatch: {trace_path}")
        else:
            print(f"[parity {index}/{len(samples)}] {sample['dataset']}:{sample_id} start", flush=True)
            measured, compute_error = run_compute_unit(
                lambda: run_reference_pair(
                    runtime, sample, max_new_tokens=max_new_tokens,
                    source_anchors=args.source_anchors, recent_output=args.recent_output,
                    require_ar_match=False,
                ),
                label=f"{cohort['phase']} reference {sample['dataset']}:{sample_id}",
                result_error=_reference_result_error,
            )
            if compute_error is not None:
                error_row = {
                    "sample_id": sample_id, "dataset": str(sample["dataset"]),
                    "source_group_id": str(sample["_cad_source_group_id"]),
                    **compute_error,
                }
                results.append(error_row)
                atomic_json(
                    run_root / "evidence" / f"reference_error_{cohort['phase']}_{hashlib.sha256(sample_id.encode()).hexdigest()}.json",
                    error_row,
                )
                failures.observe(failed=True, label=sample_id)
                measured = None
                continue
            trace = {
                "schema_version": "dflash-context-interventions.trace.v1",
                "trace_signature": trace_signature,
                "sample_id": sample_id,
                "dataset": str(sample["dataset"]),
                "source_group_id": str(sample["_cad_source_group_id"]),
                "split": str(sample.get("_cad_split", "unknown")),
                "prompt_tokens": int(measured["input_ids"].shape[1]),
                "prompt_hash": measured["layout"].prompt_hash,
                "source_available": measured["layout"].source_available,
                "source_reason": measured["layout"].source_reason,
                "ar_ids": measured["ar_ids"],
                "native_ids": measured["native_ids"],
                "adapter_ids": measured["adapter_ids"],
                "replay_reference_trajectory": "adapter_ids",
                "adapter_rounds": measured["adapter_rounds"],
                "eos_ids": sorted(measured["eos_ids"]),
                "parity": measured["parity"],
            }
            atomic_json(trace_path, trace)
            del measured
        traces.append(trace)
        failures.observe(failed=False, label=sample_id)
        results.append({
            "sample_id": sample_id,
            "dataset": str(sample["dataset"]),
            "source_group_id": str(sample["_cad_source_group_id"]),
            "status": trace["parity"]["status"],
            **trace["parity"],
        })
        print(
            f"[parity {index}/{len(samples)}] {sample['dataset']}:{sample_id} "
            f"status={trace['parity']['status']} AR={trace['parity']['ar_output_tokens']} tokens",
            flush=True,
        )
    evidence = {
        "schema_version": "dflash-context-interventions.parity.v1",
        "run_id": args.run_id,
        "status": (
            "fail" if not traces or any(row["status"] == "fail" for row in results)
            else "partial_with_errors" if any(row["status"] == "error" for row in results)
            else "pass"
        ),
        "gate_basis": "native_full_dflash_equals_full_adapter",
        "ar_divergent_case_count": sum(
            not (row.get("native_equals_ar") and row.get("adapter_equals_ar"))
            for row in results if row.get("status") != "error"
        ),
        "native_adapter_mismatch_count": sum(
            not row.get("native_equals_adapter") for row in results if row.get("status") != "error"
        ),
        "mode": cohort.get("phase"),
        "max_new_tokens": max_new_tokens,
        "runtime_signature": runtime_sig,
        "code_signature": code_sig["sha256"],
        "cohort_groups": cohort.get("selected_group_count"),
        "sample_count": len(results),
        "cases": results,
    }
    evidence_path = run_root / "evidence" / f"parity_{cohort.get('phase')}.json"
    atomic_json(evidence_path, evidence)
    # Every retained trace must pass parity. Missing references remain explicit
    # incomplete evidence; the other samples can finish, and dev lock rejects it.
    return evidence["status"] in {"pass", "partial_with_errors"}, traces


def _sample_index(samples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for sample in samples:
        sample_id = str(sample["id"])
        if sample_id in indexed:
            raise CampaignIntegrityError(f"duplicate sample ID makes traces ambiguous: {sample_id}")
        indexed[sample_id] = dict(sample)
    return indexed


def _trace_index(traces: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(trace["sample_id"]): trace for trace in traces}


def _prefix_pair(trace: Mapping[str, Any], prefix, max_new_tokens: int):
    from TrainingFree.context_interventions.execution import _prefix_from_trace
    reference_ids = _replay_reference_ids(trace)
    round_map = {int(row["round_index"]): row for row in trace.get("adapter_rounds", [])
                 if row.get("type") == "round" and int(row.get("gamma_executed", 0)) >= 15}
    previous = None
    if prefix.previous_dense_round_index is not None:
        row = round_map.get(int(prefix.previous_dense_round_index))
        if row is not None:
            previous = _prefix_from_trace(
                row, output_ids=reference_ids, max_new_tokens=max_new_tokens,
                previous_round_index=None,
            )
    return previous


def _prefix_at_age(trace: Mapping[str, Any], prefix, age: int, max_new_tokens: int):
    from TrainingFree.context_interventions.execution import _prefix_from_trace
    reference_ids = _replay_reference_ids(trace)

    prior_rounds = sorted(
        (row for row in trace.get("adapter_rounds", [])
         if row.get("type") == "round" and int(row.get("gamma_executed", 0)) >= 15
         and int(row.get("round_index", -1)) < prefix.round_index),
        key=lambda row: int(row["round_index"]),
    )
    if age < 1 or len(prior_rounds) < age:
        return None
    row = prior_rounds[-age]
    return _prefix_from_trace(
        row, output_ids=reference_ids, max_new_tokens=max_new_tokens,
        previous_round_index=None,
    )


def _phase_store(run_root: Path, phase: str, kind: str, signature: str, resume: bool):
    from TrainingFree.context_interventions.artifacts import CellStore
    return CellStore(run_root / phase / kind, signature=signature, resume=resume)


def _save_rows(store, rows: Sequence[Mapping[str, Any]], existing: dict[str, Mapping[str, Any]]) -> None:
    from TrainingFree.context_interventions.resilience import sanitize_failed_numbers

    for row in rows:
        row = sanitize_failed_numbers(dict(row))
        cell_id = str(row["cell_id"])
        previous = existing.get(cell_id)
        if previous is not None and previous.get("status") != "error":
            continue
        store.save(cell_id, row)
        saved_row = dict(row)
        saved_row["run_signature"] = store.signature
        existing[cell_id] = saved_row


def _cell_signature(runtime: Mapping[str, Any], args: argparse.Namespace, phase: str, kind: str) -> str:
    return _hash_json({
        "phase": phase,
        "kind": kind,
        "runtime": runtime["runtime_signature"].get("signature"),
        "target": runtime.get("target_signature"),
        "draft": runtime.get("draft_signature"),
        "tokenizer": runtime.get("tokenizer_signature"),
        "code": _code_signature()["sha256"],
        "max_new_tokens": args.max_new_tokens,
        "source_anchors": args.source_anchors,
        "recent_output": args.recent_output,
        "a3_selector": args.a3_selector,
        "a3_budget": args.a3_budget,
        "refresh_period": args.refresh_period,
        "attention_backend": args.attention_backend,
    })


def _check_compute_reason(reason: str) -> None:
    from TrainingFree.context_interventions.resilience import CampaignIntegrityError, is_fatal_compute_error

    if is_fatal_compute_error(RuntimeError(reason)):
        raise CampaignIntegrityError(reason)


def _compute_replay_prefix(
    args, runtime, sample, ids, layout, prefix, reference_ids, *, phase,
    max_new_tokens, run_a3, trace, previous, reference_accepted, previous_accepted,
):
    from TrainingFree.context_interventions.execution import run_a1_prefix, run_a2_prefix, run_a3_prefix
    from TrainingFree.context_interventions.replay import rebuild_replay_state, evaluate_replay_action
    from TrainingFree.context_interventions.resilience import release_compute_memory

    state = None
    replay_cache = {}
    try:
        state = rebuild_replay_state(
            runtime, ids, reference_ids, prefix, layout, max_new_tokens=max_new_tokens,
        )
        full_positions = tuple(range(state.bank.length))
        full = evaluate_replay_action(
            runtime, ids, state, layout, [full_positions for _ in runtime["draft"].layers],
            requested_budget="full", selector_id="full",
            remaining_output_tokens=prefix.remaining_output_tokens, collect_attention_scores=True,
        )
        accepted = full.get("accepted_candidates")
        if full.get("status") != "ok" or accepted is None or int(accepted) != reference_accepted:
            status = "baseline_mismatch" if full.get("status") == "ok" else "baseline_error"
            reason = (
                f"{full.get('reason') or 'acceptance_mismatch'}: "
                f"replay={accepted}, reference={reference_accepted}"
            )
            _check_compute_reason(reason)
            return {"status": status, "reason": reason, "replay_accepted_candidates": accepted}
        a1 = run_a1_prefix(
            runtime, sample, ids, layout, prefix, reference_ids, phase=phase,
            max_new_tokens=max_new_tokens, state=state, full_outcome=full,
            source_anchors=args.source_anchors, recent_output=args.recent_output,
            reference_accepted=reference_accepted,
        )
        a2 = run_a2_prefix(
            runtime, sample, ids, layout, prefix, reference_ids, phase=phase,
            max_new_tokens=max_new_tokens, state=state, full_outcome=full,
            replay_cache=replay_cache, previous_prefix=previous,
            reference_accepted=reference_accepted, previous_reference_accepted=previous_accepted,
            source_anchors=args.source_anchors, recent_output=args.recent_output,
        )
        a3 = []
        if run_a3:
            round_rows = {
                int(row["round_index"]): row for row in trace["adapter_rounds"]
                if row.get("type") == "round" and int(row.get("gamma_executed", 0)) >= 15
            }
            ages = (1, 2, 4, 8, 16) if phase == "dev" and args.a3_selector == "previous_dflash" else (1,)
            for age in ages:
                age_prefix = _prefix_at_age(trace, prefix, age, max_new_tokens) if args.a3_selector == "previous_dflash" else None
                age_row = round_rows.get(age_prefix.round_index) if age_prefix is not None else None
                age_accepted = int(age_row.get("accepted_candidates", 0)) if age_row else None
                a3.extend(run_a3_prefix(
                    runtime, sample, ids, layout, prefix, reference_ids, phase=phase,
                    selector=args.a3_selector, max_new_tokens=max_new_tokens,
                    state=state, full_outcome=full, replay_cache=replay_cache,
                    previous_prefix=age_prefix if args.a3_selector == "previous_dflash" else previous,
                    ranking_age=age if args.a3_selector == "previous_dflash" else None,
                    source_anchors=args.source_anchors, recent_output=args.recent_output,
                    reference_accepted=reference_accepted,
                    previous_reference_accepted=age_accepted if args.a3_selector == "previous_dflash" else previous_accepted,
                ))
        return {"status": "ok", "a1": a1, "a2": a2, "a3_replay": a3}
    finally:
        # Release the old prefix before the next prefix is rebuilt, including
        # exception paths. Keeping it during the next prefill can double VRAM.
        state = None
        replay_cache.clear()
        release_compute_memory()


def _replay_result_error(result) -> str | None:
    from TrainingFree.context_adaptive.schema import _check_finite

    if result.get("status") == "baseline_error":
        return str(result.get("reason"))
    for kind in ("a1", "a2", "a3_replay"):
        for row in result.get(kind, []):
            if row.get("status") == "error":
                reason = str(row.get("reason") or "replay cell failed")
                _check_compute_reason(reason)
                return reason
    nonfinite = _check_finite(result)
    if nonfinite:
        return f"nonfinite replay values: {nonfinite}"
    return None


def _run_replay(
    args: argparse.Namespace,
    run_root: Path,
    runtime: Mapping[str, Any],
    samples: Sequence[dict[str, Any]],
    traces: Sequence[Mapping[str, Any]],
    *,
    phase: str,
    max_new_tokens: int,
    run_a3: bool,
) -> None:
    from TrainingFree.context_adaptive.prompt import prepare_prompt
    from TrainingFree.context_interventions.replay import choose_reference_prefixes
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.context_interventions.resilience import FailureStreak, run_compute_unit

    sample_map = _sample_index(samples)
    failures = FailureStreak()
    stores = {
        kind: _phase_store(run_root, phase, kind, _cell_signature(runtime, args, phase, kind), args.resume)
        for kind in ("a1", "a2", "a3_replay", "prefix_diagnostics")
    }
    existing = {
        kind: {str(row.get("cell_id")): row for row in store.rows()}
        for kind, store in stores.items()
    }
    for sample_index, trace in enumerate(traces, start=1):
        sample = sample_map[str(trace["sample_id"])]
        sample["_campaign_run_id"] = args.run_id
        reference_output_ids = _replay_reference_ids(trace)
        ids, layout = prepare_prompt(sample, runtime["tokenizer"], chunk_size=128)
        ids = ids.to(runtime["target"].device).contiguous()
        if layout.prompt_hash != trace["prompt_hash"]:
            raise ValueError(f"prompt token hash changed since reference trace: {sample['id']}")
        if not layout.source_available:
            raise ValueError(f"source span became unavailable during replay: {sample['id']} {layout.source_reason}")
        prefixes = choose_reference_prefixes(
            trace["adapter_rounds"], reference_output_ids, prompt_length=int(trace["prompt_tokens"]),
            max_new_tokens=max_new_tokens, eos_ids={int(value) for value in trace.get("eos_ids", [])},
            max_prefixes=args.max_prefixes,
        )
        round_rows = {
            int(row["round_index"]): row for row in trace["adapter_rounds"]
            if row.get("type") == "round" and int(row.get("gamma_executed", 0)) >= 15
        }
        for prefix_index, prefix in enumerate(prefixes, start=1):
            prefix_key = f"{sample['dataset']}|{sample['_cad_source_group_id']}|{prefix.round_index}"
            prefix_marker = run_root / phase / "prefixes" / f"{hashlib.sha256(prefix_key.encode()).hexdigest()}.json"
            if args.resume and prefix_marker.is_file():
                marker = json.loads(prefix_marker.read_text(encoding="utf-8"))
                if marker.get("signature") != _cell_signature(runtime, args, phase, "prefixes"):
                    raise ValueError(f"prefix resume signature mismatch: {prefix_marker}")
                marker_status = marker.get("status", "complete")
                if marker_status == "baseline_mismatch":
                    if marker.get("diagnostic_cell_id") in existing["prefix_diagnostics"]:
                        failures.observe(failed=True, label=prefix_key)
                        continue
                elif marker_status in {"error", "baseline_error"}:
                    pass  # A resumed phase retries failed computation.
                elif marker_status == "complete":
                    marker_cells = marker.get("cell_ids")
                    if marker_cells is None:
                        # Older markers were written only after all cells completed.
                        continue
                    if all(
                        cell_id in existing[kind] and existing[kind][cell_id].get("status") != "error"
                        for kind, cell_ids in marker_cells.items()
                        for cell_id in cell_ids
                    ):
                        failures.observe(failed=False, label=prefix_key)
                        continue
                else:
                    raise ValueError(f"unknown prefix checkpoint status {marker_status!r}: {prefix_marker}")
            previous = _prefix_pair(trace, prefix, max_new_tokens)
            reference_round = round_rows.get(prefix.round_index)
            if reference_round is None:
                raise ValueError(f"reference round {prefix.round_index} missing from saved trace")
            reference_accepted = int(reference_round.get("accepted_candidates", 0))
            previous_accepted = (
                int(round_rows[prefix.previous_dense_round_index].get("accepted_candidates", 0))
                if prefix.previous_dense_round_index in round_rows else None
            )
            print(
                f"[{phase} replay {sample_index}/{len(traces)} prefix {prefix_index}/{len(prefixes)}] "
                f"{sample['dataset']}:{sample['id']} round={prefix.round_index}", flush=True,
            )
            label = f"{phase} replay {sample['dataset']}:{sample['id']} round={prefix.round_index}"
            computed, compute_error = run_compute_unit(
                lambda: _compute_replay_prefix(
                    args, runtime, sample, ids, layout, prefix, reference_output_ids,
                    phase=phase, max_new_tokens=max_new_tokens, run_a3=run_a3, trace=trace,
                    previous=previous, reference_accepted=reference_accepted,
                    previous_accepted=previous_accepted,
                ),
                label=label, result_error=_replay_result_error,
            )
            diagnostic_id = f"{prefix_key}|full_dflash_baseline"
            if computed is None or computed.get("status") != "ok":
                baseline_mismatch = computed is not None and computed.get("status") == "baseline_mismatch"
                failure_status = "baseline_mismatch" if baseline_mismatch else "error"
                reason = str((computed or compute_error or {}).get("reason", "prefix computation failed"))
                diagnostic = {
                    "schema_version": "dflash-context-interventions.prefix-diagnostic.v1",
                    "type": "prefix_diagnostic", "experiment": "full_dflash_replay_baseline",
                    "phase": phase, "sample_id": str(sample["id"]), "dataset": str(sample["dataset"]),
                    "source_group_id": str(sample["_cad_source_group_id"]),
                    "prefix_round_index": int(prefix.round_index),
                    "processed_output_tokens": int(prefix.processed_output_tokens),
                    "prefix_hash": str(prefix.prefix_hash), "status": failure_status, "reason": reason,
                    "reference_accepted_candidates": reference_accepted,
                    "replay_accepted_candidates": (computed or {}).get("replay_accepted_candidates"),
                    "compute_error": compute_error, "run_id": args.run_id, "cell_id": diagnostic_id,
                }
                _save_rows(stores["prefix_diagnostics"], [diagnostic], existing["prefix_diagnostics"])
                prefix_marker.parent.mkdir(parents=True, exist_ok=True)
                atomic_json(prefix_marker, {
                    "prefix_key": prefix_key,
                    "signature": _cell_signature(runtime, args, phase, "prefixes"),
                    "status": failure_status, "diagnostic_cell_id": diagnostic_id,
                })
                print(
                    f"[{phase} replay] quarantined prefix sample={sample['id']} "
                    f"round={prefix.round_index} status={failure_status} {reason}; continuing",
                    file=sys.stderr, flush=True,
                )
                failures.observe(failed=True, label=label)
                continue
            prefix_cell_ids = {}
            for kind in ("a1", "a2", "a3_replay"):
                rows = computed[kind]
                _save_rows(stores[kind], rows, existing[kind])
                prefix_cell_ids[kind] = [str(row["cell_id"]) for row in rows]
            previous_check = next((
                row.get("previous_replay_check") for row in computed["a2"]
                if row.get("previous_replay_check") is not None
            ), None)
            if previous_check and previous_check.get("status") not in {"ok", "acceptance_unavailable"}:
                print(
                    f"[{phase} replay] previous_dflash evidence sample={sample['id']} "
                    f"round={previous_check.get('round_index')} status={previous_check.get('status')}",
                    flush=True,
                )
            prior_diagnostic = existing["prefix_diagnostics"].get(diagnostic_id)
            if prior_diagnostic is not None and prior_diagnostic.get("status") == "error":
                _save_rows(stores["prefix_diagnostics"], [{
                    **prior_diagnostic, "status": "resolved", "reason": "prefix recovered on resume",
                }], existing["prefix_diagnostics"])
            prefix_marker.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(prefix_marker, {
                "prefix_key": prefix_key,
                "signature": _cell_signature(runtime, args, phase, "prefixes"),
                "status": "complete", "cell_ids": prefix_cell_ids,
                "a1_rows": len(computed["a1"]), "a2_rows": len(computed["a2"]), "a3_replay": run_a3,
            })
            failures.observe(failed=False, label=label)
            del computed
    for kind, store in stores.items():
        rows = store.rows()
        statuses = Counter(str(row.get("status", "missing")) for row in rows)
        failed_prefixes = sum(1 for row in existing["prefix_diagnostics"].values()
                              if row.get("status") in {"baseline_mismatch", "baseline_error", "error"})
        store.merge(run_root / phase / f"{kind}.jsonl", summary={
            "type": "summary",
            "status": (
                "partial_with_errors"
                if statuses.get("error") or failed_prefixes
                else "complete"
            ),
            "phase": phase,
            "experiment": kind,
            "planned_or_recorded_rows": len(rows),
            "status_counts": dict(statuses),
            "run_signature": store.signature,
        })


def _rollout_parity_failed(rows) -> bool:
    arms = {str(row.get("variant")): row for row in rows}
    adapter = arms.get("adapter_full")
    native = arms.get("native_full_dflash")
    if adapter is None or native is None or adapter.get("status") != "ok" or native.get("status") != "ok":
        return False
    if not adapter.get("matches_native_full_dflash"):
        return True
    if adapter.get("output_ids") is not None and native.get("output_ids") is not None:
        return adapter["output_ids"] != native["output_ids"]
    return False


def _rollout_result_error(rows, selector) -> str | None:
    from TrainingFree.context_interventions.resilience import CampaignIntegrityError
    from TrainingFree.context_adaptive.schema import _check_finite

    variants = [str(row.get("variant")) for row in rows]
    required = {"ar", "native_full_dflash", "adapter_full", selector}
    if len(variants) != 4 or set(variants) != required:
        raise CampaignIntegrityError(f"A3 paired result has invalid arms: {variants}")
    if _rollout_parity_failed(rows):
        return None  # Save the failed parity evidence, then stop without retry.
    nonfinite = _check_finite(rows)
    if nonfinite:
        return f"nonfinite rollout values: {nonfinite}"
    for row in rows:
        if row.get("status") != "ok":
            reason = str(row.get("error") or row.get("reason") or f"{row['variant']} status={row.get('status')}")
            _check_compute_reason(reason)
            return reason
        try:
            latency = float(row["wall_e2e_ms"])
            if not math.isfinite(latency) or latency <= 0:
                return f"{row['variant']} has invalid wall latency: {latency}"
            if row["variant"] in {"native_full_dflash", selector}:
                acceptance = float(row["counters"]["mean_accepted_candidates"])
                if not math.isfinite(acceptance) or not 0 <= acceptance <= 15:
                    return f"{row['variant']} has invalid acceptance metric: {acceptance}"
        except (KeyError, TypeError, ValueError) as exc:
            return f"{row['variant']} is missing usable timing/acceptance metrics: {exc}"
    return None


def _run_rollouts(
    args: argparse.Namespace, run_root: Path, runtime: Mapping[str, Any],
    samples: Sequence[dict[str, Any]], traces: Sequence[Mapping[str, Any]], *,
    phase: str, max_new_tokens: int, periods: Sequence[int], repetitions: int, warmups: int,
) -> None:
    from TrainingFree.context_interventions.execution import run_a3_comparison
    from TrainingFree.context_interventions.resilience import CampaignIntegrityError, FailureStreak, run_compute_unit, sanitize_failed_numbers

    # Four measured arms belong to one attempt. Commit the complete bundle in
    # one atomic checkpoint so retries/resume can never mix measurement attempts.
    store = _phase_store(
        run_root, phase, "a3_rollout_pairs", _cell_signature(runtime, args, phase, "a3_rollout_pairs"), args.resume,
    )
    saved_by_id = {str(row.get("cell_id")): row for row in store.rows()}
    sample_map = _sample_index(samples)
    failures = FailureStreak()
    required = ("ar", "native_full_dflash", "adapter_full", args.a3_selector)
    for index, trace in enumerate(traces, start=1):
        sample = sample_map[str(trace["sample_id"])]
        sample_has_success = False
        for period in periods:
            for repetition in range(repetitions):
                pair_id = "|".join((
                    str(sample["dataset"]), str(sample["_cad_source_group_id"]),
                    args.a3_selector, str(args.a3_budget), str(period), str(repetition),
                ))
                previous = saved_by_id.get(pair_id)
                if previous is not None:
                    if previous.get("status") == "parity_fail":
                        raise CampaignIntegrityError(f"A3 full-adapter/native-DFlash parity failed for saved cell {sample['id']}")
                    saved_rows = previous.get("rows", [])
                    saved_error = _rollout_result_error(saved_rows, args.a3_selector)
                    if _rollout_parity_failed(saved_rows):
                        raise CampaignIntegrityError(f"A3 full-adapter/native-DFlash parity failed for saved cell {sample['id']}")
                    if previous.get("status") == "ok":
                        if saved_error is not None:
                            raise CampaignIntegrityError(f"A3 successful checkpoint has invalid evidence: {pair_id}: {saved_error}")
                        sample_has_success = True
                        continue
                print(
                    f"[{phase} A3 {index}/{len(traces)}] {sample['dataset']}:{sample['id']} "
                    f"selector={args.a3_selector} B={args.a3_budget} period={period} rep={repetition}",
                    flush=True,
                )
                attempt_count = 0

                def measure_pair():
                    nonlocal attempt_count
                    attempt_count += 1
                    return run_a3_comparison(
                        runtime, sample, max_new_tokens=max_new_tokens,
                        selector=args.a3_selector, budget=args.a3_budget,
                        refresh_period=period, repetition=repetition,
                        warmups=warmups if repetition == 0 and attempt_count == 1 else 0,
                        source_anchors=args.source_anchors, recent_output=args.recent_output,
                    )

                rows, compute_error = run_compute_unit(
                    measure_pair, label=f"{phase} A3 {pair_id}",
                    result_error=lambda values: _rollout_result_error(values, args.a3_selector),
                )
                if rows is None:
                    rows = [{"variant": variant, "status": "error"} for variant in required]
                parity_failed = _rollout_parity_failed(rows)
                for row in rows:
                    row.update({
                        "schema_version": "dflash-context-interventions.rollout.v1", "type": "rollout",
                        "sample_id": str(sample["id"]), "dataset": str(sample["dataset"]),
                        "source_group_id": str(sample["_cad_source_group_id"]),
                        "split": str(sample.get("_cad_split", "unknown")),
                        "selector": args.a3_selector, "budget": args.a3_budget,
                        "refresh_period": period, "repetition": repetition,
                        "cell_id": f"{pair_id}|{row['variant']}", "pair_id": pair_id,
                        "run_id": args.run_id, "compute_attempts": attempt_count,
                    })
                    if compute_error is not None:
                        row["attempt_status"] = row.get("status")
                        row["status"] = "error"
                        row["error"] = compute_error["reason"]
                        row["pair_complete"] = False
                    else:
                        row["pair_complete"] = True
                rows = [sanitize_failed_numbers(row) for row in rows]
                bundle = {
                    "schema_version": "dflash-context-interventions.rollout-pair.v1",
                    "type": "rollout_pair", "run_id": args.run_id, "cell_id": pair_id,
                    "status": "parity_fail" if parity_failed else ("error" if compute_error else "ok"),
                    "rows": rows, "compute_error": compute_error, "compute_attempts": attempt_count,
                }
                store.save(pair_id, bundle)
                saved_by_id[pair_id] = bundle
                if parity_failed:
                    raise CampaignIntegrityError(f"A3 full-adapter/native-DFlash parity failed for {sample['id']}")
                sample_has_success = sample_has_success or compute_error is None
        # Optional grid failures must not declare the whole GPU unusable while
        # other configurations still produce valid paired evidence for a sample.
        failures.observe(failed=not sample_has_success, label=str(sample["id"]))
    all_rows = [row for bundle in saved_by_id.values() for row in bundle["rows"]]
    statuses = Counter(str(row.get("status", "missing")) for row in all_rows)
    configurations = []
    for period in periods:
        config_rows = [row for row in all_rows if int(row["refresh_period"]) == int(period)]
        planned = len(samples) * repetitions * 4
        config_statuses = Counter(str(row.get("status", "missing")) for row in config_rows)
        configurations.append({
            "selector": args.a3_selector, "budget": args.a3_budget, "refresh_period": period,
            "rows": len(config_rows), "planned_rows": planned,
            "status": "complete" if len(config_rows) == planned and config_statuses.get("ok", 0) == planned else "partial_with_errors",
            "status_counts": dict(config_statuses),
        })
    planned_rows = len(samples) * len(periods) * repetitions * 4
    store.merge(run_root / phase / "a3_rollout.jsonl", records=all_rows, summary={
        "type": "summary",
        "status": "complete" if statuses.get("ok", 0) == planned_rows and len(all_rows) == planned_rows else "partial_with_errors",
        "phase": phase, "experiment": "A3_rollout", "rows": len(all_rows), "planned_rows": planned_rows,
        "status_counts": dict(statuses), "configurations": configurations, "run_signature": store.signature,
    })


def _run_phase(args: argparse.Namespace, run_root: Path, phase: str) -> int:
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.context_interventions.environment import run_preflight

    samples, _ = _load_samples(args)
    saved_split = run_root / "cohorts" / "split_manifest.json"
    if not args.split_manifest and saved_split.is_file():
        args.split_manifest = str(saved_split)
    partial_confirm = phase == "confirm" and args.allow_partial_confirm
    selected, split_manifest, cohort = _cohort(
        args, samples, phase, require_quota=not partial_confirm,
    )
    if partial_confirm:
        planned_quotas = dict(cohort.get("quotas", {}))
        actual_quotas = {
            dataset: sum(1 for row in cohort.get("groups", []) if row.get("dataset") == dataset)
            for dataset in planned_quotas
        }
        if any(count < 1 for count in actual_quotas.values()):
            raise ValueError(
                "partial confirm still requires at least one untouched source group per dataset; "
                f"available={cohort.get('available_groups')}"
            )
        cohort["planned_quotas"] = planned_quotas
        cohort["actual_quotas"] = actual_quotas
        cohort["status"] = "partial_confirmation" if cohort.get("shortages") else "complete"
        if cohort.get("shortages"):
            cohort["scope_note"] = (
                "Exposure audit reduced the untouched test cohort below the preregistered quota. "
                "Use the observed group counts and treat confirm conclusions as limited."
            )
    _save_cohort(run_root, cohort, split_manifest)
    locked: dict[str, Any] | None = None
    dev_runtime: dict[str, Any] = {}
    if phase == "confirm":
        locked_path = _resolve(args.locked_config) if args.locked_config else run_root / "locked_config.json"
        if locked_path is None or not locked_path.is_file():
            raise FileNotFoundError("confirm requires --locked-config from a completed dev phase")
        locked = json.loads(locked_path.read_text(encoding="utf-8"))
        if locked.get("status") != "locked_for_heldout" or locked.get("run_id") != args.run_id:
            raise ValueError("confirm locked config must be valid and use the current --run-id")
        if locked.get("code_signature") != _code_signature()["sha256"]:
            raise ValueError("campaign source code changed after lock; heldout has not been opened")
        split_path = run_root / "cohorts" / "split_manifest.json"
        if not split_path.is_file() or locked.get("split_manifest_sha256") != hashlib.sha256(split_path.read_bytes()).hexdigest():
            raise ValueError("heldout confirmation split differs from the locked dev split")
        dev_runtime_path = run_root / "runtime_dev.json"
        if not dev_runtime_path.is_file():
            raise FileNotFoundError("locked dev runtime signature is missing")
        dev_runtime = json.loads(dev_runtime_path.read_text(encoding="utf-8"))
        args.a3_selector = str(locked["selector"])
        args.a3_budget = int(locked["budget"])
        args.refresh_period = int(locked["refresh_period"])
    if args.max_new_tokens is None:
        args.max_new_tokens = 1024 if phase == "pilot" else 2048
    if args.warmups is None:
        args.warmups = 1 if phase == "pilot" else 3
    if args.repetitions is None:
        args.repetitions = 1 if phase == "pilot" else 3
    if phase in {"dev", "confirm"} and args.attention_backend != "flash_attention_4":
        raise ValueError("dev/confirm primary results require FA4; SDPA is diagnostic-only")
    data_dir, data_file = _paths(args)
    preflight = run_preflight(
        target_model=args.target_model, draft_model=args.draft_model,
        data_dir=str(data_dir) if data_dir else None,
        data_file=str(data_file) if data_file else None,
        attention_backend=args.attention_backend,
    )
    atomic_json(run_root / "preflight.json", preflight)
    if preflight.get("status") != "ready":
        print("Preflight is blocked; no model was loaded and no GPU experiment was started.", file=sys.stderr, flush=True)
        return 2
    runtime = _runtime(args)
    atomic_json(run_root / f"runtime_{phase}.json", {
        "runtime_signature": runtime["runtime_signature"],
        "target_signature": runtime["target_signature"],
        "draft_signature": runtime["draft_signature"],
        "tokenizer_signature": runtime["tokenizer_signature"],
        "target_asset": runtime["target_asset"],
        "draft_asset": runtime["draft_asset"],
        "attention_backend": runtime["attention_backend"],
    })
    if phase == "confirm":
        assert locked is not None
        locked_runtime = locked.get("runtime_signature")
        actual_runtime = runtime["runtime_signature"].get("signature")
        if locked_runtime and locked_runtime != actual_runtime:
            raise ValueError("heldout confirmation runtime differs from the locked dev runtime; heldout has not been opened")
        if (runtime.get("target_signature") != dev_runtime.get("target_signature")
                or runtime.get("draft_signature") != dev_runtime.get("draft_signature")
                or runtime.get("tokenizer_signature") != dev_runtime.get("tokenizer_signature")):
            raise ValueError("model/tokenizer assets differ from the locked dev artifacts; heldout has not been opened")
    parity_pass, traces = _run_parity_cohort(
        args, run_root, runtime, selected, cohort, max_new_tokens=args.max_new_tokens,
    )
    if not parity_pass:
        print("Parity gate failed; no A1/A2/A3 replay or performance claim was run.", file=sys.stderr, flush=True)
        return 3
    if phase == "validate":
        return 0
    _run_replay(
        args, run_root, runtime, selected, traces, phase=phase,
        max_new_tokens=args.max_new_tokens, run_a3=True,
    )
    if phase == "pilot":
        rollout_periods = (args.refresh_period,) if args.a3_selector == "previous_dflash" else (1,)
    elif phase == "dev":
        rollout_periods = (2, 4, 8, 16) if args.a3_selector == "previous_dflash" else (1,)
    else:
        locked = json.loads((_resolve(args.locked_config) or run_root / "locked_config.json").read_text(encoding="utf-8"))
        args.a3_selector = str(locked["selector"])
        args.a3_budget = int(locked["budget"])
        args.refresh_period = int(locked["refresh_period"])
        rollout_periods = (args.refresh_period,)
    _run_rollouts(
        args, run_root, runtime, selected, traces, phase=phase,
        max_new_tokens=args.max_new_tokens, periods=rollout_periods,
        repetitions=args.repetitions, warmups=args.warmups,
    )
    from TrainingFree.context_interventions.report import build_report
    report = build_report(run_root, phase)
    print(f"Report: {run_root / 'reports' / f'{phase}_decision.md'}", flush=True)
    return 0


def _lock(args: argparse.Namespace, run_root: Path) -> int:
    from TrainingFree.context_adaptive.schema import atomic_json, sha256_file
    from TrainingFree.context_interventions.report import build_report

    if not args.a3_selector or args.a3_budget < 1 or args.refresh_period < 1:
        raise ValueError("lock requires a deployable --a3-selector, positive --a3-budget and --refresh-period")
    evidence = run_root / "evidence" / "parity_dev.json"
    if not evidence.is_file() or json.loads(evidence.read_text(encoding="utf-8")).get("status") != "pass":
        raise ValueError("dev parity evidence must pass before configuration can be locked")
    report = build_report(run_root, "dev")
    rollout_path = run_root / "dev" / "a3_rollout.jsonl"
    if not rollout_path.is_file():
        raise ValueError("dev A3 rollout artifact is missing; run dev before lock")
    from TrainingFree.context_adaptive.schema import read_jsonl
    rows = read_jsonl(rollout_path)
    config_rows = [row for row in rows if row.get("type") == "rollout"
                   and row.get("selector") == args.a3_selector
                   and int(row.get("budget", -1)) == args.a3_budget
                   and int(row.get("refresh_period", -1)) == args.refresh_period]
    required_variants = {"ar", "native_full_dflash", "adapter_full", args.a3_selector}
    arms_by_group_repetition: dict[tuple[str, str, int], dict[str, Mapping[str, Any]]] = {}
    for row in config_rows:
        key = (str(row.get("dataset")), str(row.get("source_group_id")), int(row.get("repetition", -1)))
        arms_by_group_repetition.setdefault(key, {})[str(row.get("variant"))] = row
    if not arms_by_group_repetition or any(
        set(arms) != required_variants
        or any(arms[variant].get("status") != "ok" for variant in required_variants)
        or not arms["adapter_full"].get("matches_native_full_dflash")
        for arms in arms_by_group_repetition.values()
    ):
        raise ValueError("dev A3 rollout is missing a valid paired arm or full-adapter/native-DFlash parity")
    chosen = [row for row in rows if row.get("type") == "rollout" and row.get("variant") == args.a3_selector
              and row.get("selector") == args.a3_selector and int(row.get("budget", -1)) == args.a3_budget
              and int(row.get("refresh_period", -1)) == args.refresh_period]
    groups_by_dataset: dict[str, set[str]] = {}
    repetitions_by_group: dict[tuple[str, str], set[int]] = {}
    for row in chosen:
        if row.get("status") == "ok":
            groups_by_dataset.setdefault(str(row.get("dataset")), set()).add(str(row.get("source_group_id")))
            key = (str(row.get("dataset")), str(row.get("source_group_id")))
            repetitions_by_group.setdefault(key, set()).add(int(row.get("repetition", -1)))
    cohort_path = run_root / "cohorts" / "dev.json"
    if not cohort_path.is_file():
        raise ValueError("dev cohort manifest is missing")
    dev_cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    expected_groups: dict[str, set[str]] = {dataset: set() for dataset in ("gov_report", "multi_news", "qmsum")}
    for group in dev_cohort.get("groups", []):
        dataset = str(group.get("dataset", ""))
        if dataset in expected_groups:
            expected_groups[dataset].add(str(group["source_group_id"]))
    for dataset in ("gov_report", "multi_news", "qmsum"):
        expected = expected_groups.get(dataset, set())
        observed = groups_by_dataset.get(dataset, set())
        if not expected or observed != expected:
            raise ValueError(f"selected A3 policy has incomplete dev source-group coverage for {dataset}: {len(observed)}/{len(expected)}")
    if any(repetitions_by_group.get((dataset, group), set()) != {0, 1, 2}
           for dataset, groups in expected_groups.items() for group in groups):
        raise ValueError("selected A3 policy must have three complete timing repetitions for every dev source group")
    candidate_metrics = (
        report.get("rollout_metrics", {})
        .get(args.a3_selector, {})
        .get(f"B{args.a3_budget}_period{args.refresh_period}")
    )
    if not candidate_metrics or candidate_metrics.get("decision") != "acceptance_and_speedup_gate_pass":
        raise ValueError(
            "selected dev A3 configuration has not passed full-adapter parity, "
            "acceptance non-inferiority and latency gates; do not open heldout"
        )
    runtime_path = run_root / "runtime_dev.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
    if runtime.get("attention_backend") != "flash_attention_4":
        raise ValueError("only FA4 dev evidence can be locked for heldout confirmation")
    locked = {
        "schema_version": "dflash-context-interventions.locked.v1",
        "status": "locked_for_heldout",
        "run_id": args.run_id,
        "selector": args.a3_selector,
        "budget": args.a3_budget,
        "refresh_period": args.refresh_period,
        "gamma": 15,
        "block_size": 16,
        "source_anchors": args.source_anchors,
        "recent_output": args.recent_output,
        "random_seeds": [17, 42, 123],
        "dev_parity_sha256": sha256_file(evidence),
        "dev_rollout_sha256": sha256_file(rollout_path),
        "dev_cohort_sha256": sha256_file(cohort_path),
        "split_manifest_sha256": sha256_file(run_root / "cohorts" / "split_manifest.json"),
        "runtime_signature": (runtime.get("runtime_signature") or {}).get("signature"),
        "code_signature": _code_signature()["sha256"],
        "dev_report_decisions": report.get("family_metrics"),
        "dev_a3_rollout_decision": candidate_metrics,
        "dev_source_groups_by_dataset": {key: len(value) for key, value in groups_by_dataset.items()},
        "heldout_opened": False,
    }
    lock_path = run_root / "locked_config.json"
    atomic_json(lock_path, locked)
    print(f"Locked config: {lock_path}", flush=True)
    return 0


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    args.run_id = args.run_id or _utc_run_id()
    if args.gpu_ids:
        if len([item for item in args.gpu_ids.split(",") if item.strip()]) != 1:
            raise SystemExit("this campaign version runs one worker on one GPU; pass exactly one --gpu-ids value")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("FAST_INFER_CACHE_ROOT", "/tmp/fast_infer_cache")
    if args.plan_only:
        return _plan(args)
    run_root = _run_root(args)
    if args.mode == "prepare":
        return _prepare(args, run_root)
    if args.mode == "preflight":
        return _preflight(args, run_root)
    if args.mode == "report":
        from TrainingFree.context_interventions.report import build_report
        report = build_report(run_root, args.report_phase)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.mode == "lock":
        return _lock(args, run_root)
    if args.mode == "confirm":
        locked_path = _resolve(args.locked_config) if args.locked_config else run_root / "locked_config.json"
        if locked_path is None or not locked_path.is_file():
            raise FileNotFoundError("confirm requires --locked-config from a completed dev phase")
        locked = json.loads(locked_path.read_text(encoding="utf-8"))
        if locked.get("status") != "locked_for_heldout":
            raise ValueError("locked config status is invalid")
        if locked.get("run_id") != args.run_id:
            raise ValueError("confirm must use the same --run-id as dev/lock to preserve provenance")
        args.a3_selector = str(locked["selector"])
        args.a3_budget = int(locked["budget"])
        args.refresh_period = int(locked["refresh_period"])
        args.locked_config = str(locked_path)
    if args.mode == "validate":
        args.max_new_tokens = args.max_new_tokens or 1024
        args.warmups = 0
        args.repetitions = 1
        return _run_phase(args, run_root, "validate")
    if args.mode in {"pilot", "dev", "confirm"}:
        return _run_phase(args, run_root, args.mode)
    parser.error(f"unsupported mode: {args.mode}")
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; completed per-cell artifacts can be resumed with --resume.", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except CampaignIntegrityError as exc:
        print(f"campaign integrity error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(3)
    except Exception as exc:
        print(f"campaign error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
