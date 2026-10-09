#!/usr/bin/env python3
"""Prepare, calibrate and benchmark Context-Adaptive DFlash.

The ``prepare``, ``preflight``, ``lock`` and ``report`` phases are model-free.
Inference phases require local model snapshots and a CUDA device.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from TrainingFree.context_adaptive.config import AdaptiveConfig


VARIANTS = (
    "ar", "dflash_full_fixed", "best_fixed_pair", "a_only", "b_only_history",
    "b_only_entropy", "independent_ab", "joint", "joint_no_entropy",
    "joint_no_source_relevance",
)
MODEL_PHASES = {"smoke", "calibrate", "dev", "test"}
ADAPTIVE_VARIANTS = {
    "a_only", "b_only_history", "b_only_entropy", "independent_ab", "joint",
    "joint_no_entropy", "joint_no_source_relevance",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "preflight", "smoke", "calibrate", "dev", "test", "lock", "report"), default=os.environ.get("CAD_PHASE", "smoke"))
    parser.add_argument("--variant", choices=VARIANTS, default=None)
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-root", default=os.environ.get("CAD_OUTPUT_ROOT"))
    parser.add_argument("--run-id", default=os.environ.get("CAD_RUN_ID"))
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--exposure-manifest", action="append", default=[])
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument("--locked-config", default=None)
    parser.add_argument("--locked-config-out", default=None)
    parser.add_argument("--dev-report", default=None)
    parser.add_argument("--budgets", default=None)
    parser.add_argument("--gammas", default=None)
    parser.add_argument("--selector", default=None)
    parser.add_argument("--target-layer", default=None)
    parser.add_argument("--refresh-period", type=int, default=None)
    parser.add_argument("--signal-update-period", type=int, default=None)
    parser.add_argument("--source-chunk-size", type=int, default=None)
    parser.add_argument("--source-anchors", type=int, default=None)
    parser.add_argument("--recent-output", type=int, default=None)
    parser.add_argument("--prior-strength", type=float, default=None)
    parser.add_argument("--min-state-support", type=int, default=None)
    parser.add_argument("--statistics-decay", type=float, default=None)
    parser.add_argument("--controller-margin", type=float, default=None)
    parser.add_argument("--entropy-signal-temperature", type=float, default=None)
    parser.add_argument("--timing-mode", choices=("wall", "diagnostic"), default=None)
    parser.add_argument("--cost-update-mode", choices=("frozen_cost",), default=None)
    parser.add_argument("--statistics-update-mode", choices=("online", "frozen"), default=None)
    parser.add_argument("--length-mode", choices=("draft_shape",), default=None)
    parser.add_argument("--fixed-budget", default=None)
    parser.add_argument("--fixed-gamma", type=int, default=None)
    parser.add_argument("--gamma-reference", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=int(os.environ.get("RUN_MAX_INPUT_TOKENS", "0")))
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--fixed-output-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("RUN_TEMPERATURE", "0")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("LONG_BENCH_SEED", "42")))
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--report-phase", choices=("smoke", "dev", "test"), default="dev")
    parser.add_argument("--report-statistics-mode", choices=("online", "frozen"), default="online")
    parser.add_argument("--calibration-checkpoints", default="0,512,1024,1536")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--model-check", action="store_true", help="preflight local target/draft configs")
    return parser


def _path(value: str | None, fallback: str | None = None) -> Path | None:
    text = value or fallback
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_absolute() else ROOT / path


def _data_source(args: argparse.Namespace) -> tuple[Path | None, Path | None, list[Path]]:
    data_file_value = args.data_file
    data_dir_value = args.data_dir
    if not data_file_value and not data_dir_value:
        env_dir = os.environ.get("CAD_DATA_DIR")
        env_input = os.environ.get("DATA_INPUT")
        if env_dir:
            data_dir_value = env_dir
        elif env_input:
            candidate = _path(env_input)
            if candidate and candidate.is_dir():
                data_dir_value = str(candidate)
            else:
                data_file_value = env_input
        else:
            data_dir_value = "data/longbench_100_14k"
    data_file = _path(data_file_value)
    data_dir = _path(data_dir_value)
    if data_file and data_dir:
        raise ValueError("--data-file and --data-dir are mutually exclusive")
    source_paths = [data_file] if data_file else sorted(data_dir.glob("*.jsonl")) if data_dir and data_dir.is_dir() else []
    if not source_paths:
        raise FileNotFoundError(f"no JSONL data files found: {data_file or data_dir}")
    return data_file, data_dir, source_paths


def _config(args: argparse.Namespace, *, phase: str) -> AdaptiveConfig:
    base = AdaptiveConfig.from_env()
    updates: dict[str, Any] = {}
    if args.budgets is not None:
        updates["budgets"] = tuple("full" if part.strip() == "full" else int(part.strip()) for part in args.budgets.split(",") if part.strip())
    if args.gammas is not None:
        updates["gammas"] = tuple(dict.fromkeys(int(part.strip()) for part in args.gammas.split(",") if part.strip()))
    mapping = {
        "selector": args.selector,
        "refresh_period": args.refresh_period,
        "signal_update_period": args.signal_update_period,
        "source_chunk_size": args.source_chunk_size,
        "source_anchors": args.source_anchors,
        "recent_output": args.recent_output,
        "prior_strength": args.prior_strength,
        "min_state_support": args.min_state_support,
        "statistics_decay": args.statistics_decay,
        "controller_margin": args.controller_margin,
        "entropy_signal_temperature": args.entropy_signal_temperature,
        "timing_mode": args.timing_mode,
        "cost_update_mode": args.cost_update_mode,
        "statistics_update_mode": args.statistics_update_mode,
        "length_mode": args.length_mode,
    }
    updates.update({key: value for key, value in mapping.items() if value is not None})
    updates["seed"] = args.seed
    if args.target_layer is not None:
        updates["target_layer"] = "middle" if args.target_layer == "middle" else int(args.target_layer)
    if phase == "calibrate":
        updates["timing_mode"] = "diagnostic"
        updates["cost_update_mode"] = "frozen_cost"
    return replace(base, **updates)


def _variant(args: argparse.Namespace) -> str:
    return args.variant or os.environ.get("CAD_VARIANT", "dflash_full_fixed" if args.phase == "smoke" else "joint")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_provenance() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _persist_manifest(path: Path, payload: dict[str, Any]) -> None:
    from TrainingFree.context_adaptive.schema import atomic_json

    if path.is_file():
        existing = _load_json(path)
        if existing.get("config_hash") != payload.get("config_hash"):
            raise ValueError(f"refusing to replace incompatible manifest: {path}")
    atomic_json(path, payload)


def _implementation_manifest() -> dict[str, Any]:
    from TrainingFree.context_adaptive.schema import sha256_file

    relative_paths = [
        *[path.relative_to(ROOT) for path in sorted((ROOT / "src/TrainingFree/context_adaptive").glob("*.py"))],
        *[path.relative_to(ROOT) for path in sorted((ROOT / "externals/dflash").rglob("*.py"))],
        Path("scripts/infer_context_adaptive_dflash.py"),
        Path("scripts/runners/run_context_adaptive_dflash.sh"),
        Path("scripts/run.sh"),
        Path("scripts/infer_dflash.py"),
        Path("scripts/common/config.sh"),
        Path("scripts/common/runtime.sh"),
        Path("scripts/common/data_loader.py"),
        Path("scripts/common/input_utils.py"),
        Path("scripts/common/io_util.py"),
        Path("scripts/common/reproducibility.py"),
        Path("scripts/common/rouge.py"),
    ]
    unique_paths = sorted(set(relative_paths), key=lambda path: path.as_posix())
    files = [
        {"path": path.as_posix(), "sha256": sha256_file(ROOT / path)}
        for path in unique_paths
    ]
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": files}


def _prepare(args: argparse.Namespace, output_root: Path, source_paths: list[Path]) -> int:
    from TrainingFree.context_adaptive.benchmark import load_corpus, split_records
    from TrainingFree.context_adaptive.schema import atomic_json, sha256_file
    from TrainingFree.context_adaptive.splits import build_split_manifest, read_exposure_manifest

    data_file, data_dir, _ = _data_source(args)
    samples = load_corpus(data_file=data_file, data_dir=data_dir)
    exposure = [read_exposure_manifest(_path(path) or Path(path)) for path in args.exposure_manifest]
    split_manifest = build_split_manifest(split_records(samples), exposure, seed=args.seed)
    split_manifest["dataset_files"] = [
        {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in source_paths
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "split_manifest.json", split_manifest)
    atomic_json(output_root / "prepare_manifest.json", {
        "schema_version": "cadflash.prepare.v1",
        "status": split_manifest["status"],
        "run_id": args.run_id,
        "record_count": len(samples),
        "group_count": len(split_manifest["groups"]),
        "split_group_counts": {key: len(value) for key, value in split_manifest["splits"].items()},
        "data_files": split_manifest["dataset_files"],
        "exposure_sources": split_manifest["exposure_sources"],
        "unresolved_exposure_sample_ids": split_manifest["unresolved_exposure_sample_ids"],
    })
    print(f"Prepared {len(samples)} records into {len(split_manifest['groups'])} source groups; status={split_manifest['status']}", flush=True)
    print(f"Split manifest: {output_root / 'split_manifest.json'}", flush=True)
    return 0 if split_manifest["status"] == "complete" else 2


def _preflight(args: argparse.Namespace, output_root: Path, source_paths: list[Path], split_path: Path | None) -> int:
    from TrainingFree.context_adaptive.benchmark import (
        load_corpus,
        validate_dflash_target_layers,
        validate_rotary_compatibility,
    )
    from TrainingFree.context_adaptive.schema import atomic_json, sha256_file

    data_file, data_dir, _ = _data_source(args)
    samples = load_corpus(data_file=data_file, data_dir=data_dir)
    checks: dict[str, Any] = {
        "data_files": [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in source_paths],
        "record_count": len(samples),
        "python": sys.version.split()[0],
        "model_configs_checked": False,
    }
    if split_path:
        split = _load_json(split_path)
        checks["split_status"] = split.get("status")
        checks["split_counts"] = {name: len(values) for name, values in split.get("splits", {}).items()}
    if args.model_check:
        target_model = args.target_model or os.environ.get("MODEL_TARGET")
        draft_model = args.draft_model or os.environ.get("MODEL_DFLASH_DRAFT")
        if not target_model or not draft_model:
            raise ValueError("--model-check requires --target-model and --draft-model")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from transformers import AutoConfig
        target_config = AutoConfig.from_pretrained(target_model, local_files_only=True)
        draft_config = AutoConfig.from_pretrained(draft_model, local_files_only=True)
        if target_config.model_type != "qwen3" or draft_config.model_type != "qwen3":
            raise ValueError("Context-Adaptive DFlash V1 currently requires Qwen3 target and draft configs")
        validate_rotary_compatibility(target_config, draft_config)
        dflash = getattr(draft_config, "dflash_config", {}) or {}
        dflash_root = ROOT / "externals" / "dflash"
        if str(dflash_root) not in sys.path:
            sys.path.insert(0, str(dflash_root))
        from dflash.model import build_target_layer_ids

        target_layer_ids = dflash.get("target_layer_ids") or build_target_layer_ids(
            int(draft_config.num_target_layers), int(draft_config.num_hidden_layers)
        )
        layer_types = tuple(getattr(draft_config, "layer_types", ()) or ())
        checks.update({
            "model_configs_checked": True,
            "target_hidden_layers": int(target_config.num_hidden_layers),
            "target_vocab_size": int(target_config.vocab_size),
            "draft_block_size": int(getattr(draft_config, "block_size", 0)),
            "draft_vocab_size": int(draft_config.vocab_size),
            "target_layer_ids": target_layer_ids,
            "mask_token_id": dflash.get("mask_token_id"),
            "target_hidden_size": int(target_config.hidden_size),
            "draft_hidden_size": int(draft_config.hidden_size),
            "draft_layer_types": list(layer_types),
            "rope_theta": (getattr(target_config, "rope_parameters", {}) or {}).get(
                "rope_theta", getattr(target_config, "rope_theta", None)
            ),
            "rope_scaling": getattr(target_config, "rope_scaling", None),
            "rope_parameters": getattr(target_config, "rope_parameters", None),
            "target_max_position_embeddings": getattr(target_config, "max_position_embeddings", None),
            "draft_max_position_embeddings": getattr(draft_config, "max_position_embeddings", None),
            "target_model_assets": str(Path(target_model).resolve()),
            "draft_model_assets": str(Path(draft_model).resolve()),
        })
        if checks["target_vocab_size"] != checks["draft_vocab_size"]:
            raise ValueError("target and DFlash vocab sizes differ")
        if checks["target_hidden_size"] != checks["draft_hidden_size"]:
            raise ValueError("DFlash and target hidden sizes differ")
        validate_dflash_target_layers(
            target_layer_ids,
            target_hidden_layers=checks["target_hidden_layers"],
            draft_num_target_layers=draft_config.num_target_layers,
        )
        mask_token_id = dflash.get("mask_token_id")
        if not isinstance(mask_token_id, int) or not 0 <= mask_token_id < checks["target_vocab_size"]:
            raise ValueError("DFlash mask_token_id is missing or outside the shared vocabulary")
        if checks["draft_block_size"] < 2:
            raise ValueError("DFlash block_size must be at least 2")
        if layer_types and len(layer_types) != int(draft_config.num_hidden_layers):
            raise ValueError("DFlash layer_types length does not match num_hidden_layers")
        if "sliding_attention" in layer_types and int(getattr(draft_config, "sliding_window", 0) or 0) < 1:
            raise ValueError("DFlash sliding_attention layers require a positive sliding_window")
        capabilities = _config(args, phase="preflight").gammas + (
            int(args.fixed_gamma), int(args.gamma_reference),
        )
        if any(gamma >= checks["draft_block_size"] for gamma in capabilities):
            raise ValueError("configured/fixed gamma exceeds draft block_size - 1")
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "preflight.json", {"schema_version": "cadflash.preflight.v1", "status": "ok", **checks})
    print(json.dumps(checks, ensure_ascii=False, indent=2), flush=True)
    return 0


def _lock(args: argparse.Namespace, output_root: Path, config: AdaptiveConfig, variant: str) -> int:
    from TrainingFree.context_adaptive.schema import atomic_json, sha256_file
    from TrainingFree.context_adaptive.schema import validate_calibration

    if not args.calibration_file or not args.locked_config_out or not args.dev_report or not args.split_manifest:
        raise ValueError("lock phase requires --calibration-file, --dev-report, --split-manifest and --locked-config-out")
    fixed_budget: int | str = "full" if str(args.fixed_budget) == "full" else int(args.fixed_budget)
    calibration_path = _path(args.calibration_file)
    dev_report_path = _path(args.dev_report)
    calibration = _load_json(calibration_path)
    problems = validate_calibration(calibration)
    if problems:
        raise ValueError("invalid calibration: " + "; ".join(problems))
    calibration_signature = calibration.get("signature", {})
    implementation = _implementation_manifest()
    run_id = args.run_id or output_root.name
    if calibration_signature.get("run_id") != run_id:
        raise ValueError("calibration was collected under a different run_id")
    if calibration_signature.get("implementation_sha256") != implementation["sha256"]:
        raise ValueError("calibration was collected with a different implementation snapshot")
    split_path = _path(args.split_manifest)
    split_sha256 = sha256_file(split_path)
    if calibration_signature.get("split_manifest_sha256") != split_sha256:
        raise ValueError("calibration was collected with a different split manifest")
    if calibration_signature.get("max_input_tokens") != max(args.max_input_tokens, 0):
        raise ValueError("lock input-token cap differs from calibration")
    if float(calibration_signature.get("temperature", -1)) != float(args.temperature):
        raise ValueError("lock sampling temperature differs from calibration")
    if calibration_signature.get("fixed_output_tokens") != args.fixed_output_tokens:
        raise ValueError("lock output scope differs from calibration")
    if args.max_new_tokens is not None and int(calibration_signature.get("max_new_tokens", -1)) != max(
        args.max_new_tokens, args.fixed_output_tokens or 0
    ):
        raise ValueError("lock output cap differs from calibration")
    config_signature_values = {
        "selector_id": config.selector,
        "target_layer": config.target_layer,
        "refresh_period": config.refresh_period,
        "signal_update_period": config.signal_update_period,
        "source_chunk_size": config.source_chunk_size,
        "source_anchors": config.source_anchors,
        "recent_output": config.recent_output,
        "budgets": list(config.budgets),
        "gammas": list(config.gammas),
        "entropy_signal_temperature": config.entropy_signal_temperature,
        "length_mode": config.length_mode,
    }
    differences = [key for key, value in config_signature_values.items() if calibration_signature.get(key) != value]
    if differences:
        raise ValueError("locked config differs from calibration behavior: " + ", ".join(differences))
    dev_report = _load_json(dev_report_path)
    if dev_report.get("schema_version") != "cadflash.report.v1":
        raise ValueError("unsupported dev report schema")
    if dev_report.get("phase") != "dev" or dev_report.get("statistics_update_mode") != config.statistics_update_mode:
        raise ValueError("dev report phase/statistics mode does not match the locked config")
    if dev_report.get("split_manifest_sha256") != split_sha256 or dev_report.get("provenance_mixed"):
        raise ValueError("dev report provenance does not match the locked split or mixes runs")
    if dev_report.get("dataset_manifest_sha256") != calibration_signature.get("dataset_manifest_sha256"):
        raise ValueError("dev report dataset does not match calibration data")
    if dev_report.get("implementation_sha256") != implementation["sha256"]:
        raise ValueError("dev report was generated with a different implementation snapshot")
    if dev_report.get("run_id") != run_id:
        raise ValueError("dev report was generated under a different run_id")
    comparison = next(
        (row for row in dev_report.get("comparisons", []) if row.get("candidate_variant") == variant),
        None,
    )
    if comparison is None or not comparison.get("headline_valid"):
        raise ValueError(f"dev report has no complete, exact paired comparison for variant {variant}")
    dev_config = dev_report.get("configs_by_variant", {}).get(variant)
    if not isinstance(dev_config, dict) or dev_config.get("inconsistent"):
        raise ValueError(f"dev report is missing consistent config provenance for variant {variant}")
    if json.dumps(dev_config, sort_keys=True, default=str) != json.dumps(config.to_dict(), sort_keys=True, default=str):
        raise ValueError("lock config differs from the selected dev variant config")
    dev_action = dev_report.get("fixed_actions_by_variant", {}).get(variant)
    if not isinstance(dev_action, dict) or dev_action.get("inconsistent"):
        raise ValueError(f"dev report is missing consistent fixed-action provenance for variant {variant}")
    expected_action = {
        "budget": fixed_budget,
        "gamma": int(args.fixed_gamma),
        "gamma_reference": int(args.gamma_reference),
    }
    if any(str(dev_action.get(key)) != str(value) for key, value in expected_action.items()):
        raise ValueError("lock fixed-action settings differ from the selected dev variant")
    test_variants = ["ar"] + sorted({
        str(row["candidate_variant"])
        for row in dev_report.get("comparisons", [])
        if row.get("headline_valid") and row.get("candidate_variant")
    })
    repetition_count = int(dev_report.get("repetitions_by_variant", {}).get(variant, 0))
    warmup_count = int(dev_report.get("warmups_by_variant", {}).get(variant, 0))
    if repetition_count < 1:
        raise ValueError("dev report is missing repetition provenance for the selected variant")
    if args.repetitions is not None and args.repetitions != repetition_count:
        raise ValueError("lock repetition count differs from the dev report")
    if args.warmup_runs is not None and args.warmup_runs != warmup_count:
        raise ValueError("lock warmup count differs from the dev report")
    payload = {
        "schema_version": "cadflash.locked.v1",
        "status": "candidate_locked_pending_heldout",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "run_id": run_id,
        "test_variants": test_variants,
        "config": config.to_dict(),
        "fixed_budget": fixed_budget,
        "fixed_gamma": args.fixed_gamma,
        "gamma_reference": args.gamma_reference,
        "max_new_tokens": int(calibration_signature.get("max_new_tokens", args.max_new_tokens or 512)),
        "temperature": float(calibration.get("signature", {}).get("temperature", args.temperature)),
        "fixed_output_tokens": args.fixed_output_tokens,
        "max_input_tokens": args.max_input_tokens,
        "seed": args.seed,
        "repetitions": repetition_count,
        "warmup_runs": warmup_count,
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(calibration_path),
        "implementation_sha256": implementation["sha256"],
        "split_manifest_sha256": split_sha256,
        "dev_report_path": str(dev_report_path),
        "dev_report_sha256": sha256_file(dev_report_path),
        "dev_gate_status": dev_report.get("gate_status", "unknown"),
        "test_overrides_allowed": ["device", "output_root", "resume"],
    }
    output = _path(args.locked_config_out) or (output_root / "locked_config.json")
    atomic_json(output, payload)
    print(f"Created candidate lock (heldout gates still pending): {output}", flush=True)
    return 0


def _error_record(sample: dict[str, Any], variant: str, run_id: str, repetition: int, model_path: str, draft_path: str | None, exc: BaseException) -> dict[str, Any]:
    return {
        "schema_version": "cadflash.request.v1", "type": "error", "status": "error",
        "method": "context_adaptive_dflash", "variant": variant, "run_id": run_id,
        "sample_id": str(sample.get("id", "unknown")), "dataset": str(sample.get("dataset", "unknown")),
        "split": str(sample.get("_cad_split", "unknown")), "source_group_id": sample.get("_cad_source_group_id"),
        "repetition": repetition, "model": model_path, "draft_model": draft_path,
        "input_tokens": None, "retained_tokens": None, "output_tokens": None, "batch_size": 1,
        "selector_latency_ms": None, "ttft_ms": None, "tpot_ms": None, "e2e_ms": None,
        "throughput_tok_s": None, "qps": None, "peak_memory_gb": None,
        "avg_accept_length": None, "acceptance_rate": None, "rejected_draft_ratio": None,
        "draft_latency_ms": None, "verification_latency_ms": None,
        "error_type": type(exc).__name__, "error": str(exc), "component": "request_generation",
    }


def _request_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row.get("dataset", "unknown")),
        str(row.get("sample_id", "")),
        int(row.get("repetition", 0)),
    )


def _stream_prepare(path: Path, *, resume: bool, config_hash: str, retry_errors: bool = True) -> tuple[set[tuple[str, str, int]], list[dict[str, Any]]]:
    from TrainingFree.context_adaptive.schema import atomic_json, read_jsonl

    if not path.exists():
        return set(), []
    if not resume:
        raise FileExistsError(f"refusing to append to existing artifact: {path}; choose a new output root or use --resume")
    rows = read_jsonl(path)
    data_rows = [row for row in rows if row.get("type") != "summary"]
    for row in data_rows:
        if row.get("config_hash") not in {None, config_hash}:
            raise ValueError(f"resume configuration mismatch in {path}")
    if retry_errors:
        data_rows = [row for row in data_rows if row.get("type") != "error"]
    # JsonlWriter is append-only; remove an old summary while keeping verified rows.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in data_rows), encoding="utf-8")
    completed = {
        _request_key(row)
        for row in data_rows if row.get("type") == "sample" and row.get("status") == "ok"
    }
    return completed, data_rows


def _run_model_phase(args: argparse.Namespace, phase: str, variant: str, output_root: Path, source_paths: list[Path], config: AdaptiveConfig) -> int:
    from TrainingFree.context_adaptive.benchmark import (
        calibration_model_signature,
        data_manifest,
        load_corpus,
        load_runtime,
        request_record,
        select_split,
    )
    from TrainingFree.context_adaptive.calibration import fit_action_priors, profile_controller_costs
    from TrainingFree.context_adaptive.prompt import prepare_prompt
    from TrainingFree.context_adaptive.schema import atomic_json, sha256_file, validate_calibration, validate_request, validate_round
    from TrainingFree.context_adaptive.statistics import AdaptiveStatistics
    from TrainingFree.context_adaptive.generation import generate_adaptive, generate_target_only
    from common import io_util, rouge
    from common.reproducibility import seed_everything

    data_file, data_dir, _ = _data_source(args)
    samples = load_corpus(data_file=data_file, data_dir=data_dir)
    dataset_manifest = data_manifest(samples, source_paths)
    split_manifest = _load_json(_path(args.split_manifest)) if args.split_manifest else None
    if phase in {"calibrate", "dev", "test"} and split_manifest is None:
        raise ValueError(f"phase {phase} requires --split-manifest")
    if phase in {"calibrate", "dev", "test"} and split_manifest.get("status") != "complete":
        raise ValueError(f"source-group split is incomplete: {split_manifest.get('status')}")
    split_name = {"calibrate": "calibration", "dev": "dev", "test": "test"}.get(phase)
    max_samples = args.max_samples
    if phase == "smoke":
        max_samples = None
    selected = select_split(samples, split_manifest, split_name, max_samples=max_samples)
    if phase == "smoke":
        selected = _smoke_deduplicate(selected)
        if args.max_samples is not None:
            selected = selected[: args.max_samples]
    if phase == "calibrate" and args.temperature != 0:
        raise ValueError("calibration action replay is currently greedy-only")
    if phase == "test":
        if not args.locked_config:
            raise ValueError("test phase requires --locked-config")
        if args.max_samples is not None:
            raise ValueError("heldout test does not allow --max-samples; run the complete locked split")
        locked_config, locked_payload = AdaptiveConfig.from_locked(_path(args.locked_config))
        run_id = args.run_id or output_root.name
        if locked_payload.get("run_id") != run_id:
            raise ValueError("test run_id conflicts with locked config")
        permitted_variants = set(locked_payload.get("test_variants", [locked_payload.get("variant", "joint")]))
        if variant not in permitted_variants:
            raise ValueError(f"test variant {variant} is not listed in the locked test matrix")
        explicit = {
            "selector": args.selector,
            "refresh_period": args.refresh_period,
            "signal_update_period": args.signal_update_period,
            "source_chunk_size": args.source_chunk_size,
            "source_anchors": args.source_anchors,
            "recent_output": args.recent_output,
            "prior_strength": args.prior_strength,
            "min_state_support": args.min_state_support,
            "statistics_decay": args.statistics_decay,
            "controller_margin": args.controller_margin,
            "entropy_signal_temperature": args.entropy_signal_temperature,
            "timing_mode": args.timing_mode,
            "cost_update_mode": args.cost_update_mode,
            "statistics_update_mode": args.statistics_update_mode,
            "length_mode": args.length_mode,
        }
        if args.budgets is not None:
            explicit["budgets"] = tuple("full" if part.strip() == "full" else int(part.strip()) for part in args.budgets.split(",") if part.strip())
        if args.gammas is not None:
            explicit["gammas"] = tuple(dict.fromkeys(int(part.strip()) for part in args.gammas.split(",") if part.strip()))
        if args.target_layer is not None:
            explicit["target_layer"] = "middle" if args.target_layer == "middle" else int(args.target_layer)
        mismatches = [key for key, value in explicit.items() if value is not None and getattr(locked_config, key) != value]
        if mismatches:
            raise ValueError("test CLI overrides conflict with locked config: " + ", ".join(mismatches))
        config = locked_config
        locked_max_new = int(locked_payload.get("max_new_tokens", args.max_new_tokens or 512))
        if args.max_new_tokens is not None and args.max_new_tokens != locked_max_new:
            raise ValueError("test output cap conflicts with locked config")
        args.max_new_tokens = locked_max_new
        if float(args.temperature) != float(locked_payload.get("temperature", args.temperature)):
            raise ValueError("test temperature conflicts with locked config")
        if args.fixed_output_tokens != locked_payload.get("fixed_output_tokens"):
            raise ValueError("test fixed-output scope conflicts with locked config")
        if args.max_input_tokens != int(locked_payload.get("max_input_tokens", args.max_input_tokens)):
            raise ValueError("test input cap conflicts with locked config")
        if args.seed != int(locked_payload.get("seed", args.seed)):
            raise ValueError("test seed conflicts with locked config")
        if args.repetitions is not None and args.repetitions != int(locked_payload.get("repetitions", args.repetitions)):
            raise ValueError("test repetition count conflicts with locked config")
        if args.warmup_runs is not None and args.warmup_runs != int(locked_payload.get("warmup_runs", args.warmup_runs)):
            raise ValueError("test warmup count conflicts with locked config")
        if args.repetitions is None:
            args.repetitions = int(locked_payload.get("repetitions", 3))
        if args.warmup_runs is None:
            args.warmup_runs = int(locked_payload.get("warmup_runs", 3))
        if args.fixed_budget_was_set and str(args.fixed_budget) != str(locked_payload.get("fixed_budget", args.fixed_budget)):
            raise ValueError("test fixed-budget override conflicts with locked config")
        if args.fixed_gamma_was_set and args.fixed_gamma != int(locked_payload.get("fixed_gamma", args.fixed_gamma)):
            raise ValueError("test fixed-gamma override conflicts with locked config")
        if args.gamma_reference_was_set and args.gamma_reference != int(locked_payload.get("gamma_reference", args.gamma_reference)):
            raise ValueError("test gamma-reference override conflicts with locked config")
        if args.split_manifest and locked_payload.get("split_manifest_sha256"):
            if sha256_file(_path(args.split_manifest)) != locked_payload["split_manifest_sha256"]:
                raise ValueError("test split manifest does not match locked-config hash")
        args.fixed_budget = str(locked_payload.get("fixed_budget", args.fixed_budget))
        args.fixed_gamma = int(locked_payload.get("fixed_gamma", args.fixed_gamma))
        args.gamma_reference = int(locked_payload.get("gamma_reference", args.gamma_reference))
        calibration_path = _path(args.calibration_file or locked_payload.get("calibration_path"))
        if not calibration_path or sha256_file(calibration_path) != locked_payload.get("calibration_sha256"):
            raise ValueError("test calibration file does not match locked-config hash")
    else:
        calibration_path = _path(args.calibration_file)

    target_value = args.target_model or os.environ.get("MODEL_TARGET") or os.environ.get("TARGET_MODEL")
    draft_value = args.draft_model or os.environ.get("MODEL_DFLASH_DRAFT") or os.environ.get("DRAFT_MODEL")
    target_path = str(_path(target_value)) if target_value else None
    draft_path = str(_path(draft_value)) if draft_value else None
    if not target_path:
        raise ValueError("--target-model or MODEL_TARGET is required")
    runtime = load_runtime(target_path, draft_path, load_draft=(variant != "ar"))
    if variant != "ar" and runtime["draft"] is None:
        raise ValueError("DFlash draft model did not load")
    signature_max_new = args.max_new_tokens or int(os.environ.get("CAD_MAX_NEW_TOKENS", os.environ.get("RUN_MAX_NEW_TOKENS", "512")))
    if args.fixed_output_tokens is not None:
        signature_max_new = max(signature_max_new, args.fixed_output_tokens)
    implementation = _implementation_manifest()
    signature = calibration_model_signature(runtime, config, temperature=args.temperature, max_new_tokens=signature_max_new)
    run_id = args.run_id or output_root.name
    signature["run_id"] = run_id
    signature["max_input_tokens"] = max(args.max_input_tokens, 0)
    signature["split_manifest_sha256"] = sha256_file(_path(args.split_manifest)) if args.split_manifest else None
    signature["dataset_manifest_sha256"] = hashlib.sha256(
        json.dumps(dataset_manifest, sort_keys=True, default=str).encode()
    ).hexdigest()
    signature["fixed_output_tokens"] = args.fixed_output_tokens
    signature["output_scope"] = "fixed_tokens" if args.fixed_output_tokens is not None else "natural_eos"
    signature["implementation_sha256"] = implementation["sha256"]
    if phase == "test" and locked_payload.get("implementation_sha256") != implementation["sha256"]:
        raise ValueError("test implementation does not match locked config")
    if phase == "test":
        locked_calibration = _load_json(calibration_path)
        locked_signature = locked_calibration.get("signature", {})
        for key in ("target_signature", "tokenizer_signature", "runtime_signature", "run_id"):
            if signature.get(key) != locked_signature.get(key):
                raise ValueError(f"test runtime does not match calibration lock: {key}")
    calibration = None
    if phase in {"dev", "test"} and variant in ADAPTIVE_VARIANTS:
        if calibration_path is None:
            raise ValueError(f"variant {variant} requires --calibration-file")
        calibration = _load_json(calibration_path)
        issues = validate_calibration(calibration, signature)
        if issues:
            raise ValueError("invalid or incompatible calibration artifact: " + "; ".join(issues))

    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        max_new_tokens = signature_max_new
    if phase == "smoke":
        max_new_tokens = min(max_new_tokens, 32)
    if args.fixed_output_tokens is not None:
        if args.fixed_output_tokens < 1:
            raise ValueError("--fixed-output-tokens must be positive")
        max_new_tokens = args.fixed_output_tokens
    repetitions = args.repetitions if args.repetitions is not None else (3 if phase in {"dev", "test", "calibrate"} else 1)
    warmups = args.warmup_runs if args.warmup_runs is not None else (1 if phase == "smoke" else 3)
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be positive and warmup-runs nonnegative")
    fixed_budget: int | str = "full" if str(args.fixed_budget) == "full" else int(args.fixed_budget)

    config_values = config.to_dict()
    run_signature = {
        "target": runtime["target_signature"], "draft": runtime.get("draft_signature"),
        "tokenizer": runtime["tokenizer_signature"], "runtime": runtime["runtime_signature"]["signature"],
        "config": config_values, "phase": phase, "variant": variant, "run_id": run_id,
        "temperature": args.temperature, "max_new_tokens": max_new_tokens,
        "max_input_tokens": max(args.max_input_tokens, 0),
        "fixed_output_tokens": args.fixed_output_tokens,
        "fixed_budget": fixed_budget, "fixed_gamma": args.fixed_gamma,
        "gamma_reference": args.gamma_reference,
        "dataset_manifest": signature["dataset_manifest_sha256"],
        "split_manifest": signature["split_manifest_sha256"],
        "implementation": implementation["sha256"],
        "selected_sample_ids": hashlib.sha256(
            "\n".join(str(sample["_cad_record_key"]) for sample in selected).encode()
        ).hexdigest(),
        "repetitions": repetitions,
        "warmup_runs": warmups,
        "calibration": sha256_file(calibration_path) if calibration_path and calibration_path.is_file() else None,
    }
    config_hash = hashlib.sha256(json.dumps(run_signature, sort_keys=True, default=str).encode()).hexdigest()
    manifest = {
        "schema_version": "cadflash.manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "phase": phase,
        "variant": variant,
        "test_variants": [
            "ar", "dflash_full_fixed", "best_fixed_pair", "a_only",
            "b_only_history", "b_only_entropy", "independent_ab", variant,
            "joint_no_entropy", "joint_no_source_relevance",
        ],
        "config_hash": config_hash,
        "code": _git_provenance(),
        "implementation": implementation,
        "data": dataset_manifest,
        "split_manifest_sha256": signature["split_manifest_sha256"],
        "split_status": split_manifest.get("status") if split_manifest else None,
        "split_group_counts": {
            key: len(value) for key, value in split_manifest.get("splits", {}).items()
        } if split_manifest else None,
        "target_asset": runtime.get("target_asset"),
        "draft_asset": runtime.get("draft_asset"),
        "tokenizer_signature": runtime["tokenizer_signature"],
        "runtime": runtime["runtime_signature"],
        "dtype": runtime["dtype"],
        "attention_backend": runtime["attention_backend"],
        "device": runtime["device"],
        "calibration_sha256": run_signature["calibration"],
        "calibration_signature": signature if phase == "calibrate" else None,
        "config": config_values,
        "fixed_action": {
            "budget": fixed_budget, "gamma": args.fixed_gamma,
            "gamma_reference": args.gamma_reference,
        },
        "sampling": {
            "temperature": args.temperature,
            "max_new_tokens": max_new_tokens,
            "fixed_output_tokens": args.fixed_output_tokens,
            "output_scope": signature["output_scope"],
            "seed": args.seed,
        },
        "repetitions": repetitions,
        "warmup_runs": warmups,
        "selected_sample_count": len(selected),
        "request_prepare_and_model_load_in_e2e": False,
    }
    if phase == "calibrate":
        _persist_manifest(output_root / "calibration" / "run_manifest.json", manifest)
    else:
        cell_root = output_root / phase / variant / config.statistics_update_mode
        cell_root.mkdir(parents=True, exist_ok=True)
        _persist_manifest(cell_root / "manifest.json", manifest)

    if phase == "calibrate":
        calibration_root = output_root / "calibration"
        if (calibration_root / "action_observations.jsonl").exists() or (calibration_root / "calibration.json").exists():
            raise FileExistsError(f"calibration artifacts already exist under {calibration_root}; use a new output root")
        action_rows: list[dict[str, Any]] = []
        for warmup in range(warmups):
            sample = selected[warmup % len(selected)]
            ids, layout = prepare_prompt(sample, runtime["tokenizer"], chunk_size=config.source_chunk_size, max_tokens=max(args.max_input_tokens, 0))
            seed_everything(args.seed)
            generate_adaptive(
                runtime["target"], runtime["draft"], ids, runtime["tokenizer"], layout, config,
                max_new_tokens=min(max_new_tokens, 16), temperature=0.0, variant="dflash_full_fixed",
                fixed_budget="full", fixed_gamma=args.fixed_gamma, run_id=run_id,
                sample_id=str(sample["id"]), dataset=str(sample["dataset"]), split="calibration_warmup",
                stop_token_ids=() if args.fixed_output_tokens is not None else None,
            )
        for sample_index, sample in enumerate(selected):
            for repetition in range(repetitions):
                ids, layout = prepare_prompt(sample, runtime["tokenizer"], chunk_size=config.source_chunk_size, max_tokens=max(args.max_input_tokens, 0))
                seed_everything(args.seed + repetition)
                row_start = len(action_rows)
                result = generate_adaptive(
                    runtime["target"], runtime["draft"], ids, runtime["tokenizer"], layout, config,
                    max_new_tokens=max_new_tokens, temperature=0.0, variant="dflash_full_fixed",
                    fixed_budget="full", fixed_gamma=args.fixed_gamma, gamma_reference=args.gamma_reference,
                    run_id=run_id, sample_id=str(sample["id"]), dataset=str(sample["dataset"]),
                    split="calibration", repetition=repetition, calibration_action_rows=action_rows,
                    calibration_checkpoints=tuple(int(value) for value in args.calibration_checkpoints.split(",") if value.strip()),
                    stop_token_ids=() if args.fixed_output_tokens is not None else None,
                )
                if result.status != "ok":
                    raise RuntimeError(f"calibration trajectory failed for {sample['id']}: {result.error}")
                for row in action_rows[row_start:]:
                    row.update({
                        "sample_id": str(sample["id"]),
                        "dataset": str(sample["dataset"]),
                        "source_group_id": str(sample.get("_cad_source_group_id")),
                        "allocation_stratum": str(sample.get("_cad_allocation_stratum", sample["dataset"])),
                        "prompt_hash": layout.prompt_hash,
                        "repetition": repetition,
                        "split": "calibration",
                    })
                print(f"[calibrate] {sample_index + 1}/{len(selected)} {sample['dataset']}:{sample['id']} action_rows={len(action_rows)}", flush=True)
        calibration = fit_action_priors(action_rows, signature=signature, config=config)
        calibration = profile_controller_costs(
            calibration, action_rows, config, fixed_budget=fixed_budget,
            fixed_gamma=args.fixed_gamma, gamma_reference=args.gamma_reference,
        )
        issues = validate_calibration(calibration, signature)
        if issues:
            raise ValueError("generated calibration artifact is invalid: " + "; ".join(issues))
        calibration_root.mkdir(parents=True, exist_ok=True)
        obs_path = calibration_root / "action_observations.jsonl"
        if obs_path.exists():
            raise FileExistsError(f"calibration observations already exist: {obs_path}; use a new output root")
        observation_writer = io_util.JsonlWriter(obs_path)
        for row in action_rows:
            observation_writer.add(row)
        observation_writer.finalize({"type": "summary", "status": "complete", "rows": len(action_rows)})
        atomic_json(calibration_root / "calibration.json", calibration)
        manifest["calibration_sha256"] = sha256_file(calibration_root / "calibration.json")
        manifest["action_observations_sha256"] = sha256_file(obs_path)
        atomic_json(calibration_root / "run_manifest.json", manifest)
        print(f"Calibration artifact: {calibration_root / 'calibration.json'} ({len(action_rows)} replay rows)", flush=True)
        return 0

    if phase == "test" and _load_json(_path(args.locked_config)).get("status") == "candidate_locked_pending_heldout":
        print("[test] using candidate lock; G0-G5 evidence remains pending in the lock artifact", flush=True)

    mode = config.statistics_update_mode
    if warmups:
        warmup_statistics = None
        if calibration is not None:
            warmup_statistics = AdaptiveStatistics(
                calibration,
                prior_strength=config.prior_strength,
                decay=config.statistics_decay,
                online=False,
                min_state_support=config.min_state_support,
                min_cost_repetitions=3,
            )
        for warmup in range(warmups):
            sample = selected[warmup % len(selected)]
            ids, layout = prepare_prompt(
                sample, runtime["tokenizer"], chunk_size=config.source_chunk_size,
                max_tokens=max(args.max_input_tokens, 0),
            )
            seed_everything(args.seed + warmup)
            if variant == "ar":
                generate_target_only(
                    runtime["target"], ids, max_new_tokens=min(max_new_tokens, 16),
                    temperature=args.temperature,
                    stop_token_ids=() if args.fixed_output_tokens is not None else None,
                )
            else:
                generate_adaptive(
                    runtime["target"], runtime["draft"], ids, runtime["tokenizer"], layout, config,
                    max_new_tokens=min(max_new_tokens, 16), temperature=args.temperature,
                    statistics=warmup_statistics, variant=variant, fixed_budget=fixed_budget,
                    fixed_gamma=args.fixed_gamma, gamma_reference=args.gamma_reference,
                    run_id=run_id, sample_id=str(sample["id"]), dataset=str(sample["dataset"]),
                    split="warmup", repetition=warmup,
                    stop_token_ids=() if args.fixed_output_tokens is not None else None,
                )
            print(f"[{phase} warmup {warmup + 1}/{warmups}] complete", flush=True)
    phase_exit_code = 0
    for repetition in range(repetitions):
        rep_root = cell_root / f"rep_{repetition}"
        rep_root.mkdir(parents=True, exist_ok=True)
        request_path = rep_root / "requests.jsonl"
        rounds_path = rep_root / "rounds.jsonl"
        token_path = rep_root / "output_token_ids.jsonl"
        completed, existing = _stream_prepare(request_path, resume=args.resume, config_hash=config_hash)
        if rounds_path.exists() and args.resume:
            _prune_request_artifact(rounds_path, completed)
        elif rounds_path.exists() and not args.resume:
            raise FileExistsError(f"refusing to append to existing artifact: {rounds_path}")
        if token_path.exists() and args.resume:
            _prune_request_artifact(token_path, completed)
        elif token_path.exists() and not args.resume:
            raise FileExistsError(f"refusing to append to existing artifact: {token_path}")
        req_writer = io_util.JsonlWriter(request_path)
        round_writer = io_util.JsonlWriter(rounds_path)
        token_writer = io_util.JsonlWriter(token_path)
        rep_rows = list(existing)
        for sample_index, sample in enumerate(selected):
            key = (str(sample["dataset"]), str(sample["id"]), repetition)
            if key in completed:
                continue
            try:
                ids, layout = prepare_prompt(
                    sample, runtime["tokenizer"], chunk_size=config.source_chunk_size,
                    max_tokens=max(args.max_input_tokens, 0),
                )
                if layout.source_group_id is None:
                    object.__setattr__(layout, "source_group_id", str(sample.get("_cad_source_group_id")))
                if variant == "ar":
                    seed_everything(args.seed + repetition)
                    result = generate_target_only(
                        runtime["target"], ids, max_new_tokens=max_new_tokens, temperature=args.temperature,
                        stop_token_ids=() if args.fixed_output_tokens is not None else None,
                    )
                    baseline_ids = None
                else:
                    request_statistics = None
                    if calibration is not None:
                        request_statistics = AdaptiveStatistics(
                            calibration,
                            prior_strength=config.prior_strength,
                            decay=config.statistics_decay,
                            online=config.statistics_update_mode == "online",
                            min_state_support=config.min_state_support,
                            min_cost_repetitions=3,
                        )
                    seed_everything(args.seed + repetition)
                    result = generate_adaptive(
                        runtime["target"], runtime["draft"], ids, runtime["tokenizer"], layout, config,
                        max_new_tokens=max_new_tokens, temperature=args.temperature, statistics=request_statistics,
                        variant=variant, fixed_budget=fixed_budget, fixed_gamma=args.fixed_gamma,
                        gamma_reference=args.gamma_reference, run_id=run_id, sample_id=str(sample["id"]),
                        dataset=str(sample["dataset"]), split=str(sample.get("_cad_split")), repetition=repetition,
                        stop_token_ids=() if args.fixed_output_tokens is not None else None,
                    )
                    baseline_ids = None
                record = request_record(
                    result=result, sample=sample, input_ids=ids, layout=layout, prompt_hash=layout.prompt_hash,
                    variant=variant, run_id=run_id, repetition=repetition,
                    model_path=target_path, draft_path=(draft_path if variant != "ar" else None),
                    runtime=runtime, temperature=args.temperature, max_new_tokens=max_new_tokens,
                    fixed_output_tokens=args.fixed_output_tokens, config_hash=config_hash,
                    baseline_ids=baseline_ids,
                )
                if result.status != "ok":
                    record["status"] = "error"
                    record["type"] = "error"
                record.update({
                    "phase": phase,
                    "implementation_sha256": implementation["sha256"],
                    "fixed_budget": fixed_budget,
                    "fixed_gamma": args.fixed_gamma,
                    "gamma_reference": args.gamma_reference,
                    "statistics_update_mode": config.statistics_update_mode,
                    "cost_update_mode": config.cost_update_mode,
                    "timing_mode": config.timing_mode,
                    "split_manifest_sha256": signature["split_manifest_sha256"],
                    "dataset_manifest_sha256": signature["dataset_manifest_sha256"],
                    "calibration_sha256": run_signature["calibration"],
                })
                problems = validate_request(record)
                if problems:
                    raise ValueError("request schema validation failed: " + "; ".join(problems))
                prepared_rounds = []
                for round_row in result.rounds:
                    round_row.update({
                        "run_id": run_id, "sample_id": str(sample["id"]), "dataset": str(sample["dataset"]),
                        "split": str(sample.get("_cad_split", phase)), "variant": variant,
                        "repetition": repetition, "source_group_id": sample.get("_cad_source_group_id"),
                    })
                    problems = validate_round(round_row)
                    if problems:
                        raise ValueError("round schema validation failed: " + "; ".join(problems))
                    prepared_rounds.append(round_row)
                req_writer.add(record)
                rep_rows.append(record)
                token_writer.add({
                    "schema_version": "cadflash.output.v1", "type": "output_tokens", "run_id": run_id,
                    "sample_id": str(sample["id"]), "dataset": str(sample["dataset"]),
                    "variant": variant, "repetition": repetition, "prompt_hash": layout.prompt_hash,
                    "output_ids": [int(value) for value in result.output_ids.reshape(-1).tolist()],
                })
                for round_row in prepared_rounds:
                    round_writer.add(round_row)
                print(f"[{phase} {sample_index + 1}/{len(selected)} rep={repetition}] {sample['dataset']}:{sample['id']} status={record['status']} tokens={record['output_tokens']} e2e_ms={record['e2e_ms']}", flush=True)
            except Exception as exc:
                error = _error_record(sample, variant, run_id, repetition, target_path, draft_path, exc)
                error.update({
                    "phase": phase,
                    "implementation_sha256": implementation["sha256"],
                    "fixed_budget": fixed_budget,
                    "fixed_gamma": args.fixed_gamma,
                    "gamma_reference": args.gamma_reference,
                    "statistics_update_mode": config.statistics_update_mode,
                    "cost_update_mode": config.cost_update_mode,
                    "timing_mode": config.timing_mode,
                    "output_scope": "fixed_tokens" if args.fixed_output_tokens is not None else "natural_eos",
                    "output_cap": max_new_tokens,
                    "generation_temperature": float(args.temperature),
                    "split_manifest_sha256": signature["split_manifest_sha256"],
                    "dataset_manifest_sha256": signature["dataset_manifest_sha256"],
                    "calibration_sha256": run_signature["calibration"],
                    "config_hash": config_hash,
                })
                req_writer.add(error)
                rep_rows.append(error)
                print(f"[{phase} error] {sample.get('dataset')}:{sample.get('id')} {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        success_rows = [row for row in rep_rows if row.get("type") == "sample" and row.get("status") == "ok"]
        summary = {
            "type": "summary", "schema_version": "cadflash.summary.v1",
            "status": "complete" if len(success_rows) == len(selected) else "partial",
            "method": "context_adaptive_dflash", "variant": variant, "phase": phase,
            "run_id": run_id, "requested_samples": len(selected), "successes": len(success_rows),
            "errors": sum(row.get("type") == "error" for row in rep_rows),
            "repetition": repetition, "config_hash": config_hash,
            "statistics_update_mode": config.statistics_update_mode,
            "cost_update_mode": config.cost_update_mode,
            "output_scope": "fixed_tokens" if args.fixed_output_tokens is not None else "natural_eos",
            "rouge": rouge.aggregate_rouge(success_rows),
            "mean_e2e_ms": sum(float(row["e2e_ms"]) for row in success_rows if row.get("e2e_ms") is not None) / len(success_rows) if success_rows else None,
        }
        req_writer.finalize(summary)
        round_writer.finalize({"type": "summary", "schema_version": "cadflash.round_summary.v1", "status": summary["status"], "round_count": sum(int(row.get("round_count", 0) or 0) for row in success_rows), "request_count": len(success_rows)})
        token_writer.finalize({
            "type": "summary", "schema_version": "cadflash.output_summary.v1",
            "status": summary["status"], "request_count": len(success_rows),
            "output_scope": summary["output_scope"],
        })
        if summary["status"] != "complete":
            phase_exit_code = 1
    return phase_exit_code


def _prune_request_artifact(path: Path, completed: set[tuple[str, str, int]]) -> None:
    from TrainingFree.context_adaptive.schema import read_jsonl
    rows = read_jsonl(path)
    kept = [
        row for row in rows
        if row.get("type") != "summary"
        and _request_key(row) in completed
    ]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept), encoding="utf-8")


def _smoke_deduplicate(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, set[str]] = {}
    output = []
    for sample in samples:
        dataset = str(sample.get("dataset", "unknown"))
        raw = sample.get("raw") or {}
        context = " ".join(str(raw.get("context", "")).split())
        group = hashlib.sha256((context or str(sample.get("_cad_record_key", sample.get("id")))).encode()).hexdigest()
        dataset_groups = seen.setdefault(dataset, set())
        if group in dataset_groups or len(dataset_groups) >= 2:
            continue
        dataset_groups.add(group)
        sample["_cad_source_group_id"] = group
        sample["_cad_split"] = "smoke"
        sample["_cad_allocation_stratum"] = dataset
        output.append(sample)
    if not output:
        raise ValueError("smoke selection has no distinct source documents")
    return output


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.smoke and args.full:
        parser.error("--smoke and --full cannot both be set")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.max_input_tokens < 0:
        parser.error("--max-input-tokens must be nonnegative")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.fixed_gamma is not None and args.fixed_gamma < 1:
        parser.error("--fixed-gamma must be positive")
    if args.gamma_reference is not None and args.gamma_reference < 1:
        parser.error("--gamma-reference must be positive")
    if args.temperature < 0:
        parser.error("--temperature must be nonnegative")
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be nonnegative")
    if args.smoke:
        args.phase = "smoke"
    elif args.full:
        args.phase = "test"
    args.fixed_budget_was_set = args.fixed_budget is not None
    args.fixed_gamma_was_set = args.fixed_gamma is not None
    args.gamma_reference_was_set = args.gamma_reference is not None
    if args.fixed_budget is None:
        args.fixed_budget = os.environ.get("CAD_FIXED_BUDGET", "full")
    if args.fixed_gamma is None:
        args.fixed_gamma = int(os.environ.get("CAD_FIXED_GAMMA", "15"))
    if args.gamma_reference is None:
        args.gamma_reference = int(os.environ.get("CAD_GAMMA_REFERENCE", "15"))
    if args.fixed_gamma < 1 or args.gamma_reference < 1:
        parser.error("CAD_FIXED_GAMMA and CAD_GAMMA_REFERENCE must be positive")
    if args.fixed_output_tokens is not None and args.fixed_output_tokens < 1:
        parser.error("--fixed-output-tokens must be positive")
    if args.phase == "smoke" and args.fixed_output_tokens is not None and args.fixed_output_tokens > 32:
        parser.error("smoke fixed-output cap cannot exceed 32 tokens")
    if args.phase == "test" and args.variant is None and args.locked_config:
        locked_payload = _load_json(_path(args.locked_config))
        args.variant = str(locked_payload.get("variant", os.environ.get("CAD_VARIANT", "joint")))
    variant = _variant(args)
    config = _config(args, phase=args.phase)
    output_root = _path(args.output_root) or (ROOT / config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    split_path = _path(args.split_manifest) if args.split_manifest else None

    if args.phase == "report":
        from TrainingFree.context_adaptive.report import report_artifacts
        result = report_artifacts(
            output_root,
            phase=args.report_phase,
            statistics_mode=args.report_statistics_mode,
            bootstrap_samples=args.bootstrap_samples,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return 0
    if args.phase == "lock":
        return _lock(args, output_root, config, variant)

    data_file, data_dir, source_paths = _data_source(args)
    if args.phase == "prepare":
        return _prepare(args, output_root, source_paths)
    if args.phase == "preflight":
        return _preflight(args, output_root, source_paths, split_path)
    if args.phase not in MODEL_PHASES:
        parser.error(f"unsupported phase {args.phase}")
    return _run_model_phase(args, args.phase, variant, output_root, source_paths, config)


if __name__ == "__main__":
    raise SystemExit(main())
