#!/usr/bin/env python3
"""Offline preflight for SyncSpec on the canonical B200 server."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


def _resolve_asset(value: str | None) -> dict:
    if not value:
        return {"provided": False, "exists": False, "reason": "not_set"}
    path = Path(value).expanduser()
    if not path.is_absolute():
        repo_path = ROOT / path
        if repo_path.exists() or value.startswith((".", "data/", "outputs/", "checkpoints/")):
            path = repo_path
        else:
            path = None
    if path is None:
        try:
            from common.paths import snapshot_dir
            cached = snapshot_dir(value)
        except Exception:
            cached = None
        path = cached if cached is not None else Path(value).expanduser()
    return {"provided": True, "exists": path.exists(), "path": str(path)}


def _read_model_config(asset: dict) -> dict | None:
    if not asset.get("exists"):
        return None
    path = Path(asset["path"]) / "config.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_component_config(asset: dict, filename: str) -> dict | None:
    if not asset.get("exists") or not Path(asset["path"]).is_dir():
        return None
    try:
        value = json.loads((Path(asset["path"]) / filename).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _tokenizer_files(asset: dict) -> list[str]:
    if not asset.get("exists"):
        return []
    root = Path(asset["path"])
    # tokenizer_config.json only contains metadata; AutoTokenizer still needs
    # one of these vocabulary/serialization artifacts to load offline.
    names = (
        "tokenizer.json", "tokenizer.model", "spiece.model",
        "sentencepiece.bpe.model", "vocab.json", "vocab.txt",
    )
    return [name for name in names if (root / name).is_file()]


def _model_weights_present(asset: dict) -> bool:
    """Accept the common Hugging Face single-file and sharded layouts."""
    if not asset.get("exists") or not Path(asset["path"]).is_dir():
        return False
    root = Path(asset["path"])
    names = (
        "pytorch_model.bin", "model.safetensors", "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    )
    return any((root / name).is_file() for name in names) or any(root.glob("*.safetensors"))


def _missing_component_artifacts(asset: dict, names: tuple[str, ...]) -> list[str]:
    if not asset.get("exists") or not Path(asset["path"]).is_dir():
        return list(names)
    root = Path(asset["path"])
    return [name for name in names if not (root / name).is_file()]


def _profile_is_valid(
    asset: dict,
    *,
    expected_model: str | tuple[str, ...] | None = None,
    expected_checkpoint: str | tuple[str, ...] | None = None,
    expected_gpu: str | None = None,
    expected_precision: str | None = None,
    expected_batch_size: int | None = None,
    expected_selector_checkpoint: str | tuple[str, ...] | None = None,
    expected_survival_checkpoint: str | tuple[str, ...] | None = None,
) -> bool:
    """Check the structural contract before a profile can gate CUDA inference."""
    if not asset.get("exists") or not Path(asset["path"]).is_file():
        return False
    try:
        payload = json.loads(Path(asset["path"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    records = payload if isinstance(payload, list) else [payload]
    if not records:
        return False
    for record in records:
        if not isinstance(record, dict):
            return False
        if record.get("schema_version") != 1:
            return False
        if record.get("source") != "measured":
            return False
        key = record.get("key")
        measurements = record.get("measurements_ms")
        if not isinstance(key, dict) or not isinstance(measurements, dict):
            return False
        required_key = (
            "model", "checkpoint", "gpu", "precision", "context_bin",
            "batch_bin", "kd", "kv",
        )
        if any(field not in key for field in required_key):
            return False
        for actual, expected in (
            (key.get("model"), expected_model),
            (key.get("checkpoint"), expected_checkpoint),
            (key.get("selector_checkpoint"), expected_selector_checkpoint),
            (key.get("survival_checkpoint"), expected_survival_checkpoint),
        ):
            if expected is None:
                continue
            accepted = expected if isinstance(expected, tuple) else (expected,)
            if actual is None or str(actual) not in {str(value) for value in accepted}:
                return False
        if expected_gpu is not None:
            actual_gpu = str(key.get("gpu", "")).strip().lower()
            wanted_gpu = str(expected_gpu).strip().lower()
            if not actual_gpu or (actual_gpu not in wanted_gpu and wanted_gpu not in actual_gpu):
                return False
        if expected_precision is not None and str(key.get("precision", "")).lower() != str(expected_precision).lower():
            return False
        if expected_batch_size is not None and str(key.get("batch_bin")) != f"batch{int(expected_batch_size)}":
            return False
        if "target_ar" not in measurements or not (
            "verify" in measurements or "e2e" in measurements
        ):
            return False
    return True


def _cache_report() -> dict[str, dict]:
    """Check that local model/kernel caches have a writable directory."""
    configured = {
        "huggingface": os.environ.get("HF_HOME") or os.environ.get("FI_HF_HOME"),
        "transformers": os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("FI_TRANSFORMERS_CACHE"),
        "triton": os.environ.get("TRITON_CACHE_DIR") or os.environ.get("FI_TRITON_CACHE"),
        "flashinfer": os.environ.get("FLASHINFER_WORKSPACE_BASE") or os.environ.get("FI_FLASHINFER_CACHE"),
        "torch_extensions": os.environ.get("TORCH_EXTENSIONS_DIR") or os.environ.get("FI_TORCH_EXTENSIONS_CACHE"),
    }
    defaults = {
        "huggingface": "~/.cache/huggingface",
        "transformers": "~/.cache/huggingface/transformers",
        "triton": "~/.cache/triton",
        "flashinfer": "~/.cache/flashinfer",
        "torch_extensions": "~/.cache/torch_extensions",
    }
    report = {}
    for name, value in configured.items():
        raw = value or defaults[name]
        path = Path(raw).expanduser()
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        writable = probe.is_dir() and os.access(probe, os.W_OK)
        report[name] = {
            "path": str(path), "exists": path.exists(), "writable": writable,
            "probe": str(probe),
        }
    return report


def build_report(
    target_model: str | None = None,
    drafter_checkpoint: str | None = None,
    data_file: str | None = None,
    selector_checkpoint: str | None = None,
    survival_checkpoint: str | None = None,
    runtime_profile: str | None = None,
    target_gpu: str = "B200",
    phase: str = "infer",
    min_compute_major: int = 10,
    precision: str | None = None,
    batch_size: int | None = None,
) -> dict:
    if phase not in {"infer", "train"}:
        raise ValueError("phase must be infer or train")
    target_model = target_model or os.environ.get("SYNCSPEC_TARGET_MODEL") or os.environ.get("MODEL_TARGET")
    drafter_checkpoint = drafter_checkpoint or os.environ.get("SYNCSPEC_DRAFTER_CHECKPOINT")
    data_file = data_file or os.environ.get("SYNCSPEC_DATA_FILE") or os.environ.get("DATA_INPUT")
    selector_checkpoint = selector_checkpoint or os.environ.get("SYNCSPEC_SELECTOR_CHECKPOINT")
    survival_checkpoint = survival_checkpoint or os.environ.get("SYNCSPEC_SURVIVAL_CHECKPOINT")
    runtime_profile = runtime_profile or os.environ.get("SYNCSPEC_PROFILE")
    precision = precision or os.environ.get("DTYPE") or "bfloat16"
    if batch_size is None:
        raw_batch_size = os.environ.get("BATCH_SIZE") or os.environ.get("SYNCSPEC_BATCH_SIZE")
        batch_size = int(raw_batch_size) if raw_batch_size else None
    python = {
        "path": sys.executable,
        "version": ".".join(str(x) for x in sys.version_info[:3]),
        "ok": sys.version_info[:2] == (3, 12),
    }
    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        cuda = {
            "available": cuda_available,
            "reason": "ok" if cuda_available else "hardware_unavailable",
            "devices": [], "capability_match": False,
        }
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                cuda["devices"].append({"index": index, "name": str(props.name), "major": int(props.major), "minor": int(props.minor), "memory_gb": round(props.total_memory / 1024**3, 2)})
            cuda["target_match"] = any(target_gpu.lower() in item["name"].lower() for item in cuda["devices"])
            cuda["capability_match"] = any(
                item["major"] >= int(min_compute_major) for item in cuda["devices"]
            )
            try:
                probe = torch.zeros(1, device="cuda") + 1
                torch.cuda.synchronize()
                cuda["tensor_probe"] = {"ok": bool(probe.item() == 1)}
            except Exception as exc:
                cuda["reason"] = "cuda_tensor_probe_failed"
                cuda["tensor_probe"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        else:
            cuda["target_match"] = False
    except Exception as exc:
        torch = None
        cuda = {
            "available": False, "reason": "torch_unavailable",
            "error": f"{type(exc).__name__}: {exc}", "target_match": False,
            "capability_match": False,
        }

    imports = {}
    for name in ("torch", "transformers"):
        try:
            module = importlib.import_module(name)
            imports[name] = {"ok": True, "version": str(getattr(module, "__version__", "unknown"))}
        except Exception as exc:
            imports[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    assets = {
        "target_model": _resolve_asset(target_model),
        "drafter_checkpoint": _resolve_asset(drafter_checkpoint),
        "data_file": _resolve_asset(data_file),
        "selector_checkpoint": _resolve_asset(selector_checkpoint),
        "survival_checkpoint": _resolve_asset(survival_checkpoint),
        "runtime_profile": _resolve_asset(runtime_profile),
    }
    caches = _cache_report()
    target_config = _read_model_config(assets["target_model"])
    drafter_config = _read_model_config(assets["drafter_checkpoint"])
    selector_config = _read_component_config(
        assets["selector_checkpoint"], "selector_config.json",
    )
    target_tokenizer = _tokenizer_files(assets["target_model"])
    artifact_checks = {}
    for name in ("target_model", "drafter_checkpoint"):
        asset = assets[name]
        artifact_checks[name] = {
            "exists": bool(asset.get("exists")),
            "config": bool(asset.get("exists") and (Path(asset["path"]) / "config.json").is_file()),
            "model_weights": _model_weights_present(asset),
        }
    artifact_checks["selector_checkpoint"] = {
        "exists": bool(assets["selector_checkpoint"].get("exists")),
        "config": selector_config is not None,
        "missing": _missing_component_artifacts(
            assets["selector_checkpoint"], ("selector.pt", "selector_config.json")
        ),
    }
    artifact_checks["survival_checkpoint"] = {
        "exists": bool(assets["survival_checkpoint"].get("exists")),
        "missing": _missing_component_artifacts(
            assets["survival_checkpoint"], ("survival.pt",)
        ),
    }
    profile_model_values = tuple(
        value for value in (
            target_model, assets["target_model"].get("path"),
        ) if value
    )
    profile_checkpoint_values = tuple(
        value for value in (
            drafter_checkpoint, assets["drafter_checkpoint"].get("path"),
        ) if value
    )
    profile_selector_values = tuple(
        value for value in (
            selector_checkpoint, assets["selector_checkpoint"].get("path"),
        ) if value
    )
    profile_survival_values = tuple(
        value for value in (
            survival_checkpoint, assets["survival_checkpoint"].get("path"),
        ) if value
    )
    artifact_checks["runtime_profile"] = {
        "exists": bool(assets["runtime_profile"].get("exists")),
        "valid": _profile_is_valid(
            assets["runtime_profile"],
            expected_model=profile_model_values or None,
            expected_checkpoint=profile_checkpoint_values or None,
            expected_gpu=target_gpu,
            expected_precision=precision,
            expected_batch_size=batch_size,
            expected_selector_checkpoint=profile_selector_values or None,
            expected_survival_checkpoint=profile_survival_values or None,
        ),
    }
    compatibility = {"status": "not_checked", "mismatches": []}
    compatibility_checked = False
    if assets["target_model"]["provided"] and assets["target_model"]["exists"]:
        compatibility_checked = True
        if target_config is None:
            compatibility["mismatches"].append("target_config_missing_or_invalid")
        if not target_tokenizer:
            compatibility["mismatches"].append("target_tokenizer_missing")
    if assets["drafter_checkpoint"]["provided"] and assets["drafter_checkpoint"]["exists"]:
        compatibility_checked = True
        if drafter_config is None:
            compatibility["mismatches"].append("drafter_config_missing_or_invalid")
    if target_config is not None and drafter_config is not None:
        for field in ("vocab_size", "hidden_size"):
            target_value = target_config.get(field)
            drafter_value = drafter_config.get(field)
            if target_value is not None and drafter_value is not None and int(target_value) != int(drafter_value):
                compatibility["mismatches"].append(
                    f"{field}_mismatch_target_{target_value}_drafter_{drafter_value}"
                )
        target_positions = target_config.get("max_position_embeddings")
        drafter_positions = drafter_config.get("max_positions")
        if (
            target_positions is not None and drafter_positions is not None
            and int(drafter_positions) < int(target_positions)
        ):
            compatibility["mismatches"].append(
                "max_positions_insufficient_"
                f"drafter_{drafter_positions}_target_{target_positions}"
            )
    if selector_config is not None:
        compatibility_checked = True
        selector_vocab = selector_config.get("vocab_size")
        selector_hidden = selector_config.get("hidden_size")
        target_vocab = target_config.get("vocab_size") if target_config else None
        drafter_hidden = drafter_config.get("hidden_size") if drafter_config else None
        if selector_vocab is not None and target_vocab is not None and int(selector_vocab) != int(target_vocab):
            compatibility["mismatches"].append(
                f"selector_vocab_size_mismatch_target_{target_vocab}_selector_{selector_vocab}"
            )
        if selector_hidden is not None and drafter_hidden is not None and int(selector_hidden) != int(drafter_hidden):
            compatibility["mismatches"].append(
                f"selector_hidden_size_mismatch_drafter_{drafter_hidden}_selector_{selector_hidden}"
            )
    compatibility["status"] = (
        "FAIL" if compatibility["mismatches"]
        else "PASS" if compatibility_checked else "not_checked"
    )
    errors = []
    if not python["ok"]:
        errors.append("python_3.12_required")
    if not cuda.get("available"):
        errors.append("hardware_unavailable")
    elif cuda.get("reason") != "ok":
        errors.append(str(cuda.get("reason")))
    elif not cuda.get("target_match"):
        errors.append(f"target_gpu_{target_gpu}_not_found")
    elif not cuda.get("capability_match"):
        errors.append(f"compute_capability_below_sm{min_compute_major}0")
    if not all(item["ok"] for item in imports.values()):
        errors.append("required_import_unavailable")
    required_assets = (
        ("target_model", "data_file")
        if phase == "train"
        else (
            "target_model", "drafter_checkpoint", "selector_checkpoint",
            "survival_checkpoint", "data_file", "runtime_profile",
        )
    )
    optional_assets = ("selector_checkpoint", "survival_checkpoint", "runtime_profile")
    for name in required_assets:
        if not assets[name]["provided"]:
            errors.append(f"{name}_not_set")
        elif not assets[name]["exists"]:
            errors.append(f"{name}_not_found")
    if phase in {"train", "infer"} and assets["target_model"]["provided"]:
        if not artifact_checks["target_model"]["model_weights"]:
            errors.append("target_model_artifacts_missing")
    if phase == "infer":
        if assets["drafter_checkpoint"]["exists"] and not artifact_checks["drafter_checkpoint"]["model_weights"]:
            errors.append("drafter_checkpoint_artifacts_missing")
        if assets["selector_checkpoint"]["exists"] and artifact_checks["selector_checkpoint"]["missing"]:
            errors.append("selector_checkpoint_artifacts_missing")
        elif assets["selector_checkpoint"]["exists"] and selector_config is None:
            errors.append("selector_checkpoint_config_invalid")
        if assets["survival_checkpoint"]["exists"] and artifact_checks["survival_checkpoint"]["missing"]:
            errors.append("survival_checkpoint_artifacts_missing")
        if assets["runtime_profile"]["exists"] and not artifact_checks["runtime_profile"]["valid"]:
            errors.append("runtime_profile_invalid")
    for name in optional_assets:
        if name not in required_assets and assets[name]["provided"] and not assets[name]["exists"]:
            errors.append(f"{name}_not_found")
    errors.extend(compatibility["mismatches"])
    if not all(item["writable"] for item in caches.values()):
        errors.append("cache_not_writable")
    status = "PASS" if not errors else ("BLOCKED" if "hardware_unavailable" in errors else "FAIL")
    return {
        "schema_version": 1,
        "method": "syncspec",
        "target_gpu": target_gpu,
        "precision": precision,
        "batch_size": batch_size,
        "phase": phase,
        "min_compute_major": int(min_compute_major),
        "status": status,
        "errors": errors,
        "interpreter": python,
        "cuda": cuda,
        "imports": imports,
        "assets": assets,
        "artifact_checks": artifact_checks,
        "target_tokenizer_files": target_tokenizer,
        "compatibility": compatibility,
        "offline": True,
        "caches": caches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model")
    parser.add_argument("--drafter-checkpoint")
    parser.add_argument("--data-file")
    parser.add_argument("--selector-checkpoint")
    parser.add_argument("--survival-checkpoint")
    parser.add_argument("--profile")
    parser.add_argument("--target-gpu", default=os.environ.get("B200_TARGET_GPU", "B200"))
    parser.add_argument("--precision", default=os.environ.get("DTYPE", "bfloat16"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--phase", choices=("infer", "train"), default="infer")
    parser.add_argument("--min-compute-major", type=int,
                        default=int(os.environ.get("B200_MIN_COMPUTE_MAJOR", "10")))
    parser.add_argument("--output", type=Path, default=Path("outputs/syncspec_b200_preflight.json"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = build_report(
        args.target_model, args.drafter_checkpoint, args.data_file,
        args.selector_checkpoint, args.survival_checkpoint, args.profile, args.target_gpu,
        args.phase,
        args.min_compute_major,
        args.precision,
        args.batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.strict or report["status"] == "PASS":
        return 0
    # Keep a distinct exit code for an unavailable external environment.  This
    # lets CI/orchestration distinguish a retryable B200 handoff block from a
    # real asset/configuration failure while retaining the structured report.
    return 2 if report["status"] == "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
