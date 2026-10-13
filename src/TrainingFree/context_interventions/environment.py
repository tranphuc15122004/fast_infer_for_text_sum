"""Offline B200 dependency, asset, data and kernel preflight."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def _model_asset_check(value: str | None, label: str) -> dict[str, Any]:
    if not value:
        return {"status": "asset_missing", "path": None, "reason": f"{label} path is unset"}
    path = Path(value).expanduser()
    if not path.is_dir():
        return {"status": "asset_missing", "path": str(path), "reason": "directory is absent"}
    config = path / "config.json"
    if not config.is_file():
        return {"status": "asset_missing", "path": str(path), "reason": "config.json is absent"}
    try:
        config_payload = json.loads(config.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "asset_missing", "path": str(path), "reason": f"invalid config.json: {exc}"}
    indexes = list(path.glob("*.safetensors.index.json")) + list(path.glob("pytorch_model.bin.index.json"))
    if indexes:
        for index_path in indexes:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            absent = sorted({str(path / shard) for shard in payload.get("weight_map", {}).values() if not (path / shard).is_file()})
            if absent:
                return {"status": "asset_missing", "path": str(path), "reason": f"missing checkpoint shard(s): {absent[:5]}"}
    elif not list(path.glob("*.safetensors")) and not list(path.glob("pytorch_model*.bin")):
        return {"status": "asset_missing", "path": str(path), "reason": "no model weights found"}
    tokenizer_files = (
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json", "vocab.json",
        "merges.txt", "special_tokens_map.json",
    )
    if label == "target model" and not any((path / name).is_file() for name in tokenizer_files):
        return {"status": "asset_missing", "path": str(path), "reason": "target tokenizer files are absent"}
    if config_payload.get("model_type") != "qwen3":
        return {"status": "asset_missing", "path": str(path), "reason": f"expected model_type=qwen3, got {config_payload.get('model_type')}"}
    if label == "DFlash draft":
        dflash_config = config_payload.get("dflash_config") or {}
        block_size = config_payload.get("block_size", dflash_config.get("block_size"))
        if block_size is not None and int(block_size) != 16:
            return {"status": "asset_missing", "path": str(path), "reason": f"expected DFlash block_size=16, got {block_size}"}
    return {
        "status": "ok",
        "path": str(path.resolve()),
        "model_type": config_payload.get("model_type"),
        "weight_files": len(list(path.glob("*.safetensors"))) + len(list(path.glob("pytorch_model*.bin"))),
        "block_size": config_payload.get("block_size", (config_payload.get("dflash_config") or {}).get("block_size")),
        "config_sha256": __import__("hashlib").sha256(config.read_bytes()).hexdigest(),
    }


def run_preflight(
    *,
    target_model: str | None,
    draft_model: str | None,
    data_dir: str | None,
    data_file: str | None,
    attention_backend: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "dflash-context-interventions.preflight.v1",
        "status": "ok",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "attention_backend_requested": attention_backend,
        "checks": {},
    }
    if platform.python_version_tuple()[:2] != ("3", "12"):
        result["checks"]["python"] = {"status": "package_missing", "reason": "server runtime must be Python 3.12"}
    else:
        result["checks"]["python"] = {"status": "ok"}
    package_versions: dict[str, str | None] = {}
    for distribution in ("torch", "transformers", "safetensors", "numpy", "accelerate", "huggingface-hub"):
        try:
            package_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            package_versions[distribution] = None
    result["packages"] = package_versions
    missing = [name for name, version in package_versions.items() if version is None]
    result["checks"]["packages"] = {
        "status": "package_missing" if missing else "ok",
        "missing": missing,
    }

    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpu = None
        if cuda_available:
            properties = torch.cuda.get_device_properties(0)
            gpu = {
                "name": properties.name,
                "uuid": str(getattr(properties, "uuid", "unknown")),
                "capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_gib": round(properties.total_memory / (1024**3), 3),
            }
        result["torch"] = {"version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": cuda_available, "gpu": gpu}
        if not cuda_available:
            result["checks"]["gpu"] = {"status": "package_missing", "reason": "CUDA device is not visible"}
        elif tuple(gpu["capability"])[0] < 10 or "B200" not in str(gpu["name"]).upper():
            result["checks"]["gpu"] = {
                "status": "kernel_incompatible",
                "reason": f"expected NVIDIA B200 (SM100+), got {gpu['name']} capability={gpu['capability']}",
            }
        else:
            result["checks"]["gpu"] = {"status": "ok"}
    except Exception as exc:
        result["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
        result["checks"]["gpu"] = {"status": "package_missing", "reason": result["torch"]["error"]}

    if attention_backend == "flash_attention_4":
        try:
            from common.vanilla_inference import _install_flash_attention_4_cutlass_compat, _probe_flash_attention_4

            _install_flash_attention_4_cutlass_compat()
            ok, reason = _probe_flash_attention_4()
            result["checks"]["flash_attention_4"] = {
                "status": "ok" if ok else "kernel_incompatible",
                "reason": reason,
            }
        except Exception as exc:
            result["checks"]["flash_attention_4"] = {
                "status": "kernel_incompatible",
                "reason": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
            }
    elif attention_backend == "sdpa":
        result["checks"]["attention_backend"] = {"status": "ok", "scope": "diagnostic_only"}
    else:
        result["checks"]["attention_backend"] = {"status": "kernel_incompatible", "reason": f"unsupported backend: {attention_backend}"}

    result["assets"] = {
        "target": _model_asset_check(target_model, "target model"),
        "draft": _model_asset_check(draft_model, "DFlash draft"),
    }
    for name, asset in result["assets"].items():
        result["checks"][f"{name}_asset"] = {"status": asset["status"], "reason": asset.get("reason")}

    target_asset = result["assets"]["target"]
    draft_asset = result["assets"]["draft"]
    mapping_status = "ok"
    mapping_reason = None
    if target_asset.get("status") == "ok" and draft_asset.get("status") == "ok":
        try:
            target_config = json.loads((Path(str(target_asset["path"])) / "config.json").read_text(encoding="utf-8"))
            draft_config = json.loads((Path(str(draft_asset["path"])) / "config.json").read_text(encoding="utf-8"))
            target_depth = int(target_config.get("num_hidden_layers", -1))
            dflash = draft_config.get("dflash_config") or {}
            target_layer_ids = tuple(int(value) for value in dflash.get("target_layer_ids", ()))
            expected_layer_ids = (1, 9, 17, 25, 33)
            if target_depth != 36:
                mapping_status, mapping_reason = "asset_missing", f"campaign expects a 36-layer Qwen3-4B target, got {target_depth}"
            elif int(draft_config.get("num_target_layers", -1)) != target_depth:
                mapping_status, mapping_reason = "asset_missing", "DFlash num_target_layers does not match target depth"
            elif target_layer_ids != expected_layer_ids:
                mapping_status, mapping_reason = "asset_missing", f"expected DFlash target_layer_ids={expected_layer_ids}, got {target_layer_ids}"
        except Exception as exc:
            mapping_status, mapping_reason = "asset_missing", f"could not validate target/draft mapping: {type(exc).__name__}: {exc}"
    else:
        mapping_status = "asset_missing"
        mapping_reason = "target/draft mapping cannot be checked until both model snapshots pass asset checks"
    result["checks"]["target_draft_mapping"] = {"status": mapping_status, "reason": mapping_reason}

    try:
        from TrainingFree.context_adaptive.benchmark import load_corpus

        if bool(data_dir) == bool(data_file):
            raise ValueError("set exactly one of DATA_INPUT (directory) or --data-file")
        samples = load_corpus(data_dir=Path(data_dir) if data_dir else None, data_file=Path(data_file) if data_file else None)
        datasets = sorted({str(row.get("dataset", "unknown")) for row in samples})
        expected = {"gov_report", "multi_news", "qmsum"}
        missing_datasets = sorted(expected - set(datasets))
        result["data"] = {"record_count": len(samples), "datasets": datasets, "missing_required_datasets": missing_datasets}
        result["checks"]["data"] = {"status": "asset_missing" if missing_datasets else "ok"}
    except Exception as exc:
        result["data"] = {"error": f"{type(exc).__name__}: {exc}"}
        result["checks"]["data"] = {"status": "asset_missing", "reason": result["data"]["error"]}

    try:
        result["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip().splitlines()
    except Exception as exc:
        result["nvidia_smi"] = {"unavailable": f"{type(exc).__name__}: {exc}"}

    failure_statuses = {"package_missing", "asset_missing", "kernel_incompatible", "parity_fail"}
    result["status"] = "ready" if not any(
        check.get("status") in failure_statuses for check in result["checks"].values()
    ) else "blocked"
    return result
