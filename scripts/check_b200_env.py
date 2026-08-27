#!/usr/bin/env python3
"""Offline B200/server preflight for the shared benchmark runtime.

This command never downloads models or packages. It reports the selected
interpreter, CUDA device, baseline-specific imports, local assets supplied via
environment variables, and writable compiler caches. A host without CUDA is a
valid simulation input but returns BLOCKED rather than pretending to be B200.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BASELINES = (
    "eagle3",
    "dflash",
    "llmlingua",
    "fastkv",
    "rocketkv",
    "gemfilter",
    "specprefill",
    "minference",
    "magicdec",
    "longspec",
    "specextend",
    "higoe",
    "semantic_selection",
    "flexprefill",
)

GPU_ONLY = {
    "eagle3",
    "dflash",
    "specprefill",
    "minference",
    "magicdec",
    "longspec",
    "specextend",
    "flexprefill",
}

BASELINE_IMPORTS = {
    "eagle3": ("torch", "transformers", "eagle.model.ea_model"),
    "dflash": ("torch", "transformers", "dflash", "flash_attn"),
    "llmlingua": ("torch", "transformers", "llmlingua"),
    "fastkv": ("torch", "transformers"),
    "rocketkv": ("torch", "triton"),
    "gemfilter": ("torch", "transformers"),
    "specprefill": ("torch", "vllm"),
    "minference": ("torch", "transformers", "kivi_gemv"),
    "magicdec": ("torch", "flashinfer"),
    "longspec": ("torch", "triton", "liger_kernel"),
    "specextend": ("torch", "transformers", "termcolor", "eagle"),
    "higoe": ("torch", "sentence_transformers", "faiss", "dgl"),
    "semantic_selection": ("torch", "transformers", "sentence_transformers"),
    "flexprefill": ("torch", "triton"),
}

ASSET_ENV = {
    "eagle3": ("BASE_MODEL", "EAGLE_MODEL", "DATA_FILE"),
    "dflash": ("TARGET_MODEL", "DRAFT_MODEL", "DATA_FILE"),
    "llmlingua": ("COMPRESSOR_MODEL", "TARGET_MODEL", "DOC_FILE"),
    "fastkv": ("MODEL", "DATA_FILE"),
    "gemfilter": ("MODEL", "DATA_FILE"),
    "specprefill": ("TARGET_MODEL", "SPEC_MODEL", "DATA_FILE"),
    "minference": ("MODEL", "DATA_FILE"),
    "magicdec": ("MODEL_PTH",),
    "longspec": ("TARGET_MODEL", "DRAFT_MODEL", "DATA_FILE"),
    "specextend": ("BASE_MODEL", "DRAFT_MODEL", "INPUT_FILE"),
    "semantic_selection": ("MODEL", "EMBEDDING_MODEL", "INPUT_FILE"),
    "flexprefill": ("MODEL", "DATA_FILE"),
}

PROFILE_ASSET_ENV = {
    "eagle3": {"BASE_MODEL": "B200_TARGET_MODEL", "EAGLE_MODEL": "B200_EAGLE_MODEL"},
    "dflash": {
        "TARGET_MODEL": "B200_TARGET_MODEL",
        "DRAFT_MODEL": "B200_DFLASH_MODEL",
        "DATA_FILE": "B200_DATA_FILE",
    },
    "llmlingua": {
        "COMPRESSOR_MODEL": "B200_COMPRESSOR_MODEL",
        "TARGET_MODEL": "B200_TARGET_MODEL",
        "DOC_FILE": "B200_DATA_FILE",
    },
    "fastkv": {"MODEL": "B200_TARGET_MODEL", "DATA_FILE": "B200_DATA_FILE"},
    "gemfilter": {"MODEL": "B200_TARGET_MODEL", "DATA_FILE": "B200_DATA_FILE"},
    "specprefill": {
        "TARGET_MODEL": "B200_TARGET_MODEL",
        "SPEC_MODEL": "B200_SPEC_MODEL",
        "DATA_FILE": "B200_DATA_FILE",
    },
    "minference": {"MODEL": "B200_TARGET_MODEL", "DATA_FILE": "B200_DATA_FILE"},
    "magicdec": {"MODEL_PTH": "B200_MAGICDEC_MODEL_PTH"},
    "longspec": {
        "TARGET_MODEL": "B200_VICUNA_MODEL",
        "DRAFT_MODEL": "B200_LONGSPEC_DRAFT_MODEL",
        "DATA_FILE": "B200_DATA_FILE",
    },
    "specextend": {
        "BASE_MODEL": "B200_TARGET_MODEL",
        "DRAFT_MODEL": "B200_EAGLE_MODEL",
        "INPUT_FILE": "B200_DATA_FILE",
    },
    "semantic_selection": {
        "MODEL": "B200_TARGET_MODEL",
        "EMBEDDING_MODEL": "B200_EMBEDDING_MODEL",
        "INPUT_FILE": "B200_DATA_FILE",
    },
    "flexprefill": {"MODEL": "B200_TARGET_MODEL", "DATA_FILE": "B200_DATA_FILE"},
}

CACHE_ENV = (
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "TRITON_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "TORCH_EXTENSIONS_DIR",
)

WRITABLE_CACHE_ENV = {
    "TRITON_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "TORCH_EXTENSIONS_DIR",
}


def _add_external_paths() -> None:
    for relative in (
        "externals/EAGLE",
        "externals/dflash",
        "externals/LLMLingua",
        "externals/FastKV",
        "externals/GemFilter",
        "externals/MInference",
        "externals/SpecExtend",
        "externals/LongSpec",
        "externals/HiGOE",
        "externals/FlexPrefill",
        "externals/MagicDec",
        "externals/RocketKV",
    ):
        path = ROOT / relative
        if path.is_dir():
            sys.path.insert(0, str(path))


def _parse_baselines(value: str) -> list[str]:
    names = value.replace(",", " ").split()
    unknown = sorted(set(names) - set(BASELINES))
    if unknown:
        raise SystemExit(f"Unknown baseline(s): {', '.join(unknown)}")
    return list(dict.fromkeys(names))


def _check_torch() -> tuple[dict, object | None]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        return {
            "available": False,
            "reason": "torch_unavailable",
            "error": f"{type(exc).__name__}: {exc}",
        }, None

    cuda = {"available": False, "reason": "hardware_unavailable", "devices": []}
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        cuda["reason"] = "cuda_probe_error"
        cuda["error"] = f"{type(exc).__name__}: {exc}"
        return cuda, torch
    if not available:
        return cuda, torch

    try:
        count = int(torch.cuda.device_count())
        devices = []
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": str(props.name),
                    "major": int(props.major),
                    "minor": int(props.minor),
                    "total_memory_gb": round(props.total_memory / 1024**3, 2),
                }
            )
        cuda["available"] = True
        cuda["reason"] = "ok"
        cuda["devices"] = devices
    except Exception as exc:
        cuda["reason"] = "cuda_probe_error"
        cuda["error"] = f"{type(exc).__name__}: {exc}"
    return cuda, torch


def _probe_cuda_tensor(cuda: dict, torch: object | None) -> None:
    if not cuda.get("available") or torch is None:
        return
    try:
        tensor = torch.zeros(1, device="cuda")
        result = tensor + 1
        torch.cuda.synchronize()
        cuda["tensor_probe"] = {"ok": bool(result.item() == 1)}
        if not cuda["tensor_probe"]["ok"]:
            cuda["reason"] = "cuda_tensor_probe_failed"
    except Exception as exc:
        cuda["tensor_probe"] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        cuda["reason"] = "cuda_tensor_probe_failed"


def _check_imports(names: tuple[str, ...]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            result[name] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            result[name] = {
                "ok": True,
                "version": str(getattr(module, "__version__", "unknown")),
            }
    return result


def _check_caches() -> dict[str, dict]:
    result = {}
    for name in CACHE_ENV:
        value = os.environ.get(name)
        if not value:
            result[name] = {
                "ok": False,
                "required": name in WRITABLE_CACHE_ENV,
                "reason": "not_set",
            }
            continue
        path = Path(value).expanduser()
        required = name in WRITABLE_CACHE_ENV
        if not required:
            result[name] = {
                "ok": path.is_dir() and os.access(path, os.R_OK),
                "required": False,
                "path": str(path),
                "access": "read",
            }
            if not result[name]["ok"]:
                result[name]["reason"] = "not_readable"
            continue
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".fast_infer_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as exc:
            result[name] = {
                "ok": False,
                "required": True,
                "path": str(path),
                "reason": "not_writable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            result[name] = {"ok": True, "required": True, "path": str(path), "access": "write"}
    return result


def _resolve_asset(value: str, variable: str) -> dict:
    path = Path(value).expanduser()
    is_model = variable.endswith("MODEL") or variable in {
        "BASE_MODEL", "DRAFT_MODEL", "TARGET_MODEL", "SPEC_MODEL",
        "EAGLE_MODEL", "COMPRESSOR_MODEL", "EMBEDDING_MODEL",
    }
    if path.is_absolute() or value.startswith((".", "data/", "outputs/")):
        if not path.is_absolute():
            path = ROOT / path
        return {"value": value, "exists": path.exists(), "path": str(path)}
    if is_model and "/" in value:
        try:
            from common.paths import snapshot_dir

            cached = snapshot_dir(value)
        except Exception:
            cached = None
        return {
            "value": value,
            "exists": cached is not None,
            "path": str(cached) if cached is not None else None,
            "offline_lookup": "huggingface_cache",
        }
    return {"value": value, "exists": path.exists(), "path": str(path)}


def _asset_usable(info: dict, variable: str) -> tuple[bool, str | None]:
    if info.get("exists") is not True:
        return bool(info.get("exists") is None), None
    path = Path(info["path"])
    if variable == "MODEL_PTH":
        return path.is_file(), "missing_checkpoint_file" if not path.is_file() else None
    if variable.endswith("MODEL") or variable in {
        "BASE_MODEL", "DRAFT_MODEL", "TARGET_MODEL", "SPEC_MODEL",
        "EAGLE_MODEL", "COMPRESSOR_MODEL", "EMBEDDING_MODEL",
    }:
        has_config = (path / "config.json").is_file()
        if not has_config:
            return False, "missing_model_config"
        if variable in {"EAGLE_MODEL", "DRAFT_MODEL"}:
            has_weights = any(
                (path / name).is_file()
                for name in ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")
            )
            if not has_weights:
                return False, "missing_model_weights"
    if variable.endswith("FILE") or variable in {"DOC_FILE", "DATA_FILE", "INPUT_FILE"}:
        if not path.is_file():
            return False, "missing_data_file"
    return True, None


def _check_assets(baseline: str) -> dict[str, dict]:
    result = {}
    for variable in ASSET_ENV.get(baseline, ()):
        profile_variable = PROFILE_ASSET_ENV.get(baseline, {}).get(variable)
        value = os.environ.get(variable)
        if not value and profile_variable:
            value = os.environ.get(profile_variable)
        if not value:
            result[variable] = {
                "ok": False,
                "reason": "not_set",
                "profile_variable": profile_variable,
            }
            continue
        info = _resolve_asset(value, variable)
        usable, unusable_reason = _asset_usable(info, variable)
        if info.get("exists") is False or not usable:
            info["ok"] = False
            info["reason"] = unusable_reason or "missing_local_asset"
        elif info.get("exists") is True:
            info["ok"] = True
        else:
            info["ok"] = True
        result[variable] = info
    return result


def build_report(baselines: list[str], target_gpu: str) -> dict:
    _add_external_paths()
    cuda, torch = _check_torch()
    _probe_cuda_tensor(cuda, torch)
    interpreter_ok = sys.version_info[:2] == (3, 12)
    interpreter = {
        "path": sys.executable,
        "version": ".".join(str(x) for x in sys.version_info[:3]),
        "ok": interpreter_ok,
        "required": "3.12.*",
    }

    device_names = [d["name"] for d in cuda.get("devices", [])]
    target_match = any(target_gpu.upper() in name.upper() for name in device_names)
    if cuda.get("reason") == "cuda_tensor_probe_failed":
        pass
    elif cuda.get("available") and not target_match:
        cuda["reason"] = "wrong_gpu"
    elif cuda.get("available") and target_match:
        cuda["target_match"] = True
    else:
        cuda["target_match"] = False

    baselines_report = {}
    for baseline in baselines:
        imports = _check_imports(BASELINE_IMPORTS[baseline])
        assets = _check_assets(baseline)
        import_failures = [name for name, info in imports.items() if not info["ok"]]
        asset_failures = [name for name, info in assets.items() if not info["ok"]]
        if baseline in GPU_ONLY and cuda.get("reason") != "ok":
            status, reason = "BLOCKED", cuda.get("reason", "hardware_unavailable")
        elif import_failures:
            status, reason = "BLOCKED", "missing_dependency"
        elif asset_failures:
            status, reason = "BLOCKED", "missing_asset"
        else:
            status, reason = "PASS", "ready"
        baselines_report[baseline] = {
            "status": status,
            "reason": reason,
            "gpu_required": baseline in GPU_ONLY,
            "imports": imports,
            "assets": assets,
        }

    cache_report = _check_caches()
    required_failures = []
    if not interpreter_ok:
        required_failures.append("python_version")
    if not cuda.get("available"):
        required_failures.append("cuda")
    elif cuda.get("reason") != "ok":
        required_failures.append(f"cuda:{cuda.get('reason')}")
    if cuda.get("available") and not target_match:
        required_failures.append("target_gpu")
    if torch is None:
        required_failures.append("torch")
    if any(info["required"] and not info["ok"] for info in cache_report.values()):
        required_failures.append("cache")
    if any(info["status"] != "PASS" for info in baselines_report.values()):
        required_failures.append("baseline_requirements")

    status = "PASS" if not required_failures else "BLOCKED"
    return {
        "status": status,
        "target_gpu": target_gpu,
        "interpreter": interpreter,
        "cuda": cuda,
        "caches": cache_report,
        "baselines": baselines_report,
        "errors": required_failures,
    }


def _write_json(report: dict, destination: str | None) -> None:
    if not destination:
        return
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if destination == "-":
        print(text, end="")
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baselines", default=" ".join(BASELINES))
    parser.add_argument("--target-gpu", default=os.environ.get("B200_TARGET_GPU", "B200"))
    parser.add_argument("--json", nargs="?", const="-", default=None, metavar="PATH")
    args = parser.parse_args()
    report = build_report(_parse_baselines(args.baselines), args.target_gpu)
    _write_json(report, args.json)

    print(f"B200 preflight: {report['status']}")
    print(f"  interpreter: {report['interpreter']['path']} ({report['interpreter']['version']})")
    print(f"  cuda: {report['cuda'].get('reason')}")
    for name, info in report["baselines"].items():
        print(f"  {name}: {info['status']} ({info['reason']})")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
