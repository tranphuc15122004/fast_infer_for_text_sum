#!/usr/bin/env python3
"""Offline preflight for the single server runtime.

This command checks the interpreter and imports packages without loading a
model, resolving a Hugging Face repo, or calling any network API.  The import
set covers both the shared benchmark runners and the local MR-DFlash
train/cache/inference modules.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "torch",
    "transformers",
    # MR-DFlash train/cache/preprocess dependencies.
    "numpy",
    "yaml",
    "safetensors",
    "tqdm",
    "MR_DFlash",
    "MR_DFlash.capture",
    "MR_DFlash.checkpoint",
    "MR_DFlash.config",
    "MR_DFlash.data",
    "MR_DFlash.inference",
    "MR_DFlash.memory",
    "MR_DFlash.mr_model",
    "MR_DFlash.offline_features",
    "MR_DFlash.online_features",
    "MR_DFlash.run_train",
    "MR_DFlash.tokenized_data",
    "MR_DFlash.trainer",
    "MR_DFlash.training",
    # Shared benchmark/runtime dependencies.
    "vllm",
    "triton",
    "flashinfer",
    "flash_attn",
    "gguf",
    "sgl_kernel",
    "dflash",
    "llmlingua",
    "sentence_transformers",
)
DIST_NAMES = {
    "torch": "torch",
    "transformers": "transformers",
    "numpy": "numpy",
    "yaml": "PyYAML",
    "safetensors": "safetensors",
    "tqdm": "tqdm",
    "vllm": "vllm",
    "triton": "triton",
    "flashinfer": "flashinfer-python",
    "flash_attn": "flash-attn",
    "gguf": "gguf",
    "sgl_kernel": "sglang-kernel",
    "dflash": "dflash",
    "llmlingua": "llmlingua",
    "sentence_transformers": "sentence-transformers",
}


def _version(module_name: str, module: object) -> str:
    value = getattr(module, "__version__", None)
    if value:
        return str(value)
    distribution = DIST_NAMES.get(module_name)
    if distribution is None:
        distribution = DIST_NAMES.get(module_name.split(".", 1)[0])
    if distribution is None:
        return "local"
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> int:
    cache_root = Path(
        os.environ.get("FAST_INFER_CACHE_ROOT", "/tmp/fast_infer_cache")
    )
    os.environ.setdefault(
        "FLASHINFER_WORKSPACE_BASE", str(cache_root / "flashinfer")
    )
    os.environ.setdefault("TRITON_CACHE_DIR", str(cache_root / "triton"))
    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions")
    )
    for path in (
        os.environ["FLASHINFER_WORKSPACE_BASE"],
        os.environ["TRITON_CACHE_DIR"],
        os.environ["TORCH_EXTENSIONS_DIR"],
    ):
        Path(path).mkdir(parents=True, exist_ok=True)

    # Make local baseline packages discoverable without installing them.
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "externals" / "dflash"))
    sys.path.insert(0, str(ROOT / "externals" / "LLMLingua"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    failures: list[str] = []
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print("mode: offline import-only (no model loading; MR-DFlash included)")
    if sys.version_info[:2] != (3, 12):
        failures.append("Python 3.12 is required")

    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # binary imports can fail with varied errors
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
            print(f"FAIL {module_name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {module_name} {_version(module_name, module)}")
            if module_name == "torch":
                try:
                    print(f"CUDA available: {module.cuda.is_available()}")
                except Exception as exc:
                    failures.append(f"torch.cuda: {type(exc).__name__}: {exc}")
                    print(f"FAIL torch.cuda: {type(exc).__name__}: {exc}")

    if failures:
        print("\nShared environment preflight: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nShared environment preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
