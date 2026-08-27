#!/usr/bin/env python3
"""Offline preflight for the single server runtime.

This command checks the interpreter and imports packages without loading a
model, resolving a Hugging Face repo, or calling any network API.
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
    "vllm",
    "triton",
    "flashinfer",
    "flash_attn",
    "dflash",
    "llmlingua",
    "sentence_transformers",
)
DIST_NAMES = {
    "torch": "torch",
    "transformers": "transformers",
    "vllm": "vllm",
    "triton": "triton",
    "flashinfer": "flashinfer-python",
    "flash_attn": "flash-attn",
    "dflash": "dflash",
    "llmlingua": "llmlingua",
    "sentence_transformers": "sentence-transformers",
}


def _version(module_name: str, module: object) -> str:
    value = getattr(module, "__version__", None)
    if value:
        return str(value)
    try:
        return importlib.metadata.version(DIST_NAMES[module_name])
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main() -> int:
    # Make local baseline packages discoverable without installing them.
    sys.path.insert(0, str(ROOT / "externals" / "dflash"))
    sys.path.insert(0, str(ROOT / "externals" / "LLMLingua"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    failures: list[str] = []
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print("mode: offline import-only (no model loading)")
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
