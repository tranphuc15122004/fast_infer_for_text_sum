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
import argparse
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
OPTIONAL_MODULES = {"flash_attn"}

# The server profile intentionally remains broad.  Modal's canonical
# LongBench image does not contain unrelated training/serving stacks such as
# vLLM or SGLang; checking those modules would reject a valid benchmark image
# before the selected adapters run.
PROFILE_MODULES = {
    "server": MODULES,
    "modal-longbench": (
        "torch",
        "transformers",
        "numpy",
        "yaml",
        "safetensors",
        "tqdm",
        "accelerate",
        "datasets",
        "rouge_score",
        "sentencepiece",
        "tokenizers",
        "flashinfer",
        "flash_attn",
        "dflash",
        "llmlingua",
        "pipeline.fafo.decoding",
    ),
}
PROFILE_OPTIONAL_MODULES = {
    "server": OPTIONAL_MODULES,
    "modal-longbench": set(),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILE_MODULES),
        default="server",
        help="dependency profile to check (default: server)",
    )
    args = parser.parse_args(argv)
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
    sys.path.insert(0, str(ROOT / "externals" / "FAFO"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    failures: list[str] = []
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print(
        "mode: offline import-only "
        f"(profile={args.profile}; no model loading)"
    )
    if sys.version_info[:2] != (3, 12):
        failures.append("Python 3.12 is required")

    optional_modules = PROFILE_OPTIONAL_MODULES[args.profile]
    for module_name in PROFILE_MODULES[args.profile]:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # binary imports can fail with varied errors
            level = "WARN OPTIONAL" if module_name in optional_modules else "FAIL"
            print(f"{level} {module_name}: {type(exc).__name__}: {exc}")
            if module_name not in optional_modules:
                failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
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
