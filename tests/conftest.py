"""Test isolation for vendored packages with generic top-level names."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FAFO_ROOT = str(ROOT / "externals" / "FAFO")


@pytest.fixture(autouse=True)
def isolate_fafo_pipeline_namespace(request):
    """Make FAFO imports deterministic after other vendored ``pipeline`` trees.

    RocketKV and FAFO both expose a top-level ``pipeline`` package.  Pytest
    collects all test modules in one interpreter, so a prior import can leave
    RocketKV's package in ``sys.modules`` even when FAFO is first on
    ``sys.path``.  FAFO tests need the same child-process-like import boundary
    as the benchmark adapters.
    """

    if "fafo" not in request.node.name.lower():
        yield
        return

    old_path = list(sys.path)
    old_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "pipeline" or name.startswith("pipeline.")
    }
    sys.path[:] = [
        FAFO_ROOT,
        str(ROOT),
    ] + [
        entry for entry in sys.path
        if entry not in {FAFO_ROOT, str(ROOT)}
    ]
    for name in list(old_modules):
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "pipeline" or name.startswith("pipeline."):
                sys.modules.pop(name, None)
        sys.modules.update(old_modules)
        sys.path[:] = old_path
