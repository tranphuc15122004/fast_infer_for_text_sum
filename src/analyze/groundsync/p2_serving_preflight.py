"""Preflight for the optional production-serving P2 benchmark.

The direct EAGLE benchmark is intentionally separate from this check.  A
serving result is only valid when the requested server package and canonical
server mount are present; this module records an explicit UNAVAILABLE result
otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def check_serving_environment() -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[3]
    canonical_server_repo = Path("/workspace/storage-shared/nlp/dungdx4/phuc_projects/fast_infer_text_sum")
    vllm_available = importlib.util.find_spec("vllm") is not None
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception as exc:  # pragma: no cover - depends on runtime install
        cuda_available = False
        torch_version = None
        cuda_version = None
        torch_error = f"{type(exc).__name__}: {exc}"
    else:
        torch_error = None
    server_present = canonical_server_repo.is_dir()
    checks = {
        "vllm_importable": vllm_available,
        "canonical_server_repo_mounted": server_present,
        "repository": str(repo),
        "canonical_server_repo": str(canonical_server_repo),
    }
    if vllm_available and server_present:
        status = "READY"
        decision = "READY_FOR_SERVING_BENCHMARK"
        reason = None
    else:
        status = "UNAVAILABLE"
        decision = "UNAVAILABLE"
        missing = []
        if not vllm_available:
            missing.append("vllm is not importable in the active system runtime")
        if not server_present:
            missing.append("canonical server repository is not mounted on this host")
        reason = "; ".join(missing)
    return {
        "schema_version": "groundsync.p2.serving_preflight.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "decision": decision,
        "reason": reason,
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch_version,
            "torch_cuda": cuda_version,
            "cuda_available": cuda_available,
            "torch_error": torch_error,
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = check_serving_environment()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
