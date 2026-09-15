"""Offline preflight for the SpecExtend Horizon-CMR run."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TARGET = Path("/home/tuantb/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5-16k/snapshots/c8df3ca4436a3bce5c4b5877e0117032081852b4")
DEFAULT_DRAFT = Path("/home/tuantb/.cache/huggingface/hub/models--double7--vicuna-68m/snapshots/f35c45e548302e8edd0a31db7490b42ea2ddd109")
SPECEXTEND = ROOT / "externals" / "SpecExtend" / "specextend"


def model_files(path: Path) -> dict[str, object]:
    files = {item.name: item.stat().st_size for item in path.iterdir()} if path.is_dir() else {}
    weight_bytes = sum(size for name, size in files.items() if name.endswith((".bin", ".safetensors")))
    return {
        "path": str(path),
        "exists": path.is_dir(),
        "file_count": len(files),
        "weight_bytes": weight_bytes,
        "has_config": (path / "config.json").is_file(),
        "has_tokenizer": any((path / name).is_file() for name in ("tokenizer.model", "tokenizer.json")),
    }


def torch_probe() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"imported": False, "error": f"{type(exc).__name__}: {exc}"}
    cuda = bool(torch.cuda.is_available())
    result: dict[str, object] = {
        "imported": True,
        "version": torch.__version__,
        "cuda_available": cuda,
        "cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()),
    }
    if cuda:
        result["devices"] = [
            {
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    return result


def dependency_probe() -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("transformers", "accelerate", "termcolor", "flash_attn", "triton"):
        try:
            module = __import__(name)
            result[name] = {
                "available": True,
                "version": getattr(module, "__version__", "unknown"),
                "role": "optional cosmetic fallback" if name == "termcolor" else "runtime/accelerator",
            }
        except Exception as exc:
            result[name] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
                "role": "optional cosmetic fallback" if name == "termcolor" else "runtime/accelerator",
            }
    return result


def data_probe(data_dir: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    for level in ("512", "1K", "2K", "4K", "8K", "16K"):
        path = data_dir / f"govreport_{level}.jsonl"
        result[level] = {
            "path": str(path),
            "exists": path.is_file(),
            "records": sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.is_file() else 0,
        }
    return result


def make_report(target: Path, draft: Path, data_dir: Path) -> dict[str, object]:
    torch_info = torch_probe()
    target_info = model_files(target)
    draft_info = model_files(draft)
    report: dict[str, object] = {
        "status": "PASS" if torch_info.get("cuda_available") and target_info["exists"] and draft_info["exists"] else "BLOCKED_RUNTIME",
        "python": {"version": sys.version, "executable": sys.executable},
        "platform": platform.platform(),
        "model_choice": {
            "family": "Vicuna classic SpecExtend summarization path",
            "target": target_info,
            "draft": draft_info,
            "reason": "official classic path; smallest compatible cached pair; Llama+EAGLE is not T4-safe and Qwen is not supported by this loader",
        },
        "torch": torch_info,
        "dependencies": dependency_probe(),
        "data": data_probe(data_dir),
        "required_commands": {
            "smoke": "python run_classic.py --input_file data/govreport/govreport_512.jsonl --model_name vicuna_7b --use_specextend --max_gen_len 64",
            "pilot_4k": "python run_classic.py --input_file data/govreport/govreport_4K.jsonl --model_name vicuna_7b --use_specextend --max_gen_len 256",
            "pilot_8k": "python run_classic.py --input_file data/govreport/govreport_8K.jsonl --model_name vicuna_7b --use_specextend --max_gen_len 256",
        },
        "limitations": [
            "Không gọi CPU/fallback là GPU benchmark.",
            "CUDA false trong runtime hiện tại làm 4K/8K inference chưa được xác nhận.",
            "Attention tracing chỉ có ý nghĩa khi SpecExtend target forward trả attention scores.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--data-dir", type=Path, default=SPECEXTEND / "data" / "govreport")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = make_report(args.target, args.draft, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 2)


if __name__ == "__main__":
    main()
