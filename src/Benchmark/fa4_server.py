"""CLI for the native Transformers + FA4 LongBench benchmark."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any


SERVER_MODEL_ENV = {
    "vanilla_hf": "MODEL_TARGET",
    "eagle3": "MODEL_EAGLE_DRAFT",
    "dflash": "MODEL_DFLASH_DRAFT",
    "domino": "MODEL_DOMINO_DRAFT",
    "dspark": "MODEL_DSPARK_DRAFT",
    "dflash_phase_b": "MODEL_DFLASH_DRAFT",
}

SERVER_MODEL_ALIASES = {
    "vanilla_hf": ("MODEL_TARGET", "B200_TARGET_MODEL", "LONG_BENCH_MODEL"),
    "eagle3": ("MODEL_EAGLE_DRAFT", "B200_EAGLE_MODEL"),
    "dflash": ("MODEL_DFLASH_DRAFT", "B200_DFLASH_MODEL"),
    "domino": ("MODEL_DOMINO_DRAFT", "B200_DOMINO_MODEL"),
    "dspark": ("MODEL_DSPARK_DRAFT", "B200_DSPARK_MODEL"),
    "dflash_phase_b": ("MODEL_DFLASH_DRAFT", "B200_DFLASH_MODEL"),
}

SUPPORTED_METHODS = ("vanilla_hf", "eagle3", "dflash", "domino", "dspark", "dflash_phase_b")

LENGTH_BIN_DATASETS = (
    "bin1_00k_02k",
    "bin2_02k_04k",
    "bin3_04k_08k",
    "bin4_08k_12k",
    "bin5_12k_16k",
)


def auto_load_master_config(config_arg: str | None = None) -> Path | None:
    """Auto-load master config into os.environ if models or paths are missing."""
    root = Path(__file__).resolve().parents[2]
    candidate: Path | None = None
    if config_arg and Path(config_arg).expanduser().is_file():
        candidate = Path(config_arg).expanduser()
    elif os.environ.get("FAST_INFER_MASTER_CONFIG") and Path(os.environ["FAST_INFER_MASTER_CONFIG"]).expanduser().is_file():
        candidate = Path(os.environ["FAST_INFER_MASTER_CONFIG"]).expanduser()
    else:
        pointer = root / "config" / "master.path"
        if pointer.is_file():
            for line in pointer.read_text(encoding="utf-8").splitlines():
                target = line.split("#", 1)[0].strip()
                if target:
                    p = Path(target).expanduser()
                    if not p.is_absolute():
                        p = root / p
                    if p.is_file():
                        candidate = p
                    break
        if candidate is None:
            default_canonical = Path("/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/fast_infer_master.env")
            if default_canonical.is_file():
                candidate = default_canonical

    if candidate and candidate.is_file():
        import subprocess
        sub_cmd = f'set -a && source "{candidate}" && python3 -c "import os, json; print(json.dumps(dict(os.environ)))"'
        proc = subprocess.run(["bash", "-c", sub_cmd], capture_output=True, text=True)
        if proc.returncode == 0:
            try:
                loaded = json.loads(proc.stdout)
                for k, v in loaded.items():
                    if k not in os.environ or not str(os.environ[k]).strip():
                        os.environ[k] = str(v)
                return candidate
            except Exception:
                pass
    return None


def resolve_server_models(
    environ: Mapping[str, str], methods: str = "all"
) -> dict[str, str]:
    requested = [part for part in methods.replace(",", " ").split() if part]
    if requested == ["all"]:
        selected = SUPPORTED_METHODS
    else:
        unknown = sorted(set(requested) - set(SUPPORTED_METHODS))
        if unknown:
            raise ValueError(f"Unknown methods: {unknown}")
        if len(requested) != len(set(requested)):
            raise ValueError("Duplicate method in --methods")
        if "vanilla_hf" not in requested or "dflash" not in requested:
            raise ValueError("FA4 comparisons require both vanilla_hf and dflash")
        selected = tuple(method for method in SUPPORTED_METHODS if method in requested)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for method in selected:
        value = None
        for var in SERVER_MODEL_ALIASES.get(method, (SERVER_MODEL_ENV[method],)):
            candidate = str(environ.get(var, "")).strip()
            if candidate:
                value = candidate
                if isinstance(environ, dict):
                    environ.setdefault(SERVER_MODEL_ENV[method], value)
                elif hasattr(os, "environ"):
                    os.environ.setdefault(SERVER_MODEL_ENV[method], value)
                break
        if value:
            resolved[method] = value
        else:
            missing.append(SERVER_MODEL_ENV[method])

    if missing:
        dedup_missing = sorted(set(missing))
        raise ValueError("Missing model paths from master config: " + ", ".join(dedup_missing))
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chạy benchmark LongBench batch-1 native Transformers + FA4."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "representative", "full"),
        default=os.environ.get("LONG_BENCH_MODE", "smoke"),
    )
    parser.add_argument("--smoke", dest="mode", action="store_const", const="smoke")
    parser.add_argument("--representative", dest="mode", action="store_const", const="representative")
    parser.add_argument("--full", dest="mode", action="store_const", const="full")
    parser.add_argument("--datasets", default=os.environ.get("LONG_BENCH_DATASETS", "all"))
    parser.add_argument("--samples-per-dataset", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--methods", default=os.environ.get("FA4_METHODS", "all"))
    parser.add_argument("--warmup-tokens", type=int, default=int(os.environ.get("FA4_WARMUP_TOKENS", "8")))
    parser.add_argument("--repetitions", type=int, default=int(os.environ.get("FA4_REPETITIONS", "1")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("LONG_BENCH_SEED", "42")))
    parser.add_argument("--sample-retries", type=int, default=int(os.environ.get("FA4_SAMPLE_RETRIES", "1")))
    parser.add_argument("--checkpoint-interval", type=int, default=int(os.environ.get("FA4_CHECKPOINT_INTERVAL", "20")))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--direct-target-audit", action="store_true")
    parser.add_argument("--verifier-audit", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--debug-cuda-launch-blocking", action="store_true")
    parser.add_argument("--config", default=os.environ.get("FAST_INFER_MASTER_CONFIG", ""), help="Đường dẫn file master config shell-env (mặc định lấy theo config/master.path)")
    parser.add_argument("--data-dir", default=os.environ.get("FA4_DATA_DIR", os.environ.get("LONG_BENCH_DATA_DIR", "data/longbench_100_14k")))
    parser.add_argument("--output-dir", default=os.environ.get("FA4_OUTPUT_ROOT", "outputs/fa4_native_benchmark"))
    return parser


def runner_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    mode = args.mode
    default_samples = {
        "smoke": int(os.environ.get("LONG_BENCH_SMOKE_SAMPLES", "1")),
        "representative": int(os.environ.get("LONG_BENCH_REPRESENTATIVE_SAMPLES", "20")),
        "full": 0,
    }
    max_new_tokens = args.max_new_tokens
    if max_new_tokens is None:
        if mode == "smoke":
            max_new_tokens = int(os.environ.get("LONG_BENCH_SMOKE_MAX_NEW_TOKENS", "8"))
        else:
            max_new_tokens = int(os.environ.get("RUN_MAX_NEW_TOKENS", os.environ.get("LONG_BENCH_MAX_NEW_TOKENS", "512")))
    max_input_tokens = args.max_input_tokens
    if max_input_tokens is None:
        max_input_tokens = int(os.environ.get("LONG_BENCH_MAX_INPUT_TOKENS", "0"))
    return {
        "mode": mode,
        "datasets": args.datasets,
        "samples_per_dataset": args.samples_per_dataset if args.samples_per_dataset is not None else default_samples[mode],
        "max_new_tokens": max_new_tokens,
        "max_input_tokens": max_input_tokens,
        "methods": args.methods,
        "warmup_tokens": args.warmup_tokens,
        "repetitions": args.repetitions,
        "seed": args.seed,
        "sample_retries": args.sample_retries,
        "checkpoint_interval": args.checkpoint_interval,
        "run_id": args.run_id,
        "resume": args.resume,
        "direct_target_audit": args.direct_target_audit,
        "verifier_audit": args.verifier_audit,
        "preflight_only": args.preflight_only,
        "debug_cuda_launch_blocking": args.debug_cuda_launch_blocking,
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    auto_load_master_config(args.config)
    args.methods = ",".join(args.methods.replace(",", " ").split())
    try:
        models = resolve_server_models(os.environ, args.methods)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    root = Path(__file__).resolve().parents[2]
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_absolute():
        data_dir = root / data_dir

    requested_datasets = [d.strip() for d in args.datasets.replace(",", " ").split() if d.strip()]
    if any(d in LENGTH_BIN_DATASETS for d in requested_datasets):
        has_bins = False
        manifest_file = data_dir / "manifest.json"
        if manifest_file.is_file():
            try:
                has_bins = "bins" in json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        if not has_bins:
            for candidate in (
                Path("/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/length_bins"),
                root / "data" / "length_bins",
                data_dir.parent / "length_bins",
                data_dir / "length_bins",
            ):
                if candidate.is_dir() and (candidate / "manifest.json").is_file():
                    data_dir = candidate
                    args.data_dir = str(candidate)
                    break

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    os.environ["FA4_EXECUTION_BACKEND"] = "server"
    os.environ["FA4_DATA_DIR"] = str(data_dir.resolve())
    os.environ["FA4_OUTPUT_ROOT"] = str(output_dir.resolve())
    for method, model_path in models.items():
        os.environ[SERVER_MODEL_ENV[method]] = model_path
    if os.environ.get("FI_OFFLINE", "1") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(root / "src"))
    from Benchmark.native_fa4_benchmark import run_flashattn_benchmark

    result = run_flashattn_benchmark(**runner_kwargs(args))
    if args.preflight_only:
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return 0
    run_dir = Path(result["remote_run_dir"])
    report_path = run_dir / "report_vi.md"
    if report_path.is_file():
        print(report_path.read_text(encoding="utf-8"))
    print(f"Kết quả: {run_dir}")
    if result["summary"].get("run_passed") is not True:
        print(f"Benchmark gate chưa đạt; xem {report_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
