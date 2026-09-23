#!/usr/bin/env python3
"""Run the Qwen3-4B paired LongBench matrix in one short command.

The runner executes Vanilla HF first, extracts a strict per-sample sidecar,
then runs Vanilla FA, DFlash, Domino, and EAGLE3 sequentially.  Sequential
execution is intentional: it avoids GPU contention and makes every online
speedup compare against the same batch-1 request contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common.data_loader import load_records  # noqa: E402
from common.paired_reference import extract_reference_sidecar  # noqa: E402
from common.qwen3_paired import (  # noqa: E402
    DEFAULT_INPUT_CAP,
    QWEN3_PAIRED_BASELINES,
    QWEN3_PAIRED_DATASETS,
    build_run_config,
    input_distribution,
    profile_defaults,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--target-model", default=os.environ.get("MODEL_TARGET"))
    parser.add_argument(
        "--dflash-model",
        default=os.environ.get("MODEL_DFLASH") or os.environ.get("MODEL_DFLASH_DRAFT"),
    )
    parser.add_argument(
        "--domino-model",
        default=os.environ.get("MODEL_DOMINO") or os.environ.get("MODEL_DOMINO_DRAFT"),
    )
    parser.add_argument(
        "--eagle-model",
        default=os.environ.get("MODEL_EAGLE") or os.environ.get("MODEL_EAGLE_DRAFT"),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=ROOT / "data" / "longbench_100_14k"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs" / "Benchmark_results" / "qwen3_4b_paired_online",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--datasets", default=",".join(QWEN3_PAIRED_DATASETS))
    parser.add_argument("--baselines", default=",".join(QWEN3_PAIRED_BASELINES))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_INPUT_CAP)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--vanilla-fa-backend", default=os.environ.get("LONG_BENCH_FA_BACKEND", "flash_attention_2"))
    parser.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    parser.add_argument("--python", dest="python_executable", default=os.environ.get("FAST_INFER_PYTHON", sys.executable))
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _csv(value: str, allowed: tuple[str, ...], flag: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise SystemExit(f"{flag}: unsupported value(s): {', '.join(invalid)}")
    if not values:
        raise SystemExit(f"{flag} must not be empty")
    return values


def _run_id(value: str | None) -> str:
    if value:
        return value
    return datetime.now(timezone.utc).strftime("qwen3_4b_%Y%m%dT%H%M%SZ")


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def _common_args(args, *, data_file: Path, output: Path, max_samples: int, budget: int, run_config_hash: str) -> list[str]:
    return [
        "--data-file", str(data_file),
        "--max-samples", str(max_samples),
        "--max-input-tokens", str(args.max_input_tokens),
        "--max-new-tokens", str(budget),
        "--fixed-output-tokens", str(budget),
        "--temperature", str(args.temperature),
        "--seed", str(args.seed),
        "--run-config-hash", run_config_hash,
        "--run-id", args._run_id,
        "--output", str(output),
    ]


def _make_question_file(data_file: Path, output: Path, max_samples: int) -> int:
    rows = load_records(data_file, max_samples)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            raw = row.get("raw", {})
            handle.write(json.dumps({
                "question_id": str(row["id"]),
                "dataset": raw.get("dataset"),
                "task_type": raw.get("task_type"),
                "turns": [row["prompt"]],
                "answer": row.get("reference"),
                "reference": row.get("reference"),
            }, ensure_ascii=False) + "\n")
    return len(rows)


def build_commands(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profiles = profile_defaults(args.mode)
    budget = int(profiles["max_new_tokens"])
    max_samples = int(args.max_samples or profiles["max_samples"])
    datasets = _csv(args.datasets, QWEN3_PAIRED_DATASETS, "--datasets")
    baselines = _csv(args.baselines, QWEN3_PAIRED_BASELINES, "--baselines")
    if "vanilla_hf" not in baselines:
        raise SystemExit("--baselines must include vanilla_hf to create the reference sidecar")
    if not args.target_model:
        raise SystemExit("--target-model or MODEL_TARGET is required")
    for name, value in (("dflash", args.dflash_model), ("domino", args.domino_model), ("eagle3", args.eagle_model)):
        if name in baselines and not value:
            raise SystemExit(f"--{name.replace('_', '-')}-model or MODEL_{name.upper()} is required")

    run_id = _run_id(args.run_id)
    args._run_id = run_id
    run_root = Path(args.output_dir) / run_id
    commands: list[dict[str, Any]] = []
    for dataset in datasets:
        data_file = Path(args.data_dir) / f"{dataset}.jsonl"
        if not data_file.exists():
            raise SystemExit(f"missing dataset file: {data_file}")
        count = min(max_samples, sum(1 for line in data_file.open(encoding="utf-8") if line.strip()))
        cfg = build_run_config(
            dataset=dataset,
            target_model=args.target_model,
            input_cap=args.max_input_tokens,
            output_tokens=budget,
            seed=args.seed,
            dtype=args.dtype,
            attention_backend="paired",
            temperature=args.temperature,
        )
        cfg_hash = str(cfg["run_config_hash"])
        reference_output = run_root / "vanilla_hf" / f"{dataset}.jsonl"
        reference_sidecar = run_root / "references" / f"{dataset}.jsonl"
        common = _common_args(
            args, data_file=data_file, output=reference_output,
            max_samples=count, budget=budget, run_config_hash=cfg_hash,
        )
        common_without_output = common[:common.index("--output")]
        commands.append({
            "dataset": dataset, "baseline": "vanilla_hf",
            "output": str(reference_output),
            "command": [args.python_executable, str(SCRIPTS / "infer_vanilla_hf.py"),
                         "--model", args.target_model, "--device", "cuda:0",
                         "--dtype", args.dtype, "--attention-backend", "eager",
                         "--warmup-runs", str(args.warmup_runs), "--local-files-only", *common],
            "reference_sidecar": str(reference_sidecar),
        })
        if "vanilla_fa" in baselines:
            output = run_root / "vanilla_fa" / f"{dataset}.jsonl"
            commands.append({
                "dataset": dataset, "baseline": "vanilla_fa", "output": str(output),
                "command": [args.python_executable, str(SCRIPTS / "infer_vanilla_fa.py"),
                             "--model", args.target_model, "--device", "cuda:0",
                             "--dtype", args.dtype, "--attention-backend", args.vanilla_fa_backend,
                             "--warmup-runs", str(args.warmup_runs), "--local-files-only",
                             "--reference-file", str(reference_sidecar), *common[:common.index("--output")],
                             "--output", str(output)],
            })
        if "dflash" in baselines:
            output = run_root / "dflash" / f"{dataset}.jsonl"
            commands.append({
                "dataset": dataset, "baseline": "dflash", "output": str(output),
                "command": [args.python_executable, str(SCRIPTS / "infer_dflash.py"),
                             "--target-model", args.target_model, "--draft-model", args.dflash_model,
                             "--skip-reference", "--reference-file", str(reference_sidecar), *common_without_output,
                             "--output", str(output)],
            })
        if "domino" in baselines:
            output = run_root / "domino" / f"{dataset}.jsonl"
            commands.append({
                "dataset": dataset, "baseline": "domino", "output": str(output),
                "command": [args.python_executable, str(SCRIPTS / "infer_domino.py"),
                             "--target-model", args.target_model, "--draft-model", args.domino_model,
                             "--reference-file", str(reference_sidecar), "--attention-backend", "sdpa",
                             "--use-bias", *common_without_output, "--output", str(output)],
            })
        if "eagle3" in baselines:
            question_file = run_root / "inputs" / f"{dataset}.eagle.jsonl"
            # The file is generated by main() before execution; keep the
            # command declarative here so --dry-run shows the exact matrix.
            output = run_root / "eagle3" / f"{dataset}.jsonl"
            eagle_common = [
                "--max-new-tokens", str(budget), "--fixed-output-tokens", str(budget),
                "--max-input-tokens", str(args.max_input_tokens), "--temperature", str(args.temperature),
                "--dtype", args.dtype, "--seed", str(args.seed), "--question-begin", "0", "--question-end", str(count),
                "--dataset-name", dataset, "--run-config-hash", cfg_hash, "--run-id", run_id,
                "--reference-file", str(reference_sidecar), "--skip-naive", "--output", str(output),
            ]
            commands.append({
                "dataset": dataset, "baseline": "eagle3", "output": str(output),
                "question_file": str(question_file),
                "command": [args.python_executable, str(SCRIPTS / "eagle3_infer_qwen3.py"),
                             "--base-model", args.target_model, "--eagle-model", args.eagle_model,
                             "--question-file", str(question_file), *eagle_common],
            })
    manifest = {
        "run_id": run_id,
        "mode": args.mode,
        "target_model": args.target_model,
        "datasets": datasets,
        "baselines": baselines,
        "max_samples": max_samples,
        "max_input_tokens": args.max_input_tokens,
        "fixed_output_tokens": budget,
        "batch_size": 1,
        "reference_baseline": "vanilla_hf",
        "speedup_scope": "qwen3_4b_batch1_fixed_budget",
        "input_distribution": input_distribution(args.data_dir),
        "run_root": str(run_root),
    }
    return commands, manifest


def main() -> int:
    args = build_parser().parse_args()
    if not args.data_dir.is_absolute():
        args.data_dir = ROOT / args.data_dir
    if not args.output_dir.is_absolute():
        args.output_dir = ROOT / args.output_dir
    commands, manifest = build_commands(args)
    run_root = Path(manifest["run_root"])
    run_root.mkdir(parents=True, exist_ok=False)
    for item in commands:
        if item.get("question_file"):
            _make_question_file(
                Path(args.data_dir) / f"{item['dataset']}.jsonl",
                Path(item["question_file"]),
                int(manifest["max_samples"]),
            )
    manifest["commands"] = [_command_text(item["command"]) for item in commands]
    (run_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for item in commands:
        print(f"[{item['dataset']}][{item['baseline']}] {_command_text(item['command'])}")
    if args.dry_run:
        print(f"Dry-run manifest: {run_root / 'run_manifest.json'}")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    for item in commands:
        output = Path(item["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nRunning {item['dataset']} / {item['baseline']} ...", flush=True)
        subprocess.run(item["command"], cwd=ROOT, env=env, check=True)
        if item["baseline"] == "vanilla_hf":
            sidecar = Path(item["reference_sidecar"])
            count = extract_reference_sidecar(output, sidecar)
            expected = int(manifest["max_samples"])
            if count != min(expected, sum(1 for line in (Path(args.data_dir) / f"{item['dataset']}.jsonl").open(encoding="utf-8") if line.strip())):
                raise RuntimeError(f"reference sidecar sample count mismatch for {item['dataset']}: {count}")
            print(f"Reference sidecar: {sidecar} ({count} samples)", flush=True)
    print(f"\nCompleted paired matrix. Manifest: {run_root / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
