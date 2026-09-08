"""Run the E23/E24 objective matrix with one shared training config.

The matrix changes only the training objective knobs.  Model architecture,
data, seed, batch, schedule, and evaluation paths come from the base config.
Run this on the B200 server after the pilot feature/tokenized stores exist.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


OBJECTIVE_VARIANTS: dict[str, dict[str, str]] = {
    "e23_fixed_decay": {"loss_type": "dflash"},
    "e23_dpace": {"loss_type": "dpace", "dpace_alpha": "0.5"},
    "e23_spec_auf": {"loss_type": "spec-auf"},
    "e24_dpace_hn_all": {
        "loss_type": "dpace-hard-negative-all",
        "hard_negative_k": "32",
        "hard_negative_lambda": "0.25",
    },
    "e24_dpace_hn_shallow": {
        "loss_type": "dpace-hard-negative-shallow",
        "hard_negative_k": "32",
        "hard_negative_lambda": "0.25",
    },
}


def build_command(
    *,
    python: str,
    config: str,
    output_dir: str,
    variant: str,
    max_steps: int | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
) -> list[str]:
    if variant not in OBJECTIVE_VARIANTS:
        raise ValueError(f"unknown E23/E24 variant: {variant}")
    command = [
        python,
        "-m",
        "MR_DFlash.run_train",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
    ]
    for name, value in OBJECTIVE_VARIANTS[variant].items():
        command.extend([f"--{name.replace('_', '-')}", value])
    if max_steps is not None:
        command.extend(["--max-steps", str(int(max_steps))])
    if num_samples is not None:
        command.extend(["--num-samples", str(int(num_samples))])
    if seed is not None:
        command.extend(["--seed", str(int(seed))])
    return command


def build_eval_command(
    *,
    python: str,
    config: str,
    checkpoint: str,
    input_path: str,
    output_path: str,
    device: str,
    max_new_tokens: int,
    max_samples: int | None = None,
    local_files_only: bool = False,
) -> list[str]:
    command = [
        python,
        "scripts/mr_dflash/evaluate_pilot.py",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--device",
        str(device),
        "--max-new-tokens",
        str(int(max_new_tokens)),
        "--exactness-check",
    ]
    if max_samples is not None:
        command.extend(["--max-samples", str(int(max_samples))])
    if local_files_only:
        command.append("--local-files-only")
    return command


def _run_one(command: Sequence[str], *, cwd: Path, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    variants = list(args.variants or OBJECTIVE_VARIANTS)
    env = os.environ.copy()
    source_root = str(repo_root / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    results: dict[str, Any] = {}
    for variant in variants:
        if variant not in OBJECTIVE_VARIANTS:
            raise ValueError(f"unknown E23/E24 variant: {variant}")
        variant_dir = output_root / variant
        command = build_command(
            python=args.python,
            config=args.config,
            output_dir=str(variant_dir),
            variant=variant,
            max_steps=args.max_steps,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        log_path = variant_dir / "run.log"
        print(f"\n[E23/E24] {variant}: {' '.join(command)}")
        if args.dry_run:
            results[variant] = {"status": "dry_run", "command": command}
            continue
        returncode = _run_one(command, cwd=repo_root, log_path=log_path, env=env)
        results[variant] = {
            "status": "pass" if returncode == 0 else "fail",
            "returncode": returncode,
            "command": command,
            "run_log": str(log_path),
            "metrics": str(variant_dir / "metrics.jsonl"),
            "train_eval_metrics": str(variant_dir / "eval_metrics.json"),
        }
        if returncode == 0 and args.eval_input:
            eval_output = variant_dir / "generation_eval.jsonl"
            eval_log = variant_dir / "generation_eval.log"
            eval_command = build_eval_command(
                python=args.python,
                config=args.config,
                checkpoint=str(variant_dir / "checkpoint_final.pt"),
                input_path=args.eval_input,
                output_path=str(eval_output),
                device=args.eval_device,
                max_new_tokens=args.eval_max_new_tokens,
                max_samples=args.eval_max_samples,
                local_files_only=args.local_files_only,
            )
            print(f"\n[E23/E24 eval] {variant}: {' '.join(eval_command)}")
            eval_returncode = _run_one(eval_command, cwd=repo_root, log_path=eval_log, env=env)
            results[variant].update(
                {
                    "eval_status": "pass" if eval_returncode == 0 else "fail",
                    "eval_returncode": eval_returncode,
                    "eval_command": eval_command,
                    "generation_eval": str(eval_output),
                    "generation_eval_log": str(eval_log),
                }
            )
            if eval_returncode != 0 and args.stop_on_failure:
                break
        if returncode != 0 and args.stop_on_failure:
            break
    manifest = {
        "experiment": "E23-E24",
        "config": str(args.config),
        "variants": results,
        "max_steps": args.max_steps,
        "num_samples": args.num_samples,
        "seed": args.seed,
    }
    (output_root / "matrix_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="shared pilot YAML config")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--variants", nargs="+", choices=sorted(OBJECTIVE_VARIANTS))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--eval-input", help="holdout JSONL for generation evaluation")
    parser.add_argument("--eval-device", default="cuda")
    parser.add_argument("--eval-max-new-tokens", type=int, default=128)
    parser.add_argument("--eval-max-samples", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(argv)
    manifest = run_matrix(args)
    print(json.dumps({"experiment": manifest["experiment"], "variants": manifest["variants"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
