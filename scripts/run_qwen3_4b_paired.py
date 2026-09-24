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
import hashlib
import json
import math
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
from common.benchmark_runtime import runtime_metadata  # noqa: E402
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
    parser.add_argument("--eagle-total-token", type=int, default=32)
    parser.add_argument("--eagle-depth", type=int, default=5)
    parser.add_argument("--eagle-top-k", type=int, default=4)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_dataset(source: Path, destination: Path, *, max_samples: int) -> int:
    """Copy the exact first N non-empty JSONL rows used by this run."""

    selected: list[bytes] = []
    with Path(source).open("rb") as handle:
        for line in handle:
            if line.strip():
                selected.append(line)
                if len(selected) >= int(max_samples):
                    break
    if not selected:
        raise ValueError(f"no JSONL samples to snapshot from {source}")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(
        line if line.endswith((b"\n", b"\r")) else line + b"\n"
        for line in selected
    )
    destination.write_bytes(payload)
    return len(selected)


def _read_sample_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "summary" and row.get("scope") != "aggregate":
                rows.append(row)
    return rows


def validate_paired_records(
    records: list[dict[str, Any]],
    *,
    baseline: str,
    expected_samples: int,
    output_tokens: int,
) -> list[str]:
    """Reject incomplete rows before the run is reported as benchmark-valid."""

    errors: list[str] = []
    successful = [row for row in records if row.get("status", "success") == "success"]
    if len(successful) != expected_samples:
        errors.append(
            f"{baseline}: successful samples {len(successful)}/{expected_samples}"
        )
    sample_ids = [str(row.get("sample_id")) for row in successful]
    if len(set(sample_ids)) != len(sample_ids):
        errors.append(f"{baseline}: duplicate sample_id values")
    for index, row in enumerate(records):
        label = f"{baseline} record {row.get('sample_id', index)}"
        if row.get("status", "success") != "success":
            errors.append(f"{label}: status={row.get('status')}")
            continue
        if row.get("batch_size") != 1:
            errors.append(f"{label}: batch_size must be 1")
        if row.get("measurement_scope") != "full_e2e":
            errors.append(f"{label}: measurement_scope must be full_e2e")
        if int(row.get("output_tokens") or 0) != output_tokens:
            errors.append(f"{label}: output_tokens must equal {output_tokens}")
        if int(row.get("dense_output_tokens") or 0) != output_tokens:
            errors.append(f"{label}: dense_output_tokens must equal {output_tokens}")
        if row.get("speed_output_tokens") != output_tokens:
            errors.append(f"{label}: speed_output_tokens must equal {output_tokens}")
        for field in (
            "prefill_ms", "decode_ms", "e2e_ms", "throughput_tok_s",
            "decode_throughput_tok_s", "dense_prefill_ms", "dense_decode_ms",
            "dense_e2e_ms",
        ):
            value = row.get(field)
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                errors.append(f"{label}: required timing/throughput {field} is missing or invalid")
        token_ids = row.get("generated_token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != output_tokens:
            errors.append(f"{label}: raw generated_token_ids missing or wrong length")
        if not isinstance(row.get("full_output_text"), str):
            errors.append(f"{label}: full_output_text is missing")
        if row.get("fixed_budget_reached") is not True:
            errors.append(f"{label}: fixed output budget was not reached")
        if not isinstance(row.get("quality_text"), str):
            errors.append(f"{label}: quality_text is missing")
        if row.get("output_parity_available") is not True:
            errors.append(f"{label}: output token parity vs Vanilla HF is unavailable")
        if not isinstance(row.get("fixed_continuation_exact_match"), bool):
            errors.append(f"{label}: fixed continuation exact-match metric is missing")
        if not isinstance(row.get("quality_prefix_exact_match"), bool):
            errors.append(f"{label}: quality-prefix exact-match metric is missing")
        first_eos = row.get("first_eos_index")
        quality_tokens = row.get("quality_output_tokens")
        expected_quality_tokens = (
            int(first_eos) if first_eos is not None else output_tokens
        )
        if quality_tokens != expected_quality_tokens:
            errors.append(f"{label}: quality_output_tokens is inconsistent with EOS")
        if first_eos is not None and (
            not isinstance(token_ids, list)
            or first_eos < 0
            or first_eos >= len(token_ids)
            or token_ids[first_eos] not in (
                row.get("eos_token_id")
                if isinstance(row.get("eos_token_id"), list)
                else [row.get("eos_token_id")]
            )
        ):
            errors.append(f"{label}: first_eos_index does not point to EOS")
        if not row.get("prompt_hash") or not row.get("run_config_hash"):
            errors.append(f"{label}: pairing hashes are missing")
        if row.get("reference_baseline") != "vanilla_hf":
            errors.append(f"{label}: reference_baseline must be vanilla_hf")
        if not row.get("reference_run_id"):
            errors.append(f"{label}: reference_run_id is missing")
        if row.get("speedup_scope") != "qwen3_4b_batch1_fixed_budget":
            errors.append(f"{label}: speedup_scope is invalid")
        if baseline in {"dflash", "domino", "eagle3"}:
            acceptance = row.get("acceptance_lengths")
            if not isinstance(acceptance, list) or not acceptance:
                errors.append(f"{label}: speculative acceptance trace is missing")
            if row.get("avg_accept_length") is None:
                errors.append(f"{label}: tau/avg_accept_length is missing")
            if row.get("tau_scope") != "fixed_budget":
                errors.append(f"{label}: tau scope must be fixed_budget")
        if row.get("speedup_valid") is not True:
            errors.append(
                f"{label}: speedup_valid is false; online pair invalid "
                f"({row.get('speedup_invalid_reason') or 'missing reason'})"
            )
        for metric in ("esr", "dsr"):
            value = row.get(metric)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                errors.append(f"{label}: online {metric.upper()} missing or invalid")
        if row.get("speedup_valid") is True:
            dense_e2e, method_e2e = row.get("dense_e2e_ms"), row.get("e2e_ms")
            if isinstance(dense_e2e, (int, float)) and isinstance(method_e2e, (int, float)) and method_e2e > 0:
                expected_esr = dense_e2e / method_e2e
                if not math.isclose(float(row.get("esr") or 0), expected_esr, rel_tol=1e-6):
                    errors.append(f"{label}: stored ESR does not match online paired timings")
            method_rate = row.get("method_decode_tok_s")
            dense_rate = row.get("dense_decode_tok_s")
            if isinstance(method_rate, (int, float)) and isinstance(dense_rate, (int, float)) and dense_rate > 0:
                expected_dsr = method_rate / dense_rate
                if not math.isclose(float(row.get("dsr") or 0), expected_dsr, rel_tol=1e-6):
                    errors.append(f"{label}: stored DSR does not match online paired throughput")
    return errors


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


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
    dataset_manifest: list[dict[str, Any]] = []
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
            extra={"dataset_sha256": _sha256(data_file)},
        )
        cfg_hash = str(cfg["run_config_hash"])
        dataset_manifest.append({
            "dataset": dataset,
            "path": str(data_file.resolve()),
            "sha256": _sha256(data_file),
            "sample_count": count,
            "run_config": cfg,
        })
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
                "--total-token", str(args.eagle_total_token), "--depth", str(args.eagle_depth),
                "--top-k", str(args.eagle_top_k),
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
        "dataset_inputs": dataset_manifest,
        "model_paths": {
            "target": args.target_model,
            "dflash_draft": args.dflash_model if "dflash" in baselines else None,
            "domino_draft": args.domino_model if "domino" in baselines else None,
            "eagle_draft": args.eagle_model if "eagle3" in baselines else None,
        },
        "method_parameters": {
            "warmup_runs": args.warmup_runs,
            "vanilla_fa_attention_backend": args.vanilla_fa_backend,
            "dflash_block_size": "from loaded draft checkpoint unless overridden",
            "domino_block_size": "from loaded draft checkpoint unless overridden",
            "eagle3": {
                "total_token": args.eagle_total_token,
                "depth": args.eagle_depth,
                "top_k": args.eagle_top_k,
                "skip_naive_internal_reference": True,
            },
        },
        "measurement_contract": {
            "batch_size": 1,
            "input_truncation": "deterministic head-tail",
            "fixed_output_tokens": budget,
            "fixed_generation_stops_on_eos": False,
            "quality_text": "decode generated_token_ids before the first EOS",
            "decode_tok_s": "sum(output_tokens) / sum(decode_ms) * 1000 over valid samples",
            "esr": "paired Vanilla HF e2e_ms / method e2e_ms, computed per sample during inference",
            "dsr": "method decode tok/s / paired Vanilla HF decode tok/s, computed per sample during inference",
            "pair_key": ["dataset", "sample_id", "prompt_hash", "run_config_hash"],
            "tau_scope": "mean speculative acceptance length over the fixed-K decode trace",
        },
        "stored_sample_fields": [
            "input_tokens", "original_input_tokens", "input_truncated",
            "prompt_hash", "run_config_hash", "generated_token_ids",
            "first_eos_index", "quality_output_tokens", "quality_text",
            "full_output_text", "prefill_ms", "decode_ms", "e2e_ms",
            "throughput_tok_s", "decode_throughput_tok_s", "acceptance_lengths",
            "avg_accept_length", "acceptance_rate", "esr", "dsr",
            "dense_prefill_ms", "dense_decode_ms", "dense_e2e_ms",
            "output_parity_available", "fixed_continuation_exact_match",
            "quality_prefix_exact_match", "fixed_continuation_token_match_ratio",
            "target_model_revision", "draft_model_revision",
            "tokenizer_name_or_path", "tokenizer_revision",
        ],
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
    manifest_path = run_root / "run_manifest.json"
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["gpu_id"] = str(args.gpu_id)
    manifest["runtime"] = runtime_metadata()
    manifest["status"] = "preparing"
    _write_manifest(manifest_path, manifest)
    snapshot_paths: dict[str, Path] = {}
    for dataset in manifest["dataset_inputs"]:
        source = Path(dataset["path"])
        if _sha256(source) != dataset["sha256"]:
            raise RuntimeError(f"dataset changed after preflight: {source}")
        snapshot = run_root / "inputs" / "source" / f"{dataset['dataset']}.jsonl"
        count = snapshot_dataset(
            source,
            snapshot,
            max_samples=int(dataset["sample_count"]),
        )
        if count != int(dataset["sample_count"]):
            raise RuntimeError(
                f"dataset snapshot sample count mismatch for {dataset['dataset']}: "
                f"{count}/{dataset['sample_count']}"
            )
        dataset["snapshot_path"] = str(snapshot)
        dataset["snapshot_sha256"] = _sha256(snapshot)
        snapshot_paths[dataset["dataset"]] = snapshot
    for item in commands:
        command = item["command"]
        if "--data-file" in command:
            data_index = command.index("--data-file")
            command[data_index + 1] = str(snapshot_paths[item["dataset"]])
        if item.get("question_file"):
            expected_count = next(
                int(value["sample_count"])
                for value in manifest["dataset_inputs"]
                if value["dataset"] == item["dataset"]
            )
            _make_question_file(
                snapshot_paths[item["dataset"]],
                Path(item["question_file"]),
                expected_count,
            )
    manifest["commands"] = [_command_text(item["command"]) for item in commands]
    manifest["command_plan"] = [
        {
            "dataset": item["dataset"],
            "baseline": item["baseline"],
            "output": item["output"],
            "argv": item["command"],
        }
        for item in commands
    ]
    manifest["status"] = "planned" if args.dry_run else "running"
    manifest["command_results"] = []
    _write_manifest(manifest_path, manifest)
    for item in commands:
        print(f"[{item['dataset']}][{item['baseline']}] {_command_text(item['command'])}")
    if args.dry_run:
        print(f"Dry-run manifest: {manifest_path}")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    manifest["started_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    sample_counts = {
        item["dataset"]: int(item["sample_count"])
        for item in manifest["dataset_inputs"]
    }
    try:
        for item in commands:
            output = Path(item["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            print(f"\nRunning {item['dataset']} / {item['baseline']} ...", flush=True)
            started = datetime.now(timezone.utc).isoformat()
            completed = subprocess.run(
                item["command"], cwd=ROOT, env=env, check=False
            )
            result = {
                "dataset": item["dataset"],
                "baseline": item["baseline"],
                "output": str(output),
                "started_at": started,
                "returncode": completed.returncode,
            }
            if completed.returncode != 0:
                result["status"] = "failed"
                manifest["command_results"].append(result)
                raise RuntimeError(
                    f"{item['baseline']}/{item['dataset']} exited "
                    f"with code {completed.returncode}"
                )

            rows = _read_sample_rows(output)
            errors = validate_paired_records(
                rows,
                baseline=item["baseline"],
                expected_samples=sample_counts[item["dataset"]],
                output_tokens=int(manifest["fixed_output_tokens"]),
            )
            result["sample_records"] = len(rows)
            if errors:
                result["status"] = "invalid"
                result["validation_errors"] = errors
                manifest["command_results"].append(result)
                raise RuntimeError("paired output validation failed: " + "; ".join(errors))
            if item["baseline"] == "vanilla_hf":
                sidecar = Path(item["reference_sidecar"])
                count = extract_reference_sidecar(output, sidecar)
                if count != sample_counts[item["dataset"]]:
                    result["status"] = "invalid"
                    result["validation_errors"] = [
                        f"reference sidecar sample count {count}/"
                        f"{sample_counts[item['dataset']]}"
                    ]
                    manifest["command_results"].append(result)
                    raise RuntimeError(result["validation_errors"][0])
                result["reference_sidecar"] = str(sidecar)
                print(f"Reference sidecar: {sidecar} ({count} samples)", flush=True)
            result["status"] = "complete"
            result["completed_at"] = datetime.now(timezone.utc).isoformat()
            manifest["command_results"].append(result)
            _write_manifest(manifest_path, manifest)

        selected_datasets = ",".join(manifest["datasets"])
        collector_command = [
            args.python_executable,
            str(SCRIPTS / "collect_metrics.py"),
            "--outputs-dir", str(run_root),
            "--data-dir", str(run_root / "inputs" / "source"),
            "--datasets", selected_datasets,
        ]
        subprocess.run(collector_command, cwd=ROOT, env=env, check=True)
        manifest["postprocess"] = {
            "status": "complete",
                "command": collector_command,
                "immutable_data_dir": str(run_root / "inputs" / "source"),
            "artifacts": [
                str(run_root / "metrics_summary.json"),
                str(run_root / "metrics_summary.csv"),
                str(run_root / "metrics_summary.md"),
            ],
        }
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_manifest(manifest_path, manifest)
        raise
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_manifest(manifest_path, manifest)
    print(f"\nCompleted paired matrix. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
