#!/usr/bin/env python3
"""Smoke/full adapter for the vendored FAFO implementation.

FAFO's public upstream runner is GSM8K-oriented and expects ``question`` /
``answer`` JSONL.  The adapter converts one or more unified repository records
to that input, creates an isolated FAFO config, invokes the upstream runner,
and normalizes its log metrics into the shared JSONL schema.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from common import io_util, verify
from common.data_loader import load_records
from common.paths import ROOT


FAFO_ROOT = ROOT / "externals" / "FAFO"
FAFO_PIPELINE_MAIN = "pipeline/fafo/main.py"


def _config_path(kv_method: str) -> Path:
    return (
        FAFO_ROOT
        / "config/pipeline_config/fafo/gsm8k/Llama-3.1-8B-Instruct"
        / kv_method
        / "default.json"
    )


def build_pipeline_config(
    model: str,
    max_new_tokens: int,
    kv_method: str = "stream-llm",
    use_flash: bool = False,
) -> dict[str, Any]:
    """Load an upstream FAFO config and apply run-specific values."""

    if kv_method not in {"stream-llm", "quest"}:
        raise ValueError(f"unsupported FAFO KV method: {kv_method}")
    config = json.loads(_config_path(kv_method).read_text(encoding="utf-8"))
    params = config["pipeline_params"]
    params["model_name"] = model
    params["n_new_tokens"] = max(1, int(max_new_tokens))
    params["use_flash"] = bool(use_flash)
    return config


def build_eval_config(dataset_path: str) -> dict[str, Any]:
    """Build the upstream GSM8K evaluator config for a generated JSONL file."""

    return {
        "eval_params": {
            "dataset": "gsm8k",
            "dataset_path": dataset_path,
            "max_new_tokens": 1024,
            "eval_metrics": ["throughput", "avg_acceptance_len"],
        },
        "management": {
            "sub_dir": {
                "input_config": "input_config/",
                "raw_results": "raw_results.json",
                "output_config": "output_config.json",
            }
        },
    }


def build_command(
    *,
    python: str,
    pipeline_config: str,
    eval_config: str,
    output_dir: str,
    exp_desc: str,
) -> list[str]:
    """Build the upstream FAFO command used by the adapter."""

    return [
        python,
        FAFO_PIPELINE_MAIN,
        "--exp_desc",
        exp_desc,
        "--pipeline_config_dir",
        pipeline_config,
        "--eval_config_dir",
        eval_config,
        "--output_folder_dir",
        output_dir,
    ]


def _resolve_repo_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _write_fafo_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(
                {
                    "question": record["prompt"],
                    "answer": str(record.get("reference") or record.get("answer") or ""),
                },
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    entries = [str(FAFO_ROOT), str(ROOT / "scripts")]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _parse_log(log: str) -> dict[str, float | int | None]:
    """Extract the one-sample timing line emitted by upstream FAFO."""

    time_match = re.findall(r"time:\s*([0-9.eE+-]+)", log)
    token_match = re.findall(r"generated tokens:\s*([0-9]+)", log)
    average_match = re.findall(
        r"AVERAGE THROUGHPUT2\s+([0-9.eE+-]+)", log
    )
    e2e_s = float(time_match[-1]) if time_match else None
    output_tokens = int(token_match[-1]) if token_match else None
    throughput = float(average_match[-1]) if average_match else None
    if throughput is None and e2e_s and output_tokens is not None:
        throughput = output_tokens / e2e_s
    return {
        "e2e_s": e2e_s,
        "output_tokens": output_tokens,
        "throughput": throughput,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("FAFO_MODEL", os.environ.get("MODEL_TARGET", "meta-llama/Llama-3.1-8B-Instruct")))
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--prompt", default="What is 2 + 2? Give the answer briefly.")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--kv-method", choices=["stream-llm", "quest"], default="stream-llm")
    parser.add_argument("--use-flash", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.smoke:
        args.max_samples = 1
        args.max_new_tokens = min(args.max_new_tokens, 32)

    data_file = _resolve_repo_path(args.data_file)
    if data_file:
        records = load_records(data_file, args.max_samples)
    else:
        records = [{"id": "prompt", "prompt": args.prompt, "reference": None}]
    if not records:
        raise SystemExit("FAFO input contains no usable records")

    output_path = Path(args.output)
    writer = io_util.JsonlWriter(output_path)
    runtime_dir = output_path if output_path.is_absolute() else ROOT / output_path
    runtime_dir = runtime_dir.parent / f"{runtime_dir.stem}_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    process_returncode = 1
    process_log = ""
    raw_result: Any = None
    with tempfile.TemporaryDirectory(prefix="fafo-input-") as temp_dir:
        temp_root = Path(temp_dir)
        dataset_path = temp_root / "one_sample.jsonl"
        pipeline_path = temp_root / "pipeline.json"
        eval_path = temp_root / "eval.json"
        _write_fafo_dataset(dataset_path, records)
        pipeline_path.write_text(
            json.dumps(
                build_pipeline_config(
                    args.model,
                    args.max_new_tokens,
                    args.kv_method,
                    args.use_flash,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        eval_path.write_text(
            json.dumps(build_eval_config(str(dataset_path)), indent=2),
            encoding="utf-8",
        )
        command = build_command(
            python=os.environ.get("FAST_INFER_PYTHON", sys.executable),
            pipeline_config=str(pipeline_path),
            eval_config=str(eval_path),
            output_dir=str(runtime_dir),
            exp_desc=f"fafo_{args.kv_method}_{'smoke' if args.smoke else 'run'}",
        )
        print("+ " + " ".join(command))
        proc = subprocess.run(
            command,
            cwd=FAFO_ROOT,
            env=_runtime_env(),
            capture_output=True,
            text=True,
        )
        process_returncode = proc.returncode
        process_log = (proc.stdout or "") + (proc.stderr or "")
        print(process_log[-4000:])

    raw_path = runtime_dir / "raw_results.json"
    if raw_path.is_file():
        try:
            raw_result = json.loads(raw_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw_result = None
    parsed = _parse_log(process_log)
    e2e_s = parsed["e2e_s"]
    output_tokens = parsed["output_tokens"]
    e2e_ms = round(float(e2e_s) * 1000, 3) if e2e_s is not None else None
    output_number = int(output_tokens) if output_tokens is not None else 0
    record = {
        "method": f"fafo_{args.kv_method}",
        "dataset": data_file.name if data_file else "prompt",
        "task_type": records[0].get("raw", {}).get("task_type"),
        "status": "success" if process_returncode == 0 and output_number > 0 else "failed",
        "scope": "aggregate",
        "model": args.model,
        "input_tokens": None,
        "retained_tokens": None,
        "output_tokens": output_tokens,
        "batch_size": len(records),
        "selector_latency_ms": None,
        "ttft_ms": None,
        "tpot_ms": round(e2e_ms / output_number, 3) if e2e_ms and output_number else None,
        "e2e_ms": e2e_ms,
        "throughput_tok_s": parsed["throughput"],
        "qps": round(1 / float(e2e_s), 6) if e2e_s and e2e_s > 0 else None,
        "peak_memory_gb": None,
        "avg_accept_length": None,
        "acceptance_rate": None,
        "draft_latency_ms": None,
        "verification_latency_ms": None,
        "rejected_draft_ratio": None,
        "sample_ids": [sample["id"] for sample in records],
        "kv_method": args.kv_method,
        "smoke": args.smoke,
        "returncode": process_returncode,
        "raw_result": raw_result,
        "log_tail": process_log[-3000:],
    }
    writer.add(record)

    checks: list[tuple[bool, str]] = [
        (process_returncode == 0, f"FAFO process exit code = {process_returncode}"),
        (output_number > 0, f"generated tokens = {output_number} (> 0)"),
        (parsed["throughput"] is not None, "FAFO timing/throughput line parsed"),
    ]
    summary = {
        "type": "summary",
        "method": f"fafo_{args.kv_method}",
        "returncode": process_returncode,
        "num_samples": len(records),
        "checks_passed": all(ok for ok, _ in checks),
        "runtime_dir": str(runtime_dir),
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {output_path}")
    verify.finish("FAFO", checks)


if __name__ == "__main__":
    main()
