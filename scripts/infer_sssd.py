#!/usr/bin/env python3
"""Smoke/full adapter for the vendored SSSD SGLang fork.

The upstream benchmark accepts ShareGPT-like conversations.  This adapter
converts the repository's unified JSONL records to that format, runs exactly
the upstream offline-throughput entrypoint, and writes one aggregate record
plus a summary to the repository schema.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from common import io_util, verify
from common.data_loader import load_records
from common.paths import ROOT


SSSD_ROOT = ROOT / "externals" / "SSSD"
SSSD_PYTHON = SSSD_ROOT / "python"
SSSD_SPECULATOR = SSSD_ROOT / "sssd_speculator"

# Standalone stdlib-``json`` implementation of the ``orjson`` API subset the
# vendored SGLang fork uses.  It is copied to a temp dir as ``orjson.py`` and
# prepended to the child PYTHONPATH only when the real package is missing.
ORJSON_COMPAT_SOURCE = ROOT / "scripts" / "common" / "orjson_compat.py"


def _child_imports_orjson(python: str) -> bool:
    """Return True when ``python`` can import the real ``orjson`` package."""
    try:
        probe = subprocess.run(
            [python, "-c", "import orjson"],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return probe.returncode == 0


@contextlib.contextmanager
def _with_orjson_compat(python: str, env: dict[str, str]):
    """Yield ``env``, prepending a stdlib-``json`` ``orjson`` shim when needed.

    The vendored SGLang fork imports ``orjson`` at module scope even for the
    offline throughput entrypoint, so without a fallback an SSSD run dies with
    ``ModuleNotFoundError: No module named 'orjson'`` on dependency-light
    servers.  When the real package is importable this is a no-op.
    """
    if not ORJSON_COMPAT_SOURCE.is_file() or _child_imports_orjson(python):
        yield env
        return
    with tempfile.TemporaryDirectory(prefix="orjson-compat-") as tmp_dir:
        shutil.copyfile(ORJSON_COMPAT_SOURCE, Path(tmp_dir) / "orjson.py")
        patched = dict(env)
        entries = [tmp_dir] + [
            entry
            for entry in patched.get("PYTHONPATH", "").split(os.pathsep)
            if entry
        ]
        patched["PYTHONPATH"] = os.pathsep.join(entries)
        print(
            f"[SSSD] orjson unavailable; using stdlib-json compat module "
            f"({ORJSON_COMPAT_SOURCE})"
        )
        yield patched


def build_command(
    *,
    python: str,
    model: str,
    dataset: str,
    result_file: str,
    max_new_tokens: int,
    datastore_path: str | None = None,
    num_draft_tokens: int = 8,
    num_steps: int = 5,
    topk: int = 5,
    adaptive: bool = False,
    num_prompts: int = 1,
    context_len: int = 0,
) -> list[str]:
    """Build the upstream SSSD offline benchmark command."""

    command = [
        python,
        "-m",
        "sglang.bench_offline_throughput",
        "--model-path",
        model,
        "--dataset-name",
        "custom",
        "--dataset-path",
        dataset,
        "--num-prompts",
        str(num_prompts),
        "--sharegpt-output-len",
        str(max(4, max_new_tokens)),
        "--result-filename",
        result_file,
        "--apply-chat-template",
        "--skip-warmup",
        "--disable-ignore-eos",
        "--temperature",
        "0",
        "--speculative-algorithm",
        "SSSD",
        "--speculative-num-draft-tokens",
        str(num_draft_tokens),
        "--speculative-num-steps",
        str(num_steps),
        "--speculative-eagle-topk",
        str(topk),
    ]
    if context_len > 0:
        command += ["--sharegpt-context-len", str(context_len)]
    if datastore_path:
        command += ["--speculative-draft-model-path", datastore_path]
    if adaptive:
        command.append("--speculative-adaptive")
    return command


def _resolve_repo_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _write_custom_dataset(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            completion = record.get("reference") or record.get("answer") or "ok"
            json.dump(
                {
                    "conversations": [
                        {"from": "human", "value": record["prompt"]},
                        {"from": "gpt", "value": str(completion)},
                    ]
                },
                handle,
                ensure_ascii=False,
            )
            handle.write("\n")


def _last_json_line(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    entries = [str(SSSD_PYTHON), str(SSSD_SPECULATOR), str(ROOT / "scripts")]
    if env.get("PYTHONPATH"):
        entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("MODEL_TARGET", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--prompt", default="Summarize the following document in one sentence.")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--datastore-path", default=os.environ.get("SSSD_DATASTORE_PATH", ""))
    parser.add_argument("--num-draft-tokens", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--adaptive", action="store_true")
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
        raise SystemExit("SSSD input contains no usable records")

    output_path = Path(args.output)
    writer = io_util.JsonlWriter(output_path)
    result: dict[str, Any] | None = None
    process_log = ""
    process_returncode = 1

    with tempfile.TemporaryDirectory(prefix="sssd-smoke-") as temp_dir:
        temp_root = Path(temp_dir)
        dataset_path = temp_root / "custom.jsonl"
        raw_result_path = temp_root / "sssd_result.jsonl"
        _write_custom_dataset(dataset_path, records)
        command = build_command(
            python=os.environ.get("FAST_INFER_PYTHON", sys.executable),
            model=args.model,
            dataset=str(dataset_path),
            result_file=str(raw_result_path),
            max_new_tokens=args.max_new_tokens,
            datastore_path=args.datastore_path or None,
            num_draft_tokens=args.num_draft_tokens,
            num_steps=args.num_steps,
            topk=args.topk,
            adaptive=args.adaptive,
            num_prompts=len(records),
            context_len=args.max_input_tokens,
        )
        print("+ " + " ".join(command))
        with _with_orjson_compat(
            python=command[0] if command else sys.executable,
            env=_runtime_env(),
        ) as child_env:
            proc = subprocess.run(
                command,
                cwd=SSSD_ROOT,
                env=child_env,
                capture_output=True,
                text=True,
            )
        process_returncode = proc.returncode
        process_log = (proc.stdout or "") + (proc.stderr or "")
        print(process_log[-4000:])
        result = _last_json_line(raw_result_path)

    result = result or {}
    successful = int(result.get("successful_requests", 0) or 0)
    input_tokens = result.get("total_input_tokens")
    output_tokens = result.get("total_output_tokens")
    e2e_s = _float_or_none(result.get("total_latency"))
    throughput = _float_or_none(result.get("output_throughput"))
    extra = result.get("extra_metrics") or {}
    e2e_ms = round(e2e_s * 1000, 3) if e2e_s is not None else None
    output_number = int(output_tokens) if output_tokens is not None and output_tokens >= 0 else 0
    record = {
        "method": "sssd",
        "dataset": data_file.name if data_file else "prompt",
        "task_type": records[0].get("raw", {}).get("task_type"),
        "status": "success" if process_returncode == 0 and successful == len(records) else "failed",
        "scope": "aggregate",
        "model": args.model,
        "input_tokens": input_tokens,
        "retained_tokens": None,
        "output_tokens": output_tokens,
        "batch_size": len(records),
        "selector_latency_ms": None,
        "ttft_ms": None,
        "tpot_ms": round(e2e_ms / output_number, 3) if e2e_ms and output_number else None,
        "e2e_ms": e2e_ms,
        "throughput_tok_s": throughput,
        "qps": _float_or_none(result.get("request_throughput")),
        "peak_memory_gb": None,
        "avg_accept_length": _float_or_none(extra.get("avg_acceptance_length")),
        "acceptance_rate": None,
        "draft_latency_ms": None,
        "verification_latency_ms": None,
        "rejected_draft_ratio": None,
        "sample_ids": [sample["id"] for sample in records],
        "smoke": args.smoke,
        "returncode": process_returncode,
        "successful_requests": successful,
        "raw_result": result,
        "log_tail": process_log[-3000:],
    }
    writer.add(record)

    checks: list[tuple[bool, str]] = [
        (process_returncode == 0, f"SSSD process exit code = {process_returncode}"),
        (successful == len(records), f"successful requests = {successful}/{len(records)}"),
        (output_number > 0, f"generated tokens = {output_number} (> 0)"),
    ]
    summary = {
        "type": "summary",
        "method": "sssd",
        "returncode": process_returncode,
        "num_samples": len(records),
        "checks_passed": all(ok for ok, _ in checks),
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {output_path}")
    verify.finish("SSSD", checks)


if __name__ == "__main__":
    main()
