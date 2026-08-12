#!/usr/bin/env python3
"""SpecExtend verification / smoke script.

Runs the repo's own `run_classic.py` (or `run_eagle.py`) on one of the bundled
govreport jsonl files and verifies that a generated summary line appears.

Smoke (T4): vicuna-7b + govreport_512 + max_gen_len 64 (marginal on 16GB).
Full: govreport_1K..16K or eval_eagle.py sweep on a bigger GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from common import io_util, verify
from common.paths import ROOT

SPECEXTEND = ROOT / "externals" / "SpecExtend" / "specextend"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", default="run_classic.py",
                        choices=["run_classic.py", "run_eagle.py"])
    parser.add_argument("--model-name", default="vicuna_7b",
                        choices=["vicuna_7b", "longchat_7b"])
    parser.add_argument("--base-model", default=None,
                        help="override base model path/id (T4 smoke: TinyLlama)")
    parser.add_argument("--draft-model", default=None,
                        help="override draft model path/id (T4 smoke: TinyLlama)")
    parser.add_argument("--input-file", default="data/govreport/govreport_512.jsonl")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-gen-len", type=int, default=64)
    parser.add_argument("--use-specextend", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.smoke:
        args.max_samples = 1
        args.max_gen_len = min(args.max_gen_len, 64)

    cmd = [
        "python", args.script,
        "--input_file", str(SPECEXTEND / args.input_file),
        "--model_name", args.model_name,
        "--max_samples", str(args.max_samples),
        "--max_gen_len", str(args.max_gen_len),
        "--output_result_line",
    ]
    if args.use_specextend:
        cmd += ["--use_specextend"]

    print("+ " + " ".join(cmd))
    env = dict(__import__("os").environ)
    # Optional model overrides (T4 smoke uses TinyLlama via the wrapper).
    if args.base_model:
        env["SPECEXTEND_BASE_MODEL"] = args.base_model
    if args.draft_model:
        env["SPECEXTEND_DRAFT_MODEL"] = args.draft_model
    proc = subprocess.run(cmd, cwd=SPECEXTEND, env=env,
                          capture_output=True, text=True)
    stdout = proc.stdout or ""
    log = stdout + (proc.stderr or "")
    print(log[-4000:])

    # run_classic prints one result line per sample (JSON-ish).
    result_lines = [
        line.strip() for line in stdout.splitlines()
        if line.strip() and "{" in line and "summary" in line.lower()
    ]
    got_output = bool(result_lines)

    writer = io_util.JsonlWriter(Path(args.output))
    record = {
        "method": "specextend_classic",
        "dataset": Path(args.input_file).name,
        "model": args.model_name,
        "input_tokens": None,
        "retained_tokens": None,
        "output_tokens": None,
        "batch_size": 1,
        "selector_latency_ms": None,
        "ttft_ms": None,
        "tpot_ms": None,
        "e2e_ms": None,
        "throughput_tok_s": None,
        "qps": None,
        "peak_memory_gb": None,
        "returncode": proc.returncode,
        "result_lines": result_lines[:3],
        "log_tail": log[-2000:],
    }
    writer.add(record)

    checks: list[tuple[bool, str]] = [
        (proc.returncode == 0, f"process exit code = {proc.returncode}"),
        (got_output, f"generated summary line(s): {len(result_lines)}"),
    ]
    summary = {"type": "summary", "method": record["method"],
               "returncode": proc.returncode, "got_output": got_output}
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("SpecExtend", checks)


if __name__ == "__main__":
    main()
