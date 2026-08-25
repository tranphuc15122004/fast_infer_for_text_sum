#!/usr/bin/env python3
"""SpecExtend verification / smoke adapter.

Runs SpecExtend's classic or EAGLE path on a JSONL file and normalizes its
human-readable result into this repository's JSONL schema.  The Llama-3.1
configuration uses the EAGLE path; an EAGLE checkpoint is not a classic
``AutoModelForCausalLM`` draft model and must not be passed to ``run_classic``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

from common import io_util, verify
from common.paths import ROOT

SPECEXTEND = ROOT / "externals" / "SpecExtend" / "specextend"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", default="run_classic.py",
                        choices=["run_classic.py", "run_eagle.py"])
    parser.add_argument("--model-name", default="vicuna_7b",
                        choices=["vicuna_7b", "longchat_7b", "llama3_1_8b"])
    parser.add_argument("--base-model", default=None,
                        help="override base model path/id")
    parser.add_argument("--draft-model", default=None,
                        help="override draft model path/id")
    parser.add_argument("--input-file", default="data/govreport/govreport_512.jsonl")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--max-gen-len", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=0,
                        help="truncate each input before inference; 0 disables truncation")
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--use-specextend",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="enable/disable SpecExtend hybrid attention")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.smoke:
        args.max_samples = 1
        args.max_gen_len = min(args.max_gen_len, 64)
        args.warmup_runs = 0

    input_file = Path(args.input_file)
    if not input_file.is_absolute():
        # Direct SpecExtend data lives below externals/SpecExtend, while the
        # representative runner writes converted data below the repo root.
        candidates = (ROOT / input_file, SPECEXTEND / input_file)
        input_file = next((candidate for candidate in candidates if candidate.exists()),
                          candidates[0])

    cmd = [
        "python", args.script,
        "--input_file", str(input_file),
        "--model_name", args.model_name,
        "--max_samples", str(args.max_samples),
        "--max_gen_len", str(args.max_gen_len),
        "--max_input_tokens", str(args.max_input_tokens),
        "--output_result_line",
    ]
    if args.use_specextend:
        cmd += ["--use_specextend"]

    print("+ " + " ".join(cmd))
    env = dict(__import__("os").environ)
    # Model paths are consumed by both SpecExtend entrypoints.  The EAGLE
    # entrypoint uses the draft path as an EAGLE checkpoint, not a classic LLM.
    if args.base_model:
        env["SPECEXTEND_BASE_MODEL"] = args.base_model
    if args.draft_model:
        env["SPECEXTEND_DRAFT_MODEL"] = args.draft_model
    env["SPECEXTEND_WARMUP_RUNS"] = str(args.warmup_runs)
    env["SPECEXTEND_MAX_INPUT_TOKENS"] = str(args.max_input_tokens)
    proc = subprocess.run(cmd, cwd=SPECEXTEND, env=env,
                          capture_output=True, text=True)
    stdout = proc.stdout or ""
    log = stdout + (proc.stderr or "")
    print(log[-4000:])

    # Both SpecExtend entrypoints print metrics as human-readable lines rather
    # than JSON.
    result_lines = [
        line.strip() for line in stdout.splitlines()
        if line.strip() and "Generated " in line and " tokens in " in line
    ]
    got_output = bool(result_lines)
    generated_tokens = None
    if result_lines:
        match = re.search(r"Generated\s+(\d+)\s+tokens", result_lines[-1])
        if match:
            generated_tokens = int(match.group(1))

    writer = io_util.JsonlWriter(Path(args.output))
    record = {
        "method": "specextend_eagle" if args.script == "run_eagle.py"
        else "specextend_classic",
        "dataset": Path(args.input_file).name,
        "model": args.base_model or args.model_name,
        "draft_model": args.draft_model,
        "model_name": args.model_name,
        "script": args.script,
        "max_input_tokens": args.max_input_tokens,
        "smoke": args.smoke,
        "input_tokens": None,
        "retained_tokens": None,
        "output_tokens": generated_tokens,
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
        (got_output, f"generated result line(s): {len(result_lines)}"),
    ]
    summary = {"type": "summary", "method": record["method"],
               "returncode": proc.returncode, "got_output": got_output}
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("SpecExtend", checks)


if __name__ == "__main__":
    main()
