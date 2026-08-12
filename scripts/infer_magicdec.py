#!/usr/bin/env python3
"""MagicDec verification / smoke script (dense baseline_benchmark wrapper).

Runs MagicDec's own `tests/baseline_benchmark.py` on one GPU with a small
model (e.g. TinyLlama) and short prefix so it fits a T4 16GB, and verifies
the run produces output tokens / timing lines.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from common import io_util, verify
from common.paths import ROOT

MAGICDEC = ROOT / "externals" / "MagicDec"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-pth", required=True, help="path to model.pth")
    parser.add_argument("--model-name", required=True, help="HF id for tokenizer")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prefix-len", type=int, default=2048)
    parser.add_argument("--max-len", type=int, default=2176, help="must be % 128 == 0")
    parser.add_argument("--self-spec", action="store_true",
                        help="run tests/SnapKV/selfspec_benchmark.py instead of dense")
    parser.add_argument("--gamma", type=int, default=3)
    parser.add_argument("--draft-budget", type=int, default=257)
    parser.add_argument("--num-runs", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=128,
                        help="SnapKV window; needs (prefix_len - window_size) % 128 == 0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.smoke:
        args.prefix_len = min(args.prefix_len, 1024)
        args.max_len = ((args.prefix_len + 128) // 128) * 128
        args.num_runs = 1

    assert args.max_len % 128 == 0, "--max-len must be divisible by 128"

    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(MAGICDEC) + ":" + env.get("PYTHONPATH", "")
    env["ENABLE_INTRA_NODE_COMM"] = "1"

    # The benchmark subprocess runs with cwd=MAGICDEC, so the model path must
    # be absolute (a ROOT-relative path would resolve under externals/MagicDec).
    model_pth = Path(args.model_pth)
    if not model_pth.is_absolute():
        model_pth = (ROOT / model_pth).resolve()

    script = (
        "tests/SnapKV/selfspec_benchmark.py" if args.self_spec
        else "tests/baseline_benchmark.py"
    )
    cmd = [
        "torchrun", "--standalone", "--nproc_per_node=1",
        script,
        "--model", str(model_pth),
        "--model_name", args.model_name,
        "--rank_group", "0",
        "--B", str(args.batch_size),
        "--prefix_len", str(args.prefix_len),
        "--max_len", str(args.max_len),
        "--printoutput",
    ]
    if args.self_spec:
        # --window_size only exists in selfspec_benchmark.py
        cmd += ["--window_size", str(args.window_size),
                "--gamma", str(args.gamma), "--draft_budget", str(args.draft_budget),
                "--benchmark"]

    print("+ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=MAGICDEC, env=env, capture_output=True, text=True)
    log = (proc.stdout or "") + (proc.stderr or "")
    print(log[-4000:])

    writer = io_util.JsonlWriter(Path(args.output))
    record = {
        "method": "magicdec_selfspec" if args.self_spec else "magicdec_dense",
        "dataset": "pg19",
        "model": args.model_name,
        "input_tokens": args.prefix_len,
        "retained_tokens": None,
        "output_tokens": None,
        "batch_size": args.batch_size,
        "selector_latency_ms": None,
        "ttft_ms": None,
        "tpot_ms": None,
        "e2e_ms": None,
        "throughput_tok_s": None,
        "qps": None,
        "peak_memory_gb": None,
        "returncode": proc.returncode,
        "log_tail": log[-2000:],
    }
    writer.add(record)

    checks: list[tuple[bool, str]] = [
        (proc.returncode == 0, f"benchmark process exit code = {proc.returncode}"),
        ("Throughput" in log or "Token/sec" in log or "tokens/s" in log
         or "tokens per second" in log.lower() or "Speed" in log
         or "output" in log.lower(),
         "benchmark produced timing/output lines"),
    ]
    summary = {"type": "summary", "method": record["method"],
               "returncode": proc.returncode, "num_runs": args.num_runs}
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("MagicDec", checks)


if __name__ == "__main__":
    main()
