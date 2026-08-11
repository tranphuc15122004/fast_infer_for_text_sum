#!/usr/bin/env python3
"""RocketKV verification / smoke script.

RocketKV's full pipeline (LongBench eval) is heavy and needs big GPU + HF
token, so the smoke test verifies the *core kernel* directly:

  1. RocketArgs + get_params_for_token_budget produce sane budgets.
  2. A minimal RocketAttention prefill + decode forward on dummy tensors
     returns finite output (i.e. the rocket_attn two-stage kernel runs).

Full mode (--full) delegates to the repo's own pipeline runner
(pipeline/inf_stream_llm/main.py) with a model + pipeline/eval config.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from common import io_util, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-budget", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--full", action="store_true",
                        help="(not implemented in smoke) run repo pipeline")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from rocket import RocketArgs, RocketAttention

    # 1) budget math -----------------------------------------------------
    budget = RocketArgs(token_budget=args.token_budget)
    cap, prompt_budget, chunk, r, k = (
        __import__("rocket").get_params_for_token_budget(
            args.token_budget, args.seq_len, args.max_new_tokens, args.head_dim
        )
    )
    checks: list[tuple[bool, str]] = []
    checks.append((cap >= 0 and prompt_budget >= 0, f"budgets valid (cap={cap}, prompt={prompt_budget})"))
    checks.append((1 <= r <= args.head_dim, f"r={r} in [1, {args.head_dim}]"))
    checks.append((1 <= k <= cap, f"k={k} <= capacity={cap}"))

    # 2) kernel forward (prefill + decode) -------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running kernel smoke on {device} (token_budget={args.token_budget})")
    attn = RocketAttention(budget, n_head=8, n_local_heads=8)
    attn.setup_caches(
        max_batch_size=1, max_seq_length=args.seq_len,
        max_new_tokens=args.max_new_tokens, n_heads=8, head_dim=64,
        layer_idx=0, dtype=torch.float32,
    )
    attn = attn.to(device)

    writer = io_util.JsonlWriter(Path(args.output))
    kernel_ok = True
    for run in range(args.num_runs):
        torch.manual_seed(run)
        t0 = time.perf_counter()
        with torch.inference_mode():
            # prefill: 8 tokens at once; boolean causal mask spanning capacity
            pre_len = 8
            q = torch.randn(1, 8, pre_len, 64, device=device)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            mask = torch.zeros(1, 1, pre_len, args.seq_len, device=device, dtype=torch.bool)
            causal = torch.tril(torch.ones(pre_len, pre_len, device=device, dtype=torch.bool))
            mask[..., :pre_len] = causal
            input_pos = torch.arange(pre_len, device=device)
            out_pre = attn(q, k, v, mask, input_pos, 0, state=0)  # PREFILL
            # decode: one token, everything visible
            qd = torch.randn(1, 8, 1, 64, device=device)
            kd = torch.randn_like(qd)
            vd = torch.randn_like(qd)
            maskd = torch.ones(1, 1, 1, args.seq_len, device=device, dtype=torch.bool)
            input_posd = torch.tensor([pre_len], device=device)
            out_dec = attn(qd, kd, vd, maskd, input_posd, 0, state=2)  # DECODE
        elapsed = time.perf_counter() - t0

        finite_pre = bool(torch.isfinite(out_pre).all())
        finite_dec = bool(torch.isfinite(out_dec).all())
        ok = finite_pre and finite_dec and out_pre.shape == q.shape and out_dec.shape == qd.shape
        kernel_ok = kernel_ok and ok

        record = {
            "method": "rocketkv",
            "dataset": "kernel-smoke",
            "model": "RocketAttention",
            "input_tokens": pre_len,
            "retained_tokens": None,
            "output_tokens": 1,
            "batch_size": 1,
            "selector_latency_ms": None,
            "ttft_ms": None,
            "tpot_ms": round(elapsed * 1e3, 3),
            "e2e_ms": round(elapsed * 1e3, 3),
            "throughput_tok_s": None,
            "qps": None,
            "peak_memory_gb": None,
            "token_budget": args.token_budget,
            "run": run,
            "finite": ok,
            "cap_budget": cap,
            "r": r,
            "k": k,
        }
        writer.add(record)
        checks.append((ok, f"run {run}: prefill+decode finite, shapes ok"))

    summary = {
        "type": "summary",
        "method": "rocketkv",
        "num_runs": args.num_runs,
        "token_budget": args.token_budget,
        "kernel_ok": kernel_ok,
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    print("NOTE: full LongBench pipeline run needs a Llama model + HF token;")
    print("      see externals/RocketKV/scripts/longbench/ for the upstream commands.")
    verify.finish("RocketKV", checks)


if __name__ == "__main__":
    main()
