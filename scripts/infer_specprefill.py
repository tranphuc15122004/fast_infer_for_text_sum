#!/usr/bin/env python3
"""Speculative Prefill (vLLM patch) verification / smoke script.

Applies speculative_prefill's monkey-patch to vLLM, then runs a short
generation with a small Llama target + spec (draft) model and reports tokens,
latency and whether speculative prefill engaged.

NOTE: this repo supports Llama-family models only. On T4 16GB use a small
target (e.g. Llama-3.2-3B-Instruct) + Llama-3.2-1B-Instruct draft.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from common import io_util, verify
from common.data_loader import load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-model", required=True, help="draft/speculator model (Llama)")
    parser.add_argument("--target-model", required=True, help="target model (Llama)")
    parser.add_argument("--spec-config", default="configs/config_p1_full_lah8.yaml")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--data-file", default=None,
                        help="jsonl of records (id/prompt); generates over all "
                             "prompts in one vLLM batch")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.smoke:
        args.max_tokens = min(args.max_tokens, 32)

    # patch MUST happen before importing vllm
    from speculative_prefill import enable_prefill_spec

    enable_prefill_spec(
        spec_model=args.spec_model,
        spec_config_path=args.spec_config,
    )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.target_model,
        tokenizer=args.target_model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="half",  # bf16 needs sm80+; T4 (sm75) must use float16
        enforce_eager=True,
        enable_chunked_prefill=False,
        tensor_parallel_size=1,
    )
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)

    if args.data_file:
        prompts = load_records(args.data_file, args.max_samples)
    else:
        prompts = [{"id": "prompt", "prompt": args.prompt}]

    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []

    t0 = time.perf_counter()
    outputs = llm.generate([p["prompt"] for p in prompts], params)
    elapsed = time.perf_counter() - t0

    for p, out in zip(prompts, outputs):
        text = out.outputs[0].text
        n_tok = len(out.outputs[0].token_ids)
        record = {
            "method": "speculative_prefill",
            "dataset": "data-file" if args.data_file else "prompt",
            "model": args.target_model,
            "input_tokens": len(out.prompt_token_ids),
            "retained_tokens": None,
            "output_tokens": n_tok,
            "batch_size": len(prompts),
            "selector_latency_ms": None,
            "ttft_ms": None,
            "tpot_ms": round(elapsed / n_tok * 1e3, 3) if n_tok else None,
            "e2e_ms": round(elapsed * 1e3, 3),
            "throughput_tok_s": round(n_tok / elapsed, 2) if elapsed else 0.0,
            "qps": None,
            "peak_memory_gb": None,
            "spec_model": args.spec_model,
            "sample_id": p["id"],
            "text": text,
        }
        writer.add(record)
        print(f"[sample {p['id']}] tokens={n_tok} | {text[:60]!r}")

        checks.append(verify.check_new_tokens(n_tok))
        checks.append(verify.check_output_text(text))

    summary = {
        "type": "summary",
        "method": "speculative_prefill",
        "num_samples": len(prompts),
        "batch_e2e_ms": round(elapsed * 1e3, 3),
        "throughput_tok_s": round(
            sum(r["output_tokens"] for r in writer.records) / elapsed, 2
        ) if elapsed else 0.0,
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("SpeculativePrefill", checks)


if __name__ == "__main__":
    main()
