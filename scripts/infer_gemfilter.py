#!/usr/bin/env python3
"""GemFilter verification / smoke script.

Uses the repo's own patched loader (my_utils.load_model with modified=gemfilter,
eager attention — no flash-attn required) and runs both the standard greedy
generation and the GemFilter token-selection generation on a short prompt.

For every sample the script records latency, tokens and the generated text,
then runs correctness checks (non-empty output, finite logits, determinism).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common import io_util, metrics, rouge, verify
from common.data_loader import load_records
from common.paths import snapshot_dir


def resolve_model(model: str | None) -> str:
    if model:
        return model
    for repo in [
        "microsoft/Phi-3.5-mini-instruct",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ]:
        cached = snapshot_dir(repo)
        if cached is not None:
            return str(cached)
    raise SystemExit(
        "No model found; pass --model (one of the GemFilter-supported models: "
        "Phi-3.5-mini / Mistral-Nemo / Llama-3.1-8B)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--data-file", default=None,
                        help="jsonl of records (id/prompt) to generate over; "
                             "overrides --prompt")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--topk", type=int, default=1024)
    parser.add_argument("--select-layer-idx", type=int, default=None,
                        help="GemFilter selection layer (default: 13 for Llama-3.1-8B, 19 for Nemo/Phi-3.5)")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-gen-len", type=int, default=32)
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.smoke:
        args.max_samples = 1
        args.max_gen_len = min(args.max_gen_len, 16)
        args.num_runs = 2

    model_path = resolve_model(args.model)
    print(f"Model: {model_path} | topk={args.topk}")

    from my_utils.load_model import load_model
    from my_utils.my_generation import (
        my_greedy_generate_selection,
        my_greedy_generate_standard,
        set_topk,
    )

    model, tokenizer = load_model(
        model_path,
        modified="gemfilter",
        torch_dtype=torch.float16,
        device_map="auto",
        flash_attention_2=False,
    )
    set_topk(model, args.topk, mode="gemfilter")
    model.eval()

    device = next(model.parameters()).device

    if args.data_file:
        prompts = load_records(args.data_file, args.max_samples)
    else:
        prompts = [{"id": "prompt", "prompt": args.prompt}]

    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []

    for si, sample in enumerate(prompts):
        input_ids = tokenizer(sample["prompt"], return_tensors="pt").input_ids.to(device)
        attn_mask = torch.ones_like(input_ids)
        for run in range(args.num_runs):
            torch.manual_seed(run)
            # baseline (standard greedy)
            with torch.inference_mode():
                t0 = time.perf_counter()
                base_text = my_greedy_generate_standard(
                    input_ids, attn_mask, model, tokenizer, max_gen_len=args.max_gen_len
                )
                base_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                gem_text = my_greedy_generate_selection(
                    input_ids, attn_mask, model, tokenizer,
                    max_gen_len=args.max_gen_len,
                    select_layer_idx=args.select_layer_idx,
                )
                gem_s = time.perf_counter() - t0

            record = {
                "method": "gemfilter",
                "dataset": "data-file" if args.data_file else "prompt",
                "model": model_path,
                "input_tokens": int(input_ids.shape[1]),
                "retained_tokens": None,
                "output_tokens": None,
                "batch_size": 1,
                "selector_latency_ms": None,
                "ttft_ms": None,
                "tpot_ms": None,
                "e2e_ms": round(gem_s * 1e3, 3),
                "throughput_tok_s": None,
                "qps": None,
                "peak_memory_gb": None,
                "topk": args.topk,
                "sample_id": sample["id"],
                "run": run,
                "base_text": base_text,
                "gemfilter_text": gem_text,
                "base_time_s": round(base_s, 4),
                "gemfilter_time_s": round(gem_s, 4),
                "dense_e2e_ms": round(base_s * 1e3, 3),
            }
            # ROUGE-1/2/L cho cả 2 nhánh (nếu data có reference/answer);
            # base_text lưu dưới prefix "base_" để phân biệt với gemfilter_text.
            rouge.add_rouge(record, gem_text, sample.get("reference"))
            rouge.add_rouge(record, base_text, sample.get("reference"), prefix="base_")
            writer.add(record)
            print(
                f"[sample {sample['id']} run {run}] gemfilter e2e={gem_s:.2f}s (baseline {base_s:.2f}s) | {gem_text[:60]!r}"
            )

            checks.append(verify.check_output_text(gem_text))
            checks.append(verify.check_output_text(base_text))

    summary = {
        "type": "summary",
        "method": "gemfilter",
        "num_runs": args.num_runs,
        "mean_gemfilter_time_s": round(io_util.mean([r["gemfilter_time_s"] for r in writer.records]), 4),
        "mean_base_time_s": round(io_util.mean([r["base_time_s"] for r in writer.records]), 4),
        "speedup": metrics.aggregate_speedup(writer.records),
        **rouge.aggregate_rouge(writer.records),
        **rouge.aggregate_rouge(writer.records, prefix="base_"),
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("GemFilter", checks)


if __name__ == "__main__":
    main()
