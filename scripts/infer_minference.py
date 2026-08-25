#!/usr/bin/env python3
"""MInference verification / smoke script.

Patches a HF model with MInference (dynamic sparse attention) and runs a
short generation. MInference targets long-context prefill; a short-context
run here only verifies the patch path + kernels produce a correct output.

Supported models are listed by ``minference.get_support_models()``; on T4 use
the smallest supported one (e.g. Qwen2.5-7B-Instruct with a short context).
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import torch

from common import io_util, rouge, verify
from common.data_loader import load_records


def _select_attention_implementation(requested: str) -> str:
    """Select a Transformers attention backend that works on this GPU.

    FlashAttention-2 is not available on the T4 setup used by this repo.
    MInference replaces the attention forward path after model loading, so
    loading with SDPA/eager is a valid fallback for the patch path.
    """
    if requested != "auto":
        return requested

    has_flash_attn = importlib.util.find_spec("flash_attn") is not None
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if has_flash_attn and major >= 8:
            return "flash_attention_2"
    return "sdpa"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="supported HF model id or local snapshot")
    parser.add_argument("--attn-type", default="minference",
                        choices=["minference", "vertical_slash", "block_sparse", "streaming"])
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--data-file", default=None,
                        help="jsonl of records (id/prompt) to generate over")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-input-tokens", type=int, default=0,
                        help="truncate each prompt to this many tokens before "
                             "generation (0 = no limit; use on T4 smoke runs)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="auto",
                        choices=["auto", "flash_attention_2", "sdpa", "eager"],
                        help="Transformers backend used while loading; auto "
                             "falls back to SDPA on T4/sm75")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.smoke:
        args.max_new_tokens = min(args.max_new_tokens, 16)
        args.max_model_len = min(args.max_model_len, 4096)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from minference import MInference, get_support_models

    supported = set(get_support_models())
    print(f"Model: {args.model} (in support list: {args.model in supported})")

    attn_implementation = _select_attention_implementation(
        args.attn_implementation
    )
    print(f"Transformers attention backend: {attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        _attn_implementation=attn_implementation,
        max_position_embeddings=args.max_model_len,
    )
    model.eval()

    patch = MInference(args.attn_type, args.model)
    model = patch(model)

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(model.device)
    input_len = input_ids.shape[1]

    if args.data_file:
        prompts = load_records(args.data_file, args.max_samples)
    else:
        prompts = [{"id": "prompt", "prompt": args.prompt}]

    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []

    for sample in prompts:
        encoded = tokenizer(sample["prompt"], return_tensors="pt")
        ids = encoded.input_ids.to(model.device)
        attention_mask = encoded.attention_mask.to(model.device)
        if args.max_input_tokens and args.max_input_tokens > 0 \
                and ids.shape[1] > args.max_input_tokens:
            ids = ids[:, : args.max_input_tokens]
            attention_mask = attention_mask[:, : args.max_input_tokens]
        ilen = ids.shape[1]
        with torch.inference_mode():
            t0 = time.perf_counter()
            out = model.generate(
                input_ids=ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            elapsed = time.perf_counter() - t0

        output_ids = out[0, ilen:]
        n_tok = int(output_ids.shape[0])
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        record = {
            "method": f"minference_{args.attn_type}",
            "dataset": "data-file" if args.data_file else "prompt",
            "model": args.model,
            "input_tokens": ilen,
            "retained_tokens": None,
            "output_tokens": n_tok,
            "batch_size": 1,
            "selector_latency_ms": None,
            "ttft_ms": None,
            "tpot_ms": round(elapsed / n_tok * 1e3, 3) if n_tok else None,
            "e2e_ms": round(elapsed * 1e3, 3),
            "throughput_tok_s": round(n_tok / elapsed, 2) if elapsed else 0.0,
            "qps": None,
            "peak_memory_gb": None,
            "sample_id": sample["id"],
            "text": text,
        }
        # ROUGE-1/2/L vs reference summary (nếu data có reference/answer)
        rouge.add_rouge(record, text, sample.get("reference"))
        writer.add(record)
        print(f"[sample {sample['id']}] tokens={n_tok} e2e={elapsed:.2f}s | {text[:60]!r}")

        checks.append(verify.check_new_tokens(n_tok))
        checks.append(verify.check_output_text(text))
        if n_tok:
            checks.append(verify.check_finite_logits(out[0, ilen - 1]))

    summary = {
        "type": "summary",
        "method": record["method"],
        "num_samples": len(prompts),
        "mean_e2e_ms": round(io_util.mean([r["e2e_ms"] for r in writer.records]), 3),
        "mean_throughput_tok_s": round(io_util.mean([r["throughput_tok_s"] for r in writer.records]), 2),
        **rouge.aggregate_rouge(writer.records),
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("MInference", checks)


if __name__ == "__main__":
    main()
