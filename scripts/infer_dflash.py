#!/usr/bin/env python3
"""DFlash representative benchmark adapter.

Runs the paired target + DFlash draft checkpoints on unified JSONL prompts and
records both DFlash and target-only timings.  ``block_size=1`` is the paired
autoregressive reference used for the speedup fields.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common import io_util, metrics, rouge, verify
from common.data_loader import load_records


def _dtype_and_attention() -> tuple[torch.dtype, str]:
    if not torch.cuda.is_available():
        raise SystemExit("DFlash Transformers adapter requires CUDA")
    capability = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return dtype, "sdpa"
    return dtype, "flash_attention_2" if capability[0] >= 8 else "sdpa"


def _chat_prompt(tokenizer, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def _run_generation(dflash_generate, draft, target, input_ids, *, max_new_tokens,
                    temperature, block_size):
    torch.cuda.synchronize()
    start = time.perf_counter()
    eos_id = target.config.eos_token_id
    stop_token_ids = eos_id if isinstance(eos_id, list) else [eos_id]
    result = dflash_generate(
        draft,
        target=target,
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        block_size=block_size,
        return_stats=True,
    )
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


def _timings(result, elapsed_s: float) -> tuple[float, float, float]:
    e2e_ms = elapsed_s * 1e3
    prefill_ms = float(result.time_to_first_token) * 1e3
    decode_ms = max(e2e_ms - prefill_ms, 0.0)
    return prefill_ms, decode_ms, e2e_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=0,
                        help="truncate each prompt to this many tokens before "
                             "generation (0 = no limit; use on T4 smoke runs)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.smoke:
        args.max_new_tokens = min(args.max_new_tokens, 32)

    dtype, attn_impl = _dtype_and_attention()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel, dflash_generate

    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        dtype=dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        dtype=dtype,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.data_file:
        prompts = load_records(Path(args.data_file), args.max_samples)
    else:
        prompts = [{"id": "prompt", "prompt": args.prompt, "reference": None}]

    block_size = args.block_size or int(draft.block_size)
    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []

    for sample in prompts:
        prompt = _chat_prompt(tokenizer, sample["prompt"])
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(device)
        if args.max_input_tokens and args.max_input_tokens > 0 \
                and input_ids.shape[1] > args.max_input_tokens:
            input_ids = input_ids[:, : args.max_input_tokens]
        input_len = int(input_ids.shape[1])

        torch.manual_seed(0)
        baseline, baseline_elapsed = _run_generation(
            dflash_generate, draft, target, input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            block_size=1,
        )
        torch.manual_seed(0)
        result, elapsed = _run_generation(
            dflash_generate, draft, target, input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            block_size=block_size,
        )

        output_ids = result.output_ids[0, input_len:]
        baseline_ids = baseline.output_ids[0, input_len:]
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        baseline_text = tokenizer.decode(
            baseline_ids, skip_special_tokens=True
        ).strip()
        n_tok = int(output_ids.shape[0])
        baseline_n_tok = int(baseline_ids.shape[0])
        prefill_ms, decode_ms, e2e_ms = _timings(result, elapsed)
        base_prefill_ms, base_decode_ms, base_e2e_ms = _timings(
            baseline, baseline_elapsed
        )

        record = {
            "method": "dflash",
            "dataset": "data-file" if args.data_file else "prompt",
            "model": args.target_model,
            "draft_model": args.draft_model,
            "input_tokens": input_len,
            "retained_tokens": None,
            "output_tokens": n_tok,
            "baseline_output_tokens": baseline_n_tok,
            "batch_size": 1,
            "selector_latency_ms": None,
            "ttft_ms": round(prefill_ms, 3),
            "prefill_ms": round(prefill_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "e2e_ms": round(e2e_ms, 3),
            "baseline_ttft_ms": round(base_prefill_ms, 3),
            "baseline_prefill_ms": round(base_prefill_ms, 3),
            "baseline_decode_ms": round(base_decode_ms, 3),
            "baseline_e2e_ms": round(base_e2e_ms, 3),
            "dense_prefill_ms": round(base_prefill_ms, 3),
            "dense_decode_ms": round(base_decode_ms, 3),
            "dense_e2e_ms": round(base_e2e_ms, 3),
            "tpot_ms": round(decode_ms / n_tok, 3) if n_tok else None,
            "throughput_tok_s": round(n_tok / (decode_ms / 1e3), 2)
            if decode_ms > 0 and n_tok else 0.0,
            "qps": None,
            "peak_memory_gb": None,
            "sample_id": sample["id"],
            "text": text,
            "baseline_text": baseline_text,
            "block_size": block_size,
            "acceptance_lengths": list(result.acceptance_lengths),
        }
        rouge.add_rouge(record, text, sample.get("reference"))
        writer.add(record)
        print(
            f"[sample {sample['id']}] dflash={e2e_ms:.1f}ms "
            f"baseline={base_e2e_ms:.1f}ms tokens={n_tok}"
        )
        checks.append(verify.check_new_tokens(n_tok))
        checks.append(verify.check_output_text(text))

    summary = {
        "type": "summary",
        "method": "dflash",
        "num_samples": len(prompts),
        "block_size": block_size,
        "speedup": metrics.aggregate_speedup(writer.records),
        **rouge.aggregate_rouge(writer.records),
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("DFlash", checks)


if __name__ == "__main__":
    main()
