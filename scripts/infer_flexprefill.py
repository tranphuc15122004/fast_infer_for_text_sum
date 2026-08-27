#!/usr/bin/env python3
"""FlexPrefill verification / benchmark script.

Patches a HF model with FlexPrefill sparse attention
(``flex_prefill.patch_model``, ICLR 2025 Oral) and generates summaries over a
JSONL data file, comparing against the dense (unpatched) baseline on the same
model for paired speedup metrics.

Smoke-safe on T4 16GB:
  * uses a small cached qwen2-arch model (Qwen2.5-3B-Instruct) in bf16;
  * does not require flash-attn — disable_hf_flash_attention_check()
    monkey-patches HF so ``_attn_implementation="flash_attention_2"`` passes
    without the kernel being installed (the patch replaces attention anyway);
  * ``--max-input-tokens`` caps long documents (e.g. govreport) before the
    KV cache / prefill.

Records follow the baseline_repo_guide.md §13 schema.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common import io_util, metrics, rouge, verify
from common.data_loader import load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="HF id or local snapshot (qwen2/llama/glm arch)")
    parser.add_argument("--pattern", default="flex_prefill",
                        choices=["flex_prefill", "streaming_llm",
                                 "vertical_slash", "minfer", "default", "flash"])
    parser.add_argument("--data-file", default=None,
                        help="jsonl of records (id/document or prompt)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=0,
                        help="truncate each prompt to this many tokens before "
                             "generation (0 = no limit; use on T4 smoke runs)")
    parser.add_argument("--skip-naive", action="store_true",
                        help="skip the dense baseline (disables paired speedup)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.smoke:
        args.max_samples = 1
        args.max_new_tokens = min(args.max_new_tokens, 32)

    if not torch.cuda.is_available():
        raise RuntimeError("FlexPrefill requires a visible CUDA GPU (triton kernels)")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from flex_prefill import (
        disable_hf_flash_attention_check,
        get_config_example,
        patch_model,
    )

    disable_hf_flash_attention_check()
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # SDPA works without flash-attn (T4/sm75); FlexPrefill replaces the
        # attention modules after loading anyway.
        _attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.data_file:
        prompts = load_records(Path(args.data_file), args.max_samples)
    else:
        prompts = [{"id": "prompt", "prompt": args.prompt}]
    print(f"Loaded {len(prompts)} record(s)")

    def _build_input(prompt: str) -> torch.Tensor:
        try:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=False,
            ).to("cuda")
        except TypeError:
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        if args.max_input_tokens and args.max_input_tokens > 0 \
                and ids.shape[1] > args.max_input_tokens:
            ids = ids[:, : args.max_input_tokens]
        return ids

    def _generate(ids: torch.Tensor) -> tuple[torch.Tensor, float]:
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        return out, time.perf_counter() - t0

    warmup_ids = _build_input("Hello")
    with torch.inference_mode():
        model.forward(warmup_ids, use_cache=False)

    input_ids_list = [_build_input(s["prompt"]) for s in prompts]

    # ---- 1) dense baseline (model not patched yet) -----------------------
    dense_outs: list[tuple[torch.Tensor, float]] = []
    if not args.skip_naive:
        print("Running dense baseline ...")
        for i, (s, ids) in enumerate(zip(prompts, input_ids_list)):
            out, elapsed = _generate(ids)
            dense_outs.append((out, elapsed))
            print(f"  [dense {s['id']}] {elapsed * 1e3:.1f}ms")
        torch.cuda.empty_cache()

    # ---- 2) patch + method ------------------------------------------------
    attention_config = get_config_example(args.pattern)
    patch_model(model, args.pattern, attention_config)
    print(f"Patched model with pattern: {args.pattern} "
          f"cfg={attention_config}")
    with torch.inference_mode():
        model.forward(warmup_ids, use_cache=False)
    torch.cuda.empty_cache()

    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []

    for i, (sample, ids) in enumerate(zip(prompts, input_ids_list)):
        input_len = ids.shape[1]
        out, elapsed = _generate(ids)
        output_ids = out[0, input_len:]
        n_tok = int(output_ids.shape[0])
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        record = {
            "method": f"flexprefill_{args.pattern}",
            "dataset": "data-file" if args.data_file else "prompt",
            "model": args.model,
            "input_tokens": input_len,
            "retained_tokens": None,
            "output_tokens": n_tok,
            "batch_size": 1,
            "selector_latency_ms": None,
            "ttft_ms": None,
            "tpot_ms": None,
            "prefill_ms": None,
            "decode_ms": None,
            "e2e_ms": round(elapsed * 1e3, 3),
            "pipeline_e2e_ms": round(elapsed * 1e3, 3),
            "throughput_tok_s": round(n_tok / elapsed, 2) if elapsed > 0 else 0.0,
            "qps": None,
            "peak_memory_gb": None,
            "sample_id": sample["id"],
            "text": text,
        }
        if not args.skip_naive and i < len(dense_outs):
            d_out, d_elapsed = dense_outs[i]
            d_out_ids = d_out[0, input_len:]
            d_n_tok = int(d_out_ids.shape[0])
            d_text = tokenizer.decode(d_out_ids, skip_special_tokens=True).strip()
            record["dense_e2e_ms"] = round(d_elapsed * 1e3, 3)
            record["dense_output_tokens"] = d_n_tok
            record["dense_text"] = d_text

        rouge.add_rouge(record, text, sample.get("reference"))
        writer.add(record)
        checks.append(verify.check_new_tokens(n_tok))
        checks.append(verify.check_output_text(text))
        print(f"[sample {sample['id']}] flexprefill={elapsed * 1e3:.1f}ms "
              f"tokens={n_tok}")

    summary = {
        "type": "summary",
        "method": f"flexprefill_{args.pattern}",
        "num_samples": len(prompts),
        "speedup": metrics.aggregate_speedup(writer.records),
        **rouge.aggregate_rouge(writer.records),
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("FlexPrefill", checks)


if __name__ == "__main__":
    main()
