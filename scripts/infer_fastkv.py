#!/usr/bin/env python3
"""FastKV verification / smoke script.

Loads a Llama/Mistral model with the FastKV monkey-patch
(baselines.monkeypatch.set_model) and generates a short continuation.

Two modes:
  * full  : method=fastkv + attn_implementation=flash_attention_2 (needs
            flash-attn; install via `EXTRA_FLASH=1 scripts/setup_envs.sh`).
  * smoke : method=snapkv + attn_implementation=sdpa (no flash-attn needed,
            T4-runnable). Verifies the patch path + generation end-to-end.

The record includes ``kernel_engaged`` to distinguish whether FastKV's own
kernel was actually used.
"""

from __future__ import annotations

import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

import torch

from common import io_util, rouge, verify
from common.data_loader import load_records
from common.paths import snapshot_dir


def resolve_model(model: str | None) -> str:
    if model:
        return model
    # Fall back to a cached llama-arch model if any is present.
    for repo in ["mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Meta-Llama-3.1-8B-Instruct"]:
        cached = snapshot_dir(repo)
        if cached is not None:
            return str(cached)
    raise SystemExit(
        "No model found; pass --model (HF id or local snapshot of a Llama/Mistral model)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="HF id or local snapshot (Llama/Mistral)")
    parser.add_argument("--data-file", default=None,
                        help="jsonl of records (id/prompt) to generate over; "
                             "overrides --prompt")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="limit on records from --data-file")
    parser.add_argument("--method", default="fastkv",
                        choices=["fastkv", "snapkv", "h2o", "streamingllm", "fullkv"])
    parser.add_argument("--attn-implementation", default=None,
                        choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--max-capacity-prompts", type=int, default=2048)
    parser.add_argument("--kernel-size", type=int, default=63)
    parser.add_argument("--retain-rate", type=float, default=0.1)
    parser.add_argument("--eviction-mode", default="proportional",
                        choices=["constant", "proportional"])
    parser.add_argument("--num-runs", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # flash-attn availability decides the default implementation.
    try:
        import flash_attn  # noqa: F401
        has_flash_attn = True
    except Exception:
        has_flash_attn = False

    if args.attn_implementation is None:
        args.attn_implementation = (
            "flash_attention_2" if has_flash_attn else "sdpa"
        )

    if args.smoke:
        # T4-safe path: no flash-attn build required.
        args.method = "snapkv"
        args.attn_implementation = "sdpa"
        args.max_new_tokens = min(args.max_new_tokens, 32)
        args.num_runs = 2

    if args.attn_implementation == "flash_attention_2" and not has_flash_attn:
        raise SystemExit(
            "flash-attn is not installed. Run: EXTRA_FLASH=1 scripts/setup_envs.sh, "
            "or use --attn-implementation sdpa / --smoke"
        )

    model_path = resolve_model(args.model)
    print(f"Model: {model_path} | method={args.method} attn={args.attn_implementation}")

    # Patch transformers (must happen before model construction).
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "mistral" in model_path.lower() or "ministral" in model_path.lower():
        from baselines.monkeypatch import replace_mistral as replace
        from baselines.monkeypatch import set_model
    else:
        from baselines.monkeypatch import replace_llama as replace
        from baselines.monkeypatch import set_model

    replace(args.method)

    patch_args = Namespace(
        method=args.method,
        model_path=model_path,
        max_capacity_prompts=args.max_capacity_prompts,
        window_size=args.window_size,
        kernel_size=args.kernel_size,
        pooling="avgpool",
        retain_rate=args.retain_rate,
        merge=None,  # FastKV's merge_kv is undefined in the vendored repo
        eviction_mode=args.eviction_mode,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.eval()
    set_model(model, patch_args)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device

    # Plug-and-play: one generation per record in --data-file, else --prompt.
    if args.data_file:
        prompts = load_records(args.data_file, args.max_samples)
    else:
        prompts = [{"id": "prompt", "prompt": args.prompt}]

    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []

    for si, sample in enumerate(prompts):
        input_ids = tokenizer(sample["prompt"], return_tensors="pt").input_ids.to(device)
        input_len = input_ids.shape[1]
        for run in range(args.num_runs):
            torch.manual_seed(run)
            with torch.inference_mode():
                t0 = time.perf_counter()
                out = model.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            elapsed = time.perf_counter() - t0
            output_ids = out[0, input_len:]
            n_tok = int(output_ids.shape[0])
            text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            record = {
                "method": args.method,
                "kernel_engaged": args.method == "fastkv" and args.attn_implementation == "flash_attention_2",
                "attn_implementation": args.attn_implementation,
                "dataset": "data-file" if args.data_file else "prompt",
                "model": model_path,
                "input_tokens": input_len,
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
                "run": run,
                "text": text,
            }
            # ROUGE-1/2/L vs reference summary (nếu data có reference/answer)
            rouge.add_rouge(record, text, sample.get("reference"))
            writer.add(record)
            print(f"[sample {sample['id']} run {run}] tokens={n_tok} e2e={elapsed:.2f}s {record['throughput_tok_s']:.1f} tok/s | {text[:60]!r}")

            checks.append(verify.check_new_tokens(n_tok))
            checks.append(verify.check_output_text(text))
            if n_tok:
                checks.append(verify.check_finite_logits(out[0, input_len - 1]))

    summary = {
        "type": "summary",
        "method": args.method,
        "num_runs": args.num_runs,
        "mean_tpot_ms": round(io_util.mean([r["tpot_ms"] for r in writer.records if r["tpot_ms"]]), 3),
        "mean_throughput_tok_s": round(io_util.mean([r["throughput_tok_s"] for r in writer.records]), 2),
        **rouge.aggregate_rouge(writer.records),
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("FastKV", checks)


if __name__ == "__main__":
    main()
