#!/usr/bin/env python3
"""LLMLingua verification / smoke script.

Compresses long documents with LLMLingua (PromptCompressor) and then generates
a summary with a small local target LLM (transformers), so the whole chain is
verifiable on CPU or a small GPU.

For every sample the script records (baseline_repo_guide.md §13 schema):
  origin_tokens, compressed_tokens, retained ratio, compressor latency,
  target TTFT / E2E latency, output tokens, generated summary text.

Verification checks:
  * compressor returns a non-empty compressed prompt
  * compression actually removes tokens (compressed < origin)
  * a distinctive keyword/entity from the source survives compression
  * the target model produces non-empty output
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from common import io_util, metrics, rouge, verify
from common.data_loader import load_records
from common.paths import ROOT, snapshot_dir


def _fits_model_context(input_tokens: int, max_new_tokens: int, model) -> bool:
    """Return whether a prompt plus generation fits the target context."""
    limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if not isinstance(limit, int) or limit <= 0:
        return True
    return input_tokens + max_new_tokens <= limit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compressor-model", default="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
    parser.add_argument("--use-llmlingua2", action="store_true", default=True)
    parser.add_argument("--compression-rate", type=float, default=0.5)
    parser.add_argument("--target-model", default=None,
                        help="HF id or local snapshot of the target LLM "
                             "(default: cached Qwen2.5-1.5B-Instruct)")
    parser.add_argument("--doc-file", required=True, help="jsonl of documents")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=0,
                        help="truncate each document to this many tokens before "
                             "compression (0 = no limit; use on T4 smoke runs)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true",
                        help="fast mode: fewer samples, shorter generation")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.smoke:
        args.max_samples = min(args.max_samples, 2)
        args.max_new_tokens = min(args.max_new_tokens, 32)

    # ---- compressor -------------------------------------------------------
    from llmlingua import PromptCompressor

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    compressor_model = args.compressor_model
    cached_compressor = snapshot_dir(args.compressor_model)
    if cached_compressor is not None:
        compressor_model = str(cached_compressor)
    print(f"Loading compressor: {compressor_model} (device={device})")
    compressor = PromptCompressor(
        model_name=compressor_model,
        use_llmlingua2=args.use_llmlingua2,
        device_map=device,
    )

    # ---- target model -----------------------------------------------------
    target_model = args.target_model
    if target_model is None:
        cached = snapshot_dir("Qwen/Qwen2.5-1.5B-Instruct")
        if cached is not None:
            target_model = str(cached)
    if target_model is None:
        raise SystemExit(
            "No target model found; pass --target-model (HF id or local snapshot)"
        )

    print(f"Loading target model: {target_model}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(target_model)
    model = AutoModelForCausalLM.from_pretrained(
        target_model, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
    ).to(device)
    model.eval()

    # ---- run --------------------------------------------------------------
    docs = load_records(Path(args.doc_file), args.max_samples)
    print(f"Loaded {len(docs)} document(s)")

    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []

    for i, doc in enumerate(docs):
        source = doc["prompt"]
        if args.max_input_tokens and args.max_input_tokens > 0:
            # Cap very long documents (e.g. govreport) so the target model's
            # full attention fits a T4 16GB in smoke runs.
            enc = tokenizer(
                source, truncation=True, max_length=args.max_input_tokens
            )
            source = tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
        doc_id = doc["id"]
        keyword = doc["keyword"]  # distinctive entity expected to survive

        # 1) compress
        t0 = time.perf_counter()
        result = compressor.compress_prompt(
            source,
            rate=args.compression_rate,
            force_tokens=["\n"],
        )
        compress_s = time.perf_counter() - t0

        compressed = result.get("compressed_prompt", "")
        origin_tokens = int(result.get("origin_tokens", 0))
        compressed_tokens = int(result.get("compressed_tokens", 0))

        # 2) Paired dense reference with the same target model and generation
        # settings.  The compressor is intentionally excluded from this
        # reference; the collector adds selector_latency_ms back for the
        # selector-inclusive ESR denominator.
        dense_messages = [{
            "role": "user",
            "content": "Summarize the following document.\n\n" + source,
        }]
        dense_input_ids = tokenizer.apply_chat_template(
            dense_messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        dense_input_len = dense_input_ids.shape[1]
        dense_e2e_s = None
        dense_output_tokens = None
        dense_reference_status = "measured"
        if _fits_model_context(dense_input_len, args.max_new_tokens, model):
            t0 = time.perf_counter()
            with torch.inference_mode():
                dense_out = model.generate(
                    dense_input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            dense_e2e_s = time.perf_counter() - t0
            dense_output_tokens = int(dense_out[0, dense_input_len:].shape[0])
        else:
            dense_reference_status = "skipped_context_limit"
            print(
                f"[{doc_id}] dense reference skipped: prompt={dense_input_len} "
                f"+ new_tokens={args.max_new_tokens} exceeds target context"
            )

        # 3) generate summary with the compressed target prompt
        messages = [{"role": "user", "content": "Summarize the following document.\n\n" + compressed}]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        input_len = input_ids.shape[1]

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        e2e_s = time.perf_counter() - t0

        output_ids = out[0, input_len:]
        output_tokens = int(output_ids.shape[0])
        summary_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        record = {
            "method": "llmlingua",
            "model": args.compressor_model,
            "dataset": Path(args.doc_file).name,
            "input_tokens": origin_tokens,
            "retained_tokens": compressed_tokens,
            "output_tokens": output_tokens,
            "batch_size": 1,
            "selector_latency_ms": round(compress_s * 1e3, 3),
            "ttft_ms": None,
            "tpot_ms": round(e2e_s / output_tokens * 1e3, 3) if output_tokens else None,
            "e2e_ms": round(e2e_s * 1e3, 3),
            "throughput_tok_s": round(output_tokens / e2e_s, 2) if e2e_s else 0.0,
            "qps": None,
            "peak_memory_gb": None,
            "doc_id": doc_id,
            "compression_rate": args.compression_rate,
            "dense_e2e_ms": (
                round(dense_e2e_s * 1e3, 3)
                if dense_e2e_s is not None
                else None
            ),
            "dense_output_tokens": dense_output_tokens,
            "dense_reference_status": dense_reference_status,
            "pipeline_e2e_ms": round((compress_s + e2e_s) * 1e3, 3),
            "summary": summary_text,
        }
        # ROUGE-1/2/L vs reference summary (nếu data có trường reference/answer)
        if rouge.add_rouge(record, summary_text, doc.get("reference")):
            print(
                f"  [rouge] r1={record['rouge1']:.4f} "
                f"r2={record['rouge2']:.4f} rL={record['rougeL']:.4f}"
            )
        writer.add(record)
        print(
            f"[{doc_id}] origin={origin_tokens} -> retained={compressed_tokens} "
            f"({(compressed_tokens / origin_tokens if origin_tokens else 0):.0%}) "
            f"output_tokens={output_tokens} e2e={e2e_s:.2f}s"
        )

        # verify per-sample
        checks.append(verify.check_output_text(compressed, 10))
        ok_reduced = compressed_tokens < origin_tokens
        checks.append((ok_reduced, f"compression reduced tokens ({compressed_tokens} < {origin_tokens})"))
        if keyword:
            checks.append(verify.check_retention(source, compressed, [keyword]))
        checks.append(verify.check_output_text(summary_text))

    # aggregate
    n = len(writer.records)
    summary = {
        "type": "summary",
        "method": "llmlingua",
        "num_samples": n,
        "mean_origin_tokens": round(io_util.mean([r["input_tokens"] for r in writer.records]), 1),
        "mean_retained_tokens": round(io_util.mean([r["retained_tokens"] for r in writer.records]), 1),
        "mean_retained_ratio": round(
            io_util.mean([r["retained_tokens"] / r["input_tokens"] for r in writer.records if r["input_tokens"]]), 4),
        "mean_selector_latency_ms": round(io_util.mean([r["selector_latency_ms"] for r in writer.records]), 3),
        "mean_e2e_ms": round(io_util.mean([r["e2e_ms"] for r in writer.records]), 3),
        "speedup": metrics.aggregate_speedup(writer.records),
        **rouge.aggregate_rouge(writer.records),
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved records + summary to: {args.output}")

    verify.finish("LLMLingua", checks)


if __name__ == "__main__":
    main()
