#!/usr/bin/env python3
"""LongSpec verification script.

Full LongSpec inference needs an 80GB-class GPU (llama8b/longchat targets +
dedicated draft). The smoke mode therefore verifies the environment and code
path instead: every module imports (llama_glide / qwen2_glide /
triton_tree_attn / liger_kernel), and the TreeAttention triton kernel runs a
tiny dummy forward.

Full mode (--full) delegates to the repo's inference_long-bench.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import torch

from common import io_util, metrics, rouge, verify
from common.data_loader import load_records
from common.paths import ROOT

LONGSPEC = ROOT / "externals" / "LongSpec" / "longspec" / "test"
# LongSpec modules (llama_glide / qwen2_glide / triton_tree_attn) live in the
# repo's test dir and are imported as top-level modules.
sys.path.insert(0, str(LONGSPEC))

MODEL_NAMES = {
    "vicuna7b": {
        "target": "lmsys/vicuna-7b-v1.5-16k",
        "draft": "sail/longspec-vicuna-7b-v1.5-16k",
    },
}


def _representative_prompt(document: str) -> str:
    return (
        "<s>system\nYou are a helpful assistant</s>\n"
        "<s>user\nSummarize the following document in a concise paragraph.\n\n"
        f"Document:\n{document}\n\n"
        "Now, write the summary.</s>\n<s>assistant\nSummary:"
    )


def _trim_generated(ids, eos_id: int, limit: int | None = None) -> list[int]:
    values = ids[0].tolist()
    if limit is not None:
        values = values[:limit]
    try:
        end = values.index(eos_id)
    except ValueError:
        return values
    return values[: end + 1]


def _run_representative(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("LongSpec representative inference requires CUDA")

    from transformers import AutoConfig, AutoTokenizer
    from llama_glide import LlamaGlide

    pair = MODEL_NAMES.get(args.model_name)
    target_model = args.target_model or (pair or {}).get("target")
    draft_model = args.draft_model or (pair or {}).get("draft")
    if not target_model or not draft_model:
        raise SystemExit(
            f"No LongSpec target/draft pair configured for {args.model_name}; "
            "pass --target-model and --draft-model"
        )

    config = AutoConfig.from_pretrained(target_model)
    tokenizer = AutoTokenizer.from_pretrained(target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    model = LlamaGlide(config, target_model, draft_model)
    device = next(model.model.parameters()).device
    eos_id = int(tokenizer.eos_token_id)

    records = load_records(Path(args.data_file), args.max_samples)
    context_limit = int(getattr(config, "max_position_embeddings", 16384))
    max_input_tokens = args.max_input_tokens
    if max_input_tokens is None:
        max_input_tokens = max(context_limit - args.max_gen_len - 1, 1)

    writer = io_util.JsonlWriter(Path(args.output))
    checks: list[tuple[bool, str]] = []
    for sample in records:
        prompt = _representative_prompt(sample["prompt"])
        original_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        input_ids = encoded.input_ids.to(device)
        input_len = int(input_ids.shape[1])
        prompt_length = input_ids.new_tensor([input_len])

        baseline_start = time.perf_counter()
        baseline_ids, baseline_num, baseline_decode_s = model.vanilla_torch_generate(
            input_ids, prompt_length, max_gen_len=args.max_gen_len
        )
        baseline_wall_s = time.perf_counter() - baseline_start

        tree_start = time.perf_counter()
        output_ids, accepted, verified, tree_decode_s, _ = model.tree_spec_generate(
            input_ids,
            prompt_length,
            tree_shape=args.tree_shape,
            max_gen_len=args.max_gen_len,
            eos_id=eos_id,
            temperature=args.temperature,
        )
        tree_wall_s = time.perf_counter() - tree_start

        baseline_token_ids = _trim_generated(
            baseline_ids, eos_id, limit=max(1, int(baseline_num) + 1)
        )
        token_ids = _trim_generated(output_ids, eos_id)
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        baseline_text = tokenizer.decode(
            baseline_token_ids, skip_special_tokens=True
        ).strip()
        n_tok = len(token_ids)
        baseline_n_tok = len(baseline_token_ids)
        baseline_decode_ms = baseline_decode_s * 1e3
        tree_decode_ms = tree_decode_s * 1e3
        baseline_e2e_ms = baseline_wall_s * 1e3
        tree_e2e_ms = tree_wall_s * 1e3
        record = {
            "method": "longspec",
            "dataset": "data-file",
            "model": target_model,
            "draft_model": draft_model,
            "model_name": args.model_name,
            "input_tokens": input_len,
            "original_input_tokens": len(original_ids),
            "input_truncated": len(original_ids) != input_len,
            "retained_tokens": None,
            "output_tokens": n_tok,
            "baseline_output_tokens": baseline_n_tok,
            "batch_size": 1,
            "selector_latency_ms": None,
            "ttft_ms": round(max(tree_e2e_ms - tree_decode_ms, 0.0), 3),
            "prefill_ms": round(max(tree_e2e_ms - tree_decode_ms, 0.0), 3),
            "decode_ms": round(tree_decode_ms, 3),
            "e2e_ms": round(tree_e2e_ms, 3),
            "baseline_ttft_ms": round(max(baseline_e2e_ms - baseline_decode_ms, 0.0), 3),
            "baseline_prefill_ms": round(max(baseline_e2e_ms - baseline_decode_ms, 0.0), 3),
            "baseline_decode_ms": round(baseline_decode_ms, 3),
            "baseline_e2e_ms": round(baseline_e2e_ms, 3),
            "dense_prefill_ms": round(max(baseline_e2e_ms - baseline_decode_ms, 0.0), 3),
            "dense_decode_ms": round(baseline_decode_ms, 3),
            "dense_e2e_ms": round(baseline_e2e_ms, 3),
            "tpot_ms": round(tree_decode_ms / n_tok, 3) if n_tok else None,
            "throughput_tok_s": round(n_tok / (tree_decode_s), 2)
            if tree_decode_s > 0 and n_tok else 0.0,
            "qps": None,
            "peak_memory_gb": None,
            "sample_id": sample["id"],
            "text": text,
            "baseline_text": baseline_text,
            "accepted_tokens": int(accepted),
            "verified_tokens": int(verified),
            "avg_accept_length": round(float(accepted) / max(int(verified), 1), 4),
        }
        rouge.add_rouge(record, text, sample.get("reference"))
        writer.add(record)
        print(
            f"[sample {sample['id']}] longspec={tree_e2e_ms:.1f}ms "
            f"baseline={baseline_e2e_ms:.1f}ms tokens={n_tok}"
        )
        checks.append(verify.check_new_tokens(n_tok))
        checks.append(verify.check_output_text(text))

    summary = {
        "type": "summary",
        "method": "longspec",
        "num_samples": len(records),
        "model_name": args.model_name,
        "speedup": metrics.aggregate_speedup(writer.records),
        **rouge.aggregate_rouge(writer.records),
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("LongSpec", checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="run repo inference_long-bench.py (needs 80GB GPU)")
    parser.add_argument("--model-name", default="llama8b")
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--draft-model", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--method", default="tree")
    parser.add_argument("--task", default="gov_report")
    parser.add_argument("--data-path-prefix", default=None,
                        help="dir with preprocessed longbench jsonl (full mode)")
    parser.add_argument("--tree-shape", default="4 16 16 16 16")
    parser.add_argument("--max-gen-len", type=int, default=1024)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.smoke:
        args.max_gen_len = min(args.max_gen_len, 32)

    if args.data_file:
        _run_representative(args)
        return

    checks: list[tuple[bool, str]] = []
    writer = io_util.JsonlWriter(Path(args.output))

    if not args.full:
        # ---- smoke: imports + tiny kernel forward -------------------------
        try:
            import llama_glide  # noqa: F401
            import qwen2_glide  # noqa: F401
            from triton_tree_attn import attention as tree_attention  # noqa: F401
            import liger_kernel  # noqa: F401
            import_ok = True
            msg = "imports ok (llama_glide, qwen2_glide, triton_tree_attn, liger_kernel)"
        except Exception as e:
            import_ok = False
            msg = f"import failed: {e}"
        checks.append((import_ok, msg))

        # Kernel forward: the triton TreeAttention kernel compiles only on
        # sm80+ (it fails in triton's LLVM pass on sm75/T4). Report SKIP on
        # sm<80 (validated on the big-GPU server) instead of a hard FAIL.
        kernel_ok: bool | None = False
        try:
            from triton_tree_attn import attention as tree_attention

            device = "cuda" if torch.cuda.is_available() else "cpu"
            cap = torch.cuda.get_device_capability() if device == "cuda" else None
            if cap is not None and cap[0] < 8:
                print(f"kernel smoke SKIPPED: triton TreeAttention needs sm80+ "
                      f"(this GPU is sm{cap[0]}{cap[1]})")
                kernel_ok = None
            else:
                n, h, t, d = 1, 8, 8, 64
                N = 24  # realistic tree attention: M (draft) < N (full cache)
                q = torch.randn(n, h, t, d, device=device)
                k = torch.randn(n, h, N, d, device=device)
                v = torch.randn(n, h, N, d, device=device)
                mask = torch.ones(n, t, N, device=device)
                out, _ = tree_attention(q, k, v, mask)
                kernel_ok = bool(torch.isfinite(out).all()) and out.shape == q.shape
        except Exception as e:
            print(f"kernel smoke skipped/failed: {e}")
        if kernel_ok is None:
            print("NOTE: kernel check skipped on sm<80; run on A100/H100 (FULL=1 "
                  "or the smoke there)")
        else:
            checks.append((kernel_ok, f"triton TreeAttention dummy forward finite"))

        record = {
            "method": "longspec_smoke",
            "dataset": "import-check",
            "model": args.model_name,
            "input_tokens": None, "retained_tokens": None,
            "output_tokens": None, "batch_size": 1,
            "selector_latency_ms": None, "ttft_ms": None, "tpot_ms": None,
            "e2e_ms": None, "throughput_tok_s": None, "qps": None,
            "peak_memory_gb": None,
            "import_ok": import_ok, "kernel_ok": kernel_ok,
        }
        writer.add(record)
        print("LongSpec smoke: env imports + kernel forward (full inference "
              "requires 80GB-class GPU; see --full)")
        summary = {"type": "summary", "method": "longspec_smoke",
                   "import_ok": import_ok, "kernel_ok": kernel_ok}
    else:
        # ---- full: run repo inference script ------------------------------
        if not args.data_path_prefix:
            raise SystemExit("--data-path-prefix is required in full mode")
        cmd = [
            "python", "inference_long-bench.py",
            "--model_name", args.model_name,
            "--method", args.method,
            "--task", args.task,
            "--data_path_prefix", args.data_path_prefix,
            "--max_gen_len", str(args.max_gen_len),
            "--temperature", "0",
            "--tree_shape"] + args.tree_shape.split()
        print("+ " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=LONGSPEC, capture_output=True, text=True)
        print((proc.stdout or "")[-4000:])
        checks = [
            (proc.returncode == 0, f"exit code = {proc.returncode}"),
            ("Summary" in (proc.stdout or "") or "ROUGE" in (proc.stdout or ""),
             "inference produced output"),
        ]
        record = {"type": "longspec_full", "returncode": proc.returncode}
        writer.add(record)
        summary = {"type": "summary", "method": "longspec_full",
                   "returncode": proc.returncode}

    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    verify.finish("LongSpec", checks)


if __name__ == "__main__":
    main()
