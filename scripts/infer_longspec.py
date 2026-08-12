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
from pathlib import Path

import torch

from common import io_util, verify
from common.paths import ROOT

LONGSPEC = ROOT / "externals" / "LongSpec" / "longspec" / "test"
# LongSpec modules (llama_glide / qwen2_glide / triton_tree_attn) live in the
# repo's test dir and are imported as top-level modules.
sys.path.insert(0, str(LONGSPEC))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="run repo inference_long-bench.py (needs 80GB GPU)")
    parser.add_argument("--model-name", default="llama8b")
    parser.add_argument("--method", default="tree")
    parser.add_argument("--task", default="gov_report")
    parser.add_argument("--data-path-prefix", default=None,
                        help="dir with preprocessed longbench jsonl (full mode)")
    parser.add_argument("--tree-shape", default="4 16 16 16 16")
    parser.add_argument("--max-gen-len", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

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
