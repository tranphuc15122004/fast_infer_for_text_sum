#!/usr/bin/env python3
"""HiGOE verification / smoke script.

HiGOE is a multi-step pipeline (graph construction -> knowledge synthesis ->
training -> eval) that needs API keys and several external datasets for the
full run. The smoke test verifies the environment + core components instead:

  1. every HiGOE module imports in this env (retrieval, utils, data_process,
     prompt_pool, graph_construction, training_preparation),
  2. a tiny Contriever retrieval round-trip on dummy documents returns hits.

Full paper setup (--full) is documented in externals/HiGOE/README.md and
requires the datasets + an LLM judge (API key or local model).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import io_util, verify
from common.paths import ROOT

HIGOE = ROOT / "externals" / "HiGOE"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retriever-model", default="facebook/contriever")
    parser.add_argument("--num-docs", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # 1) module imports ----------------------------------------------------
    import_errs: list[str] = []
    mods = [
        "retrieval", "utils", "data_process", "prompt_pool",
        "graph_construction", "training_preparation",
    ]
    for mod in mods:
        try:
            __import__(mod)
        except Exception as e:  # noqa: BLE001
            import_errs.append(f"{mod}: {e}")
    imports_ok = not import_errs
    checks: list[tuple[bool, str]] = [
        (imports_ok, "HiGOE modules import" + ("" if imports_ok else f" FAILED: {import_errs}"))
    ]

    # 2) Contriever round-trip (tiny, CPU-friendly) ------------------------
    retrieval_ok = False
    try:
        from transformers import AutoModel, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(args.retriever_model)
        model = AutoModel.from_pretrained(args.retriever_model).to(device).eval()

        docs = [
            "The central bank raised interest rates to curb inflation.",
            "A new machine learning model predicts protein structures.",
            "NASA's Perseverance rover collected a rock sample on Mars.",
        ][: args.num_docs]
        query = "How does the rover collect samples on Mars?"

        def embed(texts: list[str]) -> torch.Tensor:
            enc = tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.inference_mode():
                out = model(**enc)
            return out.last_hidden_state[:, 0]

        q = embed([query])
        d = embed(docs)
        scores = torch.matmul(q, d.T)[0]
        top = int(scores.argmax())
        retrieval_ok = scores.numel() == len(docs) and torch.isfinite(scores).all()
        print(f"retrieval top hit: doc #{top} ({scores[top].item():.3f})")
    except Exception as e:  # noqa: BLE001
        print(f"retrieval smoke failed: {e}")
    checks.append((retrieval_ok, "Contriever round-trip finite, shape ok"))

    writer = io_util.JsonlWriter(Path(args.output))
    record = {
        "method": "higoe_smoke",
        "dataset": "dummy",
        "model": args.retriever_model,
        "input_tokens": None, "retained_tokens": None,
        "output_tokens": None, "batch_size": 1,
        "selector_latency_ms": None, "ttft_ms": None, "tpot_ms": None,
        "e2e_ms": None, "throughput_tok_s": None, "qps": None,
        "peak_memory_gb": None,
        "imports_ok": imports_ok, "retrieval_ok": retrieval_ok,
        "import_errs": import_errs,
    }
    writer.add(record)
    summary = {"type": "summary", "method": "higoe_smoke",
               "imports_ok": imports_ok, "retrieval_ok": retrieval_ok}
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    print("NOTE: full HiGOE pipeline needs datasets (QMSum/WCEP/BookSum/GovReport/SQuALITY)")
    print("      + LLM judge (API key or local model); see externals/HiGOE/README.md")
    verify.finish("HiGOE", checks)


if __name__ == "__main__":
    main()
