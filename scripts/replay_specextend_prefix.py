#!/usr/bin/env python3
"""Replay SpecExtend at fixed target prefixes under alternative draft contexts.

The state bank is a trace produced with ``SPECEXTEND_TRACE_PREFIX=1``.  Each
selected row contains the exact target prefix before a verification cycle and
the CMR chunk IDs used for that cycle.  The script loads the target/draft pair
once, then runs one speculative cycle per (state, policy).  It is therefore a
matched-prefix diagnostic, not a normal free-running generation benchmark.

The ``cmr32`` condition is forced to the chunk IDs recorded in the state bank;
``recent32``, ``shuffled32`` and ``full`` use their corresponding control
policies on the same prefix.  This avoids interpreting trajectory differences
as causal context effects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator

from common.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]
SPECEXTEND = ROOT / "externals" / "SpecExtend"
SPECEXTEND_PACKAGE = SPECEXTEND / "specextend"
if str(SPECEXTEND_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SPECEXTEND_PACKAGE))

from classic.model_classic import SPModel  # noqa: E402


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict) and value.get("type") == "cycle":
            prefix = value.get("prefix_token_ids")
            chunks = value.get("current_chunk_ids")
            if isinstance(prefix, list) and prefix and isinstance(chunks, list) and chunks:
                rows.append(value)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--policies", default="cmr32,recent32,shuffled32,full")
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--cycles-per-doc", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    policies = [item.strip().lower() for item in args.policies.split(",") if item.strip()]
    valid = {"cmr32", "recent32", "shuffled32", "full"}
    unknown = sorted(set(policies) - valid)
    if unknown:
        raise SystemExit(f"unsupported policies: {unknown}")

    all_rows = load_trace(args.state_bank)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_doc.setdefault(str(row.get("sample_id", "unknown")), []).append(row)
    state_rows: list[dict[str, Any]] = []
    for doc_id in sorted(by_doc, key=str)[: max(0, args.max_docs)]:
        state_rows.extend(by_doc[doc_id][: max(0, args.cycles_per_doc)])
    if not state_rows:
        raise SystemExit("state bank has no usable prefix rows")

    model = SPModel.from_pretrained(
        base_model_path=args.target_model,
        draft_model_path=args.draft_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    accelerator = Accelerator()
    model, _ = accelerator.prepare(model, model.tokenizer)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    for state_index, state in enumerate(state_rows):
        prefix = torch.tensor([state["prefix_token_ids"]], dtype=torch.long)
        prefix = prefix.to(accelerator.device)
        recorded_ids = [int(value) for value in state["current_chunk_ids"]]
        for policy in policies:
            os.environ["SPECEXTEND_RETRIEVAL_POLICY"] = policy
            os.environ["SPECEXTEND_SEED"] = str(args.seed)
            os.environ["SPECEXTEND_ORIGINAL_INPUT_LEN"] = str(
                state.get("source_input_tokens") or len(state["prefix_token_ids"])
            )
            os.environ.pop("SPECEXTEND_TRACE_FILE", None)
            os.environ.pop("SPECEXTEND_TRACE_SAMPLE_ID", None)
            if policy == "cmr32":
                os.environ["SPECEXTEND_FIXED_CHUNK_IDS"] = json.dumps(recorded_ids)
            else:
                os.environ.pop("SPECEXTEND_FIXED_CHUNK_IDS", None)

            started = time.perf_counter()
            error: str | None = None
            stats: dict[str, Any] = {}
            try:
                stats = model.spgenerate(
                    prefix,
                    temperature=0,
                    max_new_tokens=1,
                    output_result_line=False,
                    verbose=False,
                    use_specextend=True,
                    retrieval_chunk_size=32,
                    retrieve_top_k=32,
                    retrieve_every_n_steps=4,
                    retrieval_verbose=False,
                )
            except Exception as exc:  # keep other policies diagnosable
                error = f"{type(exc).__name__}: {str(exc)[:500]}"
            wall_s = time.perf_counter() - started
            accept_list = stats.get("accept_length_list") or []
            accepted = int(accept_list[0]) + 1 if accept_list else None
            result_rows.append(
                {
                    "type": "replay",
                    "state_index": state_index,
                    "sample_id": state.get("sample_id"),
                    "source_cycle": state.get("cycle"),
                    "prefix_length": len(state["prefix_token_ids"]),
                    "recorded_cmr_chunk_ids": recorded_ids,
                    "policy": policy,
                    "matched_prefix": True,
                    "accepted_tokens": accepted,
                    "accept_length": int(accept_list[0]) if accept_list else None,
                    "total_generated": stats.get("total_generated"),
                    "cycle_count": len(accept_list),
                    "wall_time_s": wall_s,
                    "inference_time_s": stats.get("inference_time"),
                    "timing": stats.get("timing"),
                    "peak_memory_gb": stats.get("peak_memory_gb"),
                    "status": "success" if error is None else "failed",
                    "error": error,
                }
            )
            print(
                json.dumps(result_rows[-1], ensure_ascii=False),
                flush=True,
            )

    summaries: dict[str, Any] = {}
    for policy in policies:
        values = [
            row["accepted_tokens"]
            for row in result_rows
            if row["policy"] == policy and row["accepted_tokens"] is not None
        ]
        successful = [
            row for row in result_rows if row["policy"] == policy and row["status"] == "success"
        ]
        summaries[policy] = {
            "states": len([row for row in result_rows if row["policy"] == policy]),
            "successful_states": len(successful),
            "mean_accepted_tokens": sum(values) / len(values) if values else None,
            "p_accept_at_least_2": sum(value >= 2 for value in values) / len(values) if values else None,
            "mean_wall_time_s": (
                sum(float(row["wall_time_s"]) for row in successful) / len(successful)
                if successful else None
            ),
        }

    with args.output.open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.write(
            json.dumps(
                {
                    "type": "summary",
                    "state_bank": str(args.state_bank),
                    "states": len(state_rows),
                    "policies": policies,
                    "summaries": summaries,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
