#!/usr/bin/env python3
"""Intervene on one SpecExtend cycle while preserving the baseline trajectory.

For every state from a CMR32 state bank, this script starts from the original
document prompt, follows the ordinary CMR32 trajectory until the selected
cycle, swaps only the draft working-cache chunk set for that cycle, and stops
immediately after that verification.  Thus all policies see the same prefix
and target model state up to the intervention.  This is the stronger D1/D2
causal replay; a free-running policy comparison is not used here.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator

from common.reproducibility import seed_everything

ROOT = Path(__file__).resolve().parents[1]
SPECEXTEND_PACKAGE = ROOT / "externals" / "SpecExtend" / "specextend"
if str(SPECEXTEND_PACKAGE) not in sys.path:
    sys.path.insert(0, str(SPECEXTEND_PACKAGE))
from classic.model_classic import SPModel  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def chunks_for_policy(prefix_length: int, policy: str, recorded: list[int], seed: int, cycle: int) -> list[int]:
    if policy == "cmr32":
        return list(recorded)
    chunk_count = (int(prefix_length) + 31) // 32
    all_ids = list(range(chunk_count))
    if policy == "full":
        return all_ids
    if policy in {"cmr64", "cmr128", "recent32"}:
        width = int(policy[3:]) if policy.startswith("cmr") else 32
        return all_ids[-width:]
    if policy == "shuffled32":
        random.Random(seed + cycle - 1).shuffle(all_ids)
        return all_ids[:32]
    raise ValueError(f"unsupported policy: {policy}")


def encode(tokenizer, text: str, limit: int) -> torch.Tensor:
    ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)
    return ids[:, :limit] if limit and ids.shape[1] > limit else ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-bank", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--policies", default="cmr32,recent32,shuffled32,full")
    parser.add_argument("--max-docs", type=int, default=10)
    parser.add_argument("--cycles-per-doc", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)
    policies = [p.strip().lower() for p in args.policies.split(",") if p.strip()]
    valid = {"cmr32", "cmr64", "cmr128", "recent32", "shuffled32", "full"}
    if set(policies) - valid:
        raise SystemExit(f"unsupported policies: {sorted(set(policies)-valid)}")

    bank_rows = [
        row for row in load_jsonl(args.state_bank)
        if row.get("type") == "cycle"
        and isinstance(row.get("prefix_token_ids"), list)
        and isinstance(row.get("current_chunk_ids"), list)
    ]
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bank_rows:
        by_doc[str(row.get("sample_id"))].append(row)
    selected_states: list[dict[str, Any]] = []
    for doc_id in sorted(by_doc, key=str)[: max(0, args.max_docs)]:
        selected_states.extend(by_doc[doc_id][: max(0, args.cycles_per_doc)])
    source_rows = load_jsonl(args.input_file)
    if not selected_states:
        raise SystemExit("state bank contains no usable states")

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
    outputs: list[dict[str, Any]] = []
    for state_index, state in enumerate(selected_states):
        doc_index = int(state.get("sample_id", state_index))
        if doc_index >= len(source_rows):
            continue
        prompt = source_rows[doc_index].get("text")
        if not isinstance(prompt, str):
            continue
        cycle = int(state.get("cycle", 1))
        prefix_length = len(state["prefix_token_ids"])
        recorded = [int(x) for x in state["current_chunk_ids"]]
        # Baseline inputs are re-encoded from the same source JSONL and are
        # never replaced with the target prefix.  The target trajectory is
        # therefore identical up to the intervention under greedy decoding.
        input_ids = encode(model.tokenizer, prompt, args.max_input_tokens).to(accelerator.device)
        source_len = int(state.get("source_input_tokens") or input_ids.shape[1])
        for policy in policies:
            intervention_ids = chunks_for_policy(prefix_length, policy, recorded, args.seed, cycle)
            os.environ["SPECEXTEND_RETRIEVAL_POLICY"] = "cmr32"
            os.environ["SPECEXTEND_FIXED_CHUNK_IDS"] = json.dumps(intervention_ids)
            os.environ["SPECEXTEND_FIXED_AT_TIMESTEP"] = str(cycle - 1)
            os.environ["SPECEXTEND_REPLAY_STOP_CYCLE"] = str(cycle)
            os.environ["SPECEXTEND_ORIGINAL_INPUT_LEN"] = str(source_len)
            os.environ["SPECEXTEND_SEED"] = str(args.seed)
            os.environ.pop("SPECEXTEND_TRACE_FILE", None)
            os.environ.pop("SPECEXTEND_TRACE_SAMPLE_ID", None)

            started = time.perf_counter()
            error: str | None = None
            stats: dict[str, Any] = {}
            try:
                stats = model.spgenerate(
                    input_ids,
                    temperature=0,
                    max_new_tokens=max(16, cycle * 8),
                    output_result_line=False,
                    verbose=False,
                    use_specextend=True,
                    retrieval_chunk_size=32,
                    retrieve_top_k=32,
                    retrieve_every_n_steps=4,
                    retrieval_verbose=False,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:500]}"
            elapsed = time.perf_counter() - started
            values = stats.get("accept_length_list") or []
            accepted = int(values[cycle - 1]) + 1 if len(values) >= cycle else None
            baseline_value = int(state.get("accepted_tokens")) if state.get("accepted_tokens") is not None else None
            outputs.append({
                "type": "cycle_intervention",
                "state_index": state_index,
                "sample_id": state.get("sample_id"),
                "intervention_cycle": cycle,
                "prefix_length": prefix_length,
                "policy": policy,
                "intervention_chunk_ids": intervention_ids,
                "recorded_cmr_chunk_ids": recorded,
                "matched_target_prefix": True,
                "baseline_state_accepted_tokens": baseline_value,
                "accepted_tokens": accepted,
                "accept_length": accepted - 1 if accepted is not None else None,
                "trajectory_cycles_run": len(values),
                "wall_time_s": elapsed,
                "inference_time_s": stats.get("inference_time"),
                "timing": stats.get("timing"),
                "peak_memory_gb": stats.get("peak_memory_gb"),
                "status": "success" if error is None and accepted is not None else "failed",
                "error": error,
            })
            print(json.dumps(outputs[-1], ensure_ascii=False), flush=True)

    summaries: dict[str, Any] = {}
    for policy in policies:
        rows = [row for row in outputs if row["policy"] == policy]
        vals = [row["accepted_tokens"] for row in rows if row["accepted_tokens"] is not None]
        base = [row["baseline_state_accepted_tokens"] for row in rows if row["baseline_state_accepted_tokens"] is not None]
        summaries[policy] = {
            "states": len(rows),
            "successful_states": sum(row["status"] == "success" for row in rows),
            "mean_accepted_tokens": sum(vals) / len(vals) if vals else None,
            "mean_recorded_baseline_tokens": sum(base) / len(base) if base else None,
            "mean_gain_vs_recorded_baseline": (
                (sum(vals) - sum(base)) / len(vals) if vals and len(vals) == len(base) else None
            ),
            "exact_match_to_recorded_baseline": (
                sum(row["accepted_tokens"] == row["baseline_state_accepted_tokens"] for row in rows)
                / len(rows) if rows else None
            ),
        }
    with args.output.open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.write(json.dumps({
            "type": "summary",
            "states": len(selected_states),
            "policies": policies,
            "summaries": summaries,
        }, ensure_ascii=False) + "\n")
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
