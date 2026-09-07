"""GPU screening for a self-conditioned DFlash redraft bridge.

The experiment is deliberately diagnostic rather than a claimed production
algorithm.  A normal DFlash block first produces a greedy draft.  Its first
``r`` draft tokens are then fed back as token embeddings in a second DFlash
block, which redrafts the remaining positions in parallel.  The target is
used only for verification/measurement; no target token is used to construct
the self-conditioned prefix.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from .causal_screening import (
    _load_jsonl,
    _load_models,
    _prepare_fixed_context,
    collect_fixed_block,
)
from .schema import SCHEMA_VERSION


def _block_time_fields(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"draft_s": 0.0, "verify_s": 0.0, "total_s": 0.0}
    row = rows[0]
    return {
        "draft_s": float(row.get("draft_elapsed_s", 0.0)),
        "verify_s": float(row.get("verify_elapsed_s", 0.0)),
        "total_s": float(row.get("cycle_elapsed_s", 0.0)),
    }


def _annotate(
    rows: list[dict[str, Any]],
    *,
    stage: str,
    self_condition_r: int,
    source_prefix: Sequence[int],
    timing: Mapping[str, float],
) -> None:
    for row in rows:
        row.update({
            "self_condition_stage": stage,
            "self_condition_r": int(self_condition_r),
            "self_condition_source_prefix_token_ids": [int(token) for token in source_prefix],
            "draft_elapsed_s": float(timing.get("draft_s", 0.0)),
            "verify_elapsed_s": float(timing.get("verify_s", 0.0)),
            "cycle_elapsed_s": float(timing.get("total_s", 0.0)),
        })


def run(args: argparse.Namespace) -> int:
    if args.dtype != "bfloat16":
        raise ValueError("T4 self-conditioned screening is locked to bfloat16")
    model_args = SimpleNamespace(
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device=args.device,
        target_model=args.target_model,
        draft_model=args.draft_model,
    )
    target, draft, _tokenizer, _dtype_value, _attn = _load_models(model_args)
    state_docs = _load_jsonl(args.state_bank)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()
    requested_states = 0
    timing_totals = {"baseline": {"draft_s": 0.0, "verify_s": 0.0, "total_s": 0.0},
                     "redraft": {"draft_s": 0.0, "verify_s": 0.0, "total_s": 0.0}}
    block_counts = {"baseline": 0, "redraft": 0}

    for doc in state_docs:
        document_id = str(doc["document_id"])
        dataset = str(doc.get("dataset", args.dataset))
        base = torch.tensor([doc["base_input_ids"]], dtype=torch.long, device=args.device)
        for state in doc.get("states", []):
            requested_states += 1
            fixed_id = str(state["fixed_state_id"])
            prefix = torch.tensor(
                [state["on_policy_prefix_token_ids"]], dtype=torch.long, device=args.device
            )
            input_ids = torch.cat([base, prefix], dim=1)
            target_reveal = [int(token) for token in state["on_policy_reveal_token_ids"]]
            prepared = _prepare_fixed_context(
                target, draft, input_ids, int(getattr(draft, "block_size", 16))
            )
            try:
                timing: dict[str, float] = {}
                baseline_rows = collect_fixed_block(
                    target,
                    draft,
                    input_ids,
                    run_id=f"t4-self-conditioned-{dataset}-baseline",
                    sample_id=f"{fixed_id}::self::baseline",
                    document_id=document_id,
                    dataset=dataset,
                    state_mode="on_policy",
                    fixed_state_id=fixed_id,
                    prefix_length=int(state["prefix_length"]),
                    reveal_count=0,
                    reveal_tokens=target_reveal,
                    top_m=args.top_m,
                    prepared_context=prepared,
                    timing=timing,
                )
                _annotate(
                    baseline_rows,
                    stage="baseline",
                    self_condition_r=0,
                    source_prefix=[],
                    timing={
                        "draft_s": timing.get("draft_s", 0.0),
                        "verify_s": timing.get("verify_s", 0.0),
                        "total_s": timing.get("total_s", 0.0),
                    },
                )
                rows.extend(baseline_rows)
                base_timing = _block_time_fields(baseline_rows)
                for key in timing_totals["baseline"]:
                    timing_totals["baseline"][key] += base_timing[key]
                block_counts["baseline"] += 1

                raw_by_position = {
                    int(row["draft_position"]): int(row["raw_dflash_selected_token_id"])
                    for row in baseline_rows
                }
                anchor = target_reveal[0]
                for reveal_count in args.self_condition_reveals:
                    r = int(reveal_count)
                    if r <= 0:
                        continue
                    source_prefix = [raw_by_position[position] for position in range(1, r + 1)]
                    self_reveal = [anchor] + source_prefix
                    timing = {}
                    redraft_rows = collect_fixed_block(
                        target,
                        draft,
                        input_ids,
                        run_id=f"t4-self-conditioned-{dataset}-redraft-r{r}",
                        sample_id=f"{fixed_id}::self::r{r}",
                        document_id=document_id,
                        dataset=dataset,
                        state_mode="on_policy",
                        fixed_state_id=fixed_id,
                        prefix_length=int(state["prefix_length"]),
                        reveal_count=r,
                        reveal_tokens=self_reveal,
                        top_m=args.top_m,
                        prepared_context=prepared,
                        timing=timing,
                    )
                    _annotate(
                        redraft_rows,
                        stage="redraft",
                        self_condition_r=r,
                        source_prefix=source_prefix,
                        timing={
                            "draft_s": timing.get("draft_s", 0.0),
                            "verify_s": timing.get("verify_s", 0.0),
                            "total_s": timing.get("total_s", 0.0),
                        },
                    )
                    rows.extend(redraft_rows)
                    redraft_timing = _block_time_fields(redraft_rows)
                    for key in timing_totals["redraft"]:
                        timing_totals["redraft"][key] += redraft_timing[key]
                    block_counts["redraft"] += 1
            except Exception as exc:
                errors.append({
                    "document_id": document_id,
                    "fixed_state_id": fixed_id,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            if requested_states % 10 == 0:
                print(f"self-conditioned {dataset} {requested_states} states; rows={len(rows)}", flush=True)

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for error in errors:
            handle.write(json.dumps({
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "run_id": "self-conditioned-screening",
                "sample_id": str(error["fixed_state_id"]),
                "document_id": str(error["document_id"]),
                **error,
            }, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "dflash_residual.trace.manifest.v1",
        "experiment": "E21_SELF_CONDITIONED",
        "output": str(output),
        "dataset": args.dataset,
        "documents": len(state_docs),
        "fixed_states": requested_states,
        "baseline_blocks": block_counts["baseline"],
        "redraft_blocks_per_r": block_counts["redraft"],
        "self_condition_reveals": [int(value) for value in args.self_condition_reveals],
        "ok_rows": len(rows),
        "error_rows": len(errors),
        "top_m": args.top_m,
        "elapsed_s": time.perf_counter() - started,
        "timing_totals_s": timing_totals,
        "block_counts": block_counts,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "device": args.device,
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "state_bank": args.state_bank,
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--state-bank", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-m", type=int, default=128)
    parser.add_argument("--self-condition-reveals", default="1,2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    args = parser.parse_args()
    args.self_condition_reveals = [
        int(value) for value in args.self_condition_reveals.split(",") if value.strip()
    ]
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
