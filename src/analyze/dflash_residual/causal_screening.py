"""GPU fixed-state screening for H19--H21.

This module creates a reproducible state bank and evaluates one native DFlash
block per state.  It intentionally keeps the state pair fixed: reference and
on-policy conditions use the same document, prefix length and block budget.
The collector records a scalar full-vocabulary draft rank, so E20 does not
need to persist the full vocabulary logits.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .schema import SCHEMA_VERSION
from .trace_dflash import (
    _encode,
    _prompt,
    acceptance_length_from_tokens,
    build_position_rows,
    truncate_input_ids,
)


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _load_models(args: argparse.Namespace) -> tuple[Any, Any, Any, torch.dtype, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("causal screening requires CUDA")
    root = Path(__file__).resolve().parents[3]
    dflash_root = root / "externals" / "dflash"
    if str(dflash_root) not in sys.path:
        sys.path.insert(0, str(dflash_root))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel

    dtype = _dtype(args.dtype)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    return target, draft, tokenizer, dtype, str(args.attn_implementation)


def _greedy_continuation(target: Any, input_ids: torch.Tensor, count: int, eos_id: int | None) -> list[int]:
    if count <= 0:
        return []
    with torch.inference_mode():
        output = target.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=int(count),
            do_sample=False,
            use_cache=True,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
        )
    return [int(token) for token in output[0, input_ids.shape[1]:].detach().cpu().tolist()]


def prepare_state_bank(
    records: Sequence[Mapping[str, Any]],
    target: Any,
    tokenizer: Any,
    *,
    max_docs: int,
    prefix_lengths: Sequence[int],
    max_reveal: int,
    context_cap: int,
    device: str,
) -> list[dict[str, Any]]:
    """Prepare paired fixed states and target-greedy reveal tokens."""

    result: list[dict[str, Any]] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    for index, record in enumerate(records[:max_docs]):
        sample_id = str(record.get("id", index))
        prompt = _prompt(record)
        reference = str(record.get("reference", ""))
        if not prompt or not reference:
            continue
        base = _encode(tokenizer, prompt)
        if base.shape[1] > context_cap:
            base = truncate_input_ids(base, context_cap, side="right")
        base = base.to(device)
        on_tokens = _greedy_continuation(
            target, base, max(max(prefix_lengths, default=0) + max_reveal + 1, 1), eos_id
        )
        reference_tokens = _token_ids(tokenizer(reference, add_special_tokens=False)["input_ids"])
        states: list[dict[str, Any]] = []
        for prefix_length in sorted({int(value) for value in prefix_lengths if int(value) >= 0}):
            if len(on_tokens) < prefix_length or len(reference_tokens) < prefix_length:
                continue
            on_prefix = on_tokens[:prefix_length]
            ref_prefix = reference_tokens[:prefix_length]
            on_state = torch.cat([
                base,
                torch.tensor([on_prefix], dtype=torch.long, device=base.device),
            ], dim=1)
            ref_state = torch.cat([
                base,
                torch.tensor([ref_prefix], dtype=torch.long, device=base.device),
            ], dim=1)
            ref_reveal = _greedy_continuation(
                target, ref_state, max_reveal + 1, eos_id
            )
            states.append({
                "fixed_state_id": f"{sample_id}::prefix{prefix_length}",
                "prefix_length": prefix_length,
                "on_policy_prefix_token_ids": on_prefix,
                "reference_prefix_token_ids": ref_prefix,
                "on_policy_reveal_token_ids": on_tokens[prefix_length:prefix_length + max_reveal + 1],
                "reference_reveal_token_ids": ref_reveal,
            })
        if states:
            result.append({
                "id": sample_id,
                "dataset": str(record.get("dataset", "other")),
                "document_id": sample_id,
                "base_input_ids": [int(token) for token in base[0].detach().cpu().tolist()],
                "states": states,
            })
        if (index + 1) % 5 == 0:
            print(f"state-bank {index + 1}/{min(max_docs, len(records))}", flush=True)
    return result


def _target_hidden(target: Any, output: Any, layer_ids: Sequence[int]) -> torch.Tensor:
    return torch.cat([output.hidden_states[layer_id + 1] for layer_id in layer_ids], dim=-1)


def _prepare_fixed_context(
    target: Any, draft: Any, input_ids: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Prefill one fixed state once; the target cache is cloned per reveal."""

    from transformers import DynamicCache

    device = input_ids.device
    input_length = int(input_ids.shape[1])
    position_ids = torch.arange(input_length + block_size, device=device).unsqueeze(0)
    past_target = DynamicCache()
    with torch.inference_mode():
        context_output = target(
            input_ids,
            position_ids=position_ids[:, :input_length],
            past_key_values=past_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
    layer_ids = getattr(draft, "target_layer_ids", None)
    if layer_ids is None:
        raise ValueError("DFlash checkpoint does not expose target_layer_ids")
    return position_ids, _target_hidden(target, context_output, layer_ids), past_target


def collect_fixed_block(
    target: Any,
    draft: Any,
    input_ids: torch.Tensor,
    *,
    run_id: str,
    sample_id: str,
    document_id: str,
    dataset: str,
    state_mode: str,
    fixed_state_id: str,
    prefix_length: int,
    reveal_count: int,
    reveal_tokens: Sequence[int],
    top_m: int,
    prepared_context: tuple[torch.Tensor, torch.Tensor, Any] | None = None,
) -> list[dict[str, Any]]:
    """Collect one native block at a fixed state.

    The first block token is the target-greedy anchor.  ``reveal_count``
    additional candidate positions are clamped to target-greedy tokens and
    excluded from the conditional oracle score by the offline analyzer.
    """

    from transformers import DynamicCache

    block_size = int(getattr(draft, "block_size", 0))
    if block_size <= 1:
        raise ValueError("DFlash native block size must be > 1")
    if reveal_count < 0 or reveal_count > block_size - 1:
        raise ValueError("reveal_count must be within candidate block")
    if len(reveal_tokens) < reveal_count + 1:
        raise ValueError("insufficient target reveal tokens for requested reveal_count")

    device = input_ids.device
    input_length = int(input_ids.shape[1])
    mask_token_id = getattr(draft, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(getattr(draft, "config", None), "dflash_config", {}).get("mask_token_id")
    if mask_token_id is None:
        raise ValueError("DFlash checkpoint does not define mask_token_id")
    if prepared_context is None:
        position_ids, target_hidden, context_cache = _prepare_fixed_context(
            target, draft, input_ids, block_size
        )
    else:
        position_ids, target_hidden, context_cache = prepared_context
    block = torch.full(
        (1, block_size), int(mask_token_id), dtype=torch.long, device=device
    )
    reveal = torch.tensor([list(reveal_tokens[:reveal_count + 1])], dtype=torch.long, device=device)
    block[:, :reveal_count + 1] = reveal
    # Target verification appends to its cache.  Each reveal condition gets an
    # independent copy of the same prefilled context cache.
    past_target = copy.deepcopy(context_cache)
    past_draft = DynamicCache()
    with torch.inference_mode():
        noise_embedding = target.model.embed_tokens(block)
        draft_hidden = draft(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids,
            past_key_values=past_draft,
            use_cache=True,
            is_causal=False,
        )
        draft_logits = target.lm_head(draft_hidden[:, 1 - block_size:, :])
        if not torch.isfinite(draft_logits).all():
            raise RuntimeError("non-finite fixed-state draft logits")
        candidate_logits, candidate_ids = torch.topk(
            draft_logits.float(), k=min(top_m, draft_logits.shape[-1]), dim=-1
        )
        raw_selected = torch.argmax(draft_logits, dim=-1)
        proposed = raw_selected.clone()
        if reveal_count:
            proposed[:, :reveal_count] = reveal[:, 1:reveal_count + 1]
        block[:, 1:] = proposed
        verifier_output = target(
            block,
            position_ids=position_ids[:, input_length:input_length + block_size],
            past_key_values=past_target,
            use_cache=True,
        )
        posterior = torch.argmax(verifier_output.logits, dim=-1)
        target_tokens = posterior[:, :-1]
        verifier_logits = verifier_output.logits[:, :-1, :].float()
        log_probs = torch.log_softmax(verifier_logits, dim=-1)
        target_probs = log_probs.exp()
        target_entropy = -(target_probs * log_probs).sum(dim=-1)
        target_top1_probability = log_probs.max(dim=-1).values.exp()
        target_draft_logits = torch.gather(
            draft_logits.float(), dim=-1, index=target_tokens.unsqueeze(-1)
        ).squeeze(-1)
        target_ranks = 1 + (
            draft_logits.float() > target_draft_logits.unsqueeze(-1)
        ).sum(dim=-1)
        accepted = acceptance_length_from_tokens(proposed, posterior)
        rows = build_position_rows(
            run_id=run_id,
            sample_id=sample_id,
            document_id=document_id,
            dataset=dataset,
            context_length=input_length,
            round_index=0,
            candidates=candidate_ids,
            candidate_logits=candidate_logits,
            target_tokens=target_tokens,
            dflash_selected=proposed,
            accepted_draft_len=accepted,
            block_size=block_size,
            native_block_size=block_size,
            target_entropy=target_entropy,
            target_top1_probability=target_top1_probability,
            draft_target_ranks=target_ranks,
            draft_target_logits=target_draft_logits,
            target_token_source="verifier_posterior",
            state_mode=state_mode,
        )
    for row in rows:
        row.update({
            "fixed_state_id": fixed_state_id,
            "state_prefix_length": int(prefix_length),
            "reveal_count": int(reveal_count),
            "candidate_budget": int(top_m),
            "raw_dflash_selected_token_id": int(raw_selected[0, row["draft_position"] - 1].item()),
        })
    return rows


def collect_state_bank(
    state_bank: Sequence[Mapping[str, Any]],
    target: Any,
    draft: Any,
    *,
    output: str | Path,
    top_m: int,
    reveal_counts: Sequence[int],
    device: str,
    seed: int,
) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()
    requested_states = 0
    for doc in state_bank:
        document_id = str(doc["document_id"])
        dataset = str(doc["dataset"])
        base = torch.tensor([doc["base_input_ids"]], dtype=torch.long, device=device)
        for state in doc["states"]:
            requested_states += 1
            fixed_id = str(state["fixed_state_id"])
            for mode, prefix_key, reveal_key in (
                ("on_policy", "on_policy_prefix_token_ids", "on_policy_reveal_token_ids"),
                ("reference", "reference_prefix_token_ids", "reference_reveal_token_ids"),
            ):
                prefix = torch.tensor([state[prefix_key]], dtype=torch.long, device=device)
                input_ids = torch.cat([base, prefix], dim=1)
                reveal = [int(token) for token in state[reveal_key]]
                prepared_context = _prepare_fixed_context(
                    target, draft, input_ids, int(getattr(draft, "block_size", 16))
                )
                for reveal_count in reveal_counts:
                    if len(reveal) < int(reveal_count) + 1:
                        continue
                    sample_id = f"{fixed_id}::{mode}::r{int(reveal_count)}"
                    run_id = f"t4-causal-screen-{dataset}-{mode}-r{int(reveal_count)}"
                    try:
                        rows.extend(collect_fixed_block(
                            target,
                            draft,
                            input_ids,
                            run_id=run_id,
                            sample_id=sample_id,
                            document_id=document_id,
                            dataset=dataset,
                            state_mode=mode,
                            fixed_state_id=fixed_id,
                            prefix_length=int(state["prefix_length"]),
                            reveal_count=int(reveal_count),
                            reveal_tokens=reveal,
                            top_m=top_m,
                            prepared_context=prepared_context,
                        ))
                    except Exception as exc:
                        errors.append({
                            "document_id": document_id,
                            "fixed_state_id": fixed_id,
                            "state_mode": mode,
                            "reveal_count": int(reveal_count),
                            "error": f"{type(exc).__name__}: {exc}",
                        })
            if requested_states % 10 == 0:
                print(f"fixed-state {dataset} {requested_states} states; rows={len(rows)}", flush=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for error in errors:
            handle.write(json.dumps({
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "run_id": "causal-screening",
                "sample_id": str(error["fixed_state_id"]),
                "document_id": str(error["document_id"]),
                **error,
            }, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "dflash_residual.trace.manifest.v1",
        "experiment": "E19_E20_E21",
        "output": str(output_path),
        "documents": len(state_bank),
        "fixed_states": requested_states,
        "ok_rows": len(rows),
        "error_rows": len(errors),
        "top_m": top_m,
        "reveal_counts": [int(value) for value in reveal_counts],
        "elapsed_s": time.perf_counter() - started,
        "seed": seed,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "device": device,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def run(args: argparse.Namespace) -> int:
    if args.dtype != "bfloat16":
        raise ValueError("T4 causal screening is locked to bfloat16")
    target, draft, tokenizer, _dtype_value, _attn = _load_models(args)
    records = _load_jsonl(args.input)
    bank = prepare_state_bank(
        records,
        target,
        tokenizer,
        max_docs=args.max_docs,
        prefix_lengths=[int(value) for value in args.prefix_lengths.split(",") if value.strip()],
        max_reveal=max(int(value) for value in args.reveal_counts.split(",") if value.strip()),
        context_cap=args.context_cap,
        device=args.device,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = output_dir / f"state_bank_{args.dataset}.jsonl"
    bank_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in bank),
        encoding="utf-8",
    )
    manifest = collect_state_bank(
        bank,
        target,
        draft,
        output=output_dir / f"trace_{args.dataset}.jsonl",
        top_m=args.top_m,
        reveal_counts=[int(value) for value in args.reveal_counts.split(",") if value.strip()],
        device=args.device,
        seed=args.seed,
    )
    manifest.update({
        "input": str(args.input),
        "dataset": args.dataset,
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "prefix_lengths": [int(value) for value in args.prefix_lengths.split(",") if value.strip()],
        "context_cap": args.context_cap,
        "state_bank": str(bank_path),
    })
    (output_dir / f"trace_{args.dataset}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-docs", type=int, default=30)
    parser.add_argument("--prefix-lengths", default="0,4,8")
    parser.add_argument("--reveal-counts", default="0,1,2,4")
    parser.add_argument("--context-cap", type=int, default=1024)
    parser.add_argument("--top-m", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
