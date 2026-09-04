"""Instrumented DFlash Transformers collector for candidate-coverage traces.

The collector is intentionally separate from ``scripts/infer_dflash.py``.
It runs the same target-hidden → diffusion-draft → target-verification loop,
but records the candidate lattice and verifier posterior needed by P2–P4.
Imports that require CUDA/Transformers happen only in ``run``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .schema import SCHEMA_VERSION, context_bin, task_regime_for_dataset


def parse_context_caps(value: str) -> list[int]:
    caps = sorted({int(part.strip()) for part in str(value).split(",") if part.strip()})
    if not caps or any(cap <= 0 for cap in caps):
        raise ValueError("context lengths must contain positive integers")
    return caps


def truncate_input_ids(input_ids: torch.Tensor, max_tokens: int, *, side: str = "right") -> torch.Tensor:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if side == "right":
        return input_ids[:, :max_tokens]
    if side == "left":
        return input_ids[:, -max_tokens:]
    raise ValueError("side must be 'left' or 'right'")


def acceptance_length_from_tokens(proposed: torch.Tensor, posterior: torch.Tensor) -> int:
    """Count the consecutive prefix where draft tokens equal target tokens."""

    if proposed.ndim != 2 or posterior.ndim != 2 or proposed.shape[0] != 1 or posterior.shape[0] != 1:
        raise ValueError("tokens must have shape [1, sequence]")
    length = min(proposed.shape[1], posterior.shape[1] - 1)
    accepted = 0
    for index in range(length):
        if int(proposed[0, index]) != int(posterior[0, index]):
            break
        accepted += 1
    return accepted


def _target_rank(candidates: Sequence[int], target: int) -> int | None:
    try:
        return list(candidates).index(int(target)) + 1
    except ValueError:
        return None


def build_position_rows(
    *,
    run_id: str,
    sample_id: str,
    document_id: str,
    dataset: str,
    context_length: int,
    round_index: int,
    candidates: torch.Tensor,
    candidate_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    dflash_selected: torch.Tensor,
    accepted_draft_len: int,
    block_size: int,
    native_block_size: int,
    target_token_source: str = "verifier_posterior",
) -> list[dict[str, Any]]:
    """Convert one block's tensors to JSON-safe per-position rows."""

    if candidates.ndim != 3 or candidates.shape[0] != 1:
        raise ValueError("candidates must have shape [1, depth, top_m]")
    if candidate_logits.shape != candidates.shape:
        raise ValueError("candidate_logits shape must match candidates")
    if target_tokens.shape[0] != 1 or dflash_selected.shape[0] != 1:
        raise ValueError("target/dflash tokens must have batch dimension 1")
    rows: list[dict[str, Any]] = []
    depth = candidates.shape[1]
    for index in range(depth):
        candidate_ids = [int(value) for value in candidates[0, index].detach().cpu().tolist()]
        target = int(target_tokens[0, index].item())
        row = {
            "schema_version": SCHEMA_VERSION,
            "status": "ok",
            "run_id": str(run_id),
            "sample_id": str(sample_id),
            "document_id": str(document_id),
            "dataset": str(dataset),
            "task_regime": task_regime_for_dataset(dataset),
            "context_length": int(context_length),
            "context_bin": context_bin(context_length),
            "round_index": int(round_index),
            "draft_position": index + 1,
            "max_depth": depth,
            "target_token_id": target,
            "target_token_source": target_token_source,
            "candidate_token_ids": candidate_ids,
            "candidate_logits": [float(value) for value in candidate_logits[0, index].detach().cpu().tolist()],
            "dflash_selected_token_id": int(dflash_selected[0, index].item()),
            "accepted_draft_len": int(accepted_draft_len),
            "committed_tokens": int(accepted_draft_len) + 1,
            "block_size": int(block_size),
            "native_block_size": int(native_block_size),
            "target_rank": _target_rank(candidate_ids, target),
            "target_in_top16": target in candidate_ids[:16],
        }
        rows.append(row)
    return rows


def _model_call(model: Any, **kwargs: Any) -> Any:
    try:
        return model(logits_to_keep=1, **kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        return model(**kwargs)


def collect_one(
    target: Any,
    draft: Any,
    input_ids: torch.Tensor,
    *,
    run_id: str,
    sample_id: str,
    document_id: str,
    dataset: str,
    top_m: int = 16,
    block_size: int | None = None,
    max_new_tokens: int = 32,
    stop_token_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Collect one deterministic DFlash run, including every verified block."""

    if top_m <= 0 or max_new_tokens <= 0:
        raise ValueError("top_m and max_new_tokens must be positive")
    device = input_ids.device
    native_block_size = int(getattr(draft, "block_size", 0))
    physical_block_size = int(block_size or native_block_size)
    if physical_block_size <= 1:
        raise ValueError("DFlash trace requires a block size greater than one")
    if native_block_size and physical_block_size != native_block_size:
        raise ValueError("collector requires the native DFlash block size")
    mask_token_id = getattr(draft, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(getattr(draft, "config", None), "dflash_config", {}).get("mask_token_id")
    if mask_token_id is None:
        raise ValueError("DFlash checkpoint does not define mask_token_id")
    try:
        from transformers import DynamicCache
    except Exception as exc:
        raise RuntimeError("Transformers is required for the GPU collector") from exc
    input_ids = input_ids.to(device)
    input_length = int(input_ids.shape[1])
    max_length = input_length + int(max_new_tokens)
    output_ids = torch.full((1, max_length + physical_block_size), int(mask_token_id), dtype=torch.long, device=device)
    position_ids = torch.arange(output_ids.shape[1], device=device).unsqueeze(0)
    past_target = DynamicCache()
    past_draft = DynamicCache()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        output = target(
            input_ids,
            position_ids=position_ids[:, :input_length],
            past_key_values=past_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        output_ids[:, :input_length] = input_ids
        output_ids[:, input_length:input_length + 1] = torch.argmax(output.logits, dim=-1)[:, -1:]
        target_layer_ids = getattr(draft, "target_layer_ids", None)
        if target_layer_ids is None:
            raise ValueError("DFlash checkpoint does not expose target_layer_ids")
        target_hidden = torch.cat([output.hidden_states[layer_id + 1] for layer_id in target_layer_ids], dim=-1)
        start = input_length
        round_index = 0
        while start < max_length:
            block_output_ids = output_ids[:, start:start + physical_block_size].clone()
            block_position_ids = position_ids[:, start:start + physical_block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_hidden = draft(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_draft.get_seq_length():start + physical_block_size],
                past_key_values=past_draft,
                use_cache=True,
                is_causal=False,
            )
            draft_logits = target.lm_head(draft_hidden[:, 1 - physical_block_size:, :])
            past_draft.crop(start)
            candidate_logits, candidate_ids = torch.topk(
                draft_logits.float(), k=min(top_m, draft_logits.shape[-1]), dim=-1
            )
            dflash_selected = torch.argmax(draft_logits, dim=-1)
            block_output_ids[:, 1:] = dflash_selected
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_target,
                use_cache=True,
                output_hidden_states=True,
            )
            posterior = torch.argmax(output.logits, dim=-1)
            proposed = block_output_ids[:, 1:]
            target_tokens = posterior[:, :-1]
            accepted = acceptance_length_from_tokens(proposed, posterior)
            rows.extend(build_position_rows(
                run_id=run_id,
                sample_id=sample_id,
                document_id=document_id,
                dataset=dataset,
                context_length=input_length,
                round_index=round_index,
                candidates=candidate_ids,
                candidate_logits=candidate_logits,
                target_tokens=target_tokens,
                dflash_selected=dflash_selected,
                accepted_draft_len=accepted,
                block_size=physical_block_size,
                native_block_size=native_block_size or physical_block_size,
            ))
            output_ids[:, start:start + accepted + 1] = block_output_ids[:, :accepted + 1]
            output_ids[:, start + accepted + 1] = posterior[:, accepted]
            start += accepted + 1
            past_target.crop(start)
            target_hidden = torch.cat([output.hidden_states[layer_id + 1] for layer_id in target_layer_ids], dim=-1)[:, :accepted + 1, :]
            round_index += 1
            if stop_token_ids and any(int(token) in output_ids[:, input_length:].tolist()[0] for token in stop_token_ids):
                break
    return rows


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _prompt(row: Mapping[str, Any]) -> str:
    if row.get("context") is not None:
        try:
            from scripts.common.benchmark_data import render_prompt

            return render_prompt(row)
        except Exception:
            query = str(row.get("input") or "")
            return f"Context:\n{row.get('context', '')}\n\n{query}".strip()
    for key in ("prompt", "question", "document", "text", "input"):
        value = row.get(key)
        if value:
            return str(value)
    return ""


def _encode(tokenizer: Any, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
                return_dict=False,
            )
        except TypeError:
            encoded = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            )
    else:
        encoded = tokenizer(prompt, return_tensors="pt").input_ids
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded.ndim != 2:
        raise ValueError("tokenizer must return [batch, sequence] input IDs")
    return encoded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", default="dflash-trace")
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-m", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--context-lengths", default="", help="comma-separated input caps; empty keeps full prompt")
    parser.add_argument("--truncate-side", choices=("left", "right"), default="right")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--attn-implementation", choices=("auto", "sdpa", "flash_attention_2"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.max_samples <= 0 or args.max_new_tokens <= 0:
        raise ValueError("max-samples and max-new-tokens must be positive")
    caps = parse_context_caps(args.context_lengths) if args.context_lengths else [None]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise SystemExit("DFlash residual collector requires CUDA; host reports unavailable")
    root = Path(__file__).resolve().parents[3]
    dflash_root = root / "externals" / "dflash"
    if str(dflash_root) not in sys.path:
        sys.path.insert(0, str(dflash_root))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from dflash.model import DFlashDraftModel
    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if args.attn_implementation == "auto":
        try:
            import flash_attn  # noqa: F401

            attn_implementation = "flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa"
        except Exception:
            attn_implementation = "sdpa"
    else:
        attn_implementation = args.attn_implementation
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        dtype=dtype,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    draft = DFlashDraftModel.from_pretrained(
        args.draft_model,
        dtype=dtype,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=True,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    records = _load_jsonl(Path(args.input))[:args.max_samples]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for record in records:
        full_ids = _encode(tokenizer, _prompt(record))
        for cap in caps:
            input_ids = full_ids if cap is None else truncate_input_ids(full_ids, cap, side=args.truncate_side)
            input_ids = input_ids.to(args.device)
            sample_id = str(record.get("id", len(output_rows)))
            try:
                rows = collect_one(
                    target, draft, input_ids,
                    run_id=args.run_id,
                    sample_id=sample_id,
                    document_id=sample_id,
                    dataset=str(record.get("dataset", record.get("task_regime", "other"))),
                    top_m=args.top_m,
                    block_size=args.block_size,
                    max_new_tokens=args.max_new_tokens,
                    stop_token_ids=[int(tokenizer.eos_token_id)] if tokenizer.eos_token_id is not None else None,
                )
                for row in rows:
                    row["context_cap"] = cap
                    row["truncate_side"] = args.truncate_side if cap is not None else None
                output_rows.extend(rows)
            except Exception as exc:
                output_rows.append({
                    "schema_version": SCHEMA_VERSION,
                    "status": "error",
                    "run_id": args.run_id,
                    "sample_id": sample_id,
                    "context_cap": cap,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        print(f"trace {sample_id} {len(output_rows)} rows", flush=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_path.with_suffix(".manifest.json").write_text(json.dumps({
        "schema_version": "dflash_residual.trace.manifest.v1",
        "run_id": args.run_id,
        "target_model": args.target_model,
        "draft_model": args.draft_model,
        "input": str(Path(args.input)),
        "output": str(output_path),
        "requested_samples": len(records),
        "context_caps": caps,
        "truncate_side": args.truncate_side,
        "top_m": args.top_m,
        "block_size": args.block_size or int(draft.block_size),
        "native_block_size": int(draft.block_size),
        "ok_rows": sum(row.get("status") == "ok" for row in output_rows),
        "error_rows": sum(row.get("status") != "ok" for row in output_rows),
        "elapsed_s": time.perf_counter() - started,
        "seed": args.seed,
        "attn_implementation": attn_implementation,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
