"""Controlled greedy speculative traces for GroundSync H2/H4.

The runner compares a small draft model's continuation with a canonical target
continuation.  Under deterministic greedy decoding this is the same accepted
prefix that target verification would commit, without requiring an EAGLE
checkpoint or a CUDA-only serving stack.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .trace_target import (
    RenderedPrompt,
    _chunked_prefill,
    _record_document,
    load_local_model,
    load_jsonl,
    render_document_prompt,
)
from .core import accepted_prefix_length


def acceptance_record(
    proposed: Sequence[int],
    canonical: Sequence[int],
) -> dict[str, Any]:
    """Summarize longest-prefix acceptance against the target continuation."""

    accepted = accepted_prefix_length(proposed, canonical)
    fully_accepted = accepted == len(proposed)
    return {
        "accepted_len": accepted,
        "first_reject_rel": None if fully_accepted else accepted + 1,
        "fully_accepted": fully_accepted,
    }


def select_start_positions(
    output_length: int,
    *,
    max_starts: int,
    stride: int,
    start_offset: int = 0,
    max_new_tokens: int | None = None,
) -> list[int]:
    """Choose deterministic positions whose requested continuation is complete."""

    if output_length < 0:
        raise ValueError("output_length must be non-negative")
    if max_starts <= 0:
        raise ValueError("max_starts must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if start_offset < 0:
        raise ValueError("start_offset must be non-negative")
    if max_new_tokens is not None and max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    last_start = output_length - max_new_tokens if max_new_tokens is not None else output_length - 1
    if last_start < start_offset:
        return []
    return list(range(start_offset, last_start + 1, stride))[:max_starts]


def _model_call(model: Any, **kwargs: Any) -> Any:
    try:
        return model(logits_to_keep=1, **kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        return model(**kwargs)


def generate_draft_proposal(
    model: Any,
    prefix_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    device: torch.device,
    eos_token_id: int | None = None,
    prefill_chunk_size: int = 512,
) -> tuple[list[int], list[float], float, list[float]]:
    """Generate a greedy draft continuation and max-probability confidences."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prefix_ids = prefix_ids.to(device)
    prefill_started = time.perf_counter()
    proposed: list[int] = []
    confidences: list[float] = []
    elapsed_by_k: list[float] = []
    with torch.inference_mode():
        # Keep timing semantics aligned with the target AR measurement and
        # target block verifier: prefix prefill is outside the per-round cost.
        # The final prefix token is then an incremental draft query producing
        # the first proposed token; subsequent proposed tokens are cached
        # one-token forwards.
        outputs = _chunked_prefill(
            model,
            prefix_ids[:, :-1],
            prefill_chunk_size=prefill_chunk_size,
        )
        past_key_values = outputs.past_key_values
        decode_started = time.perf_counter()
        outputs = _model_call(
            model,
            input_ids=prefix_ids[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :].float()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(max_new_tokens):
            probabilities = torch.softmax(logits, dim=-1)
            confidence, next_token = probabilities.max(dim=-1, keepdim=True)
            token_id = int(next_token[0, 0].item())
            proposed.append(token_id)
            confidences.append(float(confidence[0, 0].item()))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_by_k.append((time.perf_counter() - decode_started) * 1000.0)
            if eos_token_id is not None and token_id == int(eos_token_id):
                break
            outputs = _model_call(
                model,
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :].float()
    elapsed_ms = (time.perf_counter() - prefill_started) * 1000.0
    return proposed, confidences, elapsed_ms, elapsed_by_k


def _measure_verification_by_k(
    model: Any,
    prefix_ids: torch.Tensor,
    proposed: Sequence[int],
    *,
    device: torch.device,
    prefill_chunk_size: int = 512,
) -> list[float]:
    """Measure cached target checks for each prefix length of one proposal."""

    if not proposed:
        return []
    prefix_ids = prefix_ids.to(device)
    token_ids = torch.tensor([list(proposed)], dtype=prefix_ids.dtype, device=device)
    with torch.inference_mode():
        prefill = _chunked_prefill(
            model,
            prefix_ids,
            prefill_chunk_size=prefill_chunk_size,
        )
        base_past_key_values = prefill.past_key_values
        result: list[float] = []
        for end in range(1, token_ids.shape[1] + 1):
            # A block verifier evaluates all proposed tokens in one causal
            # forward.  Clone the prefix cache so measuring k=1 does not
            # mutate the cache used for k=2, ... on Transformers cache APIs
            # that update objects in place.
            try:
                past_key_values = copy.deepcopy(base_past_key_values)
            except (TypeError, RuntimeError):
                past_key_values = base_past_key_values
            started = time.perf_counter()
            _model_call(
                model,
                input_ids=token_ids[:, :end],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=False,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            result.append((time.perf_counter() - started) * 1000.0)
    return result


def _measure_autoregressive_time(
    model: Any,
    prefix_ids: torch.Tensor,
    *,
    device: torch.device,
    prefill_chunk_size: int = 512,
) -> float:
    """Measure the cached one-token target check used by the ``k=0`` policy."""

    prefix_ids = prefix_ids.to(device)
    if prefix_ids.shape[1] < 2:
        raise ValueError("prefix must contain at least two tokens")
    with torch.inference_mode():
        prefill = _chunked_prefill(
            model,
            prefix_ids[:, :-1],
            prefill_chunk_size=prefill_chunk_size,
        )
        started = time.perf_counter()
        _model_call(
            model,
            input_ids=prefix_ids[:, -1:],
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000.0
    if elapsed <= 0.0 or not torch.isfinite(torch.tensor(elapsed)):
        raise ValueError("autoregressive timing is non-positive or non-finite")
    return elapsed


def run_one_speculative_trace(
    model: Any,
    tokenizer: Any,
    rendered: RenderedPrompt,
    target_row: Mapping[str, Any],
    *,
    document_id: str,
    max_k: int,
    max_starts: int,
    stride: int,
    start_offset: int = 0,
    device: torch.device,
    horizon_threshold: float = 0.2,
    verification_model: Any | None = None,
    verification_rendered: RenderedPrompt | None = None,
    verification_device: torch.device | None = None,
    prefill_chunk_size: int = 512,
) -> list[dict[str, Any]]:
    """Run controlled proposals at selected positions of one target trace."""

    canonical = [int(token) for token in target_row.get("generated_token_ids", [])]
    if not canonical:
        raise ValueError("target row has no generated_token_ids")
    starts = select_start_positions(
        len(canonical), max_starts=max_starts, stride=stride,
        start_offset=start_offset, max_new_tokens=max_k,
    )
    results: list[dict[str, Any]] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    target_attention = target_row.get("attention", [])
    target_trace = [step.get("nosink") for step in target_attention if step.get("nosink")]
    target_entropy = target_row.get("target_entropy", [])
    acceptance_history: list[float] = []
    for start in starts:
        continuation = canonical[start : start + max_k]
        prefix_tokens = torch.tensor(
            canonical[:start], dtype=rendered.input_ids.dtype, device=device
        )
        prefix_ids = torch.cat([rendered.input_ids.to(device), prefix_tokens.unsqueeze(0)], dim=1)
        proposed, confidences, draft_ms, draft_time_by_k = generate_draft_proposal(
            model,
            prefix_ids,
            max_new_tokens=max_k,
            device=device,
            eos_token_id=eos_id,
            prefill_chunk_size=prefill_chunk_size,
        )
        verification_time_by_k = None
        autoregressive_time_ms = None
        if verification_model is not None and verification_rendered is not None:
            verification_prefix_ids = torch.cat(
                [
                    verification_rendered.input_ids.to(verification_device or device),
                    prefix_tokens.to(verification_device or device).unsqueeze(0),
                ],
                dim=1,
            )
            verification_time_by_k = _measure_verification_by_k(
                verification_model,
                verification_prefix_ids,
                proposed,
                device=verification_device or device,
                prefill_chunk_size=prefill_chunk_size,
            )
            autoregressive_time_ms = _measure_autoregressive_time(
                verification_model,
                verification_prefix_ids,
                device=verification_device or device,
                prefill_chunk_size=prefill_chunk_size,
            )
        result = acceptance_record(proposed, continuation)
        drift = None
        if start < len(target_attention) and start > 0:
            from .core import js_divergence

            current = target_attention[start].get("nosink")
            previous = target_attention[start - 1].get("nosink")
            if current is not None and previous is not None:
                drift = js_divergence(current, previous)
        horizon = None
        if start < len(target_trace):
            from .core import grounding_horizon

            horizon = grounding_horizon(
                target_trace,
                start=start,
                threshold=horizon_threshold,
                max_horizon=max_k,
            )
        results.append({
            "schema_version": "groundsync.spec.v1",
            "status": "ok",
            "document_id": str(document_id),
            "start_position": start,
            "max_k": max_k,
            "proposal_token_ids": proposed,
            "canonical_token_ids": continuation,
            "draft_confidence": confidences,
            "draft_time_ms": draft_ms,
            "draft_time_by_k_ms": draft_time_by_k,
            "verification_time_by_k_ms": verification_time_by_k,
            "autoregressive_time_ms": autoregressive_time_ms,
            "verification_time_ms": (
                verification_time_by_k[-1] if verification_time_by_k else None
            ),
            "timing_basis": (
                "measured_cached_target_check"
                if verification_time_by_k
                else "draft_only_no_target_check"
            ),
            "drift_at_start": drift,
            "grounding_horizon": horizon,
            "target_entropy_at_start": (
                float(target_entropy[start])
                if start < len(target_entropy) else None
            ),
            "source_concentration_at_start": (
                max(target_trace[start]) if start < len(target_trace) else None
            ),
            "recent_acceptance": (
                sum(acceptance_history) / len(acceptance_history)
                if acceptance_history else 0.0
            ),
            **result,
        })
        acceptance_history.append(
            float(result["accepted_len"]) / max(float(max_k), 1.0)
        )
    return results


def _read_trace_rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            result[str(row["sample_id"])] = row
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--verification-model", default=None)
    parser.add_argument("--input", required=True)
    parser.add_argument("--target-traces", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument(
        "--sample-offset", type=int, default=0,
        help="Skip this many input records before applying --max-samples",
    )
    parser.add_argument(
        "--sample-ids", default="",
        help="Optional comma-separated record IDs; takes precedence over offset/limit",
    )
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--max-starts", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    if args.sample_offset < 0:
        raise ValueError("sample_offset must be non-negative")
    all_records = load_jsonl(Path(args.input), limit=0)
    selected_ids = {
        item.strip() for item in str(args.sample_ids).split(",") if item.strip()
    }
    if selected_ids:
        records = [
            record for record in all_records
            if str(record.get("id")) in selected_ids
        ]
    else:
        records = all_records[args.sample_offset:]
        if args.max_samples:
            records = records[:args.max_samples]
    traces = _read_trace_rows(Path(args.target_traces))
    model, tokenizer, device = load_local_model(
        args.draft_model, device=args.device, dtype=args.dtype
    )
    verification_model = verification_tokenizer = verification_device = None
    if args.verification_model:
        verification_model, verification_tokenizer, verification_device = load_local_model(
            args.verification_model, device=args.device, dtype=args.dtype
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    output_rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sample_id = str(record.get("id", index))
        try:
            trace = traces[sample_id]
            rendered = render_document_prompt(tokenizer, _record_document(record))
            verification_rendered = None
            if verification_tokenizer is not None:
                verification_rendered = render_document_prompt(
                    verification_tokenizer, _record_document(record)
                )
            output_rows.extend(
                run_one_speculative_trace(
                    model,
                    tokenizer,
                    rendered,
                    trace,
                    document_id=sample_id,
                    max_k=args.max_k,
                    max_starts=args.max_starts,
                    stride=args.stride,
                    start_offset=args.start_offset,
                    device=device,
                    prefill_chunk_size=args.prefill_chunk_size,
                    verification_model=verification_model,
                    verification_rendered=verification_rendered,
                    verification_device=verification_device,
                )
            )
        except Exception as exc:
            output_rows.append({
                "schema_version": "groundsync.spec.v1",
                "status": "error",
                "document_id": sample_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
        print(f"spec_trace {index + 1}/{len(records)} sample={sample_id}", flush=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps({
            "schema_version": "groundsync.spec.manifest.v1",
            "draft_model": args.draft_model,
            "verification_model": args.verification_model,
            "device": str(device),
            "input": str(Path(args.input)),
            "target_traces": str(Path(args.target_traces)),
            "output": str(output_path),
            "requested_samples": len(records),
            "sample_offset": args.sample_offset,
            "sample_ids": sorted(selected_ids),
            "ok_rows": sum(row.get("status") == "ok" for row in output_rows),
            "error_rows": sum(row.get("status") != "ok" for row in output_rows),
            "max_k": args.max_k,
            "max_starts": args.max_starts,
            "start_offset": args.start_offset,
            "stride": args.stride,
            "prefill_chunk_size": args.prefill_chunk_size,
            "timing_basis": (
                "measured_cached_target_check"
                if args.verification_model else "draft_only_no_target_check"
            ),
            "elapsed_s": time.perf_counter() - started,
            "seed": args.seed,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
