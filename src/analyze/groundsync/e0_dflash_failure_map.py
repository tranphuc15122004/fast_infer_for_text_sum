#!/usr/bin/env python3
"""E0 Target-KV proposal: DFlash acceptance failure map.

This runner is intentionally separate from the older GroundSync traces.  It
uses the official Qwen3-4B/DFlash pair, runs several native block sizes on the
same prompt, and writes one auditable run directory under ``groundsync``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common.benchmark_data import render_prompt  # noqa: E402
from common.paths import ROOT as SCRIPT_ROOT  # noqa: E402

from .target_kv_experiments import (  # noqa: E402
    aggregate_e0_metrics,
    apply_input_length_limit,
    context_bucket,
    flatten_dflash_rounds,
    prepare_record_metadata,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    """Flush one row immediately so a long GPU run remains resumable/auditable."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prompt_from_record(record: Mapping[str, Any]) -> str:
    if record.get("dataset") in {"gov_report", "qmsum", "multi_news", "lcc", "repobench-p"}:
        return render_prompt(record)
    for key in ("prompt", "document", "text", "instruction", "question"):
        value = record.get(key)
        if value:
            return str(value)
    raise ValueError("record has no usable prompt")


def _chat_prompt(tokenizer: Any, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


class SelectiveHiddenTarget:
    """Proxy target that captures only DFlash's requested hidden layers.

    Transformers normally materializes every layer when
    ``output_hidden_states=True``.  That peak is too large for a 15GB T4 at
    long context.  Hooks preserve the DFlash contract while asking the target
    model to return only its normal output.
    """

    def __init__(self, target: Any, layer_ids: Sequence[int]) -> None:
        self._target = target
        self._layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if not self._layer_ids:
            raise ValueError("layer_ids must not be empty")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not kwargs.get("output_hidden_states", False):
            return self._target(*args, **kwargs)
        layers = getattr(getattr(self._target, "model", None), "layers", None)
        if layers is None:
            raise AttributeError("target.model.layers is required for selective capture")
        captured: dict[int, Any] = {}
        handles = []

        def make_hook(layer_id: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                captured[layer_id] = output[0] if isinstance(output, tuple) else output

            return hook

        for layer_id in self._layer_ids:
            handles.append(layers[layer_id].register_forward_hook(make_hook(layer_id)))
        call_kwargs = dict(kwargs)
        call_kwargs["output_hidden_states"] = False
        try:
            output = self._target(*args, **call_kwargs)
        finally:
            for handle in handles:
                handle.remove()
        missing = [layer_id for layer_id in self._layer_ids if layer_id not in captured]
        if missing:
            raise RuntimeError(f"target did not expose requested layers: {missing}")
        hidden_states = [None] * (max(self._layer_ids) + 2)
        for layer_id in self._layer_ids:
            hidden_states[layer_id + 1] = captured[layer_id]
        output.hidden_states = tuple(hidden_states)
        return output


def _encode_prompt(tokenizer: Any, prompt: str, *, device: Any) -> Any:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids
    if len(input_ids.shape) != 2 or input_ids.shape[0] != 1:
        raise ValueError("tokenizer must return one [1, sequence] input")
    return input_ids.to(device)


def apply_input_cap(input_ids: Any, *, max_tokens: int) -> Any:
    """Apply an explicit smoke cap to the tensor/list actually sent to CUDA."""

    if max_tokens <= 0:
        return input_ids
    if hasattr(input_ids, "shape") and len(input_ids.shape) == 2:
        return input_ids[:, :max_tokens]
    return input_ids[:max_tokens]


def choose_smoke_input_cap(smoke: bool, configured_cap: int) -> int:
    """Return the configured smoke cap, or zero when running the main set."""

    if configured_cap <= 0 and smoke:
        raise ValueError("smoke cap must be positive")
    return int(configured_cap) if smoke else 0


def run_inference_safe(function: Any) -> Any:
    """Run one generation without retaining an autograd graph."""

    import torch

    with torch.inference_mode():
        return function()


def release_cuda_cache(torch: Any) -> None:
    """Release allocator blocks between independent E0 K measurements."""

    cuda = getattr(torch, "cuda", None)
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


def select_raw_rows(
    rows: Sequence[dict[str, Any]],
    *,
    start_index: int = 0,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Select a deterministic contiguous slice of source rows."""

    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    selected = list(rows[start_index:])
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        selected = selected[:max_samples]
    return selected


def chunk_spans(sequence_length: int, chunk_size: int) -> list[tuple[int, int]]:
    """Return contiguous, non-overlapping prefill spans."""

    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [
        (start, min(start + chunk_size, sequence_length))
        for start in range(0, sequence_length, chunk_size)
    ]


def _load_rows(
    path: Path,
    tokenizer: Any,
    *,
    start_index: int,
    max_samples: int | None,
    max_position_embeddings: int,
    smoke: bool,
    smoke_input_tokens: int,
    input_token_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_rows = _jsonl(path)
    raw_rows = select_raw_rows(raw_rows, start_index=start_index, max_samples=max_samples)
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        sample_id = str(raw.get("id", index))
        try:
            content = _prompt_from_record(raw)
            rendered = _chat_prompt(tokenizer, content)
            input_ids = _encode_prompt(tokenizer, rendered, device="cpu")
            input_tokens = int(input_ids.shape[1])
            metadata = prepare_record_metadata(
                {"id": sample_id, "dataset": raw.get("dataset", "unknown"), "input_tokens": input_tokens},
                max_position_embeddings=max_position_embeddings,
            )
            metadata = apply_input_length_limit(metadata, input_token_limit)
            metadata.update(
                {
                    "prompt": content,
                    "prompt_sha256": _sha256_text(rendered),
                    "source_index": raw.get("source_index", index),
                }
            )
            smoke_cap = choose_smoke_input_cap(smoke, smoke_input_tokens)
            if metadata["status"] == "ok" and smoke_cap and input_tokens > smoke_cap:
                metadata.update(
                    status="smoke_truncated",
                    original_input_tokens=input_tokens,
                    input_tokens=smoke_cap,
                    context_bucket=context_bucket(smoke_cap),
                    smoke_input_cap=smoke_cap,
                    exclusion_reason="smoke_input_cap",
                )
            if metadata["status"] in {"ok", "smoke_truncated"}:
                accepted.append(metadata)
            else:
                excluded.append(metadata)
        except Exception as exc:  # keep one malformed row from hiding coverage
            excluded.append(
                {
                    "sample_id": sample_id,
                    "status": "excluded",
                    "exclusion_reason": "row_preparation_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return accepted, excluded


def _dtype_and_attention(torch: Any) -> tuple[Any, str]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; E0 requires the T4 host runtime")
    capability = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return dtype, "sdpa"
    return dtype, "flash_attention_2" if capability[0] >= 8 else "sdpa"


def _run_one(
    *,
    target: Any,
    draft: Any,
    dflash_generate: Any,
    input_ids: Any,
    block_size: int,
    max_new_tokens: int,
    temperature: float,
    prefill_chunk_size: int,
    torch: Any,
) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    eos_id = target.config.eos_token_id
    stop_ids = eos_id if isinstance(eos_id, list) else [eos_id]
    result = run_inference_safe(
        lambda: memory_safe_dflash_generate(
            draft,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_ids,
            temperature=temperature,
            block_size=block_size,
            prefill_chunk_size=prefill_chunk_size,
            return_stats=True,
        )
    )
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


def memory_safe_dflash_generate(
    model: Any,
    *,
    target: Any,
    input_ids: Any,
    max_new_tokens: int,
    stop_token_ids: Sequence[int] | None,
    temperature: float,
    block_size: int,
    prefill_chunk_size: int,
    return_stats: bool = False,
) -> Any:
    """DFlash generation with chunked target prefill and selective hidden states."""

    import copy
    from types import SimpleNamespace

    import torch
    from transformers import DynamicCache

    from dflash.model import extract_context_feature, sample
    from .trace_target import _bottom_right_causal_mask

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    input_length = int(input_ids.shape[1])
    max_length = input_length + max_new_tokens
    output_ids = torch.full(
        (1, max_length + block_size),
        target.config.dflash_config.get("mask_token_id", None)
        if hasattr(target.config, "dflash_config")
        else 0,
        dtype=torch.long,
        device=input_ids.device,
    )
    mask_token_id = getattr(target, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(model, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = int(getattr(target.config, "pad_token_id", 0) or 0)
    output_ids.fill_(int(mask_token_id))
    position_ids = torch.arange(output_ids.shape[1], device=input_ids.device).unsqueeze(0)
    target_cache = DynamicCache()
    draft_cache = DynamicCache()
    layer_ids = list(getattr(model, "target_layer_ids"))
    selected_chunks: list[list[Any]] = [[] for _ in layer_ids]
    prefill_started = time.perf_counter()
    last_output = None
    model_dtype = getattr(target, "dtype", torch.float16)
    for start, end in chunk_spans(input_length, prefill_chunk_size):
        attention_mask = _bottom_right_causal_mask(
            end - start,
            start,
            dtype=model_dtype,
            device=input_ids.device,
        )
        last_output = target(
            input_ids=input_ids[:, start:end],
            position_ids=position_ids[:, start:end],
            past_key_values=target_cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=block_size > 1,
            attention_mask=attention_mask,
        )
        target_cache = last_output.past_key_values
        if block_size > 1:
            for index, layer_id in enumerate(layer_ids):
                selected_chunks[index].append(last_output.hidden_states[layer_id + 1])
    if last_output is None:
        raise ValueError("input_ids must contain at least one token")
    output_ids[:, :input_length] = input_ids
    output_ids[:, input_length : input_length + 1] = sample(
        last_output.logits, temperature
    )
    if block_size > 1:
        # The concatenated chunks are already in the requested layer order.
        target_hidden = torch.cat(
            [torch.cat(chunks, dim=1) for chunks in selected_chunks], dim=-1
        )
    else:
        target_hidden = None
    time_to_first_token = time.perf_counter() - prefill_started
    decode_started = time.perf_counter()
    acceptance_lengths: list[int] = []
    start = input_length
    draft_prefill = True
    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(
                model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, draft_cache.get_seq_length() : start + block_size],
                    past_key_values=draft_cache,
                    use_cache=True,
                    is_causal=False,
                )[:, 1 - block_size :, :]
            )
            draft_cache.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits, temperature)
            if draft_prefill:
                draft_prefill = False
                decode_started = time.perf_counter()
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=target_cache,
            use_cache=True,
            output_hidden_states=block_size > 1,
        )
        posterior = sample(output.logits, temperature)
        accepted = (
            block_output_ids[:, 1:] == posterior[:, :-1]
        ).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + accepted + 1] = block_output_ids[:, : accepted + 1]
        output_ids[:, start + accepted + 1] = posterior[:, accepted]
        start += accepted + 1
        target_cache.crop(start)
        acceptance_lengths.append(int(accepted + 1))
        if block_size > 1:
            target_hidden = extract_context_feature(
                output.hidden_states, layer_ids
            )[:, : accepted + 1, :]
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, input_length:] for stop_token_id in stop_token_ids
        ):
            break
    output_ids = output_ids[:, : min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_ids = torch.tensor(list(stop_token_ids), device=output_ids.device)
        indices = torch.isin(output_ids[0][input_length:], stop_ids).nonzero(as_tuple=True)[0]
        if indices.numel() > 0:
            output_ids = output_ids[:, : input_length + int(indices[0]) + 1]
    if not return_stats:
        return output_ids
    num_output_tokens = output_ids.shape[1] - input_length
    total_decode_time = time.perf_counter() - decode_started
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=input_length,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / max(num_output_tokens, 1),
        acceptance_lengths=acceptance_lengths,
    )


def _hardware(torch: Any) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_gb": round(properties.total_memory / 1024**3, 3),
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "python": sys.version,
    }


def run_e0(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "experiment": "E0_target_kv_dflash_failure_map",
        "status": "RUNNING",
        "target_model": str(args.target_model),
        "draft_model": str(args.draft_model),
        "data_file": str(args.data_file),
        "candidate_ks": list(args.candidate_ks),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "max_position_embeddings": args.max_position_embeddings,
        "smoke": bool(args.smoke),
        "command": " ".join(sys.argv),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    try:
        dtype, attention = _dtype_and_attention(torch)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dflash_root = ROOT / "externals" / "dflash"
        if str(dflash_root) not in sys.path:
            sys.path.insert(0, str(dflash_root))
        from dflash.model import DFlashDraftModel, dflash_generate

        target_kwargs = {
            "dtype": dtype,
            "attn_implementation": attention,
            "low_cpu_mem_usage": True,
            "local_files_only": True,
        }
        if args.target_load_in_8bit:
            from transformers import BitsAndBytesConfig

            target_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            target_kwargs["device_map"] = {"": "cuda:0"}
            target = AutoModelForCausalLM.from_pretrained(
                args.target_model, **target_kwargs
            ).eval()
        else:
            target = AutoModelForCausalLM.from_pretrained(
                args.target_model, **target_kwargs
            ).to("cuda:0").eval()
        draft = DFlashDraftModel.from_pretrained(
            args.draft_model,
            dtype=dtype,
            attn_implementation=attention,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to("cuda:0").eval()
        target_for_dflash = SelectiveHiddenTarget(target, draft.target_layer_ids)
        tokenizer = AutoTokenizer.from_pretrained(
            args.target_model, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_position_embeddings = int(
            getattr(target.config, "max_position_embeddings", args.max_position_embeddings)
        )
        records, excluded = _load_rows(
            Path(args.data_file),
            tokenizer,
            start_index=args.start_index,
            max_samples=args.max_samples,
            max_position_embeddings=max_position_embeddings,
            smoke=args.smoke,
            smoke_input_tokens=args.smoke_input_tokens,
            input_token_limit=args.input_token_limit,
        )
        raw_rows: list[dict[str, Any]] = []
        round_rows: list[dict[str, Any]] = []
        raw_path = output_dir / "dflash_records.jsonl"
        rounds_path = output_dir / "round_records.jsonl"
        raw_path.unlink(missing_ok=True)
        rounds_path.unlink(missing_ok=True)
        for record_index, record in enumerate(records):
            input_ids = _encode_prompt(
                tokenizer, _chat_prompt(tokenizer, record["prompt"]), device="cuda:0"
            )
            input_ids = apply_input_cap(
                input_ids, max_tokens=int(record.get("smoke_input_cap", 0))
            )
            base_result, base_elapsed = _run_one(
                target=target_for_dflash,
                draft=draft,
                dflash_generate=dflash_generate,
                input_ids=input_ids,
                block_size=1,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                prefill_chunk_size=args.prefill_chunk_size,
                torch=torch,
            )
            base_output = base_result.output_ids[0, input_ids.shape[1] :].tolist()
            base_timing = {
                "baseline_e2e_ms": base_elapsed * 1000.0,
                "baseline_output_tokens": len(base_output),
            }
            del base_result
            release_cuda_cache(torch)
            for k in args.candidate_ks:
                torch.cuda.reset_peak_memory_stats()
                torch.manual_seed(args.seed)
                try:
                    result, elapsed = _run_one(
                        target=target_for_dflash,
                        draft=draft,
                        dflash_generate=dflash_generate,
                        input_ids=input_ids,
                        block_size=k,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        prefill_chunk_size=args.prefill_chunk_size,
                        torch=torch,
                    )
                    output = result.output_ids[0, input_ids.shape[1] :].tolist()
                    row = {
                        **record,
                        **base_timing,
                        "status": "ok",
                        "block_size": k,
                        "output_tokens": len(output),
                        "output_token_ids": [int(token) for token in output],
                        "baseline_token_ids": [int(token) for token in base_output],
                        "acceptance_lengths": [int(v) for v in result.acceptance_lengths],
                        "e2e_ms": elapsed * 1000.0,
                        "time_to_first_token_ms": float(result.time_to_first_token) * 1000.0,
                        "time_per_output_token_ms": float(result.time_per_output_token) * 1000.0,
                        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
                        "exact_match_target_ar": output == base_output,
                        "seed": args.seed,
                        "record_index": record_index,
                    }
                    raw_rows.append(row)
                    new_rounds = flatten_dflash_rounds(row)
                    round_rows.extend(new_rounds)
                    append_jsonl(raw_path, row)
                    for round_row in new_rounds:
                        append_jsonl(rounds_path, round_row)
                    del result
                    release_cuda_cache(torch)
                except Exception as exc:
                    error_row = {
                            **record,
                            "status": "error",
                            "block_size": k,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=3),
                        }
                    raw_rows.append(error_row)
                    append_jsonl(raw_path, error_row)
                    if "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                    release_cuda_cache(torch)
        metrics = aggregate_e0_metrics(
            round_rows,
            candidate_ks=args.candidate_ks,
            bootstrap_samples=args.bootstrap_samples,
        ) if round_rows else {"status": "UNAVAILABLE", "reason": "no_successful_rounds"}
        manifest.update(
            {
                "status": "ok",
                "hardware": _hardware(torch),
                "dtype": str(dtype),
                "attention_implementation": attention,
                "target_load_in_8bit": bool(args.target_load_in_8bit),
                "input_token_limit": args.input_token_limit,
                "records_selected": len(records),
                "records_excluded": len(excluded),
                "successful_generation_rows": len(raw_rows),
                "round_rows": len(round_rows),
            }
        )
        _write_jsonl(output_dir / "exclusions.jsonl", excluded)
        # Rows were flushed after every K; rewriting them here keeps a stable
        # canonical ordering while preserving the partial-run recovery path.
        _write_jsonl(raw_path, raw_rows)
        _write_jsonl(rounds_path, round_rows)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "UNAVAILABLE",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=5),
            }
        )
        (output_dir / "metrics.json").write_text(
            json.dumps({"status": "UNAVAILABLE", "reason": manifest["reason"]}, indent=2),
            encoding="utf-8",
        )
    finally:
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-position-embeddings", type=int, default=40960)
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--smoke-input-tokens", type=int, default=1024)
    parser.add_argument("--input-token-limit", type=int, default=0,
                        help="exclude natural prompts above this cap; 0 = no cap")
    parser.add_argument("--target-load-in-8bit", action="store_true",
                        help="quantized feasibility run; not canonical FP16 evidence")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--candidate-ks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser


def main() -> None:
    manifest = run_e0(build_parser().parse_args())
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    if manifest.get("status") == "UNAVAILABLE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
