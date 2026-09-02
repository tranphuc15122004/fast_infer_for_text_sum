"""Canonical target tracing for GroundSync.

This module keeps model-specific work at the boundary.  The trace records only
one source-attention vector per generated token, rather than materializing or
writing a full attention matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import aggregate_source_mass


DEFAULT_INSTRUCTION = (
    "Summarize the following document faithfully and concisely. "
    "Return only the summary.\n\nDocument:\n"
)
ALLOWED_QWEN3_MARKERS = ("qwen3-4b", "qwen3-1.7b", "qwen3-0.6b")


@dataclass(frozen=True)
class RenderedPrompt:
    """Tokenized chat prompt and the source span inside it."""

    prompt: str
    input_ids: Any
    source_start: int
    source_end: int
    instruction_prefix: str


def locate_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    """Return the first index of ``needle`` in ``haystack``."""

    if not needle:
        return 0
    width = len(needle)
    for start in range(len(haystack) - width + 1):
        if list(haystack[start : start + width]) == list(needle):
            return start
    return None


def _tokenizer_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    while encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(token) for token in encoded]


def _chat_input_ids(tokenizer: Any, content: str) -> Any:
    messages = [{"role": "user", "content": content}]
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": False,
    }
    try:
        encoded = tokenizer.apply_chat_template(
            messages, enable_thinking=False, **kwargs
        )
    except (TypeError, ValueError):
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if not hasattr(encoded, "shape"):
        import torch

        encoded = torch.tensor([encoded], dtype=torch.long)
    if len(encoded.shape) == 1:
        encoded = encoded.unsqueeze(0)
    return encoded


def render_document_prompt(
    tokenizer: Any,
    document: str,
    *,
    instruction_prefix: str = DEFAULT_INSTRUCTION,
) -> RenderedPrompt:
    """Render one document and locate its exact token span in the chat prompt."""

    document = str(document)
    if not document.strip():
        raise ValueError("document must not be empty")
    content = instruction_prefix + document
    input_ids = _chat_input_ids(tokenizer, content)
    prompt_ids = [int(value) for value in input_ids[0].tolist()]
    # Locate the tokenization in context first.  Standalone document token IDs
    # can differ at a whitespace/BPE boundary, while the content sequence is
    # exactly what was passed to the chat template.
    content_ids = _tokenizer_ids(tokenizer, content)
    content_start = locate_subsequence(prompt_ids, content_ids)
    if content_start is not None:
        prefix_ids = _tokenizer_ids(tokenizer, instruction_prefix)
        source_start = content_start + len(prefix_ids)
        source_end = content_start + len(content_ids)
    else:
        document_ids = _tokenizer_ids(tokenizer, document)
        source_start = locate_subsequence(prompt_ids, document_ids)
        if source_start is None:
            raise ValueError("could not locate document token span in chat template")
        source_end = source_start + len(document_ids)
    if source_end > len(prompt_ids):
        raise ValueError("document token span exceeds rendered prompt")
    return RenderedPrompt(
        prompt=content,
        input_ids=input_ids,
        source_start=source_start,
        source_end=source_end,
        instruction_prefix=instruction_prefix,
    )


def _attention_array(attention: Any) -> Any:
    """Convert a layer attention object to a CPU numpy array."""

    import numpy as np

    if hasattr(attention, "detach"):
        attention = attention.detach().float().cpu().numpy()
    return np.asarray(attention, dtype=float)


def _collapse_attention_layers(attentions: Sequence[Any]) -> list[float]:
    """Average heads/layers and select the last query for each layer."""

    import numpy as np

    vectors: list[Any] = []
    for layer in attentions:
        if layer is None:
            continue
        array = _attention_array(layer)
        if array.ndim == 4:  # batch, heads, query, key
            vector = array[0, :, -1, :].mean(axis=0)
        elif array.ndim == 3:  # heads, query, key
            vector = array[:, -1, :].mean(axis=0)
        elif array.ndim == 2:  # query, key for a single head
            vector = array[-1, :]
        elif array.ndim == 1:
            vector = array
        else:
            raise ValueError(f"unsupported attention rank: {array.ndim}")
        vectors.append(vector)
    if not vectors:
        raise ValueError("model returned no attention tensors")
    result = np.mean(np.stack(vectors, axis=0), axis=0)
    return [float(value) for value in result.tolist()]


def attention_to_source_distribution(
    attentions: Sequence[Any],
    *,
    source_start: int,
    source_end: int,
    chunk_size: int,
    skip_source_tokens: int = 0,
    positional_prior: Sequence[float] | None = None,
    sensitivity_chunk_sizes: Sequence[int] | None = None,
    sink_sizes: Sequence[int] | None = None,
) -> dict[str, list[float]]:
    """Return raw and sink-controlled source-chunk distributions.

    The default preserves the compact legacy schema.  When sensitivity sizes
    or sink sizes are supplied, the result additionally contains explicitly
    named variants such as ``raw_chunk_64`` and ``nosink_8_chunk_128``.
    """

    if source_start < 0 or source_end <= source_start:
        raise ValueError("source span must be non-empty and ordered")
    mass = _collapse_attention_layers(attentions)
    if source_end > len(mass):
        raise ValueError("source span exceeds attention key length")
    source_mass = mass[source_start:source_end]
    def aggregate(chunk: int, skip: int) -> list[float]:
        effective_skip = min(max(int(skip), 0), len(source_mass) - 1)
        prior = (
            positional_prior[effective_skip:]
            if positional_prior is not None
            else None
        )
        return aggregate_source_mass(
            source_mass,
            chunk_size=chunk,
            skip_tokens=effective_skip,
            positional_prior=prior,
        )

    if sensitivity_chunk_sizes is None and sink_sizes is None:
        return {
            "raw": aggregate_source_mass(
                source_mass,
                chunk_size=chunk_size,
                positional_prior=positional_prior,
            ),
            "nosink": aggregate(chunk_size, skip_source_tokens),
        }

    sizes = tuple(dict.fromkeys(
        [chunk_size] + [int(value) for value in (sensitivity_chunk_sizes or ())]
    ))
    sinks = tuple(dict.fromkeys(
        [skip_source_tokens] + [int(value) for value in (sink_sizes or ())]
    ))
    result: dict[str, list[float]] = {}
    for size in sizes:
        result[f"raw_chunk_{size}"] = aggregate_source_mass(
            source_mass,
            chunk_size=size,
            positional_prior=positional_prior,
        )
        for sink in sinks:
            result[f"nosink_{sink}_chunk_{size}"] = aggregate(size, sink)
    result["raw"] = result[f"raw_chunk_{chunk_size}"]
    result["nosink"] = result[f"nosink_{skip_source_tokens}_chunk_{chunk_size}"]
    return result


def _model_call(model: Any, **kwargs: Any) -> Any:
    try:
        return model(logits_to_keep=1, **kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        return model(**kwargs)


def _set_attention_implementation(model: Any, implementation: str) -> bool:
    """Best-effort switch for models exposing Transformers' attention setter."""

    setter = getattr(model, "set_attn_implementation", None)
    if setter is None:
        return False
    try:
        setter(implementation)
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        return False
    return True


def _bottom_right_causal_mask(
    query_length: int,
    past_length: int,
    *,
    dtype: Any,
    device: Any,
) -> Any:
    """Build a causal mask aligned to the bottom-right of a KV cache."""

    import torch

    if query_length <= 0 or past_length < 0:
        raise ValueError("query_length must be positive and past_length non-negative")
    key_length = past_length + query_length
    rows = torch.arange(query_length, device=device).unsqueeze(1) + past_length
    columns = torch.arange(key_length, device=device).unsqueeze(0)
    mask = torch.zeros((1, 1, query_length, key_length), dtype=dtype, device=device)
    mask.masked_fill_(columns > rows, float("-inf"))
    return mask


def _chunked_prefill(
    model: Any,
    input_ids: Any,
    *,
    prefill_chunk_size: int,
) -> Any:
    """Prefill a long prompt without materializing a full L x L attention map."""

    import torch

    if prefill_chunk_size <= 0:
        raise ValueError("prefill_chunk_size must be positive")
    model_dtype = getattr(model, "dtype", torch.float32)
    if not isinstance(model_dtype, torch.dtype):
        model_dtype = torch.float32
    switched_attention = _set_attention_implementation(model, "sdpa")
    past_key_values = None
    last_outputs = None
    try:
        for start in range(0, input_ids.shape[1], prefill_chunk_size):
            end = min(start + prefill_chunk_size, input_ids.shape[1])
            chunk = input_ids[:, start:end]
            attention_mask = _bottom_right_causal_mask(
                end - start,
                start,
                dtype=model_dtype,
                device=input_ids.device,
            )
            kwargs = {
                "input_ids": chunk,
                "use_cache": True,
                "return_dict": True,
                "output_attentions": False,
                "attention_mask": attention_mask,
            }
            if past_key_values is not None:
                kwargs["past_key_values"] = past_key_values
            outputs = _model_call(model, **kwargs)
            last_outputs = outputs
            past_key_values = outputs.past_key_values
    finally:
        if switched_attention:
            _set_attention_implementation(model, "eager")
    return last_outputs


def _entropy(logits: Any) -> float:
    import torch

    probabilities = torch.softmax(logits.float(), dim=-1)
    value = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("target entropy is non-finite")
    return result


def _sync_cuda(torch: Any, device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        torch.cuda.synchronize(device)


def generate_target_trace(
    model: Any,
    tokenizer: Any,
    rendered: RenderedPrompt,
    *,
    sample_id: str,
    document_id: str,
    max_new_tokens: int,
    chunk_size: int,
    skip_source_tokens: int,
    device: Any,
    positional_prior: Sequence[float] | None = None,
    prefill_chunk_size: int = 512,
    sensitivity_chunk_sizes: Sequence[int] | None = None,
    sink_sizes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Generate a deterministic target trace with incremental attention."""

    import torch

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    input_ids = rendered.input_ids.to(device)
    source_start = rendered.source_start
    source_end = rendered.source_end
    if input_ids.shape[1] <= 1:
        raise ValueError("rendered prompt must contain at least two tokens")

    with torch.inference_mode():
        # Build a cache without the last prompt token, then pass that token as
        # an incremental query. This avoids asking eager attention for an
        # O(L^2) prompt matrix while still measuring the query predicting y_1.
        prefix = input_ids[:, :-1]
        current = input_ids[:, -1:]
        prefill = _chunked_prefill(
            model,
            prefix,
            prefill_chunk_size=prefill_chunk_size,
        )
        past_key_values = prefill.past_key_values
        generated: list[int] = []
        entropies: list[float] = []
        sentence_boundaries: list[int] = []
        copyability: list[int] = []
        attention_rows: list[dict[str, list[float]]] = []
        eos_ids = set()
        source_token_ids = {
            int(value) for value in input_ids[0, source_start:source_end].tolist()
        }
        if tokenizer.eos_token_id is not None:
            eos_ids.add(int(tokenizer.eos_token_id))

        for _ in range(max_new_tokens):
            outputs = _model_call(
                model,
                input_ids=current,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=True,
            )
            logits = outputs.logits[:, -1, :]
            entropies.append(_entropy(logits[0]))
            attention_rows.append(
                attention_to_source_distribution(
                    outputs.attentions,
                    source_start=source_start,
                    source_end=source_end,
                    chunk_size=chunk_size,
                    skip_source_tokens=min(skip_source_tokens, source_end - source_start - 1),
                    positional_prior=positional_prior,
                    sensitivity_chunk_sizes=sensitivity_chunk_sizes,
                    sink_sizes=sink_sizes,
                )
            )
            next_token = logits.argmax(dim=-1, keepdim=True)
            token_id = int(next_token[0, 0].item())
            generated.append(token_id)
            try:
                token_text = tokenizer.decode(
                    [token_id], clean_up_tokenization_spaces=False
                )
            except (AttributeError, TypeError):
                token_text = ""
            sentence_boundaries.append(
                int(any(mark in token_text for mark in (".", "!", "?")))
            )
            copyability.append(int(token_id in source_token_ids))
            past_key_values = outputs.past_key_values
            current = next_token
            if token_id in eos_ids:
                break

    return {
        "schema_version": "groundsync.target.v1",
        "status": "ok",
        "sample_id": str(sample_id),
        "document_id": str(document_id),
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": len(generated),
        "source_start": source_start,
        "source_end": source_end,
        "source_token_ids": [
            int(value) for value in input_ids[0, source_start:source_end].tolist()
        ],
        "chunk_size": chunk_size,
        "skip_source_tokens": skip_source_tokens,
        "generated_token_ids": generated,
        "target_entropy": entropies,
        "sentence_boundary": sentence_boundaries,
        "copyability": copyability,
        "attention": attention_rows,
    }


def _record_document(record: Mapping[str, Any]) -> str:
    for key in ("document", "context", "input", "prompt", "text"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    turns = record.get("turns")
    if isinstance(turns, list) and turns and isinstance(turns[0], str):
        return turns[0]
    raise ValueError("record has no usable document/prompt field")


def load_jsonl(path: Path, *, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        _record_document(value)
        rows.append(value)
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"no usable records in {path}")
    return rows


def load_local_model(model_name: str, *, device: str, dtype: str = "auto") -> tuple[Any, Any, Any]:
    """Load a local Transformers model; never attempts a network download."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    normalized_name = str(model_name).lower().replace("_", "-")
    if "eagle" in normalized_name or not any(
        marker in normalized_name for marker in ALLOWED_QWEN3_MARKERS
    ):
        raise ValueError(
            "model must be one of Qwen3-4B, Qwen3-1.7B, Qwen3-0.6B; "
            "EAGLE heads and other model families are not valid here"
        )

    resolved_device = torch.device(device)
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        local_files_only=True,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(resolved_device).eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, resolved_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--skip-source-tokens", type=int, default=8)
    parser.add_argument("--sensitivity-chunk-sizes", default="64,128,256")
    parser.add_argument("--sink-sizes", default="4,8,16")
    parser.add_argument("--prefill-chunk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("integer list must contain positive values")
    return values


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    rows = load_jsonl(Path(args.input), limit=args.max_samples)
    model, tokenizer, device = load_local_model(args.model, device=args.device, dtype=args.dtype)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    output_rows: list[dict[str, Any]] = []
    for index, record in enumerate(rows):
        sample_id = str(record.get("id", index))
        try:
            document = _record_document(record)
            rendered = render_document_prompt(tokenizer, document)
            output_rows.append(
                generate_target_trace(
                    model,
                    tokenizer,
                    rendered,
                    sample_id=sample_id,
                    document_id=sample_id,
                    max_new_tokens=args.max_new_tokens,
                    chunk_size=args.chunk_size,
                    skip_source_tokens=args.skip_source_tokens,
                    device=device,
                    prefill_chunk_size=args.prefill_chunk_size,
                    sensitivity_chunk_sizes=_parse_int_list(args.sensitivity_chunk_sizes),
                    sink_sizes=_parse_int_list(args.sink_sizes),
                )
            )
        except Exception as exc:  # per-document isolation for long runs
            output_rows.append({
                "schema_version": "groundsync.target.v1",
                "status": "error",
                "sample_id": sample_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
        print(f"target_trace {index + 1}/{len(rows)} sample={sample_id}", flush=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": "groundsync.target.manifest.v1",
        "model": args.model,
        "device": str(device),
        "input": str(Path(args.input)),
        "output": str(output_path),
        "requested_samples": len(rows),
        "ok_samples": sum(row.get("status") == "ok" for row in output_rows),
        "chunk_size": args.chunk_size,
        "skip_source_tokens": args.skip_source_tokens,
        "sensitivity_chunk_sizes": _parse_int_list(args.sensitivity_chunk_sizes),
        "sink_sizes": _parse_int_list(args.sink_sizes),
        "prefill_chunk_size": args.prefill_chunk_size,
        "elapsed_s": time.perf_counter() - started,
        "seed": args.seed,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
