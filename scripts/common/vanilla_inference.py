"""Shared implementation for the Vanilla HF and Vanilla FA baselines."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Any

import torch

from common import io_util
from common.benchmark_runtime import (
    build_sample_record,
    measure_call,
    runtime_metadata,
)
from common.data_loader import load_records


def build_parser(default_backend: str, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        default=os.environ.get("LONG_BENCH_MODEL")
        or os.environ.get("MODEL_TARGET"),
    )
    parser.add_argument("--data-file", default=os.environ.get("LONG_BENCH_DATA_FILE"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument(
        "--device", default=os.environ.get("LONG_BENCH_DEVICE", "cuda")
    )
    parser.add_argument("--dtype", default=os.environ.get("LONG_BENCH_DTYPE", "bfloat16"))
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("LONG_BENCH_LOCAL_FILES_ONLY", "1") == "1",
    )
    parser.add_argument(
        "--attention-backend",
        choices=[default_backend],
        default=default_backend,
    )
    parser.add_argument("--run-id", default=os.environ.get("LONG_BENCH_RUN_ID"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True)
    return parser


def _dtype(name: str) -> torch.dtype:
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {name}") from exc


def _prompt_batch(tokenizer: Any, prompt: str, *, max_input_tokens: int) -> torch.Tensor:
    kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if max_input_tokens > 0:
        kwargs.update({"truncation": True, "max_length": max_input_tokens})
    encoded = tokenizer(prompt, **kwargs)
    return encoded.input_ids


def _generate(model: Any, input_ids: torch.Tensor, args: argparse.Namespace) -> Any:
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": False,
        "pad_token_id": model.generation_config.pad_token_id,
    }
    if args.temperature > 0:
        kwargs["temperature"] = args.temperature
    return model.generate(input_ids, **kwargs)


def _next_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    scores = logits[:, -1, :]
    if temperature > 0:
        probabilities = torch.softmax(scores / temperature, dim=-1)
        return torch.multinomial(probabilities, num_samples=1)
    return scores.argmax(dim=-1, keepdim=True)


def _is_eos(token: torch.Tensor, eos_token_id: int | list[int] | None) -> bool:
    if eos_token_id is None:
        return False
    eos_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
    return int(token.reshape(-1)[0]) in {int(value) for value in eos_ids}


def _timed_generate(
    model: Any,
    input_ids: torch.Tensor,
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Greedy cached decoding with explicit prefill/decode timings.

    ``generate()`` exposes only one end-to-end wall time.  The manual loop
    records the prefill and incremental decode phases needed for ESR/DSR.  A
    compatibility fallback keeps the script usable with older Transformers
    cache APIs, while honestly leaving unavailable phase timings as ``null``.
    """
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    request_start = time.perf_counter()
    attention_mask = torch.ones_like(input_ids, device=device)
    try:
        prefill_start = time.perf_counter()
        prefill = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        past = prefill.past_key_values
        next_token = _next_token(prefill.logits, args.temperature)
        generated = [next_token]
        eos_id = tokenizer.eos_token_id

        decode_start = time.perf_counter()
        if not _is_eos(next_token, eos_id):
            for _ in range(max(args.max_new_tokens - 1, 0)):
                attention_mask = torch.cat(
                    (attention_mask, torch.ones_like(next_token)), dim=1
                )
                step = model(
                    input_ids=next_token,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = step.past_key_values
                next_token = _next_token(step.logits, args.temperature)
                generated.append(next_token)
                if _is_eos(next_token, eos_id):
                    break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decode_ms = (time.perf_counter() - decode_start) * 1000.0
        output_ids = torch.cat([input_ids, *generated], dim=1)
        e2e_ms = (time.perf_counter() - request_start) * 1000.0
        peak_memory_gb = (
            torch.cuda.max_memory_allocated(device) / (1024**3)
            if device.type == "cuda"
            else None
        )
        output_tokens = len(generated)
        return output_ids, {
            "prefill_ms": round(prefill_ms, 3),
            "ttft_ms": round(prefill_ms, 3),
            "decode_ms": round(decode_ms, 3),
            "e2e_ms": round(e2e_ms, 3),
            "tpot_ms": round(decode_ms / max(output_tokens - 1, 1), 3),
            "peak_memory_gb": round(peak_memory_gb, 6)
            if peak_memory_gb is not None
            else None,
            "device": str(device),
        }
    except (AttributeError, IndexError, TypeError, ValueError):
        # Transformers 4.x and 5.x expose different cache classes/arguments.
        # Fall back to the stable public generate API rather than emitting a
        # partial record that looks like a valid phase measurement.
        output_ids, timing = measure_call(
            lambda: _generate(model, input_ids, args), device=device
        )
        timing.update(
            {
                "prefill_ms": None,
                "ttft_ms": None,
                "decode_ms": None,
                "tpot_ms": None,
            }
        )
        return output_ids, timing


def _load_model(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    if not args.model:
        raise SystemExit("--model or LONG_BENCH_MODEL/MODEL_TARGET is required")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use orchestrator smoke preflight on this host")

    if args.attention_backend == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "vanilla_fa requires the installed flash-attn wheel; no fallback is allowed"
            ) from exc

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=_dtype(args.dtype),
        attn_implementation=args.attention_backend,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    )
    model.to(device).eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def run(args: argparse.Namespace, *, method: str) -> int:
    if args.smoke:
        args.max_samples = 1
        args.max_new_tokens = min(args.max_new_tokens, 8)
    if not args.data_file:
        raise SystemExit("--data-file or LONG_BENCH_DATA_FILE is required")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    records = load_records(Path(args.data_file), args.max_samples)
    data_name = Path(args.data_file).stem

    load_start = time.perf_counter()
    model, tokenizer = _load_model(args, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_ms = round((time.perf_counter() - load_start) * 1000.0, 3)
    metadata = runtime_metadata()
    config = {
        "device": str(device),
        "gpu_name": metadata.get("gpu_name"),
        "dtype": args.dtype,
        "attention_backend": args.attention_backend,
        "seed": args.seed,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "warmup_runs": args.warmup_runs,
        "batch_size": 1,
    }

    with torch.inference_mode():
        warmup_ids = _prompt_batch(tokenizer, "Hello", max_input_tokens=0).to(device)
        for _ in range(max(args.warmup_runs, 0)):
            _generate(model, warmup_ids, args)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    writer = io_util.JsonlWriter(Path(args.output))
    successful = 0
    for sample in records:
        input_ids = _prompt_batch(
            tokenizer,
            sample["prompt"],
            max_input_tokens=max(args.max_input_tokens, 0),
        ).to(device)
        input_tokens = int(input_ids.shape[1])
        with torch.inference_mode():
            output_ids, timing = _timed_generate(
                model, input_ids, tokenizer, args, device
            )
        new_ids = output_ids[0, input_tokens:]
        output_tokens = int(new_ids.shape[0])
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        timing["model_load_ms"] = model_load_ms
        record = build_sample_record(
            method=method,
            dataset=sample.get("raw", {}).get("dataset", data_name),
            sample_id=sample["id"],
            model=str(args.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timing=timing,
            config=config,
            text=text,
            reference_output=sample.get("reference"),
        )
        record["run_id"] = args.run_id
        record["task_type"] = sample.get("raw", {}).get("task_type")
        writer.add(record)
        successful += 1
        print(
            f"[{method}][{record['dataset']}][{sample['id']}] "
            f"input={input_tokens} output={output_tokens} "
            f"e2e_ms={record['e2e_ms']} tok_s={record['throughput_tok_s']}"
        )

    summary = {
        "type": "summary",
        "method": method,
        "dataset": data_name,
        "run_id": args.run_id,
        "status": "success" if successful == len(records) else "failed",
        "num_samples": len(records),
        "successful_samples": successful,
        "model": args.model,
        "model_load_ms": model_load_ms,
        "attention_backend": args.attention_backend,
        "runtime": metadata,
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    return 0 if successful == len(records) else 1
