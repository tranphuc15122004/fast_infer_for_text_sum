"""Shared implementation for the Vanilla HF and Vanilla FA baselines."""

from __future__ import annotations

import argparse
import functools
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import time
import types
from typing import Any

import torch

from common import io_util, metrics, rouge
from common.benchmark_runtime import (
    build_sample_record,
    measure_call,
    runtime_metadata,
)
from common.data_loader import load_records
from common.input_utils import truncate_input_ids
from common.quality_guard import is_degenerate_output
from common.reproducibility import seed_everything


FLASH_ATTENTION_BACKENDS = frozenset(
    {"flash_attention_2", "flash_attention_4"}
)


def _install_flash_attention_4_cutlass_compat() -> bool:
    """Install the FA4 beta compatibility module without touching site-packages.

    ``flash-attn-4==4.0.0b15`` still imports the deprecated
    ``cutlass.utils.ampere_helpers`` module.  CUTLASS 4.5.2 removed that module
    but retained the only value used by the beta kernel: ``SMEM_CAPACITY``.
    Register the small compatibility module in ``sys.modules`` and adapt the
    legacy ``nvvm.fmax`` call convention for this process only, so the server
    environment and the FA2 installation remain unchanged.
    """

    changed = False
    module_name = "cutlass.utils.ampere_helpers"
    try:
        missing_ampere_helpers = importlib.util.find_spec(module_name) is None
    except Exception:
        missing_ampere_helpers = False

    if missing_ampere_helpers:
        try:
            cutlass_utils = importlib.import_module("cutlass.utils")
        except Exception:
            cutlass_utils = None
        if cutlass_utils is not None:
            shim = types.ModuleType(module_name)
            shim.SMEM_CAPACITY = {
                "sm80": 163840,
                "sm86": 102400,
                "sm89": 102400,
                "sm90": 232448,
                "sm100": 229376,
            }
            sys.modules[module_name] = shim
            setattr(cutlass_utils, "ampere_helpers", shim)
            changed = True

    # FA4 b15 calls nvvm.fmax directly with the optional third argument
    # positionally.  CUTLASS DSL 4.5.x made that argument keyword-only and
    # also made nvvm.fmax require MLIR Values rather than CuTe scalar objects.
    # Adapt the callable in this process instead of modifying either
    # site-packages tree.
    try:
        nvvm = importlib.import_module("cutlass._mlir.dialects.nvvm")
        fmax = nvvm.fmax
        cutlass = importlib.import_module("cutlass")
        float32 = cutlass.Float32
        mlir_ir = importlib.import_module("cutlass._mlir.ir")

        # CUTLASS only re-exports these NVVM enums from ``cute.arch`` for
        # CUDA 12.9.  FA4 b15 still imports them from that stable CuTe path,
        # while the server's CUDA 13 package keeps the same enums under
        # ``cutlass._mlir.dialects.nvvm``.  The CUDA 13 CuTe wrappers also
        # require string literals (for example ``"async.shared"``) rather
        # than enum instances.  Expose a small string-member namespace in
        # memory so FA4 can use the old member names with the new ABI.
        try:
            cute_arch = importlib.import_module("cutlass.cute.arch")
            for enum_name in (
                "ProxyKind",
                "SharedSpace",
                "RoundingModeKind",
                "ReduxKind",
                "AtomicOpKind",
            ):
                if not hasattr(cute_arch, enum_name) and hasattr(nvvm, enum_name):
                    enum_class = getattr(nvvm, enum_name)
                    string_members = {
                        member.name: str(member) for member in enum_class
                    }
                    setattr(
                        cute_arch,
                        enum_name,
                        types.SimpleNamespace(**string_members),
                    )
                    changed = True
        except Exception:
            # The fmax compatibility below remains independently useful for
            # CUTLASS builds that already expose the CuTe enum names.
            pass

        parameters = inspect.signature(fmax).parameters
        c_parameter = parameters.get("c")
        if (
            c_parameter is not None
            and c_parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and not getattr(fmax, "_fast_infer_fa4_compat", False)
        ):
            @functools.wraps(fmax)
            def fmax_compat(a, b, *args, **kwargs):
                # FA4 b15 uses the pre-CUTLASS-4.5 signature:
                #   nvvm.fmax(T.f32(), a, b, c=...)
                # Newer CUTLASS infers the result type and accepts only
                #   nvvm.fmax(a, b, c=...)
                # Drop the explicit result type before adapting operands.
                result_type = None
                if isinstance(a, mlir_ir.Type):
                    result_type = a
                    operands = (b, *args)
                    if len(operands) < 2:
                        return fmax(a, b, *args, **kwargs)
                    a, b, *args = operands
                if len(args) == 1:
                    # FA4 b15 supplies c positionally; some generated DSL
                    # paths also leave a keyword ``c`` behind.  The
                    # positional value is the authoritative third operand.
                    kwargs["c"] = args[0]
                    args = ()
                if args:
                    return fmax(a, b, *args, **kwargs)
                loc = kwargs.get("loc")
                ip = kwargs.get("ip")
                c = kwargs.get("c")

                def as_ir_value(value):
                    ir_value = getattr(value, "ir_value", None)
                    if callable(ir_value):
                        return ir_value(loc=loc, ip=ip)
                    try:
                        return float32(value).ir_value(loc=loc, ip=ip)
                    except Exception as exc:
                        value_type = type(value)
                        value_attrs = tuple(
                            name
                            for name in (
                                "value",
                                "type",
                                "dtype",
                                "shape",
                                "__extract_mlir_values__",
                                "__new_from_mlir_values__",
                            )
                            if hasattr(value, name)
                        )
                        raise TypeError(
                            "FA4 nvvm.fmax operand is not directly convertible: "
                            f"type={value_type.__module__}.{value_type.__qualname__}, "
                            f"text={value!s}, attrs={value_attrs}"
                        ) from exc

                raw_kwargs = {
                    "a": as_ir_value(a),
                    "b": as_ir_value(b),
                    "c": as_ir_value(c) if c is not None else None,
                    "ftz": kwargs.get("ftz"),
                    "nan": kwargs.get("nan"),
                    "abs": kwargs.get("abs"),
                    "loc": loc,
                    "ip": ip,
                }
                raw_parameters = inspect.signature(fmax).parameters
                first_parameter = next(iter(raw_parameters), None)
                if first_parameter == "res":
                    # Some CUTLASS 4.5 builds retain the old explicit-result
                    # argument, while others infer it from a/b.
                    if result_type is None:
                        result_type = getattr(cutlass, "T", None)
                        result_type = (
                            result_type.f32()
                            if result_type is not None
                            else None
                        )
                    if result_type is None:
                        raise TypeError(
                            "CUTLASS nvvm.fmax requires `res`, but FA4 did not "
                            "provide an explicit result type"
                        )
                    raw_kwargs["res"] = result_type
                return fmax(**raw_kwargs)

            fmax_compat._fast_infer_fa4_compat = True
            nvvm.fmax = fmax_compat
            changed = True
    except Exception:
        # The import probe below remains authoritative for unsupported or
        # otherwise incomplete CUTLASS installations.
        pass

    return changed


def _probe_flash_attention_4() -> tuple[bool, str | None]:
    """Check that FA4 and its transitive CUDA/CuTe dependencies import.

    ``find_spec("flash_attn.cute")`` only proves that a package directory is
    discoverable.  FA4 loads CUTLASS/CuTe modules during import, so a stale
    ``flash_attn.cute`` tree can be discoverable while still being unusable.
    Keep the full exception as a preflight reason for actionable launcher
    output.
    """

    try:
        if importlib.util.find_spec("flash_attn.cute") is None:
            return False, "flash_attn.cute is not installed"
    except Exception as exc:  # discovery can import a broken parent package
        detail = str(exc).strip().splitlines()[0] or repr(exc)
        return False, f"{type(exc).__name__}: {detail}"

    _install_flash_attention_4_cutlass_compat()
    try:
        importlib.import_module("flash_attn.cute")
    except Exception as exc:
        detail = str(exc).strip().splitlines()[0] or repr(exc)
        return False, f"{type(exc).__name__}: {detail}"
    return True, None


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
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("LONG_BENCH_SEED", "42"))
    )
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--max-input-tokens", type=int, default=None)
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
        choices=sorted(
            {default_backend, "flash_attention_4"}
            if default_backend == "flash_attention_2"
            else {default_backend}
        ),
        default=os.environ.get("LONG_BENCH_ATTENTION_BACKEND", default_backend),
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
    encoded = tokenizer(prompt, return_tensors="pt")
    return truncate_input_ids(encoded.input_ids, max_input_tokens)


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
    attention_mask = _attention_mask_for_backend(
        torch.ones_like(input_ids), getattr(args, "attention_backend", None)
    )
    return model.generate(input_ids, attention_mask=attention_mask, **kwargs)


def _warmup_args(args: argparse.Namespace, *, max_new_tokens: int = 8) -> argparse.Namespace:
    """Copy generation args with a short warmup budget.

    Warmup is for kernel/cache initialization, not for measuring long-form
    generation.  Reusing the benchmark's full output budget here can waste
    minutes before the first sample when the configured budget is 2048+.
    """
    values = vars(args).copy()
    values["max_new_tokens"] = min(int(args.max_new_tokens), int(max_new_tokens))
    return argparse.Namespace(**values)


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


def _build_decode_attention_mask(
    input_ids: torch.Tensor, *, max_new_tokens: int
) -> torch.Tensor:
    """Allocate the complete 1D attention mask once for one request.

    The old decode loop appended one column with ``torch.cat`` at every step.
    Besides allocating repeatedly, that made long generations progressively
    more expensive.  A full mask is tiny compared with model activations and
    can be sliced as the KV cache grows.
    """
    total_tokens = int(input_ids.shape[1]) + max(int(max_new_tokens), 0)
    return torch.ones(
        (int(input_ids.shape[0]), total_tokens),
        dtype=input_ids.dtype,
        device=input_ids.device,
    )


def _attention_mask_for_backend(
    attention_mask: torch.Tensor | None, attention_backend: str | None
) -> torch.Tensor | None:
    """Avoid routing a trivial mask through FlashAttention's varlen path.

    A mask containing only ones carries no padding information.  Passing it to
    older Transformers/FlashAttention combinations can nevertheless select
    the unpadding (varlen) kernel, including for one-token KV-cache decode.
    Returning ``None`` preserves the model's causal mask while using the
    regular dense FlashAttention call.  Non-trivial masks must be retained.
    """
    if (
        attention_backend in FLASH_ATTENTION_BACKENDS
        and attention_mask is not None
        and bool(torch.all(attention_mask != 0))
    ):
        return None
    return attention_mask


def _build_static_cache(
    model: Any,
    *,
    max_cache_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any | None, str]:
    """Build a preallocated Transformers cache when the runtime supports it.

    ``DynamicCache`` concatenates K/V tensors during every generated token.
    ``StaticCache`` writes into preallocated storage and is substantially more
    suitable for the long, single-request generations used by this benchmark.
    Older Transformers versions may not expose it or may reject a model
    configuration; those versions retain the public ``generate`` fallback.
    """
    try:
        from transformers.cache_utils import StaticCache
    except (ImportError, ModuleNotFoundError):
        return None, "dynamic"

    config = getattr(model, "config", None)
    if config is None:
        return None, "dynamic"

    try:
        cache = StaticCache(
            config=config,
            max_cache_len=max(int(max_cache_len), 1),
            device=device,
            dtype=dtype,
        )
    except RuntimeError as exc:
        # Never turn a real allocation failure into a slower second attempt.
        if "out of memory" in str(exc).lower():
            raise
        return None, "dynamic"
    except (AttributeError, TypeError, ValueError):
        return None, "dynamic"
    return cache, "static"


def _should_use_static_cache(attention_backend: str | None) -> bool:
    """Return whether the manual cache path is safe for this attention backend.

    Flash-Attention 2 combined with the Transformers ``StaticCache`` path has
    produced silently corrupted repeated-token outputs on some server
    combinations.  Keep the optimized static path for eager attention, while
    making FA use the public dynamic-cache semantics until its output parity is
    revalidated.  This only changes cache allocation; model weights, prompts,
    and decoding policy remain unchanged.
    """

    return attention_backend not in FLASH_ATTENTION_BACKENDS


def _resolve_flash_attention_backend(
    attention_backend: str,
    *,
    compute_capability: tuple[int, int] | None,
    flash_attention_4_available: bool,
) -> str:
    """Resolve a FlashAttention implementation that is valid on this GPU.

    FA2 is not a supported Blackwell implementation.  Silently continuing on
    a B200 is dangerous because the kernel can return plausible-looking but
    repeated token IDs.  Prefer FA4 when the runtime provides it; otherwise
    fail before loading the model and before writing benchmark records.
    """
    if attention_backend != "flash_attention_2" or compute_capability is None:
        return attention_backend
    if int(compute_capability[0]) < 10:
        return attention_backend
    if flash_attention_4_available:
        return "flash_attention_4"
    raise RuntimeError(
        "vanilla_fa requested FlashAttention-2 on Blackwell/B200, but FA2 is "
        "not a supported backend for this GPU and flash_attn.cute (FA4) is "
        "not installed; install a Blackwell-compatible FlashAttention-4 "
        "runtime or run vanilla_fa on Ampere/Ada/Hopper"
    )


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
    input_length = int(input_ids.shape[1])
    attention_mask = _build_decode_attention_mask(
        input_ids, max_new_tokens=args.max_new_tokens
    )
    try:
        model_dtype = next(model.parameters()).dtype
    except (AttributeError, StopIteration):
        model_dtype = _dtype(getattr(args, "dtype", "float32"))
    if _should_use_static_cache(getattr(args, "attention_backend", None)):
        static_cache, cache_backend = _build_static_cache(
            model,
            max_cache_len=input_length + int(args.max_new_tokens),
            device=device,
            dtype=model_dtype,
        )
    else:
        static_cache, cache_backend = None, "dynamic_fa_safe"
    try:
        prefill_start = time.perf_counter()
        prefill_kwargs = {
            "input_ids": input_ids,
            "attention_mask": _attention_mask_for_backend(
                attention_mask[:, :input_length],
                getattr(args, "attention_backend", None),
            ),
            "use_cache": True,
            "return_dict": True,
        }
        if static_cache is not None:
            prefill_kwargs["past_key_values"] = static_cache
        prefill = model(**prefill_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
        past = getattr(prefill, "past_key_values", None)
        if past is None:
            past = static_cache
        next_token = _next_token(prefill.logits, args.temperature)
        generated = [next_token]
        eos_id = tokenizer.eos_token_id

        decode_start = time.perf_counter()
        if not _is_eos(next_token, eos_id):
            for _ in range(max(args.max_new_tokens - 1, 0)):
                current_length = input_length + len(generated)
                step = model(
                    input_ids=next_token,
                    attention_mask=_attention_mask_for_backend(
                        attention_mask[:, :current_length],
                        getattr(args, "attention_backend", None),
                    ),
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = getattr(step, "past_key_values", None)
                if past is None:
                    past = static_cache
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
            "kv_cache_backend": cache_backend,
            "attention_mask_strategy": "preallocated_slice",
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
                "kv_cache_backend": "generate",
                "attention_mask_strategy": "generate",
            }
        )
        return output_ids, timing


def _load_model(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    if not args.model:
        raise SystemExit("--model or LONG_BENCH_MODEL/MODEL_TARGET is required")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use orchestrator smoke preflight on this host")

    if args.attention_backend in FLASH_ATTENTION_BACKENDS:
        requested_backend = args.attention_backend
        compute_capability = (
            torch.cuda.get_device_capability(device)
            if device.type == "cuda"
            else None
        )
        flash_attention_4_available, flash_attention_4_reason = (
            _probe_flash_attention_4()
            if requested_backend == "flash_attention_4"
            or (
                requested_backend == "flash_attention_2"
                and compute_capability is not None
                and int(compute_capability[0]) >= 10
            )
            else (False, None)
        )
        try:
            args.attention_backend = _resolve_flash_attention_backend(
                requested_backend,
                compute_capability=compute_capability,
                flash_attention_4_available=flash_attention_4_available,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

        if args.attention_backend == "flash_attention_4" and not flash_attention_4_available:
            detail = f" ({flash_attention_4_reason})" if flash_attention_4_reason else ""
            raise SystemExit(
                "vanilla_fa requires an importable FlashAttention-4 runtime, but "
                f"flash_attn.cute failed its import probe{detail}. The existing "
                "flash-attn-4/CuTeDSL packages are incompatible beyond the "
                "process-local b15 shim; do not mix an older flash_attn.cute "
                "source tree with a newer nvidia-cutlass-dsl."
            )

        try:
            if args.attention_backend == "flash_attention_4":
                from flash_attn import cute as _flash_attention_cute  # noqa: F401
            else:
                import flash_attn  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                f"vanilla_fa requires the installed {args.attention_backend} "
                "runtime; no fallback is allowed"
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
        if args.max_input_tokens is None:
            args.max_input_tokens = 4096
    elif args.max_input_tokens is None:
        args.max_input_tokens = 0
    if not args.data_file:
        raise SystemExit("--data-file or LONG_BENCH_DATA_FILE is required")
    if args.max_samples is not None and args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")

    seed_everything(args.seed)
    device = torch.device(args.device)
    records = load_records(Path(args.data_file), args.max_samples)
    data_name = Path(args.data_file).stem
    requested_attention_backend = args.attention_backend

    load_start = time.perf_counter()
    model, tokenizer = _load_model(args, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_ms = round((time.perf_counter() - load_start) * 1000.0, 3)
    metadata = runtime_metadata()
    effective_attention_backend = getattr(
        getattr(model, "config", None), "_attn_implementation", None
    )
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
        "extra_metrics": {
            "requested_attention_backend": requested_attention_backend,
            "effective_attention_backend": effective_attention_backend
            or "unknown",
        },
    }

    with torch.inference_mode():
        seed_everything(args.seed)
        warmup_ids = _prompt_batch(tokenizer, "Hello", max_input_tokens=0).to(device)
        warmup_args = _warmup_args(args)
        for _ in range(max(args.warmup_runs, 0)):
            _generate(model, warmup_ids, warmup_args)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    writer = io_util.JsonlWriter(Path(args.output))
    successful = 0
    for sample in records:
        seed_everything(args.seed)
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
        text = tokenizer.decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        task_type = sample.get("raw", {}).get("task_type") or sample.get("task_type")
        timing["model_load_ms"] = model_load_ms
        record = build_sample_record(
            method=method,
            dataset=sample.get("raw", {}).get("dataset", data_name),
            sample_id=sample["id"],
            model=str(args.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            timing=timing,
            config={
                **config,
                "extra_metrics": {
                    **dict(config.get("extra_metrics", {}) or {}),
                    "kv_cache_backend": timing.get("kv_cache_backend"),
                    "attention_mask_strategy": timing.get(
                        "attention_mask_strategy"
                    ),
                },
            },
            text=text,
            reference_output=sample.get("reference"),
        )
        record["output_quality_guard"] = {
            "degenerate_repetition": is_degenerate_output(text),
            "action": "annotate_only",
        }
        if task_type == "code_completion":
            metrics.add_code_completion(record, text, sample.get("reference"))
        else:
            rouge.add_rouge(record, text, sample.get("reference"))
            metrics.add_semantic(record, text, sample.get("reference"))
        if record["output_quality_guard"]["degenerate_repetition"]:
            print(
                f"[{method}][{sample['id']}] warning: output has a strong "
                "repeated-n-gram collapse signal; inference result is retained "
                "for audit but should not be used as a quality comparison.",
                flush=True,
            )
        record["run_id"] = args.run_id
        record["task_type"] = task_type
        writer.add(record)
        successful += 1
        print(
            f"[{method}][{record['dataset']}][{sample['id']}] "
            f"input={input_tokens} output={output_tokens} "
            f"prefill_ms={record['prefill_ms']} decode_ms={record['decode_ms']} "
            f"e2e_ms={record['e2e_ms']} tok_s={record['throughput_tok_s']} "
            f"decode_tok_s={record['decode_throughput_tok_s']} "
            f"cache={record['extra_metrics'].get('kv_cache_backend')} "
            f"attn={record['extra_metrics'].get('effective_attention_backend')}",
            flush=True,
        )

    quality = (
        metrics.aggregate_code_completion(writer.records)
        if any(r.get("task_type") == "code_completion" for r in writer.records)
        else metrics.aggregate_semantic(writer.records)
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
        "requested_attention_backend": requested_attention_backend,
        "effective_attention_backend": effective_attention_backend or "unknown",
        "runtime": metadata,
        **quality,
    }
    writer.finalize(summary)
    io_util.print_table(list(summary.items()))
    print(f"Saved to: {args.output}")
    return 0 if successful == len(records) else 1
