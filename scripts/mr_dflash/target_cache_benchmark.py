#!/usr/bin/env python3
"""Benchmark target generation và target hidden-state caching.

Script này được thiết kế để chạy trực tiếp trên server GPU và trả lời ba câu
hỏi bằng cùng một prompt, cùng target model và cùng attention backend:

1. ``generate`` hiện tại của Hugging Face (KV cache bật rõ ràng);
2. full hidden capture hiện tại bằng ``AutoModelForCausalLM``;
3. full hidden capture bằng backbone-only ``AutoModel`` và fused
   generate+capture bằng ``past_key_values``.

Fused path là một benchmark/verification path: nó chưa thay thế pipeline
production. Script kiểm tra token output của fused path với
``model.generate`` và so sánh hidden trajectory với full forward.

Ví dụ trên B200::

    PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python3 \
      scripts/mr_dflash/target_cache_benchmark.py \
      --target-model-path /workspace/storage-shared/models/Qwen3-4B \
      --input-jsonl /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/\
mr_dflash_pilot_full/regenerated_full/val.jsonl \
      --sample-index 0 --max-new-tokens 128 \
      --target-layer-ids 1 9 17 25 33 \
      --attn-implementation flash_attention_2 \
      --device cuda --torch-dtype bfloat16 --local-files-only \
      --report /tmp/qwen3_target_cache_benchmark.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from MR_DFlash.kv_hidden import compare_hidden_states  # noqa: E402


DEFAULT_LAYER_IDS = (1, 9, 17, 25, 33)


def generation_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Bỏ đúng assistant cuối của regenerated row để tạo prompt.

    Các assistant turn trước đó được giữ lại vì chúng là một phần ngữ cảnh
    hội thoại. Với prompt-only row, danh sách được giữ nguyên.
    """

    result = [dict(message) for message in messages]
    if result and str(result[-1].get("role", "")).lower() == "assistant":
        return result[:-1]
    return result


def compare_token_ids(reference: Iterable[int], candidate: Iterable[int]) -> dict[str, Any]:
    """So sánh hai chuỗi token và báo vị trí mismatch đầu tiên."""

    reference = [int(value) for value in reference]
    candidate = [int(value) for value in candidate]
    first_mismatch: Optional[int] = None
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            first_mismatch = index
            break
    if first_mismatch is None and len(reference) != len(candidate):
        first_mismatch = min(len(reference), len(candidate))
    return {
        "exact": first_mismatch is None,
        "reference_length": len(reference),
        "candidate_length": len(candidate),
        "first_mismatch": first_mismatch,
    }


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda nhưng torch.cuda.is_available()=False")
    return device


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _peak_memory(device: torch.device) -> dict[str, Optional[float]]:
    if device.type != "cuda":
        return {"allocated_gb": None, "reserved_gb": None}
    gib = float(1024**3)
    return {
        "allocated_gb": torch.cuda.max_memory_allocated(device) / gib,
        "reserved_gb": torch.cuda.max_memory_reserved(device) / gib,
    }


def _timed(device: torch.device, fn):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _sync(device)
    started = time.perf_counter()
    value = fn()
    _sync(device)
    elapsed = time.perf_counter() - started
    return value, elapsed, _peak_memory(device)


class LayerCapture:
    """Capture selected backbone layer outputs without retaining all layers."""

    def __init__(self, model: torch.nn.Module, layer_ids: Sequence[int]) -> None:
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        target_model = getattr(model, "model", model)
        layers = getattr(target_model, "layers", None)
        if layers is None:
            raise ValueError("không tìm thấy model.layers để capture hidden states")
        num_layers = len(layers)
        invalid = [layer_id for layer_id in self.layer_ids if layer_id < 0 or layer_id >= num_layers]
        if invalid:
            raise ValueError(
                f"target-layer-ids {invalid} nằm ngoài [0,{num_layers})"
            )
        self._values: dict[int, torch.Tensor] = {}
        self._hooks = [
            layers[layer_id].register_forward_hook(self._make_hook(layer_id))
            for layer_id in self.layer_ids
        ]

    def _make_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"layer {layer_id} không trả Tensor")
            self._values[layer_id] = value

        return hook

    def clear(self) -> None:
        self._values.clear()

    def current(self) -> torch.Tensor:
        missing = [layer_id for layer_id in self.layer_ids if layer_id not in self._values]
        if missing:
            raise RuntimeError(f"capture thiếu layer {missing}")
        return torch.cat([self._values[layer_id] for layer_id in self.layer_ids], dim=-1)

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._values.clear()


def _load_model(
    model_cls: Any,
    model_path: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    attn_implementation: str,
    local_files_only: bool,
    target_revision: Optional[str],
) -> torch.nn.Module:
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if target_revision:
        load_kwargs["revision"] = target_revision
    if attn_implementation != "auto":
        load_kwargs["attn_implementation"] = attn_implementation
    model = model_cls.from_pretrained(model_path, **load_kwargs).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _apply_chat_template(tokenizer: Any, messages: Sequence[dict[str, Any]]) -> torch.Tensor:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": False,
    }
    try:
        value = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        value = tokenizer.apply_chat_template(messages, **kwargs)
    if isinstance(value, dict):
        value = value["input_ids"]
    return torch.as_tensor(value, dtype=torch.long).reshape(1, -1)


def _load_prompt_ids(
    tokenizer: Any,
    *,
    prompt: Optional[str],
    input_jsonl: Optional[str],
    sample_index: int,
    max_input_tokens: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if input_jsonl:
        from _common import read_jsonl

        row = next(
            (row for index, row in enumerate(read_jsonl(input_jsonl)) if index == sample_index),
            None,
        )
        if row is None:
            raise IndexError(f"sample-index={sample_index} không có trong {input_jsonl}")
        messages = row.get("conversations") or row.get("messages")
        if isinstance(messages, list):
            messages = generation_messages(messages)
            input_ids = _apply_chat_template(tokenizer, messages)
        else:
            text = str(row.get("prompt") or row.get("text") or "")
            if not text:
                raise ValueError("row không có conversations/messages/prompt/text")
            input_ids = tokenizer(text, return_tensors="pt", return_attention_mask=False)["input_ids"]
        metadata = {"sample_index": sample_index, "sample_id": row.get("id")}
    else:
        input_ids = tokenizer(prompt or "", return_tensors="pt", return_attention_mask=False)["input_ids"]
        metadata = {"sample_index": None, "sample_id": "prompt"}
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError(f"prompt tokenization trả shape bất hợp lệ: {tuple(input_ids.shape)}")
    if max_input_tokens > 0 and input_ids.shape[1] > max_input_tokens:
        input_ids = input_ids[:, :max_input_tokens]
        metadata["input_truncated"] = True
    else:
        metadata["input_truncated"] = False
    return input_ids.to(dtype=torch.long), metadata


def _attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids, dtype=torch.long)


@torch.inference_mode()
def _generate_reference(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    pad_token_id: Optional[int],
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": False,
        "attention_mask": _attention_mask(input_ids),
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = int(pad_token_id)
    return model.generate(input_ids, **kwargs)


@torch.inference_mode()
def _full_capture(
    model: torch.nn.Module,
    capture: LayerCapture,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    capture.clear()
    model(
        input_ids=input_ids,
        attention_mask=_attention_mask(input_ids),
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    return capture.current().detach().cpu()


def _cached_forward(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values: Any,
    cache_position: torch.Tensor,
) -> Any:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": True,
        "output_hidden_states": False,
        "return_dict": True,
        "cache_position": cache_position,
    }
    try:
        return model(**kwargs)
    except TypeError as exc:
        # Older Transformers versions do not expose cache_position. Their
        # causal models infer the position from past_key_values.
        if "cache_position" not in str(exc):
            raise
        kwargs.pop("cache_position")
        return model(**kwargs)


@torch.inference_mode()
def _fused_generate_capture(
    model: torch.nn.Module,
    capture: LayerCapture,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate greedy tokens and capture every hidden state in one KV run."""

    capture.clear()
    prefill = model(
        input_ids=input_ids,
        attention_mask=_attention_mask(input_ids),
        output_hidden_states=False,
        use_cache=True,
        return_dict=True,
    )
    hidden_parts = [capture.current().detach().cpu()]
    past_key_values = getattr(prefill, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError("target không trả past_key_values khi use_cache=True")

    next_token = prefill.logits[:, -1:].argmax(dim=-1)
    generated: list[torch.Tensor] = []
    prompt_length = int(input_ids.shape[1])
    for step in range(max_new_tokens):
        current_length = prompt_length + step
        decode_ids = next_token
        decode_mask = torch.ones(
            (input_ids.shape[0], current_length + 1),
            dtype=torch.long,
            device=input_ids.device,
        )
        cache_position = torch.arange(
            current_length,
            current_length + 1,
            dtype=torch.long,
            device=input_ids.device,
        )
        capture.clear()
        outputs = _cached_forward(
            model,
            input_ids=decode_ids,
            attention_mask=decode_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        hidden_parts.append(capture.current().detach().cpu())
        generated.append(next_token.detach().cpu())
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError("target mất past_key_values trong decode")
        if int(next_token[0, 0]) in eos_token_ids:
            break
        next_token = outputs.logits[:, -1:].argmax(dim=-1)

    generated_ids = torch.cat(generated, dim=1) if generated else input_ids[:, :0].cpu()
    full_ids = torch.cat([input_ids.detach().cpu(), generated_ids], dim=1)
    hidden = torch.cat(hidden_parts, dim=1)
    return full_ids, hidden


def _cleanup_model(model: Optional[torch.nn.Module], device: torch.device) -> None:
    if model is not None:
        del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    device = _resolve_device(args.device)
    dtype = _dtype(args.torch_dtype)
    load_local = bool(args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        local_files_only=load_local,
        revision=args.target_revision,
    ) if args.target_revision else AutoTokenizer.from_pretrained(
        args.target_model_path,
        local_files_only=load_local,
    )
    input_ids, input_meta = _load_prompt_ids(
        tokenizer,
        prompt=args.prompt,
        input_jsonl=args.input_jsonl,
        sample_index=args.sample_index,
        max_input_tokens=args.max_input_tokens,
    )
    input_ids = input_ids.to(device)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_value = getattr(tokenizer, "eos_token_id", None)
    if eos_value is None:
        eos_value = getattr(tokenizer, "eos_token_id", None)
    eos_token_ids = {int(eos_value)} if isinstance(eos_value, int) else {int(x) for x in (eos_value or [])}

    layer_ids = tuple(args.target_layer_ids)
    result: dict[str, Any] = {
        "target_model_path": args.target_model_path,
        "device": str(device),
        "torch_dtype": args.torch_dtype,
        "requested_attention_backend": args.attn_implementation,
        "target_layer_ids": list(layer_ids),
        "input": {**input_meta, "tokens": int(input_ids.shape[1])},
        "max_new_tokens": int(args.max_new_tokens),
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
    }

    model = _load_model(
        AutoModelForCausalLM,
        args.target_model_path,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
        local_files_only=load_local,
        target_revision=args.target_revision,
    )
    result["actual_attention_backend_causal_lm"] = getattr(
        getattr(model, "config", None), "_attn_implementation", None
    )
    capture = LayerCapture(model, layer_ids)
    try:
        for _ in range(args.warmup):
            _generate_reference(model, input_ids, max_new_tokens=min(2, args.max_new_tokens), pad_token_id=pad_token_id)
        baseline_runs = []
        for _ in range(args.repeat):
            baseline, elapsed, memory = _timed(
                device,
                lambda: _generate_reference(
                    model,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=pad_token_id,
                ),
            )
            baseline_runs.append({"seconds": elapsed, "peak_memory": memory})
        baseline_ids = baseline
        baseline_generated = baseline_ids[:, input_ids.shape[1] :]
        result["hf_generate"] = {
            "runs": baseline_runs,
            "generated_tokens": int(baseline_generated.shape[1]),
            "generated_ids": baseline_generated[0].tolist(),
        }

        for _ in range(args.warmup):
            _full_capture(model, capture, baseline_ids)
        current_cache_runs = []
        current_hidden = None
        for _ in range(args.repeat):
            current_hidden, elapsed, memory = _timed(
                device,
                lambda: _full_capture(model, capture, baseline_ids),
            )
            current_cache_runs.append({"seconds": elapsed, "peak_memory": memory})
        result["current_causal_lm_full_capture"] = {
            "runs": current_cache_runs,
            "hidden_shape": list(current_hidden.shape),
            "uses_kv_cache": False,
        }

        for _ in range(args.warmup):
            _fused_generate_capture(
                model,
                capture,
                input_ids,
                max_new_tokens=min(2, args.max_new_tokens),
                eos_token_ids=eos_token_ids,
            )
        fused_runs = []
        fused_ids = fused_hidden = None
        for _ in range(args.repeat):
            (fused_ids, fused_hidden), elapsed, memory = _timed(
                device,
                lambda: _fused_generate_capture(
                    model,
                    capture,
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos_token_ids,
                ),
            )
            fused_runs.append({"seconds": elapsed, "peak_memory": memory})
        result["fused_generate_capture"] = {
            "runs": fused_runs,
            "hidden_shape": list(fused_hidden.shape),
            "uses_kv_cache": True,
        }
        result["comparisons"] = {
            "hf_vs_fused_tokens": compare_token_ids(
                baseline_generated[0].tolist(), fused_ids[0, input_ids.shape[1] :].tolist()
            ),
            "current_capture_vs_fused_capture": compare_hidden_states(
                current_hidden, fused_hidden, atol=args.atol, rtol=args.rtol
            ),
        }
    finally:
        capture.close()
        _cleanup_model(model, device)

    backbone = _load_model(
        AutoModel,
        args.target_model_path,
        dtype=dtype,
        device=device,
        attn_implementation=args.attn_implementation,
        local_files_only=load_local,
        target_revision=args.target_revision,
    )
    result["actual_attention_backend_backbone"] = getattr(
        getattr(backbone, "config", None), "_attn_implementation", None
    )
    backbone_capture = LayerCapture(backbone, layer_ids)
    try:
        for _ in range(args.warmup):
            _full_capture(backbone, backbone_capture, baseline_ids)
        backbone_runs = []
        backbone_hidden = None
        for _ in range(args.repeat):
            backbone_hidden, elapsed, memory = _timed(
                device,
                lambda: _full_capture(backbone, backbone_capture, baseline_ids),
            )
            backbone_runs.append({"seconds": elapsed, "peak_memory": memory})
        result["backbone_only_full_capture"] = {
            "runs": backbone_runs,
            "hidden_shape": list(backbone_hidden.shape),
            "uses_lm_head": False,
            "uses_kv_cache": False,
        }
        result["comparisons"]["current_vs_backbone_hidden"] = compare_hidden_states(
            current_hidden, backbone_hidden, atol=args.atol, rtol=args.rtol
        )
    finally:
        backbone_capture.close()
        _cleanup_model(backbone, device)

    current_seconds = current_cache_runs[0]["seconds"] + baseline_runs[0]["seconds"]
    fused_seconds = fused_runs[0]["seconds"]
    backbone_seconds = backbone_runs[0]["seconds"]
    result["summary"] = {
        "current_generate_plus_full_capture_s": current_seconds,
        "fused_generate_capture_s": fused_seconds,
        "backbone_only_capture_s": backbone_seconds,
        "fused_speedup_vs_current_generate_plus_capture": (
            current_seconds / fused_seconds if fused_seconds > 0 else None
        ),
        "backbone_speedup_vs_current_full_capture": (
            current_cache_runs[0]["seconds"] / backbone_seconds if backbone_seconds > 0 else None
        ),
    }
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model-path", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--input-jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=list(DEFAULT_LAYER_IDS))
    parser.add_argument(
        "--attn-implementation",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
    )
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    if args.sample_index < 0:
        raise ValueError("sample-index phải >= 0")
    if args.max_input_tokens < 0 or args.max_new_tokens < 1:
        raise ValueError("max-input-tokens phải >= 0 và max-new-tokens phải >= 1")
    if args.warmup < 0 or args.repeat < 1:
        raise ValueError("warmup phải >= 0 và repeat phải >= 1")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("atol/rtol phải >= 0")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = run_benchmark(args)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(serialized)
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(serialized + "\n", encoding="utf-8")
        print(f"[benchmark] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

