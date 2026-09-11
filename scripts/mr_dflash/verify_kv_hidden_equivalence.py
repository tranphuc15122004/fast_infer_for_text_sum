"""Verify full-forward and KV-cache hidden-state equivalence.

The script takes one canonical regenerated sample, uses its assistant
response as a fixed continuation, and runs the target in two ways:

1. one causal full forward over prompt + continuation;
2. one prefill over the prompt followed by one-token-at-a-time decode with
   ``past_key_values``.

The continuation is fixed so both paths process exactly the same tokens. This
is the relevant invariant for the cache phase: an autoregressive KV cache must
not change hidden states at positions that have already been processed.

Example on the B200 server::

    PYTHONPATH=src python3 scripts/mr_dflash/verify_kv_hidden_equivalence.py \
      --target-model-path /workspace/storage-shared/models/Qwen3-4B \
      --input /workspace/storage-shared/nlp/dungdx4/phuc_projects/data/\
mr_dflash_pilot_full/regenerated_full/train.jsonl \
      --sample-index 0 --max-length 32768 --decode-tokens 0 \
      --device cuda --torch-dtype bfloat16 --local-files-only
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from MR_DFlash.data import build_sample, iter_jsonl  # noqa: E402
from MR_DFlash.kv_hidden import compare_hidden_states  # noqa: E402


DEFAULT_LAYER_IDS = (1, 9, 17, 25, 33)


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def _load_row(path: str, sample_index: int) -> dict[str, Any]:
    if sample_index < 0:
        raise ValueError("sample-index phải >= 0")
    for row_index, row in enumerate(iter_jsonl(path)):
        if row_index == sample_index:
            return row
    raise IndexError(f"không tìm thấy sample-index={sample_index} trong {path}")


def _first_supervised_position(loss_mask: Sequence[Any]) -> int:
    for position, value in enumerate(loss_mask):
        if bool(value):
            return position
    raise ValueError("sample không có supervised response token")


def _model_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    position_ids: torch.Tensor,
    use_cache: bool,
    past_key_values: Any = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "use_cache": use_cache,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values

    # Recent Transformers models use cache_position for DynamicCache. Older
    # versions infer it from position_ids/past_key_values. Pass it only when
    # the concrete model advertises the argument.
    parameters = inspect.signature(model.forward).parameters
    accepts_cache_position = "cache_position" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if accepts_cache_position:
        start = int(position_ids[0, 0].item())
        kwargs["cache_position"] = torch.arange(
            start,
            start + input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        )
    return model(**kwargs)


def _layer_hidden(outputs: Any, layer_ids: Sequence[int]) -> list[torch.Tensor]:
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("model không trả output_hidden_states")
    result = []
    for layer_id in layer_ids:
        index = int(layer_id) + 1
        if index >= len(hidden_states):
            raise ValueError(
                f"layer id {layer_id} không hợp lệ; model chỉ trả {len(hidden_states) - 1} layers"
            )
        result.append(hidden_states[index][0].detach().float().cpu())
    return result


def _run_full(
    backbone: torch.nn.Module,
    sequence: torch.Tensor,
    layer_ids: Sequence[int],
) -> list[torch.Tensor]:
    device = sequence.device
    length = sequence.shape[1]
    attention_mask = torch.ones((1, length), dtype=torch.long, device=device)
    positions = torch.arange(length, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        outputs = _model_forward(
            backbone,
            sequence,
            attention_mask,
            position_ids=positions,
            use_cache=False,
        )
    return _layer_hidden(outputs, layer_ids)


def _run_incremental(
    backbone: torch.nn.Module,
    prompt: torch.Tensor,
    continuation: torch.Tensor,
    layer_ids: Sequence[int],
) -> list[torch.Tensor]:
    device = prompt.device
    prompt_length = prompt.shape[1]
    prompt_mask = torch.ones((1, prompt_length), dtype=torch.long, device=device)
    prompt_positions = torch.arange(
        prompt_length, dtype=torch.long, device=device
    ).unsqueeze(0)
    with torch.inference_mode():
        outputs = _model_forward(
            backbone,
            prompt,
            prompt_mask,
            position_ids=prompt_positions,
            use_cache=True,
        )
        collected = [[value] for value in _layer_hidden(outputs, layer_ids)]
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            raise RuntimeError("model không trả past_key_values khi use_cache=True")

        current_length = prompt_length
        for token in continuation[0].tolist():
            token_ids = torch.tensor([[int(token)]], dtype=torch.long, device=device)
            attention_mask = torch.ones(
                (1, current_length + 1), dtype=torch.long, device=device
            )
            positions = torch.tensor(
                [[current_length]], dtype=torch.long, device=device
            )
            outputs = _model_forward(
                backbone,
                token_ids,
                attention_mask,
                position_ids=positions,
                use_cache=True,
                past_key_values=past_key_values,
            )
            step_hidden = _layer_hidden(outputs, layer_ids)
            for layer_index, value in enumerate(step_hidden):
                collected[layer_index].append(value)
            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                raise RuntimeError("model mất past_key_values trong decode")
            current_length += 1
    return [torch.cat(values, dim=0) for values in collected]


def verify_sample(
    *,
    target_model_path: str,
    data_path: str,
    sample_index: int,
    max_length: int,
    layer_ids: Sequence[int],
    decode_tokens: int,
    supervision_mode: str,
    device_name: str,
    torch_dtype: str,
    local_files_only: bool,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    row = _load_row(data_path, sample_index)
    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path,
        local_files_only=local_files_only,
    )
    sample = build_sample(
        row,
        tokenizer,
        max_length,
        supervision_mode=supervision_mode,
    )
    if sample is None:
        raise ValueError(f"sample {row.get('id', sample_index)!r} không render được")
    response_start = _first_supervised_position(sample["loss_mask"])
    all_ids = torch.tensor(sample["input_ids"], dtype=torch.long)
    prompt_ids = all_ids[:response_start]
    stored_continuation = all_ids[response_start:]
    if prompt_ids.numel() == 0 or stored_continuation.numel() == 0:
        raise ValueError("sample phải có cả prompt và assistant continuation")
    if decode_tokens > 0:
        stored_continuation = stored_continuation[:decode_tokens]
    sequence = torch.cat([prompt_ids, stored_continuation], dim=0).unsqueeze(0)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda nhưng torch.cuda.is_available()=False")

    dtype = _dtype(torch_dtype)
    model = AutoModel.from_pretrained(
        target_model_path,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    sequence = sequence.to(device)
    prompt = prompt_ids.unsqueeze(0).to(device)
    continuation = stored_continuation.unsqueeze(0).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    full_hidden = _run_full(model, sequence, layer_ids)
    incremental_hidden = _run_incremental(model, prompt, continuation, layer_ids)
    reports = {
        str(layer_id): compare_hidden_states(
            full_value,
            incremental_value,
            atol=atol,
            rtol=rtol,
        )
        for layer_id, full_value, incremental_value in zip(
            layer_ids, full_hidden, incremental_hidden
        )
    }
    result: dict[str, Any] = {
        "status": "pass" if all(report["allclose"] for report in reports.values()) else "fail",
        "sample_id": str(sample["id"]),
        "sample_index": int(sample_index),
        "prompt_tokens": int(prompt.shape[1]),
        "continuation_tokens_compared": int(continuation.shape[1]),
        "total_tokens_compared": int(sequence.shape[1]),
        "layer_ids": [int(value) for value in layer_ids],
        "device": str(device),
        "torch_dtype": torch_dtype,
        "atol": float(atol),
        "rtol": float(rtol),
        "layers": reports,
    }
    if device.type == "cuda":
        result["peak_memory_allocated_gb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024**3), 4
        )
        result["peak_memory_reserved_gb"] = round(
            torch.cuda.max_memory_reserved(device) / (1024**3), 4
        )
    return result


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đối chiếu hidden full-forward với incremental KV-cache"
    )
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--input", required=True, help="canonical regenerated JSONL")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=list(DEFAULT_LAYER_IDS))
    parser.add_argument("--decode-tokens", type=int, default=32, help="0 = toàn bộ response")
    parser.add_argument(
        "--supervision-mode",
        choices=["all_assistant", "last_assistant"],
        default="last_assistant",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--report", default=None)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_length < 1 or args.decode_tokens < 0:
        raise ValueError("max-length phải > 0 và decode-tokens phải >= 0")
    result = verify_sample(
        target_model_path=args.target_model_path,
        data_path=args.input,
        sample_index=args.sample_index,
        max_length=args.max_length,
        layer_ids=args.target_layer_ids,
        decode_tokens=args.decode_tokens,
        supervision_mode=args.supervision_mode,
        device_name=args.device,
        torch_dtype=args.torch_dtype,
        local_files_only=args.local_files_only,
        atol=args.atol,
        rtol=args.rtol,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload, encoding="utf-8")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
