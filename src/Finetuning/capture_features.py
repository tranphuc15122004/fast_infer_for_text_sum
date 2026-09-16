"""Offline target hidden-state capture using a local Transformers snapshot."""

from __future__ import annotations

import argparse
import json
from itertools import chain
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping

import torch

from .data import DEFAULT_SUMMARY_PROMPT_TEMPLATE
from .features import (
    FEATURE_MANIFEST_FILENAME,
    FeatureManifest,
    _canonicalize_input_ids,
    _canonicalize_loss_mask,
    _dtype_name,
    _reject_symlink_components,
    validate_feature_record,
)


def _as_dtype(value: torch.dtype | str) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    name = str(value).removeprefix("torch.")
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported capture dtype: {value!r}")
    return dtype


def _as_1d_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    if value.ndim == 2 and value.shape[0] == 1:
        value = value.squeeze(0)
    if value.ndim != 1:
        raise ValueError(f"prepared example {name!r} must be one-dimensional")
    return value.detach().cpu()


def _atomic_torch_save(payload: Mapping[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_feature_output_dir(destination: Path) -> None:
    """Reject replacement of a directory containing unrelated user files."""

    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"feature output path is not a directory: {destination}")
    allowed_suffixes = (".pt", ".pth", ".ckpt", ".ckpt.gz")
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                "refusing to use feature output containing a symlink: " f"{path}"
            )
        if path.is_file() and path.name != FEATURE_MANIFEST_FILENAME and not path.name.endswith(allowed_suffixes):
            raise ValueError(
                "refusing to replace feature output containing unrelated file: "
                f"{path}"
            )


def _publish_feature_generation(
    staging: Path,
    destination: Path,
    manifest: FeatureManifest,
) -> None:
    """Publish a complete immutable generation behind an atomic manifest."""

    generations = destination / ".generations"
    generations.mkdir(parents=True, exist_ok=True)
    generation_name = staging.name.lstrip(".")
    generation_dir = generations / f"generation-{generation_name}"
    os.replace(staging, generation_dir)
    manifest.generation_dir = generation_dir.relative_to(destination).as_posix()
    # Readers either retain the old manifest/generation or observe this new
    # manifest after its complete generation has already been renamed in.
    _atomic_json_save(manifest.to_dict(), destination / FEATURE_MANIFEST_FILENAME)


def _target_config(model: Any) -> Any:
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError("local target model has no config")
    text_config = getattr(config, "text_config", None)
    return text_config if text_config is not None else config


def _extract_hidden_states(outputs: Any) -> Any:
    if isinstance(outputs, Mapping):
        return outputs.get("hidden_states")
    return getattr(outputs, "hidden_states", None)


def _load_local_target(
    target_model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
) -> Any:
    if not target_model_path.is_dir():
        raise FileNotFoundError(
            f"local target model snapshot not found: {target_model_path}"
        )
    # Import at the capture boundary only.  Data loading and feature reading
    # remain usable without Transformers, and this call cannot resolve remote
    # model identifiers because local_files_only is unconditional.
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(target_model_path),
        local_files_only=True,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model = model.to(device)
    model.eval()
    return model


def _resolve_layer_ids(target_layer_ids: Iterable[int], num_layers: int) -> list[int]:
    layer_ids = list(target_layer_ids)
    if not layer_ids:
        raise ValueError("target_layer_ids must not be empty")
    if any(
        isinstance(layer_id, bool)
        or not isinstance(layer_id, int)
        or layer_id < 0
        or layer_id >= num_layers
        for layer_id in layer_ids
    ):
        raise ValueError(
            f"target_layer_ids must be within [0, {num_layers}), got {layer_ids!r}"
        )
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("target_layer_ids must not contain duplicates")
    return layer_ids


def _capture_hidden_feature(
    model: Any,
    input_ids: torch.Tensor,
    layer_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    model_input = input_ids.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=model_input,
            output_hidden_states=True,
            use_cache=False,
        )
    hidden_states = _extract_hidden_states(outputs)
    if hidden_states is None:
        raise ValueError("target model did not return output_hidden_states")
    selected: list[torch.Tensor] = []
    for layer_id in layer_ids:
        index = layer_id + 1  # SpecForge layer output includes embedding at index 0.
        if index >= len(hidden_states):
            raise ValueError(
                f"target model hidden_states missing selected layer {layer_id}"
            )
        value = hidden_states[index]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"target layer {layer_id} output is not a tensor")
        if value.ndim != 3 or value.shape[0] != 1:
            raise ValueError(
                f"target layer {layer_id} hidden state must have shape [1, seq, hidden], "
                f"got {tuple(value.shape)}"
            )
        selected.append(value[0])
    feature = torch.cat(selected, dim=-1).detach().to(device="cpu", dtype=dtype)
    return feature


def capture_dataset(
    target_model_path: str | Path,
    prepared_examples: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    target_layer_ids: Iterable[int] | None,
    max_length: int,
    device: str | torch.device,
    dtype: torch.dtype | str,
    *,
    num_draft_layers: int | None = None,
    trust_remote_code: bool = False,
    tokenizer_id: str | None = None,
    prompt_contract: Mapping[str, Any] | None = None,
) -> FeatureManifest:
    """Capture prepared token examples into an atomic local feature store."""

    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length < 1:
        raise ValueError("max_length must be a positive integer")
    requested_dtype = _as_dtype(dtype)
    model_path = Path(target_model_path)
    device_obj = torch.device(device)
    destination = Path(output_dir)
    _reject_symlink_components(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_feature_output_dir(destination)
    else:
        destination.mkdir(parents=True)
    model = _load_local_target(
        model_path,
        device_obj,
        requested_dtype,
        trust_remote_code=trust_remote_code,
    )
    config = _target_config(model)
    num_layers = int(getattr(config, "num_hidden_layers", 0))
    hidden_size = int(getattr(config, "hidden_size", 0))
    if num_layers < 1 or hidden_size < 1:
        raise ValueError("local target model config lacks num_hidden_layers/hidden_size")
    if target_layer_ids is None:
        if num_draft_layers is None or num_draft_layers < 1:
            raise ValueError(
                "capture_dataset requires target_layer_ids or a positive "
                "num_draft_layers"
            )
        from .model import build_target_layer_ids

        target_layer_ids = build_target_layer_ids(num_layers, num_draft_layers)
    layer_ids = _resolve_layer_ids(target_layer_ids, num_layers)

    examples = iter(prepared_examples)
    try:
        first_example = next(examples)
    except StopIteration as exc:
        raise ValueError("capture_dataset requires at least one prepared example") from exc

    def prepare_example(example: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        first_ids = _as_1d_tensor(example["input_ids"], "input_ids")
        first_mask = _as_1d_tensor(example["loss_mask"], "loss_mask")
        if first_ids.shape[0] != first_mask.shape[0]:
            raise ValueError(
                "prepared example has mismatched sequence lengths: "
                f"input_ids={first_ids.shape[0]}, loss_mask={first_mask.shape[0]}"
            )
        return (
            _canonicalize_input_ids(first_ids)[:max_length],
            _canonicalize_loss_mask(first_mask)[:max_length],
        )

    first_ids, first_mask = prepare_example(first_example)
    manifest = FeatureManifest(
        model_id=str(model_path),
        revision=getattr(config, "_commit_hash", None),
        tokenizer_id=tokenizer_id,
        prompt_contract=dict(prompt_contract) if prompt_contract is not None else None,
        layer_ids=layer_ids,
        hidden_size=hidden_size,
        max_length=max_length,
        input_ids_dtype=_dtype_name(first_ids.dtype),
        loss_mask_dtype=_dtype_name(first_mask.dtype),
        hidden_states_dtype=_dtype_name(requested_dtype),
    )

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.new-", dir=destination.parent)
    )
    try:
        for index, example in enumerate(chain((first_example,), examples)):
            input_ids, loss_mask = prepare_example(example)
            hidden_states = _capture_hidden_feature(
                model, input_ids, layer_ids, device_obj, requested_dtype
            )
            if hidden_states.shape != (input_ids.shape[0], manifest.feature_width):
                raise ValueError(
                    "captured feature width mismatch: "
                    f"got {tuple(hidden_states.shape)}, expected "
                    f"({input_ids.shape[0]}, {manifest.feature_width})"
                )
            record = {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "hidden_states": hidden_states,
            }
            validate_feature_record(record, manifest)
            _atomic_torch_save(record, staging / f"feature_{index:08d}.pt")

        # Publish only a complete generation.  A loader either sees the old
        # manifest/generation or the new one, never a partial mixture.
        _publish_feature_generation(staging, destination, manifest)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _layer_ids_argument(value: str) -> list[int]:
    try:
        layer_ids = [int(part) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--target-layer-ids must be comma-separated integers"
        ) from exc
    if not layer_ids:
        raise argparse.ArgumentTypeError("--target-layer-ids must not be empty")
    return layer_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture frozen Qwen hidden states for offline DFlash training"
    )
    parser.add_argument("--input", required=True, help="Teacher-trajectory JSONL")
    parser.add_argument("--output", required=True, help="Feature-store directory")
    parser.add_argument("--target-model-path", required=True)
    layer_selection = parser.add_mutually_exclusive_group(required=True)
    layer_selection.add_argument("--target-layer-ids", type=_layer_ids_argument)
    layer_selection.add_argument("--num-draft-layers", type=int)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--max-source-tokens", required=True, type=int)
    parser.add_argument("--max-summary-tokens", required=True, type=int)
    parser.add_argument("--chat-template", default="qwen3")
    parser.add_argument("--prompt-template", default=DEFAULT_SUMMARY_PROMPT_TEMPLATE)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Capture one immutable feature generation from a local JSONL trajectory."""

    args = _parser().parse_args(argv)
    if args.max_source_tokens < 0 or args.max_summary_tokens < 1:
        raise ValueError("source budget must be non-negative and summary budget positive")
    if args.num_draft_layers is not None and args.num_draft_layers < 1:
        raise ValueError("--num-draft-layers must be positive")
    if args.max_source_tokens + args.max_summary_tokens > args.max_length:
        raise ValueError("source and summary token budgets exceed --max-length")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True,
    )
    from .prepare_data import iter_summary_examples

    prompt_contract = {
        "chat_template": args.chat_template,
        "max_source_tokens": args.max_source_tokens,
        "max_summary_tokens": args.max_summary_tokens,
        "prompt_template": args.prompt_template,
    }
    manifest = capture_dataset(
        target_model_path=args.target_model_path,
        prepared_examples=iter_summary_examples(
            args.input,
            tokenizer,
            max_length=args.max_length,
            chat_template=args.chat_template,
            max_samples=args.max_samples,
            max_source_tokens=args.max_source_tokens,
            max_summary_tokens=args.max_summary_tokens,
            prompt_template=args.prompt_template,
        ),
        output_dir=args.output,
        target_layer_ids=args.target_layer_ids,
        num_draft_layers=args.num_draft_layers,
        trust_remote_code=args.trust_remote_code,
        max_length=args.max_length,
        device=args.device,
        dtype=args.torch_dtype,
        tokenizer_id=str(Path(args.target_model_path)),
        prompt_contract=prompt_contract,
    )
    print(json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["capture_dataset", "main"]
