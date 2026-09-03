#!/usr/bin/env python3
"""E1 Target-KV representation sufficiency probe.

Extraction is model-backed and local-only; training consumes the compact
feature cache and can therefore be repeated without another Qwen forward.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common.benchmark_data import render_prompt  # noqa: E402

from .e0_dflash_failure_map import (  # noqa: E402
    SelectiveHiddenTarget,
    _chat_prompt,
    chunk_spans,
)
from .trace_target import _bottom_right_causal_mask  # noqa: E402
from .target_kv_e1 import (  # noqa: E402
    MemoryBlockProbe,
    anchor_positions,
    pool_representation_dict,
    probe_metrics,
    required_capture_layers,
    split_feature_rows_by_document,
    wrong_document_indices,
)
from .target_kv_experiments import context_bucket  # noqa: E402


REPRESENTATIONS = (
    "hidden",
    "hidden_sequence",
    "multi_layer_hidden",
    "kv",
    "kv_shuffled",
    "kv_recent",
    "kv_wrong_document",
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


def _prompt_from_record(record: Mapping[str, Any]) -> str:
    if record.get("dataset") in {"gov_report", "qmsum", "multi_news", "lcc", "repobench-p"}:
        return render_prompt(record)
    for key in ("prompt", "document", "text", "instruction", "question"):
        if record.get(key):
            return str(record[key])
    raise ValueError("record has no usable prompt")


def _encode(tokenizer: Any, content: str, *, device: Any) -> Any:
    rendered = _chat_prompt(tokenizer, content)
    encoded = tokenizer(rendered, return_tensors="pt")
    input_ids = encoded.input_ids
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("tokenizer must return [1, sequence]")
    return input_ids.to(device), rendered


def _model_call(model: Any, **kwargs: Any) -> Any:
    try:
        return model(logits_to_keep=1, **kwargs)
    except TypeError as exc:
        if "logits_to_keep" not in str(exc):
            raise
        return model(**kwargs)


def _prefill_target(
    target: Any,
    input_ids: Any,
    *,
    layer_ids: Sequence[int],
    chunk_size: int,
) -> tuple[Any, Any, list[list[Any]], Any]:
    """Prefill target and return final hidden, cache, and selected layers."""

    import torch
    from transformers import DynamicCache

    cache = DynamicCache()
    num_layers = int(getattr(target.config, "num_hidden_layers"))
    capture_layer_ids = required_capture_layers(layer_ids, num_layers=num_layers)
    target_proxy = SelectiveHiddenTarget(target, capture_layer_ids)
    final_chunks: list[Any] = []
    selected_chunks: list[list[Any]] = [[] for _ in layer_ids]
    last_output = None
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    model_dtype = getattr(target, "dtype", torch.float16)
    for start, end in chunk_spans(int(input_ids.shape[1]), chunk_size):
        mask = _bottom_right_causal_mask(
            end - start,
            start,
            dtype=model_dtype,
            device=input_ids.device,
        )
        last_output = target_proxy(
            input_ids=input_ids[:, start:end],
            position_ids=position_ids[:, start:end],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
            attention_mask=mask,
        )
        cache = last_output.past_key_values
        # Do not retain token-wise hidden states on the T4 while the full KV
        # cache is still live.  E1 consumes a compact CPU feature cache, so
        # moving each chunk here avoids an artificial GPU peak at long prefix
        # lengths without changing the representation values.
        final_chunks.append(last_output.hidden_states[num_layers].detach().cpu())
        for index, layer_id in enumerate(layer_ids):
            selected_chunks[index].append(
                last_output.hidden_states[layer_id + 1].detach().cpu()
            )
    if last_output is None:
        raise ValueError("input_ids must not be empty")
    return torch.cat(final_chunks, dim=1), cache, selected_chunks, last_output


def _kv_sequence(cache: Any, layer_ids: Sequence[int]) -> Any:
    import torch

    chunks = []
    for layer_id in layer_ids:
        key, value = cache[layer_id]
        key = key[0].transpose(0, 1).reshape(key.shape[-2], -1)
        value = value[0].transpose(0, 1).reshape(value.shape[-2], -1)
        chunks.append(torch.cat((key, value), dim=-1))
    return torch.cat(chunks, dim=-1).detach().cpu()


def _generate_labels(target: Any, cache: Any, logits: Any, *, horizon: int, device: Any) -> list[int]:
    import torch

    labels: list[int] = []
    target_proxy = target
    for _ in range(horizon):
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        labels.append(int(next_token[0, 0].item()))
        output = _model_call(
            target_proxy,
            input_ids=next_token.to(device),
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=False,
        )
        cache = output.past_key_values
        logits = output.logits
    return labels


def _build_representations(
    target: Any,
    input_ids: Any,
    *,
    layer_ids: Sequence[int],
    chunk_size: int,
    max_memory_tokens: int,
    interface_dim: int,
    horizon: int,
    seed: int,
) -> tuple[dict[str, Any], list[int]]:
    import torch

    final_hidden, cache, selected_chunks, last_output = _prefill_target(
        target,
        input_ids,
        layer_ids=layer_ids,
        chunk_size=chunk_size,
    )
    hidden_sequence = final_hidden[0]
    multi_layer_hidden = torch.cat(
        [torch.cat(chunks, dim=1)[0] for chunks in selected_chunks], dim=-1
    )
    kv = _kv_sequence(cache, layer_ids)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    shuffled = kv[torch.randperm(kv.shape[0], generator=generator)]
    recent = torch.zeros_like(kv)
    recent[-max(1, kv.shape[0] // 4) :] = kv[-max(1, kv.shape[0] // 4) :]
    raw = {
        "hidden": hidden_sequence[-1:].detach().cpu(),
        "hidden_sequence": hidden_sequence.detach().cpu(),
        "multi_layer_hidden": multi_layer_hidden.detach().cpu(),
        "kv": kv.detach().cpu(),
        "kv_shuffled": shuffled.detach().cpu(),
        "kv_recent": recent.detach().cpu(),
    }
    features, masks = pool_representation_dict(
        raw,
        max_memory_tokens=max_memory_tokens,
        interface_dim=interface_dim,
    )
    labels = _generate_labels(
        target,
        cache,
        last_output.logits,
        horizon=horizon,
        device=input_ids.device,
    )
    return features, masks, labels


def extract_features(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "experiment": "E1_target_kv_representation_probe",
        "mode": "extract",
        "status": "RUNNING",
        "target_model": str(args.target_model),
        "data_file": str(args.data_file),
        "max_samples": args.max_samples,
        "start_index": args.start_index,
        "anchors_per_document": args.anchors_per_document,
        "max_input_tokens": args.max_input_tokens,
        "max_memory_tokens": args.max_memory_tokens,
        "interface_dim": args.interface_dim,
        "horizon": args.horizon,
        "seed": args.seed,
        "command": " ".join(sys.argv),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; E1 extraction requires T4 host runtime")
        target = AutoModelForCausalLM.from_pretrained(
            args.target_model,
            dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).to("cuda:0").eval()
        tokenizer = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        layer_count = int(getattr(target.config, "num_hidden_layers", 36))
        layer_ids = [layer_id for layer_id in args.layer_ids if layer_id < layer_count]
        if not layer_ids:
            raise ValueError("no valid layer ids")
        source_rows = _jsonl(Path(args.data_file))
        source_rows = source_rows[args.start_index :]
        if args.max_samples is not None:
            source_rows = source_rows[: args.max_samples]
        feature_rows: dict[str, list[np.ndarray]] = {name: [] for name in REPRESENTATIONS}
        masks: dict[str, list[np.ndarray]] = {name: [] for name in REPRESENTATIONS}
        labels: list[list[int]] = []
        metadata: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        anchor_index = 0
        for source_index, raw in enumerate(source_rows):
            document_id = str(raw.get("id", source_index))
            try:
                content = _prompt_from_record(raw)
                input_ids, _rendered = _encode(tokenizer, content, device="cuda:0")
                input_length = int(input_ids.shape[1])
                if input_length > args.max_input_tokens:
                    excluded.append({
                        "document_id": document_id,
                        "status": "excluded",
                        "reason": "input_exceeds_t4_e1_cap",
                        "input_tokens": input_length,
                    })
                    continue
                for prefix_length in anchor_positions(
                    input_length,
                    count=args.anchors_per_document,
                    minimum_prefix=args.minimum_prefix,
                ):
                    prefix = input_ids[:, :prefix_length]
                    with torch.inference_mode():
                        features, representation_masks, target_labels = _build_representations(
                            target,
                            prefix,
                            layer_ids=layer_ids,
                            chunk_size=args.prefill_chunk_size,
                            max_memory_tokens=args.max_memory_tokens,
                            interface_dim=args.interface_dim,
                            horizon=args.horizon,
                            seed=args.seed + anchor_index,
                        )
                    for name in REPRESENTATIONS:
                        if name == "kv_wrong_document":
                            continue
                        feature_rows[name].append(features[name].numpy().astype(np.float16))
                        masks[name].append(representation_masks[name].numpy().astype(np.float16))
                    labels.append(target_labels)
                    metadata.append({
                        "document_id": document_id,
                        "dataset": str(raw.get("dataset", "unknown")),
                        "source_index": raw.get("source_index", source_index),
                        "anchor_index": anchor_index,
                        "prefix_tokens": prefix_length,
                        "input_tokens": prefix_length,
                        "context_bucket": context_bucket(prefix_length),
                    })
                    anchor_index += 1
            except Exception as exc:
                excluded.append({
                    "document_id": document_id,
                    "status": "excluded",
                    "reason": "feature_extraction_error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=3),
                })
                torch.cuda.empty_cache()
        if not labels:
            _write_jsonl(output_dir / "excluded.jsonl", excluded)
            raise RuntimeError("no feature rows were extracted")
        arrays: dict[str, np.ndarray] = {
            "labels": np.asarray(labels, dtype=np.int64),
        }
        for name in REPRESENTATIONS:
            if name == "kv_wrong_document":
                indices = wrong_document_indices([str(row["document_id"]) for row in metadata])
                arrays[name] = np.stack(feature_rows["kv"])[indices]
                arrays[f"{name}_mask"] = np.stack(masks["kv"])[indices]
            else:
                arrays[name] = np.stack(feature_rows[name])
                arrays[f"{name}_mask"] = np.stack(masks[name])
        np.savez_compressed(output_dir / "features.npz", **arrays)
        _write_jsonl(output_dir / "metadata.jsonl", metadata)
        _write_jsonl(output_dir / "excluded.jsonl", excluded)
        manifest.update({
            "status": "ok",
            "feature_rows": len(metadata),
            "excluded_rows": len(excluded),
            "representation_names": list(REPRESENTATIONS),
            "layer_ids": layer_ids,
            "vocab_size": int(target.config.vocab_size),
            "hardware": {
                "device": torch.cuda.get_device_name(0),
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
        })
        del target
        torch.cuda.empty_cache()
    except Exception as exc:
        manifest.update({
            "status": "UNAVAILABLE",
            "reason": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=5),
        })
    finally:
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return manifest


def _partition_indices(metadata: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    rows = [{"document_id": str(row["document_id"]), "index": index} for index, row in enumerate(metadata)]
    train, dev, test = split_feature_rows_by_document(rows)
    return {
        "train": [int(row["index"]) for row in train],
        "dev": [int(row["index"]) for row in dev],
        "test": [int(row["index"]) for row in test],
    }


def train_probes(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    output_dir = Path(args.output_dir)
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    arrays = np.load(output_dir / "features.npz")
    metadata = _jsonl(output_dir / "metadata.jsonl")
    partitions = _partition_indices(metadata)
    device = torch.device("cuda:0" if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    vocab_size = int(manifest["vocab_size"])
    result: dict[str, Any] = {
        "experiment": "E1_target_kv_representation_probe",
        "status": "ok",
        "device": str(device),
        "partitions": {name: len(indices) for name, indices in partitions.items()},
        "representations": {},
    }
    for name in REPRESENTATIONS:
        all_features = torch.from_numpy(arrays[name]).float()
        all_masks = torch.from_numpy(arrays[f"{name}_mask"]).float()
        all_labels = torch.from_numpy(arrays["labels"]).long()
        train_idx = partitions["train"]
        test_idx = partitions["test"]
        if not train_idx or not test_idx:
            result["representations"][name] = {"status": "INCONCLUSIVE", "reason": "empty_document_split"}
            continue
        model = MemoryBlockProbe(
            interface_dim=args.interface_dim,
            hidden_dim=args.hidden_dim,
            horizon=args.horizon,
            vocab_size=vocab_size,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        dataset = TensorDataset(all_features[train_idx], all_masks[train_idx], all_labels[train_idx])
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        losses: list[float] = []
        model.train()
        for _epoch in range(args.epochs):
            for memory, mask, target_labels in loader:
                memory, mask, target_labels = memory.to(device), mask.to(device), target_labels.to(device)
                logits = model(memory, mask)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, vocab_size), target_labels.reshape(-1)
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
        model.eval()
        with torch.inference_mode():
            test_logits = model(all_features[test_idx].to(device), all_masks[test_idx].to(device)).cpu()
        metrics = probe_metrics(test_logits, all_labels[test_idx])
        metrics.update({
            "status": "ok",
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
        })
        result["representations"][name] = metrics
        torch.save(model.state_dict(), output_dir / f"probe_{name}.pt")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    (output_dir / "probe_metrics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("extract", "train", "all"), default="all")
    parser.add_argument("--target-model", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--anchors-per-document", type=int, default=4)
    parser.add_argument("--minimum-prefix", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--max-memory-tokens", type=int, default=128)
    parser.add_argument("--interface-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--layer-ids", type=int, nargs="+", default=[1, 9, 17, 25, 33])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode in {"extract", "all"}:
        if not args.target_model or not args.data_file:
            raise SystemExit("--target-model and --data-file are required for extraction")
        manifest = extract_features(args)
        if manifest.get("status") != "ok":
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
            raise SystemExit(2)
    if args.mode in {"train", "all"}:
        result = train_probes(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
