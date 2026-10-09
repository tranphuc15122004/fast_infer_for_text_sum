"""Model loading and shared request execution for Context-Adaptive DFlash."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import AdaptiveConfig
from .generation import generate_adaptive, generate_target_only
from .prompt import prepare_prompt
from .schema import atomic_json, sha256_file, validate_request, validate_round
from .statistics import AdaptiveStatistics


def load_corpus(*, data_file: Path | None, data_dir: Path | None) -> list[dict[str, Any]]:
    from common.data_loader import load_records

    if bool(data_file) == bool(data_dir):
        raise ValueError("provide exactly one of --data-file or --data-dir")
    paths = [Path(data_file)] if data_file else sorted(Path(data_dir).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no JSONL datasets found in {data_dir}")
    samples: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        records = load_records(path)
        for sample in records:
            raw = dict(sample.get("raw") or {})
            dataset = str(raw.get("dataset") or path.stem)
            sample_id = str(sample.get("id"))
            sample["id"] = sample_id
            sample["dataset"] = dataset
            sample["_cad_record_key"] = f"{dataset}::{sample_id}"
            sample["raw"] = raw
            samples.append(sample)
    if not samples:
        raise ValueError("dataset has no records")
    return samples


def split_records(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        raw = sample.get("raw") or {}
        rows.append({
            "id": str(sample["_cad_record_key"]),
            "sample_id": str(sample.get("id", "")),
            "dataset": str(sample.get("dataset", "unknown")),
            "context": str(raw.get("context") or ""),
            "source_split": str(raw.get("source_split", "unknown")),
            "source_index": str(raw.get("source_index", raw.get("document_id", sample.get("id")))),
        })
    return rows


def validate_dflash_target_layers(
    target_layer_ids: Sequence[int],
    *,
    target_hidden_layers: int,
    draft_num_target_layers: int,
) -> None:
    """Validate the target depth and the drafter's selected feature layers."""
    target_depth = int(target_hidden_layers)
    if int(draft_num_target_layers) != target_depth:
        raise ValueError("DFlash num_target_layers does not match target num_hidden_layers")
    if not target_layer_ids or any(
        not isinstance(layer, int) or layer < 0 or layer >= target_depth
        for layer in target_layer_ids
    ):
        raise ValueError("DFlash target_layer_ids are outside the target model")
    if len(set(target_layer_ids)) != len(target_layer_ids):
        raise ValueError("DFlash target_layer_ids must be unique")


def select_split(
    samples: Sequence[dict[str, Any]],
    split_manifest: Mapping[str, Any] | None,
    split: str | None,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    group_strata = {
        str(group.get("source_group_id")): str(group.get("allocation_stratum", "unknown"))
        for group in (split_manifest or {}).get("groups", [])
    }
    for sample in samples:
        key = str(sample["_cad_record_key"])
        assigned = (split_manifest or {}).get("record_split", {}).get(key, "unsplit")
        if split is not None and assigned != split:
            continue
        sample["_cad_split"] = assigned
        sample["_cad_source_group_id"] = (split_manifest or {}).get("record_group", {}).get(key)
        if sample["_cad_source_group_id"] is None:
            context = str((sample.get("raw") or {}).get("context", ""))
            fallback_key = context if context.strip() else str(sample.get("_cad_record_key", key))
            sample["_cad_source_group_id"] = hashlib.sha256(fallback_key.encode()).hexdigest()
        sample["_cad_allocation_stratum"] = group_strata.get(
            str(sample["_cad_source_group_id"]), str(sample.get("dataset", "unknown"))
        )
        selected.append(sample)
    if not selected:
        raise ValueError(f"no samples selected for split={split!r}")
    if max_samples and len(selected) > max_samples:
        # A small calibration/dev cap must still cover datasets and independent
        # source groups. Visit one representative per group in round-robin
        # dataset order before adding additional queries from an already chosen
        # source document.
        by_dataset: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for sample in selected:
            dataset = str(sample.get("dataset", "unknown"))
            group_id = str(sample.get("_cad_source_group_id"))
            by_dataset.setdefault(dataset, {}).setdefault(group_id, []).append(sample)
        group_queues = {
            dataset: [by_dataset[dataset][group] for group in sorted(by_dataset[dataset])]
            for dataset in sorted(by_dataset)
        }
        selected_capped: list[dict[str, Any]] = []
        max_groups = max((len(groups) for groups in group_queues.values()), default=0)
        for group_index in range(max_groups):
            for dataset in sorted(group_queues):
                groups = group_queues[dataset]
                if group_index < len(groups) and len(selected_capped) < max_samples:
                    selected_capped.append(groups[group_index][0])
        if len(selected_capped) < max_samples:
            chosen = {str(row["_cad_record_key"]) for row in selected_capped}
            for dataset in sorted(group_queues):
                for group in group_queues[dataset]:
                    for sample in group[1:]:
                        if str(sample["_cad_record_key"]) not in chosen:
                            selected_capped.append(sample)
                            chosen.add(str(sample["_cad_record_key"]))
                            if len(selected_capped) >= max_samples:
                                break
                    if len(selected_capped) >= max_samples:
                        break
                if len(selected_capped) >= max_samples:
                    break
        selected = selected_capped
    return selected


def _asset_signature(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    # Include every regular file: model and tokenizer snapshots may use
    # SentencePiece, vocab/merges, Jinja templates, or custom config assets.
    files = [resolved] if resolved.is_file() else sorted(
        item for item in resolved.rglob("*") if item.is_file()
    )
    digest = hashlib.sha256()
    total_bytes = 0
    file_hashes: list[dict[str, Any]] = []
    hashed_bytes = 0
    next_progress = 2 * 1024**3
    for item in files:
        relative = item.name if resolved.is_file() else item.relative_to(resolved).as_posix()
        stat = item.stat()
        total_bytes += stat.st_size
        print(f"[provenance] SHA-256 {resolved.name}/{relative} ({stat.st_size} bytes)", flush=True)
        file_digest = hashlib.sha256()
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                file_digest.update(chunk)
                hashed_bytes += len(chunk)
                if hashed_bytes >= next_progress:
                    print(f"[provenance] hashed {hashed_bytes / (1024**3):.1f} GiB for {resolved}", flush=True)
                    next_progress += 2 * 1024**3
        content_hash = file_digest.hexdigest()
        file_hashes.append({"path": relative, "bytes": stat.st_size, "sha256": content_hash})
        digest.update(f"{relative}\0{stat.st_size}\0{content_hash}\n".encode())
    return {
        "path": str(resolved),
        "file_count": len(files),
        "bytes": total_bytes,
        "content_sha256": digest.hexdigest(),
        "files": file_hashes,
    }


def validate_rotary_compatibility(target_config: Any, draft_config: Any) -> None:
    def rope_parameters(config: Any) -> dict[str, Any]:
        value = getattr(config, "rope_parameters", None)
        if value is None:
            value = getattr(config, "rope_scaling", None)
        return dict(value) if isinstance(value, Mapping) else {}

    target_rope = rope_parameters(target_config)
    draft_rope = rope_parameters(draft_config)
    if target_rope != draft_rope:
        raise ValueError("target and DFlash RoPE parameter configs differ")
    target_theta = target_rope.get("rope_theta", getattr(target_config, "rope_theta", None))
    draft_theta = draft_rope.get("rope_theta", getattr(draft_config, "rope_theta", None))
    if target_theta is not None and draft_theta is not None and float(target_theta) != float(draft_theta):
        raise ValueError("target and DFlash rope_theta values differ")
    target_positions = int(getattr(target_config, "max_position_embeddings", 0))
    draft_positions = int(getattr(draft_config, "max_position_embeddings", 0))
    if target_positions and draft_positions and target_positions != draft_positions:
        raise ValueError("target and DFlash max_position_embeddings differ")
    rope_type = target_rope.get("rope_type", target_rope.get("type"))
    if str(rope_type).lower() in {"dynamic", "dynamic_ntk"}:
        raise ValueError("dynamic RoPE is unsupported because cached draft context keys would become stale")


def _resolve_attention_backend(torch: Any, automatic_backend: str) -> str:
    """Resolve an optional, explicitly requested backend for this benchmark.

    The DFlash adapter selects FA4 automatically on Blackwell and FA2 on
    Hopper/Ada.  An explicit override lets us reproduce a server FA4 runtime
    on Hopper for controlled backend comparisons.  Never silently downgrade
    an explicit FA4 request to SDPA or FA2.
    """
    requested = os.environ.get("CAD_ATTENTION_BACKEND", "auto").strip().lower()
    if requested in {"", "auto"}:
        return automatic_backend
    if requested not in {"sdpa", "flash_attention_2", "flash_attention_4"}:
        raise ValueError(
            "CAD_ATTENTION_BACKEND must be auto, sdpa, flash_attention_2, "
            "or flash_attention_4"
        )
    if requested == "sdpa":
        return requested

    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    if requested == "flash_attention_4":
        # FA4 supports Hopper (SM90) and Blackwell (SM100/SM110).
        if capability[0] < 9:
            raise RuntimeError(
                f"FlashAttention-4 requires Hopper/Blackwell; got capability {capability}"
            )
        from common.vanilla_inference import (
            _install_flash_attention_4_cutlass_compat,
            _probe_flash_attention_4,
        )

        _install_flash_attention_4_cutlass_compat()
        available, reason = _probe_flash_attention_4()
        if not available:
            raise RuntimeError(
                "CAD_ATTENTION_BACKEND=flash_attention_4 was requested, but "
                f"its runtime probe failed: {reason}"
            )
        return requested

    if capability[0] >= 10:
        raise RuntimeError("FlashAttention-2 is unsupported on Blackwell; request FA4 or SDPA")
    try:
        import flash_attn  # noqa: F401
    except Exception as exc:
        raise RuntimeError(f"FlashAttention-2 was requested but could not be imported: {exc}") from exc
    return requested


def load_runtime(target_path: str, draft_path: str | None, *, load_draft: bool = True) -> dict[str, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parents[3]
    dflash_root = root / "externals" / "dflash"
    scripts_root = root / "scripts"
    for path in (str(scripts_root), str(dflash_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    if not torch.cuda.is_available():
        raise RuntimeError("model phases require CUDA; this development host is CPU-only")
    from infer_dflash import _dtype_and_attention, normalize_generation_token_ids
    from dflash.model import DFlashDraftModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    dtype, attention_backend = _dtype_and_attention()
    attention_backend = _resolve_attention_backend(torch, attention_backend)
    tokenizer = AutoTokenizer.from_pretrained(target_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_config = AutoConfig.from_pretrained(target_path, local_files_only=True)
    if target_config.model_type != "qwen3":
        raise ValueError("Context-Adaptive DFlash V1 currently requires a Qwen3 target model")
    target_token_id_changes = normalize_generation_token_ids(target_config, tokenizer)
    device = torch.device("cuda:0")
    target = AutoModelForCausalLM.from_pretrained(
        target_path,
        dtype=dtype,
        attn_implementation=attention_backend,
        low_cpu_mem_usage=True,
        config=target_config,
        local_files_only=True,
    ).to(device).eval()
    draft = None
    draft_config = None
    if load_draft:
        if not draft_path:
            raise ValueError("--draft-model is required for DFlash variants")
        draft_config = AutoConfig.from_pretrained(draft_path, local_files_only=True)
        if draft_config.model_type != "qwen3":
            raise ValueError("Context-Adaptive DFlash V1 currently requires a Qwen3 DFlash checkpoint")
        validate_rotary_compatibility(target_config, draft_config)
        draft_token_id_changes = normalize_generation_token_ids(draft_config, tokenizer)
        draft = DFlashDraftModel.from_pretrained(
            draft_path,
            dtype=dtype,
            attn_implementation=attention_backend,
            low_cpu_mem_usage=True,
            config=draft_config,
            local_files_only=True,
        ).to(device).eval()
        if int(draft.config.vocab_size) != int(target.config.vocab_size):
            raise ValueError("target and DFlash vocabulary sizes differ")
        if int(draft.config.hidden_size) != int(target.config.hidden_size):
            raise ValueError("target and DFlash hidden sizes differ")
        validate_dflash_target_layers(
            draft.target_layer_ids,
            target_hidden_layers=target.config.num_hidden_layers,
            draft_num_target_layers=draft.config.num_target_layers,
        )
        if int(draft.block_size) < 2:
            raise ValueError("DFlash checkpoint block_size must be at least 2")
    else:
        draft_token_id_changes = {}
    torch.cuda.synchronize(device)
    target_signature = _asset_signature(target_path)
    draft_signature = _asset_signature(draft_path) if draft_path and load_draft else None
    tokenizer_signature = {
        "name_or_path": str(getattr(tokenizer, "name_or_path", target_path)),
        "vocab_size": len(tokenizer),
        "special_tokens": {
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id,
        },
    }
    runtime_signature = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "dtype": str(dtype),
        "attention_backend": attention_backend,
        "requested_attention_backend": os.environ.get("CAD_ATTENTION_BACKEND", "auto"),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        # Runtime identity is compared between AR and speculative variants.
        # Keep target normalization here; bind draft-specific normalization to
        # the draft signature so AR and DFlash remain pair-comparable.
        "target_token_id_normalization": target_token_id_changes,
    }
    draft_model_signature = None
    if draft_signature is not None:
        draft_model_signature = hashlib.sha256(json.dumps({
            "asset_content_sha256": draft_signature["content_sha256"],
            "token_id_normalization": draft_token_id_changes,
        }, sort_keys=True, default=str).encode()).hexdigest()
    for signature in (target_signature, draft_signature, tokenizer_signature, runtime_signature):
        if signature is not None:
            signature["signature"] = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()
    return {
        "torch": torch,
        "target": target,
        "draft": draft,
        "tokenizer": tokenizer,
        "dtype": str(dtype),
        "attention_backend": attention_backend,
        "device": str(device),
        "target_signature": target_signature["signature"],
        "draft_signature": draft_model_signature,
        "tokenizer_signature": tokenizer_signature["signature"],
        "token_id_normalization": {
            "target": target_token_id_changes,
            "draft": draft_token_id_changes,
        },
        "runtime_signature": runtime_signature,
        "target_asset": target_signature,
        "draft_asset": draft_signature,
    }


def _json_hash(values: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(value) for value in values], separators=(",", ":")).encode()).hexdigest()


def request_record(
    *,
    result: Any,
    sample: Mapping[str, Any],
    input_ids: Any,
    layout: Any,
    prompt_hash: str,
    variant: str,
    run_id: str,
    repetition: int,
    model_path: str,
    draft_path: str | None,
    runtime: Mapping[str, Any],
    temperature: float,
    max_new_tokens: int,
    fixed_output_tokens: int | None,
    config_hash: str,
    baseline_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    from common import rouge

    output_ids = [int(value) for value in result.output_ids.reshape(-1).tolist()]
    text = runtime["tokenizer"].decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    reference_text = sample.get("reference")
    if fixed_output_tokens is None:
        text = text.strip()
    rounds = result.rounds
    counters = result.counters
    timings = result.timings
    acceptance = counters.get("acceptance_rate")
    exact: bool | None
    if temperature == 0.0 and baseline_ids is not None:
        exact = output_ids == [int(value) for value in baseline_ids]
    elif variant == "ar" and temperature == 0.0:
        exact = True
    else:
        exact = None
    output_scope = "fixed_tokens" if fixed_output_tokens is not None else "natural_eos"
    record: dict[str, Any] = {
        "schema_version": "cadflash.request.v1",
        "type": "sample",
        "method": "context_adaptive_dflash",
        "variant": variant,
        "run_id": run_id,
        "sample_id": str(sample["id"]),
        "dataset": str(sample.get("dataset", "unknown")),
        "split": str(sample.get("_cad_split", "unsplit")),
        "source_group_id": str(sample.get("_cad_source_group_id", sample["id"])),
        "allocation_stratum": str(sample.get("_cad_allocation_stratum", sample.get("dataset", "unknown"))),
        "repetition": repetition,
        "model": model_path,
        "draft_model": draft_path,
        "model_signature": runtime["target_signature"],
        "draft_signature": runtime.get("draft_signature"),
        "tokenizer_signature": runtime["tokenizer_signature"],
        "runtime_signature": runtime["runtime_signature"]["signature"],
        "dtype": runtime["dtype"],
        "attention_backend": runtime["attention_backend"],
        "device": runtime["device"],
        "input_tokens": int(input_ids.shape[1]),
        "retained_tokens": int(input_ids.shape[1]),
        "output_tokens": int(result.output_tokens),
        "batch_size": 1,
        "selector_latency_ms": sum(float(timings.get(key) or 0.0) for key in ("signal_latency_ms", "selection_latency_ms", "controller_latency_ms")),
        "ttft_ms": timings.get("ttft_ms"),
        "prefill_ms": timings.get("prefill_ms"),
        "decode_ms": timings.get("decode_ms"),
        "e2e_ms": timings.get("e2e_ms"),
        "tpot_ms": timings.get("tpot_ms"),
        "throughput_tok_s": timings.get("throughput_tok_s"),
        "qps": timings.get("qps"),
        "peak_memory_gb": counters.get("peak_memory_gib"),
        "memory_unit": "gib",
        "avg_accept_length": counters.get("mean_accept_length"),
        "acceptance_rate": acceptance,
        "effective_accept_length": counters.get("mean_effective_accept_length"),
        "effective_acceptance_rate": counters.get("effective_acceptance_rate"),
        "rejected_draft_ratio": (1.0 - float(acceptance)) if acceptance is not None else None,
        "draft_latency_ms": timings.get("draft_latency_ms"),
        "verification_latency_ms": timings.get("verification_latency_ms"),
        "signal_latency_ms": timings.get("signal_latency_ms"),
        "selection_latency_ms": timings.get("selection_latency_ms"),
        "controller_latency_ms": timings.get("controller_latency_ms"),
        "bank_update_latency_ms": timings.get("bank_update_latency_ms"),
        "draft_prefill_ms": timings.get("draft_prefill_ms"),
        "draft_tokens_proposed": counters.get("draft_tokens_proposed"),
        "draft_tokens_accepted": counters.get("draft_tokens_accepted"),
        "useful_draft_tokens_accepted": counters.get("useful_draft_tokens_accepted"),
        "mean_draft_context_tokens": counters.get("mean_draft_context_tokens"),
        "physical_key_tokens": sum(
            sum(int(value) for value in row.get("physical_context_tokens_by_layer", []))
            for row in rounds
        ),
        "mean_gamma": counters.get("mean_gamma"),
        "fallback_rounds": counters.get("fallback_rounds"),
        "dense_refresh_rounds": counters.get("dense_refresh_rounds"),
        "dense_bank_bytes": counters.get("dense_bank_bytes"),
        "gather_bytes": counters.get("gather_bytes"),
        "round_count": counters.get("round_count"),
        "processed_commits": counters.get("processed_commits"),
        "final_pending_emitted_count": counters.get("final_pending_emitted_count"),
        "trimmed_commits": counters.get("trimmed_commits"),
        "output_accounting_valid": counters.get("output_accounting_valid"),
        "stopped_by": counters.get("stopped_by"),
        "output_cap": max_new_tokens,
        "speed_output_tokens": fixed_output_tokens,
        "output_scope": output_scope,
        "generation_temperature": float(temperature),
        "prompt_hash": prompt_hash,
        "source_span_available": bool(layout.source_available),
        "source_span_reason": layout.source_reason,
        "source_token_count": len(layout.source_positions),
        "source_chunk_count": len(layout.source_chunks),
        "output_ids_hash": _json_hash(output_ids),
        "config_hash": config_hash,
        "greedy_exact_match": exact,
        "correctness_status": "target_reference" if variant == "ar" else ("pending_paired_ar" if exact is None else ("pass" if exact else "mismatch")),
        "status": result.status,
        "error": result.error,
        "text": text,
        "reference_output": reference_text,
        "measurement_scope": "full_e2e",
    }
    if reference_text:
        rouge.add_rouge(record, text, str(reference_text))
    return record


def calibration_model_signature(runtime: Mapping[str, Any], config: AdaptiveConfig, *, temperature: float, max_new_tokens: int) -> dict[str, Any]:
    from .calibration import calibration_signature

    return calibration_signature(
        target_signature=str(runtime["target_signature"]),
        draft_signature=str(runtime["draft_signature"]),
        tokenizer_signature=str(runtime["tokenizer_signature"]),
        runtime_signature=str(runtime["runtime_signature"]["signature"]),
        config=config,
        sampling_temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def data_manifest(samples: Sequence[Mapping[str, Any]], source_paths: Sequence[Path]) -> dict[str, Any]:
    return {
        "data_files": [
            {"path": str(Path(path).resolve()), "sha256": sha256_file(Path(path)), "bytes": Path(path).stat().st_size}
            for path in source_paths
        ],
        "record_count": len(samples),
        "sample_ids_hash": hashlib.sha256(
            "\n".join(str(sample["_cad_record_key"]) for sample in samples).encode()
        ).hexdigest(),
    }


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_json(path, {"schema_version": "cadflash.manifest.v1", **dict(payload)})
