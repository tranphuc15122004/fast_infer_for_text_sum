"""Capture, preference labeling, and training workflows for AMR-DFlash."""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .artifacts import (
    atomic_torch_save,
    artifact_id,
    canonical_hash,
    load_torch,
    read_jsonl,
    safe_id,
    sha256_file,
    sha256_text,
    stable_split,
    write_jsonl,
)
from .candidates import generate_candidate_sets
from .checkpoint import contract_identity, feature_contract, save_memory_checkpoint, snapshot_fingerprint
from .budget import charge_phase, remaining_gpu_seconds
from .config import memory_config, resolve_env_path, resolve_run_root
from .core import alignment_kl_loss, greedy_acceptance, preference_loss, resolve_greedy_commit
from .evaluation import evaluate_one_fixed_state
from .inference import AMRDFlashEngine
from .runtime import load_runtime


LONG_BENCH_DATASETS = {"gov_report", "qmsum", "multi_news", "lcc", "repobench-p"}


def _input_path(config: dict[str, Any], supplied: str | Path | None) -> Path:
    if supplied:
        path = Path(supplied).expanduser()
    else:
        env_names = (
            str(config["data"].get("manifest_env", "AMR_DATA_MANIFEST")),
            str(config["data"].get("input_env", "DATA_FILE")),
        )
        configured = next((os.environ.get(name, "") for name in env_names if os.environ.get(name, "")), "")
        if not configured:
            raise ValueError(f"set --input or one of these environment variables: {', '.join(env_names)}")
        path = Path(configured).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"AMR-DFlash input JSONL not found: {path}")
    return path


def _run_root(config: dict[str, Any], supplied: str | Path | None, *, create: bool = False) -> Path:
    root = Path(supplied).expanduser() if supplied else resolve_run_root(config)
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _prompt_for(tokenizer: Any, sample: dict[str, Any]) -> str:
    raw = sample.get("raw") or {}
    prompt = str(sample.get("prompt") or "")
    if raw.get("dataset") in LONG_BENCH_DATASETS or not getattr(tokenizer, "chat_template", None):
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


def _tokenize_prompt(
    tokenizer: Any,
    sample: dict[str, Any],
    *,
    max_input_tokens: int,
) -> tuple[str, torch.Tensor]:
    from common.input_utils import truncate_input_ids

    prompt = _prompt_for(tokenizer, sample)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids
    if max_input_tokens > 0:
        input_ids = truncate_input_ids(input_ids, max_input_tokens)
    if input_ids.shape[1] < 1:
        raise ValueError(f"sample {sample.get('id')} tokenized to an empty prompt")
    return prompt, input_ids


def _model_dtype_name(config: dict[str, Any]) -> str:
    return str(config["model"].get("dtype", "bfloat16"))


def _split_for(raw: dict[str, Any], document_id: str, config: dict[str, Any]) -> str:
    explicit = str(raw.get("split", "")).lower()
    if explicit and explicit not in {"train", "validation", "holdout"}:
        raise ValueError(f"invalid document split: {explicit}")
    if explicit in {"train", "validation", "holdout"}:
        return explicit
    if str(raw.get("source_split", "")).lower() in {"validation", "valid", "dev"}:
        return "validation"
    if str(raw.get("source_split", "")).lower() in {"test", "holdout"}:
        return "holdout"
    data_cfg = config["data"]
    return stable_split(
        str(raw.get("source_document_id") or raw.get("document_id") or raw.get("source_id") or document_id),
        train_fraction=float(data_cfg.get("train_fraction", 0.8)),
        validation_fraction=float(data_cfg.get("validation_fraction", 0.1)),
    )


def validate_document_records(records: list[dict[str, Any]], config: dict[str, Any]) -> None:
    """Check the entire input before slicing, splitting or loading model weights."""
    ids: set[str] = set()
    sources: dict[str, str] = {}
    contents: dict[str, str] = {}
    for index, record in enumerate(records):
        document_id = str(record.get("id", index))
        if document_id in ids:
            raise ValueError(f"duplicate document id: {document_id}")
        ids.add(document_id)
        raw = record.get("raw") or {}
        split = _split_for(raw, document_id, config)
        source_id = str(raw.get("source_document_id") or raw.get("document_id") or raw.get("source_id") or document_id)
        content = next((str(raw[key]) for key in ("context", "document", "text")
                        if raw.get(key)), str(record.get("prompt") or ""))
        content_hash = sha256_text(" ".join(content.split()))
        identities_to_check = [(sources, source_id)]
        if content.strip():
            identities_to_check.append((contents, content_hash))
        for identities, identity in identities_to_check:
            if identity in identities and identities[identity] != split:
                raise ValueError(f"cross-split source/content overlap for document {document_id}")
            identities[identity] = split


def _capture_contract(config: dict[str, Any], *, input_sha256: str,
                      max_new_tokens: int, max_input_tokens: int,
                      max_states_per_document: int) -> dict[str, Any]:
    return {
        "version": 2,
        "input_manifest_sha256": input_sha256,
        "target_sha256": snapshot_fingerprint(resolve_env_path(config, "target_model_env"))["sha256"],
        "draft_sha256": snapshot_fingerprint(resolve_env_path(config, "draft_model_env"))["sha256"],
        "model": {key: value for key, value in config["model"].items()
                  if not key.endswith("_env")},
        "memory_config": asdict(memory_config(config)),
        "index_dim": int(config["memory"].get("index_dim", 64)),
        "max_new_tokens": max_new_tokens,
        "max_input_tokens": max_input_tokens,
        "max_states_per_document": max_states_per_document,
        "state_sampling": "evenly_spaced_including_endpoints",
        "prompt_policy": "chat_template_no_thinking_or_longbench_v1",
        "truncate_policy": "common_truncate_input_ids_v1",
        "seed": int(config["training"].get("seed", 17)),
        "train_fraction": float(config["data"].get("train_fraction", 0.8)),
        "validation_fraction": float(config["data"].get("validation_fraction", 0.1)),
        "alignment_history": config["training"].get("alignment_history"),
        "temperature": 0.0,
    }


def _validate_run_index(root: Path, *, metadata: dict[str, Any] | None = None,
                        related_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise ValueError("capture manifest is missing; recapture this run")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    contract = manifest.get("capture_contract")
    digest = manifest.get("capture_contract_sha256")
    if not contract or canonical_hash(contract) != digest:
        raise ValueError("capture contract is missing or invalid; recapture this run")
    if metadata is not None:
        saved_metadata = manifest.get("model_metadata") or {}
        for key, expected in metadata.items():
            if contract_identity(saved_metadata.get(key)) != contract_identity(expected):
                raise ValueError(f"capture runtime contract differs for {key}; recapture this run")
    documents = read_jsonl(root / "documents.jsonl")
    states = read_jsonl(root / "states.jsonl")
    document_by_id = {}
    state_by_id = {}
    for document in documents:
        key = str(document["document_id"])
        if key in document_by_id:
            raise ValueError(f"duplicate captured document id: {key}")
        if document.get("capture_contract_sha256") != digest:
            raise ValueError("document capture contract differs from manifest")
        if document.get("split") not in {"train", "validation", "holdout"}:
            raise ValueError("invalid captured document split")
        document_by_id[key] = document
    for state in states:
        key = str(state["state_id"])
        if key in state_by_id:
            raise ValueError(f"duplicate captured state id: {key}")
        document = document_by_id.get(str(state["document_id"]))
        if document is None or any(state.get(field) != document.get(field)
                                   for field in ("split", "bundle", "input_sha256", "capture_contract_sha256")):
            raise ValueError(f"state/document split or capture contract mismatch: {key}")
        state_by_id[key] = state
    for key, document in document_by_id.items():
        saved_states = [state for state in states if str(state["document_id"]) == key]
        if len(saved_states) != document.get("states_saved") or sorted(
                int(state["state_index"]) for state in saved_states) != list(range(len(saved_states))):
            raise ValueError(f"incomplete captured state index for document {key}; recapture this run")
    seen = set()
    for row in related_rows or []:
        state = state_by_id.get(str(row["state_id"]))
        if state is None or any(row.get(field) != state.get(field)
                               for field in ("document_id", "split", "capture_contract_sha256")):
            raise ValueError("label/preference state split or capture contract mismatch")
        key = (row.get("state_id"), row.get("candidate_id"),
               row.get("positive_candidate_id"), row.get("negative_candidate_id"))
        if key in seen:
            raise ValueError("duplicate candidate/label/preference identity")
        seen.add(key)
    return manifest


def select_records_for_split(
    records: list[dict[str, Any]],
    *,
    split: str,
    config: dict[str, Any],
    max_samples: int | None,
) -> list[dict[str, Any]]:
    """Filter normalized records by their stable document split, then apply the limit."""
    if split not in {"train", "validation", "holdout", "all"}:
        raise ValueError("split must be train, validation, holdout, or all")
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    validate_document_records(records, config)

    if split == "all":
        selected = list(records)
    else:
        selected = []
        for index, record in enumerate(records):
            raw = record.get("raw") or {}
            document_id = str(record.get("id", index))
            if _split_for(raw, document_id, config) == split:
                selected.append(record)
    if max_samples is not None:
        selected = selected[:max_samples]
    return selected


def select_capture_state_indices(
    num_states: int, max_states: int | None
) -> list[int]:
    """Select deterministic, evenly spaced states across one trajectory."""
    if num_states < 0:
        raise ValueError("num_states must be non-negative")
    if max_states is not None and max_states < 1:
        raise ValueError("max_states must be positive when provided")
    if num_states == 0:
        return []
    if max_states is None or max_states >= num_states:
        return list(range(num_states))
    if max_states == 1:
        return [(num_states - 1) // 2]
    return [
        int(index * (num_states - 1) / (max_states - 1) + 0.5)
        for index in range(max_states)
    ]


def capture_run(
    config: dict[str, Any],
    *,
    input_path: str | Path | None,
    run_root: str | Path | None,
    max_samples: int | None,
    max_new_tokens: int | None,
    max_input_tokens: int | None,
    max_states_per_document: int | None,
    device: torch.device,
    resume: bool = False,
) -> dict[str, Any]:
    """Capture one verifier-consistent target-feature trajectory per document."""
    source_path = _input_path(config, input_path)
    root = _run_root(config, run_root, create=True)
    phase_started = time.perf_counter()
    max_gpu_hours = float(config["training"].get("max_gpu_hours", 24))
    if remaining_gpu_seconds(root, max_gpu_hours, device=device) <= 0:
        raise RuntimeError("AMR-DFlash GPU-hour budget is exhausted before capture")
    state_index_path = root / "states.jsonl"
    document_index_path = root / "documents.jsonl"
    manifest_path = root / "manifest.json"
    input_manifest_sha256 = sha256_file(source_path)
    records = read_records_for_pipeline(source_path, max_samples=None)
    validate_document_records(records, config)
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        records = records[:max_samples]
    max_new_tokens = int(config["data"].get("max_new_tokens", 256)) if max_new_tokens is None else max_new_tokens
    max_input_tokens = int(config["data"].get("max_input_tokens", 0)) if max_input_tokens is None else max_input_tokens
    max_states_per_document = int(config["training"].get("max_states_per_document", 6)) if max_states_per_document is None else max_states_per_document
    if max_new_tokens < 1 or max_input_tokens < 0 or max_states_per_document < 1:
        raise ValueError("capture token/state caps must be positive (input cap may be zero)")
    capture_contract = _capture_contract(
        config, input_sha256=input_manifest_sha256, max_new_tokens=max_new_tokens,
        max_input_tokens=max_input_tokens, max_states_per_document=max_states_per_document,
    )
    capture_digest = canonical_hash(capture_contract)
    if not resume and (state_index_path.exists() or document_index_path.exists()):
        raise FileExistsError(
            f"capture artifacts already exist under {root}; choose another run root or pass --resume"
        )
    previous_manifest = None
    if resume:
        if not all(path.exists() for path in (manifest_path, state_index_path, document_index_path)):
            raise ValueError("capture resume requires complete manifest/document/state indexes; recapture in a new run root")
        previous_manifest = _validate_run_index(root)
        if previous_manifest.get("capture_contract_sha256") != capture_digest:
            raise ValueError("capture resume contract differs from the existing run; recapture in a new run root")
    old_states = read_jsonl(state_index_path) if resume and state_index_path.exists() else []
    old_documents = read_jsonl(document_index_path) if resume and document_index_path.exists() else []
    completed = {str(row["document_id"]) for row in old_documents}

    target_path = resolve_env_path(config, "target_model_env")
    draft_path = resolve_env_path(config, "draft_model_env")
    assert target_path is not None and draft_path is not None
    target, tokenizer, draft, memory, metadata = load_runtime(
        config, device=device, checkpoint_path=None
    )
    if previous_manifest is not None:
        _validate_run_index(root, metadata=metadata)
        for state in old_states:
            bundle = load_torch(root / state["bundle"])
            _validate_feature_bundle(bundle, metadata, draft)
            _validate_bundle_state(bundle, state)
    engine = AMRDFlashEngine(
        target,
        tokenizer,
        draft,
        memory,
        device=device,
        mode="dense",
        use_cost_gate=False,
    )
    eos_value = getattr(target.config, "eos_token_id", None)
    new_states = []
    new_documents = []
    total_state_count = 0
    total_states_observed = 0
    total_output_tokens = 0
    budget_exhausted = False
    features_dir = root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    for index, sample in enumerate(records):
        if remaining_gpu_seconds(
            root,
            max_gpu_hours,
            device=device,
            in_flight_seconds=time.perf_counter() - phase_started,
        ) <= 0:
            budget_exhausted = True
            break
        raw = sample.get("raw") or {}
        document_id = str(sample.get("id", index))
        if document_id in completed:
            continue
        prompt, input_ids = _tokenize_prompt(
            tokenizer, sample, max_input_tokens=max_input_tokens
        )
        input_hash = sha256_text(prompt)
        result = engine.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_value,
            capture_states=True,
        )
        safe_document_id = artifact_id(document_id)
        feature_relative = Path("features") / f"{safe_document_id}.pt"
        assert result.projected_context is not None
        captured_state_indices = select_capture_state_indices(
            len(result.states), max_states_per_document
        )
        captured_states = [result.states[index] for index in captured_state_indices]
        state_tensors = []
        for state in captured_states:
            state_tensors.append(
                {
                    "anchor_embedding": state.pop("anchor_embedding").to(torch.float16),
                    "context_length": int(state["context_length"]),
                }
            )
        feature_payload = {
            "format": "amr-v0-feature-trajectory",
            "document_id": document_id,
            "trajectory_ids": result.output_ids[0].detach().cpu(),
            "projected_context": result.projected_context[0].detach().cpu(),
            "states": state_tensors,
            "model_fingerprint_sha256": metadata["target_fingerprint"]["sha256"],
            "feature_layer_ids": draft.target_layer_ids,
            "feature_contract": feature_contract(metadata),
            "capture_contract_sha256": capture_digest,
        }
        atomic_torch_save(feature_payload, root / feature_relative)
        split = _split_for(raw, document_id, config)
        dataset = str(raw.get("dataset") or source_path.stem)
        for state_index, state in enumerate(captured_states):
            state_id = f"{safe_document_id}-s{state_index:05d}"
            new_states.append(
                {
                    "record_type": "state",
                    "schema_version": "amr-v0",
                    "capture_contract_sha256": capture_digest,
                    "state_id": state_id,
                    "document_id": document_id,
                    "split": split,
                    "bundle": str(feature_relative),
                    "state_index": state_index,
                    "context_length": int(state["context_length"]),
                    "anchor_id": int(state["anchor_id"]),
                    "remaining_output_budget": int(state["remaining_output_budget"]),
                    "input_sha256": input_hash,
                    "feature_hidden_size": int(result.projected_context.shape[-1]),
                    "selection_was_bypassed": bool(state["bypassed"]),
                }
            )
        reference = sample.get("reference")
        new_documents.append(
            {
                "record_type": "document",
                "schema_version": "amr-v0",
                "capture_contract_sha256": capture_digest,
                "document_id": document_id,
                "dataset": dataset,
                "split": split,
                "bundle": str(feature_relative),
                "input_sha256": input_hash,
                "input_tokens": int(input_ids.shape[1]),
                "generated_tokens": int(result.output_ids.shape[1] - input_ids.shape[1]),
                "reference": reference,
                "states_observed": len(result.states),
                "states_saved": len(captured_states),
                "termination_acceptance": result.accepted_tokens,
            }
        )
        completed.add(document_id)
        total_state_count += len(captured_states)
        total_states_observed += len(result.states)
        total_output_tokens += int(result.output_ids.shape[1] - input_ids.shape[1])
        print(
            f"[amr-capture] {index + 1}/{len(records)} id={document_id} "
            f"input={input_ids.shape[1]} output={result.output_ids.shape[1] - input_ids.shape[1]} "
            f"states={len(captured_states)}/{len(result.states)}",
            flush=True,
        )

    ledger = charge_phase(
        root,
        "capture",
        elapsed_seconds=time.perf_counter() - phase_started,
        device=device,
        status="budget_exhausted" if budget_exhausted else "complete",
    )
    all_states = old_states + new_states
    all_documents = old_documents + new_documents
    write_jsonl(
        state_index_path,
        all_states,
        summary={
            "documents": len(all_documents),
            "states": len(all_states),
            "new_states": total_state_count,
            "new_states_observed": total_states_observed,
            "status": "budget_exhausted" if budget_exhausted else "complete",
        },
    )
    write_jsonl(
        document_index_path,
        all_documents,
        summary={
            "documents": len(all_documents),
            "new_documents": len(new_documents),
            "new_output_tokens": total_output_tokens,
        },
    )
    manifest_path = root / "manifest.json"
    manifest = {
        "schema_version": "amr-v0",
        "capture_contract": capture_contract,
        "capture_contract_sha256": capture_digest,
        "model_metadata": metadata,
        "input_manifest": str(source_path.resolve()),
        "input_manifest_sha256": input_manifest_sha256,
        "target_model": metadata["target_fingerprint"],
        "draft_model": metadata["draft_fingerprint"],
        "target_layer_ids": draft.target_layer_ids,
        "block_size": draft.block_size,
        "dtype": _model_dtype_name(config),
        "attention_backend": config["model"].get("attention_backend", "sdpa"),
        "temperature": 0.0,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "max_states_per_document": max_states_per_document,
        "state_sampling": "evenly_spaced_including_endpoints",
        "seed": int(config["training"].get("seed", 17)),
        "split_policy": "explicit split/source split, otherwise stable hash by document id",
        "raw_budget": memory.config.raw_budget,
        "num_slots": memory.config.num_slots,
        "local_window": memory.config.local_window,
        "memory_config": asdict(memory.config),
        "alignment_history": config["training"].get("alignment_history"),
        "resource_ledger": str(root / "resource_ledger.json"),
        "capture_gpu_hours": ledger["phases"]["capture"]["gpu_seconds"] / 3600.0,
    }
    if previous_manifest is None:
        _write_json(manifest_path, manifest)
    return {
        "run_root": str(root),
        "documents": len(all_documents),
        "new_documents": len(new_documents),
        "states": len(all_states),
        "new_states": total_state_count,
        "new_states_observed": total_states_observed,
        "status": "budget_exhausted" if budget_exhausted else "complete",
        "manifest": str(manifest_path),
    }


def read_records_for_pipeline(path: Path, *, max_samples: int | None) -> list[dict[str, Any]]:
    from common.data_loader import load_records

    return load_records(path, max_samples=max_samples)


def _validate_feature_bundle(bundle: dict[str, Any], metadata: dict[str, Any], draft: Any) -> None:
    if bundle.get("feature_contract") != feature_contract(metadata):
        raise ValueError("feature bundle projection/execution contract differs or is missing; recapture this run")
    expected_fingerprint = metadata["target_fingerprint"]["sha256"]
    if bundle.get("model_fingerprint_sha256") != expected_fingerprint:
        raise ValueError("feature bundle target fingerprint differs from the loaded model")
    if [int(value) for value in bundle.get("feature_layer_ids", [])] != draft.target_layer_ids:
        raise ValueError("feature bundle layer ids differ from the loaded DFlash checkpoint")


def _validate_bundle_state(bundle: dict[str, Any], state: dict[str, Any]) -> None:
    if any(bundle.get(key) != state.get(key) for key in ("document_id", "capture_contract_sha256")):
        raise ValueError("feature bundle identity/capture contract differs from the state")
    index = int(state["state_index"])
    context_length = int(state["context_length"])
    if (index < 0 or index >= len(bundle["states"])
            or bundle["states"][index]["context_length"] != context_length
            or context_length > bundle["projected_context"].shape[0]
            or context_length >= bundle["trajectory_ids"].numel()
            or int(bundle["trajectory_ids"][context_length]) != int(state["anchor_id"])):
        raise ValueError("feature bundle state/anchor does not match captured trajectory")


def evaluate_fixed_run(
    config: dict[str, Any],
    *,
    run_root: str | Path | None,
    split: str,
    mode: str,
    checkpoint_path: str | Path | None,
    output_path: str | Path | None,
    max_states: int | None,
    overwrite: bool,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate one hard memory policy on captured, identical draft states."""
    from common import io_util

    if split not in {"train", "validation", "holdout", "all"}:
        raise ValueError("split must be train, validation, holdout, or all")
    if mode not in {"dense", "selection", "compressor", "amr"}:
        raise ValueError("mode must be dense, selection, compressor, or amr")
    if mode != "dense" and not checkpoint_path:
        raise ValueError(f"fixed-state {mode} evaluation requires an AMR checkpoint")
    if max_states is not None and max_states < 1:
        raise ValueError("max_states must be positive")
    root = _run_root(config, run_root)
    states = read_jsonl(root / "states.jsonl")
    _validate_run_index(root)
    if split != "all":
        states = [row for row in states if row.get("split") == split]
    if max_states is not None:
        states = states[:max_states]
    if not states:
        raise ValueError(f"no captured states for split={split!r} under {root}")

    target, tokenizer, draft, memory, metadata = load_runtime(
        config, device=device, checkpoint_path=checkpoint_path
    )
    _validate_run_index(root, metadata=metadata)
    del tokenizer
    eos_value = getattr(target.config, "eos_token_id", None)
    eos_ids = set(
        eos_value
        if isinstance(eos_value, (list, tuple, set))
        else ([] if eos_value is None else [eos_value])
    )
    records = []
    for index, state in enumerate(states, start=1):
        bundle = load_torch(root / state["bundle"])
        _validate_feature_bundle(bundle, metadata, draft)
        _validate_bundle_state(bundle, state)
        context_length = int(state["context_length"])
        trajectory_ids = bundle["trajectory_ids"]
        if context_length >= trajectory_ids.numel():
            raise ValueError(f"state {state['state_id']} lacks its pending anchor token")
        if int(trajectory_ids[context_length]) != int(state["anchor_id"]):
            raise ValueError(f"state {state['state_id']} anchor differs from saved trajectory")
        context_ids = trajectory_ids[:context_length].to(
            device=device, dtype=torch.long
        ).unsqueeze(0)
        projected_context = bundle["projected_context"][:context_length].to(
            device=device
        ).unsqueeze(0)
        record = evaluate_one_fixed_state(
            target,
            draft,
            memory,
            context_ids=context_ids,
            projected_context=projected_context,
            anchor_id=int(state["anchor_id"]),
            remaining_output_budget=int(state["remaining_output_budget"]),
            mode=mode,
            eos_token_ids=eos_ids,
        )
        record.update(
            {
                "state_id": state["state_id"],
                "document_id": state["document_id"],
                "split": state["split"],
                "target_fingerprint": metadata["target_fingerprint"]["sha256"],
                "draft_fingerprint": metadata["draft_fingerprint"]["sha256"],
            }
        )
        records.append(record)
        print(
            f"[amr-fixed-eval] {index}/{len(states)} state={state['state_id']} "
            f"A={record['accepted_proposals_raw']} G={record['committed_tokens']} "
            f"censored={record['censored']}",
            flush=True,
        )

    if output_path:
        path = Path(output_path).expanduser()
        if not path.is_absolute():
            from common.paths import ROOT

            path = ROOT / path
    else:
        path = root / "evaluation" / f"fixed_state_{split}_{mode}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output exists: {path}; pass --overwrite to replace it")
        path.unlink()
    writer = io_util.JsonlWriter(path)
    for record in records:
        writer.add(record)
    summary: dict[str, Any] = {
        "record_type": "summary",
        "split": split,
        "mode": mode,
        "states": len(records),
        "censored_states": sum(bool(row["censored"]) for row in records),
        "target_fingerprint": metadata["target_fingerprint"]["sha256"],
        "draft_fingerprint": metadata["draft_fingerprint"]["sha256"],
        "output": str(path),
    }
    for key in ("accepted_proposals_raw", "accepted_proposals_committed", "committed_tokens"):
        summary[f"mean_{key}"] = round(
            sum(float(row[key]) for row in records) / len(records), 4
        )
    summary["mean_survival"] = [
        round(sum(row["survival"][position] for row in records) / len(records), 6)
        for position in range(draft.block_size - 1)
    ]
    writer.finalize(summary)
    return summary


def generate_candidates_for_run(
    config: dict[str, Any], *, run_root: str | Path | None = None
) -> dict[str, Any]:
    root = _run_root(config, run_root)
    states = read_jsonl(root / "states.jsonl")
    manifest = _validate_run_index(root)
    memory = memory_config(config)
    if asdict(memory) != manifest["memory_config"]:
        raise ValueError("candidate memory budget/config differs from capture contract")
    candidate_dir = root / "candidate_positions"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, Any]] = []
    degenerate = 0
    for state in states:
        candidates = generate_candidate_sets(
            context_length=int(state["context_length"]),
            raw_budget=memory.raw_budget,
            local_window=memory.local_window,
            seed=int(config["training"].get("seed", 17)),
            max_candidates=int(config["training"].get("max_candidates", 12)),
        )
        if len(candidates) <= 1:
            degenerate += 1
        state_key = artifact_id(state["state_id"])
        positions_path = candidate_dir / f"{state_key}.pt"
        position_payload = {
            candidate.candidate_id: torch.tensor(candidate.positions, dtype=torch.long)
            for candidate in candidates
        }
        atomic_torch_save(position_payload, positions_path)
        for candidate in candidates:
            candidate_rows.append(
                {
                    "record_type": "candidate",
                    "schema_version": "amr-v0",
                    "capture_contract_sha256": state["capture_contract_sha256"],
                    "state_id": state["state_id"],
                    "document_id": state["document_id"],
                    "split": state["split"],
                    "candidate_id": candidate.candidate_id,
                    "method": candidate.method,
                    "positions_ref": str(positions_path.relative_to(root)),
                    "positions_sha256": sha256_file(positions_path),
                    "raw_budget": len(candidate.positions),
                    "oracle": False,
                    "deployable": True,
                    "status": "pending",
                }
            )
    path = write_jsonl(
        root / "candidates.jsonl",
        candidate_rows,
        summary={
            "states": len(states),
            "candidate_rows": len(candidate_rows),
            "degenerate_states": degenerate,
        },
    )
    return {
        "candidate_file": str(path),
        "states": len(states),
        "candidate_rows": len(candidate_rows),
        "degenerate_states": degenerate,
    }


def label_candidates(
    config: dict[str, Any],
    *,
    run_root: str | Path | None,
    device: torch.device,
) -> dict[str, Any]:
    root = _run_root(config, run_root)
    phase_started = time.perf_counter()
    max_gpu_hours = float(config["training"].get("max_gpu_hours", 24))
    if remaining_gpu_seconds(root, max_gpu_hours, device=device) <= 0:
        raise RuntimeError("AMR-DFlash GPU-hour budget is exhausted before labeling")
    state_rows = read_jsonl(root / "states.jsonl")
    candidate_rows = read_jsonl(root / "candidates.jsonl") if (root / "candidates.jsonl").exists() else []
    if not candidate_rows:
        generate_candidates_for_run(config, run_root=root)
        candidate_rows = read_jsonl(root / "candidates.jsonl")
    _validate_run_index(root, related_rows=candidate_rows)
    candidates_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        candidates_by_state[str(row["state_id"])].append(row)

    target, tokenizer, draft, memory, metadata = load_runtime(
        config, device=device, checkpoint_path=None
    )
    _validate_run_index(root, metadata=metadata)
    eos_value = getattr(target.config, "eos_token_id", None)
    eos_ids = set(
        eos_value
        if isinstance(eos_value, (list, tuple, set))
        else ([] if eos_value is None else [eos_value])
    )
    labeled_rows: list[dict[str, Any]] = []
    teacher_dir = root / "teacher"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    candidate_position_cache: dict[str, dict[str, torch.Tensor]] = {}
    budget_exhausted = False
    completed_states = 0

    for state_num, state in enumerate(state_rows, start=1):
        if remaining_gpu_seconds(
            root,
            max_gpu_hours,
            device=device,
            in_flight_seconds=time.perf_counter() - phase_started,
        ) <= 0:
            budget_exhausted = True
            break
        state_id = str(state["state_id"])
        bundle = load_torch(root / state["bundle"])
        _validate_feature_bundle(bundle, metadata, draft)
        _validate_bundle_state(bundle, state)
        context_length = int(state["context_length"])
        trajectory_ids = bundle["trajectory_ids"]
        if context_length >= trajectory_ids.numel():
            raise ValueError(f"state {state_id} lacks its pending anchor token")
        if int(trajectory_ids[context_length]) != int(state["anchor_id"]):
            raise ValueError(f"state {state_id} anchor does not match saved trajectory")
        context_ids = trajectory_ids[:context_length].to(device=device, dtype=torch.long).unsqueeze(0)
        if context_ids.shape[1] < 1:
            continue
        feature_bank = bundle["projected_context"][:context_length].to(device=device)
        feature_bank = feature_bank.unsqueeze(0)
        prefix_positions = torch.arange(context_length, device=device).unsqueeze(0)
        prefix = target(
            input_ids=context_ids,
            position_ids=prefix_positions,
            use_cache=True,
            output_hidden_states=False,
            logits_to_keep=1,
            return_dict=True,
        )
        past = prefix.past_key_values
        del prefix
        anchor_id = int(state["anchor_id"])
        live_positions = torch.arange(
            context_length,
            context_length + draft.block_size,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        state_candidate_rows = candidates_by_state.get(state_id, [])
        state_labels: list[dict[str, Any]] = []
        best_teacher: tuple[int, torch.Tensor, torch.Tensor, str] | None = None
        for candidate in state_candidate_rows:
            if remaining_gpu_seconds(
                root,
                max_gpu_hours,
                device=device,
                in_flight_seconds=time.perf_counter() - phase_started,
            ) <= 0:
                budget_exhausted = True
                break
            pos_ref = str(candidate["positions_ref"])
            position_path = str((root / pos_ref).resolve())
            if position_path not in candidate_position_cache:
                if candidate.get("positions_sha256") != sha256_file(position_path):
                    raise ValueError("candidate positions fingerprint differs from candidate index")
                candidate_position_cache[position_path] = load_torch(position_path)
            positions = candidate_position_cache[position_path][candidate["candidate_id"]].to(device)
            raw_count = int(positions.numel())
            if raw_count != int(candidate["raw_budget"]):
                raise ValueError(f"candidate {candidate['candidate_id']} has inconsistent raw budget")
            anchor_embedding = target.get_input_embeddings()(
                torch.tensor([[anchor_id]], device=device, dtype=torch.long)
            )[:, 0, :]
            draft_memory = memory.build(
                feature_bank,
                anchor_embedding,
                live_positions,
                mode="selection",
                use_cost_gate=False,
                candidate_positions=positions.unsqueeze(0),
            )
            block_ids = torch.full(
                (1, draft.block_size), draft.mask_token_id, device=device, dtype=torch.long
            )
            block_ids[0, 0] = anchor_id
            noise = target.get_input_embeddings()(block_ids)
            draft_position_ids = torch.cat((draft_memory.positions, live_positions), dim=1)
            draft_hidden = draft.forward_projected(
                projected_context=draft_memory.features,
                noise_embedding=noise,
                position_ids=draft_position_ids,
                attention_mask=draft_memory.attention_mask,
            )
            proposal_logits = target.get_output_embeddings()(draft_hidden)[:, 1:, :]
            proposals = proposal_logits.argmax(dim=-1)
            candidate_ids = torch.cat(
                (torch.tensor([[anchor_id]], device=device), proposals), dim=1
            )
            verification = target(
                input_ids=candidate_ids,
                position_ids=live_positions,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            target_logits = verification.logits[:, :-1, :]
            target_choices = target_logits.argmax(dim=-1)
            acceptance = greedy_acceptance(proposals, target_choices)
            correction = int(verification.logits[0, acceptance.accepted].argmax().item())
            accepted_proposals = [int(value) for value in proposals[0, :acceptance.accepted].tolist()]
            eos_censored = any(token in eos_ids for token in accepted_proposals) or correction in eos_ids
            remaining = int(state["remaining_output_budget"])
            budget_censored = remaining <= draft.block_size - 1
            censored = bool(eos_censored or budget_censored)
            commit = resolve_greedy_commit(
                accepted_proposals,
                accepted=acceptance.accepted,
                correction=correction,
                remaining=remaining,
                eos_token_ids=eos_ids,
            )
            row = {
                "record_type": "candidate_label",
                "schema_version": "amr-v0",
                "capture_contract_sha256": state["capture_contract_sha256"],
                "state_id": state_id,
                "document_id": state["document_id"],
                "split": state["split"],
                "candidate_id": candidate["candidate_id"],
                "method": candidate["method"],
                "positions_ref": candidate["positions_ref"],
                "positions_sha256": candidate["positions_sha256"],
                "raw_budget": raw_count,
                "oracle": False,
                "deployable": True,
                "accepted_proposals_raw": acceptance.accepted,
                "accepted_proposals_committed": len(commit.committed_proposals),
                "committed_tokens": len(commit.emitted_tokens),
                "survival": [
                    int(index < acceptance.accepted) for index in range(draft.block_size - 1)
                ],
                "first_mismatch": acceptance.first_mismatch,
                "correction_token_id": correction,
                "proposal_ids": [int(value) for value in proposals[0].tolist()],
                "censored": censored,
                "status": "success",
                "timing_scope": "instrumented_reference",
            }
            state_labels.append(row)
            new_reward = acceptance.accepted
            if not censored and (
                best_teacher is None or new_reward > best_teacher[0]
            ):
                best_teacher = (
                    new_reward,
                    target_logits[0].detach().to(torch.float16).cpu(),
                    proposals[0].detach().to(torch.long).cpu(),
                    str(candidate["candidate_id"]),
                )
            past.crop(context_length)
            if int(past.get_seq_length()) != context_length:
                raise RuntimeError(f"target cache failed to restore state {state_id}")
            del verification, target_logits, target_choices, draft_hidden, proposal_logits

        if budget_exhausted:
            break
        if best_teacher is not None:
            teacher_path = teacher_dir / f"{artifact_id(state_id)}.pt"
            atomic_torch_save(
                {
                    "state_id": state_id,
                    "capture_contract_sha256": state["capture_contract_sha256"],
                    "feature_contract": feature_contract(metadata),
                    "target_logits": best_teacher[1],
                    "candidate_proposals": best_teacher[2],
                    "candidate_id": best_teacher[3],
                    "alignment_history": "captured_verifier_candidates",
                },
                teacher_path,
            )
            for row in state_labels:
                if row["candidate_id"] == best_teacher[3]:
                    row["teacher_logits_ref"] = str(teacher_path.relative_to(root))
                    row["teacher_logits_sha256"] = sha256_file(teacher_path)
                    row["teacher_candidate_proposals"] = best_teacher[2].tolist()
                    break
        labeled_rows.extend(state_labels)
        completed_states += 1
        print(
            f"[amr-label] {state_num}/{len(state_rows)} state={state_id} "
            f"candidates={len(state_labels)} "
            f"best={best_teacher[0] if best_teacher else 'censored/degenerate'}",
            flush=True,
        )

    ledger = charge_phase(
        root,
        "label",
        elapsed_seconds=time.perf_counter() - phase_started,
        device=device,
        status="budget_exhausted" if budget_exhausted else "complete",
    )
    label_path = write_jsonl(
        root / "candidate_labels.jsonl",
        labeled_rows,
        summary={
            "states": len(state_rows),
            "states_completed": completed_states,
            "candidate_labels": len(labeled_rows),
            "successful": sum(row["status"] == "success" for row in labeled_rows),
            "censored": sum(bool(row["censored"]) for row in labeled_rows),
            "target_fingerprint": metadata["target_fingerprint"]["sha256"],
            "status": "budget_exhausted" if budget_exhausted else "complete",
            "label_gpu_hours": ledger["phases"]["label"]["gpu_seconds"] / 3600.0,
        },
    )
    preferences = build_preferences(
        labeled_rows,
        max_pairs_per_state=int(config["training"].get("max_pairs_per_state", 8)),
    )
    for pair in preferences:
        pair["candidate_labels_sha256"] = sha256_file(label_path)
    pref_path = write_jsonl(
        root / "preferences.jsonl",
        preferences,
        summary={
            "pairs": len(preferences),
            "states_with_preferences": len({row["state_id"] for row in preferences}),
        },
    )
    return {
        "candidate_labels": len(labeled_rows),
        "preferences": len(preferences),
        "labels": str(label_path),
        "preferences_file": str(pref_path),
    }


def build_preferences(
    labeled_rows: list[dict[str, Any]], *, max_pairs_per_state: int
) -> list[dict[str, Any]]:
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labeled_rows:
        by_state[str(row["state_id"])].append(row)
    preferences: list[dict[str, Any]] = []
    for state_id, rows in sorted(by_state.items()):
        eligible = [row for row in rows if not row.get("censored") and row.get("status") == "success"]
        pairs = []
        for left_index, left in enumerate(eligible):
            for right in eligible[left_index + 1 :]:
                left_reward = int(left["accepted_proposals_committed"])
                right_reward = int(right["accepted_proposals_committed"])
                if left_reward == right_reward:
                    continue
                positive, negative = (left, right) if left_reward > right_reward else (right, left)
                pairs.append(
                    (
                        abs(left_reward - right_reward),
                        positive,
                        negative,
                    )
                )
        pairs.sort(
            key=lambda value: (
                -value[0],
                value[1]["candidate_id"],
                value[2]["candidate_id"],
            )
        )
        pairs = pairs[:max(0, max_pairs_per_state)]
        normalizer = 1.0 / len(pairs) if pairs else 0.0
        for gap, positive, negative in pairs:
            preferences.append(
                {
                    "record_type": "preference",
                    "schema_version": "amr-v0",
                    **({"capture_contract_sha256": positive["capture_contract_sha256"]}
                       if "capture_contract_sha256" in positive else {}),
                    "state_id": state_id,
                    "document_id": positive["document_id"],
                    "split": positive["split"],
                    "positive_candidate_id": positive["candidate_id"],
                    "negative_candidate_id": negative["candidate_id"],
                    "reward_positive": int(positive["accepted_proposals_committed"]),
                    "reward_negative": int(negative["accepted_proposals_committed"]),
                    "reward_gap": gap,
                    "weight": normalizer,
                }
            )
    return preferences


def train_selector(
    config: dict[str, Any],
    *,
    run_root: str | Path | None,
    checkpoint_in: str | Path | None,
    checkpoint_out: str | Path | None,
    steps: int | None,
    device: torch.device,
) -> dict[str, Any]:
    """Fit the pre-draft selector from within-state verifier preferences."""
    from .core import preference_loss

    root = _run_root(config, run_root)
    phase_started = time.perf_counter()
    max_gpu_hours = float(config["training"].get("max_gpu_hours", 24))
    if remaining_gpu_seconds(root, max_gpu_hours, device=device) <= 0:
        raise RuntimeError("AMR-DFlash GPU-hour budget is exhausted before selector training")
    state_rows = read_jsonl(root / "states.jsonl")
    label_rows = read_jsonl(root / "candidate_labels.jsonl")
    preferences = read_jsonl(root / "preferences.jsonl")
    if not state_rows or not label_rows or not preferences:
        raise ValueError("capture states, label candidates, and create preferences before selector training")
    _validate_run_index(root, related_rows=label_rows)
    _validate_run_index(root, related_rows=preferences)
    label_hash = sha256_file(root / "candidate_labels.jsonl")
    for pair in preferences:
        if pair.get("candidate_labels_sha256") != label_hash:
            raise ValueError("preference labels fingerprint differs; rebuild preferences")
    target, tokenizer, draft, memory, metadata = load_runtime(
        config, device=device, checkpoint_path=checkpoint_in
    )
    manifest = _validate_run_index(root, metadata=metadata)
    preference_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in preferences:
        if row.get("split") == "train":
            preference_by_state[str(row["state_id"])].append(row)
    train_pairs = [pair for rows in preference_by_state.values() for pair in rows]
    if not train_pairs:
        raise ValueError("no train-split preference pairs; check document-level splits")
    label_by_key = {
        (str(row["state_id"]), str(row["candidate_id"])): row for row in label_rows
    }
    state_by_id = {str(row["state_id"]): row for row in state_rows}
    for pair in preferences:
        for key in ("positive_candidate_id", "negative_candidate_id"):
            label = label_by_key.get((str(pair["state_id"]), str(pair[key])))
            if label is None or label.get("censored") or label.get("status") != "success":
                raise ValueError("preference refers to a missing or ineligible candidate label")
    positions_cache: dict[str, dict[str, torch.Tensor]] = {}
    bundle_cache: dict[str, dict[str, Any]] = {}
    rng = random.Random(int(config["training"].get("seed", 17)))
    torch.manual_seed(int(config["training"].get("seed", 17)))
    train_steps = int(steps or config["training"].get("selector_steps", 200))
    batch_size = max(1, int(config["training"].get("selector_batch_size", 8)))
    optimizer = torch.optim.AdamW(
        memory.selector.parameters(),
        lr=float(config["training"].get("selector_lr", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    memory.selector.train()
    training_log: list[dict[str, Any]] = []
    budget_exhausted = False
    completed_steps = 0

    def _load_state_bundle(state_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        state = state_by_id[state_id]
        bundle_path = str((root / state["bundle"]).resolve())
        if bundle_path not in bundle_cache:
            bundle = load_torch(bundle_path)
            _validate_feature_bundle(bundle, metadata, draft)
            bundle_cache[bundle_path] = bundle
        _validate_bundle_state(bundle_cache[bundle_path], state)
        return state, bundle_cache[bundle_path]

    def _positions(label: dict[str, Any], candidate_id: str) -> torch.Tensor:
        path = str((root / label["positions_ref"]).resolve())
        if path not in positions_cache:
            if label.get("positions_sha256") != sha256_file(path):
                raise ValueError("candidate positions fingerprint differs from labels")
            positions_cache[path] = load_torch(path)
        return positions_cache[path][candidate_id].to(dtype=torch.long)

    for step in range(1, train_steps + 1):
        if remaining_gpu_seconds(
            root,
            max_gpu_hours,
            device=device,
            in_flight_seconds=time.perf_counter() - phase_started,
        ) <= 0:
            budget_exhausted = True
            break
        if len(train_pairs) <= batch_size:
            batch = train_pairs
        else:
            batch = rng.sample(train_pairs, batch_size)
        positive_scores: list[torch.Tensor] = []
        negative_scores: list[torch.Tensor] = []
        pair_weights: list[float] = []
        for pair in batch:
            state_id = str(pair["state_id"])
            state, bundle = _load_state_bundle(state_id)
            context_length = int(state["context_length"])
            context = bundle["projected_context"][:context_length]
            state_tensor = bundle["states"][int(state["state_index"])]
            anchor = state_tensor["anchor_embedding"].float().to(device).unsqueeze(0)
            query_state = memory.selector.context_query(
                context.unsqueeze(0).to(device), window=memory.config.query_window
            )
            positive = label_by_key[(state_id, str(pair["positive_candidate_id"]))]
            negative = label_by_key[(state_id, str(pair["negative_candidate_id"]))]

            def score(label: dict[str, Any]) -> torch.Tensor:
                positions = _positions(label, str(label["candidate_id"]))
                selected = context[positions].float().to(device).unsqueeze(0)
                position_tensor = positions.to(device).unsqueeze(0)
                return memory.selector(
                    selected,
                    anchor,
                    position_tensor,
                    query_state=query_state,
                ).mean()

            positive_scores.append(score(positive))
            negative_scores.append(score(negative))
            pair_weights.append(float(pair.get("weight", 1.0)))

        pos = torch.stack(positive_scores)
        neg = torch.stack(negative_scores)
        weights = torch.tensor(pair_weights, device=device, dtype=pos.dtype)
        loss = preference_loss(pos, neg, weights)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite selector preference loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            memory.selector.parameters(),
            float(config["training"].get("max_grad_norm", 1.0)),
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"non-finite selector gradient at step {step}")
        optimizer.step()
        record = {
            "record_type": "train_step",
            "phase": "selector",
            "step": step,
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "pairs_in_batch": len(batch),
            "elapsed_s": time.perf_counter() - phase_started,
        }
        training_log.append(record)
        completed_steps = step
        if step == 1 or step % 10 == 0 or step == train_steps:
            print(
                f"[amr-train-selector] step={step}/{train_steps} loss={record['loss']:.5f} "
                f"grad={record['grad_norm']:.4f}",
                flush=True,
            )

    if completed_steps == 0:
        charge_phase(
            root,
            "train_selector",
            elapsed_seconds=time.perf_counter() - phase_started,
            device=device,
            status="budget_exhausted",
        )
        raise RuntimeError("GPU-hour budget expired before a selector optimizer step")
    validation_pairs = [row for row in preferences if row.get("split") == "validation"]
    validation_accuracy = _selector_pair_accuracy(
        memory, validation_pairs, state_by_id, label_by_key, root, metadata, draft
    )
    output = _checkpoint_output(root, checkpoint_out, phase="selector")
    memory.selector.eval()
    ledger = charge_phase(
        root,
        "train_selector",
        elapsed_seconds=time.perf_counter() - phase_started,
        device=device,
        status="budget_exhausted" if budget_exhausted else "complete",
    )
    save_memory_checkpoint(
        output,
        memory,
        metadata={**metadata, "phase": "selector", "run_root": str(root.resolve()),
                  "capture_contract_sha256": manifest["capture_contract_sha256"],
                  "training_seed": int(config["training"].get("seed", 17)),
                  "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
        optimizer_state=optimizer.state_dict(),
        step=completed_steps,
    )
    training_log.append(
        {
            "record_type": "phase_summary",
            "phase": "selector",
            "steps": completed_steps,
            "train_pairs": len(train_pairs),
            "validation_pairwise_accuracy": validation_accuracy,
            "checkpoint": str(output),
            "status": "budget_exhausted" if budget_exhausted else "complete",
            "gpu_hours_total": ledger["total_gpu_seconds"] / 3600.0,
            "elapsed_s": time.perf_counter() - phase_started,
        }
    )
    write_jsonl(root / "train_selector.jsonl", training_log)
    return {
        "checkpoint": str(output),
        "steps": completed_steps,
        "train_pairs": len(train_pairs),
        "validation_pairwise_accuracy": validation_accuracy,
    }


def _selector_pair_accuracy(
    memory: Any,
    pairs: list[dict[str, Any]],
    state_by_id: dict[str, dict[str, Any]],
    label_by_key: dict[tuple[str, str], dict[str, Any]],
    root: Path,
    metadata: dict[str, Any],
    draft: Any,
) -> float | None:
    if not pairs:
        return None
    device = next(memory.selector.parameters()).device
    correct = 0
    with torch.no_grad():
        for pair in pairs:
            state_id = str(pair["state_id"])
            state = state_by_id[state_id]
            bundle = load_torch(root / state["bundle"])
            _validate_feature_bundle(bundle, metadata, draft)
            _validate_bundle_state(bundle, state)
            context = bundle["projected_context"][: int(state["context_length"])]
            anchor = bundle["states"][int(state["state_index"])]["anchor_embedding"].float().to(device).unsqueeze(0)
            query_state = memory.selector.context_query(
                context.unsqueeze(0).to(device), window=memory.config.query_window
            )
            scores = []
            for candidate_id in (pair["positive_candidate_id"], pair["negative_candidate_id"]):
                label = label_by_key[(state_id, str(candidate_id))]
                if label.get("positions_sha256") != sha256_file(root / label["positions_ref"]):
                    raise ValueError("validation candidate positions fingerprint differs from labels")
                positions = load_torch(root / label["positions_ref"])[str(candidate_id)]
                scores.append(
                    memory.selector(
                        context[positions].float().to(device).unsqueeze(0),
                        anchor,
                        positions.to(device).unsqueeze(0),
                        query_state=query_state,
                    ).mean()
                )
            correct += int(scores[0] > scores[1])
    return correct / len(pairs)


def train_compressor(
    config: dict[str, Any],
    *,
    run_root: str | Path | None,
    checkpoint_in: str | Path,
    checkpoint_out: str | Path | None,
    steps: int | None,
    device: torch.device,
) -> dict[str, Any]:
    """Train global slots/gate against saved captured-verifier logits."""
    root = _run_root(config, run_root)
    phase_started = time.perf_counter()
    max_gpu_hours = float(config["training"].get("max_gpu_hours", 24))
    if remaining_gpu_seconds(root, max_gpu_hours, device=device) <= 0:
        raise RuntimeError("AMR-DFlash GPU-hour budget is exhausted before compressor training")
    if memory_config(config).num_slots == 0:
        raise ValueError("compressor training requires memory.num_slots > 0")
    state_rows = read_jsonl(root / "states.jsonl")
    label_rows = read_jsonl(root / "candidate_labels.jsonl")
    _validate_run_index(root, related_rows=label_rows)
    state_by_id = {str(row["state_id"]): row for row in state_rows}
    rows = [
        row
        for row in label_rows
        if row.get("split") == "train"
        and not row.get("censored")
        and row.get("teacher_logits_ref")
    ]
    if not rows:
        raise ValueError("no train-split uncensored teacher rows for compressor training")
    target, tokenizer, draft, memory, metadata = load_runtime(
        config, device=device, checkpoint_path=checkpoint_in
    )
    manifest = _validate_run_index(root, metadata=metadata)
    if checkpoint_in:
        saved = load_torch(checkpoint_in).get("metadata") or {}
        if saved.get("capture_contract_sha256") != manifest["capture_contract_sha256"]:
            raise ValueError("selector checkpoint capture contract differs from compressor training run")
    for parameter in memory.selector.parameters():
        parameter.requires_grad_(False)
    memory.selector.eval()
    for parameter in memory.compressor.parameters():
        parameter.requires_grad_(True)
    memory.slot_gate.requires_grad_(True)
    train_steps = int(steps or config["training"].get("compressor_steps", 200))
    rng = random.Random(int(config["training"].get("seed", 17)) + 1)
    optimizer_parameters = list(memory.compressor.parameters()) + [memory.slot_gate]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=float(config["training"].get("compressor_lr", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )
    training_log: list[dict[str, Any]] = []
    budget_exhausted = False
    completed_steps = 0
    positions_cache: dict[str, dict[str, torch.Tensor]] = {}
    bundle_cache: dict[str, dict[str, Any]] = {}
    teacher_cache: dict[str, dict[str, torch.Tensor]] = {}
    vocab_size = int(target.get_output_embeddings().weight.shape[0])

    for step in range(1, train_steps + 1):
        if remaining_gpu_seconds(
            root,
            max_gpu_hours,
            device=device,
            in_flight_seconds=time.perf_counter() - phase_started,
        ) <= 0:
            budget_exhausted = True
            break
        row = rng.choice(rows)
        state = state_by_id[str(row["state_id"])]
        bundle_path = str((root / state["bundle"]).resolve())
        if bundle_path not in bundle_cache:
            bundle = load_torch(bundle_path)
            _validate_feature_bundle(bundle, metadata, draft)
            bundle_cache[bundle_path] = bundle
        bundle = bundle_cache[bundle_path]
        _validate_bundle_state(bundle, state)
        context_length = int(state["context_length"])
        projected = bundle["projected_context"][:context_length].to(device).unsqueeze(0)
        anchor_id = int(state["anchor_id"])
        anchor_embedding = target.get_input_embeddings()(
            torch.tensor([[anchor_id]], device=device, dtype=torch.long)
        )[:, 0, :]
        positions_path = str((root / row["positions_ref"]).resolve())
        if positions_path not in positions_cache:
            if row.get("positions_sha256") != sha256_file(positions_path):
                raise ValueError("candidate positions fingerprint differs from labels")
            positions_cache[positions_path] = load_torch(positions_path)
        candidate_positions = positions_cache[positions_path][str(row["candidate_id"])].to(device)
        live_positions = torch.arange(
            context_length,
            context_length + draft.block_size,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        draft_memory = memory.build(
            projected,
            anchor_embedding,
            live_positions,
            mode="amr",
            use_cost_gate=False,
            candidate_positions=candidate_positions.unsqueeze(0),
        )
        noise_ids = torch.full(
            (1, draft.block_size), draft.mask_token_id, device=device, dtype=torch.long
        )
        noise_ids[0, 0] = anchor_id
        noise = target.get_input_embeddings()(noise_ids)
        position_ids = torch.cat((draft_memory.positions, live_positions), dim=1)
        draft_hidden = draft.forward_projected(
            projected_context=draft_memory.features,
            noise_embedding=noise,
            position_ids=position_ids,
            attention_mask=draft_memory.attention_mask,
        )
        draft_logits = target.get_output_embeddings()(draft_hidden)[:, 1:, :]
        teacher_path = str((root / row["teacher_logits_ref"]).resolve())
        if teacher_path not in teacher_cache:
            if row.get("teacher_logits_sha256") != sha256_file(teacher_path):
                raise ValueError("teacher logits fingerprint differs from labels")
            teacher = load_torch(teacher_path)
            teacher_cache[teacher_path] = teacher
        teacher = teacher_cache[teacher_path]
        if (teacher.get("feature_contract") != feature_contract(metadata)
                or teacher.get("capture_contract_sha256") != state["capture_contract_sha256"]
                or teacher.get("state_id") != state["state_id"]
                or teacher.get("candidate_id") != row["candidate_id"]
                or teacher.get("alignment_history") != "captured_verifier_candidates"):
            raise ValueError("teacher capture/projection/state contract differs from labels")
        target_logits = teacher_cache[teacher_path]["target_logits"].to(
            device=device, dtype=torch.float32
        ).unsqueeze(0)
        if draft_logits.shape != target_logits.shape:
            raise ValueError(
                f"alignment logits differ: draft={tuple(draft_logits.shape)}, target={tuple(target_logits.shape)}, vocab={vocab_size}"
            )
        valid = torch.ones(draft_logits.shape[:2], device=device, dtype=torch.bool)
        loss = alignment_kl_loss(draft_logits, target_logits, valid)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite compressor alignment loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            optimizer_parameters,
            float(config["training"].get("max_grad_norm", 1.0)),
        )
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"non-finite compressor gradient at step {step}")
        optimizer.step()
        record = {
            "record_type": "train_step",
            "phase": "compressor",
            "step": step,
            "loss": float(loss.detach().cpu()),
            "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
            "slot_gate": float(memory.slot_gate.detach().cpu()),
            "state_id": row["state_id"],
            "candidate_id": row["candidate_id"],
            "elapsed_s": time.perf_counter() - phase_started,
        }
        training_log.append(record)
        completed_steps = step
        if step == 1 or step % 10 == 0 or step == train_steps:
            print(
                f"[amr-train-compressor] step={step}/{train_steps} loss={record['loss']:.5f} "
                f"grad={record['grad_norm']:.4f} gate={record['slot_gate']:.3f}",
                flush=True,
            )

    if completed_steps == 0:
        charge_phase(
            root,
            "train_compressor",
            elapsed_seconds=time.perf_counter() - phase_started,
            device=device,
            status="budget_exhausted",
        )
        raise RuntimeError("GPU-hour budget expired before a compressor optimizer step")
    output = _checkpoint_output(root, checkpoint_out, phase="compressor")
    ledger = charge_phase(
        root,
        "train_compressor",
        elapsed_seconds=time.perf_counter() - phase_started,
        device=device,
        status="budget_exhausted" if budget_exhausted else "complete",
    )
    save_memory_checkpoint(
        output,
        memory,
        metadata={
            **metadata,
            "phase": "compressor",
            "run_root": str(root.resolve()),
            "capture_contract_sha256": manifest["capture_contract_sha256"],
            "training_seed": int(config["training"].get("seed", 17)),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "alignment_history": config["training"].get("alignment_history"),
        },
        optimizer_state=optimizer.state_dict(),
        step=completed_steps,
    )
    training_log.append(
        {
            "record_type": "phase_summary",
            "phase": "compressor",
            "steps": completed_steps,
            "teacher_rows": len(rows),
            "alignment_history": config["training"].get("alignment_history"),
            "checkpoint": str(output),
            "status": "budget_exhausted" if budget_exhausted else "complete",
            "gpu_hours_total": ledger["total_gpu_seconds"] / 3600.0,
            "elapsed_s": time.perf_counter() - phase_started,
        }
    )
    write_jsonl(root / "train_compressor.jsonl", training_log)
    return {
        "checkpoint": str(output),
        "steps": completed_steps,
        "teacher_rows": len(rows),
        "final_loss": training_log[-2]["loss"] if len(training_log) > 1 else None,
    }


def _checkpoint_output(
    root: Path, requested: str | Path | None, *, phase: str
) -> Path:
    if requested:
        path = Path(requested).expanduser()
    else:
        from common.paths import ROOT

        path = ROOT / "checkpoints" / "amr_dflash" / safe_id(root.name) / f"{phase}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
