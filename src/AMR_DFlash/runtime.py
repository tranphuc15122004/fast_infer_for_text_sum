"""Common local-only model loading used by inference, labeling, and training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
from typing import Any

import torch

from .checkpoint import build_model_metadata, load_memory_checkpoint, snapshot_fingerprint
from .config import memory_config, resolve_env_path
from .memory import AMRMemory
from .model import freeze_backbones, load_pretrained_models


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = name.lower().replace("torch.", "")
    choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if normalized not in choices:
        raise ValueError(f"unsupported AMR dtype: {name}")
    dtype = choices[normalized]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU execution is not supported by the AMR-DFlash smoke path")
    return dtype


def load_runtime(
    config: dict[str, Any],
    *,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
) -> tuple[Any, Any, Any, AMRMemory, dict[str, Any]]:
    # Seed before constructing either backbone or the trainable memory modules.
    seed = int(config["training"].get("seed", 17))
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    target_path = resolve_env_path(config, "target_model_env")
    draft_path = resolve_env_path(config, "draft_model_env")
    assert target_path is not None and draft_path is not None
    model_cfg = config["model"]
    dtype = resolve_dtype(str(model_cfg.get("dtype", "bfloat16")), device)
    target, tokenizer, draft = load_pretrained_models(
        target_path,
        draft_path,
        device=device,
        dtype=dtype,
        attention_backend=str(model_cfg.get("attention_backend", "sdpa")),
    )
    memory_cfg = memory_config(config)
    index_dim = int(config["memory"].get("index_dim", 64))
    memory = AMRMemory(draft.hidden_size, memory_cfg, index_dim=index_dim).to(device)
    target_fingerprint = snapshot_fingerprint(target_path)
    draft_fingerprint = snapshot_fingerprint(draft_path)
    metadata = build_model_metadata(
        target_fingerprint,
        draft_fingerprint,
        target_layer_ids=draft.target_layer_ids,
        block_size=draft.block_size,
        memory_config=asdict(memory_cfg),
        index_dim=index_dim,
    )
    metadata.update(dtype=str(dtype).replace("torch.", ""),
                    attention_backend=str(model_cfg.get("attention_backend", "sdpa")))
    if checkpoint_path:
        load_memory_checkpoint(
            checkpoint_path,
            memory,
            expected_metadata=metadata,
            device=device,
            strict=True,
        )
    freeze_backbones(target, draft)
    return target, tokenizer, draft, memory, metadata
