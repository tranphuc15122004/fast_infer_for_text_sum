"""Checkpoint train DFlash: draft weights (weights-only) + trạng thái trainer.

Tương ứng checkpoint của SpecForge: khi persist "draft weights", chỉ giữ các
key dưới module draft (do ``DFlashTrainStrategy.checkpoint_state_filter`` quyết
định) — target lm_head/embedding không được lưu làm draft weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def save_training_checkpoint(
    path: str,
    *,
    draft_state_dict: Dict[str, torch.Tensor],
    global_step: int,
    optimizer_state: Optional[dict] = None,
    scheduler_state: Optional[dict] = None,
    run_id: str = "",
    config_yaml: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Ghi checkpoint đầy đủ (weights + trainer state) ra một file .pt."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format": "mr_dflash_checkpoint_v1",
        "run_id": run_id,
        "global_step": int(global_step),
        "draft_state_dict": draft_state_dict,
        "metrics": metrics or {},
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if scheduler_state is not None:
        payload["scheduler_state"] = scheduler_state
    if config_yaml is not None:
        payload["config_yaml"] = config_yaml
    torch.save(payload, path)


def load_training_checkpoint(path: str) -> Dict[str, Any]:
    """Đọc checkpoint (map_location='cpu')."""
    return torch.load(path, map_location="cpu", weights_only=False)


def save_draft_weights(
    path: str,
    draft_state_dict: Dict[str, torch.Tensor],
    *,
    config_yaml: Optional[str] = None,
) -> None:
    """Ghi weights-only kèm config để serving tự dựng đúng architecture."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "format": "mr_dflash_draft_weights_v1",
        "draft_state_dict": draft_state_dict,
    }
    if config_yaml is not None:
        payload["config_yaml"] = config_yaml
    torch.save(payload, path)


def _convert_dflash_state_to_mr(
    state: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    """Map one-layer/multi-layer DFlash weights into every MR stage.

    ``fc`` and ``hidden_norm`` initialize the two feature adapters; the
    compressors and indexer intentionally remain at their MR initialization
    because DFlash has no equivalent parameters.
    """
    current = model.state_dict()
    source_layers: Dict[int, Dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        if not key.startswith("layers."):
            continue
        parts = key.split(".", 2)
        if len(parts) == 3 and parts[1].isdigit():
            source_layers.setdefault(int(parts[1]), {})[parts[2]] = value
    if not source_layers:
        raise ValueError("DFlash checkpoint không có layers.* để chuyển sang MR-DFlash")

    target_stage_count = len(getattr(model, "stages", []))
    if target_stage_count < 1:
        raise ValueError("model đích không có stages MR-DFlash")
    converted: Dict[str, torch.Tensor] = {}

    group_map = {
        "input_layernorm": "input_layernorm",
        "self_attn": "joint_attn",
        "post_attention_layernorm": "post_attention_layernorm",
        "mlp": "mlp",
    }
    source_layer_ids = sorted(source_layers)
    for stage_idx in range(target_stage_count):
        source_idx = source_layer_ids[min(stage_idx, len(source_layer_ids) - 1)]
        for source_key, value in source_layers[source_idx].items():
            source_group, separator, suffix = source_key.partition(".")
            destination_group = group_map.get(source_group)
            if destination_group is None or not separator:
                continue
            destination_key = f"stages.{stage_idx}.{destination_group}.{suffix}"
            if destination_key in current and current[destination_key].shape == value.shape:
                converted[destination_key] = value

    top_level_map = {
        "fc.weight": "memory.adapter.hca.weight",
        "hidden_norm.weight": "memory.adapter.hca_norm.weight",
        "norm.weight": "norm.weight",
    }
    for source_key, destination_key in top_level_map.items():
        value = state.get(source_key)
        if value is None:
            continue
        if destination_key in current and current[destination_key].shape == value.shape:
            converted[destination_key] = value
    # Share the same DFlash adapter initialization across the two MR views.
    if "memory.adapter.hca.weight" in converted:
        if current["memory.adapter.csa.weight"].shape == converted["memory.adapter.hca.weight"].shape:
            converted["memory.adapter.csa.weight"] = converted["memory.adapter.hca.weight"]
    if "memory.adapter.hca_norm.weight" in converted:
        if current["memory.adapter.csa_norm.weight"].shape == converted["memory.adapter.hca_norm.weight"].shape:
            converted["memory.adapter.csa_norm.weight"] = converted["memory.adapter.hca_norm.weight"]
    return converted


def warm_start_draft_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    key_prefix: str = "draft_model.",
    strategy_name: str = "dflash",
) -> Tuple[List[str], List[str]]:
    """Nạp draft weights từ checkpoint (training checkpoint hoặc weights-only).

    Trả về ``(missing, unexpected)``. Native ``mr_dflash`` checkpoint được
    load strict; khi source là DFlash, converter cho phép thiếu các module MR
    mới (pool/indexer) nhưng vẫn fail nếu thiếu tensor có thể chuyển.
    """
    raw = load_training_checkpoint(checkpoint_path)
    state = raw.get("draft_state_dict")
    if state is None and "draft_state_dict" not in raw:
        # weights-only file lưu draft_state_dict trực tiếp ở top-level
        state = raw if raw.get("format", "").startswith("mr_dflash_draft") else None
    if state is None:
        raise ValueError(
            f"checkpoint {checkpoint_path} không chứa draft weights"
        )

    # Chuyển key có tiền tố về dạng model.state_dict().
    loadable: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith(key_prefix):
            loadable[key[len(key_prefix):]] = value
        else:
            loadable[key] = value
    current = model.state_dict()
    is_mr_conversion = strategy_name == "mr_dflash" and any(
        key.startswith("layers.") for key in loadable
    )
    if is_mr_conversion:
        converted = _convert_dflash_state_to_mr(loadable, model)
        incompatible = model.load_state_dict(converted, strict=False)
        # Các tensor mới của MR không có trong DFlash và được khởi tạo riêng.
        allowed_new = (
            "memory.hca_pool.",
            "memory.csa_pool.",
            "memory.indexer.",
        )
        missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_new)
        ]
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "strict DFlash→MR-DFlash conversion failed: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return [], []

    incompatible = model.load_state_dict(loadable, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if strategy_name == "mr_dflash" and (missing or unexpected):
        raise RuntimeError(
            "strict MR-DFlash checkpoint load failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if strategy_name and strategy_name != "mr_dflash":
        # fc/hidden_norm có thể vắng ở checkpoint DFlash cũ.
        missing = [k for k in missing if "fc." not in k and "hidden_norm." not in k]
    return missing, unexpected


__all__ = [
    "save_training_checkpoint",
    "load_training_checkpoint",
    "save_draft_weights",
    "warm_start_draft_model",
]
