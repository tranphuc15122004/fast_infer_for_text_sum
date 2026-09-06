"""Checkpoint train DFlash: draft weights (weights-only) + trạng thái trainer.

Tương ứng checkpoint của SpecForge: khi persist "draft weights", chỉ giữ các
key dưới module draft (do ``DFlashTrainStrategy.checkpoint_state_filter`` quyết
định) — target lm_head/embedding không được lưu làm draft weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

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


def save_draft_weights(path: str, draft_state_dict: Dict[str, torch.Tensor]) -> None:
    """Ghi weights-only (warm start / export cho serving sau này)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"format": "mr_dflash_draft_weights_v1", "draft_state_dict": draft_state_dict},
        path,
    )


def warm_start_draft_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    key_prefix: str = "draft_model.",
    strategy_name: str = "dflash",
) -> List[str]:
    """Nạp draft weights từ checkpoint (training checkpoint hoặc weights-only).

    Trả về danh sách key đã nạp; chỉ báo lỗi nếu thiếu key bắt buộc.
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
    missing_required = [k for k in current if k not in loadable and "fc." not in k and "hidden_norm." not in k]
    incompatible = model.load_state_dict(loadable, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if missing_required and strategy_name:
        # fc/hidden_norm có thể vắng ở checkpoint cũ; không báo lỗi.
        missing = [k for k in missing if "fc." not in k and "hidden_norm." not in k]
    return missing, unexpected


__all__ = [
    "save_training_checkpoint",
    "load_training_checkpoint",
    "save_draft_weights",
    "warm_start_draft_model",
]
