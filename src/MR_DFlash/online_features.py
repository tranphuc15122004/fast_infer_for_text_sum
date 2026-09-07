"""Online target hidden-feature provider cho pilot MR-DFlash."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence

import torch


class OnlineTargetFeatureProvider:
    """Lấy hidden states từ backbone target frozen, không chạy LM head.

    HF causal-LM thường expose backbone ở ``target_model.model``. Provider
    fallback về chính object được truyền vào để dùng được với tiny test model
    và các backbone custom.
    """

    def __init__(
        self,
        target_model: torch.nn.Module,
        feature_layer_ids: Sequence[int],
        *,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if not feature_layer_ids:
            raise ValueError("feature_layer_ids không được rỗng")
        self.target_model = target_model
        self.backbone = getattr(target_model, "model", target_model)
        self.feature_layer_ids = [int(value) for value in feature_layer_ids]
        if any(value < 0 for value in self.feature_layer_ids):
            raise ValueError("feature layer id phải không âm")
        self.dtype = dtype
        self.target_model.eval()
        for parameter in self.target_model.parameters():
            parameter.requires_grad_(False)
        self._captured_layers: dict[int, torch.Tensor] = {}
        self._hooks = []
        target_layers = getattr(self.backbone, "layers", None)
        if target_layers is not None:
            num_layers = len(target_layers)
            if any(value >= num_layers for value in self.feature_layer_ids):
                raise ValueError(
                    f"feature layer vượt số backbone layers: {self.feature_layer_ids} vs {num_layers}"
                )
            for layer_id in self.feature_layer_ids:
                self._hooks.append(
                    target_layers[layer_id].register_forward_hook(
                        self._make_capture_hook(layer_id)
                    )
                )

    def _make_capture_hook(self, layer_id: int):
        def capture_hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"hidden output layer {layer_id} không phải Tensor")
            self._captured_layers[layer_id] = value

        return capture_hook

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids phải có dạng [batch, sequence]")
        device = next(self.target_model.parameters()).device
        ids = input_ids.to(device=device, dtype=torch.long)
        self._captured_layers.clear()
        kwargs = {
            "input_ids": ids,
            "output_hidden_states": not bool(self._hooks),
            "use_cache": False,
            "return_dict": True,
        }
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(device=device)
        # ``inference_mode`` tiết kiệm overhead ở target, nhưng tensor tạo ra
        # trong mode này không được autograd lưu làm input cho draft Linear.
        # Clone sau khi thoát context để trả về regular detached tensor.
        with torch.inference_mode():
            outputs = self.backbone(**kwargs)
            if self._hooks:
                if len(self._captured_layers) != len(self.feature_layer_ids):
                    raise RuntimeError(
                        "online feature hook thiếu layer: "
                        f"expected={self.feature_layer_ids}, got={sorted(self._captured_layers)}"
                    )
                selected = [self._captured_layers[layer_id] for layer_id in self.feature_layer_ids]
            else:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states is None:
                    raise ValueError("target backbone không trả hidden_states")
                selected = []
                for layer_id in self.feature_layer_ids:
                    index = layer_id + 1
                    if index >= len(hidden_states):
                        raise ValueError(
                            f"feature layer {layer_id} vượt số hidden states {len(hidden_states)}"
                        )
                    selected.append(hidden_states[index])
            features = torch.cat(selected, dim=-1).detach()
        features = features.clone()
        if self.dtype is not None and features.dtype != self.dtype:
            features = features.to(dtype=self.dtype)
        return features


__all__ = ["OnlineTargetFeatureProvider"]
