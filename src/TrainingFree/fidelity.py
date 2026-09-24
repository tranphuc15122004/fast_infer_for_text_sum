"""Source-only fake quantization primitives for the FidelityKV experiments."""

from __future__ import annotations

from collections.abc import Mapping
from types import MethodType
from typing import Any


_ALLOWED_PRECISIONS = (4, 8, 16)


def _validate_precision(bits: int) -> int:
    resolved = int(bits)
    if resolved not in _ALLOWED_PRECISIONS:
        raise ValueError("bits must be one of 4, 8, or 16")
    return resolved


def fake_quantize_symmetric(values: Any, *, bits: int, quant_dim: int = -2) -> Any:
    """Symmetric min/max fake quantization, returned in the input dtype.

    Scales are independent for every slice orthogonal to ``quant_dim``. The
    signed range uses ``[-qmax, qmax]`` (e.g. [-7, 7] for four bits), which is
    symmetric around zero and avoids a zero-point.
    """

    import torch

    resolved = _validate_precision(bits)
    if not torch.is_tensor(values) or values.numel() == 0:
        raise ValueError("values must be a non-empty tensor")
    if resolved == 16:
        return values
    if not values.is_floating_point():
        raise ValueError("values must have a floating-point dtype")
    if not -values.ndim <= int(quant_dim) < values.ndim:
        raise ValueError("quant_dim is outside the tensor rank")

    work = values.float()
    qmax = (1 << (resolved - 1)) - 1
    scale = work.abs().amax(dim=int(quant_dim), keepdim=True) / qmax
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(work / safe_scale).clamp(-qmax, qmax) * safe_scale
    return quantized.to(dtype=values.dtype)


def fake_quantize_source_span_(
    cache_tensor: Any,
    *,
    source_start: int,
    source_end: int,
    bits: int,
    kv_heads: tuple[int, ...] | None = None,
) -> Any:
    """Fake-quantize only selected source positions in a KV tensor.

    ``cache_tensor`` has shape ``[batch, kv_heads, sequence, head_dim]``.
    The scale is per batch, KV head, and head-dimension channel over source
    positions. Prefix, suffix, generated positions, and unselected KV heads
    remain byte-for-byte unchanged.
    """

    import torch

    resolved = _validate_precision(bits)
    if not torch.is_tensor(cache_tensor) or cache_tensor.ndim != 4:
        raise ValueError("cache_tensor must have shape [batch, kv_heads, sequence, head_dim]")
    start, end = int(source_start), int(source_end)
    if start < 0 or end <= start or end > int(cache_tensor.shape[-2]):
        raise ValueError("source span is invalid")
    selected = tuple(range(int(cache_tensor.shape[1]))) if kv_heads is None else tuple(
        dict.fromkeys(int(head) for head in kv_heads)
    )
    if not selected or any(head < 0 or head >= int(cache_tensor.shape[1]) for head in selected):
        raise ValueError("kv_heads contains an invalid KV-head index")
    if resolved == 16:
        return cache_tensor
    for head in selected:
        source = cache_tensor[:, head, start:end, :]
        source.copy_(fake_quantize_symmetric(source, bits=resolved, quant_dim=-2))
    return cache_tensor


def _precision_for(spec: int | Mapping[tuple[int, int], int], layer: int, head: int) -> int:
    if isinstance(spec, Mapping):
        value = spec.get((int(layer), int(head)), 16)
    else:
        value = spec
    return _validate_precision(value)


def install_source_cache_quantizer(
    cache: Any,
    *,
    source_start: int,
    source_end: int,
    k_bits: int | Mapping[tuple[int, int], int],
    v_bits: int | Mapping[tuple[int, int], int],
) -> None:
    """Intercept cache appends and quantize each layer's source span once.

    DynamicCache keeps key/value tensors in the returned cache object. The
    wrapper first performs the model's normal append, then overwrites only the
    source slice with fake-quantized/dequantized values. Later generated KV
    appends are therefore left in their original BF16/FP16 dtype.

    ``k_bits`` and ``v_bits`` may be uniform integers or mappings from
    ``(layer_index, kv_head_index)`` to bit width; missing entries mean 16 bit.
    """

    start, end = int(source_start), int(source_end)
    if start < 0 or end <= start:
        raise ValueError("source span is invalid")
    if isinstance(k_bits, Mapping):
        for value in k_bits.values():
            _validate_precision(value)
    else:
        _validate_precision(k_bits)
    if isinstance(v_bits, Mapping):
        for value in v_bits.values():
            _validate_precision(value)
    else:
        _validate_precision(v_bits)
    update = getattr(cache, "update", None)
    if not callable(update):
        raise TypeError("cache must expose a callable update method")
    if getattr(cache, "_fidelity_quantizer_installed", False):
        raise ValueError("source cache quantizer is already installed")

    original_update = update
    quantized_layers: set[int] = set()

    def wrapped_update(
        self: Any,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: Any = None,
    ) -> tuple[Any, Any]:
        keys, values = original_update(key_states, value_states, layer_idx, cache_kwargs)
        layer = int(layer_idx)
        if layer in quantized_layers:
            return keys, values
        if end > int(keys.shape[-2]):
            raise ValueError("source span exceeds the updated KV sequence")
        kv_head_count = int(keys.shape[1])
        for head in range(kv_head_count):
            key_precision = _precision_for(k_bits, layer, head)
            value_precision = _precision_for(v_bits, layer, head)
            if key_precision < 16:
                fake_quantize_source_span_(
                    keys,
                    source_start=start,
                    source_end=end,
                    bits=key_precision,
                    kv_heads=(head,),
                )
            if value_precision < 16:
                fake_quantize_source_span_(
                    values,
                    source_start=start,
                    source_end=end,
                    bits=value_precision,
                    kv_heads=(head,),
                )
        quantized_layers.add(layer)
        return keys, values

    cache.update = MethodType(wrapped_update, cache)
    cache._fidelity_quantizer_installed = True
