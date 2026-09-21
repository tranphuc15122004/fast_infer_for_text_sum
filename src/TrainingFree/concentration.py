"""Per-head source-attention concentration metrics for controlled search."""

from __future__ import annotations

import math
from typing import Any, Sequence


def _last_query_attention(attention: Any):
    import torch

    tensor = (
        attention.detach().float()
        if hasattr(attention, "detach")
        else torch.as_tensor(attention).float()
    )
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("attention batch dimension must be one")
        return tensor[0, :, -1, :]
    if tensor.ndim == 3:
        if tensor.shape[0] != 1:
            raise ValueError("attention batch dimension must be one")
        return tensor[0]
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"unsupported attention rank: {tensor.ndim}")


def _quantile_count(values: Sequence[float], level: float) -> int:
    if not 0.0 < level <= 1.0 or not math.isfinite(level):
        raise ValueError("mass levels must be finite and in (0, 1]")
    ordered = sorted((float(value) for value in values), reverse=True)
    total = sum(ordered)
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("source attention mass must be positive and finite")
    cumulative = 0.0
    for count, value in enumerate(ordered, start=1):
        cumulative += value / total
        if cumulative + 1e-12 >= level:
            return count
    return len(ordered)


def head_source_concentration(
    attention: Any,
    *,
    source_start: int,
    source_end: int,
    mass_levels: Sequence[float] = (0.90, 0.95, 0.99),
) -> list[dict[str, float | int]]:
    """Summarize source-token concentration separately for every attention head.

    The K-values are computed after normalizing each head over the source span.
    ``source_mass`` is retained so callers can distinguish concentrated source
    attention from a head that mostly attends to non-source/live tokens.
    """

    tensor = _last_query_attention(attention)
    start, end = int(source_start), int(source_end)
    if start < 0 or end <= start or end > tensor.shape[-1]:
        raise ValueError("source span is invalid")
    levels = tuple(float(level) for level in mass_levels)
    if not levels or any(not 0.0 < level <= 1.0 or not math.isfinite(level) for level in levels):
        raise ValueError("mass levels must be finite and in (0, 1]")
    source = tensor[:, start:end]
    if not bool(torch_is_finite(source)) or bool((source < 0.0).any()):
        raise ValueError("source attention must be finite and non-negative")

    rows: list[dict[str, float | int]] = []
    source_tokens = int(source.shape[1])
    for head in range(int(source.shape[0])):
        values = [float(value) for value in source[head].tolist()]
        source_mass = sum(values)
        if source_mass <= 0.0 or not math.isfinite(source_mass):
            raise ValueError("source attention mass must be positive and finite")
        row: dict[str, float | int] = {
            "head": head,
            "source_tokens": source_tokens,
            "source_mass": source_mass,
        }
        for level in levels:
            label = int(round(level * 100))
            count = _quantile_count(values, level)
            row[f"k{label}"] = count
            row[f"k{label}_fraction"] = count / source_tokens
        rows.append(row)
    return rows


def head_source_oracle_metrics(
    attention: Any,
    values: Any,
    *,
    source_start: int,
    source_end: int,
    routed_fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30),
) -> list[dict[str, float | int]]:
    """Compute exact top-source oracle metrics for each query head.

    The oracle retains every non-source position and retains the exact
    highest-attention source positions for each requested fraction. Attention
    is renormalized over retained positions before measuring context error.
    This is an offline upper-bound experiment, not a physical executor.
    """

    import torch

    attention_tensor = _last_query_attention(attention)
    values_tensor = values.detach().float() if hasattr(values, "detach") else torch.as_tensor(values).float()
    if values_tensor.ndim == 3:
        values_tensor = values_tensor.unsqueeze(0)
    if values_tensor.ndim != 4 or values_tensor.shape[0] != 1:
        raise ValueError("values must have shape [1, kv_heads, tokens, dimension]")
    start, end = int(source_start), int(source_end)
    if start < 0 or end <= start or end > attention_tensor.shape[-1]:
        raise ValueError("source span is invalid")
    if values_tensor.shape[2] != attention_tensor.shape[-1]:
        raise ValueError("attention/value sequence lengths must match")
    fractions = tuple(float(value) for value in routed_fractions)
    if not fractions or any(
        not math.isfinite(value) or not 0.0 < value <= 1.0 for value in fractions
    ):
        raise ValueError("routed fractions must be finite and in (0, 1]")
    if not bool(torch.isfinite(attention_tensor).all()) or bool((attention_tensor < 0.0).any()):
        raise ValueError("attention must be finite and non-negative")
    if not bool(torch.isfinite(values_tensor).all()):
        raise ValueError("values must be finite")

    from .lease_collector import map_query_heads_to_kv

    query_heads = int(attention_tensor.shape[0])
    kv_heads = int(values_tensor.shape[1])
    mapping = map_query_heads_to_kv(query_heads, kv_heads)
    source = attention_tensor[:, start:end]
    source_tokens = int(source.shape[1])
    full_values = values_tensor[0, mapping]
    full_context = torch.einsum("hs,hsd->hd", attention_tensor, full_values)
    rows: list[dict[str, float | int]] = []
    for head in range(query_heads):
        source_mass = float(source[head].sum().item())
        if source_mass <= 0.0 or not math.isfinite(source_mass):
            raise ValueError("source attention mass must be positive and finite")
        row: dict[str, float | int] = {
            "head": head,
            "source_tokens": source_tokens,
            "source_mass": source_mass,
        }
        for fraction in fractions:
            count = min(source_tokens, max(1, math.ceil(source_tokens * fraction)))
            top = torch.topk(source[head], k=count, largest=True, sorted=False).indices + start
            keep = torch.ones(attention_tensor.shape[-1], dtype=torch.bool, device=attention_tensor.device)
            keep[start:end] = False
            keep[top] = True
            weights = attention_tensor[head] * keep.to(attention_tensor.dtype)
            denominator = float(weights.sum().item())
            if denominator <= 0.0 or not math.isfinite(denominator):
                raise ValueError("retained attention mass must be positive and finite")
            approximate = torch.einsum("s,sd->d", weights, full_values[head]) / denominator
            exact = full_context[head]
            exact_norm = float(torch.linalg.vector_norm(exact).item())
            error = 0.0 if exact_norm == 0.0 else float(
                torch.linalg.vector_norm(approximate - exact).item() / exact_norm
            )
            label = str(fraction).rstrip("0").rstrip(".")
            row[f"routed_fraction_{label}"] = count / source_tokens
            row[f"routed_missed_mass_{label}"] = float(
                source[head].sum().item() - source[head][top - start].sum().item()
            )
            row[f"routed_output_error_{label}"] = error
        rows.append(row)
    return rows


def torch_is_finite(value: Any) -> Any:
    """Keep torch import local so offline schema/unit tooling stays lightweight."""

    import torch

    return torch.isfinite(value).all()
