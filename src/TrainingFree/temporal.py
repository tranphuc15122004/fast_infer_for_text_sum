"""Offline temporal-support and GQA-aware source-cost analysis."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


def _tensor(value: Any) -> Any:
    import torch

    tensor = value.detach().float() if hasattr(value, "detach") else torch.as_tensor(value).float()
    if tensor.ndim != 2:
        raise ValueError("source attention must have shape [query_heads, source_tokens]")
    if not bool(torch.isfinite(tensor).all()) or bool((tensor < 0.0).any()):
        raise ValueError("source attention must be finite and non-negative")
    return tensor


def _fmt(value: float | int) -> str:
    rendered = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


def _p99(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.99 * len(ordered)) - 1))
    return ordered[index]


def _validate_mapping(mapping: Sequence[int], query_heads: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in mapping)
    if len(result) != query_heads or not result or any(value < 0 for value in result):
        raise ValueError("query_to_kv must map every query head to a non-negative KV head")
    return result


def select_top_supports(
    source_attention: Any,
    *,
    fractions: Sequence[float],
) -> dict[float, list[set[int]]]:
    """Select per-query-head source positions by current attention mass."""

    tensor = _tensor(source_attention)
    source_tokens = int(tensor.shape[1])
    result: dict[float, list[set[int]]] = {}
    for requested in fractions:
        fraction = float(requested)
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise ValueError("support fractions must be finite and in (0, 1]")
        count = min(source_tokens, max(1, math.ceil(source_tokens * fraction)))
        supports = [
            {int(index) for index in tensor[head].topk(count, largest=True, sorted=False).indices.tolist()}
            for head in range(int(tensor.shape[0]))
        ]
        result[fraction] = supports
    return result


def _k95_count(values: Any, mass_level: float) -> int:
    import torch

    if not math.isfinite(mass_level) or not 0.0 < mass_level <= 1.0:
        raise ValueError("mass_level must be finite and in (0, 1]")
    ordered = torch.sort(values, descending=True).values
    total = float(ordered.sum().item())
    if total <= 0.0 or not math.isfinite(total):
        return int(values.shape[0])
    cumulative = torch.cumsum(ordered, dim=0) / total
    reached = torch.nonzero(cumulative >= mass_level, as_tuple=False)
    return int(reached[0, 0].item() + 1) if reached.numel() else int(values.shape[0])


def select_adaptive_supports(
    source_attention: Any,
    *,
    alpha: float,
    mass_level: float = 0.95,
) -> list[set[int]]:
    """Select ceil(alpha * K_mass) positions independently per query head."""

    tensor = _tensor(source_attention)
    multiplier = float(alpha)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("alpha must be finite and positive")
    source_tokens = int(tensor.shape[1])
    supports: list[set[int]] = []
    for head in range(int(tensor.shape[0])):
        k_mass = _k95_count(tensor[head], float(mass_level))
        count = min(source_tokens, max(1, math.ceil(multiplier * k_mass)))
        supports.append(
            {int(index) for index in tensor[head].topk(count, largest=True, sorted=False).indices.tolist()}
        )
    return supports


def gqa_union_supports(
    supports: Sequence[set[int]],
    *,
    query_to_kv: Sequence[int],
) -> dict[int, set[int]]:
    """Union query-head support at the physical KV-head granularity."""

    mapping = _validate_mapping(query_to_kv, len(supports))
    unions: dict[int, set[int]] = {}
    for head, support in enumerate(supports):
        kv_head = mapping[head]
        unions.setdefault(kv_head, set()).update(int(index) for index in support)
    return unions


def blockify_supports(
    supports: Sequence[set[int]],
    *,
    source_tokens: int,
    block_size: int,
) -> list[set[int]]:
    """Round token supports up to contiguous source blocks."""

    size = int(block_size)
    total = int(source_tokens)
    if size <= 0 or total <= 0:
        raise ValueError("source_tokens and block_size must be positive")
    expanded: list[set[int]] = []
    for support in supports:
        blocks = {int(index) // size for index in support}
        tokens: set[int] = set()
        for block in blocks:
            start = block * size
            tokens.update(range(start, min(total, start + size)))
        expanded.append(tokens)
    return expanded


def evaluate_supports(
    source_attention: Any,
    supports: Sequence[set[int]],
    *,
    query_to_kv: Sequence[int],
) -> dict[str, float]:
    """Measure query-head and physical GQA-union missed mass/cost."""

    tensor = _tensor(source_attention)
    mapping = _validate_mapping(query_to_kv, int(tensor.shape[0]))
    source_tokens = int(tensor.shape[1])
    if len(supports) != int(tensor.shape[0]):
        raise ValueError("support count must match query-head count")
    import torch

    query_mask = torch.zeros(
        (int(tensor.shape[0]), source_tokens), dtype=torch.bool, device=tensor.device
    )
    for head, support in enumerate(supports):
        indices = [int(index) for index in support]
        if any(index < 0 or index >= source_tokens for index in indices):
            raise ValueError("support index is outside source span")
        if indices:
            query_mask[head, indices] = True
    query_missed = (tensor.masked_fill(query_mask, 0.0)).sum(dim=1).tolist()
    unions = gqa_union_supports(supports, query_to_kv=mapping)
    kv_heads = max(unions) + 1
    kv_mask = torch.zeros(
        (kv_heads, source_tokens), dtype=torch.bool, device=tensor.device
    )
    for kv_head, support in unions.items():
        indices = [int(index) for index in support]
        if any(index < 0 or index >= source_tokens for index in indices):
            raise ValueError("GQA support index is outside source span")
        if indices:
            kv_mask[kv_head, indices] = True
    gqa_mask = kv_mask[list(mapping)]
    gqa_missed = tensor.masked_fill(gqa_mask, 0.0).sum(dim=1).tolist()
    query_expansion = sum(len(support) for support in supports) / (
        len(supports) * source_tokens
    )
    gqa_expansion = sum(len(support) for support in unions.values()) / (
        len(unions) * source_tokens
    )
    return {
        "query_expansion_fraction": float(query_expansion),
        "query_missed_mass": float(sum(query_missed) / len(query_missed)),
        "query_p99_missed_mass": float(_p99(query_missed)),
        "gqa_expansion_fraction": float(gqa_expansion),
        "gqa_missed_mass": float(sum(gqa_missed) / len(gqa_missed)),
        "gqa_p99_missed_mass": float(_p99(gqa_missed)),
        "mean_missed_mass": float(sum(gqa_missed) / len(gqa_missed)),
        "p99_missed_mass": float(_p99(gqa_missed)),
    }


@dataclass(frozen=True)
class TemporalConfig:
    lags: tuple[int, ...] = (1, 2, 4, 8, 16)
    budgets: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    alphas: tuple[float, ...] = (1.0, 1.25, 1.50)
    refresh_intervals: tuple[int, ...] = (2, 4, 8, 16)
    block_sizes: tuple[int, ...] = (16, 32, 64)
    mass_level: float = 0.95

    def __post_init__(self) -> None:
        if any(int(value) <= 0 for value in (*self.lags, *self.refresh_intervals, *self.block_sizes)):
            raise ValueError("lags, refresh intervals and block sizes must be positive")
        if any(not 0.0 < float(value) <= 1.0 for value in self.budgets):
            raise ValueError("budgets must be in (0, 1]")
        if any(float(value) <= 0.0 or not math.isfinite(float(value)) for value in self.alphas):
            raise ValueError("alphas must be finite and positive")
        if not 0.0 < float(self.mass_level) <= 1.0:
            raise ValueError("mass_level must be in (0, 1]")


class TemporalAnalyzer:
    """Stateful, audit-only analyzer for one document and one layer."""

    def __init__(
        self,
        *,
        query_to_kv: Sequence[int],
        source_tokens: int,
        config: TemporalConfig,
    ) -> None:
        self.query_to_kv = tuple(int(value) for value in query_to_kv)
        self.source_tokens = int(source_tokens)
        if self.source_tokens <= 0:
            raise ValueError("source_tokens must be positive")
        _validate_mapping(self.query_to_kv, len(self.query_to_kv))
        self.config = config
        self._history: list[dict[str, Any]] = []
        self._working: dict[str, list[set[int]]] = {}

    def observe(self, source_attention: Any) -> dict[str, Any]:
        tensor = _tensor(source_attention)
        if int(tensor.shape[1]) != self.source_tokens:
            raise ValueError("source attention width does not match source_tokens")
        fixed = select_top_supports(tensor, fractions=self.config.budgets)
        adaptive = {
            float(alpha): select_adaptive_supports(
                tensor, alpha=float(alpha), mass_level=self.config.mass_level
            )
            for alpha in self.config.alphas
        }
        result: dict[str, Any] = {
            "step": len(self._history),
            "gqa_oracle": {},
            "lag": {},
            "adaptive_lag": {},
            "recurrent": {},
        }
        for budget, supports in fixed.items():
            result["gqa_oracle"][f"B{_fmt(budget)}"] = evaluate_supports(
                tensor, supports, query_to_kv=self.query_to_kv
            )
        for lag in self.config.lags:
            if len(self._history) < int(lag):
                continue
            previous = self._history[-int(lag)]
            for budget, supports in previous["fixed"].items():
                result["lag"][f"L{int(lag)}_B{_fmt(budget)}"] = evaluate_supports(
                    tensor, supports, query_to_kv=self.query_to_kv
                )
            for alpha, supports in previous["adaptive"].items():
                result["adaptive_lag"][f"L{int(lag)}_A{_fmt(alpha)}"] = evaluate_supports(
                    tensor, supports, query_to_kv=self.query_to_kv
                )
        for refresh_interval in self.config.refresh_intervals:
            for block_size in self.config.block_sizes:
                for alpha in self.config.alphas:
                    key = f"R{int(refresh_interval)}_B{int(block_size)}_A{_fmt(alpha)}"
                    refresh = (
                        key not in self._working
                        or len(self._history) % int(refresh_interval) == 0
                    )
                    if refresh:
                        token_supports = adaptive[float(alpha)]
                        self._working[key] = blockify_supports(
                            token_supports,
                            source_tokens=self.source_tokens,
                            block_size=int(block_size),
                        )
                    metrics = evaluate_supports(
                        tensor,
                        self._working[key],
                        query_to_kv=self.query_to_kv,
                    )
                    metrics.update(
                        {
                            "refresh": bool(refresh),
                            "source_cost_ratio": (
                                1.0 if refresh else metrics["gqa_expansion_fraction"]
                            ),
                            "working_source_cost_ratio": metrics["gqa_expansion_fraction"],
                            "block_size": int(block_size),
                            "refresh_interval": int(refresh_interval),
                            "alpha": float(alpha),
                        }
                    )
                    result["recurrent"][key] = metrics
                for budget, token_supports in fixed.items():
                    key = f"R{int(refresh_interval)}_C{int(block_size)}_F{_fmt(budget)}"
                    refresh = (
                        key not in self._working
                        or len(self._history) % int(refresh_interval) == 0
                    )
                    if refresh:
                        self._working[key] = blockify_supports(
                            token_supports,
                            source_tokens=self.source_tokens,
                            block_size=int(block_size),
                        )
                    metrics = evaluate_supports(
                        tensor,
                        self._working[key],
                        query_to_kv=self.query_to_kv,
                    )
                    metrics.update(
                        {
                            "refresh": bool(refresh),
                            "source_cost_ratio": (
                                1.0 if refresh else metrics["gqa_expansion_fraction"]
                            ),
                            "working_source_cost_ratio": metrics["gqa_expansion_fraction"],
                            "block_size": int(block_size),
                            "refresh_interval": int(refresh_interval),
                            "budget": float(budget),
                        }
                    )
                    result["recurrent"][key] = metrics
        self._history.append({"fixed": fixed, "adaptive": adaptive})
        return result
