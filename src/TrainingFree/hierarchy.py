"""Target-native multi-resolution source hierarchy for RECAP-KV V3."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


def _matrix(value: Any, name: str):
    import torch

    tensor = value.detach().float() if hasattr(value, "detach") else torch.as_tensor(value).float()
    if tensor.ndim != 2 or tensor.shape[0] <= 0 or tensor.shape[1] <= 0:
        raise ValueError(f"{name} must have shape [tokens, dimension]")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain finite values")
    return tensor


def _vector(value: Any, name: str):
    import torch

    tensor = value.detach().float() if hasattr(value, "detach") else torch.as_tensor(value).float()
    if tensor.ndim != 1 or tensor.shape[0] <= 0:
        raise ValueError(f"{name} must have shape [dimension]")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain finite values")
    return tensor


def select_representatives(points: Any, count: int) -> list[int]:
    """Select deterministic farthest-point representatives from a point set."""

    import torch

    matrix = _matrix(points, "points")
    requested = int(count)
    if requested <= 0:
        raise ValueError("count must be positive")
    requested = min(requested, int(matrix.shape[0]))
    norms = torch.sum(matrix * matrix, dim=-1)
    first = min(range(int(matrix.shape[0])), key=lambda index: (-float(norms[index]), index))
    selected = [first]
    nearest = torch.sum((matrix - matrix[first]) ** 2, dim=-1)
    nearest[first] = 0.0
    while len(selected) < requested:
        candidate = min(
            range(int(matrix.shape[0])),
            key=lambda index: (-float(nearest[index]), index),
        )
        selected.append(candidate)
        distances = torch.sum((matrix - matrix[candidate]) ** 2, dim=-1)
        nearest = torch.minimum(nearest, distances)
        nearest[selected] = 0.0
    return selected


def coverage_radius(points: Any, representatives: Sequence[int]) -> float:
    """Return max distance from a point to its nearest representative."""

    import torch

    matrix = _matrix(points, "points")
    indices = tuple(int(index) for index in representatives)
    if not indices or any(index < 0 or index >= matrix.shape[0] for index in indices):
        raise ValueError("representatives must be non-empty and in range")
    if len(set(indices)) != len(indices):
        raise ValueError("representatives must not contain duplicates")
    reps = matrix[list(indices)]
    distances = torch.cdist(matrix, reps, p=2).min(dim=1).values
    result = float(distances.max().item())
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("coverage radius must be finite and non-negative")
    return result


@dataclass(frozen=True)
class Cluster:
    """A contiguous source span and representatives in global source indices."""

    start: int
    end: int
    representatives: tuple[int, ...]
    radius: float
    children: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        return self.end - self.start


@dataclass
class SourceHierarchy:
    """Multi-resolution hierarchy for one target layer/KV head."""

    keys: Any
    region_size: int
    block_size: int
    reps_per_block: int
    reps_per_region: int
    regions: tuple[Cluster, ...]
    blocks: tuple[Cluster, ...]

    @property
    def source_tokens(self) -> int:
        return int(self.keys.shape[0])

    @property
    def total_representatives(self) -> int:
        indices = {
            index
            for cluster in self.regions + self.blocks
            for index in cluster.representatives
        }
        return len(indices)

    @property
    def index_overhead(self) -> float:
        return self.total_representatives / self.source_tokens


def _cluster(points: Any, start: int, end: int, count: int, children: Sequence[int] = ()) -> Cluster:
    local = _matrix(points, "cluster points")
    representatives = select_representatives(local, count)
    radius = coverage_radius(local, representatives)
    return Cluster(
        start=int(start),
        end=int(end),
        representatives=tuple(int(start + index) for index in representatives),
        radius=radius,
        children=tuple(int(index) for index in children),
    )


def build_source_hierarchy(
    source_keys: Any,
    *,
    region_size: int,
    block_size: int,
    reps_per_block: int,
    reps_per_region: int | None = None,
) -> SourceHierarchy:
    """Build a deterministic region/block hierarchy from source keys."""

    keys = _matrix(source_keys, "source_keys")
    region_size = int(region_size)
    block_size = int(block_size)
    reps_per_block = int(reps_per_block)
    reps_per_region = reps_per_block if reps_per_region is None else int(reps_per_region)
    if region_size <= 0 or block_size <= 0:
        raise ValueError("region_size and block_size must be positive")
    if block_size > region_size:
        raise ValueError("block_size must not exceed region_size")
    if reps_per_block <= 0 or reps_per_region <= 0:
        raise ValueError("representative counts must be positive")

    blocks: list[Cluster] = []
    for start in range(0, int(keys.shape[0]), block_size):
        end = min(start + block_size, int(keys.shape[0]))
        blocks.append(_cluster(keys[start:end], start, end, reps_per_block))

    regions: list[Cluster] = []
    for start in range(0, int(keys.shape[0]), region_size):
        end = min(start + region_size, int(keys.shape[0]))
        children = [
            index
            for index, block in enumerate(blocks)
            if block.start >= start and block.end <= end
        ]
        regions.append(_cluster(keys[start:end], start, end, reps_per_region, children))

    return SourceHierarchy(
        keys=keys,
        region_size=region_size,
        block_size=block_size,
        reps_per_block=reps_per_block,
        reps_per_region=reps_per_region,
        regions=tuple(regions),
        blocks=tuple(blocks),
    )


def cluster_log_bounds(query: Any, keys: Any, cluster: Cluster) -> tuple[float, float]:
    """Return log upper and lower bounds for a cluster partition contribution."""

    upper, lower = cluster_log_bounds_batch(query, keys, (cluster,))
    return upper[0], lower[0]


def cluster_log_bounds_batch(
    query: Any, keys: Any, clusters: Sequence[Cluster]
) -> tuple[list[float], list[float]]:
    """Compute cluster bounds with one representative matmul."""

    import torch

    vector = _vector(query, "query")
    matrix = _matrix(keys, "keys")
    if not clusters:
        return [], []
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("query/key dimensions must match")
    if any(
        cluster.start < 0 or cluster.end > matrix.shape[0] or cluster.end <= cluster.start
        for cluster in clusters
    ):
        raise ValueError("cluster span is invalid")
    scale = math.sqrt(float(matrix.shape[1]))
    query_norm = float(torch.linalg.vector_norm(vector).item())
    representative_indices = [
        index for cluster in clusters for index in cluster.representatives
    ]
    representative_logits = matrix[representative_indices] @ vector / scale
    upper_values: list[float] = []
    lower_values: list[float] = []
    offset = 0
    for cluster in clusters:
        width = len(cluster.representatives)
        logits = representative_logits[offset : offset + width]
        offset += width
        upper = math.log(cluster.size) + float(logits.max().item())
        upper += query_norm * float(cluster.radius) / scale
        lower = float(torch.logsumexp(logits, dim=0).item())
        if not math.isfinite(upper) or not math.isfinite(lower):
            raise ValueError("cluster bounds must be finite")
        upper_values.append(upper)
        lower_values.append(lower)
    return upper_values, lower_values


def exact_cluster_log_partition(query: Any, keys: Any, cluster: Cluster) -> float:
    """Compute exact log partition for audit only; never used by routing."""

    import torch

    vector = _vector(query, "query")
    matrix = _matrix(keys, "keys")
    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("query/key dimensions must match")
    logits = matrix[cluster.start : cluster.end] @ vector / math.sqrt(float(matrix.shape[1]))
    return float(torch.logsumexp(logits, dim=0).item())
