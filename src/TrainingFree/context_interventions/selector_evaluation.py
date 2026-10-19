"""Các phép đo Phase A từ snapshot có metadata tường minh.

Coverage là attention mass, không phải semantic accuracy, acceptance hay GPU I/O.
Không suy đoán layer từ số hàng và không tái dựng query đã bị gộp mất.
"""

from collections import defaultdict
from collections.abc import Mapping

import numpy as np


def normalized(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
        raise ValueError("attention must be a nonempty finite nonnegative vector with positive mass")
    return values / values.sum()


def layer_index(snapshot: Mapping, layer: int) -> int:
    if "target_layer_ids" not in snapshot:
        raise ValueError("snapshot thiếu target_layer_ids; cần manifest hoặc mapping legacy tường minh")
    ids = [int(value) for value in snapshot["target_layer_ids"]]
    if len(ids) != len(set(ids)) or layer not in ids:
        raise ValueError(f"target layer {layer} unavailable; recorded layers={ids}")
    return ids.index(layer)


def layer_vector(snapshot: Mapping, layer: int, field: str = "target"):
    if field not in snapshot:
        raise ValueError(f"snapshot thiếu {field}")
    index = layer_index(snapshot, layer)
    values = np.asarray(snapshot[field])
    if values.ndim != 2 or values.shape[0] != len(snapshot["target_layer_ids"]):
        raise ValueError(f"invalid shape for {field}")
    return values[index].astype(np.float64)


def pool_queries(snapshot: Mapping, layer: int, *, mode: str, scope: str):
    if mode not in {"max", "mean"} or scope not in {"accepted", "all"}:
        raise ValueError("invalid query pooling mode/scope")
    if "target_query_attention" not in snapshot:
        raise ValueError("snapshot lacks per-query attention; max/query ablation unavailable")
    values = np.asarray(snapshot["target_query_attention"])
    if values.ndim != 3 or values.shape[0] != len(snapshot["target_layer_ids"]):
        raise ValueError("invalid per-query attention shape")
    query_count = values.shape[1]
    if scope == "accepted":
        if "accepted_candidates" not in snapshot:
            raise ValueError("accepted query scope requires accepted_candidates")
        query_count = int(snapshot["accepted_candidates"]) + 1
    else:
        # q[0:15] predicts the 15 candidates; q[15] is the bonus query.
        query_count = min(15, query_count)
    if not 1 <= query_count <= values.shape[1]:
        raise ValueError("accepted_candidates outside captured query block")
    rows = values[layer_index(snapshot, layer), :query_count].astype(np.float64)
    if not np.isfinite(rows).all() or (rows < 0).any():
        raise ValueError("invalid query attention values")
    pooled = rows.max(axis=0) if mode == "max" else rows.mean(axis=0)
    return normalized(pooled)


def protected_topk(scores, budget: int, *, sink: int = 16, recent: int = 32):
    scores = normalized(scores)
    if budget < 1 or sink < 0 or recent < 0:
        raise ValueError("invalid budget/protection")
    n = len(scores)
    protected = set(range(min(sink, n))) | set(range(max(0, n - recent), n))
    if len(protected) > budget:
        raise ValueError("budget smaller than protected positions")
    candidates = np.array([i for i in range(n) if i not in protected], dtype=int)
    order = np.lexsort((candidates, -scores[candidates]))
    picked = set(candidates[order[:max(0, min(n, budget) - len(protected))]]) | protected
    return np.array(sorted(picked), dtype=int)


def select_chunks(scores, budget: int, *, chunk_size: int, sink: int = 16, recent: int = 32):
    scores = normalized(scores)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if chunk_size == 1:
        return protected_topk(scores, budget, sink=sink, recent=recent)
    n = len(scores)
    selected = set(range(min(sink, n))) | set(range(max(0, n - recent), n))
    if len(selected) > budget:
        raise ValueError("budget smaller than protected positions")
    chunks = [tuple(range(start, min(n, start + chunk_size))) for start in range(0, n, chunk_size)]
    order = sorted(range(len(chunks)), key=lambda c: (-scores[list(chunks[c])].sum(), c))
    for c in order:
        additions = set(chunks[c]) - selected
        if len(selected) + len(additions) <= min(budget, n):
            selected.update(additions)
    # Không bù token rời rạc: biến thể này thực sự chọn nguyên chunk còn vừa budget.
    return np.array(sorted(selected), dtype=int)


def spatial_continuity(indices):
    indices = np.sort(np.asarray(indices))
    return float(np.mean(np.diff(indices) == 1)) if len(indices) > 1 else 1.


def pool_keys(scores, mode: str, kernel: int = 7):
    """Ablation spatial pooling; không đại diện toàn bộ SnapKV/RocketKV."""
    if kernel < 1 or kernel % 2 == 0 or mode not in {"mean", "max"}:
        raise ValueError("invalid spatial pooling")
    # PyTorch avg_pool1d padding counts zero padding; max padding is -inf.
    padded = np.pad(scores, kernel // 2, constant_values=0. if mode == "mean" else -np.inf)
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
    return windows.mean(-1) if mode == "mean" else windows.max(-1)


def summarize_rows(rows):
    by_document = defaultdict(list)
    for row in rows:
        by_document[row["sample_id"]].append(float(row["coverage"]))
    return {"documents": len(by_document), "snapshots": len(rows),
            "mean_coverage": float(np.mean([np.mean(v) for v in by_document.values()])) if rows else None}


CANDIDATES = {
    "C0_L0": (0,), "C4_L12": (12,), "C1_L18": (18,), "C3_L27": (27,),
    "C2_L35": (35,), "E1_L18_L35": (18, 35),
    "E2_L18_L27_L35": (18, 27, 35), "E3_L14_L16_L18_L20": (14, 16, 18, 20),
}


def reference_attention(snapshot, layers, prefix):
    """Mean-query reference, từng layer conditional trên cùng prefix trước khi trộn."""
    return np.mean([normalized(layer_vector(snapshot, layer)[:prefix]) for layer in layers], axis=0)


def selector_scores(snapshot, prefix):
    """Trả cả available và unavailable; không thay thuật toán khi thiếu dữ liệu."""
    available, unavailable = {}, {}
    builders = {
        "parent_L18": lambda: parent_query_attention(snapshot, 18),
        "max_L18_accepted": lambda: pool_queries(snapshot, 18, mode="max", scope="accepted"),
        "mean_L18_accepted": lambda: pool_queries(snapshot, 18, mode="mean", scope="accepted"),
        "max_L18_all_diagnostic": lambda: pool_queries(snapshot, 18, mode="max", scope="all"),
        "mean_L18_all": lambda: pool_queries(snapshot, 18, mode="mean", scope="all"),
        "ensemble_L18_L35_accepted": lambda: .5 * normalized(pool_queries(snapshot, 18, mode="max", scope="accepted")[:prefix]) + .5 * normalized(pool_queries(snapshot, 35, mode="max", scope="accepted")[:prefix]),
    }
    if "target_query_attention" not in snapshot:
        builders["mean_L18_legacy"] = lambda: layer_vector(snapshot, 18)
        builders["mean_L35_legacy"] = lambda: layer_vector(snapshot, 35)
    for name, build in builders.items():
        try:
            available[name] = normalized(build()[:prefix])
        except ValueError as exc:
            unavailable[name] = str(exc)
    # Các adaptation dùng cùng tín hiệu đầu vào; đặt nhãn khác baseline gốc.
    if "mean_L18_all" in available:
        for mode in ("mean", "max"):
            available[f"spatial_{mode}_k7_mean_L18"] = normalized(pool_keys(available["mean_L18_all"], mode))
    return available, unavailable


def parent_query_attention(snapshot, layer):
    """Query r=L ở vòng hiện tại, cùng thời điểm chọn với multi-query."""
    if "target_query_attention" not in snapshot or "accepted_candidates" not in snapshot:
        raise ValueError("parent control requires same-round query attention and accepted_candidates")
    rows = np.asarray(snapshot["target_query_attention"])[layer_index(snapshot, layer)]
    index = int(snapshot["accepted_candidates"])
    if not 0 <= index < len(rows):
        raise ValueError("parent query index outside recorded block")
    return normalized(rows[index])
