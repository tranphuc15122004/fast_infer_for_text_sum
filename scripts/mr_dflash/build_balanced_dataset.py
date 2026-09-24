#!/usr/bin/env python3
"""Preview toàn bộ và build dataset MR-DFlash 50k theo quota source/bin.

ShareGPT/ArXiv được chuẩn hóa bằng normalizer hiện hành. Prompt length được
đếm bằng tokenizer sau khi áp dụng Qwen chat template; preview quét đủ dữ liệu,
không lấy mẫu trước khi tạo phân phối và báo quota khả thi.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from _common import read_jsonl, write_json


DEFAULT_SHAREGPT_SOURCE = (
    "/workspace/storage-shared/nlp/tungdd11/tungdecoder/ShareGPT/"
    "ShareGPT_V3_unfiltered_cleaned_split.json"
)
DEFAULT_ARXIV_SOURCE = (
    "/workspace/storage-shared/nlp/dungdx4/datasets/arxiv/train.label.jsonl"
)
DEFAULT_OUTPUT_ROOT = (
    "/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/"
    "mr_dflash_phase1_50k_50_50_10k"
)
DEFAULT_SHAREGPT_COUNT = 25_000
DEFAULT_ARXIV_COUNT = 25_000
DEFAULT_MAX_PROMPT_TOKENS = 10 * 1_024
BIN_WIDTH = 2 * 1_024
SOURCES = ("sharegpt", "arxiv")
SPLITS = ("train", "val", "test")
SOURCE_TARGETS = {"sharegpt": DEFAULT_SHAREGPT_COUNT, "arxiv": DEFAULT_ARXIV_COUNT}
BIN_TARGETS = {index: 10_000 for index in range(5)}


def _length_edges(max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS) -> tuple[int, ...]:
    if max_prompt_tokens != DEFAULT_MAX_PROMPT_TOKENS:
        raise ValueError(
            f"max_prompt_tokens phải là {DEFAULT_MAX_PROMPT_TOKENS} để giữ 5 bin đã chốt"
        )
    return tuple(BIN_WIDTH * (index + 1) for index in range(len(BIN_TARGETS)))


def length_bin(prompt_tokens: int, max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS) -> int | None:
    """Gán prompt vào 5 khoảng 2 Ki-token; None nếu vượt trần 10 Ki-token."""
    value = int(prompt_tokens)
    if value < 0:
        raise ValueError("prompt_tokens không được âm")
    _length_edges(max_prompt_tokens)
    if value > max_prompt_tokens:
        return None
    return min(max(0, (value - 1) // BIN_WIDTH), len(BIN_TARGETS) - 1)


def plan_source_bin_quotas(
    available: Mapping[str, Mapping[int, int]],
) -> dict[str, Any]:
    """Plan exact source and length-bin quotas or report why they cannot fit."""
    counts = {
        source: {bucket: int(available.get(source, {}).get(bucket, 0)) for bucket in BIN_TARGETS}
        for source in SOURCES
    }
    source_available = {source: sum(counts[source].values()) for source in SOURCES}
    bin_available = {
        bucket: sum(counts[source][bucket] for source in SOURCES) for bucket in BIN_TARGETS
    }
    source_shortages = {
        source: max(0, SOURCE_TARGETS[source] - source_available[source]) for source in SOURCES
    }
    bin_shortages = {
        bucket: max(0, BIN_TARGETS[bucket] - bin_available[bucket]) for bucket in BIN_TARGETS
    }
    reasons = []
    for source, shortage in source_shortages.items():
        if shortage:
            reasons.append(
                f"{source}: thiếu {shortage:,} eligible prompt (cần {SOURCE_TARGETS[source]:,}, "
                f"có {source_available[source]:,})"
            )
    for bucket, shortage in bin_shortages.items():
        if shortage:
            reasons.append(
                f"bin {bucket}: thiếu {shortage:,} prompt (cần {BIN_TARGETS[bucket]:,}, "
                f"có {bin_available[bucket]:,})"
            )

    share = "sharegpt"
    arxiv = "arxiv"
    lower: dict[int, int] = {}
    upper: dict[int, int] = {}
    ideal: dict[int, float] = {}
    for bucket, target in BIN_TARGETS.items():
        lower[bucket] = max(0, target - counts[arxiv][bucket])
        upper[bucket] = min(target, counts[share][bucket])
        total = bin_available[bucket]
        ideal[bucket] = target * counts[share][bucket] / total if total else 0.0
    min_share = sum(lower.values())
    max_share = sum(upper.values())
    if not source_shortages[share] and not any(bin_shortages.values()):
        target_share = SOURCE_TARGETS[share]
        if not min_share <= target_share <= max_share:
            reasons.append(
                f"không thể thỏa quota đồng thời: ShareGPT cần {target_share:,}, "
                f"khoảng khả thi theo bin là {min_share:,}–{max_share:,}"
            )

    plan: dict[str, Any] = {
        "feasible": not reasons,
        "source_targets": dict(SOURCE_TARGETS),
        "bin_targets": dict(BIN_TARGETS),
        "source_available": source_available,
        "bin_available": bin_available,
        "source_shortages": source_shortages,
        "bin_shortages": bin_shortages,
        "reasons": reasons,
        "source_bin_quotas": {},
    }
    if reasons:
        return plan

    # Begin near each bin's observed source composition, then adjust by the
    # smallest squared-error step until the global 25k/25k margin is exact.
    share_quotas = {
        bucket: min(upper[bucket], max(lower[bucket], round(ideal[bucket])))
        for bucket in BIN_TARGETS
    }
    delta = SOURCE_TARGETS[share] - sum(share_quotas.values())
    while delta > 0:
        choices = [
            (2 * (share_quotas[bucket] - ideal[bucket]) + 1, bucket)
            for bucket in BIN_TARGETS
            if share_quotas[bucket] < upper[bucket]
        ]
        if not choices:
            raise AssertionError("quota planner exhausted ShareGPT capacity")
        _cost, bucket = min(choices)
        share_quotas[bucket] += 1
        delta -= 1
    while delta < 0:
        choices = [
            (-2 * (share_quotas[bucket] - ideal[bucket]) + 1, bucket)
            for bucket in BIN_TARGETS
            if share_quotas[bucket] > lower[bucket]
        ]
        if not choices:
            raise AssertionError("quota planner exhausted ArXiv capacity")
        _cost, bucket = min(choices)
        share_quotas[bucket] -= 1
        delta += 1

    plan["source_bin_quotas"] = {
        "sharegpt": share_quotas,
        "arxiv": {
            bucket: BIN_TARGETS[bucket] - share_quotas[bucket] for bucket in BIN_TARGETS
        },
    }
    return plan


def select_candidates_by_bin_quota(
    candidates: Iterable[Mapping[str, Any]],
    *,
    source: str,
    bin_quotas: Mapping[int, int],
    seed: int,
) -> dict[str, dict[str, int]]:
    """Choose the smallest seeded ID ranks independently inside every bin."""
    quotas = {int(bucket): int(count) for bucket, count in bin_quotas.items()}
    if any(bucket not in BIN_TARGETS for bucket in quotas):
        raise ValueError("bin_quotas có bin không hợp lệ")
    if any(count < 0 for count in quotas.values()):
        raise ValueError("bin quota không được âm")
    heaps: dict[int, list[tuple[int, tuple[int, ...], str, int]]] = {
        bucket: [] for bucket in BIN_TARGETS
    }
    seen: set[str] = set()
    for row in candidates:
        sample_id = str(row.get("id", ""))
        if not sample_id or sample_id in seen:
            continue
        seen.add(sample_id)
        token_length = int(row["prompt_tokens"])
        bucket = length_bin(token_length)
        if bucket is None or not quotas.get(bucket, 0):
            continue
        rank = _stable_rank(seed, source, sample_id, "sample")
        heap = heaps[bucket]
        reverse_id = tuple(-ord(char) for char in sample_id) + (1,)
        entry = (-rank, reverse_id, sample_id, token_length)
        if len(heap) < quotas[bucket]:
            heapq.heappush(heap, entry)
        elif (rank, sample_id) < (-heap[0][0], heap[0][2]):
            heapq.heapreplace(heap, entry)

    selected: dict[str, dict[str, int]] = {}
    for bucket, heap in heaps.items():
        if len(heap) != quotas.get(bucket, 0):
            raise ValueError(
                f"{source}: bin {bucket} chỉ có {len(heap)}/{quotas.get(bucket, 0)} sample; "
                "dừng để không lặp hoặc đổi quota"
            )
        for _negative_rank, _reverse_id, sample_id, token_length in heap:
            selected[sample_id] = {"prompt_tokens": token_length, "length_bin": bucket}
    return dict(sorted(selected.items()))


def _flow_counts(
    capacities: Mapping[str, Mapping[int, int]],
    source_demands: Mapping[str, int],
    bin_demands: Mapping[int, int],
) -> dict[str, dict[int, int]]:
    """Deterministic integer max-flow for small source-by-bin split margins."""
    sources = tuple(source_demands)
    bins = tuple(bin_demands)
    source_node = 0
    source_nodes = {name: index + 1 for index, name in enumerate(sources)}
    bin_offset = 1 + len(sources)
    bin_nodes = {bucket: bin_offset + index for index, bucket in enumerate(bins)}
    sink = bin_offset + len(bins)
    graph: list[list[list[int]]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int) -> list[int]:
        forward = [end, capacity, len(graph[end])]
        reverse = [start, 0, len(graph[start])]
        graph[start].append(forward)
        graph[end].append(reverse)
        return forward

    for source in sources:
        add_edge(source_node, source_nodes[source], int(source_demands[source]))
    refs: dict[tuple[str, int], tuple[list[int], int]] = {}
    for source in sources:
        for bucket in bins:
            capacity = int(capacities.get(source, {}).get(bucket, 0))
            edge = add_edge(source_nodes[source], bin_nodes[bucket], capacity)
            refs[(source, bucket)] = (edge, capacity)
    for bucket in bins:
        add_edge(bin_nodes[bucket], sink, int(bin_demands[bucket]))

    target = sum(int(value) for value in source_demands.values())
    flow = 0
    while flow < target:
        parent: list[tuple[int, int] | None] = [None] * len(graph)
        queue = [source_node]
        parent[source_node] = (source_node, -1)
        for node in queue:
            for edge_index, edge in enumerate(graph[node]):
                if edge[1] > 0 and parent[edge[0]] is None:
                    parent[edge[0]] = (node, edge_index)
                    queue.append(edge[0])
                    if edge[0] == sink:
                        break
            if parent[sink] is not None:
                break
        if parent[sink] is None:
            break
        increment = target - flow
        node = sink
        while node != source_node:
            previous, edge_index = parent[node]
            increment = min(increment, graph[previous][edge_index][1])
            node = previous
        node = sink
        while node != source_node:
            previous, edge_index = parent[node]
            edge = graph[previous][edge_index]
            edge[1] -= increment
            graph[node][edge[2]][1] += increment
            node = previous
        flow += increment

    if flow != target or sum(bin_demands.values()) != target:
        raise ValueError("không thể tạo split đúng quota đồng thời theo source và bin")
    result = {source: {} for source in sources}
    for key, (edge, original_capacity) in refs.items():
        result[key[0]][key[1]] = original_capacity - edge[1]
    return result


def _allocate_five_percent(
    quotas: Mapping[str, Mapping[int, int]],
    already_assigned: Mapping[str, Mapping[int, int]] | None = None,
) -> dict[str, dict[int, int]]:
    """Allocate exact 5% split margins, rounding each source-bin cell fairly."""
    already_assigned = already_assigned or {}
    sources = tuple(quotas)
    bins = tuple(BIN_TARGETS)
    capacities = {
        source: {
            bucket: int(quotas[source][bucket]) - int(already_assigned.get(source, {}).get(bucket, 0))
            for bucket in bins
        }
        for source in sources
    }
    base = {
        source: {bucket: int(quotas[source][bucket]) // 20 for bucket in bins}
        for source in sources
    }
    for source in sources:
        if any(base[source][bucket] > capacities[source][bucket] for bucket in bins):
            raise ValueError("split 5% vượt source-bin capacity")

    source_targets = {source: sum(quotas[source].values()) // 20 for source in sources}
    bin_targets = {
        bucket: sum(quotas[source][bucket] for source in sources) // 20 for bucket in bins
    }
    source_residual = {
        source: source_targets[source] - sum(base[source].values()) for source in sources
    }
    bin_residual = {
        bucket: bin_targets[bucket] - sum(base[source][bucket] for source in sources)
        for bucket in bins
    }
    fractional_caps = {
        source: {
            bucket: int(
                quotas[source][bucket] % 20 != 0
                and capacities[source][bucket] - base[source][bucket] > 0
            )
            for bucket in bins
        }
        for source in sources
    }
    extras = _flow_counts(fractional_caps, source_residual, bin_residual)
    return {
        source: {bucket: base[source][bucket] + extras[source][bucket] for bucket in bins}
        for source in sources
    }


def _split_counts(total: int) -> dict[str, int]:
    train = int(total * 0.90)
    val = int(total * 0.05)
    return {"train": train, "val": val, "test": total - train - val}


def assign_stratified_splits(
    selections: Mapping[str, Mapping[str, Mapping[str, int]]],
    *,
    source_bin_quotas: Mapping[str, Mapping[int, int]],
    seed: int,
) -> dict[str, dict[str, str]]:
    """Assign deterministic splits with exact 90/5/5 source and bin margins."""
    val_counts = _allocate_five_percent(source_bin_quotas)
    test_counts = _allocate_five_percent(source_bin_quotas, val_counts)
    cell_counts = {
        source: {
            bucket: {
                "val": val_counts[source][bucket],
                "test": test_counts[source][bucket],
                "train": int(source_bin_quotas[source][bucket])
                - val_counts[source][bucket]
                - test_counts[source][bucket],
            }
            for bucket in BIN_TARGETS
        }
        for source in source_bin_quotas
    }
    result: dict[str, dict[str, str]] = {}
    for source, items in selections.items():
        result[source] = {}
        for bucket in BIN_TARGETS:
            ids = [
                sample_id
                for sample_id, item in items.items()
                if int(item["length_bin"]) == bucket
            ]
            expected = int(source_bin_quotas[source][bucket])
            if len(ids) != expected:
                raise ValueError(
                    f"{source} bin {bucket}: selected {len(ids)}, quota {expected}"
                )
            ids.sort(
                key=lambda sample_id: (
                    _stable_rank(seed, source, sample_id, f"split-bin-{bucket}"), sample_id
                )
            )
            val_count = cell_counts[source][bucket]["val"]
            test_count = cell_counts[source][bucket]["test"]
            for sample_id in ids[:val_count]:
                result[source][sample_id] = "val"
            for sample_id in ids[val_count : val_count + test_count]:
                result[source][sample_id] = "test"
            for sample_id in ids[val_count + test_count :]:
                result[source][sample_id] = "train"
    expected_by_source = {
        source: _split_counts(sum(source_bin_quotas[source].values()))
        for source in source_bin_quotas
    }
    for source, expected in expected_by_source.items():
        actual = Counter(result[source].values())
        if any(actual.get(split, 0) != count for split, count in expected.items()):
            raise AssertionError(f"split source {source} lệch quota: {dict(actual)} != {expected}")
    for bucket in BIN_TARGETS:
        actual = Counter(
            result[source][sample_id]
            for source, items in selections.items()
            for sample_id, item in items.items()
            if int(item["length_bin"]) == bucket
        )
        total = sum(source_bin_quotas[source][bucket] for source in source_bin_quotas)
        expected = _split_counts(total)
        if any(actual.get(split, 0) != count for split, count in expected.items()):
            raise AssertionError(f"split bin {bucket} lệch quota: {dict(actual)} != {expected}")
    return result


def _stable_rank(seed: int, source: str, sample_id: str, purpose: str) -> int:
    payload = f"{seed}\0{source}\0{purpose}\0{sample_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def prompt_token_counts(
    rows: Sequence[Mapping[str, Any]], tokenizer: Any
) -> list[int]:
    """Count exact rendered input tokens, including the assistant generation prefix."""
    rendered: list[str] = []
    for row in rows:
        conversations = row.get("conversations") or []
        prompt = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(prompt, str):
            raise TypeError("apply_chat_template(tokenize=False) phải trả về text")
        rendered.append(prompt)
    if not rendered:
        return []
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    if len(input_ids) != len(rendered):
        raise ValueError("tokenizer trả số input_ids không khớp batch")
    return [len(ids) for ids in input_ids]


def _iter_batches(rows: Iterable[dict[str, Any]], batch_size: int):
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _tqdm(rows: Iterable[Any], *, desc: str):
    try:
        from tqdm import tqdm

        return tqdm(rows, desc=desc, unit="sample", dynamic_ncols=True)
    except ImportError:  # pragma: no cover - tqdm is installed on B200
        return rows


def _repair_trailing_partial_jsonl(path: Path) -> None:
    """Drop only a malformed, non-newline-terminated final cache record."""
    if not path.exists() or path.stat().st_size == 0:
        return
    last_good_end = 0
    with path.open("rb") as handle:
        line_number = 0
        while True:
            line = handle.readline()
            if not line:
                break
            line_number += 1
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                if line.endswith(b"\n") or handle.peek(1):
                    raise ValueError(f"JSONL cache hỏng tại {path}:{line_number}") from exc
                with path.open("r+b") as repair:
                    repair.truncate(last_good_end)
                break
            last_good_end = handle.tell()


def _iter_cache(path: Path):
    _repair_trailing_partial_jsonl(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} phải là JSON object")
                yield row


def _cache_identity(source_path: Path, tokenizer_ref: str) -> dict[str, Any]:
    stat = source_path.stat()
    return {
        "normalized_source": str(source_path.resolve()),
        "normalized_size_bytes": stat.st_size,
        "normalized_mtime_ns": stat.st_mtime_ns,
        "tokenizer": str(tokenizer_ref),
    }


def _load_existing_scan(
    cache_path: Path,
    info_path: Path,
    identity: Mapping[str, Any],
    *,
    resume: bool,
) -> tuple[int, set[str]]:
    if not resume:
        cache_path.unlink(missing_ok=True)
        info_path.unlink(missing_ok=True)
        return 0, set()
    if not cache_path.exists():
        return 0, set()
    if not info_path.exists():
        raise ValueError(f"thiếu metadata của length cache {info_path}; dùng output-root mới")
    previous = json.loads(info_path.read_text(encoding="utf-8"))
    if previous != dict(identity):
        raise ValueError(
            f"length cache không khớp source/tokenizer: {cache_path}; dùng output-root mới"
        )
    completed = 0
    seen: set[str] = set()
    for row in _iter_cache(cache_path):
        completed += 1
        if row.get("status") == "unique":
            seen.add(str(row["id"]))
    return completed, seen


def scan_normalized_source(
    source_path: Path,
    cache_path: Path,
    info_path: Path,
    tokenizer: Any,
    *,
    source: str,
    tokenizer_ref: str,
    batch_size: int,
    resume: bool,
    max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
) -> dict[str, Any]:
    """Tokenize a full normalized source and checkpoint lengths as JSONL."""
    identity = _cache_identity(source_path, tokenizer_ref)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    completed, seen_ids = _load_existing_scan(
        cache_path, info_path, identity, resume=resume
    )
    if not info_path.exists():
        write_json(info_path, identity)

    total_rows = sum(1 for _ in read_jsonl(source_path))
    if completed > total_rows:
        raise ValueError(f"length cache nhiều dòng hơn source: {source_path}")
    if completed == total_rows and cache_path.exists():
        pass
    else:
        rows = read_jsonl(source_path)
        pending_rows = []
        for index, row in enumerate(_tqdm(rows, desc=f"Tokenize {source}")):
            if index < completed:
                continue
            pending_rows.append(row)
            if len(pending_rows) >= batch_size:
                _scan_batch(
                    pending_rows,
                    cache_path,
                    tokenizer,
                    seen_ids,
                    source=source,
                )
                pending_rows = []
        if pending_rows:
            _scan_batch(
                pending_rows,
                cache_path,
                tokenizer,
                seen_ids,
                source=source,
            )

    cache_path.touch(exist_ok=True)
    return summarize_scan(cache_path, max_prompt_tokens=max_prompt_tokens)


def _scan_batch(
    rows: Sequence[Mapping[str, Any]],
    cache_path: Path,
    tokenizer: Any,
    seen_ids: set[str],
    *,
    source: str,
) -> None:
    unique_indices: list[int] = []
    unique_rows: list[Mapping[str, Any]] = []
    statuses = [""] * len(rows)
    for index, row in enumerate(rows):
        sample_id = str(row.get("id", ""))
        if not sample_id:
            raise ValueError(f"{source}: normalized row thiếu id")
        if sample_id in seen_ids:
            statuses[index] = "duplicate"
        else:
            statuses[index] = "unique"
            seen_ids.add(sample_id)
            unique_indices.append(index)
            unique_rows.append(row)

    token_lengths = prompt_token_counts(unique_rows, tokenizer)
    length_by_index = dict(zip(unique_indices, token_lengths))
    payload = []
    for index, row in enumerate(rows):
        entry: dict[str, Any] = {
            "id": str(row["id"]),
            "status": statuses[index],
        }
        if statuses[index] == "unique":
            entry["prompt_tokens"] = int(length_by_index[index])
        payload.append(entry)

    from _common import append_jsonl_durable

    append_jsonl_durable(cache_path, payload)


def summarize_scan(cache_path: Path, max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS) -> dict[str, Any]:
    lengths: list[int] = []
    bins: Counter[int] = Counter()
    unique_rows = duplicates = over_cap = 0
    eligible_lengths: list[int] = []
    for row in _iter_cache(cache_path):
        if row.get("status") == "duplicate":
            duplicates += 1
            continue
        if row.get("status") != "unique":
            continue
        token_length = int(row["prompt_tokens"])
        unique_rows += 1
        lengths.append(token_length)
        bucket = length_bin(token_length, max_prompt_tokens)
        if bucket is None:
            over_cap += 1
        else:
            bins[bucket] += 1
            eligible_lengths.append(token_length)
    return {
        "normalized_rows": unique_rows + duplicates,
        "unique_rows": unique_rows,
        "duplicate_ids": duplicates,
        "over_cap": over_cap,
        "eligible_rows": len(eligible_lengths),
        "length_stats_all": _length_stats(lengths),
        "length_stats_eligible": _length_stats(eligible_lengths),
        "eligible_bin_counts": {key: bins.get(key, 0) for key in range(len(_length_edges(max_prompt_tokens)))},
    }


def _length_stats(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "p25": None, "median": None, "p75": None, "p90": None, "max": None, "mean": None}
    ordered = sorted(int(value) for value in values)

    def quantile(q: float) -> float:
        position = (len(ordered) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p25": round(quantile(0.25), 2),
        "median": round(quantile(0.50), 2),
        "p75": round(quantile(0.75), 2),
        "p90": round(quantile(0.90), 2),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 2),
    }


def _prepare_source(
    script_name: str,
    input_path: Path,
    output_path: Path,
    *,
    resume: bool,
) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name(script_name)),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]
    if resume and output_path.exists():
        command.append("--resume")
    print("[build_balanced_dataset] normalize:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _merge_source_splits(
    share_path: Path,
    arxiv_path: Path,
    output_path: Path,
    *,
    start_with: str,
) -> int:
    paths = {"sharegpt": share_path, "arxiv": arxiv_path}
    handles = {source: path.open("r", encoding="utf-8") for source, path in paths.items()}
    count = 0
    try:
        temporary = output_path.with_name(f".{output_path.name}.tmp")
        with temporary.open("w", encoding="utf-8") as output:
            turn = start_with
            exhausted = set()
            while len(exhausted) < 2:
                source = turn
                line = handles[source].readline()
                if not line:
                    exhausted.add(source)
                    turn = "arxiv" if source == "sharegpt" else "sharegpt"
                    continue
                output.write(line if line.endswith("\n") else line + "\n")
                count += 1
                turn = "arxiv" if source == "sharegpt" else "sharegpt"
            output.flush()
            os.fsync(output.fileno())
    finally:
        for handle in handles.values():
            handle.close()
    os.replace(temporary, output_path)
    return count


def _write_selected_source(
    source_path: Path,
    stage_dir: Path,
    *,
    source: str,
    selected: Mapping[str, Mapping[str, int]],
    splits: Mapping[str, str],
) -> dict[str, int]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        split: (stage_dir / f"{source}_{split}.jsonl").open("w", encoding="utf-8")
        for split in SPLITS
    }
    written = Counter()
    seen: set[str] = set()
    try:
        for row in _tqdm(read_jsonl(source_path), desc=f"Write {source}"):
            sample_id = str(row.get("id", ""))
            if sample_id in seen:
                continue
            seen.add(sample_id)
            item = selected.get(sample_id)
            if item is None:
                continue
            split = splits[sample_id]
            metadata = dict(row.get("metadata") or {})
            metadata.update(
                {
                    "prompt_token_length": int(item["prompt_tokens"]),
                    "prompt_length_bin": int(item["length_bin"]),
                    "split": split,
                }
            )
            result = {**row, "metadata": metadata}
            outputs[split].write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            written[split] += 1
    finally:
        for handle in outputs.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
    expected = Counter(splits.values())
    if written != expected:
        raise ValueError(
            f"{source}: ghi được {dict(written)}, dự kiến {dict(expected)}; "
            "kiểm tra source và length cache"
        )
    return dict(written)


def _write_distribution_figure(path: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on server profile
        raise RuntimeError("cần matplotlib để ghi biểu đồ phân phối (PNG)") from exc

    labels = ["0–2 Ki", "2–4 Ki", "4–6 Ki", "6–8 Ki", "8–10 Ki", ">10 Ki"]
    positions = list(range(len(labels)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for offset, source in ((-width / 2, "sharegpt"), (width / 2, "arxiv")):
        summary = summaries[source]
        counts = [
            int(summary["eligible_bin_counts"].get(bucket, 0)) for bucket in BIN_TARGETS
        ] + [int(summary["over_cap"])]
        axis.bar([position + offset for position in positions], counts, width, label=source)
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Số prompt duy nhất")
    axis.set_xlabel("Độ dài prompt sau chat template (token)")
    axis.set_title("Phân phối độ dài toàn bộ dữ liệu hợp lệ")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Nguồn")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _write_reports(
    root: Path,
    *,
    summaries: Mapping[str, Mapping[str, Any]],
    selections: Mapping[str, Mapping[str, Mapping[str, int]]],
    split_assignments: Mapping[str, Mapping[str, str]],
    quota_plan: Mapping[str, Any],
) -> dict[str, Any]:
    edges = _length_edges()
    report: dict[str, Any] = {
        "schema_version": "mr_dflash_length_distribution_v2",
        "max_prompt_tokens": DEFAULT_MAX_PROMPT_TOKENS,
        "bin_width": BIN_WIDTH,
        "length_edges_inclusive": list(edges),
        "quota_feasible": bool(quota_plan["feasible"]),
        "quota_plan": dict(quota_plan),
        "sources": {},
    }
    csv_path = root / "reports" / "length_distribution.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source", "population", "bin", "token_range", "count", "fraction"),
        )
        writer.writeheader()
        for source in SOURCES:
            summary = dict(summaries[source])
            selected = selections.get(source, {})
            selected_bins = Counter(item["length_bin"] for item in selected.values())
            selected_lengths = [int(item["prompt_tokens"]) for item in selected.values()]
            split_counts = Counter(split_assignments.get(source, {}).values())
            summary["selected_rows"] = len(selected)
            summary["selected_length_stats"] = _length_stats(selected_lengths)
            summary["selected_bin_counts"] = {
                index: selected_bins.get(index, 0) for index in BIN_TARGETS
            }
            summary["selected_split_counts"] = {
                split: split_counts.get(split, 0) for split in SPLITS
            }
            planned = quota_plan["source_bin_quotas"].get(source, {})
            summary["planned_bin_quotas"] = {
                index: int(planned.get(index, 0)) for index in BIN_TARGETS
            }
            report["sources"][source] = summary

            populations: dict[str, tuple[int, Mapping[Any, int]]] = {}
            all_counts: Counter[Any] = Counter()
            cache_path = root / ".build_work" / "lengths" / f"{source}.jsonl"
            for row in _iter_cache(cache_path):
                if row.get("status") != "unique":
                    continue
                value = int(row["prompt_tokens"])
                bucket = length_bin(value)
                all_counts[bucket if bucket is not None else "over_cap"] += 1
            populations["all_unique"] = (summary["unique_rows"], all_counts)
            populations["eligible"] = (
                summary["eligible_rows"], summary["eligible_bin_counts"]
            )
            populations["planned_quota"] = (sum(planned.values()), planned)
            populations["selected"] = (len(selected), selected_bins)
            for population, (denominator, counts) in populations.items():
                for index, edge in enumerate(edges):
                    lower = 0 if index == 0 else edges[index - 1] + 1
                    count = int(counts.get(index, 0))
                    writer.writerow(
                        {
                            "source": source,
                            "population": population,
                            "bin": index,
                            "token_range": f"{lower}-{edge}",
                            "count": count,
                            "fraction": round(count / denominator, 6) if denominator else 0,
                        }
                    )
                if population == "all_unique":
                    count = int(counts.get("over_cap", 0))
                    writer.writerow(
                        {
                            "source": source,
                            "population": population,
                            "bin": "over_cap",
                            "token_range": f">{DEFAULT_MAX_PROMPT_TOKENS}",
                            "count": count,
                            "fraction": round(count / denominator, 6) if denominator else 0,
                        }
                    )

    global_bin_counts = {
        bucket: sum(int(summaries[source]["eligible_bin_counts"].get(bucket, 0)) for source in SOURCES)
        for bucket in BIN_TARGETS
    }
    report["global"] = {
        "unique_rows": sum(int(summaries[source]["unique_rows"]) for source in SOURCES),
        "duplicate_ids": sum(int(summaries[source]["duplicate_ids"]) for source in SOURCES),
        "eligible_rows": sum(int(summaries[source]["eligible_rows"]) for source in SOURCES),
        "over_cap": sum(int(summaries[source]["over_cap"]) for source in SOURCES),
        "eligible_bin_counts": global_bin_counts,
        "selected_rows": sum(len(selections.get(source, {})) for source in SOURCES),
        "selected_split_counts": {
            split: sum(
                1
                for source in SOURCES
                for value in split_assignments.get(source, {}).values()
                if value == split
            )
            for split in SPLITS
        },
    }
    write_json(root / "reports" / "length_distribution.json", report)
    _write_distribution_figure(root / "reports" / "length_distribution.png", summaries)
    return report


def _input_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _preview_identity(
    sharegpt_raw: Path, arxiv_raw: Path, *, tokenizer_ref: str, seed: int
) -> dict[str, Any]:
    return {
        "schema_version": "mr_dflash_build_preview_v1",
        "inputs": {
            "sharegpt": _input_identity(sharegpt_raw),
            "arxiv": _input_identity(arxiv_raw),
        },
        "tokenizer": str(tokenizer_ref),
        "seed": int(seed),
        "max_prompt_tokens": DEFAULT_MAX_PROMPT_TOKENS,
        "source_targets": dict(SOURCE_TARGETS),
        "bin_targets": [
            {"bin": bucket, "count": BIN_TARGETS[bucket]} for bucket in BIN_TARGETS
        ],
    }

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview rồi build dataset MR-DFlash 50k theo quota source và độ dài"
    )
    parser.add_argument("--sharegpt-source", default=DEFAULT_SHAREGPT_SOURCE)
    parser.add_argument("--arxiv-source", default=DEFAULT_ARXIV_SOURCE)
    parser.add_argument("--tokenizer", required=True, help="đường dẫn local tới tokenizer Qwen3-4B")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="tiếp tục từ normalized data/cache đã có")
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="quét toàn bộ dữ liệu, ghi báo cáo/quota khả thi, không tạo split",
    )
    parser.add_argument(
        "--confirm-build",
        action="store_true",
        help="xác nhận build đủ 50k sau khi đã xem preview report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("batch-size phải dương")
    if args.preview_only and args.confirm_build:
        raise SystemExit("không dùng --confirm-build cùng --preview-only")
    if not args.preview_only and not args.confirm_build:
        raise SystemExit(
            "build đầy đủ cần --confirm-build sau khi đã xem preview trên B200"
        )

    sharegpt_raw = Path(args.sharegpt_source)
    arxiv_raw = Path(args.arxiv_source)
    for path in (sharegpt_raw, arxiv_raw):
        if not path.is_file():
            raise FileNotFoundError(f"không tìm thấy raw source: {path}")

    root = Path(args.output_root)
    work = root / ".build_work"
    final_paths = [root / "normalized" / f"{split}_prompts.jsonl" for split in SPLITS]
    existing_final = [path for path in final_paths if path.exists()]
    manifest_path = root / "manifests" / "build_manifest.json"
    preview_manifest_path = root / "manifests" / "preview_manifest.json"
    identity = _preview_identity(
        sharegpt_raw, arxiv_raw, tokenizer_ref=args.tokenizer, seed=args.seed
    )
    if args.preview_only:
        if root.exists() and any(root.iterdir()) and not args.resume:
            raise FileExistsError(
                f"output-root không rỗng: {root}; dùng --resume hoặc chọn thư mục mới"
            )
        if manifest_path.exists():
            raise FileExistsError(f"dataset đã build xong theo {manifest_path}; dùng output-root mới")
        if work.exists() and any(work.iterdir()) and not args.resume:
            raise FileExistsError(f"đã có preview work cache ở {work}; dùng --resume hoặc output-root mới")
        if preview_manifest_path.exists() and not args.resume:
            raise FileExistsError(
                f"preview đã tồn tại tại {preview_manifest_path}; dùng --resume hoặc output-root mới"
            )
    else:
        if manifest_path.exists():
            raise FileExistsError(f"dataset đã hoàn tất theo {manifest_path}; dùng output-root mới")
        if not preview_manifest_path.exists():
            raise FileNotFoundError(
                f"chưa có preview {preview_manifest_path}; chạy --preview-only và xem report trước"
            )
        previous_preview = json.loads(preview_manifest_path.read_text(encoding="utf-8"))
        if not previous_preview.get("quota_feasible", False):
            raise ValueError(
                "preview cho thấy quota không khả thi; build bị dừng trước khi xử lý lại dữ liệu"
            )
        if previous_preview.get("identity") != identity:
            raise ValueError(
                "preview không khớp raw source/tokenizer/seed hiện tại; chạy lại --preview-only"
            )
        if existing_final and not (args.resume and work.exists()):
            raise FileExistsError(
                "output đã có split file; dùng output-root mới để tránh ghi đè: "
                + ", ".join(str(path) for path in existing_final)
            )
        if work.exists() and any(work.iterdir()) and not args.resume:
            raise FileExistsError(f"đã có preview work cache ở {work}; build cần --resume")

    work.mkdir(parents=True, exist_ok=True)
    (work / "source").mkdir(parents=True, exist_ok=True)
    (work / "lengths").mkdir(parents=True, exist_ok=True)
    source_identity_path = work / "source" / "source_identity.json"
    if args.resume and source_identity_path.exists():
        previous_identity = json.loads(source_identity_path.read_text(encoding="utf-8"))
        if previous_identity != identity:
            raise ValueError(
                "raw source/tokenizer/seed đã đổi so với cache; dùng output-root mới"
            )
    elif args.resume and any(
        (work / "source" / name).exists()
        for name in ("sharegpt_prompts.jsonl", "arxiv_prompts.jsonl")
    ):
        raise ValueError(f"thiếu source identity trong {work / 'source'}; dùng output-root mới")
    else:
        write_json(source_identity_path, identity)
    normalized_paths = {
        "sharegpt": work / "source" / "sharegpt_prompts.jsonl",
        "arxiv": work / "source" / "arxiv_prompts.jsonl",
    }
    _prepare_source(
        "prepare_sharegpt.py",
        sharegpt_raw,
        normalized_paths["sharegpt"],
        resume=args.resume,
    )
    _prepare_source(
        "prepare_arxiv.py",
        arxiv_raw,
        normalized_paths["arxiv"],
        resume=args.resume,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        use_fast=True,
    )
    summaries: dict[str, dict[str, Any]] = {}
    cache_paths: dict[str, Path] = {}
    for source in SOURCES:
        cache_path = work / "lengths" / f"{source}.jsonl"
        info_path = work / "lengths" / f"{source}.info.json"
        cache_paths[source] = cache_path
        print(
            f"[build_balanced_dataset] exact-token scan source={source} "
            f"batch_size={args.batch_size} cap={DEFAULT_MAX_PROMPT_TOKENS}",
            flush=True,
        )
        summaries[source] = scan_normalized_source(
            normalized_paths[source],
            cache_path,
            info_path,
            tokenizer,
            source=source,
            tokenizer_ref=args.tokenizer,
            batch_size=args.batch_size,
            resume=args.resume,
        )

    available_by_source = {
        source: summaries[source]["eligible_bin_counts"] for source in SOURCES
    }
    quota_plan = plan_source_bin_quotas(available_by_source)
    empty_selections: dict[str, dict[str, dict[str, int]]] = {source: {} for source in SOURCES}
    empty_splits: dict[str, dict[str, str]] = {source: {} for source in SOURCES}
    report = _write_reports(
        root,
        summaries=summaries,
        selections=empty_selections,
        split_assignments=empty_splits,
        quota_plan=quota_plan,
    )
    preview_record = {
        "identity": identity,
        "quota_feasible": quota_plan["feasible"],
        "quota_plan": quota_plan,
        "report": str(root / "reports" / "length_distribution.json"),
        "csv": str(root / "reports" / "length_distribution.csv"),
        "figure": str(root / "reports" / "length_distribution.png"),
    }
    write_json(preview_manifest_path, preview_record)

    for source in SOURCES:
        summary = summaries[source]
        print(
            f"[distribution] {source}: normalized={summary['normalized_rows']} "
            f"unique={summary['unique_rows']} duplicates={summary['duplicate_ids']} "
            f"eligible<={DEFAULT_MAX_PROMPT_TOKENS}={summary['eligible_rows']} "
            f"over_cap={summary['over_cap']} bins={summary['eligible_bin_counts']}",
            flush=True,
        )
    print(
        f"[quota] feasible={quota_plan['feasible']} "
        f"source_available={quota_plan['source_available']} "
        f"bin_available={quota_plan['bin_available']} "
        f"source_bin_quotas={quota_plan['source_bin_quotas']}",
        flush=True,
    )
    if quota_plan["reasons"]:
        for reason in quota_plan["reasons"]:
            print(f"[quota][shortage] {reason}", file=sys.stderr, flush=True)
    if args.preview_only:
        print(
            f"[build_balanced_dataset] preview complete; report="
            f"{root / 'reports' / 'length_distribution.json'}; "
            f"csv={root / 'reports' / 'length_distribution.csv'}; "
            f"figure={root / 'reports' / 'length_distribution.png'}; no split files written",
            flush=True,
        )
        return 0
    if not quota_plan["feasible"]:
        raise ValueError(
            "không đủ dữ liệu để đồng thời đạt 25k/source và 10k/bin; "
            "đã dừng, không lặp mẫu hoặc đổi phân phối"
        )

    source_bin_quotas = quota_plan["source_bin_quotas"]
    selections: dict[str, dict[str, dict[str, int]]] = {}
    for source in SOURCES:
        selections[source] = select_candidates_by_bin_quota(
            (
                {"id": row["id"], "prompt_tokens": row["prompt_tokens"]}
                for row in _iter_cache(cache_paths[source])
                if row.get("status") == "unique"
            ),
            source=source,
            bin_quotas=source_bin_quotas[source],
            seed=args.seed,
        )
    split_assignments = assign_stratified_splits(
        selections, source_bin_quotas=source_bin_quotas, seed=args.seed
    )
    report = _write_reports(
        root,
        summaries=summaries,
        selections=selections,
        split_assignments=split_assignments,
        quota_plan=quota_plan,
    )

    selected_dir = work / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    staged_counts = {}
    for source in SOURCES:
        staged_counts[source] = _write_selected_source(
            normalized_paths[source],
            selected_dir,
            source=source,
            selected=selections[source],
            splits=split_assignments[source],
        )

    normalized_output = root / "normalized"
    normalized_output.mkdir(parents=True, exist_ok=True)
    source_split_counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        first = "sharegpt" if args.seed % 2 == 0 else "arxiv"
        count = _merge_source_splits(
            selected_dir / f"sharegpt_{split}.jsonl",
            selected_dir / f"arxiv_{split}.jsonl",
            normalized_output / f"{split}_prompts.jsonl",
            start_with=first,
        )
        expected = staged_counts["sharegpt"][split] + staged_counts["arxiv"][split]
        if count != expected:
            raise ValueError(f"{split}: merged {count} rows, expected {expected}")
        source_split_counts[split] = {
            source: staged_counts[source][split] for source in SOURCES
        }

    expected_split_counts = {"train": 45_000, "val": 2_500, "test": 2_500}
    actual_split_counts = {
        split: sum(source_split_counts[split].values()) for split in SPLITS
    }
    if actual_split_counts != expected_split_counts:
        raise ValueError(f"split totals {actual_split_counts} != {expected_split_counts}")
    all_ids_hash: dict[str, str] = {}
    for split in SPLITS:
        digest = hashlib.sha256()
        for row in read_jsonl(normalized_output / f"{split}_prompts.jsonl"):
            digest.update(str(row["id"]).encode("utf-8"))
            digest.update(b"\n")
        all_ids_hash[split] = digest.hexdigest()
    manifest = {
        "schema_version": "mr_dflash_balanced_dataset_v2",
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "token_measurement": "rendered Qwen3 chat template plus generation prompt, then tokenizer with add_special_tokens=false",
        "max_prompt_tokens": DEFAULT_MAX_PROMPT_TOKENS,
        "source_targets": dict(SOURCE_TARGETS),
        "bin_targets": dict(BIN_TARGETS),
        "source_bin_quotas": source_bin_quotas,
        "split_ratio": {"train": 0.90, "val": 0.05, "test": 0.05},
        "selected_counts": {source: len(selections[source]) for source in SOURCES},
        "selected_split_counts": actual_split_counts,
        "source_split_counts": source_split_counts,
        "total_selected": sum(len(items) for items in selections.values()),
        "input_sources": {
            "sharegpt": _input_identity(sharegpt_raw),
            "arxiv": _input_identity(arxiv_raw),
        },
        "preview_manifest": str(preview_manifest_path),
        "length_distribution_report": str(root / "reports" / "length_distribution.json"),
        "length_distribution_csv": str(root / "reports" / "length_distribution.csv"),
        "length_distribution_figure": str(root / "reports" / "length_distribution.png"),
        "split_id_sha256": all_ids_hash,
    }
    write_json(manifest_path, manifest)

    for path in normalized_paths.values():
        path.unlink(missing_ok=True)
    for path in selected_dir.glob("*.jsonl"):
        path.unlink(missing_ok=True)
    try:
        (work / "source").rmdir()
        selected_dir.rmdir()
        work.rmdir() if not any(work.iterdir()) else None
    except OSError:
        pass
    print(
        f"[build_balanced_dataset] done total={manifest['total_selected']} "
        f"splits={actual_split_counts}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
