"""Stable protocol identities and source-grouped folds for campaign B."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SUPPORTED_GAMMAS = (3, 7, 11, 15)


@dataclass(frozen=True)
class CampaignConfig:
    """Protocol values that affect the B campaign's action grid."""

    gammas: tuple[int, ...] = SUPPORTED_GAMMAS
    verify_prefixes: tuple[int, ...] = SUPPORTED_GAMMAS
    n_splits: int = 4
    seed: int = 42
    bootstrap_seed: int = 20261013

    def __post_init__(self) -> None:
        for name in ("gammas", "verify_prefixes"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique supported gamma values")
            if any(value not in SUPPORTED_GAMMAS for value in values):
                raise ValueError(f"{name} contains an unsupported gamma value")
        if tuple(sorted(self.gammas)) != self.gammas:
            raise ValueError("gammas must be sorted from shortest to longest")
        if tuple(sorted(self.verify_prefixes)) != self.verify_prefixes:
            raise ValueError("verify_prefixes must be sorted from shortest to longest")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least two groups")

    @property
    def draft_actions(self) -> tuple[str, ...]:
        return tuple(f"gamma{gamma}" for gamma in self.gammas)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "dflash-speculation-b.config.v1",
            "gammas": list(self.gammas),
            "verify_prefixes": list(self.verify_prefixes),
            "n_splits": self.n_splits,
            "seed": self.seed,
            "bootstrap_seed": self.bootstrap_seed,
        }


def _group_order(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def assign_group_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_splits: int = 4,
    seed: int = 42,
) -> dict[str, int]:
    """Assign each source document to one deterministic, dataset-balanced fold.

    Rows can contain multiple rounds or repetitions. They are collapsed by
    source_group_id before assignment, and cross-dataset duplicate groups keep
    one shared fold.
    """

    if n_splits < 2:
        raise ValueError("n_splits must be at least two groups")
    group_datasets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_id = str(row.get("source_group_id", "")).strip()
        dataset = str(row.get("dataset", "")).strip()
        if not group_id:
            raise ValueError("every row requires a nonempty source_group_id")
        if not dataset:
            raise ValueError(f"row for source_group_id={group_id!r} requires dataset")
        group_datasets[group_id].add(dataset)
    if len(group_datasets) < n_splits:
        raise ValueError(
            f"need at least {n_splits} unique source groups, got {len(group_datasets)}"
        )

    dataset_group_counts: dict[str, int] = defaultdict(int)
    for datasets in group_datasets.values():
        for dataset in datasets:
            dataset_group_counts[dataset] += 1

    ordered_groups = sorted(
        group_datasets,
        key=lambda group_id: (
            -len(group_datasets[group_id]),
            _group_order(group_id, seed),
            group_id,
        ),
    )
    fold_counts: dict[str, list[int]] = {
        dataset: [0] * n_splits for dataset in dataset_group_counts
    }
    fold_group_counts = [0] * n_splits
    assignments: dict[str, int] = {}

    def score(fold: int, datasets: set[str]) -> tuple[float, int, str, int]:
        imbalance = 0.0
        for dataset, total in dataset_group_counts.items():
            expected = total / n_splits
            after = fold_counts[dataset][fold] + int(dataset in datasets)
            imbalance += (after - expected) ** 2
        after_total = fold_group_counts[fold] + 1
        expected_total = len(group_datasets) / n_splits
        imbalance += (after_total - expected_total) ** 2
        tie_break = hashlib.sha256(f"{seed}:{fold}".encode("ascii")).hexdigest()
        # The squared-error objective can tie when adding to a lighter and a
        # heavier fold (for example counts 1 and 2 around a target of 3).
        # Prefer the lighter fold to keep the group counts balanced.
        return imbalance, fold_group_counts[fold], tie_break, fold

    for group_id in ordered_groups:
        datasets = group_datasets[group_id]
        empty_folds = [fold for fold, count in enumerate(fold_group_counts) if count == 0]
        candidates = empty_folds or list(range(n_splits))
        selected_fold = min(candidates, key=lambda fold: score(fold, datasets))
        assignments[group_id] = selected_fold
        fold_group_counts[selected_fold] += 1
        for dataset in datasets:
            fold_counts[dataset][selected_fold] += 1
    return assignments


def _token_digest(token_ids: Sequence[int]) -> str:
    normalized: list[int] = []
    for token in token_ids:
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError("token IDs used in state identity must be integers")
        normalized.append(int(token))
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def state_identity(
    *,
    dataset: str,
    source_group_id: str,
    prompt_token_ids: Sequence[int],
    output_prefix_ids: Sequence[int],
    pending_anchor: int,
    anchor_position: int,
    round_index: int,
) -> str:
    """Hash a causal state independently of its tested gamma/action."""

    if not dataset or not source_group_id:
        raise ValueError("state identity requires dataset and source_group_id")
    if anchor_position < 0 or round_index < 0:
        raise ValueError("anchor_position and round_index must be nonnegative")
    if anchor_position != len(prompt_token_ids) + len(output_prefix_ids):
        raise ValueError("anchor_position must equal prompt plus processed output length")
    payload = {
        "dataset": str(dataset),
        "source_group_id": str(source_group_id),
        "prompt_sha256": _token_digest(prompt_token_ids),
        "output_prefix_sha256": _token_digest(output_prefix_ids),
        "pending_anchor": int(pending_anchor),
        "anchor_position": int(anchor_position),
        "round_index": int(round_index),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
