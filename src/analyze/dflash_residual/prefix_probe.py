"""Tiny candidate-lattice probes for E13 recoverability experiments.

The probes are deliberately small and operate on frozen DFlash traces.  The
target candidate logits are used only as supervision; they are never included
in the inference features.  Evaluation selects one of the already recorded
Top-16 candidates and measures longest target-compatible prefix length.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

from .metrics import _blocks, _observed_acceptance
from .prefix_alignment import _mean
from .prefix_gap import prefix_oracle_length


OBJECTIVES = ("pointwise", "pairwise", "listwise", "prefix_utility")


def _finite_row(row: Mapping[str, Any]) -> bool:
    candidate = row.get("candidate_logits")
    target = row.get("target_candidate_logits")
    candidates = row.get("candidate_token_ids")
    if not isinstance(candidate, list) or not isinstance(target, list) or not isinstance(candidates, list):
        return False
    if len(candidate) != len(target) or len(candidate) != len(candidates) or len(candidate) < 2:
        return False
    try:
        return all(math.isfinite(float(value)) for value in candidate + target)
    except (TypeError, ValueError):
        return False


def _usable(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if row.get("status", "ok") == "ok" and _finite_row(row)]


def _document_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("task_regime", row.get("dataset", "other"))), str(row.get("document_id"))


def split_documents(
    rows: Sequence[Mapping[str, Any]],
    *,
    test_fraction: float = 0.3,
    seed: int = 42,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any]]:
    """Make a deterministic document-disjoint split."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie in (0, 1)")
    keys = sorted({_document_key(row) for row in rows})
    if len(keys) < 2:
        return [], list(rows), {"train_documents": [], "test_documents": [list(key) for key in keys]}
    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, min(len(keys) - 1, round(len(keys) * test_fraction)))
    test_keys = set(shuffled[:test_count])
    train = [row for row in rows if _document_key(row) not in test_keys]
    test = [row for row in rows if _document_key(row) in test_keys]
    return train, test, {
        "train_documents": [list(key) for key in sorted(set(keys) - test_keys)],
        "test_documents": [list(key) for key in sorted(test_keys)],
    }


def _feature_row(row: Mapping[str, Any]) -> list[list[float]]:
    """Build test-time features from DFlash-side fields only."""

    logits = [float(value) for value in row["candidate_logits"]]
    mean = sum(logits) / len(logits)
    variance = sum((value - mean) ** 2 for value in logits) / max(1, len(logits))
    scale = math.sqrt(variance) or 1.0
    position = float(row.get("draft_position", 1)) / 16.0
    result: list[list[float]] = []
    for rank, (token_id, logit) in enumerate(zip(row["candidate_token_ids"], logits), start=1):
        token = int(token_id)
        phase = (token % 997) / 997.0 * 2.0 * math.pi
        result.append([
            (logit - mean) / scale,
            1.0 - (rank - 1) / max(1, len(logits) - 1),
            position,
            math.sin(phase),
            math.cos(phase),
        ])
    return result


def _tensorize(rows: Sequence[Mapping[str, Any]], device: Any) -> tuple[Any, Any, Any, Any]:
    import torch

    features = torch.tensor([_feature_row(row) for row in rows], dtype=torch.float32, device=device)
    target_logits = torch.tensor(
        [[float(value) for value in row["target_candidate_logits"]] for row in rows],
        dtype=torch.float32,
        device=device,
    )
    target_ids = torch.tensor(
        [int(row["target_token_id"]) for row in rows], dtype=torch.long, device=device
    )
    candidate_ids = torch.tensor(
        [[int(value) for value in row["candidate_token_ids"]] for row in rows],
        dtype=torch.long,
        device=device,
    )
    return features, target_logits, target_ids, candidate_ids


def _target_indices(target_ids: Any, candidate_ids: Any) -> Any:
    import torch

    matches = candidate_ids.eq(target_ids.unsqueeze(1))
    labels = matches.float().argmax(dim=1).long()
    return labels, matches.any(dim=1)


def _prefix_weights(rows: Sequence[Mapping[str, Any]], max_prefix: int = 16) -> Any:
    import torch

    blocks = _blocks(rows)
    if not blocks:
        return torch.ones(max_prefix, dtype=torch.float32)
    oracle_lengths = [min(prefix_oracle_length(block, max_prefix), max_prefix) for block in blocks]
    weights = [sum(length >= position for length in oracle_lengths) / len(oracle_lengths) for position in range(1, max_prefix + 1)]
    return torch.tensor(weights, dtype=torch.float32)


class _TinySelector:
    def __init__(self, input_dim: int = 5, hidden_dim: int = 16) -> None:
        import torch.nn as nn

        self.module = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)


def _loss(scores: Any, target_logits: Any, labels: Any, valid_labels: Any, objective: str, weights: Any) -> Any:
    import torch
    import torch.nn.functional as F

    if objective == "pointwise":
        if not bool(valid_labels.any()):
            return scores.sum() * 0.0
        return F.cross_entropy(scores[valid_labels], labels[valid_labels])
    target_probs = F.softmax(target_logits, dim=-1)
    log_probs = F.log_softmax(scores, dim=-1)
    kl = F.kl_div(log_probs, target_probs, reduction="none").sum(dim=-1)
    if objective == "listwise":
        return kl.mean()
    if objective == "prefix_utility":
        row_positions = torch.clamp(torch.arange(scores.shape[0], device=scores.device) * 0 + 0, min=0)
        # Caller replaces this zero vector with a position-aware multiplier by
        # passing a repeated weight tensor when batching full trace rows.
        return (kl * weights.to(scores.device)).mean()
    if objective == "pairwise":
        preferred = target_logits.unsqueeze(2) > target_logits.unsqueeze(1)
        score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
        pair_loss = -F.logsigmoid(score_diff[preferred])
        return pair_loss.mean() if pair_loss.numel() else kl.mean()
    raise ValueError(f"unknown objective: {objective}")


def _train_model(
    train_rows: Sequence[Mapping[str, Any]],
    *,
    objective: str,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> tuple[Any, list[float]]:
    import torch

    torch.manual_seed(seed)
    random.seed(seed)
    features, target_logits, target_ids, candidate_ids = _tensorize(train_rows, device)
    labels, valid_labels = _target_indices(target_ids, candidate_ids)
    model = _TinySelector().module.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    losses: list[float] = []
    position_weights = _prefix_weights(train_rows).to(device)
    row_weights = position_weights[
        torch.tensor(
            [max(1, min(16, int(row.get("draft_position", 1)))) - 1 for row in train_rows],
            dtype=torch.long,
            device=device,
        )
    ]
    for _ in range(epochs):
        order = torch.randperm(features.shape[0], device=device)
        shuffled_features = features[order]
        shuffled_targets = target_logits[order]
        shuffled_labels = labels[order]
        shuffled_valid = valid_labels[order]
        shuffled_weights = row_weights[order]
        optimizer.zero_grad(set_to_none=True)
        scores = model(shuffled_features).squeeze(-1)
        loss = _loss(
            scores,
            shuffled_targets,
            shuffled_labels,
            shuffled_valid,
            objective,
            shuffled_weights if objective == "prefix_utility" else torch.ones_like(shuffled_weights),
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, losses


def _select_rows(model: Any, rows: Sequence[Mapping[str, Any]], device: str) -> list[int]:
    import torch

    if not rows:
        return []
    features, _, _, candidate_ids = _tensorize(rows, device)
    with torch.no_grad():
        scores = model(features).squeeze(-1)
    chosen = scores.argmax(dim=1)
    return [int(candidate_ids[index, chosen[index]].item()) for index in range(len(rows))]


def _evaluate(model: Any, rows: Sequence[Mapping[str, Any]], device: str, max_prefix: int = 16) -> dict[str, Any]:
    selected = _select_rows(model, rows, device)
    blocks = _blocks(rows)
    cursor = 0
    probe_lengths: list[int] = []
    d_lengths: list[int] = []
    oracle_lengths: list[int] = []
    for block in blocks:
        block_selected = selected[cursor:cursor + len(block)]
        cursor += len(block)
        target = [int(row["target_token_id"]) for row in block]
        probe_length = 0
        for predicted, expected in zip(block_selected, target):
            if predicted != expected:
                break
            probe_length += 1
        probe_lengths.append(min(probe_length, max_prefix))
        d_lengths.append(min(_observed_acceptance(block), max_prefix))
        oracle_lengths.append(min(prefix_oracle_length(block, max_prefix), max_prefix))
    mat_d = _mean(d_lengths)
    mat_probe = _mean(probe_lengths)
    mat_oracle = _mean(oracle_lengths)
    gap = (mat_oracle - mat_d) if mat_oracle is not None and mat_d is not None else None
    recovery = ((mat_probe - mat_d) / gap) if gap and gap > 0 else None
    return {
        "blocks": len(blocks),
        "mat_d": mat_d,
        "mat_probe": mat_probe,
        "mat_o16": mat_oracle,
        "oracle_gap": gap,
        "oracle_recovery": recovery,
        "probe_survival": {
            str(position): sum(length >= position for length in probe_lengths) / len(probe_lengths)
            for position in range(1, max_prefix + 1)
        } if probe_lengths else {},
    }


def run_probe_suite(
    rows: Iterable[Mapping[str, Any]],
    *,
    objectives: Sequence[str] = OBJECTIVES,
    test_fraction: float = 0.3,
    epochs: int = 16,
    learning_rate: float = 1e-2,
    seed: int = 42,
    device: str = "auto",
    max_prefix: int = 16,
) -> dict[str, Any]:
    """Train/evaluate all E13 probes with document-disjoint splits."""

    usable = _usable(rows)
    if device == "auto":
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not usable:
        return {"status": "unavailable", "reason": "no_finite_target_and_draft_logits"}
    regimes: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in usable:
        regimes[str(row.get("task_regime", row.get("dataset", "other")))].append(row)
    output: dict[str, Any] = {
        "status": "ok",
        "experiment": "E13",
        "config": {
            "objectives": list(objectives),
            "test_fraction": test_fraction,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
            "device": device,
            "max_prefix": max_prefix,
            "features": "normalized DFlash logit, candidate rank, draft position, hashed token phase",
            "target_logits_as_features": False,
        },
        "regimes": {},
    }
    for regime, regime_rows in sorted(regimes.items()):
        train_rows, test_rows, split = split_documents(regime_rows, test_fraction=test_fraction, seed=seed)
        regime_result: dict[str, Any] = {
            "rows": len(regime_rows),
            "documents": len({_document_key(row) for row in regime_rows}),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_documents": len(split["train_documents"]),
            "test_documents": len(split["test_documents"]),
            "split": split,
            "objectives": {},
        }
        if not train_rows or not test_rows:
            regime_result["status"] = "inconclusive_small_split"
            output["regimes"][regime] = regime_result
            continue
        for objective in objectives:
            if objective not in OBJECTIVES:
                raise ValueError(f"unknown objective: {objective}")
            model, losses = _train_model(
                train_rows,
                objective=objective,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
            )
            regime_result["objectives"][objective] = {
                "train_loss_initial": losses[0] if losses else None,
                "train_loss_final": losses[-1] if losses else None,
                "train": _evaluate(model, train_rows, device, max_prefix=max_prefix),
                "test": _evaluate(model, test_rows, device, max_prefix=max_prefix),
            }
        output["regimes"][regime] = regime_result
    summary_rows = [row for regime, regime_rows in regimes.items() if regime != "canonical" for row in regime_rows]
    if len({_document_key(row) for row in summary_rows}) >= 2:
        train_rows, test_rows, split = split_documents(summary_rows, test_fraction=test_fraction, seed=seed)
        pooled: dict[str, Any] = {
            "rows": len(summary_rows),
            "documents": len({_document_key(row) for row in summary_rows}),
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_documents": len(split["train_documents"]),
            "test_documents": len(split["test_documents"]),
            "split": split,
            "objectives": {},
        }
        for objective in objectives:
            model, losses = _train_model(
                train_rows,
                objective=objective,
                epochs=epochs,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
            )
            pooled["objectives"][objective] = {
                "train_loss_initial": losses[0] if losses else None,
                "train_loss_final": losses[-1] if losses else None,
                "train": _evaluate(model, train_rows, device, max_prefix=max_prefix),
                "test": _evaluate(model, test_rows, device, max_prefix=max_prefix),
            }
        output["pooled_summarization"] = pooled
    return output
