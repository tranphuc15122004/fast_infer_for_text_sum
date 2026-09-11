from __future__ import annotations

import numpy as np

from src.analyze.safe_budget.e27c_residual_cover import (
    ACTION_ORDER,
    _action_budget,
    _selection_features,
    _select_priority_from_embeddings,
)


class _Selector:
    separator = "\n"
    allow_partial_fallback = True

    @staticmethod
    def count_tokens(text: str) -> int:
        return len(text)

    @staticmethod
    def _document_centroid(embeddings, token_counts):
        del token_counts
        centroid = embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm else centroid


class _Unit:
    def __init__(self, text: str, token_count: int):
        self.text = text
        self.token_count = token_count


def test_action_budget_is_deterministic_and_uses_rounding_rule():
    assert ACTION_ORDER == ("full", "ratio_0.25", "ratio_0.4", "ratio_0.55", "ratio_0.7", "ratio_0.85")
    assert _action_budget(101, "ratio_0.25") == 25
    assert _action_budget(101, "ratio_0.4") == 40
    assert _action_budget(101, "full") == 101


def test_mmr_priority_respects_budget_and_deterministic_tie_break():
    selector = _Selector()
    embeddings = np.eye(3, dtype=np.float32)
    priority = _select_priority_from_embeddings(
        selector,
        ["a", "b", "c"],
        [1, 1, 1],
        embeddings,
        3,
        mmr_lambda=0.7,
    )
    # With equal relevance and no redundancy, earlier sentence wins ties.
    assert priority == [0, 1]


def test_residual_cover_is_zero_when_every_sentence_is_selected():
    sentences = [_Unit("a", 1), _Unit("b", 1)]
    embeddings = np.eye(2, dtype=np.float32)
    features = _selection_features(
        text="a. b.",
        source_tokens=2,
        selected_indices=[0, 1],
        sentences=sentences,
        embeddings=embeddings,
        previous_residual=None,
        base_features={"source_tokens": 2.0},
    )
    assert features["residual_cover"] == 0.0
    assert features["facility_coverage"] == 1.0
    assert features["marginal_residual_gain"] == 0.0


def test_residual_gain_is_drop_from_previous_action():
    sentences = [_Unit("a", 1), _Unit("b", 1)]
    embeddings = np.eye(2, dtype=np.float32)
    features = _selection_features(
        text="a. b.",
        source_tokens=4,
        selected_indices=[0, 1],
        sentences=sentences,
        embeddings=embeddings,
        previous_residual=0.75,
        base_features={"source_tokens": 4.0},
    )
    assert features["marginal_residual_gain"] == 0.75
    assert all(np.isfinite(value) for value in features.values())
