from __future__ import annotations

import pytest

from src.analyze.dflash_residual.prefix_gap import (
    analyze_matched_context,
    analyze_prefix_oracle,
    analyze_rank_ambiguity,
    conditional_prefix_survival,
    independent_prefix_survival,
    joint_prefix_survival,
    prefix_oracle_length,
)
from src.analyze.dflash_residual.schema import normalize_trace_row


def _row(
    *,
    sample_id: str,
    document_id: str,
    position: int,
    target: int,
    candidates: list[int],
    logits: list[float] | None = None,
    context_cap: int = 1024,
    dataset: str = "cnn_dailymail",
    accepted: int = 1,
) -> dict:
    return normalize_trace_row({
        "status": "ok",
        "run_id": "run",
        "sample_id": sample_id,
        "document_id": document_id,
        "dataset": dataset,
        "context_length": context_cap,
        "context_cap": context_cap,
        "round_index": 0,
        "draft_position": position,
        "max_depth": 3,
        "target_token_id": target,
        "candidate_token_ids": candidates,
        "candidate_logits": logits or [3.0, 2.0, 1.0][:len(candidates)],
        "dflash_selected_token_id": candidates[0],
        "accepted_draft_len": accepted,
        "target_token_source": "verifier_posterior",
    })


def _three_pattern_rows() -> list[dict]:
    # For K=1, block hit patterns are 110, 101, 011.
    patterns = ((1, 1, 0), (1, 0, 1), (0, 1, 1))
    rows: list[dict] = []
    for block_index, pattern in enumerate(patterns):
        for position, hit in enumerate(pattern, start=1):
            target = 100 + position
            candidates = [target, 900] if hit else [800, target]
            rows.append(_row(
                sample_id=f"s{block_index}",
                document_id=f"d{block_index}",
                position=position,
                target=target,
                candidates=candidates,
            ))
    return rows


def test_joint_and_independent_prefix_survival_are_distinct() -> None:
    rows = _three_pattern_rows()
    joint = joint_prefix_survival(rows, k=1)
    independent = independent_prefix_survival(rows, k=1)
    conditional = conditional_prefix_survival(rows, k=1)

    assert joint == {1: pytest.approx(2 / 3), 2: pytest.approx(1 / 3), 3: 0.0}
    assert independent[2] == pytest.approx(4 / 9)
    assert joint[2] < independent[2]
    assert conditional[1] == pytest.approx(2 / 3)
    assert conditional[2] == pytest.approx(0.5)
    assert conditional[3] == pytest.approx(0.0)


def test_prefix_oracle_length_and_mat_are_longest_joint_prefixes() -> None:
    rows = _three_pattern_rows()
    assert prefix_oracle_length(rows[:3], k=1) == 2
    result = analyze_prefix_oracle(rows, k_values=(1, 2), min_documents=2)
    metrics = result["groups"]["cnn_dm|0-2k"]["k_values"]
    assert metrics["1"]["mat_oracle"] == pytest.approx(1.0)
    assert metrics["2"]["mat_oracle"] == pytest.approx(3.0)
    assert metrics["2"]["prefix_gap"]["3"] == pytest.approx(0.0)
    assert result["groups"]["cnn_dm|0-2k"]["oracle_ratio_k16"] is None


def test_prefix_oracle_is_tie_aware_at_topk_boundary() -> None:
    row = _row(
        sample_id="tie",
        document_id="tie",
        position=1,
        target=10,
        candidates=[20, 10],
        logits=[5.0, 5.0],
    )
    row["dflash_selected_token_id"] = 10
    assert prefix_oracle_length([row], k=1) == 1


def test_matched_context_filters_caps_and_bootstrap_is_deterministic() -> None:
    rows: list[dict] = []
    for dataset, prefix in (("canonical", "c"), ("cnn_dailymail", "s")):
        for index in range(3):
            rows.append(_row(
                sample_id=f"{prefix}{index}",
                document_id=f"{prefix}{index}",
                position=1,
                target=10,
                candidates=[10, 9] if dataset == "canonical" else [9, 10],
                dataset=dataset,
                context_cap=1024,
            ))
            rows.append(_row(
                sample_id=f"{prefix}{index}",
                document_id=f"{prefix}{index}",
                position=1,
                target=10,
                candidates=[9, 10],
                dataset=dataset,
                context_cap=2048,
            ))

    first = analyze_matched_context(rows, context_cap=1024, bootstrap_samples=30, seed=7, min_documents=2)
    second = analyze_matched_context(rows, context_cap=1024, bootstrap_samples=30, seed=7, min_documents=2)
    assert first == second
    assert first["rows"] == 6
    assert first["context_cap"] == 1024
    assert first["h1_gate"]["canonical_documents"] == 3
    assert first["h1_gate"]["summarization_documents"] == 3


def test_rank_ambiguity_excludes_top16_misses_from_rank_conditioned_metrics() -> None:
    rows = [
        _row(sample_id="s1", document_id="d1", position=1, target=10, candidates=[11, 10, 9], logits=[4.0, 3.0, 1.0]),
        _row(sample_id="s2", document_id="d2", position=1, target=10, candidates=[11, 12, 13], logits=[4.0, 2.0, 1.0]),
    ]
    result = analyze_rank_ambiguity(rows)
    metrics = result["regimes"]["cnn_dm"]
    assert metrics["rows"] == 2
    assert metrics["top16_miss_rows"] == 1
    assert metrics["rank_conditioned_rows"] == 1
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["mean_target_rank"] == pytest.approx(2.0)
    assert metrics["mean_target_logit_deficit"] == pytest.approx(1.0)
    assert metrics["mean_top1_top2_margin"] == pytest.approx(1.0)
