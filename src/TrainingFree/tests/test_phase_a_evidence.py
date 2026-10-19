"""Guard against fabricated layer evidence and lost query dimensions."""

import numpy as np
import pytest
import json

from TrainingFree.context_interventions.selector_evaluation import (
    layer_vector,
    pool_queries,
    protected_topk,
    summarize_rows,
    selector_scores,
)


def test_missing_layer_is_unavailable_instead_of_reusing_middle_layer():
    snapshot = {"target_layer_ids": np.array([0, 18, 35]),
                "target": np.array([[1., 0.], [0., 1.], [.5, .5]])}
    with pytest.raises(ValueError, match="27"):
        layer_vector(snapshot, 27)
    np.testing.assert_array_equal(layer_vector(snapshot, 18), [0., 1.])


def test_max_pooling_requires_query_rows_and_differs_from_mean():
    snapshot = {"target_layer_ids": np.array([18]), "target": np.array([[.7, .3]])}
    with pytest.raises(ValueError, match="query"):
        pool_queries(snapshot, 18, mode="max", scope="accepted")
    snapshot.update(target_query_attention=np.array([[[.9, .1], [.2, .8], [.1, .9]]]),
                    accepted_candidates=np.array(1))
    np.testing.assert_allclose(pool_queries(snapshot, 18, mode="max", scope="accepted"), [.9/1.7, .8/1.7])
    np.testing.assert_allclose(pool_queries(snapshot, 18, mode="mean", scope="accepted"), [.55, .45])


def test_budget_smaller_than_protection_is_reported_infeasible():
    with pytest.raises(ValueError, match="protected"):
        protected_topk(np.arange(10.), 3, sink=2, recent=2)
    assert set(protected_topk(np.arange(10.), 4, sink=2, recent=2)) == {0, 1, 8, 9}


def test_summary_weights_documents_equally_and_counts_rounds_separately():
    rows = [{"sample_id": "a", "coverage": 1.}] * 9 + [{"sample_id": "b", "coverage": 0.}]
    summary = summarize_rows(rows)
    assert summary["documents"] == 2
    assert summary["snapshots"] == 10
    assert summary["mean_coverage"] == .5


def test_parent_control_uses_same_round_anchor_query_as_multi_query_selector():
    snapshot = {"target_layer_ids": np.array([18]), "target": np.array([[.5,.5]]),
        "target_query_attention": np.array([[[.9,.1],[.2,.8]]]),
        "accepted_candidates": np.array(1), "parent_target": np.array([[.95,.05]])}
    scores, _ = selector_scores(snapshot, 2)
    np.testing.assert_allclose(scores["parent_L18"], [.2,.8])


def test_evaluator_keeps_missing_candidates_unavailable_and_selects_from_past(tmp_path):
    from scripts.run_candidate_benchmarks import evaluate
    source = tmp_path / "vectors"
    source.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"target_layers": [0,18,35], "samples": ["doc"]}))
    first = np.array([[.25,.25,.25,.25], [.4,.3,.2,.1], [.1,.2,.3,.4]])
    following = first.copy()
    following[1] = [.1,.2,.3,.4]
    for suffix, step, values in [("round_001",1,first), ("round_002",2,following)]:
        np.savez(source / f"sample_00_context_4_{suffix}.npz", target=values, round=step, prefix_length=4)
    report = evaluate(source, tmp_path/"report", budgets=(2,), chunk_sizes=(1,), all_rounds=True, sink=0, recent=0)
    assert report["unique_documents"] == 1
    assert report["unique_snapshots"] == 2
    assert "C3_L27" not in {r["candidate"] for r in report["results"]}
    assert "max_L18_accepted" not in {r["method"] for r in report["results"]}
    row = next(r for r in report["results"] if r["method"] == "mean_L18_legacy"
               and r["candidate"] == "C1_L18" and r["evaluation_scope"] == "next_round")
    assert row["mean_coverage"] == pytest.approx(.3)
