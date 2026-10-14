from __future__ import annotations

from types import SimpleNamespace

import pytest

import TrainingFree.context_interventions.execution as execution
from TrainingFree.context_interventions.execution import _action_row, _previous_dflash_evidence
from TrainingFree.context_interventions.replay import ReferencePrefix


def test_action_row_accepts_and_records_ranking_origin_round() -> None:
    prefix = ReferencePrefix(
        round_index=4,
        processed_output_tokens=48,
        pending_anchor=123,
        remaining_output_tokens=976,
        prefix_hash="prefix-hash",
        previous_dense_round_index=2,
    )

    row = _action_row(
        experiment="A2_selector",
        phase="pilot",
        sample={
            "id": "sample-1",
            "dataset": "gov_report",
            "_cad_source_group_id": "group-1",
        },
        prefix=prefix,
        label="previous_dflash_B1024",
        baseline_accepted=8,
        outcome={"status": "ok", "accepted_candidates": 7},
        nominal_budget=1024,
        ranking_origin_round=2,
    )

    assert row["ranking_origin_round"] == 2


def test_action_row_keeps_ranking_origin_from_outcome_when_not_overridden() -> None:
    prefix = ReferencePrefix(
        round_index=4,
        processed_output_tokens=48,
        pending_anchor=123,
        remaining_output_tokens=976,
        prefix_hash="prefix-hash",
        previous_dense_round_index=2,
    )

    row = _action_row(
        experiment="A2_selector",
        phase="pilot",
        sample={
            "id": "sample-1",
            "dataset": "gov_report",
            "_cad_source_group_id": "group-1",
        },
        prefix=prefix,
        label="previous_dflash_B1024",
        baseline_accepted=8,
        outcome={"status": "ok", "accepted_candidates": 7, "ranking_origin_round": 2},
        nominal_budget=1024,
    )

    assert row["ranking_origin_round"] == 2


def test_previous_dflash_evidence_keeps_attention_scores_on_acceptance_mismatch() -> None:
    scores = {0: {0: 0.25, 1: 0.75}, 1: {0: 0.5, 1: 0.5}}

    usable, diagnostic = _previous_dflash_evidence(
        {"status": "ok", "accepted_candidates": 0, "attention_scores_by_layer": scores},
        reference_accepted=1,
        layer_count=2,
        round_index=98,
    )

    assert usable == scores
    assert diagnostic == {
        "round_index": 98,
        "status": "acceptance_mismatch",
        "reference_accepted_candidates": 1,
        "replay_accepted_candidates": 0,
        "acceptance_matches": False,
        "missing_attention_layers": [],
        "replay_status": "ok",
    }


def test_previous_dflash_evidence_marks_missing_attention_unavailable() -> None:
    usable, diagnostic = _previous_dflash_evidence(
        {"status": "ok", "accepted_candidates": 1, "attention_scores_by_layer": {0: {0: 0.5}}},
        reference_accepted=1,
        layer_count=2,
        round_index=98,
    )

    assert usable is None
    assert diagnostic["status"] == "attention_unavailable"
    assert diagnostic["missing_attention_layers"] == [1]


def test_previous_dflash_rebuild_exception_becomes_local_failure_evidence(monkeypatch) -> None:
    def fail_rebuild(*args, **kwargs):
        raise RuntimeError("synthetic prior prefix failure")

    monkeypatch.setattr(execution, "rebuild_replay_state", fail_rebuild)
    prefix = ReferencePrefix(2, 16, 123, 1008, "previous-prefix", None)

    outcome = execution._replay_previous_dflash_round(
        {}, None, [], prefix, None, max_new_tokens=1024,
    )

    assert outcome["status"] == "error"
    assert outcome["reason"] == "RuntimeError:synthetic prior prefix failure"


def test_a2_continues_and_records_previous_acceptance_mismatch(monkeypatch) -> None:
    scores = {layer: {position: float(position) for position in range(8)} for layer in range(2)}
    state = SimpleNamespace(
        bank=SimpleNamespace(length=8),
        target_scores={18: {position: float(position) for position in range(8)}},
    )
    runtime = {"draft": SimpleNamespace(layers=[object(), object()])}
    sample = {"id": "sample-1", "dataset": "gov_report", "_cad_source_group_id": "group-1"}
    prefix = ReferencePrefix(4, 48, 123, 976, "current-prefix", 2)
    previous_prefix = ReferencePrefix(2, 16, 123, 1008, "previous-prefix", None)

    monkeypatch.setattr(execution, "global_protected_positions", lambda *args, **kwargs: ())
    monkeypatch.setattr(execution, "a2_matrix", lambda phase: {
        "budgets": (4,), "selectors": ("recency", "previous_dflash"), "random_seeds": (17,),
    })
    monkeypatch.setattr(execution, "evaluate_replay_action", lambda runtime, ids, state, layout, masks, **kwargs: {
        "status": "ok", "accepted_candidates": 1,
        "actual_context_tokens_by_layer": [len(mask) for mask in masks],
    })

    rows = execution.run_a2_prefix(
        runtime, sample, None, SimpleNamespace(prompt_tokens=4), prefix, [], phase="pilot", max_new_tokens=1024,
        state=state,
        full_outcome={"status": "ok", "accepted_candidates": 1, "attention_scores_by_layer": scores},
        replay_cache={2: {"status": "ok", "accepted_candidates": 0, "attention_scores_by_layer": scores}},
        previous_prefix=previous_prefix,
        reference_accepted=1,
        previous_reference_accepted=1,
    )

    recency = next(row for row in rows if row["selector_id"] == "recency_B4")
    previous = next(row for row in rows if row["selector_id"] == "previous_dflash_B4")
    assert recency["status"] == "ok"
    assert previous["status"] == "ok"
    assert previous["previous_replay_check"]["status"] == "acceptance_mismatch"


def test_a3_continues_and_records_previous_acceptance_mismatch(monkeypatch) -> None:
    scores = {layer: {position: float(position) for position in range(4096)} for layer in range(2)}
    state = SimpleNamespace(bank=SimpleNamespace(length=4096))
    runtime = {"draft": SimpleNamespace(layers=[object(), object()])}
    sample = {"id": "sample-1", "dataset": "gov_report", "_cad_source_group_id": "group-1"}
    prefix = ReferencePrefix(4, 48, 123, 976, "current-prefix", 2)
    previous_prefix = ReferencePrefix(2, 16, 123, 1008, "previous-prefix", None)

    monkeypatch.setattr(execution, "global_protected_positions", lambda *args, **kwargs: ())
    monkeypatch.setattr(execution, "evaluate_replay_action", lambda runtime, ids, state, layout, masks, **kwargs: {
        "status": "ok", "accepted_candidates": 1,
        "actual_context_tokens_by_layer": [len(mask) for mask in masks],
    })

    rows = execution.run_a3_prefix(
        runtime, sample, None, SimpleNamespace(prompt_tokens=4), prefix, [], phase="pilot",
        selector="previous_dflash", max_new_tokens=1024, state=state,
        full_outcome={"status": "ok", "accepted_candidates": 1},
        replay_cache={2: {"status": "ok", "accepted_candidates": 0, "attention_scores_by_layer": scores}},
        previous_prefix=previous_prefix, reference_accepted=1, previous_reference_accepted=1,
    )

    assert rows
    assert all(row["status"] == "ok" for row in rows)
    assert all(row["previous_replay_check"]["status"] == "acceptance_mismatch" for row in rows)


@pytest.mark.parametrize("family", ("A2", "A3"))
@pytest.mark.parametrize("transient", (False, True))
def test_selector_cell_failure_is_retried_without_losing_other_budgets(monkeypatch, family, transient):
    state = SimpleNamespace(bank=SimpleNamespace(length=4096))
    runtime = {"draft": SimpleNamespace(layers=[object(), object()])}
    sample = {"id": "sample-1", "dataset": "gov_report", "_cad_source_group_id": "group-1"}
    prefix = ReferencePrefix(4, 48, 123, 976, "current-prefix", None)
    monkeypatch.setattr(execution, "global_protected_positions", lambda *args, **kwargs: ())
    monkeypatch.setattr(execution, "a2_matrix", lambda phase: {
        "budgets": (1024, 2048), "selectors": ("recency",), "random_seeds": (17,),
    })
    attempts = []

    def evaluate(*args, **kwargs):
        budget = kwargs["requested_budget"]
        attempts.append(budget)
        if budget == 1024 and (not transient or attempts.count(1024) == 1):
            raise RuntimeError("temporary cell computation failure")
        return {"status": "ok", "accepted_candidates": 1}

    monkeypatch.setattr(execution, "evaluate_replay_action", evaluate)
    kwargs = dict(phase="pilot", max_new_tokens=1024, state=state,
                  full_outcome={"status": "ok", "accepted_candidates": 1,
                                "attention_scores_by_layer": {0: {0: 1.0}, 1: {0: 1.0}}})
    if family == "A2":
        rows = execution.run_a2_prefix(runtime, sample, None, SimpleNamespace(prompt_tokens=4), prefix, [], **kwargs)
    else:
        rows = execution.run_a3_prefix(runtime, sample, None, SimpleNamespace(prompt_tokens=4), prefix, [],
                                       selector="recency", **kwargs)
    first = next(row for row in rows if row["nominal_budget"] == 1024)
    second = next(row for row in rows if row["nominal_budget"] == 2048)
    assert first["status"] == ("ok" if transient else "error")
    assert first["L_action"] == (1 if transient else None)
    assert second["status"] == "ok"
    assert attempts == [1024, 1024, 2048]


def test_evaluate_mask_retries_local_failure(monkeypatch):
    prefix = ReferencePrefix(4, 48, 123, 976, "prefix", None)
    attempts = []

    def evaluate(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("temporary cell failure")
        return {"status": "ok", "accepted_candidates": 2}

    monkeypatch.setattr(execution, "evaluate_replay_action", evaluate)
    row = execution._evaluate_mask(
        {}, None, SimpleNamespace(bank=SimpleNamespace(length=4)), None,
        {"id": "sample-1", "dataset": "gov_report", "_cad_source_group_id": "group-1"}, prefix,
        phase="pilot", experiment="A1_region", label="source_only", nominal_budget=4,
        masks=[(0, 1, 2, 3)], baseline_accepted=1,
    )
    assert row["status"] == "ok"
    assert row["L_action"] == 2
    assert len(attempts) == 2


def test_replay_cell_cannot_continue_after_bank_mutation(monkeypatch):
    from TrainingFree.context_interventions.resilience import CampaignIntegrityError

    state = SimpleNamespace(bank=SimpleNamespace(length=4))

    def mutate_then_fail(*args, **kwargs):
        state.bank.length = 5
        raise RuntimeError("temporary computation failure")

    monkeypatch.setattr(execution, "evaluate_replay_action", mutate_then_fail)
    with pytest.raises(CampaignIntegrityError, match="draft-bank"):
        execution._evaluate_mask(
            {}, None, state, None,
            {"id": "sample-1", "dataset": "gov_report", "_cad_source_group_id": "group-1"},
            ReferencePrefix(4, 48, 123, 976, "prefix", None),
            phase="pilot", experiment="A1_region", label="source_only", nominal_budget=4,
            masks=[(0, 1, 2, 3)], baseline_accepted=1,
        )
