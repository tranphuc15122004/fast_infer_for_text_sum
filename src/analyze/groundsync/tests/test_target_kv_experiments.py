from __future__ import annotations

import math

import pytest

from src.analyze.groundsync.target_kv_experiments import (
    aggregate_e0_metrics,
    apply_input_length_limit,
    bootstrap_document_mean_ci,
    context_bucket,
    dflash_acceptance_to_draft_tokens,
    flatten_dflash_rounds,
    first_rejection_position,
    prepare_record_metadata,
    representation_parameter_audit,
)
from src.analyze.groundsync.e0_dflash_failure_map import (
    SelectiveHiddenTarget,
    apply_input_cap,
    chunk_spans,
    choose_smoke_input_cap,
    run_inference_safe,
    release_cuda_cache,
    select_raw_rows,
    append_jsonl,
)


def test_context_bucket_has_explicit_model_limit_bucket() -> None:
    assert context_bucket(2048) == "0-2k"
    assert context_bucket(2049) == "2-4k"
    assert context_bucket(4096) == "2-4k"
    assert context_bucket(4097) == "4-8k"
    assert context_bucket(16384) == "8-16k"
    assert context_bucket(32768) == "16-32k"
    assert context_bucket(40961) == ">40k"


def test_prepare_record_metadata_rejects_overlength_without_silent_truncation() -> None:
    accepted = prepare_record_metadata(
        {"id": "a", "dataset": "gov_report", "input_tokens": 40960},
        max_position_embeddings=40960,
    )
    rejected = prepare_record_metadata(
        {"id": "b", "dataset": "gov_report", "input_tokens": 40961},
        max_position_embeddings=40960,
    )
    assert accepted["status"] == "ok"
    assert rejected["status"] == "excluded"
    assert rejected["exclusion_reason"] == "input_exceeds_model_limit"


def test_input_length_limit_marks_t4_exclusions_without_relabeling_bucket() -> None:
    metadata = prepare_record_metadata(
        {"id": "b", "dataset": "gov_report", "input_tokens": 9000},
    )
    limited = apply_input_length_limit(metadata, 8192)
    assert limited["status"] == "excluded"
    assert limited["exclusion_reason"] == "t4_input_cap"
    assert limited["context_bucket"] == "8-16k"


def test_dflash_acceptance_converts_fallback_inclusive_length() -> None:
    assert dflash_acceptance_to_draft_tokens(1, block_size=4) == 0
    assert dflash_acceptance_to_draft_tokens(5, block_size=4) == 4
    assert first_rejection_position(1, block_size=4) == 1
    assert first_rejection_position(3, block_size=4) == 3
    assert first_rejection_position(5, block_size=4) is None


def test_flatten_dflash_rounds_keeps_round_and_bucket_metadata() -> None:
    record = {
        "status": "ok",
        "sample_id": "doc-1",
        "input_tokens": 5000,
        "block_size": 4,
        "acceptance_lengths": [1, 3, 5],
        "exact_match_target_ar": True,
    }
    rows = flatten_dflash_rounds(record)
    assert [row["round_index"] for row in rows] == [0, 1, 2]
    assert [row["accepted_draft_tokens"] for row in rows] == [0, 2, 4]
    assert all(row["context_bucket"] == "4-8k" for row in rows)
    assert rows[0]["first_rejection_rel"] == 1
    assert rows[-1]["first_rejection_rel"] is None
    assert all(row["exact_match_target_ar"] is True for row in rows)


def test_bootstrap_document_ci_weights_documents_equally() -> None:
    rows = [
        {"document_id": "a", "value": 0.0},
        {"document_id": "a", "value": 0.0},
        {"document_id": "b", "value": 1.0},
    ]
    result = bootstrap_document_mean_ci(rows, value_key="value", samples=100, seed=7)
    assert result["document_count"] == 2
    assert result["mean"] == pytest.approx(0.5)
    assert 0.0 <= result["ci_low"] <= result["mean"] <= result["ci_high"] <= 1.0


def test_e0_metrics_reports_survival_and_relative_long_context_drop() -> None:
    rows = []
    for doc_id in ("a", "b", "c", "d"):
        for bucket, accepted in (("0-2k", 8), ("8-16k", 2)):
            rows.append(
                {
                    "document_id": doc_id,
                    "context_bucket": bucket,
                    "block_size": 8,
                    "accepted_draft_tokens": accepted,
                }
            )
    result = aggregate_e0_metrics(rows, candidate_ks=(8,), bootstrap_samples=200)
    long_metrics = result["by_k"]["8"]["by_bucket"]["8-16k"]
    assert long_metrics["survival"]["8"] == pytest.approx(0.0)
    assert result["context_drop"]["8"]["relative_drop"] == pytest.approx(1.0)
    assert result["decision"]["status"] == "PASS"


def test_e0_metrics_is_inconclusive_when_long_natural_bucket_is_absent() -> None:
    rows = []
    for document_id in ("short-a", "short-b", "short-c"):
        for round_index in range(4):
            rows.append(
                {
                    "document_id": document_id,
                    "context_bucket": "4-8k",
                    "block_size": 4,
                    "accepted_draft_tokens": 0,
                    "round_index": round_index,
                }
            )
    result = aggregate_e0_metrics(rows, candidate_ks=(4,), bootstrap_samples=50)
    assert result["decision"] == {
        "status": "INCONCLUSIVE",
        "reason": "insufficient_natural_bucket_coverage",
    }
    assert result["context_drop"]["4"]["status"] == "INCONCLUSIVE"


def test_representation_parameter_audit_requires_equal_trainable_budget() -> None:
    audit = representation_parameter_audit(
        {"hidden": 100, "hidden_sequence": 100, "multi_layer_hidden": 100, "kv": 100}
    )
    assert audit["equal"] is True
    assert audit["max"] == 100
    with pytest.raises(ValueError, match="parameter budget"):
        representation_parameter_audit({"hidden": 100, "kv": 101})


def test_smoke_input_cap_is_applied_to_actual_ids() -> None:
    ids = list(range(10))
    assert apply_input_cap(ids, max_tokens=4) == [0, 1, 2, 3]
    assert apply_input_cap(ids, max_tokens=0) == ids


def test_selective_hidden_target_does_not_request_all_hidden_layers() -> None:
    torch = pytest.importorskip("torch")

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(4)])

        def forward(self, input_ids, **kwargs):
            value = input_ids.float().unsqueeze(-1)
            for layer in self.layers:
                value = layer(value + 1.0)
            return value

    class FakeTarget(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = FakeModel()
            self.config = type("Config", (), {})()
            self.lm_head = torch.nn.Identity()
            self.requested_hidden_states = None

        def forward(self, input_ids, **kwargs):
            self.requested_hidden_states = kwargs.get("output_hidden_states")
            value = self.model(input_ids)
            return type(
                "Output", (), {"hidden_states": None, "last_hidden_state": value}
            )()

    target = FakeTarget()
    wrapped = SelectiveHiddenTarget(target, layer_ids=[1, 3])
    output = wrapped(torch.tensor([[2, 3]]), output_hidden_states=True)
    assert target.requested_hidden_states is False
    assert output.hidden_states[2] is not None
    assert output.hidden_states[4] is not None
    assert output.hidden_states[1] is None


def test_chunk_spans_cover_sequence_without_overlap() -> None:
    assert chunk_spans(10, 4) == [(0, 4), (4, 8), (8, 10)]
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_spans(10, 0)


def test_smoke_cap_is_explicit_and_positive() -> None:
    assert choose_smoke_input_cap(True, 1024) == 1024
    assert choose_smoke_input_cap(False, 1024) == 0
    with pytest.raises(ValueError, match="smoke cap"):
        choose_smoke_input_cap(True, 0)


def test_generation_wrapper_does_not_build_autograd_graph() -> None:
    torch = pytest.importorskip("torch")
    value = run_inference_safe(lambda: torch.ones(1, requires_grad=True) * 2)
    assert value.grad_fn is None


def test_release_cuda_cache_calls_empty_cache_when_available() -> None:
    calls: list[str] = []

    class FakeCuda:
        def empty_cache(self) -> None:
            calls.append("empty")

    class FakeTorch:
        cuda = FakeCuda()

    release_cuda_cache(FakeTorch())
    assert calls == ["empty"]


def test_select_raw_rows_supports_reproducible_nonzero_start() -> None:
    rows = [{"id": str(i)} for i in range(5)]
    assert [row["id"] for row in select_raw_rows(rows, start_index=2, max_samples=2)] == ["2", "3"]
    with pytest.raises(ValueError, match="start_index"):
        select_raw_rows(rows, start_index=-1, max_samples=None)


def test_append_jsonl_flushes_partial_rows_for_resume(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    append_jsonl(path, {"id": 1})
    append_jsonl(path, {"id": 2})
    assert [line for line in path.read_text().splitlines()] == ['{"id": 1}', '{"id": 2}']
