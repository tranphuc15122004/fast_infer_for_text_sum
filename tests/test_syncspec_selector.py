from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.controller import (  # noqa: E402
    PostDraftController,
    PreDraftGate,
    RuntimeFeedback,
    fit_empirical_gate_table,
)
from SyncSpec.config import BudgetProfile  # noqa: E402
from SyncSpec.evidence import SourceNgramIndex  # noqa: E402
from SyncSpec.selector import SourceCoherentSelector  # noqa: E402
from SyncSpec.survival import SurvivalHead, survival_from_hazard  # noqa: E402


def test_selector_returns_normalized_q_only_on_top_m_candidates() -> None:
    torch.manual_seed(7)
    selector = SourceCoherentSelector(hidden_size=8, rank=4, ngram_dim=6)
    index = SourceNgramIndex([1, 2, 3, 4, 5, 4], min_n=2, max_n=6)
    ids = torch.tensor([[4, 9, 5], [6, 7, 8]])
    logits = torch.tensor([[0.2, 0.1, -0.4], [0.0, 0.5, 0.2]])
    hidden = torch.randn(2, 8)
    output = selector.select(hidden, ids, logits, [1, 2, 3], index)
    assert output.token_ids.shape == (2,)
    assert output.q.shape == (2, 3)
    assert torch.allclose(output.q.sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.isfinite(output.q).all()
    assert torch.equal(output.candidate_ids, ids)


def test_selector_stochastic_mode_samples_from_q() -> None:
    selector = SourceCoherentSelector(hidden_size=4, rank=2, ngram_dim=6)
    for parameter in selector.parameters():
        parameter.data.zero_()
    ids = torch.tensor([[10, 11]])
    logits = torch.zeros((1, 2))
    hidden = torch.zeros((1, 4))
    generator = torch.Generator().manual_seed(0)
    expected_generator = torch.Generator().manual_seed(0)
    expected_index = torch.multinomial(
        torch.tensor([0.5, 0.5]), 1, generator=expected_generator,
    ).item()
    output = selector.select(
        hidden, ids, logits, [], SourceNgramIndex([1, 2]),
        stochastic=True, generator=generator,
    )
    assert output.token_ids.item() == ids[0, expected_index].item()


def test_selector_has_learned_predecessor_and_successor_token_embeddings() -> None:
    selector = SourceCoherentSelector(hidden_size=4, rank=2, ngram_dim=6, vocab_size=32)
    assert selector.predecessor_embedding.num_embeddings == 32
    assert selector.successor_embedding.num_embeddings == 32
    assert selector.predecessor_embedding.embedding_dim == 2
    assert selector.successor_embedding.embedding_dim == 2


def test_selector_supports_teacher_forced_and_self_conditioned_history() -> None:
    class RecordingIndex(SourceNgramIndex):
        def __init__(self):
            super().__init__([1, 2, 3, 4])
            self.seen: list[tuple[int, ...]] = []

        def features(self, history, candidate_ids):
            self.seen.append(tuple(history))
            return super().features(history, candidate_ids)

    selector = SourceCoherentSelector(hidden_size=4, rank=2, ngram_dim=6, vocab_size=16)
    for parameter in selector.parameters():
        parameter.data.zero_()
    hidden = torch.zeros(2, 4)
    ids = torch.tensor([[2, 3], [4, 5]])
    logits = torch.zeros(2, 2)
    targets = torch.tensor([3, 4])

    teacher_index = RecordingIndex()
    selector.select(
        hidden, ids, logits, [1], teacher_index,
        target_ids=targets, teacher_forcing=1.0,
    )
    assert teacher_index.seen[1][-1] == 3

    self_index = RecordingIndex()
    selector.select(
        hidden, ids, logits, [1], self_index,
        target_ids=targets, teacher_forcing=0.0,
    )
    assert self_index.seen[1][-1] == 2


def test_survival_is_monotonic_and_controller_respects_kd() -> None:
    hazard = torch.tensor([0.1, 0.2, 0.3])
    survival = survival_from_hazard(hazard)
    assert torch.all(survival[:-1] >= survival[1:])
    assert torch.all((survival >= 0) & (survival <= 1))

    head = SurvivalHead(feature_size=5, hidden_size=8)
    assert head(torch.randn(3, 5)).shape == (3,)

    gate = PreDraftGate(epsilon=0.03)
    assert gate.choose(context_length=100, batch_size=1, predicted_gain=0.0).kd == 0
    assert gate.choose(context_length=100, batch_size=1, predicted_gain=0.5).kd > 0

    controller = PostDraftController()
    choice = controller.choose(
        kd=8,
        survival=torch.tensor([0.95, 0.8, 0.5, 0.2]),
        costs={1: 1.0, 2: 1.1, 4: 1.4, 8: 2.5},
    )
    assert choice.kd == 8
    assert choice.kv in {1, 2, 4, 8}


def test_post_draft_controller_falls_back_when_spec_utility_loses_to_ar() -> None:
    choice = PostDraftController().choose(
        kd=4,
        survival=torch.tensor([0.1, 0.0]),
        costs={2: 10.0},
        profiles=(BudgetProfile(0, 0), BudgetProfile(4, 2)),
        ar_cost=1.0,
        ar_margin=0.03,
    )
    assert choice.kv == 0


def test_pre_gate_can_read_calibrated_context_batch_table() -> None:
    gate = PreDraftGate(
        epsilon=0.03,
        gain_table={"short:batch1": 0.0, "long:batch1": 0.4},
    )
    assert gate.choose(64, 1).kd == 0
    assert gate.choose(4096, 1).kd > 0


def test_pre_gate_selects_best_profile_specific_kd() -> None:
    gate = PreDraftGate(
        epsilon=0.03,
        profiles=(BudgetProfile(0, 0), BudgetProfile(8, 4), BudgetProfile(16, 8)),
        gain_table={
            "long:batch1:kd8": 0.25,
            "long:batch1:kd16": 0.05,
        },
    )
    choice = gate.choose(4096, 1)
    assert choice.kd == 8
    assert choice.reason == "predicted_utility_gain"


def test_pre_gate_does_not_enable_uncalibrated_kd_when_profile_table_exists() -> None:
    gate = PreDraftGate(
        epsilon=0.03,
        profiles=(BudgetProfile(0, 0), BudgetProfile(8, 4), BudgetProfile(16, 8)),
        default_gain=0.9,
        gain_table={"long:batch1:kd8": 0.25},
    )
    assert gate.choose(4096, 1).kd == 8


def test_pre_gate_can_restrict_candidates_to_profiles_with_measured_costs() -> None:
    gate = PreDraftGate(
        epsilon=0.03,
        profiles=(BudgetProfile(0, 0), BudgetProfile(8, 4), BudgetProfile(16, 8)),
        gain_table={
            "long:batch1:kd8": 0.25,
            "long:batch1:kd16": 0.40,
        },
    )
    choice = gate.choose(4096, 1, allowed_kds={8})
    assert choice.kd == 8


def test_pre_gate_keeps_profile_specific_calibration_conservative_after_filtering() -> None:
    gate = PreDraftGate(
        epsilon=0.03,
        profiles=(BudgetProfile(0, 0), BudgetProfile(8, 4), BudgetProfile(16, 8)),
        default_gain=0.9,
        gain_table={"long:batch1:kd8": 0.25},
    )
    assert gate.choose(4096, 1, allowed_kds={16}).kd == 0


def test_empirical_gate_table_aggregates_realized_gain_by_context_and_batch() -> None:
    table = fit_empirical_gate_table([
        {"input_tokens": 64, "batch_size": 1, "realized_gain": 0.0},
        {"input_tokens": 64, "batch_size": 1, "realized_gain": 0.4},
        {"input_tokens": 2048, "batch_size": 2, "throughput_tok_s": 140.0,
         "ar_throughput_tok_s": 100.0},
    ])
    assert table["short:batch1"] == 0.2
    assert abs(table["long:batch2"] - 0.4) < 1e-9


def test_empirical_gate_table_keeps_kd_axis_when_trace_has_profile() -> None:
    table = fit_empirical_gate_table([
        {"input_tokens": 2048, "batch_size": 1, "kd": 8, "realized_gain": 0.25},
        {"input_tokens": 2048, "batch_size": 1, "kd": 16, "realized_gain": 0.05},
    ])
    assert table == {
        "long:batch1:kd16": 0.05,
        "long:batch1:kd8": 0.25,
    }


def test_empirical_gate_table_reads_single_kd_from_engine_budget_trace() -> None:
    table = fit_empirical_gate_table([
        {
            "input_tokens": 2048,
            "batch_size": 1,
            "budgets": [{"kd": 8, "kv": 4}, {"kd": 8, "kv": 4}],
            "realized_gain": 0.25,
        },
    ])
    assert table == {"long:batch1:kd8": 0.25}


def test_runtime_feedback_tracks_acceptance_and_component_latency_ema() -> None:
    feedback = RuntimeFeedback(alpha=0.5)
    feedback.update(
        accepted_length=2, proposed_length=4,
        timings_ms={"draft": 4.0, "selector": 2.0, "survival": 1.0, "verify": 8.0},
    )
    assert feedback.rounds == 1
    assert feedback.acceptance_ema == 0.5
    assert feedback.accepted_length_ema == 2.0
    assert feedback.draft_latency_ema_ms == 4.0
    assert feedback.adjusted_gain(0.4) == 0.2

    feedback.update(
        accepted_length=4, proposed_length=4,
        timings_ms={"draft": 6.0, "selector": 4.0, "survival": 2.0, "verify": 10.0},
    )
    assert feedback.rounds == 2
    assert feedback.acceptance_ema == 0.75
    assert feedback.draft_latency_ema_ms == 5.0
    payload = feedback.to_dict()
    assert payload["rounds"] == 2
    assert payload["last_accepted_length"] == 4
