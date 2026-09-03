from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.config import SyncSpecConfig  # noqa: E402
from SyncSpec.controller import RuntimeFeedback  # noqa: E402
from SyncSpec.engine import SyncSpecEngine  # noqa: E402
from SyncSpec.synthetic import SyntheticDrafter, SyntheticTarget  # noqa: E402


def test_cpu_engine_is_greedy_lossless_against_vanilla_ar() -> None:
    target = SyntheticTarget(vocab_size=32, eos_token_id=31)
    drafter = SyntheticDrafter(target)
    cfg = SyncSpecConfig(
        vocab_size=32,
        hidden_size=16,
        top_m=4,
        budget_profiles=((0, 0), (4, 2), (4, 4)),
    )
    engine = SyncSpecEngine(target, drafter, cfg)
    source = torch.tensor([3, 4, 5, 6])
    accelerated = engine.generate(source, max_new_tokens=8, seed=11)
    vanilla = target.generate_greedy(source, max_new_tokens=8)
    assert accelerated.token_ids.tolist() == vanilla.tolist()
    assert accelerated.status == "ok"
    assert accelerated.rounds >= 1
    assert accelerated.committed_tokens == len(vanilla)
    assert accelerated.timing_ms["verify"] >= 0.0
    assert accelerated.runtime_feedback["rounds"] >= 1
    assert 0.0 <= accelerated.runtime_feedback["acceptance_ema"] <= 1.0


def test_engine_honors_zero_max_new_tokens() -> None:
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(vocab_size=16, hidden_size=16, top_m=4),
    )
    result = engine.generate(torch.tensor([1, 2]), max_new_tokens=0)
    assert result.token_ids.numel() == 0
    assert result.rounds == 0


def test_engine_caps_generation_to_target_context_capacity() -> None:
    class ContextLimitedSyntheticTarget(SyntheticTarget):
        def __init__(self, max_context: int):
            super().__init__(vocab_size=32)
            self.max_context = int(max_context)

        def remaining_context_tokens(self, state):
            return max(0, self.max_context - state.source_ids.numel() - len(state.generated))

    target = ContextLimitedSyntheticTarget(max_context=4)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(
            vocab_size=32, hidden_size=16, top_m=4,
            budget_profiles=((0, 0), (4, 2), (4, 4)),
        ),
    )
    result = engine.generate(torch.tensor([1, 2, 3]), max_new_tokens=8)
    assert result.committed_tokens == 1


def test_batch_engine_caps_each_request_to_its_context_capacity() -> None:
    class ContextLimitedSyntheticTarget(SyntheticTarget):
        def __init__(self, max_context: int):
            super().__init__(vocab_size=32)
            self.max_context = int(max_context)

        def remaining_context_tokens(self, state):
            return max(0, self.max_context - state.source_ids.numel() - len(state.generated))

    target = ContextLimitedSyntheticTarget(max_context=4)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(
            vocab_size=32, hidden_size=16, top_m=4,
            budget_profiles=((0, 0), (4, 2), (4, 4)),
        ),
    )
    results = engine.generate_batch(
        [torch.tensor([1, 2, 3]), torch.tensor([1, 2])], max_new_tokens=8,
    )
    assert [result.committed_tokens for result in results] == [1, 2]


def test_engine_uses_effective_batch_for_pre_gate() -> None:
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(
            vocab_size=16, hidden_size=16, top_m=4,
            gate_table={"short:batch2": 0.0},
        ),
    )
    result = engine.generate(torch.tensor([1, 2]), max_new_tokens=2, batch_size=2)
    assert result.batch_size == 2
    assert result.fallback_rounds == 2
    assert all(budget["kd"] == 0 for budget in result.budgets)


def test_engine_preserves_profile_specific_pre_gate_priors() -> None:
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(
            vocab_size=16, hidden_size=16, top_m=4,
            budget_profiles=((0, 0), (8, 4), (16, 8)),
            gate_table={
                "long:batch1:kd8": 0.25,
                "long:batch1:kd16": 0.05,
            },
        ),
    )
    gains = engine._feedback_gains(
        RuntimeFeedback(),
        context_length=4096,
        batch_size=1,
    )
    assert gains == {8: 0.25, 16: 0.05}


def test_engine_falls_back_to_another_profile_when_preferred_kd_lacks_measurement(
    monkeypatch,
) -> None:
    target = SyntheticTarget(vocab_size=32)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(
            vocab_size=32, hidden_size=16, top_m=4,
            budget_profiles=((0, 0), (8, 4), (16, 8)),
            gate_table={
                "short:batch1:kd8": 0.40,
                "short:batch1:kd16": 0.10,
            },
        ),
    )
    monkeypatch.setattr(
        engine, "_can_speculate", lambda kd, batch_size, context_length, max_kv=None: kd == 16,
    )
    result = engine.generate(torch.tensor([1, 2]), max_new_tokens=8)
    assert result.budgets[0]["kd"] == 16


def test_engine_uses_low_acceptance_feedback_to_fall_back_to_ar() -> None:
    class WrongDrafter(SyntheticDrafter):
        def draft(self, state, kd, **kwargs):
            output = super().draft(state, kd, **kwargs)
            # Keep the proposal IDs in-vocabulary but exclude the synthetic
            # target's next token so the first speculative round has zero
            # accepted prefix length.
            output.candidate_ids = torch.zeros_like(output.candidate_ids)
            return output

    target = SyntheticTarget(vocab_size=32)
    engine = SyncSpecEngine(
        target, WrongDrafter(target),
        SyncSpecConfig(
            vocab_size=32, hidden_size=16, top_m=4,
            feedback_alpha=1.0,
            budget_profiles=((0, 0), (4, 2), (4, 4)),
        ),
    )
    result = engine.generate(torch.tensor([10]), max_new_tokens=3)
    assert result.token_ids.numel() == 3
    assert result.runtime_feedback["acceptance_ema"] == 0.0
    assert result.fallback_rounds >= 2


def test_cpu_engine_exposes_stochastic_exact_path() -> None:
    target = SyntheticTarget(vocab_size=16, eos_token_id=15)
    engine = SyncSpecEngine(
        target,
        SyntheticDrafter(target, top_m=4),
        SyncSpecConfig(vocab_size=16, hidden_size=16, top_m=4, predicted_spec_gain=0.2,
                       budget_profiles=((0, 0), (4, 2), (4, 4))),
    )
    result = engine.generate(torch.tensor([1, 2]), max_new_tokens=4, stochastic=True, seed=5)
    assert result.status == "ok"
    assert result.committed_tokens == 4


def test_engine_commits_target_bonus_after_all_accepted_proposals() -> None:
    class AlwaysSurvive(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def survival(self, features):
            return torch.ones(
                features.shape[:-1], dtype=features.dtype, device=features.device,
            ) * self.scale

    target = SyntheticTarget(vocab_size=32)
    engine = SyncSpecEngine(
        target,
        SyntheticDrafter(target, top_m=4),
        SyncSpecConfig(
            vocab_size=32, hidden_size=16, top_m=4,
            budget_profiles=((0, 0), (4, 2), (4, 4)), predicted_spec_gain=0.2,
        ),
        survival_head=AlwaysSurvive(),
    )
    result = engine.generate(torch.tensor([1, 2]), max_new_tokens=5)
    assert result.committed_tokens == 5
    assert result.rounds == 1
    assert result.accepted_lengths == [4]


def test_engine_supports_fixed_verification_window_for_profile_measurement() -> None:
    target = SyntheticTarget(vocab_size=32)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target, top_m=4),
        SyncSpecConfig(
            vocab_size=32, hidden_size=16, top_m=4,
            budget_profiles=((0, 0), (4, 2), (4, 4)), predicted_spec_gain=0.2,
        ),
    )
    result = engine.generate(
        torch.tensor([1, 2]), max_new_tokens=4, force_kv=4, max_rounds=1,
    )
    assert result.rounds == 1
    assert result.budgets == [{"kd": 4, "kv": 4}]
    assert result.committed_tokens == 4


def test_batch_engine_supports_fixed_window_and_single_round_profile_measurement() -> None:
    target = SyntheticTarget(vocab_size=32)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target, top_m=4),
        SyncSpecConfig(
            vocab_size=32, hidden_size=16, top_m=4,
            budget_profiles=((0, 0), (4, 2), (4, 4)), predicted_spec_gain=0.2,
        ),
    )
    results = engine.generate_batch(
        [torch.tensor([1, 2]), torch.tensor([3, 4])],
        max_new_tokens=4, force_kv=4, max_rounds=1,
    )
    assert all(result.rounds == 1 for result in results)
    assert all(result.budgets == [{"kd": 4, "kv": 4}] for result in results)
    assert all(result.committed_tokens == 4 for result in results)


def test_engine_stops_at_eos_inside_an_all_accepted_block() -> None:
    target = SyntheticTarget(vocab_size=8, eos_token_id=3)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target, top_m=4),
        SyncSpecConfig(
            vocab_size=8, hidden_size=16, top_m=4,
            budget_profiles=((0, 0), (4, 2), (4, 4)), predicted_spec_gain=0.2,
        ),
    )
    result = engine.generate(torch.tensor([2]), max_new_tokens=4)
    assert result.token_ids.tolist() == [3]
    assert result.committed_tokens == 1
    assert result.accepted_lengths == [1]


def test_batch_engine_stops_each_request_at_eos_inside_a_block() -> None:
    target = SyntheticTarget(vocab_size=8, eos_token_id=3)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target, top_m=4),
        SyncSpecConfig(
            vocab_size=8, hidden_size=16, top_m=4,
            budget_profiles=((0, 0), (4, 2), (4, 4)), predicted_spec_gain=0.2,
        ),
    )
    results = engine.generate_batch(
        [torch.tensor([2]), torch.tensor([4])], max_new_tokens=4,
    )
    assert results[0].token_ids.tolist() == [3]
    assert results[0].committed_tokens == 1


def test_engine_stops_at_eos_in_ar_fallback() -> None:
    target = SyntheticTarget(vocab_size=8, eos_token_id=3)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target, top_m=4),
        SyncSpecConfig(
            vocab_size=8, hidden_size=16, top_m=4,
            gate_table={"short:batch1": 0.0},
        ),
    )
    result = engine.generate(torch.tensor([2]), max_new_tokens=4)
    assert result.token_ids.tolist() == [3]
    assert result.committed_tokens == 1


def test_engine_ignores_profile_for_different_model_or_checkpoint(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(
        '{"key": {"model": "other", "checkpoint": "other", "precision": "float32", "kd": 4, "kv": 4}, '
        '"measurements_ms": {"verify": {"mean": 0.001}}}',
        encoding="utf-8",
    )
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(vocab_size=16, hidden_size=16, target_model="target",
                       drafter_checkpoint="drafter", runtime_profile=str(profile)),
    )
    costs = engine._profile_costs(4, 4)
    assert costs[1] == 1.0 + 0.12 + 0.08


def test_engine_profile_cost_includes_all_measured_round_components(tmp_path: Path) -> None:
    profile = tmp_path / "profile-round-cost.json"
    profile.write_text(
        json.dumps({
            "key": {
                "model": "target", "checkpoint": "drafter", "precision": "float32",
                "context_bin": "short", "batch_bin": "batch1", "kd": 4, "kv": 2,
            },
            "measurements_ms": {
                "draft": {"mean": 3.0}, "selector": {"mean": 4.0},
                "survival": {"mean": 5.0}, "verify": {"mean": 2.0},
                "scheduler": {"mean": 1.0}, "e2e": {"mean": 99.0},
            },
        }),
        encoding="utf-8",
    )
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(
            vocab_size=16, hidden_size=16, target_model="target",
            drafter_checkpoint="drafter", runtime_profile=str(profile),
        ),
    )
    costs = engine._profile_costs(4, 4, batch_size=1, context_length=64)
    assert costs[2] == 15.0


def test_engine_normalizes_shared_profile_cost_for_batch_controller(tmp_path: Path) -> None:
    profile = tmp_path / "profile-batch-round-cost.json"
    profile.write_text(json.dumps({
        "schema_version": 1,
        "source": "measured",
        "key": {
            "model": "target", "checkpoint": "drafter", "precision": "float32",
            "context_bin": "short", "batch_bin": "batch2", "kd": 4, "kv": 2,
        },
        "measurements_ms": {
            "draft": {"p50": 2.0}, "selector": {"p50": 2.0},
            "survival": {"p50": 2.0}, "verify": {"p50": 2.0},
            "scheduler": {"p50": 2.0},
        },
    }), encoding="utf-8")
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(
            vocab_size=16, hidden_size=16, target_model="target",
            drafter_checkpoint="drafter", runtime_profile=str(profile),
        ),
    )
    costs = engine._profile_costs(4, 4, batch_size=2, context_length=64)
    assert costs[2] == 5.0


def test_engine_falls_back_when_measured_spec_utility_loses_to_ar(tmp_path: Path) -> None:
    profile = tmp_path / "profile-ar-gate.json"
    profile.write_text(
        json.dumps({
            "schema_version": 1, "source": "measured",
            "key": {
                "model": "target", "checkpoint": "drafter", "precision": "float32",
                "context_bin": "short", "batch_bin": "batch1", "kd": 4, "kv": 2,
            },
            "measurements_ms": {
                "draft": {"p50": 4.0}, "selector": {"p50": 2.0},
                "survival": {"p50": 1.0}, "verify": {"p50": 3.0},
                "scheduler": {"p50": 1.0}, "target_ar": {"p50": 1.0},
            },
            "target_ar_tokens": 1,
        }),
        encoding="utf-8",
    )
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(
            vocab_size=16, hidden_size=16, target_model="target",
            drafter_checkpoint="drafter", runtime_profile=str(profile),
            predicted_spec_gain=0.2,
            budget_profiles=((0, 0), (4, 2)),
        ),
    )
    result = engine.generate(torch.tensor([1, 2]), max_new_tokens=2)
    assert result.fallback_rounds >= 1
    assert result.budgets[0] == {"kd": 4, "kv": 0}
    assert result.budgets[1] == {"kd": 0, "kv": 0}
    assert result.runtime_feedback["rounds"] == 1
    assert result.runtime_feedback["acceptance_ema"] == 0.0


def test_batch_engine_uses_measured_ar_gate_per_request(tmp_path: Path) -> None:
    profile = tmp_path / "profile-batch-ar-gate.json"
    profile.write_text(
        json.dumps({
            "schema_version": 1, "source": "measured",
            "key": {
                "model": "target", "checkpoint": "drafter", "precision": "float32",
                "context_bin": "short", "batch_bin": "batch2", "kd": 4, "kv": 2,
            },
            "measurements_ms": {
                "draft": {"p50": 8.0}, "selector": {"p50": 4.0},
                "survival": {"p50": 2.0}, "verify": {"p50": 6.0},
                "scheduler": {"p50": 1.0}, "target_ar": {"p50": 2.0},
            },
            "target_ar_tokens": 1,
        }),
        encoding="utf-8",
    )
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(
            vocab_size=16, hidden_size=16, target_model="target",
            drafter_checkpoint="drafter", runtime_profile=str(profile),
            predicted_spec_gain=0.2,
            budget_profiles=((0, 0), (4, 2)),
        ),
    )
    results = engine.generate_batch(
        [torch.tensor([1, 2]), torch.tensor([3, 4])], max_new_tokens=2,
    )
    assert all(result.budgets[0] == {"kd": 4, "kv": 0} for result in results)
    assert all(result.budgets[1] == {"kd": 0, "kv": 0} for result in results)


def test_engine_ignores_profile_for_different_context_bin(tmp_path: Path) -> None:
    profile = tmp_path / "profile-context.json"
    profile.write_text(
        '{"key": {"model": "target", "checkpoint": "drafter", "precision": "float32", '
        '"context_bin": "long", "batch_bin": "batch1", "kd": 4, "kv": 4}, '
        '"measurements_ms": {"verify": {"mean": 0.001}}}',
        encoding="utf-8",
    )
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(vocab_size=16, hidden_size=16, target_model="target",
                       drafter_checkpoint="drafter", runtime_profile=str(profile)),
    )
    assert not engine._profile_matches(
        json.loads(profile.read_text()), 4, batch_size=1, context_length=64,
    )


def test_engine_ignores_legacy_profile_without_batch_axis(tmp_path: Path) -> None:
    profile = tmp_path / "profile-legacy.json"
    profile.write_text(
        '{"key": {"model": "target", "checkpoint": "drafter", "precision": "float32", '
        '"kd": 4, "kv": 4}, "measurements_ms": {"verify": {"mean": 0.001}}}',
        encoding="utf-8",
    )
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(vocab_size=16, hidden_size=16, target_model="target",
                       drafter_checkpoint="drafter", runtime_profile=str(profile)),
    )
    assert not engine._profile_matches(
        json.loads(profile.read_text()), 4, batch_size=1, context_length=64,
    )


def test_cuda_profile_requirement_falls_back_to_ar_without_matching_measurement(tmp_path: Path) -> None:
    profile = tmp_path / "profile-missing.json"
    profile.write_text(
        '{"key": {"model": "target", "checkpoint": "drafter", "precision": "bfloat16", '
        '"context_bin": "long", "batch_bin": "batch1", "kd": 16, "kv": 8}, '
        '"measurements_ms": {"verify": {"mean": 1.0}}}',
        encoding="utf-8",
    )
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(vocab_size=16, hidden_size=16, target_model="target",
                       drafter_checkpoint="drafter", dtype="bfloat16",
                       runtime_profile=str(profile), require_measured_profile=True),
    )
    assert engine._profile_costs(16, 16, batch_size=1, context_length=64) == {}


def test_cuda_profile_requirement_rejects_unmeasured_profile(tmp_path: Path) -> None:
    profile = tmp_path / "profile-unmeasured.json"
    profile.write_text(
        json.dumps({
            "source": "synthetic",
            "key": {
                "model": "target", "checkpoint": "drafter", "gpu": "B200",
                "precision": "bfloat16", "context_bin": "short",
                "batch_bin": "batch1", "kd": 16, "kv": 8,
            },
            "measurements_ms": {
                "target_ar": {"mean": 2.0}, "verify": {"mean": 1.0},
            },
        }),
        encoding="utf-8",
    )
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(
            vocab_size=16, hidden_size=16, target_model="target",
            drafter_checkpoint="drafter", dtype="bfloat16",
            runtime_profile=str(profile), require_measured_profile=True,
        ),
    )
    assert engine._profile_costs(16, 16, batch_size=1, context_length=64) == {}


def test_cuda_profile_requirement_rejects_unmeasured_kv_candidate(tmp_path: Path) -> None:
    profile = tmp_path / "profile-wrong-kv.json"
    profile.write_text(
        '{"key": {"model": "target", "checkpoint": "drafter", "precision": "bfloat16", '
        '"context_bin": "short", "batch_bin": "batch1", "kd": 16, "kv": 4}, '
        '"measurements_ms": {"verify": {"mean": 1.0}}}',
        encoding="utf-8",
    )
    target = SyntheticTarget(vocab_size=16)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target),
        SyncSpecConfig(
            vocab_size=16, hidden_size=16, target_model="target",
            drafter_checkpoint="drafter", dtype="bfloat16",
            runtime_profile=str(profile), require_measured_profile=True,
            budget_profiles=((0, 0), (16, 8)),
        ),
    )
    assert not engine._can_speculate(16, batch_size=1, context_length=64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/B200 is not available on this host")
def test_cuda_engine_smoke_uses_same_exact_path() -> None:
    target = SyntheticTarget(vocab_size=32, eos_token_id=31, device="cuda")
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target, top_m=4),
        SyncSpecConfig(vocab_size=32, hidden_size=16, top_m=4, device="cuda",
                       predicted_spec_gain=0.2, budget_profiles=((0, 0), (4, 2), (4, 4))),
    )
    result = engine.generate(torch.tensor([1, 2], device="cuda"), max_new_tokens=3)
    assert result.token_ids.is_cuda
    assert result.committed_tokens == 3
