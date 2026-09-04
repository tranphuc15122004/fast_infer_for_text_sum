from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.config import SyncSpecConfig  # noqa: E402
from SyncSpec.engine import SyncSpecEngine  # noqa: E402
from SyncSpec.model import SyncSpecDrafter, SyncSpecDrafterConfig, top_m_candidates  # noqa: E402
from SyncSpec.synthetic import SyntheticTarget  # noqa: E402
from SyncSpec.synthetic import SyntheticDrafter  # noqa: E402
from SyncSpec.evidence import SourceNgramIndex  # noqa: E402
from SyncSpec.selector import SourceCoherentSelector  # noqa: E402
from SyncSpec.survival import SurvivalHead  # noqa: E402
from SyncSpec.training import (  # noqa: E402
    SyncSpecTrainer,
    TrajectoryCache,
    TrajectoryRecord,
    _target_anchor_batch,
    anchor_position_offsets,
    build_stage1_batch,
)
from SyncSpec.trajectory import TargetTrajectoryBuilder  # noqa: E402


def test_stage0_builder_generates_target_owned_trajectories(tmp_path: Path) -> None:
    target = SyntheticTarget(vocab_size=24)
    builder = TargetTrajectoryBuilder(target, seed=9)
    records = builder.build_records(
        [("doc-a", torch.tensor([1, 2, 3])), ("doc-b", torch.tensor([4, 5]))],
        max_new_tokens=5,
    )
    assert [r.sample_id for r in records] == ["doc-a", "doc-b"]
    assert all(len(r.target_ids) == 5 for r in records)
    assert records[0].metadata["target_generated"] is True
    assert records[0].metadata["source_boundaries"] == [{"start": 0, "end": 3, "kind": "prompt"}]
    assert records[0].metadata["decoding"]["strategy"] == "greedy"
    assert records[0].anchors
    assert records[0].metadata["anchor_position_offsets"] == [
        records[0].metadata["context_length"] - 1 + anchor for anchor in records[0].anchors
    ]


def test_stage0_builder_caps_target_generation_to_context_headroom() -> None:
    class ContextLimitedTarget(SyntheticTarget):
        def __init__(self):
            super().__init__(vocab_size=32)
            self.max_context = 4

        def remaining_context_tokens(self, state):
            return max(0, self.max_context - state.source_ids.numel() - len(state.generated))

    records = TargetTrajectoryBuilder(ContextLimitedTarget()).build_records(
        [("limited", torch.tensor([1, 2, 3]))], max_new_tokens=8,
    )
    assert len(records[0].target_ids) == 1


def test_stage0_can_cache_target_anchor_features_when_requested() -> None:
    class FeatureTarget(SyntheticTarget):
        def prefill(self, source_ids):
            state = super().prefill(source_ids)
            state.anchor_hidden = torch.ones(6)
            return state

        def commit(self, state, result):
            super().commit(state, result)
            state.anchor_hidden = torch.full((6,), float(len(state.generated)))

    target = FeatureTarget(vocab_size=24)
    record = TargetTrajectoryBuilder(target, seed=1).build_record(
        "with-features", torch.tensor([1, 2]), max_new_tokens=3, include_target_features=True
    )
    assert record.target_features is not None
    assert len(record.target_features) == 3


def test_stage0_stores_only_selected_anchor_features() -> None:
    class FeatureTarget(SyntheticTarget):
        def prefill(self, source_ids):
            state = super().prefill(source_ids)
            state.anchor_hidden = torch.ones(6)
            return state

        def commit(self, state, result):
            super().commit(state, result)
            state.anchor_hidden = torch.full((6,), float(len(state.generated)))

    record = TargetTrajectoryBuilder(FeatureTarget(vocab_size=24), seed=7).build_record(
        "anchor-only", torch.tensor([1, 2]), max_new_tokens=6,
        include_target_features=True,
    )
    assert record.target_features is not None
    assert len(record.target_features) == len(record.anchors)
    assert record.metadata["target_feature_positions"] == record.anchors
    selected = _target_anchor_batch(
        [record], [record.anchors[-1]], hidden_size=6, device="cpu",
    )
    assert selected is not None
    assert selected[0, 0].item() == record.target_features[-1][0]


def test_stage0_anchor_token_and_recent_hidden_match_selected_target_states(
    tmp_path: Path,
) -> None:
    class DFlashStateTarget(SyntheticTarget):
        def prefill(self, source_ids):
            state = super().prefill(source_ids)
            state.anchor_hidden = torch.tensor([30.0, 0.0])
            state.recent_hidden = torch.tensor([[20.0, 0.0], [30.0, 0.0]])
            return state

        def commit(self, state, result):
            super().commit(state, result)
            token = int(result.committed_ids[-1].item())
            state.anchor_hidden = torch.tensor([float(token), float(len(state.generated))])
            state.recent_hidden = torch.cat(
                [state.recent_hidden, state.anchor_hidden.unsqueeze(0)], dim=0,
            )[-3:]

    record = TargetTrajectoryBuilder(
        DFlashStateTarget(vocab_size=32), seed=3, num_anchors=8,
    ).build_record(
        "dflash-state", torch.tensor([2, 3]), max_new_tokens=3,
        include_target_features=True,
    )

    assert record.anchors == [0, 1, 2]
    assert record.anchor_token_ids == [3, record.target_ids[0], record.target_ids[1]]
    assert record.target_features == [[30.0, 0.0], [4.0, 1.0], [5.0, 2.0]]
    assert record.target_recent_hidden == [
        [[20.0, 0.0], [30.0, 0.0]],
        [[20.0, 0.0], [30.0, 0.0], [4.0, 1.0]],
        [[30.0, 0.0], [4.0, 1.0], [5.0, 2.0]],
    ]
    assert record.metadata["anchor_token_positions"] == record.anchors
    assert record.metadata["recent_hidden_positions"] == record.anchors
    assert record.metadata["trajectory_contract_version"] == 2

    cache = TrajectoryCache(tmp_path / "dflash-state.jsonl", fingerprint="contract-v2")
    cache.write([record])
    assert list(cache.read()) == [record]


def test_stage0_configurable_anchor_count_limits_selected_states() -> None:
    record = TargetTrajectoryBuilder(
        SyntheticTarget(vocab_size=32), seed=11, num_anchors=2,
    ).build_record("two-anchors", torch.tensor([1, 2]), max_new_tokens=7)

    assert len(record.anchors) == 2
    assert len(record.anchor_token_ids) == 2
    assert record.metadata["num_anchors_requested"] == 2


def test_stage0_anchor_token_contract_handles_eos_and_empty_suffix() -> None:
    eos_record = TargetTrajectoryBuilder(
        SyntheticTarget(vocab_size=16, eos_token_id=5), num_anchors=4,
    ).build_record("eos", torch.tensor([3, 4]), max_new_tokens=8)
    empty_record = TargetTrajectoryBuilder(
        SyntheticTarget(vocab_size=16), num_anchors=4,
    ).build_record("empty-contract", torch.tensor([3, 4]), max_new_tokens=0)

    assert eos_record.target_ids == [5]
    assert eos_record.anchors == [0]
    assert eos_record.anchor_token_ids == [4]
    assert empty_record.target_ids == []
    assert empty_record.anchors == []
    assert empty_record.anchor_token_ids == []


def test_legacy_anchor_token_cache_loads_with_explicit_warning() -> None:
    raw = {
        "sample_id": "legacy",
        "source_ids": [1, 2],
        "target_ids": [3, 4],
        "anchors": [0],
        "target_features": [[0.1, 0.2]],
        "metadata": {"target_feature_positions": [0]},
    }

    with pytest.warns(UserWarning, match="legacy trajectory contract"):
        record = TrajectoryRecord.from_json(raw)

    assert record.contract_version == 1
    assert record.anchor_token_ids is None
    assert record.target_recent_hidden is None
    with pytest.warns(UserWarning, match="deriving anchor token"):
        masked, targets, valid = build_stage1_batch(
            [record], kd=2, mask_token_id=31, anchor_indices=[1],
        )
    assert masked.tolist() == [[3, 31, 31]]
    assert targets.tolist() == [[3, 4, 0]]
    assert valid.tolist() == [[False, True, False]]


def test_stage0_can_cache_target_derived_source_memory_descriptors() -> None:
    class MemoryTarget(SyntheticTarget):
        def prefill(self, source_ids):
            state = super().prefill(source_ids)
            state.source_hidden = torch.arange(
                source_ids.numel() * 4, dtype=torch.float32,
            ).reshape(source_ids.numel(), 4)
            state.anchor_hidden = state.source_hidden[-1]
            return state

    record = TargetTrajectoryBuilder(
        MemoryTarget(vocab_size=24), source_chunk_size=2,
    ).build_record(
        "memory", torch.tensor([1, 2, 3, 4, 5]), max_new_tokens=2,
        include_source_memory=True,
    )
    assert record.source_memory is not None
    assert len(record.source_memory) == 3
    assert record.metadata["source_memory_source"] == "target_final_hidden"
    assert record.metadata["source_memory_chunk_offsets"] == [
        [0, 2], [2, 4], [4, 5],
    ]
    assert record.source_memory[0] == [2.0, 3.0, 4.0, 5.0]


def test_stage0_honors_zero_generation_limit() -> None:
    record = TargetTrajectoryBuilder(SyntheticTarget(vocab_size=24)).build_record(
        "empty", torch.tensor([1, 2]), max_new_tokens=0,
    )
    assert record.target_ids == []
    assert record.anchors == []


def test_stage1_batch_and_short_cpu_train_step(tmp_path: Path) -> None:
    target = SyntheticTarget(vocab_size=24)
    records = TargetTrajectoryBuilder(target, seed=2).build_records(
        [("doc", torch.tensor([1, 2, 3]))], max_new_tokens=6
    )
    records[0].anchors = [0]
    masked, targets, valid = build_stage1_batch(records, kd=4, mask_token_id=23)
    assert masked.shape == targets.shape == valid.shape == (1, 5)
    assert masked.tolist() == [[3, 23, 23, 23, 23]]
    assert targets.tolist() == [[3, *records[0].target_ids[:4]]]
    assert valid.tolist() == [[False, True, True, True, True]]

    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=24, hidden_size=8, layers=1, heads=2, groups=2, top_m=4
    ))
    trainer = SyncSpecTrainer(model, device="cpu", learning_rate=1e-3)
    summary = trainer.fit_diffusion(records, kd=4, mask_token_id=23, steps=2)
    assert summary["steps"] == 2
    assert summary["loss"] >= 0.0
    checkpoint = tmp_path / "checkpoint"
    trainer.save_checkpoint(checkpoint)
    assert (checkpoint / "pytorch_model.bin").is_file()
    assert (checkpoint / "optimizer_state.pt").is_file()
    resumed = SyncSpecTrainer(
        SyncSpecDrafter.from_pretrained(checkpoint), device="cpu", learning_rate=1e-3,
    )
    resumed.load_training_state(checkpoint)
    assert resumed.completed_steps == 2
    resumed_summary = resumed.fit_diffusion(records, kd=4, mask_token_id=23, steps=1)
    assert resumed_summary["completed_steps"] == 3


def test_stage1_batch_supports_explicit_anchor_selection() -> None:
    record = TargetTrajectoryBuilder(SyntheticTarget(vocab_size=32)).build_record(
        "anchors", torch.tensor([1, 2]), max_new_tokens=5,
    )
    record.anchors = [0, 2]
    _, targets, valid = build_stage1_batch(
        [record], kd=2, mask_token_id=31, anchor_indices=[2],
    )
    assert targets.tolist() == [[
        record.target_ids[1], record.target_ids[2], record.target_ids[3],
    ]]
    assert valid.tolist() == [[False, True, True]]


def test_stage1_anchor_offsets_use_real_target_context_position() -> None:
    records = [
        TargetTrajectoryBuilder(SyntheticTarget(vocab_size=32)).build_record(
            "offset-a", torch.tensor([1, 2, 3]), max_new_tokens=5,
        ),
        TargetTrajectoryBuilder(SyntheticTarget(vocab_size=32)).build_record(
            "offset-b", torch.tensor([4, 5]), max_new_tokens=5,
        ),
    ]
    offsets = anchor_position_offsets(records, [2, 1])
    assert offsets.tolist() == [4, 2]


def test_anchor_offsets_fall_back_when_anchor_metadata_was_rebound() -> None:
    record = TargetTrajectoryBuilder(SyntheticTarget(vocab_size=32)).build_record(
        "rebound", torch.tensor([1, 2, 3]), max_new_tokens=5,
    )
    record.anchors = [2]
    # This mirrors selector-stage expansion, where the original metadata can
    # still contain offsets for several anchors.
    record.metadata["anchor_position_offsets"] = [3, 4, 5]
    assert anchor_position_offsets([record], [2]).tolist() == [5]


def test_selector_stage_updates_selector_parameters() -> None:
    selector = SourceCoherentSelector(hidden_size=8, rank=4, ngram_dim=6)
    hidden = torch.randn(4, 8)
    candidate_ids = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    candidate_logits = torch.zeros(4, 3)
    targets = torch.tensor([1, 4, 7, 10])
    before = [param.detach().clone() for param in selector.parameters()]
    trainer = SyncSpecTrainer(selector, device="cpu", learning_rate=1e-2)
    summary = trainer.fit_selector_module(
        selector, hidden, candidate_ids, candidate_logits, targets,
        SourceNgramIndex([1, 2, 3, 4, 5]), steps=2,
    )
    assert summary["loss"] >= 0.0
    assert any(not torch.equal(old, new) for old, new in zip(before, selector.parameters()))


def test_selector_stage_trains_all_lattices_in_batch() -> None:
    selector = SourceCoherentSelector(hidden_size=8, rank=4, ngram_dim=6)
    hidden = torch.randn(2, 3, 8)
    candidate_ids = torch.tensor([
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [[2, 3, 4], [5, 6, 7], [8, 9, 10]],
    ])
    candidate_logits = torch.zeros_like(candidate_ids, dtype=torch.float32)
    targets = torch.tensor([[1, 5, 9], [2, 6, 10]])
    selector_indices = [SourceNgramIndex([1, 2, 3, 4]), SourceNgramIndex([2, 3, 4, 5])]
    before = [param.detach().clone() for param in selector.parameters()]
    trainer = SyncSpecTrainer(selector, device="cpu", learning_rate=1e-2)
    summary = trainer.fit_selector_module(
        selector, hidden, candidate_ids, candidate_logits, targets,
        selector_indices, history=[[1, 2], [2, 3]], steps=2,
    )
    assert summary["loss"] >= 0.0
    assert any(not torch.equal(old, new) for old, new in zip(before, selector.parameters()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/B200 is not available on this host")
def test_diffusion_training_smoke_on_cuda() -> None:
    target = SyntheticTarget(vocab_size=24, device="cuda")
    records = TargetTrajectoryBuilder(target, seed=5).build_records(
        [("cuda", torch.tensor([1, 2, 3], device="cuda"))], max_new_tokens=4
    )
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=24, hidden_size=8, layers=1, heads=2, groups=2, top_m=4,
    ))
    summary = SyncSpecTrainer(model, device="cuda", learning_rate=1e-3).fit_diffusion(
        records, kd=4, mask_token_id=23, steps=1,
    )
    assert summary["steps"] == 1
    assert summary["loss"] >= 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/B200 is not available on this host")
def test_full_syncspec_pipeline_smoke_on_cuda() -> None:
    device = torch.device("cuda")
    target = SyntheticTarget(vocab_size=24, device=device)
    records = TargetTrajectoryBuilder(target, seed=6).build_records(
        [("cuda-full", torch.tensor([1, 2, 3], device=device))], max_new_tokens=4
    )
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=24, hidden_size=8, layers=1, heads=2, groups=2, top_m=4,
    ))
    trainer = SyncSpecTrainer(model, device=device, learning_rate=1e-3)
    trainer.fit_diffusion(records, kd=4, mask_token_id=23, steps=1)
    masked, targets, valid = build_stage1_batch(records, 4, 23, device=device)
    with torch.no_grad():
        draft = model(masked)
    proposal_hidden = draft.hidden[:, 1:]
    proposal_targets = targets[:, 1:]
    proposal_valid = valid[:, 1:]
    candidate_ids, candidate_logits = top_m_candidates(draft.logits[:, 1:], 4)
    selector = SourceCoherentSelector(hidden_size=8, rank=4, ngram_dim=6)
    trainer.fit_selector_module(
        selector, proposal_hidden, candidate_ids, candidate_logits, proposal_targets,
        [SourceNgramIndex(records[0].source_ids)], history=[records[0].source_ids],
        valid_mask=proposal_valid, steps=1,
    )
    survival = SurvivalHead(8, hidden_size=8)
    features = torch.rand((4, 8), device=device)
    labels = torch.tensor([1, 1, 0, 0], dtype=torch.float32, device=device)
    trainer.fit_survival(survival, features, labels, steps=1)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target, top_m=4, hidden_size=8),
        SyncSpecConfig(
            vocab_size=24, hidden_size=8, top_m=4, device="cuda",
            budget_profiles=((0, 0), (4, 2), (4, 4)), predicted_spec_gain=0.2,
        ), selector=selector, survival_head=survival,
    )
    result = engine.generate(torch.tensor([1, 2, 3], device=device), max_new_tokens=3)
    assert result.status == "ok"
    assert result.token_ids.is_cuda
    assert result.committed_tokens == 3
