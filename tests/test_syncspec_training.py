from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.synthetic import SyntheticTarget  # noqa: E402
from SyncSpec.config import SyncSpecConfig  # noqa: E402
from SyncSpec.engine import SyncSpecEngine  # noqa: E402
from SyncSpec.synthetic import SyntheticDrafter  # noqa: E402
from SyncSpec.training import (  # noqa: E402
    SyncSpecTrainer,
    TrajectoryCache,
    TrajectoryRecord,
    artifact_fingerprint,
    diffusion_loss,
    selector_loss,
    survival_loss,
    calibration_metrics,
    _source_memory_batch,
)
from SyncSpec.model import SyncSpecDrafter, SyncSpecDrafterConfig  # noqa: E402
from SyncSpec.selector import SourceCoherentSelector  # noqa: E402
from SyncSpec.evidence import SourceNgramIndex  # noqa: E402
from SyncSpec.survival import SurvivalHead  # noqa: E402
from SyncSpec.trajectory import TargetTrajectoryBuilder  # noqa: E402


def test_trajectory_cache_round_trip_has_fingerprint(tmp_path: Path) -> None:
    record = TrajectoryRecord(
        sample_id="a",
        source_ids=[1, 2, 3],
        target_ids=[4, 5, 6],
        anchors=[0, 1],
        metadata={"model": "toy", "seed": 4},
    )
    path = tmp_path / "trajectories.jsonl"
    cache = TrajectoryCache(path, fingerprint="toy-v1")
    cache.write([record])
    loaded = list(cache.read())
    assert loaded == [record]
    assert json.loads(path.read_text().splitlines()[0])["schema_version"] == 1


def test_trajectory_cache_append_is_idempotent_by_sample_id(tmp_path: Path) -> None:
    record = TrajectoryRecord(sample_id="repeat", source_ids=[1], target_ids=[2])
    path = tmp_path / "trajectories.jsonl"
    cache = TrajectoryCache(path, fingerprint="toy-v1")
    cache.write([record])
    cache.write([record], append=True)
    assert list(cache.read()) == [record]


def test_jsonl_trajectory_cache_rejects_unknown_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "bad-schema.jsonl"
    path.write_text(
        json.dumps({
            "schema_version": 2, "fingerprint": "toy-v1",
            "sample_id": "a", "source_ids": [1], "target_ids": [2],
        }) + "\n",
        encoding="utf-8",
    )
    cache = TrajectoryCache(path, fingerprint="toy-v1")
    with pytest.raises(ValueError, match="schema"):
        cache.read_fingerprint()
    with pytest.raises(ValueError, match="schema"):
        list(cache.read())


def test_torch_trajectory_cache_round_trip_and_fingerprint(tmp_path: Path) -> None:
    record = TrajectoryRecord(
        sample_id="torch-cache", source_ids=[1, 2], target_ids=[3, 4],
        anchors=[0], target_features=[[0.1, 0.2]],
        source_memory=[[0.3, 0.4]], metadata={"context_length": 2},
    )
    path = tmp_path / "trajectories.pt"
    cache = TrajectoryCache(path, fingerprint="torch-v1")
    cache.write([record])
    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert torch.is_tensor(payload["records"][0]["target_features"])
    assert torch.is_tensor(payload["records"][0]["source_memory"])
    loaded = list(cache.read())
    assert loaded[0].sample_id == record.sample_id
    assert loaded[0].source_ids == record.source_ids
    assert loaded[0].target_ids == record.target_ids
    assert torch.allclose(
        torch.tensor(loaded[0].target_features),
        torch.tensor(record.target_features),
    )
    assert torch.allclose(
        torch.tensor(loaded[0].source_memory),
        torch.tensor(record.source_memory),
    )
    assert TrajectoryCache(path, fingerprint="wrong").read_fingerprint() == "torch-v1"
    with pytest.raises(ValueError, match="fingerprint"):
        list(TrajectoryCache(path, fingerprint="wrong").read())
    second = TrajectoryRecord(sample_id="second", source_ids=[5], target_ids=[6])
    cache.write([record, second], append=True)
    assert [item.sample_id for item in cache.read()] == ["torch-cache", "second"]


def test_artifact_fingerprint_changes_when_local_model_manifest_changes(tmp_path: Path) -> None:
    artifact = tmp_path / "model"
    artifact.mkdir()
    (artifact / "config.json").write_text('{"hidden_size": 16}', encoding="utf-8")
    (artifact / "model.safetensors").write_bytes(b"weights-v1")

    first = artifact_fingerprint(artifact)
    (artifact / "config.json").write_text('{"hidden_size": 32}', encoding="utf-8")
    second = artifact_fingerprint(artifact)

    assert first != second
    assert artifact_fingerprint(tmp_path / "missing-model") == artifact_fingerprint(
        tmp_path / "missing-model"
    )


def test_training_reuses_cached_target_source_memory_descriptors() -> None:
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=16, hidden_size=4, layers=1, heads=2, groups=2,
    ))
    record = TrajectoryRecord(
        sample_id="cached-memory", source_ids=[1, 2, 3, 4], target_ids=[5],
        source_memory=[[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]],
        metadata={"source_memory_chunk_offsets": [[0, 2], [2, 4]]},
    )
    result = _source_memory_batch(
        model, [record], torch.tensor([[1.0, 0.0, 0.0, 0.0]]), "cpu",
        top_r=1, chunk_size=2,
    )
    assert result is not None
    assert result.shape == (1, 1, 4)
    assert result[0, 0].tolist() == [10.0, 0.0, 0.0, 0.0]


def test_training_cached_source_memory_without_target_anchor_uses_embedding_query() -> None:
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=16, hidden_size=4, layers=1, heads=2, groups=2,
    ))
    record = TrajectoryRecord(
        sample_id="cached-memory-no-anchor", source_ids=[1, 2, 3, 4], target_ids=[5],
        source_memory=[[10.0, 0.0, 0.0, 0.0], [0.0, 10.0, 0.0, 0.0]],
        metadata={"source_memory_chunk_offsets": [[0, 2], [2, 4]]},
    )
    result = _source_memory_batch(
        model, [record], None, "cpu", top_r=1, chunk_size=2,
    )
    assert result is not None
    assert result.shape == (1, 1, 4)


def test_stage_losses_are_finite_and_trainable() -> None:
    logits = torch.randn(2, 4, 13, requires_grad=True)
    target = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    loss = diffusion_loss(logits, target, mask)
    candidate_logits = torch.randn(2, 4, 5, requires_grad=True)
    candidate_ids = torch.tensor([[[1, 2, 3, 4, 5]] * 4] * 2)
    selector = selector_loss(candidate_logits, candidate_ids, target, mask)
    hazard = torch.sigmoid(torch.randn(2, 4, requires_grad=True))
    surv = survival_loss(hazard, torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float32))
    total = loss + selector + surv
    assert torch.isfinite(total)
    total.backward()
    assert logits.grad is not None and candidate_logits.grad is not None


def test_survival_loss_trains_cumulative_survival_not_raw_hazard() -> None:
    hazard = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    labels = torch.tensor([[1.0, 1.0, 0.0]])
    expected = torch.nn.functional.binary_cross_entropy(
        torch.cumprod(1.0 - hazard, dim=-1), labels,
    )
    loss = survival_loss(hazard, labels)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert hazard.grad is not None


def test_diffusion_loss_supports_teacher_kl_and_top_m_margin() -> None:
    logits = torch.zeros((1, 2, 8), requires_grad=True)
    targets = torch.tensor([[2, 3]])
    teacher = torch.zeros_like(logits)
    teacher[..., 2] = 3.0
    teacher[..., 3] = 3.0
    loss = diffusion_loss(
        logits, targets, teacher_logits=teacher, kl_weight=0.25,
        rank_margin=0.5, rank_weight=0.1, rank_top_m=2,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_calibration_metrics_report_ece_and_brier() -> None:
    metrics = calibration_metrics(
        torch.tensor([0.9, 0.8, 0.2, 0.1]), torch.tensor([1.0, 1.0, 0.0, 0.0]), bins=2
    )
    assert set(metrics) >= {"ece", "brier"}
    assert metrics["ece"] >= 0.0
    assert metrics["brier"] >= 0.0


def test_on_policy_survival_examples_use_actual_engine_acceptance() -> None:
    target = SyntheticTarget(vocab_size=24)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target, top_m=4),
        SyncSpecConfig(vocab_size=24, hidden_size=16, top_m=4, predicted_spec_gain=0.2,
                       budget_profiles=((0, 0), (4, 2), (4, 4))),
    )
    from SyncSpec.training import collect_on_policy_survival_examples
    features, labels = collect_on_policy_survival_examples(
        engine, [("doc", torch.tensor([1, 2, 3]))], max_new_tokens=4
    )
    assert features.shape[-1] == 8
    assert features.shape[0] == labels.shape[0]
    assert labels.max().item() <= 1.0


def test_on_policy_survival_collection_can_cover_full_draft_block() -> None:
    target = SyntheticTarget(vocab_size=24)
    engine = SyncSpecEngine(
        target, SyntheticDrafter(target, top_m=4),
        SyncSpecConfig(vocab_size=24, hidden_size=16, top_m=4, predicted_spec_gain=0.2,
                       budget_profiles=((0, 0), (4, 2), (4, 4))),
    )
    from SyncSpec.training import collect_on_policy_survival_examples
    features, labels = collect_on_policy_survival_examples(
        engine, [("doc", torch.tensor([1, 2, 3]))], max_new_tokens=4,
        force_kv=4,
    )
    assert features.shape == (4, 8)
    assert labels.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_on_policy_survival_collection_propagates_reproducible_seed() -> None:
    from SyncSpec.training import collect_on_policy_survival_examples

    calls: list[dict] = []

    class Engine:
        def generate(self, source_ids, **kwargs):
            calls.append({"source": source_ids.tolist(), **kwargs})
            return type(
                "Result", (), {
                    "survival_features": [[0.0] * 8],
                    "survival_labels": [1.0],
                }
            )()

    features, labels = collect_on_policy_survival_examples(
        Engine(), [("a", torch.tensor([1])), ("b", torch.tensor([2]))],
        max_new_tokens=2, seed=37, force_kv=2,
    )
    assert calls == [
        {"source": [1], "max_new_tokens": 2, "seed": 37, "force_kv": 2},
        {"source": [2], "max_new_tokens": 2, "seed": 38, "force_kv": 2},
    ]
    assert features.shape == (2, 8)
    assert labels.tolist() == [1.0, 1.0]


def test_trainer_supports_accumulation_and_gradient_clipping() -> None:
    target = SyntheticTarget(vocab_size=16)
    records = TargetTrajectoryBuilder(target, seed=3).build_records(
        [("train", torch.tensor([1, 2, 3]))], max_new_tokens=4
    )
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=16, hidden_size=8, layers=1, heads=2, groups=2, top_m=4,
    ))
    trainer = SyncSpecTrainer(
        model, device="cpu", learning_rate=1e-3,
        grad_accumulation_steps=2, grad_clip_norm=0.5, amp=False,
    )
    summary = trainer.fit_diffusion(records, kd=2, mask_token_id=15, steps=3)
    assert summary["steps"] == 3
    assert summary["grad_accumulation_steps"] == 2


def test_trainer_supports_microbatch_training() -> None:
    target = SyntheticTarget(vocab_size=24)
    records = TargetTrajectoryBuilder(target, seed=13).build_records(
        [(f"doc-{index}", torch.tensor([1 + index, 2 + index, 3 + index])) for index in range(3)],
        max_new_tokens=4,
    )
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=24, hidden_size=8, layers=1, heads=2, groups=2, top_m=4,
    ))
    trainer = SyncSpecTrainer(model, device="cpu", learning_rate=1e-3)
    diffusion = trainer.fit_diffusion(
        records, kd=2, mask_token_id=23, steps=3, batch_size=1,
    )
    assert diffusion["batch_size"] == 1

    selector = SourceCoherentSelector(hidden_size=8, rank=4, ngram_dim=6)
    hidden = torch.randn(4, 3, 8)
    candidate_ids = torch.tensor([
        [[1, 2], [3, 4], [5, 6]],
        [[2, 3], [4, 5], [6, 7]],
        [[3, 4], [5, 6], [7, 8]],
        [[4, 5], [6, 7], [8, 9]],
    ])
    candidate_logits = torch.zeros_like(candidate_ids, dtype=torch.float32)
    targets = candidate_ids[..., 0]
    selector_summary = trainer.fit_selector_module(
        selector, hidden, candidate_ids, candidate_logits, targets,
        [SourceNgramIndex([1, 2, 3])] * 4, history=[[]] * 4,
        steps=2, batch_size=2,
    )
    assert selector_summary["batch_size"] == 2

    survival_summary = trainer.fit_survival(
        SurvivalHead(8, hidden_size=8), torch.rand(5, 8),
        torch.tensor([1, 1, 0, 0, 1], dtype=torch.float32), steps=2,
        batch_size=2,
    )
    assert survival_summary["batch_size"] == 2

    joint_model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=24, hidden_size=8, layers=1, heads=2, groups=2, top_m=4,
    ))
    joint_summary = SyncSpecTrainer(joint_model, device="cpu", learning_rate=1e-3).fit_joint(
        records, kd=2, mask_token_id=23,
        selector=SourceCoherentSelector(hidden_size=8, rank=4, ngram_dim=6),
        steps=2, batch_size=2,
    )
    assert joint_summary["batch_size"] == 2


def test_optional_joint_finetuning_updates_drafter_and_selector() -> None:
    target = SyntheticTarget(vocab_size=16)
    records = TargetTrajectoryBuilder(target, seed=8).build_records(
        [("joint", torch.tensor([1, 2, 3]))], max_new_tokens=4
    )
    model = SyncSpecDrafter(SyncSpecDrafterConfig(
        vocab_size=16, hidden_size=8, layers=1, heads=2, groups=2, top_m=4,
    ))
    selector = SourceCoherentSelector(hidden_size=8, rank=4, ngram_dim=6)
    before = [param.detach().clone() for param in model.layers.parameters()]
    trainer = SyncSpecTrainer(model, device="cpu", learning_rate=1e-3)
    summary = trainer.fit_joint(
        records, kd=2, mask_token_id=15, selector=selector, steps=1,
        selector_weight=0.1,
    )
    assert summary["stage"] == "joint_finetune"
    assert any(not torch.equal(old, new) for old, new in zip(before, model.layers.parameters()))
