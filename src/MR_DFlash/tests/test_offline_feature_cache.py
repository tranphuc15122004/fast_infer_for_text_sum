"""Tests cho target-feature cache dạng shard, dùng lại giữa nhiều run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def _sample(sample_id: str, length: int, width: int):
    return {
        "id": sample_id,
        "input_ids": torch.arange(length, dtype=torch.long),
        "loss_mask": torch.ones(length, dtype=torch.float32),
        "hidden_states": torch.randn(length, width, dtype=torch.bfloat16),
        "length": length,
    }


def test_sharded_feature_store_roundtrip_and_collate(tmp_path: Path) -> None:
    from MR_DFlash.offline_features import (
        SHARDED_FEATURE_SCHEMA_VERSION,
        ShardedDFlashFeatureDataset,
        write_feature_shard,
        write_sharded_feature_manifest,
    )

    root = tmp_path / "features"
    write_feature_shard(
        root / "shard_00000.pt",
        [_sample("a", 4, 6), _sample("b", 6, 6)],
    )
    write_sharded_feature_manifest(
        root,
        target_model_path="tiny-target",
        feature_layer_ids=[1, 2],
        hidden_size=3,
        feature_width=6,
        max_length=8,
        requested_torch_dtype="bfloat16",
        shards=[{"path": "shard_00000.pt", "count": 2, "ids": ["a", "b"]}],
        sample_ids=["a", "b"],
    )

    dataset = ShardedDFlashFeatureDataset(
        str(root),
        max_len=8,
        expected_feature_width=6,
        expected_feature_layer_ids=[1, 2],
        expected_target_model_path="tiny-target",
    )
    assert dataset.manifest["schema_version"] == SHARDED_FEATURE_SCHEMA_VERSION
    assert len(dataset) == 2
    assert dataset.lengths == [4, 6]
    assert dataset[0]["hidden_states"].shape == (1, 4, 6)
    batch = dataset.collate([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == (2, 6)
    assert batch["loss_mask"].shape == (2, 6)
    assert batch["hidden_states"].shape == (2, 6, 6)
    assert batch["hidden_states"].dtype == torch.bfloat16

    with pytest.raises(ValueError, match="feature_width"):
        ShardedDFlashFeatureDataset(
            str(root), max_len=8, expected_feature_width=7
        )


def test_sharded_writer_resume_does_not_duplicate_ids(tmp_path: Path) -> None:
    from MR_DFlash.offline_features import (
        ShardedFeatureWriter,
        ShardedDFlashFeatureDataset,
    )

    root = tmp_path / "features"
    common = {
        "target_model_path": "tiny-target",
        "feature_layer_ids": [1, 2],
        "hidden_size": 3,
        "feature_width": 6,
        "max_length": 8,
        "requested_torch_dtype": "bfloat16",
    }
    writer = ShardedFeatureWriter(root, shard_size=2, resume=False, **common)
    writer.add(_sample("a", 4, 6))
    writer.add(_sample("b", 5, 6))
    writer.close()

    resumed = ShardedFeatureWriter(root, shard_size=2, resume=True, **common)
    assert resumed.existing_ids == {"a", "b"}
    assert not resumed.add(_sample("a", 4, 6))
    assert resumed.add(_sample("c", 3, 6))
    resumed.close()

    dataset = ShardedDFlashFeatureDataset(str(root), max_len=8)
    assert len(dataset) == 3
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["sample_ids"] == ["a", "b", "c"]


def test_hf_capture_batch_keeps_all_valid_offsets() -> None:
    from types import SimpleNamespace

    from MR_DFlash.capture import HFTargetCapture

    class TinyBackbone(torch.nn.Module):
        def forward(self, input_ids, **_kwargs):
            base = input_ids.to(torch.float32).unsqueeze(-1)
            # hidden_states[0] is embeddings; layer ids 0 and 1 follow the
            # same +1 offset as the production hook implementation.
            return SimpleNamespace(
                hidden_states=[
                    base,
                    base + 1,
                    base + 2,
                ]
            )

    capturer = object.__new__(HFTargetCapture)
    capturer.device = torch.device("cpu")
    capturer.layer_ids = [0, 1]
    capturer._hooks = []
    capturer._captured_layers = {}
    capturer.model = TinyBackbone().eval()

    values = capturer.capture_batch(
        torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    )
    assert [tuple(value.shape) for value in values] == [(3, 2), (2, 2)]
    assert values[0].dtype == torch.float32
    assert torch.equal(values[0][:, 0], torch.tensor([2.0, 3.0, 4.0]))
