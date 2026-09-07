from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

try:
    from Finetuning.capture_features import capture_dataset
    from Finetuning.features import (
        FeatureManifest,
        OfflineFeatureDataset,
        collate_features,
        validate_feature_record,
    )
except ModuleNotFoundError as exc:  # Red phase: the feature contract is absent.
    _FEATURE_IMPORT_ERROR = exc


def _require_feature_api() -> None:
    if "_FEATURE_IMPORT_ERROR" in globals():
        pytest.fail(f"offline feature API is not implemented: {_FEATURE_IMPORT_ERROR}")


def tiny_manifest() -> FeatureManifest:
    _require_feature_api()
    return FeatureManifest(
        model_id="tiny-qwen3",
        revision="rev-1",
        tokenizer_id="tiny-tokenizer",
        layer_ids=[1, 3],
        hidden_size=4,
        max_length=8,
        hidden_states_dtype="torch.float32",
    )


def test_feature_manifest_round_trip_preserves_contract_fields() -> None:
    manifest = tiny_manifest()

    restored = FeatureManifest.from_dict(manifest.to_dict())

    assert restored.model_id == "tiny-qwen3"
    assert restored.revision == "rev-1"
    assert restored.tokenizer_id == "tiny-tokenizer"
    assert restored.layer_ids == [1, 3]
    assert restored.max_length == 8
    assert restored.feature_width == 8
    assert restored.hidden_states_dtype == "torch.float32"


def test_invalid_feature_sequence_lengths_are_rejected() -> None:
    manifest = tiny_manifest()
    record = {
        "input_ids": torch.ones(7, dtype=torch.long),
        "loss_mask": torch.ones(8),
        "hidden_states": torch.ones(8, 8),
    }

    with pytest.raises(ValueError, match="sequence lengths"):
        validate_feature_record(record, manifest)


def test_invalid_feature_width_and_dtypes_are_rejected() -> None:
    manifest = tiny_manifest()
    base = {
        "input_ids": torch.ones(4, dtype=torch.long),
        "loss_mask": torch.tensor([0, 1, 1, 0], dtype=torch.float32),
    }

    with pytest.raises(ValueError, match="feature width"):
        validate_feature_record(
            {**base, "hidden_states": torch.ones(4, 7)}, manifest
        )
    with pytest.raises(ValueError, match="dtype"):
        validate_feature_record(
            {
                **base,
                "input_ids": base["input_ids"].to(torch.int32),
                "hidden_states": torch.ones(4, 8),
            },
            manifest,
        )


def test_feature_validation_rejects_missing_adjacent_supervision() -> None:
    manifest = tiny_manifest()
    record = {
        "input_ids": torch.ones(4, dtype=torch.long),
        "loss_mask": torch.tensor([1, 0, 1, 0], dtype=torch.float32),
        "hidden_states": torch.ones(4, 8),
    }

    with pytest.raises(ValueError, match="two consecutive supervised tokens"):
        validate_feature_record(record, manifest)


def test_collator_right_pads_without_mixing_feature_width() -> None:
    _require_feature_api()

    batch = collate_features(
        [
            {
                "input_ids": torch.tensor([1, 2], dtype=torch.long),
                "loss_mask": torch.tensor([0, 1], dtype=torch.float32),
                "hidden_states": torch.ones(2, 4),
            },
            {
                "input_ids": torch.tensor([3], dtype=torch.long),
                "loss_mask": torch.tensor([1], dtype=torch.float32),
                "hidden_states": torch.ones(1, 4) * 2,
            },
        ]
    )

    assert batch["input_ids"].shape == (2, 2)
    assert batch["loss_mask"].shape == (2, 2)
    assert batch["hidden_states"].shape == (2, 2, 4)
    assert batch["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert batch["loss_mask"].tolist() == [[0, 1], [1, 0]]
    assert torch.equal(batch["hidden_states"][1, 1], torch.zeros(4))


def test_offline_dataset_reads_cpu_tensor_records(tmp_path) -> None:
    _require_feature_api()
    manifest = tiny_manifest()
    tmp_path.joinpath("manifest.json").write_text(
        __import__("json").dumps(manifest.to_dict()), encoding="utf-8"
    )
    torch.save(
        {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "loss_mask": torch.tensor([0, 1, 1, 0], dtype=torch.float32),
            "hidden_states": torch.ones(4, 8),
        },
        tmp_path / "feature_000000.pt",
    )

    dataset = OfflineFeatureDataset(tmp_path)
    record = dataset[0]

    assert len(dataset) == 1
    assert all(value.device.type == "cpu" for value in record.values())


def test_offline_dataset_requires_manifest_before_loading(tmp_path) -> None:
    _require_feature_api()

    with pytest.raises(FileNotFoundError, match="manifest"):
        OfflineFeatureDataset(tmp_path)


class FakeTargetModel(nn.Module):
    def __init__(self, *, bad_width: bool = False) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_hidden_layers=3,
            hidden_size=4,
            _commit_hash="local-rev",
        )
        self.bad_width = bad_width
        self.seen_no_grad = False
        self.seen_output_hidden_states = False

    def forward(self, input_ids, *, output_hidden_states, use_cache=False):
        del use_cache
        self.seen_no_grad = not torch.is_grad_enabled()
        self.seen_output_hidden_states = output_hidden_states
        batch, sequence = input_ids.shape
        width = 5 if self.bad_width else 4
        states = tuple(
            torch.full((batch, sequence, width), float(layer))
            for layer in range(self.config.num_hidden_layers + 1)
        )
        return SimpleNamespace(hidden_states=states)


def test_capture_dataset_is_local_eval_no_grad_and_manifest_first(
    tmp_path, monkeypatch
) -> None:
    _require_feature_api()
    model = FakeTargetModel()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(snapshot)
            assert kwargs["local_files_only"] is True
            return model

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    output_dir = tmp_path / "features"
    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "loss_mask": torch.tensor([0, 1, 1, 0], dtype=torch.float32),
        }
    ]

    manifest = capture_dataset(
        str(snapshot), examples, output_dir, [0, 2], 4, "cpu", torch.float32
    )

    assert manifest.feature_width == 8
    assert model.training is False
    assert model.seen_no_grad is True
    assert model.seen_output_hidden_states is True
    assert (output_dir / "manifest.json").is_file()
    stored = torch.load(
        output_dir / "feature_00000000.pt", map_location="cpu", weights_only=True
    )
    assert torch.equal(stored["hidden_states"][:, :4], torch.ones(4, 4))
    assert torch.equal(stored["hidden_states"][:, 4:], torch.full((4, 4), 3.0))
    assert len(OfflineFeatureDataset(output_dir, manifest=manifest)) == 1


def test_capture_dataset_fails_on_missing_snapshot_and_width_mismatch(
    tmp_path, monkeypatch
) -> None:
    _require_feature_api()
    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3], dtype=torch.long),
            "loss_mask": torch.tensor([0, 1, 1], dtype=torch.float32),
        }
    ]
    with pytest.raises(FileNotFoundError, match="local target model snapshot"):
        capture_dataset(
            str(tmp_path / "missing"), examples, tmp_path / "out", [0], 3, "cpu", torch.float32
        )

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    model = FakeTargetModel(bad_width=True)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(_path, **_kwargs):
            return model

    import transformers

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeAutoModel)
    with pytest.raises(ValueError, match="feature width"):
        capture_dataset(
            str(snapshot), examples, tmp_path / "out", [0], 3, "cpu", torch.float32
        )
