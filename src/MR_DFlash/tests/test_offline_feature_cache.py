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


def test_cache_script_uses_batched_target_capture(tmp_path: Path, monkeypatch) -> None:
    import sys
    from types import SimpleNamespace

    script_dir = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import cache_target_features

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, conversation, **_kwargs):
            values = []
            for message in conversation:
                values.extend([10 if message["role"] == "user" else 11])
                values.extend(self(message["content"], add_special_tokens=False)["input_ids"])
            return values

        def __call__(self, text, **_kwargs):
            return {"input_ids": [20 + (ord(char) % 20) for char in str(text)]}

    class FakeCapturer:
        instances = []

        def __init__(self, *_args, **_kwargs):
            self.tokenizer = TinyTokenizer()
            self.device = torch.device("cpu")
            self.layer_ids = [0, 1]
            self.context_feature_dim = 6
            self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=3))
            self.batch_calls = 0
            self.__class__.instances.append(self)

        def capture_batch(self, input_ids, attention_mask):
            self.batch_calls += 1
            return [
                torch.full((int(length), 6), 1, dtype=torch.bfloat16)
                for length in attention_mask.sum(dim=-1).tolist()
            ]

        def capture_one(self, input_ids):
            return torch.ones(1, len(input_ids), 6, dtype=torch.bfloat16)

        def close(self):
            return None

    monkeypatch.setattr(cache_target_features, "HFTargetCapture", FakeCapturer, raising=False)
    data = tmp_path / "regenerated.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": sample_id,
                    "conversations": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": "answer"},
                    ],
                }
            )
            for sample_id, question in (("a", "short"), ("b", "a longer question"))
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "cache"
    stats = cache_target_features.cache_dataset(
        target_model_path="tiny-target",
        data_path=str(data),
        output_path=str(output),
        max_length=64,
        batch_size=2,
        shard_size=2,
        layer_ids=[0, 1],
        device="cpu",
    )
    assert stats["captured"] == 2
    assert FakeCapturer.instances[0].batch_calls == 1
    assert json.loads((output / "manifest.json").read_text())["num_samples"] == 2


def test_trainer_accepts_sharded_feature_dataset(tmp_path: Path) -> None:
    from MR_DFlash.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
    from MR_DFlash.offline_features import ShardedFeatureWriter, ShardedDFlashFeatureDataset
    from MR_DFlash.run_train import build_online_model
    from MR_DFlash.trainer import Trainer
    from MR_DFlash.training import DFlashTrainStrategy

    from transformers import Qwen3Config, Qwen3ForCausalLM

    target = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            use_qk_norm=False,
            attention_bias=False,
        )
    ).eval()
    cache = tmp_path / "features"
    writer = ShardedFeatureWriter(
        cache,
        shard_size=2,
        target_model_path="tiny",
        feature_layer_ids=[1, 2],
        hidden_size=16,
        feature_width=32,
        max_length=8,
        requested_torch_dtype="float32",
    )
    for sample_id in ("a", "b"):
        writer.add(_sample(sample_id, 8, 32) | {"loss_mask": torch.ones(8)})
    writer.close()

    cfg = RunConfig(
        output_dir=str(tmp_path / "run"),
        model=ModelConfig(
            target_model_path="tiny",
            architecture="dflash",
            block_size=4,
            mask_token_id=63,
            feature_layer_ids=[1, 2],
            torch_dtype="float32",
        ),
        data=DataConfig(
            hidden_states_path=str(cache),
            max_length=8,
        ),
        training=TrainingConfig(
            strategy="dflash",
            num_epochs=1,
            max_steps=1,
            batch_size=1,
            num_anchors=2,
            objective_chunk_blocks=0,
            save_interval=0,
            log_interval=1,
        ),
    )

    class Tokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            return 63 if token == "[MASK]" else -1

    model = build_online_model(
        cfg,
        tokenizer=Tokenizer(),
        target_config=target.config,
        embed_tokens=target.get_input_embeddings(),
        lm_head=target.get_output_embeddings(),
        device=torch.device("cpu"),
    )
    dataset = ShardedDFlashFeatureDataset(
        str(cache),
        max_len=8,
        expected_feature_width=32,
        expected_feature_layer_ids=[1, 2],
        expected_target_model_path="tiny",
        expected_max_length=8,
    )
    summary = Trainer(
        cfg,
        DFlashTrainStrategy(model),
        dataset,
        device=torch.device("cpu"),
    ).fit()
    assert summary["global_step"] == 1
