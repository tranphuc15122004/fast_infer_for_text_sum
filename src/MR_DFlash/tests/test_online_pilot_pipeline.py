"""Smoke tests cho data/tokenized/online-feature pilot path."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from MR_DFlash.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from MR_DFlash.data import render_conversation
from MR_DFlash.inference import DFlashInferenceEngine
from MR_DFlash.model import DFlashDraftModel
from MR_DFlash.online_features import OnlineTargetFeatureProvider
from MR_DFlash.run_train import build_online_model
from MR_DFlash.sampler import LengthBucketBatchSampler
from MR_DFlash.tokenized_data import TokenizedDFlashDataset, write_tokenized_manifest
from MR_DFlash.trainer import Trainer
from MR_DFlash.training import DFlashTrainStrategy


class _ToyTokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        ids = [10 + len(word) for word in str(text).split()]
        return {"input_ids": ids}

    def convert_tokens_to_ids(self, token):
        return 99 if token == "<|im_end|>" else -1

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False, return_dict=False):
        ids = []
        for message in conversation:
            role = {"system": 1, "user": 2, "assistant": 3}[message["role"]]
            ids.extend([role, *self(message["content"])["input_ids"], 99])
        if add_generation_prompt:
            ids.append(3)
        return ids


def _tiny_target():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    return Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            tie_word_embeddings=False,
            use_qk_norm=False,
            attention_bias=False,
        )
    ).float().eval()


def _write_tokenized(root: Path, n: int = 4) -> None:
    samples = []
    for index in range(n):
        ids = torch.arange(1, 25, dtype=torch.long) + index
        mask = torch.zeros(24, dtype=torch.float32)
        mask[8:23] = 1
        samples.append({"id": f"s{index}", "input_ids": ids, "loss_mask": mask, "length": 24})
    torch.save({"samples": samples}, root / "shard_00000.pt")
    write_tokenized_manifest(
        root,
        shards=[{"path": "shard_00000.pt", "count": n}],
        num_samples=n,
        target_model="tiny",
        feature_layer_ids=[1, 2],
        chat_template="toy",
        max_length=24,
        supervision_mode="last_assistant",
    )


def test_last_assistant_masks_only_final_turn() -> None:
    tokenizer = _ToyTokenizer()
    conversation = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
        {"role": "assistant", "content": "new answer"},
    ]
    _, all_mask = render_conversation(conversation, tokenizer, 128, supervision_mode="all_assistant")
    _, last_mask = render_conversation(conversation, tokenizer, 128, supervision_mode="last_assistant")
    assert sum(all_mask) > sum(last_mask) >= 2


def test_tokenized_dataset_and_bucket_sampler(tmp_path: Path) -> None:
    root = tmp_path / "tokenized"
    root.mkdir()
    _write_tokenized(root)
    dataset = TokenizedDFlashDataset(str(root), pad_token_id=0)
    batch = dataset.collate([dataset[0], dataset[1]])
    assert batch["input_ids"].shape == (2, 24)
    assert torch.equal(batch["attention_mask"], torch.ones(2, 24, dtype=torch.long))
    sampler = LengthBucketBatchSampler(dataset.lengths, batch_size=2, seed=42)
    batches = list(sampler)
    assert len(batches) == 2
    assert sorted(index for batch in batches for index in batch) == [0, 1, 2, 3]


def test_online_provider_uses_frozen_backbone() -> None:
    target = _tiny_target()
    provider = OnlineTargetFeatureProvider(target, [1, 2], dtype=torch.float32)
    ids = torch.randint(1, 63, (2, 12))
    mask = torch.ones_like(ids)
    features = provider(ids, mask)
    with torch.inference_mode():
        outputs = target.model(ids, attention_mask=mask, output_hidden_states=True, use_cache=False, return_dict=True)
    expected = torch.cat([outputs.hidden_states[2], outputs.hidden_states[3]], dim=-1)
    assert features.shape == (2, 12, 32)
    assert torch.allclose(features, expected)
    assert all(not parameter.requires_grad for parameter in target.parameters())


def test_online_training_smoke(tmp_path: Path) -> None:
    target = _tiny_target()
    tokenized_root = tmp_path / "tokenized"
    tokenized_root.mkdir()
    _write_tokenized(tokenized_root)
    cfg = RunConfig(
        run_id="online-smoke",
        output_dir=str(tmp_path / "out"),
        model=ModelConfig(
            target_model_path="tiny",
            architecture="dflash",
            draft_num_hidden_layers=1,
            block_size=4,
            mask_token_id=63,
            feature_layer_ids=[1, 2],
            torch_dtype="float32",
        ),
        data=DataConfig(
            feature_mode="online",
            tokenized_data_path=str(tokenized_root),
            max_length=24,
            num_workers=0,
        ),
        training=TrainingConfig(
            strategy="dflash",
            num_epochs=1,
            max_steps=1,
            batch_size=1,
            accumulation_steps=1,
            learning_rate=1e-3,
            num_anchors=4,
            objective_chunk_blocks=0,
            save_interval=0,
            log_interval=1,
        ),
    )
    class _Tokenizer:
        def convert_tokens_to_ids(self, token):
            return 63 if token == "[MASK]" else -1
        pad_token_id = 0

    configured = build_online_model(
        cfg,
        tokenizer=_Tokenizer(),
        target_config=target.config,
        embed_tokens=target.get_input_embeddings(),
        lm_head=target.get_output_embeddings(),
        device=torch.device("cpu"),
    )
    dataset = TokenizedDFlashDataset(str(tokenized_root), pad_token_id=0)
    provider = OnlineTargetFeatureProvider(target, [1, 2], dtype=torch.float32)
    sampler = LengthBucketBatchSampler(dataset.lengths, batch_size=1, seed=42)
    trainer = Trainer(
        cfg,
        DFlashTrainStrategy(configured),
        dataset,
        device=torch.device("cpu"),
        feature_provider=provider,
        batch_sampler=sampler,
    )
    summary = trainer.fit()
    assert summary["global_step"] == 1
    metrics = json.loads((tmp_path / "out" / "metrics.jsonl").read_text().splitlines()[0])
    assert "acc_at_1" in metrics
    assert "accept_ge_1" in metrics


def test_dflash_reference_inference_is_exact_to_greedy_target() -> None:
    target = _tiny_target()
    from MR_DFlash.training import build_draft_spec_from_target_config

    spec = build_draft_spec_from_target_config(
        target.config,
        draft_num_hidden_layers=1,
        block_size=4,
        target_layer_ids=[1, 2],
        mask_token_id=63,
    )
    draft = DFlashDraftModel(spec).float().eval()
    prompt = torch.randint(1, 63, (1, 8))
    result = DFlashInferenceEngine(
        target,
        draft,
        mask_token_id=63,
        device=torch.device("cpu"),
    ).generate(prompt, max_new_tokens=4, eos_token_id=None)
    vanilla = target.generate(prompt, max_new_tokens=4, do_sample=False)
    assert torch.equal(result.input_ids, vanilla)
