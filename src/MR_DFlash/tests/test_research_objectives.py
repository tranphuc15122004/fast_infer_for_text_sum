from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from MR_DFlash.training import (
    DFlashTrainStrategy,
    OnlineDFlashModel,
    build_spec_auf_support_mask,
    hard_negative_components,
)


def test_spec_auf_keeps_tokens_through_first_prediction_failure() -> None:
    predicted = torch.tensor([[[1, 9, 3, 8, 5]]])
    target = torch.tensor([[[1, 2, 3, 4, 5]]])
    valid = torch.ones_like(predicted, dtype=torch.bool)
    active = build_spec_auf_support_mask(predicted, target, valid)
    assert active.tolist() == [[[False, True, False, False, False]]]


def test_spec_auf_stops_at_invalid_prefix_slot() -> None:
    predicted = torch.tensor([[[1, 2, 3, 4]]])
    target = torch.tensor([[[1, 2, 3, 4]]])
    valid = torch.tensor([[[False, False, True, True]]])
    active = build_spec_auf_support_mask(predicted, target, valid)
    assert not active.any()


def test_hard_negative_components_selects_shallow_rank_band() -> None:
    logits = torch.arange(40.0, 0.0, -1.0).reshape(1, 1, 1, 40)
    target = torch.tensor([[[23]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    loss, gate, rank = hard_negative_components(
        logits,
        target,
        valid,
        k=4,
        mode="shallow",
    )
    assert loss.shape == target.shape
    assert gate.tolist() == [[[True]]]
    assert rank.tolist() == [[[24]]]
    assert torch.isfinite(loss).all()


def test_hard_negative_components_all_mode_ignores_rank() -> None:
    logits = torch.tensor([[[[10.0, 9.0, 8.0, 7.0, 6.0, 5.0]]]])
    target = torch.tensor([[[5]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    _, gate, rank = hard_negative_components(
        logits,
        target,
        valid,
        k=4,
        mode="all",
    )
    assert gate.tolist() == [[[True]]]
    assert rank.tolist() == [[[6]]]


def test_all_e23_e24_objectives_produce_finite_backward() -> None:
    model = OnlineDFlashModel.__new__(OnlineDFlashModel)
    nn.Module.__init__(model)
    model.lm_head = nn.Linear(8, 40, bias=False)
    model.loss_type = "dflash"
    model.loss_decay_gamma = 7.0
    model.dpace_alpha = 0.5
    model.hard_negative_k = 32
    model.hard_negative_lambda = 0.25

    hidden = torch.randn(1, 2, 4, 8, requires_grad=True)
    targets = torch.randint(0, 40, (1, 2, 4))
    weights = torch.ones(1, 2, 4)
    weights[..., 0] = 0
    for objective in (
        "dflash",
        "dpace",
        "spec-auf",
        "dpace-hard-negative-all",
        "dpace-hard-negative-shallow",
    ):
        model.loss_type = objective
        terms = model._objective_chunk_terms(hidden, targets, weights)
        assert len(terms) == 10
        assert all(torch.isfinite(value).all() for value in terms)
        loss_num, loss_den, *_rest, hard_num, hard_den = terms
        if objective in {"dflash", "spec-auf"}:
            loss = loss_num / loss_den.clamp_min(1e-12)
        elif objective.startswith("dpace-hard-negative"):
            loss = loss_num / 1.0 + model.hard_negative_lambda * hard_num / hard_den.clamp_min(1e-12)
        else:
            loss = loss_num / 1.0
        loss.backward(retain_graph=True)
        assert torch.isfinite(hidden.grad).all()
        hidden.grad.zero_()


def test_e23_e24_objective_matrix_cpu_train_smoke(tmp_path) -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from MR_DFlash.config import DataConfig, ModelConfig, RunConfig, TrainingConfig
    from MR_DFlash.data import DFlashFeatureDataset, save_feature_file
    from MR_DFlash.model import DFlashDraftModel
    from MR_DFlash.run_train import build_draft_spec_from_target_config
    from MR_DFlash.trainer import Trainer

    target = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=40,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            use_qk_norm=False,
            attention_bias=False,
        )
    ).eval()
    spec = build_draft_spec_from_target_config(
        target.config,
        draft_num_hidden_layers=1,
        block_size=4,
        target_layer_ids=[1, 2],
    )
    spec.mask_token_id = 39
    features_dir = tmp_path / "features"
    for index in range(2):
        save_feature_file(
            str(features_dir / f"sample_{index}.ckpt"),
            {
                "input_ids": torch.randint(1, 39, (20,)),
                "loss_mask": torch.ones(20),
                "hidden_states": torch.randn(20, spec.context_feature_dim),
            },
        )
    dataset = DFlashFeatureDataset(str(features_dir), max_len=20, run_id="objective-smoke")

    objectives = (
        "dflash",
        "dpace",
        "spec-auf",
        "dpace-hard-negative-all",
        "dpace-hard-negative-shallow",
    )
    for objective in objectives:
        draft = DFlashDraftModel(spec).float()
        model = OnlineDFlashModel(
            draft,
            target_lm_head=target.get_output_embeddings(),
            target_embed_tokens=target.get_input_embeddings(),
            mask_token_id=39,
            block_size=4,
            num_anchors=4,
            loss_decay_gamma=7.0,
            objective_chunk_blocks=0,
            loss_type=objective,
            attention_backend="sdpa",
        )
        cfg = RunConfig(
            run_id=f"objective-{objective}",
            output_dir=str(tmp_path / objective),
            model=ModelConfig(
                target_model_path="tiny",
                block_size=4,
                mask_token_id=39,
                torch_dtype="float32",
                target_layer_ids=[1, 2],
            ),
            data=DataConfig(hidden_states_path=str(features_dir), max_length=20),
            training=TrainingConfig(
                max_steps=1,
                num_epochs=1,
                batch_size=1,
                accumulation_steps=1,
                learning_rate=1e-3,
                num_anchors=4,
                objective_chunk_blocks=0,
                loss_type=objective,
                save_interval=0,
                log_interval=1,
            ),
        )
        summary = Trainer(
            cfg,
            DFlashTrainStrategy(model),
            dataset,
            device=torch.device("cpu"),
        ).fit()
        assert summary["global_step"] == 1
        line = (tmp_path / objective / "metrics.jsonl").read_text().splitlines()[0]
        assert torch.isfinite(torch.tensor(float(__import__("json").loads(line)["loss"])))
