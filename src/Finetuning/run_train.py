"""CLI and assembly for the offline single-GPU Qwen3 DFlash trainer."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

from .capture_features import capture_dataset
from .config import RunConfig, apply_cli_overrides, load_run_config
from .dflash_family_model import OnlineDFlashModel
from .evaluation import Evaluator
from .features import OfflineFeatureDataset, collate_features
from .model import DFlashDraftModel, build_target_layer_ids
from .prepare_data import prepare_summary_examples
from .strategy import DFlashTrainStrategy
from .trainer import Trainer


def _dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {name}")
    return value


def _resolve_device(config: RunConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _make_draft_config(
    target_config: Qwen3Config,
    config: RunConfig,
    target_layer_ids: list[int],
    mask_token_id: int,
    *,
    attention_backend: str | None = None,
) -> Qwen3Config:
    payload = target_config.to_dict()
    payload.update(
        {
            "architectures": ["DFlashDraftModel"],
            "num_hidden_layers": config.model.num_draft_layers,
            "intermediate_size": config.model.draft_intermediate_size
            or int(getattr(target_config, "intermediate_size")),
            "layer_types": config.model.layer_types
            or ["full_attention"] * config.model.num_draft_layers,
            "num_target_layers": int(getattr(target_config, "num_hidden_layers")),
            "block_size": config.model.block_size,
            "dflash_config": {
                "target_layer_ids": target_layer_ids,
                "mask_token_id": mask_token_id,
            },
        }
    )
    draft_config = Qwen3Config.from_dict(payload)
    draft_config._attn_implementation = (
        attention_backend or config.training.attention_backend
    )
    return draft_config


def _resolve_mask_token_id(config: RunConfig, tokenizer: Any, vocab_size: int) -> int:
    if config.model.mask_token_id is not None:
        token_id = int(config.model.mask_token_id)
    else:
        token_id = getattr(tokenizer, "mask_token_id", None)
        if token_id is None and hasattr(tokenizer, "convert_tokens_to_ids"):
            candidate = tokenizer.convert_tokens_to_ids("[MASK]")
            token_id = candidate if isinstance(candidate, int) and candidate >= 0 else None
        if token_id is None:
            raise ValueError(
                "model.mask_token_id is required when tokenizer has no mask token"
            )
    if not 0 <= int(token_id) < vocab_size:
        raise ValueError(f"mask_token_id={token_id} is outside vocab_size={vocab_size}")
    return int(token_id)


def _target_layer_ids(config: RunConfig, target_config: Any) -> list[int]:
    if config.model.target_layer_ids is not None:
        return list(config.model.target_layer_ids)
    return build_target_layer_ids(
        int(getattr(target_config, "num_hidden_layers")),
        config.model.num_draft_layers,
    )


def _build_strategy(
    config: RunConfig,
    target_config: Qwen3Config,
    tokenizer: Any,
    embed_tokens: nn.Module,
    lm_head: nn.Module,
    device: torch.device,
) -> DFlashTrainStrategy:
    layer_ids = _target_layer_ids(config, target_config)
    mask_token_id = _resolve_mask_token_id(
        config,
        tokenizer,
        int(getattr(target_config, "vocab_size")),
    )
    draft_config = _make_draft_config(
        target_config,
        config,
        layer_ids,
        mask_token_id,
    )
    draft = DFlashDraftModel(draft_config).to(device=device, dtype=_dtype(config.model.torch_dtype))
    target_dtype = _dtype(config.model.torch_dtype)
    model = OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=lm_head.to(device=device, dtype=target_dtype),
        target_embed_tokens=embed_tokens.to(device=device, dtype=target_dtype),
        mask_token_id=mask_token_id,
        block_size=config.model.block_size,
        attention_backend=config.training.attention_backend,
        num_anchors=config.training.num_anchors,
        loss_decay_gamma=config.training.loss_decay_gamma,
        objective_chunk_blocks=config.training.objective_chunk_blocks,
        loss_type=config.training.loss_type,
        dpace_alpha=config.training.dpace_alpha,
    ).to(device)
    return DFlashTrainStrategy(model)


def _loader(dataset: OfflineFeatureDataset, batch_size: int) -> list[dict[str, torch.Tensor]]:
    if len(dataset) < batch_size:
        raise ValueError("feature dataset is smaller than batch_size")
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_features,
    )
    batches = list(data_loader)
    if not batches:
        raise ValueError("feature dataset produced no complete batches")
    return batches


def _build_synthetic(config: RunConfig, device: torch.device):
    target_config = Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        layer_types=["full_attention"] * 4,
        attention_dropout=0.0,
    )
    target_config._attn_implementation = config.training.attention_backend
    tokenizer = type("SyntheticTokenizer", (), {"mask_token_id": 0})()
    embed_tokens = nn.Embedding(target_config.vocab_size, target_config.hidden_size)
    lm_head = nn.Linear(target_config.hidden_size, target_config.vocab_size, bias=False)
    strategy = _build_strategy(
        config,
        target_config,
        tokenizer,
        embed_tokens,
        lm_head,
        device,
    )
    torch.manual_seed(config.training.seed + 1)
    sequence_length = config.data.max_length
    feature_width = len(_target_layer_ids(config, target_config)) * target_config.hidden_size
    input_ids = torch.arange(1, sequence_length + 1).remainder(target_config.vocab_size - 1) + 1
    loss_mask = torch.ones(sequence_length, dtype=torch.float32)
    hidden = torch.randn(sequence_length, feature_width, dtype=torch.float32) * 0.05
    batch = {
        "input_ids": input_ids.unsqueeze(0),
        "loss_mask": loss_mask.unsqueeze(0),
        "hidden_states": hidden.unsqueeze(0),
    }
    batches = [
        {key: value.clone() for key, value in batch.items()}
        for _ in range(max(config.training.max_steps or 4, 4))
    ]
    return strategy, batches, [batches[0]]


def _load_real_runtime(config: RunConfig, device: torch.device):
    target_path = config.model.target_model_path
    assert target_path is not None
    tokenizer = AutoTokenizer.from_pretrained(
        target_path,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.offline,
    )
    target = AutoModelForCausalLM.from_pretrained(
        target_path,
        trust_remote_code=config.model.trust_remote_code,
        torch_dtype=_dtype(config.model.torch_dtype),
        low_cpu_mem_usage=True,
        local_files_only=config.offline,
    ).to(device)
    target.eval()
    target_config = target.config
    embed_tokens = target.get_input_embeddings()
    lm_head = target.get_output_embeddings()
    if embed_tokens is None or lm_head is None:
        raise ValueError("target model must expose input embeddings and lm_head")
    layer_ids = _target_layer_ids(config, target_config)
    train_feature_path = config.data.hidden_states_path
    if train_feature_path is None:
        examples = prepare_summary_examples(
            config.data.train_data_path,
            tokenizer,
            max_length=config.data.max_length,
            chat_template=config.data.chat_template,
        )
        train_feature_path = str(config.resolved_output_dir / "features")
        capture_dataset(
            target_model_path=target_path,
            prepared_examples=examples,
            output_dir=train_feature_path,
            target_layer_ids=layer_ids,
            max_length=config.data.max_length,
            device=device,
            dtype=config.data.feature_dtype,
        )
    train_dataset = OfflineFeatureDataset(train_feature_path)
    strategy = _build_strategy(
        config,
        target_config,
        tokenizer,
        embed_tokens,
        lm_head,
        device,
    )
    expected_width = strategy.dflash_model.draft_model.fc.in_features
    if train_dataset.manifest.feature_width != expected_width:
        raise ValueError(
            "feature width does not match DFlash draft: "
            f"{train_dataset.manifest.feature_width} != {expected_width}"
        )
    train_batches = _loader(train_dataset, config.training.batch_size)
    eval_path = config.data.eval_hidden_states_path
    if eval_path is None and config.data.eval_data_path is not None:
        eval_examples = prepare_summary_examples(
            config.data.eval_data_path,
            tokenizer,
            max_length=config.data.max_length,
            chat_template=config.data.chat_template,
        )
        eval_path = str(config.resolved_output_dir / "eval_features")
        capture_dataset(
            target_model_path=target_path,
            prepared_examples=eval_examples,
            output_dir=eval_path,
            target_layer_ids=layer_ids,
            max_length=config.data.max_length,
            device=device,
            dtype=config.data.feature_dtype,
        )
    eval_batches = _loader(OfflineFeatureDataset(eval_path), config.training.batch_size) if eval_path else None
    del target
    return strategy, train_batches, eval_batches


def _assemble(config: RunConfig, device: torch.device):
    if config.synthetic:
        return _build_synthetic(config, device)
    return _load_real_runtime(config, device)


def run_training(config: RunConfig, *, resume_from: str | Path | None = None) -> Path:
    """Run config validation, assembly, optimization, evaluation and save."""

    config.validate()
    device = _resolve_device(config)
    _seed_everything(config.training.seed, device)
    strategy, train_batches, eval_batches = _assemble(config, device)
    trainer = Trainer(
        strategy=strategy,
        train_dataloader=train_batches,
        validation_dataloader=eval_batches,
        output_dir=config.resolved_output_dir,
        run_id=config.run_id,
        batch_size=config.training.batch_size,
        accumulation_steps=config.training.accumulation_steps,
        num_epochs=config.training.num_epochs,
        max_steps=config.training.max_steps,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        scheduler_type=config.training.scheduler_type,
        max_grad_norm=config.training.max_grad_norm,
        save_interval=config.training.save_interval,
        log_interval=config.training.log_interval,
        eval_interval=config.training.eval_interval,
        hardware_peak_tflops=config.training.hardware_peak_tflops,
        device=device,
        extra_checkpoint_state={"resolved_config": config.to_dict()},
        resume_from=resume_from,
    )
    trainer.fit()
    if eval_batches is not None and trainer._last_eval_step != trainer.global_step:
        metrics = trainer.evaluate()
        with trainer.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"type": "evaluation", "step": trainer.global_step, **metrics},
                    allow_nan=False,
                )
                + "\n"
            )
    return trainer.checkpoint_manager.latest_dir()


def evaluate_checkpoint(checkpoint_path: str | Path, config: RunConfig) -> dict[str, float]:
    config.validate()
    device = _resolve_device(config)
    _seed_everything(config.training.seed, device)
    strategy, _train_batches, eval_batches = _assemble(config, device)
    if eval_batches is None:
        raise ValueError("evaluation requires a validation feature source")
    return Evaluator().evaluate_checkpoint(
        str(checkpoint_path),
        lambda: strategy,
        lambda: eval_batches,
        device,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Qwen3 DFlash draft offline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--target-model-path")
    parser.add_argument("--train-data-path")
    parser.add_argument("--hidden-states-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--attention-backend")
    parser.add_argument("--run-id")
    parser.add_argument("--resume-from")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = load_run_config(args.config)
    apply_cli_overrides(
        config,
        target_model_path=args.target_model_path,
        train_data_path=args.train_data_path,
        hidden_states_path=args.hidden_states_path,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        device=args.device,
        attention_backend=args.attention_backend,
        run_id=args.run_id,
    )
    if args.smoke:
        config.training.max_steps = min(config.training.max_steps or 1, 1)
        config.training.attention_backend = "eager"
        config.training.save_interval = 1
        config.training.eval_interval = 1
        config.validate()
    path = run_training(config, resume_from=args.resume_from)
    print(f"final_checkpoint={path}")


if __name__ == "__main__":
    main()


__all__ = ["evaluate_checkpoint", "main", "run_training"]
