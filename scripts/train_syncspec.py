#!/usr/bin/env python3
"""Train a SyncSpec drafter/selector/survival component from local cache."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.model import SyncSpecDrafter, SyncSpecDrafterConfig, top_m_candidates  # noqa: E402
from SyncSpec.config import SyncSpecConfig  # noqa: E402
from SyncSpec.engine import SyncSpecEngine  # noqa: E402
from SyncSpec.evidence import SourceNgramIndex  # noqa: E402
from SyncSpec.selector import SourceCoherentSelector  # noqa: E402
from SyncSpec.survival import SurvivalHead  # noqa: E402
from SyncSpec.synthetic import SyntheticDrafter, SyntheticTarget  # noqa: E402
from SyncSpec.training import (  # noqa: E402
    SyncSpecTrainer,
    TrajectoryCache,
    build_stage1_batch,
    anchor_position_offsets,
    calibration_metrics,
    cache_fingerprint,
    collect_on_policy_survival_examples,
    _target_anchor_batch,
    _source_memory_batch,
)
from SyncSpec.transformers_adapter import (  # noqa: E402
    NativeDrafterAdapter,
    TransformersTargetAdapter,
)


def _load_records(path: Path, fingerprint: str | None):
    probe = TrajectoryCache(path, fingerprint or "")
    fp = fingerprint or probe.read_fingerprint()
    if not fp:
        raise ValueError("trajectory cache has no fingerprint; pass --fingerprint")
    return list(TrajectoryCache(path, fp).read()), fp


def _load_selector_checkpoint(
    path: str | Path, hidden_size: int, device: str | torch.device,
    vocab_size: int | None = None,
):
    """Load the selector trained by the separate Stage-2 CLI."""
    root = Path(path)
    metadata_path = root / "selector_config.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    checkpoint_hidden = int(metadata.get("hidden_size", hidden_size))
    if checkpoint_hidden != int(hidden_size):
        raise ValueError(
            f"selector checkpoint width {checkpoint_hidden} does not match drafter width {hidden_size}"
        )
    checkpoint_vocab = int(metadata.get("vocab_size", 65536))
    if vocab_size and checkpoint_vocab != int(vocab_size):
        raise ValueError("selector checkpoint vocabulary does not match target")
    selector = SourceCoherentSelector(
        checkpoint_hidden,
        rank=int(metadata.get("rank", min(128, checkpoint_hidden))),
        ngram_dim=int(metadata.get("ngram_dim", 6)),
        vocab_size=checkpoint_vocab,
        temperature=float(metadata.get("temperature", 1.0)),
    )
    state_path = root / "selector.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"selector checkpoint not found: {state_path}")
    selector.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    return selector.to(device).eval()


def _train_selector_stage(
    trainer: SyncSpecTrainer, model: SyncSpecDrafter, records, args,
    vocab_size: int, hidden_size: int, mask_id: int,
) -> tuple[SourceCoherentSelector, dict]:
    # Every stored target-generation anchor is a distinct serving state.  Keep
    # the true Top-M lattice for all of them so selector training does not
    # collapse to the first output position of each document.
    expanded = [
        replace(record, anchors=[int(anchor)])
        for record in records
        for anchor in (record.anchors or [0])
    ]
    hidden_chunks = []
    candidate_id_chunks = []
    candidate_logit_chunks = []
    target_chunks = []
    valid_chunks = []
    batch_size = int(args.train_batch_size)
    for start in range(0, len(expanded), batch_size):
        chunk = expanded[start:start + batch_size]
        anchor_indices = [record.anchors[0] if record.anchors else 0 for record in chunk]
        masked, target_ids, valid = build_stage1_batch(
            chunk, args.kd, mask_id, device=args.device, anchor_indices=anchor_indices,
        )
        anchor_tensor = _target_anchor_batch(
            chunk, anchor_indices, hidden_size, args.device,
        )
        position_offsets = anchor_position_offsets(chunk, anchor_indices, args.device)
        source_memory = _source_memory_batch(
            model, chunk, anchor_tensor, args.device,
        )
        with torch.no_grad():
            output = model(
                masked, target_anchor=anchor_tensor, source_memory=source_memory,
                position_offset=position_offsets,
            )
        top_ids, top_values = top_m_candidates(output.logits, max(2, min(16, vocab_size)))
        hidden_chunks.append(output.hidden.detach().cpu())
        candidate_id_chunks.append(top_ids.detach().cpu())
        candidate_logit_chunks.append(top_values.detach().cpu())
        target_chunks.append(target_ids.detach().cpu())
        valid_chunks.append(valid.detach().cpu())
    hidden = torch.cat(hidden_chunks, dim=0)
    top_ids = torch.cat(candidate_id_chunks, dim=0)
    top_values = torch.cat(candidate_logit_chunks, dim=0)
    target_ids = torch.cat(target_chunks, dim=0)
    valid = torch.cat(valid_chunks, dim=0)
    # Keep the true serving Top-M lattice. If the target token is absent,
    # fit_selector_module masks that position instead of injecting the
    # answer into the candidate set and overstating recall.
    selector = SourceCoherentSelector(
        hidden_size, rank=min(128, hidden_size), ngram_dim=6, vocab_size=vocab_size,
    )
    summary = trainer.fit_selector_module(
        selector, hidden, top_ids, top_values, target_ids,
        [SourceNgramIndex(record.source_ids) for record in expanded],
        history=[
            record.source_ids + record.target_ids[: int(record.anchors[0])]
            for record in expanded
        ], valid_mask=valid,
        steps=args.steps, batch_size=batch_size,
    )
    summary["anchor_count"] = len(expanded)
    torch.save(selector.state_dict(), args.output_dir / "selector.pt")
    (args.output_dir / "selector_config.json").write_text(
        json.dumps({
            "hidden_size": hidden_size, "rank": min(128, hidden_size),
            "ngram_dim": 6, "vocab_size": vocab_size, "temperature": 1.0,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return selector, summary


def _train_survival_stage(
    trainer: SyncSpecTrainer, model: SyncSpecDrafter, target, records, args,
    vocab_size: int, hidden_size: int, max_id: int, selector=None,
) -> tuple[SurvivalHead, dict, torch.Tensor, torch.Tensor]:
    if target is None:
        rollout_target = SyntheticTarget(vocab_size=max(vocab_size, max_id + 2))
        rollout_drafter = SyntheticDrafter(
            rollout_target, top_m=min(16, vocab_size), hidden_size=hidden_size,
        )
        rollout_config = SyncSpecConfig(
            vocab_size=rollout_target.vocab_size, hidden_size=hidden_size,
            top_m=min(16, vocab_size), predicted_spec_gain=0.2,
            budget_profiles=((0, 0), (args.kd, max(1, args.kd // 2)), (args.kd, args.kd)),
        )
    else:
        # Survival labels must come from the same target/drafter/selector
        # path used in serving. This is intentionally on-policy; a target
        # trajectory length is not a proxy for prefix acceptance.
        rollout_target = target
        rollout_drafter = NativeDrafterAdapter(model, target)
        rollout_config = SyncSpecConfig(
            vocab_size=vocab_size, hidden_size=hidden_size,
            top_m=model.config.top_m, device=args.device, dtype=args.dtype,
            target_model=args.target_model,
            budget_profiles=((0, 0), (args.kd, max(1, args.kd // 2)), (args.kd, args.kd)),
        )
    rollout_engine = SyncSpecEngine(
        rollout_target, rollout_drafter, rollout_config, selector=selector,
    )
    features, labels = collect_on_policy_survival_examples(
        rollout_engine, [(record.sample_id, torch.tensor(record.source_ids)) for record in records],
        max_new_tokens=args.kd, seed=args.seed, force_kv=args.kd,
    )
    if features.numel() == 0:
        raise RuntimeError("on-policy rollout produced no survival examples")
    head = SurvivalHead(8, hidden_size=min(64, hidden_size))
    summary = trainer.fit_survival(
        head, features, labels, steps=args.steps, batch_size=args.train_batch_size,
    )
    torch.save(head.state_dict(), args.output_dir / "survival.pt")
    summary["calibration"] = calibration_metrics(
        head.survival(features.to(args.device)), labels.to(args.device)
    )
    return head, summary, features, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("diffusion", "selector", "survival", "joint"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--train-batch-size", type=int, default=1,
        help="number of trajectory/lattice rows per training forward; use 1 for long contexts",
    )
    parser.add_argument("--kd", type=int, default=16)
    parser.add_argument(
        "--position-decay", type=float, default=0.0,
        help="optional exponential position weighting gamma; 0 keeps uniform CE",
    )
    parser.add_argument("--kl-weight", type=float, default=0.0,
                        help="optional cached-teacher KL weight (requires Stage-0 logits)")
    parser.add_argument("--rank-weight", type=float, default=0.0,
                        help="optional Top-M recoverability margin weight")
    parser.add_argument("--rank-margin", type=float, default=0.0,
                        help="optional Top-M recoverability margin")
    parser.add_argument("--rank-top-m", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument(
        "--max-positions", type=int, default=0,
        help="drafter positional capacity; default: target max_position_embeddings",
    )
    parser.add_argument("--mask-token-id", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-model")
    parser.add_argument("--init-checkpoint", help="resume a prior diffusion checkpoint for selector/joint training")
    parser.add_argument(
        "--selector-checkpoint",
        help="Stage-2 selector directory to reuse when collecting Stage-3 survival labels",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--fingerprint")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--joint-finetune", action="store_true",
                        help="run optional low-LR Stage-4 refinement after staged training")
    parser.add_argument("--joint-steps", type=int, default=100)
    parser.add_argument("--joint-learning-rate", type=float, default=0.0,
                        help="default: 0.1 * --learning-rate")
    parser.add_argument("--joint-selector-weight", type=float, default=0.1)
    parser.add_argument("--joint-survival-weight", type=float, default=0.1)
    args = parser.parse_args()

    if args.joint_finetune and args.stage != "joint":
        raise SystemExit("--joint-finetune is only valid with --stage joint")
    if args.selector_checkpoint and args.stage != "survival":
        raise SystemExit("--selector-checkpoint is currently valid only with --stage survival")
    if args.stage in {"selector", "survival"} and not args.init_checkpoint:
        raise SystemExit(
            f"--stage {args.stage} requires --init-checkpoint from diffusion training"
        )
    if args.stage == "survival" and not args.selector_checkpoint:
        raise SystemExit("--stage survival requires --selector-checkpoint from selector training")
    if args.position_decay < 0 or args.kl_weight < 0 or args.rank_weight < 0 or args.rank_margin < 0:
        raise SystemExit("position/teacher/rank loss weights must be non-negative")
    if args.rank_top_m <= 0:
        raise SystemExit("--rank-top-m must be positive")
    if args.max_positions < 0:
        raise SystemExit("--max-positions must be positive when provided")
    if args.train_batch_size <= 0:
        raise SystemExit("--train-batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but unavailable; run training smoke with --device cpu "
            "or execute training on the canonical CUDA server"
        )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, fingerprint = _load_records(args.data, args.fingerprint)
    max_id = max((max(r.source_ids + r.target_ids, default=0) for r in records), default=0)
    target = None
    target_embedding = None
    target_head = None
    target_max_positions = 0
    model = None
    if args.target_model:
        target = TransformersTargetAdapter.from_pretrained(
            args.target_model, device=args.device, dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
        target_embedding = target.model.get_input_embeddings()
        target_head = target.model.get_output_embeddings()
        vocab_size = int(target_embedding.num_embeddings)
        hidden_size = int(target_embedding.embedding_dim)
        target_max_positions = int(
            getattr(getattr(target.model, "config", None), "max_position_embeddings", 0) or 0
        )
    elif args.init_checkpoint:
        model = SyncSpecDrafter.from_pretrained(args.init_checkpoint, map_location=args.device)
        vocab_size = model.config.vocab_size
        hidden_size = model.config.hidden_size
    else:
        vocab_size = int(args.vocab_size or max_id + 2)
        hidden_size = int(args.hidden_size)
    if args.init_checkpoint and target is not None:
        model = SyncSpecDrafter.from_pretrained(args.init_checkpoint, map_location=args.device)
        if model.config.vocab_size != vocab_size or model.config.hidden_size != hidden_size:
            raise ValueError("initial drafter checkpoint does not match target model")
        if args.max_positions and args.max_positions != model.config.max_positions:
            raise ValueError("--max-positions does not match initial drafter checkpoint")
    checkpoint_mask_id = (
        getattr(model.config, "mask_token_id", None) if model is not None else None
    )
    mask_id = int(
        args.mask_token_id
        if args.mask_token_id is not None
        else (checkpoint_mask_id if checkpoint_mask_id is not None else vocab_size - 1)
    )
    if not args.init_checkpoint:
        max_positions = args.max_positions or target_max_positions or 4096
        config = SyncSpecDrafterConfig(
            vocab_size=vocab_size, hidden_size=hidden_size, layers=args.layers,
            heads=args.heads, groups=args.groups, max_positions=max_positions,
            mask_token_id=mask_id,
        )
        model = SyncSpecDrafter(config)
    if target is not None:
        model.tie_target_weights(target_embedding, target_head)
    trainer = SyncSpecTrainer(
        model, device=args.device, learning_rate=args.learning_rate,
        grad_accumulation_steps=args.grad_accumulation_steps,
        grad_clip_norm=args.grad_clip_norm, amp=args.amp, seed=args.seed,
    )
    if args.init_checkpoint and args.stage in {"diffusion", "joint"}:
        trainer.load_training_state(args.init_checkpoint)
    position_weight = None
    if args.position_decay > 0:
        position_weight = torch.exp(
            -torch.arange(args.kd, dtype=torch.float32, device=args.device)
            / float(args.position_decay)
        )
    selector_checkpoint = None
    if args.selector_checkpoint:
        selector_checkpoint = _load_selector_checkpoint(
            args.selector_checkpoint, hidden_size, args.device, vocab_size=vocab_size,
        )
    checkpoint_trainer = trainer
    if args.stage == "diffusion":
        summary = trainer.fit_diffusion(
            records, args.kd, mask_id, steps=args.steps,
            position_weight=position_weight, kl_weight=args.kl_weight,
            rank_margin=args.rank_margin, rank_weight=args.rank_weight,
            rank_top_m=args.rank_top_m, batch_size=args.train_batch_size,
        )
    elif args.stage == "selector":
        _, summary = _train_selector_stage(
            trainer, model, records, args, vocab_size, hidden_size, mask_id,
        )
    elif args.stage == "survival":
        _, summary, _, _ = _train_survival_stage(
            trainer, model, target, records, args, vocab_size, hidden_size, max_id,
            selector=selector_checkpoint,
        )
    else:
        diffusion_summary = trainer.fit_diffusion(
            records, args.kd, mask_id, steps=args.steps,
            position_weight=position_weight, kl_weight=args.kl_weight,
            rank_margin=args.rank_margin, rank_weight=args.rank_weight,
            rank_top_m=args.rank_top_m, batch_size=args.train_batch_size,
        )
        selector, selector_summary = _train_selector_stage(
            trainer, model, records, args, vocab_size, hidden_size, mask_id,
        )
        survival_head, survival_summary, survival_features, survival_labels = _train_survival_stage(
            trainer, model, target, records, args, vocab_size, hidden_size, max_id,
            selector=selector,
        )
        summary = {
            "stage": "joint",
            "steps": int(args.steps),
            "diffusion": diffusion_summary,
            "selector": selector_summary,
            "survival": survival_summary,
        }
        checkpoint_trainer = trainer
        if args.joint_finetune:
            joint_trainer = SyncSpecTrainer(
                model, device=args.device,
                learning_rate=(args.joint_learning_rate or args.learning_rate * 0.1),
                grad_accumulation_steps=args.grad_accumulation_steps,
                grad_clip_norm=args.grad_clip_norm, amp=args.amp, seed=args.seed,
            )
            joint_summary = joint_trainer.fit_joint(
                records, args.kd, mask_id, selector=selector,
                survival_head=survival_head, survival_features=survival_features,
                survival_labels=survival_labels, steps=args.joint_steps,
                selector_weight=args.joint_selector_weight,
                survival_weight=args.joint_survival_weight,
                batch_size=args.train_batch_size,
            )
            summary["joint_finetune"] = joint_summary
            torch.save(selector.state_dict(), args.output_dir / "selector.pt")
            torch.save(survival_head.state_dict(), args.output_dir / "survival.pt")
            checkpoint_trainer = joint_trainer
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage in {"diffusion", "joint"}:
        checkpoint_trainer.save_checkpoint(args.output_dir)
    summary.update({
        "fingerprint": fingerprint, "data": str(args.data), "output_dir": str(args.output_dir),
        "seed": args.seed, "amp": trainer.amp,
        "grad_accumulation_steps": trainer.grad_accumulation_steps,
        "grad_clip_norm": trainer.grad_clip_norm,
    })
    (args.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
