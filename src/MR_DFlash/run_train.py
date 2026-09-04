"""Entry point train DFlash offline: ``python -m MR_DFlash.run_train [flags]``.

Tương ứng ``specforge train --config <yaml>`` nhưng tự đóng gói:

1. Đọc config (mặc định dataclass hoặc file YAML khớp cây
   ``model/data/training``).
2. Nếu chưa có feature offline (``data.hidden_states_path``) → chạy capture
   bằng HF (``capture.capture_dataset``).
3. Nạp target parts (embed_tokens + lm_head, frozen) và dựng draft model +
   ``OnlineDFlashModel``.
4. Chạy ``Trainer.fit()`` (accumulation, cosine+warmup, checkpoint).

Tham số CLI phổ biến có thể override trực tiếp, ví dụ:
``python -m MR_DFlash.run_train --target-model-path Qwen/Qwen3-8B \
   --train-data-path data/user_prompts.jsonl --max-steps 100 --output-dir out``
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import DataConfig, ModelConfig, RunConfig, TrainingConfig
from .model import DFlashDraftModel
from .training import (
    DFlashTrainStrategy,
    OnlineDFlashModel,
    build_draft_spec_from_target_config,
)


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def _assign_dataclass(obj: Any, values: Dict[str, Any], section: str) -> None:
    """Gán dict vào dataclass; field lạ → lỗi (giống SpecForge không cho silent)."""
    allowed = {f.name for f in fields(obj)}
    for key, value in values.items():
        if key not in allowed:
            raise ValueError(f"field không hợp lệ {section}.{key}")
        setattr(obj, key, value)


def load_run_config(path: str) -> RunConfig:
    """Đọc YAML (model/data/training) thành RunConfig."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cần pyyaml: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    cfg = RunConfig()
    if "run_id" in payload:
        cfg.run_id = str(payload["run_id"])
    if "output_dir" in payload:
        cfg.output_dir = str(payload["output_dir"])
    for section, obj in (
        ("model", cfg.model),
        ("data", cfg.data),
        ("training", cfg.training),
    ):
        if section in payload and isinstance(payload[section], dict):
            _assign_dataclass(obj, payload[section], section)
    return cfg


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def apply_cli_overrides(cfg: RunConfig, args: argparse.Namespace) -> None:
    """Áp các override CLI phổ biến lên config."""
    mapping = {
        "target_model_path": (cfg.model, "target_model_path"),
        "draft_num_hidden_layers": (cfg.model, "draft_num_hidden_layers"),
        "block_size": (cfg.model, "block_size"),
        "mask_token_id": (cfg.model, "mask_token_id"),
        "draft_checkpoint_path": (cfg.model, "draft_checkpoint_path"),
        "init_draft_from_target": (cfg.model, "init_draft_from_target"),
        "torch_dtype": (cfg.model, "torch_dtype"),
        "train_data_path": (cfg.data, "train_data_path"),
        "eval_data_path": (cfg.data, "eval_data_path"),
        "hidden_states_path": (cfg.data, "hidden_states_path"),
        "max_length": (cfg.data, "max_length"),
        "num_samples": (cfg.data, "num_samples"),
        "cache_dir": (cfg.data, "cache_dir"),
        "num_epochs": (cfg.training, "num_epochs"),
        "max_steps": (cfg.training, "max_steps"),
        "batch_size": (cfg.training, "batch_size"),
        "accumulation_steps": (cfg.training, "accumulation_steps"),
        "learning_rate": (cfg.training, "learning_rate"),
        "warmup_ratio": (cfg.training, "warmup_ratio"),
        "max_grad_norm": (cfg.training, "max_grad_norm"),
        "num_anchors": (cfg.training, "num_anchors"),
        "loss_decay_gamma": (cfg.training, "loss_decay_gamma"),
        "objective_chunk_blocks": (cfg.training, "objective_chunk_blocks"),
        "attention_backend": (cfg.training, "attention_backend"),
        "loss_type": (cfg.training, "loss_type"),
        "save_interval": (cfg.training, "save_interval"),
        "log_interval": (cfg.training, "log_interval"),
        "seed": (cfg.training, "seed"),
    }
    for name, (obj, attr) in mapping.items():
        value = getattr(args, name, None)
        if value is None:
            continue
        if isinstance(getattr(obj, attr), bool):
            value = _coerce_bool(str(value))
        setattr(obj, attr, value)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DFlash offline (MR-DFlash).")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config (model/data/training).")
    parser.add_argument("--target-model-path", type=str, default=None)
    parser.add_argument("--draft-num-hidden-layers", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--draft-checkpoint-path", type=str, default=None)
    parser.add_argument("--init-draft-from-target", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--hidden-states-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--num-anchors", type=int, default=None)
    parser.add_argument("--loss-decay-gamma", type=float, default=None)
    parser.add_argument("--objective-chunk-blocks", type=int, default=None)
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--loss-type", type=str, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Target parts + model assembly
# --------------------------------------------------------------------------- #

def resolve_mask_token_id(
    cfg: ModelConfig,
    tokenizer: Any,
    embedding: torch.nn.Module,
) -> int:
    """mask_token_id: ưu tiên config; nếu thiếu thử '[MASK]' rồi báo lỗi rõ."""
    if cfg.mask_token_id is not None:
        token_id = int(cfg.mask_token_id)
    else:
        candidate = tokenizer.convert_tokens_to_ids("[MASK]")
        token_id = int(candidate) if candidate is not None and candidate >= 0 else -1
    if token_id < 0 or token_id >= embedding.num_embeddings:
        raise ValueError(
            f"mask_token_id={token_id} nằm ngoài vocab {embedding.num_embeddings}; "
            "đặt model.mask_token_id trong config (VD 151669 với Qwen3)."
        )
    return token_id


def load_target_parts(
    target_model_path: str,
    *,
    cache_dir: str = "./cache",
    trust_remote_code: bool = False,
    torch_dtype: str = "bfloat16",
    device: torch.device,
):
    """Nạp tokenizer + config + embed_tokens/lm_head (frozen) của target.

    Nạp đủ model để lấy head/embed rồi giải phóng phần còn lại. Với run GPU
    thật có thể tối ưu bằng cách đọc trực tiếp các slice từ safetensors.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path, cache_dir=cache_dir, trust_remote_code=trust_remote_code
    )
    target = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    target.eval()

    config = target.config
    embed_tokens = target.get_input_embeddings()
    lm_head = target.get_output_embeddings()
    if lm_head is None or embed_tokens is None:
        raise ValueError("target model thiếu embedding hoặc lm_head")
    # Giữ lại module head/embed; giải phóng phần còn lại của target.
    del target
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return tokenizer, config, embed_tokens, lm_head


def build_online_model(
    cfg: RunConfig,
    *,
    tokenizer: Any,
    target_config: Any,
    embed_tokens: torch.nn.Module,
    lm_head: torch.nn.Module,
    device: torch.device,
) -> OnlineDFlashModel:
    """Dựng draft model + wrapper OnlineDFlashModel theo config."""
    mcfg = cfg.model
    tcfg = cfg.training

    spec = build_draft_spec_from_target_config(
        target_config,
        draft_num_hidden_layers=mcfg.draft_num_hidden_layers,
        block_size=mcfg.block_size,
        target_layer_ids=mcfg.target_layer_ids,
        layer_types=mcfg.layer_types,
        sliding_window=mcfg.sliding_window,
    )
    mask_token_id = resolve_mask_token_id(mcfg, tokenizer, embed_tokens)
    spec.mask_token_id = mask_token_id

    draft = DFlashDraftModel(spec)

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[mcfg.torch_dtype]
    draft.to(device=device, dtype=dtype)

    model = OnlineDFlashModel(
        draft,
        lm_head.to(device=device, dtype=dtype),
        embed_tokens.to(device=device, dtype=dtype),
        mask_token_id=mask_token_id,
        block_size=spec.block_size,
        num_anchors=tcfg.num_anchors,
        loss_decay_gamma=tcfg.loss_decay_gamma,
        objective_chunk_blocks=tcfg.objective_chunk_blocks,
        loss_type=tcfg.loss_type,
        attention_backend=tcfg.attention_backend,
    ).to(device=device)
    return model


def run(cfg: RunConfig, *, device: torch.device, resume_from: Optional[str] = None) -> Dict[str, Any]:
    """Chạy cả pipeline (capture nếu cần → build model → fit)."""
    from .capture import capture_dataset
    from .checkpoint import warm_start_draft_model
    from .data import DFlashFeatureDataset
    from .trainer import Trainer

    out_dir = cfg.resolved_output_dir()
    features_path = cfg.data.hidden_states_path
    if not features_path or not Path(features_path).is_dir():
        if not cfg.data.train_data_path:
            raise ValueError(
                "cần data.hidden_states_path (đã capture) hoặc data.train_data_path"
            )
        features_path = str(out_dir / "captured_features")
        print(f"[run] chưa có feature; capture vào {features_path} ...")
        stats = capture_dataset(
            target_model_path=cfg.model.target_model_path,
            data_path=cfg.data.train_data_path,
            output_path=features_path,
            max_length=cfg.data.max_length,
            layer_ids=cfg.model.target_layer_ids,
            num_samples=cfg.data.num_samples,
            cache_dir=cfg.data.cache_dir,
            torch_dtype=cfg.model.torch_dtype,
            device="cuda" if device.type == "cuda" else "cpu",
        )
        print(f"[run] capture xong: {stats}")
        if stats["captured"] == 0:
            raise RuntimeError("capture không tạo được mẫu nào — kiểm tra dữ liệu")

    tokenizer, target_config, embed_tokens, lm_head = load_target_parts(
        cfg.model.target_model_path,
        cache_dir=cfg.data.cache_dir,
        torch_dtype=cfg.model.torch_dtype,
        device=device,
    )
    model = build_online_model(
        cfg,
        tokenizer=tokenizer,
        target_config=target_config,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        device=device,
    )

    if cfg.model.init_draft_from_target:
        from transformers import AutoModelForCausalLM

        print("[run] init draft từ target layers ...")
        dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[cfg.model.torch_dtype]
        target = AutoModelForCausalLM.from_pretrained(
            cfg.model.target_model_path,
            cache_dir=cfg.data.cache_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        copied = model.draft_model.init_from_target(target)
        del target
        gc.collect()
        print(f"[run] đã copy {len(copied)} tham số từ target")

    if cfg.model.draft_checkpoint_path:
        print("[run] warm-start draft weights ...")
        warm_start_draft_model(
            model.draft_model,
            cfg.model.draft_checkpoint_path,
            strategy_name=cfg.training.strategy,
        )

    strategy = DFlashTrainStrategy(model)
    dataset = DFlashFeatureDataset(
        features_path,
        max_len=cfg.data.max_length,
        run_id=cfg.run_id,
        sample_limit=cfg.data.num_samples,
    )
    print(
        f"[run] dataset: {len(dataset)} mẫu, max_len={cfg.data.max_length}, "
        f"block_size={model.block_size}, anchors={cfg.training.num_anchors}"
    )
    if len(dataset) == 0:
        raise RuntimeError("dataset rỗng — kiểm tra feature/capture")

    trainer = Trainer(
        cfg,
        strategy,
        dataset,
        device=device,
        resume_from=resume_from,
    )
    return trainer.fit()


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.config:
        cfg = load_run_config(args.config)
    else:
        cfg = RunConfig()
    apply_cli_overrides(cfg, args)
    if args.run_id:
        cfg.run_id = args.run_id
    if args.output_dir:
        cfg.output_dir = args.output_dir

    device = torch.device(
        args.device if args.device != "auto" else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run(cfg, device=device, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
