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
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .config import (
    DataConfig,
    ModelConfig,
    RunConfig,
    TrainingConfig,
    resolve_draft_init_layer_ids,
    resolve_feature_layer_ids,
)
from .model import DFlashDraftModel
from .mr_model import MRDFlashDraftModel
from .distributed import (
    barrier,
    current_context,
    destroy_process_group,
    setup_distributed,
)
from .training import (
    DFlashTrainStrategy,
    MRDFlashTrainStrategy,
    OnlineDFlashModel,
    OnlineMRDFlashModel,
    build_draft_spec_from_target_config,
    build_mr_draft_spec_from_target_config,
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
    # YAML assignment bypasses dataclass __init__, vì vậy phải validate lại
    # sau khi override để lỗi schema xuất hiện trước khi nạp model lớn.
    cfg.model.__post_init__()
    cfg.data.__post_init__()
    cfg.training.__post_init__()
    return cfg


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def apply_cli_overrides(cfg: RunConfig, args: argparse.Namespace) -> None:
    """Áp các override CLI phổ biến lên config."""
    # ``--target-layer-ids`` là alias CLI legacy của feature schema. Nếu
    # người dùng không truyền field mới, đồng bộ cả hai để override không bị
    # coi là cấu hình mâu thuẫn.
    if args.target_layer_ids is not None and args.feature_layer_ids is None:
        cfg.model.feature_layer_ids = list(args.target_layer_ids)
    mapping = {
        "target_model_path": (cfg.model, "target_model_path"),
        "architecture": (cfg.model, "architecture"),
        "draft_num_hidden_layers": (cfg.model, "draft_num_hidden_layers"),
        "target_layer_ids": (cfg.model, "target_layer_ids"),
        "feature_layer_ids": (cfg.model, "feature_layer_ids"),
        "draft_init_layer_ids": (cfg.model, "draft_init_layer_ids"),
        "block_size": (cfg.model, "block_size"),
        "mask_token_id": (cfg.model, "mask_token_id"),
        "draft_checkpoint_path": (cfg.model, "draft_checkpoint_path"),
        "init_draft_from_target": (cfg.model, "init_draft_from_target"),
        "torch_dtype": (cfg.model, "torch_dtype"),
        "mr_num_stages": (cfg.model, "mr_num_stages"),
        "hca_compression_ratio": (cfg.model, "hca_compression_ratio"),
        "csa_compression_ratio": (cfg.model, "csa_compression_ratio"),
        "memory_local_window": (cfg.model, "memory_local_window"),
        "csa_top_k": (cfg.model, "csa_top_k"),
        "indexer_dim": (cfg.model, "indexer_dim"),
        "train_data_path": (cfg.data, "train_data_path"),
        "eval_data_path": (cfg.data, "eval_data_path"),
        "eval_hidden_states_path": (cfg.data, "eval_hidden_states_path"),
        "hidden_states_path": (cfg.data, "hidden_states_path"),
        "max_length": (cfg.data, "max_length"),
        "num_samples": (cfg.data, "num_samples"),
        "cache_dir": (cfg.data, "cache_dir"),
        "num_workers": (cfg.data, "num_workers"),
        "prefetch_factor": (cfg.data, "prefetch_factor"),
        "pin_memory": (cfg.data, "pin_memory"),
        "persistent_workers": (cfg.data, "persistent_workers"),
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
        "eval_interval": (cfg.training, "eval_interval"),
        "seed": (cfg.training, "seed"),
    }
    for name, (obj, attr) in mapping.items():
        value = getattr(args, name, None)
        if value is None:
            continue
        if isinstance(getattr(obj, attr), bool):
            value = _coerce_bool(str(value))
        setattr(obj, attr, value)
    cfg.model.__post_init__()
    cfg.data.__post_init__()
    cfg.training.__post_init__()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DFlash offline (MR-DFlash).")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config (model/data/training).")
    parser.add_argument("--target-model-path", type=str, default=None)
    parser.add_argument("--architecture", type=str, choices=["dflash", "mr_dflash"], default=None)
    parser.add_argument("--draft-num-hidden-layers", type=int, default=None)
    parser.add_argument("--target-layer-ids", type=int, nargs="+", default=None)
    parser.add_argument("--feature-layer-ids", type=int, nargs="+", default=None)
    parser.add_argument("--draft-init-layer-ids", type=int, nargs="+", default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--draft-checkpoint-path", type=str, default=None)
    parser.add_argument("--init-draft-from-target", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None)
    parser.add_argument("--mr-num-stages", type=int, default=None)
    parser.add_argument("--hca-compression-ratio", type=int, default=None)
    parser.add_argument("--csa-compression-ratio", type=int, default=None)
    parser.add_argument("--memory-local-window", type=int, default=None)
    parser.add_argument("--csa-top-k", type=int, default=None)
    parser.add_argument("--indexer-dim", type=int, default=None)
    parser.add_argument("--train-data-path", type=str, default=None)
    parser.add_argument("--eval-data-path", type=str, default=None)
    parser.add_argument("--eval-hidden-states-path", type=str, default=None)
    parser.add_argument("--hidden-states-path", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--pin-memory", type=str, default=None)
    parser.add_argument("--persistent-workers", type=str, default=None)
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
    parser.add_argument("--eval-interval", type=int, default=None)
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
    local_files_only: Optional[bool] = None,
    keep_target_model: bool = False,
):
    """Nạp tokenizer + config + embed_tokens/lm_head (frozen) của target.

    Nạp đủ model để lấy head/embed rồi giải phóng phần còn lại. Với run GPU
    thật có thể tối ưu bằng cách đọc trực tiếp các slice từ safetensors.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if local_files_only is None:
        local_files_only = os.environ.get("FI_OFFLINE", "0").lower() in {
            "1", "true", "yes", "on"
        }

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[torch_dtype]

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    target = AutoModelForCausalLM.from_pretrained(
        target_model_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    ).to(device)
    target.eval()

    config = target.config
    embed_tokens = target.get_input_embeddings()
    lm_head = target.get_output_embeddings()
    if lm_head is None or embed_tokens is None:
        raise ValueError("target model thiếu embedding hoặc lm_head")
    if keep_target_model:
        # Dùng cùng instance cho init draft rồi giải phóng phần còn lại.
        return tokenizer, config, embed_tokens, lm_head, target

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
    """Dựng DFlash hoặc MR-DFlash model theo ``model.architecture``."""
    mcfg = cfg.model
    tcfg = cfg.training
    feature_layer_ids = resolve_feature_layer_ids(mcfg)

    mask_token_id = resolve_mask_token_id(mcfg, tokenizer, embed_tokens)
    if mcfg.architecture == "mr_dflash":
        spec = build_mr_draft_spec_from_target_config(
            target_config,
            draft_num_hidden_layers=mcfg.draft_num_hidden_layers,
            block_size=mcfg.block_size,
            target_layer_ids=feature_layer_ids,
            layer_types=mcfg.layer_types,
            sliding_window=mcfg.sliding_window,
            mask_token_id=mask_token_id,
            num_stages=mcfg.mr_num_stages,
            hca_compression_ratio=mcfg.hca_compression_ratio,
            csa_compression_ratio=mcfg.csa_compression_ratio,
            local_window=mcfg.memory_local_window,
            csa_top_k=mcfg.csa_top_k,
            indexer_dim=mcfg.indexer_dim,
        )
        draft = MRDFlashDraftModel(spec)
    else:
        spec = build_draft_spec_from_target_config(
            target_config,
            draft_num_hidden_layers=mcfg.draft_num_hidden_layers,
            block_size=mcfg.block_size,
            target_layer_ids=feature_layer_ids,
            layer_types=mcfg.layer_types,
            sliding_window=mcfg.sliding_window,
            mask_token_id=mask_token_id,
        )
        draft = DFlashDraftModel(spec)

    dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[mcfg.torch_dtype]
    draft.to(device=device, dtype=dtype)

    wrapper_cls = OnlineMRDFlashModel if mcfg.architecture == "mr_dflash" else OnlineDFlashModel
    model = wrapper_cls(
        draft,  # type: ignore[arg-type]
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
    from .data import DFlashFeatureDataset, list_feature_files
    from .trainer import Trainer

    dist_ctx = current_context(device)
    local_files_only = os.environ.get("FI_OFFLINE", "0").lower() in {
        "1", "true", "yes", "on"
    }
    out_dir = cfg.resolved_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    def ensure_capture(
        configured_path: Optional[str],
        raw_data_path: Optional[str],
        default_name: str,
        *,
        sample_limit: Optional[int],
    ) -> Optional[str]:
        """Capture một split đúng một lần rồi đồng bộ các rank."""
        if configured_path and list_feature_files(configured_path):
            return configured_path
        if not raw_data_path:
            if configured_path:
                raise ValueError(
                    f"feature path {configured_path!r} chưa có file và thiếu raw data"
                )
            return None
        features = Path(configured_path) if configured_path else out_dir / default_name
        error_file = out_dir / f".{default_name}.error"
        if dist_ctx.is_main:
            error_file.unlink(missing_ok=True)
            try:
                print(f"[run] chưa có feature; capture vào {features} ...")
                stats = capture_dataset(
                    target_model_path=cfg.model.target_model_path,
                    data_path=raw_data_path,
                    output_path=str(features),
                    max_length=cfg.data.max_length,
                    layer_ids=resolve_feature_layer_ids(cfg.model),
                    num_samples=sample_limit,
                    cache_dir=cfg.data.cache_dir,
                    torch_dtype=cfg.model.torch_dtype,
                    device="cuda" if device.type == "cuda" else "cpu",
                    local_files_only=local_files_only,
                )
                print(f"[run] capture xong: {stats}")
                if stats["captured"] == 0:
                    raise RuntimeError(
                        f"capture {default_name} không tạo được mẫu nào"
                    )
            except Exception as exc:
                error_file.write_text(repr(exc), encoding="utf-8")
        barrier()
        if error_file.exists():
            raise RuntimeError(
                f"capture {default_name} thất bại: "
                f"{error_file.read_text(encoding='utf-8')}"
            )
        if not list_feature_files(str(features)):
            raise RuntimeError(f"feature store rỗng sau capture: {features}")
        barrier()
        return str(features)

    features_path = cfg.data.hidden_states_path
    features_path = ensure_capture(
        features_path,
        cfg.data.train_data_path,
        "captured_features",
        sample_limit=cfg.data.num_samples,
    )
    if features_path is None:
        raise ValueError(
            "cần data.hidden_states_path (đã capture) hoặc data.train_data_path"
        )

    eval_features_path = ensure_capture(
        cfg.data.eval_hidden_states_path,
        cfg.data.eval_data_path or None,
        "captured_eval_features",
        sample_limit=None,
    )

    target_for_init = None
    loaded_target = load_target_parts(
        cfg.model.target_model_path,
        cache_dir=cfg.data.cache_dir,
        torch_dtype=cfg.model.torch_dtype,
        device=device,
        local_files_only=local_files_only,
        keep_target_model=cfg.model.init_draft_from_target,
    )
    if cfg.model.init_draft_from_target:
        tokenizer, target_config, embed_tokens, lm_head, target_for_init = loaded_target
    else:
        tokenizer, target_config, embed_tokens, lm_head = loaded_target
    model = build_online_model(
        cfg,
        tokenizer=tokenizer,
        target_config=target_config,
        embed_tokens=embed_tokens,
        lm_head=lm_head,
        device=device,
    )

    if cfg.model.init_draft_from_target:
        print("[run] init draft từ target layers ...")
        draft_init_layer_ids = resolve_draft_init_layer_ids(
            cfg.model,
            num_target_layers=int(target_for_init.config.num_hidden_layers),
        )
        copied = model.draft_model.init_from_target(
            target_for_init,
            target_layer_ids=draft_init_layer_ids,
        )
        del target_for_init
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[run] đã copy {len(copied)} tham số từ target")

    if cfg.model.draft_checkpoint_path:
        print("[run] warm-start draft weights ...")
        warm_start_draft_model(
            model.draft_model,
            cfg.model.draft_checkpoint_path,
            strategy_name=cfg.training.strategy,
        )

    if cfg.model.architecture == "mr_dflash":
        if cfg.training.strategy not in {"mr_dflash", "dflash"}:
            raise ValueError("MR-DFlash yêu cầu training.strategy=mr_dflash")
        strategy = MRDFlashTrainStrategy(model)  # type: ignore[arg-type]
    else:
        strategy = DFlashTrainStrategy(model)
    dataset = DFlashFeatureDataset(
        features_path,
        max_len=cfg.data.max_length,
        run_id=cfg.run_id,
        sample_limit=cfg.data.num_samples,
        expected_feature_width=model.draft_model.spec.context_feature_dim,
        expected_feature_layer_ids=resolve_feature_layer_ids(cfg.model),
    )
    print(
        f"[run] dataset: {len(dataset)} mẫu, max_len={cfg.data.max_length}, "
        f"block_size={model.block_size}, anchors={cfg.training.num_anchors}"
    )
    if len(dataset) == 0:
        raise RuntimeError("dataset rỗng — kiểm tra feature/capture")

    eval_dataset = None
    if eval_features_path is not None:
        eval_dataset = DFlashFeatureDataset(
            eval_features_path,
            max_len=cfg.data.max_length,
            run_id=f"{cfg.run_id}:eval",
            expected_feature_width=model.draft_model.spec.context_feature_dim,
            expected_feature_layer_ids=resolve_feature_layer_ids(cfg.model),
        )
        if len(eval_dataset) == 0:
            raise RuntimeError("eval dataset rỗng — kiểm tra eval capture")

    trainer = Trainer(
        cfg,
        strategy,
        dataset,
        device=device,
        resume_from=resume_from,
    )
    return trainer.fit(eval_dataset=eval_dataset)


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

    dist_ctx = setup_distributed(args.device)
    try:
        run(cfg, device=dist_ctx.device, resume_from=args.resume_from)
    finally:
        destroy_process_group()


if __name__ == "__main__":
    main()
