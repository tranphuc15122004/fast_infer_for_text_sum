"""Kiểm tra contract công bằng trước khi chạy matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from _common import write_json


def _load(path: str):
    from MR_DFlash.run_train import load_run_config
    return load_run_config(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Check fair DFlash/MR-DFlash configs")
    parser.add_argument("configs", nargs="+", help="at least DFlash-2L, MR-2S, DFlash-5L")
    parser.add_argument("--report", default=None)
    args = parser.parse_args(argv)
    if len(args.configs) < 3:
        raise ValueError("cần tối thiểu ba config trong experiment matrix")
    configs = [(Path(path), _load(path)) for path in args.configs]
    common_paths = {
        "target_model_path": lambda c: c.model.target_model_path,
        "feature_layer_ids": lambda c: tuple(c.model.feature_layer_ids or c.model.target_layer_ids or []),
        "block_size": lambda c: c.model.block_size,
        "mask_token_id": lambda c: c.model.mask_token_id,
        "torch_dtype": lambda c: c.model.torch_dtype,
        "init_draft_from_target": lambda c: c.model.init_draft_from_target,
        "feature_mode": lambda c: c.data.feature_mode,
        "tokenized_data_path": lambda c: c.data.tokenized_data_path,
        "eval_tokenized_data_path": lambda c: c.data.eval_tokenized_data_path,
        "hidden_states_path": lambda c: c.data.hidden_states_path,
        "eval_hidden_states_path": lambda c: c.data.eval_hidden_states_path,
        "max_length": lambda c: c.data.max_length,
        "supervision_mode": lambda c: c.data.supervision_mode,
        "strategy_objective": lambda c: (c.training.loss_type, c.training.loss_decay_gamma),
        "num_anchors": lambda c: c.training.num_anchors,
        "objective_chunk_blocks": lambda c: c.training.objective_chunk_blocks,
        "batch_size": lambda c: c.training.batch_size,
        "accumulation_steps": lambda c: c.training.accumulation_steps,
        "learning_rate": lambda c: c.training.learning_rate,
        "warmup_ratio": lambda c: c.training.warmup_ratio,
        "weight_decay": lambda c: c.training.weight_decay,
        "seed": lambda c: c.training.seed,
    }
    report: Dict[str, Any] = {"configs": [str(path) for path, _ in configs], "common": {}, "variants": {}}
    failures: List[str] = []
    for name, getter in common_paths.items():
        values = [getter(cfg) for _, cfg in configs]
        report["common"][name] = [*values]
        if len(set(values)) != 1:
            failures.append(f"common field {name} differs: {values}")
    for path, cfg in configs:
        variant = {
            "architecture": cfg.model.architecture,
            "draft_num_hidden_layers": cfg.model.draft_num_hidden_layers,
            "mr_num_stages": cfg.model.mr_num_stages,
            "indexer_dim": cfg.model.indexer_dim,
        }
        report["variants"][path.stem] = variant
        if cfg.data.feature_mode == "offline" and not cfg.data.hidden_states_path:
            failures.append(f"{path}: offline pilot cần data.hidden_states_path")
        if cfg.data.feature_mode == "online" and not cfg.data.tokenized_data_path:
            failures.append(f"{path}: online pilot cần data.tokenized_data_path")
        if cfg.model.init_draft_from_target:
            failures.append(f"{path}: pilot comparison bắt buộc init_draft_from_target=false")
    if args.report:
        write_json(args.report, report)
    if failures:
        raise SystemExit("FAIRNESS CHECK FAILED\n" + "\n".join(failures))
    print("[check_fairness] PASS")
    for path, cfg in configs:
        print(f"  {path.name}: {cfg.model.architecture}/{cfg.model.draft_num_hidden_layers}L, MR stages={cfg.model.mr_num_stages}")


if __name__ == "__main__":
    main()
