#!/usr/bin/env python3
"""Run SyncSpec-v1 inference with a synthetic or offline Transformers backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from SyncSpec.config import (  # noqa: E402
    SyncSpecConfig, parse_budget_profiles,
)
from SyncSpec.engine import SyncSpecEngine  # noqa: E402
from SyncSpec.model import SyncSpecDrafter  # noqa: E402
from SyncSpec.synthetic import SyntheticDrafter, SyntheticTarget  # noqa: E402
from SyncSpec.transformers_adapter import NativeDrafterAdapter, TransformersTargetAdapter  # noqa: E402
from SyncSpec.selector import SourceCoherentSelector  # noqa: E402
from SyncSpec.survival import SurvivalHead  # noqa: E402
from SyncSpec.prompt import encode_record as _encode_record, format_record_text as _format_record_text  # noqa: E402
from common.io_util import JsonlWriter  # noqa: E402
from common.paths import snapshot_dir  # noqa: E402
from common import rouge  # noqa: E402


def _local_target_available(value: str) -> bool:
    path = Path(value)
    if path.exists():
        return True
    return snapshot_dir(value) is not None if "/" in value else False


def _samples(path: Path | None, tokenizer, count: int, max_input_tokens: int = 0):
    if int(count) < 0:
        raise ValueError("sample count must be non-negative")
    if int(count) == 0:
        return []
    if path is None:
        return [
            (f"synthetic-{i}", torch.tensor([2 + i, 3 + i, 4 + i]), {"dataset": "syncspec"})
            for i in range(count)
        ]
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        sample_id = str(raw.get("id", raw.get("sample_id", len(rows))))
        ids = _encode_record(raw, tokenizer, max_input_tokens=max_input_tokens)
        rows.append((sample_id, ids, raw))
        if len(rows) >= int(count):
            break
    return rows


def _record(result, source_ids, model_name, tokenizer=None, raw=None):
    summary_text = (
        tokenizer.decode(result.token_ids.detach().cpu().tolist(), skip_special_tokens=True).strip()
        if tokenizer is not None else None
    )
    record = result.to_record(model=model_name, input_tokens=int(source_ids.numel()))
    record.update({
        "dataset": (raw or {}).get("dataset", "syncspec"),
        "retained_tokens": int(source_ids.numel()),
        "batch_size": result.batch_size,
        "selector_latency_ms": result.timing_ms.get("selector", 0.0),
        "ttft_ms": result.timing_ms.get("prefill", 0.0),
        "tpot_ms": max(
            0.0, result.timing_ms.get("e2e", 0.0) - result.timing_ms.get("prefill", 0.0)
        ) / max(1, result.committed_tokens),
        "e2e_ms": result.timing_ms.get("e2e", 0.0),
        "throughput_tok_s": result.committed_tokens / max(1e-9, result.timing_ms.get("e2e", 0.0) / 1000.0),
        "qps": None,
        "peak_memory_gb": None,
        "draft_latency_ms": result.timing_ms.get("draft", 0.0),
        "verification_latency_ms": result.timing_ms.get("verify", 0.0),
        "avg_accept_length": sum(result.accepted_lengths) / max(1, len(result.accepted_lengths)),
        "acceptance_rate": sum(result.accepted_lengths) / max(1, sum(b["kv"] for b in result.budgets)),
        "rejected_draft_ratio": 1.0 - (sum(result.accepted_lengths) / max(1, sum(b["kv"] for b in result.budgets))),
        "summary": summary_text,
    })
    if summary_text is not None:
        rouge.add_rouge(record, summary_text, (raw or {}).get("reference"))
    return record


def _load_trained_components(args, config):
    selector = None
    survival = None
    if args.selector_checkpoint:
        root = Path(args.selector_checkpoint)
        metadata_path = root / "selector_config.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        hidden_size = int(metadata.get("hidden_size", config.hidden_size))
        if hidden_size != config.hidden_size:
            raise ValueError("selector checkpoint width does not match drafter")
        selector_vocab_size = int(metadata.get("vocab_size", config.vocab_size or 65536))
        if config.vocab_size and selector_vocab_size != config.vocab_size:
            raise ValueError("selector checkpoint vocabulary does not match target")
        selector = SourceCoherentSelector(
            hidden_size, rank=int(metadata.get("rank", min(128, hidden_size))),
            ngram_dim=int(metadata.get("ngram_dim", 6)),
            vocab_size=selector_vocab_size,
            temperature=float(metadata.get("temperature", config.selector_temperature)),
        )
        selector.load_state_dict(torch.load(root / "selector.pt", map_location="cpu", weights_only=True))
    if args.survival_checkpoint:
        root = Path(args.survival_checkpoint)
        survival = SurvivalHead(8, hidden_size=min(64, config.hidden_size))
        survival.load_state_dict(torch.load(root / "survival.pt", map_location="cpu", weights_only=True))
    return selector, survival


def _load_gate_table(path: str | None) -> dict[str, float]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    table = payload.get("gate_table", payload) if isinstance(payload, dict) else {}
    if not isinstance(table, dict):
        raise ValueError("gate calibration must contain a gate_table object")
    return {str(key): float(value) for key, value in table.items()}


def _budget_profiles(
    backend: str,
    kd: int | None,
    kv: int | None,
    budget_profiles: str | None = None,
):
    """Resolve the serving budget contract used by both profiling and infer."""
    if (kd is None) != (kv is None):
        raise ValueError("--kd and --kv must be provided together")
    if budget_profiles and kd is not None:
        raise ValueError("--budget-profiles cannot be combined with --kd/--kv")
    if kd is not None and (kd <= 0 or kv <= 0 or kv > kd):
        raise ValueError("serving budget must satisfy K_d >= K_v >= 1")
    if kd is not None:
        return parse_budget_profiles(f"{int(kd)}:{int(kv)}")
    if budget_profiles:
        return parse_budget_profiles(budget_profiles)
    if backend == "synthetic":
        return parse_budget_profiles("4:2,4:4")
    # The production default exposes the finite v1.1 profile set.  A measured
    # profile is still required for each profile before CUDA speculation is
    # enabled; missing hardware/context/batch entries fall back to AR.
    return parse_budget_profiles("8:4,8:8,16:4,16:8,16:12,16:16")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("synthetic", "transformers"), default="synthetic")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-model")
    parser.add_argument("--drafter-checkpoint")
    parser.add_argument("--selector-checkpoint")
    parser.add_argument("--survival-checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="microbatch size used for grouped draft/verify forwards")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--kd", type=int, help="draft length used by the serving profile")
    parser.add_argument("--kv", type=int, help="verification length used by the serving profile")
    parser.add_argument(
        "--budget-profiles",
        help="comma-separated finite serving profiles, e.g. 8:4,8:8,16:8",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--check-exactness", action="store_true",
        help="compare greedy output with a fresh vanilla target-AR reference",
    )
    parser.add_argument("--profile")
    parser.add_argument("--gate-table", help="Stage-4 empirical pre-gate calibration JSON")
    parser.add_argument(
        "--allow-untrained-components", action="store_true",
        help="development-only: permit random selector/survival when using Transformers",
    )
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested but unavailable; run CPU smoke with --device cpu "
            "or execute the GPU smoke on the canonical CUDA server"
        )
    if args.check_exactness and args.stochastic:
        raise SystemExit("--check-exactness requires greedy decoding; disable --stochastic")
    if args.smoke:
        args.max_samples, args.max_new_tokens = 1, min(args.max_new_tokens, 8)
    try:
        budget_profiles = _budget_profiles(
            args.backend, args.kd, args.kv, args.budget_profiles,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.backend == "synthetic":
        runtime_device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        target = SyntheticTarget(vocab_size=64, eos_token_id=63, device=runtime_device)
        drafter = SyntheticDrafter(target, top_m=4)
        cfg = SyncSpecConfig(vocab_size=64, hidden_size=16, top_m=4, max_new_tokens=args.max_new_tokens,
                             device=runtime_device,
                             budget_profiles=budget_profiles, runtime_profile=args.profile,
                             gate_table=_load_gate_table(args.gate_table))
        tokenizer = None
        model_name = "synthetic"
    else:
        if args.local_files_only and args.target_model and not _local_target_available(args.target_model):
            raise SystemExit(f"offline target model not found: {args.target_model}")
        if args.local_files_only and args.drafter_checkpoint and not Path(args.drafter_checkpoint).exists():
            raise SystemExit(f"offline drafter checkpoint not found: {args.drafter_checkpoint}")
        if not args.target_model or not args.drafter_checkpoint:
            raise SystemExit("--target-model and --drafter-checkpoint are required for transformers backend")
        missing_components = [
            label for label, value in (
                ("--selector-checkpoint", args.selector_checkpoint),
                ("--survival-checkpoint", args.survival_checkpoint),
            ) if not value
        ]
        if missing_components and not args.allow_untrained_components:
            raise SystemExit(
                "production Transformers inference requires trained "
                + " and ".join(missing_components)
                + "; pass --allow-untrained-components only for development"
            )
        target = TransformersTargetAdapter.from_pretrained(
            args.target_model, device=args.device, dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
        model = SyncSpecDrafter.from_pretrained(args.drafter_checkpoint, map_location=args.device)
        if model.config.vocab_size != target.model.get_input_embeddings().num_embeddings or model.config.hidden_size != target.model.get_input_embeddings().embedding_dim:
            raise ValueError("drafter checkpoint width/vocabulary does not match target model")
        model.tie_target_weights(target.model.get_input_embeddings(), target.model.get_output_embeddings())
        model.to(args.device)
        drafter = NativeDrafterAdapter(model, target)
        cfg = SyncSpecConfig(vocab_size=model.config.vocab_size, hidden_size=model.config.hidden_size,
                             top_m=model.config.top_m, max_new_tokens=args.max_new_tokens, device=args.device,
                             target_model=args.target_model, drafter_checkpoint=args.drafter_checkpoint,
                             selector_checkpoint=args.selector_checkpoint,
                             survival_checkpoint=args.survival_checkpoint,
                             runtime_profile=args.profile,
                             require_measured_profile=args.device.startswith("cuda"),
                             budget_profiles=budget_profiles,
                             gate_table=_load_gate_table(args.gate_table))
        tokenizer = target.tokenizer
        model_name = str(args.target_model)
    selector, survival = _load_trained_components(args, cfg)
    if args.check_exactness and not callable(getattr(target, "generate_greedy", None)):
        raise SystemExit("--check-exactness requires a target adapter with generate_greedy()")
    engine = SyncSpecEngine(target, drafter, cfg, selector=selector, survival_head=survival)
    writer = JsonlWriter(args.output)
    rows = _samples(args.input, tokenizer, args.max_samples, args.max_input_tokens)
    selected_rows = rows[: args.max_samples]
    if args.batch_size == 1:
        batches = [selected_rows[index:index + 1] for index in range(0, len(selected_rows), 1)]
    else:
        batches = [
            selected_rows[index:index + args.batch_size]
            for index in range(0, len(selected_rows), args.batch_size)
        ]
    exactness_failures = []
    for batch in batches:
        if len(batch) == 1:
            results = [engine.generate(
                batch[0][1], max_new_tokens=args.max_new_tokens, stochastic=args.stochastic,
            )]
        else:
            results = engine.generate_batch(
                [row[1] for row in batch], max_new_tokens=args.max_new_tokens,
                stochastic=args.stochastic,
            )
        for (sample_id, source_ids, raw), result in zip(batch, results):
            record = _record(result, source_ids, model_name, tokenizer, raw) | {"input_id": sample_id}
            if args.check_exactness:
                reference = target.generate_greedy(
                    source_ids, max_new_tokens=args.max_new_tokens,
                )
                exact = torch.equal(
                    result.token_ids.detach().cpu(), reference.detach().cpu(),
                )
                record.update({
                    "exactness_checked": True,
                    "exact_match_vanilla_ar": bool(exact),
                    "vanilla_output_tokens": int(reference.numel()),
                })
                if not exact:
                    exactness_failures.append(str(sample_id))
            writer.add(record)
    status = "ok" if not exactness_failures else "fail"
    writer.finalize({
        "method": "syncspec", "status": status,
        "record_count": min(len(rows), args.max_samples), "backend": args.backend,
        "exactness_checked": bool(args.check_exactness),
        "exactness_failures": len(exactness_failures),
        **rouge.aggregate_rouge(writer.records),
    })
    print(json.dumps({
        "status": status, "records": min(len(rows), args.max_samples),
        "exactness_failures": len(exactness_failures), "output": str(args.output),
    }))
    return 0 if not exactness_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
