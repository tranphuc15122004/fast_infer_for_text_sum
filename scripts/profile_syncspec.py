#!/usr/bin/env python3
"""Measure SyncSpec component costs for one hardware/context/batch profile."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.config import (  # noqa: E402
    SyncSpecConfig, parse_budget_profiles,
)
from SyncSpec.controller import context_bin  # noqa: E402
from SyncSpec.engine import SyncSpecEngine  # noqa: E402
from SyncSpec.profile import ProfileKey, RuntimeProfiler  # noqa: E402
from SyncSpec.synthetic import SyntheticDrafter, SyntheticTarget  # noqa: E402
from SyncSpec.model import SyncSpecDrafter  # noqa: E402
from SyncSpec.selector import SourceCoherentSelector  # noqa: E402
from SyncSpec.survival import SurvivalHead  # noqa: E402
from SyncSpec.transformers_adapter import NativeDrafterAdapter, TransformersTargetAdapter  # noqa: E402
from SyncSpec.prompt import encode_record  # noqa: E402
from SyncSpec.verifier import VerificationResult  # noqa: E402


def _sync_if_cuda(device: str | torch.device) -> None:
    """Make a CUDA wall-clock boundary explicit for profiler measurements."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _measure_target_ar(profiler, target, states, device: str | torch.device):
    """Measure one post-prefill target AR token without charging prefill."""
    # ``prefill`` launches asynchronous CUDA work.  Establish the boundary
    # before starting the host timer; the trailing sync in ``decode`` closes
    # the other side of the interval.
    _sync_if_cuda(device)

    def decode():
        result = []
        for state in states:
            token = target.next_logits(state).argmax().reshape(1)
            target.commit(state, VerificationResult(token, 0))
            result.append(token)
        _sync_if_cuda(device)
        return result

    return profiler.measure("target_ar", decode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("synthetic", "transformers"), default="synthetic")
    parser.add_argument("--target-model")
    parser.add_argument("--drafter-checkpoint")
    parser.add_argument("--selector-checkpoint")
    parser.add_argument("--survival-checkpoint")
    parser.add_argument(
        "--allow-untrained-components", action="store_true",
        help="development-only: allow profiling a random selector/survival head",
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--kd", type=int)
    parser.add_argument("--kv", type=int)
    parser.add_argument(
        "--budget-profiles",
        help="comma-separated finite profiles, e.g. 8:4,8:8,16:8",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; use --device cpu only for a local non-B200 profile")
    if args.backend == "transformers" and not args.allow_untrained_components:
        missing_components = [
            label for label, value in (
                ("--selector-checkpoint", args.selector_checkpoint),
                ("--survival-checkpoint", args.survival_checkpoint),
            ) if not value
        ]
        if missing_components:
            raise SystemExit(
                "production Transformers profiling requires trained "
                + " and ".join(missing_components)
                + "; pass --allow-untrained-components only for development"
            )
    try:
        if args.budget_profiles:
            if args.kd is not None or args.kv is not None:
                raise ValueError("--budget-profiles cannot be combined with --kd/--kv")
            profile_specs = parse_budget_profiles(args.budget_profiles)
        elif args.kd is not None or args.kv is not None:
            if args.kd is None or args.kv is None:
                raise ValueError("--kd and --kv must be provided together")
            profile_specs = parse_budget_profiles(f"{args.kd}:{args.kv}")
        else:
            profile_specs = parse_budget_profiles("16:8")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if len(profile_specs) <= 1:
        raise SystemExit("at least one speculative budget profile is required")
    tokenizer = None
    if args.backend == "synthetic":
        target = SyntheticTarget(vocab_size=64, device=args.device)
        drafter = SyntheticDrafter(target, top_m=4)
        config = SyncSpecConfig(
            vocab_size=64, hidden_size=16, top_m=4, device=args.device,
            dtype="float32", budget_profiles=profile_specs,
        )
        model_name, checkpoint_name = "synthetic", "synthetic"
    else:
        if not args.target_model or not args.drafter_checkpoint:
            raise SystemExit("--target-model and --drafter-checkpoint are required for transformers profiling")
        target = TransformersTargetAdapter.from_pretrained(
            args.target_model, device=args.device, dtype=args.dtype,
            local_files_only=args.local_files_only,
        )
        model = SyncSpecDrafter.from_pretrained(args.drafter_checkpoint, map_location=args.device)
        embedding = target.model.get_input_embeddings()
        if model.config.vocab_size != embedding.num_embeddings or model.config.hidden_size != embedding.embedding_dim:
            raise SystemExit("drafter checkpoint width/vocabulary does not match target model")
        model.tie_target_weights(embedding, target.model.get_output_embeddings())
        model.to(args.device)
        drafter = NativeDrafterAdapter(model, target)
        config = SyncSpecConfig(
            vocab_size=model.config.vocab_size, hidden_size=model.config.hidden_size,
            top_m=model.config.top_m, device=args.device, dtype=args.dtype,
            target_model=args.target_model, drafter_checkpoint=args.drafter_checkpoint,
            budget_profiles=profile_specs,
        )
        tokenizer = target.tokenizer
        model_name, checkpoint_name = str(args.target_model), str(args.drafter_checkpoint)
    selector = None
    survival = None
    if args.selector_checkpoint:
        root = Path(args.selector_checkpoint)
        metadata = json.loads((root / "selector_config.json").read_text(encoding="utf-8"))
        selector_vocab_size = int(metadata.get("vocab_size", config.vocab_size or 65536))
        if config.vocab_size and selector_vocab_size != config.vocab_size:
            raise SystemExit("selector checkpoint vocabulary does not match target")
        selector = SourceCoherentSelector(
            int(metadata["hidden_size"]), rank=int(metadata.get("rank", 128)),
            ngram_dim=int(metadata.get("ngram_dim", 6)),
            vocab_size=selector_vocab_size,
            temperature=float(metadata.get("temperature", config.selector_temperature)),
        )
        selector.load_state_dict(torch.load(
            root / "selector.pt", map_location="cpu", weights_only=True,
        ))
    if args.survival_checkpoint:
        root = Path(args.survival_checkpoint)
        survival = SurvivalHead(8, hidden_size=min(64, config.hidden_size))
        survival.load_state_dict(torch.load(
            root / "survival.pt", map_location="cpu", weights_only=True,
        ))
    gpu_name = torch.cuda.get_device_name() if args.device.startswith("cuda") else "cpu"
    if args.input:
        raw_rows = [
            json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(raw_rows) < args.batch_size:
            raise SystemExit(
                f"profile input contains {len(raw_rows)} request(s), "
                f"but --batch-size={args.batch_size} requires at least that many"
            )
        sources = [
            encode_record(raw, tokenizer, max_input_tokens=args.max_input_tokens).to(target.device)
            for raw in raw_rows[:args.batch_size]
        ]
    else:
        sources = [
            (torch.arange(args.context_length, device=target.device) + index)
            % min(32, int(config.vocab_size))
            for index in range(args.batch_size)
        ]
    context_length = int(sources[0].numel())
    if args.batch_size > 1 and len({int(source.numel()) for source in sources}) != 1:
        raise SystemExit(
            "batch profile requires equal tokenized context lengths; "
            "use homogeneous input rows or profile batch 1"
        )
    payloads = []
    profile_source = "diagnostic" if args.allow_untrained_components else "measured"
    for profile in profile_specs[1:]:
        profile_config = replace(
            config,
            budget_profiles=(profile_specs[0], profile),
        )
        engine = SyncSpecEngine(
            target, drafter, profile_config, selector=selector,
            survival_head=survival,
        )
        key = ProfileKey(
            model_name, checkpoint_name, gpu_name, config.dtype,
            context_bin(context_length), f"batch{args.batch_size}",
            profile.kd, profile.kv, "pytorch",
            str(args.selector_checkpoint) if args.selector_checkpoint else None,
            str(args.survival_checkpoint) if args.survival_checkpoint else None,
        )
        profiler = RuntimeProfiler(key, source=profile_source)
        # The helper performs one post-prefill AR step per request.  Record the
        # number of measured tokens so ar_cost_from_profile normalizes batch
        # latency to the per-request opportunity cost correctly.
        profiler.target_ar_tokens = len(sources)

        def run_batch():
            if args.batch_size == 1:
                return [engine.generate(
                    sources[0], max_new_tokens=profile.kv,
                    force_kv=profile.kv, max_rounds=1,
                )]
            return engine.generate_batch(
                sources, max_new_tokens=profile.kv,
                force_kv=profile.kv, max_rounds=1,
            )

        for warmup in range(max(0, args.warmup_runs)):
            if args.batch_size == 1:
                engine.generate(
                    sources[0], max_new_tokens=profile.kv, seed=warmup,
                    force_kv=profile.kv, max_rounds=1,
                )
            else:
                engine.generate_batch(
                    sources, max_new_tokens=profile.kv, seed=warmup,
                    force_kv=profile.kv, max_rounds=1,
                )
        for repeat in range(max(0, args.repeats)):
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            results = profiler.measure("e2e", lambda: run_batch())
            # For a homogeneous microbatch, each result carries the shared
            # batch draft/verify wall time.  Recording one row preserves the
            # batch cost axis instead of multiplying by request count.
            for component, elapsed_ms in results[0].timing_ms.items():
                if component != "e2e":
                    profiler.record(component, elapsed_ms)
            if args.device.startswith("cuda"):
                profiler.record_peak_memory(
                    torch.cuda.max_memory_allocated(torch.device(args.device)) / 1024**2
                )
            # Prefill is intentionally outside target_ar timing; the helper
            # first synchronizes the post-prefill CUDA boundary so no pending
            # prefill work leaks into the opportunity-cost measurement.
            ar_states = [target.prefill(source) for source in sources]
            _measure_target_ar(profiler, target, ar_states, args.device)
        payloads.append(profiler.to_dict())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payloads[0] if len(payloads) == 1 else payloads, indent=2) + "\n",
        encoding="utf-8",
    )
    print(payloads[0] if len(payloads) == 1 else payloads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
