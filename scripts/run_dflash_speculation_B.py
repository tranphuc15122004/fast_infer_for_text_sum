#!/usr/bin/env python3
"""Run the DFlash speculation B0-B4 experiments on the shared B200 runtime.

This entrypoint is offline by design. It does not install packages or download
models/data. Use ``all`` for discovery and eligible confirm branches on B200,
``run`` for B0/B1/B2/B1-V, or execute individual phases separately.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts", ROOT / "externals" / "dflash"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "preflight", "validate", "b1", "b1v", "analyze", "b3-dev", "b3-confirm", "b4", "b4-dev", "b4-confirm", "run", "all"), default="prepare")
    parser.add_argument("--config", default=os.environ.get("FAST_INFER_MASTER_CONFIG"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", default="outputs/dflash_speculation_B")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--exposure-manifest", action="append", default=[])
    parser.add_argument("--cohort-phase", choices=("pilot", "dev", "confirm"), default="pilot")
    parser.add_argument("--allow-partial-cohort", action="store_true")
    parser.add_argument("--target-model", default=os.environ.get("MODEL_TARGET") or os.environ.get("TARGET_MODEL"))
    parser.add_argument("--draft-model", default=os.environ.get("MODEL_DFLASH_DRAFT") or os.environ.get("DRAFT_MODEL"))
    parser.add_argument("--attention-backend", choices=("flash_attention_4", "sdpa"), default="flash_attention_4")
    parser.add_argument("--gpu-ids", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--validation-tokens", type=int, default=128)
    parser.add_argument("--max-prefixes", type=int, default=15)
    parser.add_argument("--gammas", default="3,7,11,15")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--selection-repetitions", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--policy-lock", default=None, help="locked dev policy JSON required by b3-confirm/b4-confirm")
    parser.add_argument("--e2e-repetitions", type=int, default=5)
    parser.add_argument("--e2e-warmups", type=int, default=3)
    parser.add_argument("--opportunity-threshold-eta", type=float, default=0.0, help="hysteresis and cost-opportunity bypass threshold")
    parser.add_argument("--allow-gated-b3", action="store_true", help="allow B3 E2E evaluation with cost-opportunity gating even when global pooled B1-V gap is below 5%%")
    parser.add_argument("--b3-variants", default=None, help="optional comma-separated subset of B3 variants to benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("b200_%Y%m%dT%H%M%SZ")


def _is_fatal_campaign_error(exc: BaseException) -> bool:
    from TrainingFree.context_interventions.resilience import is_fatal_compute_error

    return isinstance(exc, Exception) and is_fatal_compute_error(exc)


def _clear_exception_traceback(exc: Exception) -> str:
    rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-12000:]
    exc.__traceback__ = None
    return rendered


def _compute_failure_exception(failure: Mapping[str, Any]) -> RuntimeError:
    attempts = int(failure.get("attempts", 1))
    message = f"{failure.get('reason', 'compute failed')} (attempts={attempts})"
    previous_traceback = failure.get("traceback")
    if previous_traceback:
        message += f"\nlast attempt traceback:\n{previous_traceback}"
    return RuntimeError(message)


def _recover_after_cell_error() -> None:
    """Release temporary tensors after a cell-local failure before continuing."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as cleanup_error:
        if _is_fatal_campaign_error(cleanup_error):
            raise


def _record_cell_error(
    path: Path,
    *,
    phase: str,
    campaign_signature: str,
    state_key: str,
    sample_id: str,
    exc: Exception,
) -> None:
    """Persist one failed cell and let the caller continue with other cells."""
    if _is_fatal_campaign_error(exc):
        raise exc
    from TrainingFree.context_adaptive.schema import atomic_json

    rendered_traceback = _clear_exception_traceback(exc)
    _recover_after_cell_error()
    atomic_json(path, {
        "schema_version": "dflash-speculation-b.failed-cell.v1",
        "status": "failed",
        "phase": phase,
        "campaign_signature": campaign_signature,
        "state_key": state_key,
        "sample_id": sample_id,
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "traceback": rendered_traceback,
    })
    print(
        f"[{phase}] cell failed; continuing: sample={sample_id} state={state_key} "
        f"{type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )


def _resolve(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _run_root(args: argparse.Namespace) -> Path:
    output_root = _resolve(args.output_root) or ROOT / "outputs/dflash_speculation_B"
    return output_root / str(args.run_id)


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _locked_benchmark_signature(campaign: Mapping[str, Any]) -> dict[str, Any]:
    corpus = campaign.get("corpus") or []
    return {
        "corpus_file_sha256": sorted(str(item["sha256"]) for item in corpus),
        "exposure_sha256": str(campaign.get("exposure_sha256", "")),
    }


def _locked_e2e_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_new_tokens": int(args.max_new_tokens),
        "gammas": list(_parse_gammas(args.gammas)),
        "attention_backend": str(args.attention_backend),
        "seed": int(args.seed),
        "e2e_repetitions": int(args.e2e_repetitions),
        "e2e_warmups": int(args.e2e_warmups),
    }


def _validate_policy_lock(
    lock: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    campaign: Mapping[str, Any],
    phase: str,
) -> None:
    if lock.get("protocol_parameters") != _locked_e2e_protocol(args):
        raise ValueError(f"{phase} policy lock protocol parameters differ from this run")
    if lock.get("benchmark_signature") != _locked_benchmark_signature(campaign):
        raise ValueError(f"{phase} policy lock corpus/exposure signature differs from this run")


def _load_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[Path]]:
    from TrainingFree.context_adaptive.benchmark import load_corpus

    if args.data_dir and args.data_file:
        raise ValueError("provide exactly one --data-dir/--data-file")
    if args.data_dir or args.data_file:
        # An explicit source replaces every legacy source from the master env.
        data_dir_value, data_file_value = args.data_dir, args.data_file
    else:
        data_dir_value = os.environ.get("DATA_INPUT") or os.environ.get("LONG_BENCH_DATA_DIR")
        data_file_value = os.environ.get("DATA_FILE") or os.environ.get("LONG_BENCH_DATA_FILE")
    if data_dir_value and Path(data_dir_value).is_file() and not data_file_value:
        data_file_value, data_dir_value = data_dir_value, None
    data_dir = _resolve(data_dir_value) if data_dir_value else None
    data_file = _resolve(data_file_value) if data_file_value else None
    if data_dir is None and data_file is None:
        data_dir = Path("/workspace/storage-shared/nlp/dungdx4/phuc_projects/data/longbench_100_14k")
    if bool(data_dir) == bool(data_file):
        raise ValueError("provide exactly one --data-dir/--data-file or DATA_INPUT")
    if data_file:
        paths = [data_file]
    else:
        paths = sorted(data_dir.glob("*.jsonl"))
    samples = load_corpus(data_dir=data_dir, data_file=data_file)
    available = {str(row.get("dataset")) for row in samples}
    missing = {"gov_report", "multi_news", "qmsum"} - available
    if missing and not args.allow_partial_cohort:
        raise ValueError(f"B campaign corpus is missing datasets: {sorted(missing)}")
    if not samples:
        raise ValueError("B campaign corpus is empty")
    return samples, paths


def _read_exposures(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.exposure_manifest:
        raise ValueError("prepare/run requires reviewed --exposure-manifest input from the A/B exposure audit")
    from TrainingFree.context_adaptive.splits import read_exposure_manifest

    manifests = []
    for value in args.exposure_manifest:
        path = _resolve(value) or Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"exposure manifest not found: {path}")
        manifests.append(read_exposure_manifest(path))
    return manifests


def _corpus_signature(paths: Sequence[Path]) -> list[dict[str, str]]:
    from TrainingFree.context_adaptive.schema import sha256_file

    return [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in paths]


def _prepare(args: argparse.Namespace, *, allow_existing: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.context_interventions.cohort import build_campaign_cohort

    run_root = _run_root(args)
    samples, paths = _load_samples(args)
    exposures = _read_exposures(args)
    split_path = str(_resolve(args.split_manifest)) if args.split_manifest else None
    selected, split_manifest, cohort = build_campaign_cohort(
        samples,
        exposures,
        phase=args.cohort_phase,
        seed=args.seed,
        require_quota=not args.allow_partial_cohort,
        split_manifest_path=split_path,
    )
    if not selected:
        raise ValueError("cohort selection returned no documents")
    if cohort.get("status") != "complete" and not args.allow_partial_cohort:
        raise ValueError(f"cohort is incomplete: {cohort.get('shortages')}")
    exposure_digest = _hash_json(exposures)
    campaign = {
        "schema_version": "dflash-speculation-b.campaign.v1",
        "run_id": args.run_id,
        "cohort_phase": args.cohort_phase,
        "seed": args.seed,
        "gammas": [int(value) for value in _parse_gammas(args.gammas)],
        "max_new_tokens": args.max_new_tokens,
        "validation_tokens": args.validation_tokens,
        "max_prefixes": args.max_prefixes,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "selection_repetitions": args.selection_repetitions,
        "n_splits": args.n_splits,
        "bootstrap_replicates": args.bootstrap_replicates,
        "attention_backend": args.attention_backend,
        "corpus": _corpus_signature(paths),
        "exposure_sha256": exposure_digest,
        "cohort_sha256": _hash_json(cohort),
        "split_manifest_sha256": _hash_json(split_manifest),
        "selected_sample_keys": sorted(f"{row['dataset']}::{row['id']}" for row in selected),
        "partial_cohort": cohort.get("status") != "complete",
        "thresholds": {
            "b1_positive_gap": 0.05,
            "b1_practical_null": 0.02,
            "b2_positive_skill": 0.05,
            "b2_incremental_skill": 0.02,
        },
    }
    run_root.mkdir(parents=True, exist_ok=True)
    campaign_path = run_root / "campaign.json"
    cohort_path = run_root / "cohort_manifest.json"
    split_output = run_root / "split_manifest.json"
    if campaign_path.exists():
        existing = json.loads(campaign_path.read_text(encoding="utf-8"))
        if _hash_json(existing) != _hash_json(campaign):
            raise ValueError("run id already contains a different campaign/cohort signature")
        if not allow_existing and not args.resume:
            raise FileExistsError(f"run already prepared: {run_root}; use --resume or choose a new --run-id")
        if not cohort_path.exists():
            atomic_json(cohort_path, cohort)
        if not split_output.exists():
            atomic_json(split_output, split_manifest)
    else:
        atomic_json(campaign_path, campaign)
        atomic_json(cohort_path, cohort)
        atomic_json(split_output, split_manifest)
    sample_by_key = {(str(row["dataset"]), str(row["id"])): row for row in samples}
    hydrated = []
    for row in cohort["groups"]:
        sample = sample_by_key[(str(row["dataset"]), str(row["sample_id"]))]
        item = dict(sample)
        item["_cad_source_group_id"] = str(row["source_group_id"])
        item["_cad_split"] = str(row["split"])
        hydrated.append(item)
    print(f"[prepare] run={args.run_id} cohort={len(hydrated)} groups; artifacts={run_root}", flush=True)
    return hydrated, campaign, run_root


def _load_prepared(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    # A prepared run is resumed from its saved protocol/cohort. Rebuilding the
    # campaign from current CLI defaults can produce a different campaign hash
    # and incorrectly block later phases such as B1-V.
    return _load_resume_campaign(args)


def _load_resume_campaign(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    """Load an existing run's saved cohort/protocol for a one-command resume."""
    if not args.run_id:
        raise ValueError("--mode run --resume requires the existing --run-id")
    run_root = _run_root(args)
    campaign_path = run_root / "campaign.json"
    cohort_path = run_root / "cohort_manifest.json"
    split_path = run_root / "split_manifest.json"
    if not campaign_path.is_file() or not cohort_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(f"cannot resume an unprepared run: {run_root}")

    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if campaign.get("run_id") != args.run_id:
        raise ValueError("saved campaign run_id does not match --run-id")
    if campaign.get("cohort_sha256") != _hash_json(cohort):
        raise ValueError("saved cohort manifest does not match campaign.json")
    if campaign.get("split_manifest_sha256") != _hash_json(split_manifest):
        raise ValueError("saved split manifest does not match campaign.json")

    args.allow_partial_cohort = bool(campaign.get("partial_cohort", False))
    samples, paths = _load_samples(args)
    if campaign.get("corpus") != _corpus_signature(paths):
        raise ValueError("current corpus differs from the saved campaign; resume with its original data path")
    sample_by_key = {(str(row["dataset"]), str(row["id"])): row for row in samples}
    selected: list[dict[str, Any]] = []
    for row in cohort.get("groups", []):
        key = (str(row["dataset"]), str(row["sample_id"]))
        sample = sample_by_key.get(key)
        if sample is None:
            raise ValueError(f"prepared cohort sample missing from current corpus: {key}")
        item = dict(sample)
        item["_cad_source_group_id"] = str(row["source_group_id"])
        item["_cad_split"] = str(row["split"])
        selected.append(item)
    selected_keys = sorted(f"{row['dataset']}::{row['id']}" for row in selected)
    if selected_keys != sorted(str(key) for key in campaign.get("selected_sample_keys", [])):
        raise ValueError("saved cohort rows differ from campaign.json")

    # Resume uses the protocol that created the run, even if CLI defaults have
    # since changed. Model paths and runtime environment still come from config.
    for name in (
        "cohort_phase", "seed", "max_new_tokens", "validation_tokens", "max_prefixes",
        "warmups", "repetitions", "selection_repetitions", "n_splits",
        "bootstrap_replicates", "attention_backend",
    ):
        setattr(args, name, campaign[name])
    args.gammas = ",".join(str(value) for value in campaign["gammas"])
    print(
        f"[resume] run={args.run_id} cohort={len(selected)} groups; "
        f"protocol loaded from {campaign_path}",
        flush=True,
    )
    return selected, campaign, run_root


def _parse_gammas(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("--gammas must be a comma-separated integer list") from exc
    from TrainingFree.speculation_experiments.protocol import SUPPORTED_GAMMAS

    if values != SUPPORTED_GAMMAS:
        raise ValueError(f"the preregistered B grid is exactly {SUPPORTED_GAMMAS}; reduced grids require a new protocol")
    return values


def _runtime(args: argparse.Namespace) -> dict[str, Any]:
    if not args.target_model or not args.draft_model:
        raise ValueError("target/draft model paths must be supplied by master config or command line")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["CAD_ATTENTION_BACKEND"] = args.attention_backend
    from TrainingFree.context_adaptive.benchmark import load_runtime

    runtime = load_runtime(args.target_model, args.draft_model, load_draft=True)
    draft = runtime["draft"]
    if int(draft.block_size) != 16:
        raise RuntimeError(f"B protocol requires DFlash-b16, received block_size={draft.block_size}")
    if int(runtime["target"].config.num_hidden_layers) != 36:
        raise RuntimeError("B protocol currently supports the 36-layer Qwen3-4B target checkpoint")
    if tuple(int(value) for value in draft.target_layer_ids) != (1, 9, 17, 25, 33):
        raise RuntimeError(f"unexpected DFlash target feature layers: {draft.target_layer_ids}")
    torch = runtime["torch"]
    properties = torch.cuda.get_device_properties(0)
    runtime["b_campaign_signature"] = {
        "runtime_signature": runtime["runtime_signature"].get("signature"),
        "target_signature": runtime.get("target_signature"),
        "draft_signature": runtime.get("draft_signature"),
        "tokenizer_signature": runtime.get("tokenizer_signature"),
        "gpu_name": str(properties.name),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory": int(properties.total_memory),
        "cuda_build": str(torch.version.cuda),
        "visible_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "attention_backend": runtime.get("attention_backend"),
    }
    print(f"[runtime] {json.dumps(runtime['b_campaign_signature'], ensure_ascii=False)}", flush=True)
    return runtime


def _sample_trace_path(run_root: Path, sample: Mapping[str, Any]) -> Path:
    key = f"{sample['dataset']}::{sample['id']}"
    name = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return run_root / "traces" / f"{name}.json"


def _trace_signature(
    *,
    sample: Mapping[str, Any],
    prompt_hash: str,
    args: argparse.Namespace,
    runtime_signature: Mapping[str, Any],
) -> str:
    return _hash_json({
        "sample": f"{sample['dataset']}::{sample['id']}",
        "prompt_hash": prompt_hash,
        "gammas": _parse_gammas(args.gammas),
        "max_new_tokens": args.max_new_tokens,
        "validation_tokens": args.validation_tokens,
        "runtime": runtime_signature,
    })


def _trace_signature_matches(trace: Mapping[str, Any], expected: str) -> bool:
    return str(trace.get("trace_signature", "")) == str(expected)


def _complete_b0_trace(trace: Mapping[str, Any], gammas: Sequence[int]) -> bool:
    required_sequences = ("ar_ids", "native_ids", "adapter_ids", "adapter_rounds", "eos_ids")
    if (
        trace.get("b0_status") != "complete"
        or not isinstance(trace.get("prompt_tokens"), int)
        or trace.get("prompt_tokens", 0) < 1
        or not isinstance(trace.get("prompt_hash"), str)
        or not isinstance(trace.get("sample_id"), str)
        or not isinstance(trace.get("dataset"), str)
        or not isinstance(trace.get("source_group_id"), str)
        or any(not isinstance(trace.get(key), list) for key in required_sequences)
        or not isinstance(trace.get("parity_by_gamma"), dict)
    ):
        return False
    if set(trace["parity_by_gamma"]) != {str(gamma) for gamma in gammas}:
        return False
    return all(isinstance(row, Mapping) for row in trace["adapter_rounds"])


def _b0_trace_matches_sample(
    trace: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    prompt_hash: str,
    gammas: Sequence[int],
) -> bool:
    """Check trace identity and protocol shape without comparing source hashes."""
    return (
        _complete_b0_trace(trace, gammas)
        and trace.get("sample_id") == str(sample["id"])
        and trace.get("dataset") == str(sample["dataset"])
        and trace.get("source_group_id") == str(sample["_cad_source_group_id"])
        and trace.get("prompt_hash") == prompt_hash
    )


def _validate(args: argparse.Namespace, samples: Sequence[dict[str, Any]], run_root: Path, runtime: Mapping[str, Any]) -> bool:
    from TrainingFree.context_adaptive.prompt import prepare_prompt
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.context_interventions.execution import run_reference_pair
    from TrainingFree.context_interventions.resilience import run_compute_unit

    gammas = _parse_gammas(args.gammas)
    validation_rows = []
    runtime_signature = runtime["b_campaign_signature"]
    previous_parity_path = run_root / "evidence" / "parity.json"
    previous_parity = (
        _read_json_artifact(run_root, "evidence", previous_parity_path)
        if previous_parity_path.is_file() else None
    )
    if (
        previous_parity
        and previous_parity.get("status") in {"complete", "partial"}
        and previous_parity.get("runtime") != runtime_signature
    ):
        raise ValueError("B0 runtime signature differs from saved measurements; use a new run ID")
    reuse_existing_traces = bool(
        previous_parity
        and previous_parity.get("status") in {"complete", "partial"}
        and previous_parity.get("runtime") == runtime_signature
    )
    for sample_index, sample in enumerate(samples, start=1):
        path = _sample_trace_path(run_root, sample)
        try:
            (ids, layout), setup_error = run_compute_unit(
                lambda: prepare_prompt(sample, runtime["tokenizer"], chunk_size=128),
                label=f"B0 prompt setup sample={sample['id']}",
            )
            if setup_error is not None:
                raise _compute_failure_exception(setup_error)
        except Exception as exc:
            if _is_fatal_campaign_error(exc):
                raise
            error_traceback = _clear_exception_traceback(exc)
            try:
                del ids
            except UnboundLocalError:
                pass
            _recover_after_cell_error()
            validation_rows.append({
                "sample_id": sample["id"],
                "status": "failed",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "traceback": error_traceback,
                "comparisons_by_gamma": {},
            })
            print(
                f"[B0 {sample_index}/{len(samples)}] prompt setup failed for "
                f"{sample['dataset']}:{sample['id']}; continuing: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        expected_signature = _trace_signature(
            sample=sample,
            prompt_hash=layout.prompt_hash,
            args=args,
            runtime_signature=runtime_signature,
        )
        if path.exists():
            saved = _read_json_artifact(run_root, "traces", path)
            if saved is not None:
                if reuse_existing_traces and _b0_trace_matches_sample(
                    saved, sample, prompt_hash=layout.prompt_hash, gammas=gammas,
                ):
                    parity = saved["parity_by_gamma"]
                    validation_rows.append({"sample_id": sample["id"], "status": "complete", "comparisons_by_gamma": parity})
                    print(f"[B0 {sample_index}/{len(samples)}] reuse {sample['dataset']}:{sample['id']}", flush=True)
                    continue
                # Preserve a complete trace across source edits when the saved
                # B0 runtime still matches. Rebuild only malformed or mismatched data.
                _archive_stale_cell(run_root, "traces", path, saved)

        parity_by_gamma: dict[str, Any] = {}
        reference_ar: list[int] | None = None
        reference_trace: dict[str, Any] | None = None
        measured: dict[str, Any] | None = None
        sample_error: Exception | None = None
        sample_error_traceback: str | None = None
        gamma_order = (15, *[gamma for gamma in gammas if gamma != 15])
        for gamma in gamma_order:
            limit = args.max_new_tokens if gamma == 15 else min(args.validation_tokens, args.max_new_tokens)
            try:
                measured, compute_error = run_compute_unit(
                    lambda: run_reference_pair(
                        runtime,
                        sample,
                        max_new_tokens=limit,
                        gamma=gamma,
                        reference_ar_ids=reference_ar,
                        collect_uncertainty=(gamma == 15),
                        require_ar_match=False,
                    ),
                    label=f"B0 reference pair sample={sample['id']} gamma={gamma}",
                )
                if compute_error is not None:
                    raise _compute_failure_exception(compute_error)
            except Exception as exc:
                if _is_fatal_campaign_error(exc):
                    raise
                sample_error_traceback = _clear_exception_traceback(exc)
                reference_trace = None
                measured = None
                try:
                    del ids
                except UnboundLocalError:
                    pass
                _recover_after_cell_error()
                sample_error = exc
                print(
                    f"[B0 {sample_index}/{len(samples)}] failed {sample['dataset']}:{sample['id']} "
                    f"gamma={gamma}; continuing with next sample: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            if gamma == 15:
                reference_ar = measured["ar_ids"]
                reference_trace = measured
            parity_by_gamma[str(gamma)] = measured["parity"]
            print(
                f"[B0 {sample_index}/{len(samples)}] {sample['dataset']}:{sample['id']} "
                f"gamma={gamma} comparison={measured['parity']['status']} (diagnostic only); "
                f"AR/native={measured['parity']['native_equals_ar']} "
                f"AR/adapter={measured['parity']['adapter_equals_ar']}",
                flush=True,
            )
        if sample_error is not None:
            validation_rows.append({
                "sample_id": sample["id"],
                "status": "failed",
                "error": {"type": type(sample_error).__name__, "message": str(sample_error)},
                "traceback": sample_error_traceback,
                "comparisons_by_gamma": parity_by_gamma,
            })
            del reference_trace, measured
            continue
        if reference_trace is None:
            raise RuntimeError("gamma15 reference trace was not produced despite successful validation")
        trace = {
            "schema_version": "dflash-speculation-b.reference.v1",
            "trace_signature": expected_signature,
            "b0_status": "complete",
            "sample_id": str(sample["id"]),
            "dataset": str(sample["dataset"]),
            "source_group_id": str(sample["_cad_source_group_id"]),
            "split": str(sample.get("_cad_split", "unknown")),
            "prompt_tokens": int(reference_trace["input_ids"].shape[1]),
            "prompt_hash": reference_trace["layout"].prompt_hash,
            "ar_ids": reference_trace["ar_ids"],
            "native_ids": reference_trace["native_ids"],
            "adapter_ids": reference_trace["adapter_ids"],
            "adapter_rounds": reference_trace["adapter_rounds"],
            "eos_ids": sorted(reference_trace["eos_ids"]),
            "parity_by_gamma": parity_by_gamma,
        }
        atomic_json(path, trace)
        validation_rows.append({"sample_id": sample["id"], "status": "complete", "comparisons_by_gamma": parity_by_gamma})
        del reference_trace, measured, ids
    comparisons = [
        comparison
        for sample in validation_rows
        for comparison in sample.get("comparisons_by_gamma", {}).values()
    ]
    valid_rows = [sample for sample in validation_rows if sample.get("status") == "complete"]
    failed_rows = [sample for sample in validation_rows if sample.get("status") != "complete"]
    completed = len(valid_rows) == len(samples) and all(
        len(sample.get("comparisons_by_gamma", {})) == len(gammas)
        for sample in valid_rows
    )
    summary = {
        "schema_version": "dflash-speculation-b.b0-validation.v2",
        "status": "complete" if completed else ("partial" if valid_rows else "failed"),
        "sample_count": len(validation_rows),
        "valid_sample_count": len(valid_rows),
        "failed_sample_count": len(failed_rows),
        "failed_samples": failed_rows,
        "gamma_values": list(gammas),
        "comparisons": validation_rows,
        "ar_exact_match_rate": {
            "native": (sum(bool(row.get("native_equals_ar")) for row in comparisons) / len(comparisons)) if comparisons else None,
            "adapter": (sum(bool(row.get("adapter_equals_ar")) for row in comparisons) / len(comparisons)) if comparisons else None,
        },
        "native_adapter_exact_match_rate": (
            sum(bool(row.get("native_equals_adapter")) for row in comparisons) / len(comparisons)
            if comparisons else None
        ),
        "comparison_role": "diagnostic_only",
        "runtime": runtime_signature,
    }
    atomic_json(run_root / "evidence" / "parity.json", summary)
    print(
        f"[B0] validation {summary['status']}; valid={len(valid_rows)}/{len(samples)} "
        f"failed={len(failed_rows)}; AR/native/adapter equality is diagnostic only; "
        f"AR match rates={summary['ar_exact_match_rate']}",
        flush=True,
    )
    # A partial B0 run can still provide valid independent samples to B1/B1-V.
    # False means there is no usable reference trajectory at all.
    return bool(valid_rows)


def _state_cell_path(run_root: Path, state_key: str, *, phase: str = "b1") -> Path:
    return run_root / phase / "cells" / f"{hashlib.sha256(state_key.encode('utf-8')).hexdigest()}.json"


def _mark_sample_setup_recovered(
    run_root: Path,
    phase: str,
    sample: Mapping[str, Any],
) -> None:
    state_key = f"sample|{sample['dataset']}|{sample['id']}|setup"
    path = _state_cell_path(run_root, state_key, phase=phase)
    if not path.is_file():
        return
    cell = _read_json_artifact(run_root, phase, path)
    if cell is None:
        return
    if cell.get("status") != "failed":
        return
    cell["status"] = "recovered"
    from TrainingFree.context_adaptive.schema import atomic_json

    atomic_json(path, cell)
    print(f"[{phase}] sample setup recovered: {sample['dataset']}:{sample['id']}", flush=True)


def _archive_stale_cell(run_root: Path, phase: str, path: Path, cell: Mapping[str, Any]) -> Path:
    stale_dir = run_root / "stale" / phase
    stale_dir.mkdir(parents=True, exist_ok=True)
    old_signature = str(cell.get("campaign_signature", "unknown"))[:12]
    destination = stale_dir / f"{path.stem}.{old_signature}.{time.time_ns()}{path.suffix}"
    os.replace(path, destination)
    print(f"[{phase}] archived incompatible cell: {destination}", flush=True)
    return destination


def _archive_corrupt_artifact(run_root: Path, phase: str, path: Path, reason: BaseException) -> Path:
    stale_dir = run_root / "stale" / phase
    stale_dir.mkdir(parents=True, exist_ok=True)
    destination = stale_dir / f"{path.stem}.corrupt.{time.time_ns()}{path.suffix}"
    os.replace(path, destination)
    print(
        f"[{phase}] archived unreadable artifact and will rebuild it: {path.name} "
        f"({type(reason).__name__}: {reason})",
        file=sys.stderr,
        flush=True,
    )
    return destination


def _read_json_artifact(run_root: Path, phase: str, path: Path) -> dict[str, Any] | None:
    """Read a JSON object, quarantining a torn/corrupt checkpoint for recompute."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _archive_corrupt_artifact(run_root, phase, path, exc)
        return None
    if not isinstance(value, dict):
        _archive_corrupt_artifact(
            run_root, phase, path, ValueError("checkpoint root must be a JSON object")
        )
        return None
    return value


def _retain_compatible_cells(
    run_root: Path,
    phase: str,
    *,
    is_complete: Any,
) -> None:
    """Keep complete cells whose recorded measurements match the protocol shape."""
    for path in sorted((run_root / phase / "cells").glob("*.json")):
        cell = _read_json_artifact(run_root, phase, path)
        if cell is None:
            continue
        if cell.get("status", "complete") == "complete" and not is_complete(cell):
            _archive_stale_cell(run_root, phase, path, cell)


def _complete_b1_cell(cell: Mapping[str, Any], gammas: Sequence[int], repetitions: int) -> bool:
    try:
        expected = {(int(gamma), repetition) for gamma in gammas for repetition in range(repetitions)}
        observed = {
            (int(row["gamma"]), int(row["repetition"]))
            for row in cell.get("actions", [])
        }
        return (
            len(cell.get("actions", [])) == len(expected)
            and observed == expected
            and set(cell.get("accepted_by_gamma", {})) == {str(gamma) for gamma in gammas}
            and set(cell.get("proposal_by_gamma", {})) == {str(gamma) for gamma in gammas}
            and (
                isinstance(cell.get("signal"), Mapping)
                or isinstance(cell.get("signal_error"), Mapping)
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _complete_b1v_cell(cell: Mapping[str, Any], repetitions: int) -> bool:
    try:
        expected = {(k, repetition) for k in (3, 7, 11, 15) for repetition in range(repetitions)}
        observed = {
            (int(row["k"]), int(row["repetition"]))
            for row in cell.get("actions", [])
        }
        return len(cell.get("actions", [])) == len(expected) and observed == expected
    except (KeyError, TypeError, ValueError):
        return False


def _b1(args: argparse.Namespace, samples: Sequence[dict[str, Any]], run_root: Path, runtime: Mapping[str, Any]) -> None:
    from TrainingFree.context_adaptive.prompt import prepare_prompt
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.context_interventions.replay import evaluate_replay_action, rebuild_replay_state
    from TrainingFree.context_interventions.resilience import run_compute_unit
    from TrainingFree.speculation_experiments.reference import (
        reference_history,
        reference_trajectory_ids,
        select_reference_prefixes,
    )
    from TrainingFree.speculation_experiments.replay import run_same_prefix_sweep

    parity_path = run_root / "evidence" / "parity.json"
    parity = _read_json_artifact(run_root, "evidence", parity_path) if parity_path.is_file() else None
    if not parity or parity.get("status") not in {"complete", "partial"}:
        raise RuntimeError("B1 requires at least one valid B0 reference trajectory")
    runtime_sig = runtime["b_campaign_signature"]
    if parity.get("runtime") != runtime_sig:
        raise RuntimeError("B1 runtime signature differs from the B0 runtime")
    params = {
        "gammas": _parse_gammas(args.gammas),
        "repetitions": args.repetitions,
        "selection_repetitions": args.selection_repetitions,
        "warmups": args.warmups,
        "max_prefixes": args.max_prefixes,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    campaign_signature = _hash_json({"runtime": runtime_sig, "campaign": params})
    _retain_compatible_cells(
        run_root,
        "b1",
        is_complete=lambda cell: _complete_b1_cell(cell, params["gammas"], args.repetitions),
    )
    for sample_index, sample in enumerate(samples, start=1):
        try:
            trace_path = _sample_trace_path(run_root, sample)
            if not trace_path.is_file():
                print(f"[B1 {sample_index}/{len(samples)}] skip {sample['dataset']}:{sample['id']}; B0 has no valid trace", flush=True)
                continue
            trace = _read_json_artifact(run_root, "traces", trace_path)
            if trace is None:
                print(f"[B1 {sample_index}/{len(samples)}] no readable B0 trace; run validation to rebuild", flush=True)
                continue
            if trace.get("b0_status") != "complete":
                print(f"[B1 {sample_index}/{len(samples)}] skip {sample['dataset']}:{sample['id']}; B0 trace incomplete", flush=True)
                continue
            if not _complete_b0_trace(trace, params["gammas"]):
                print(f"[B1 {sample_index}/{len(samples)}] skip malformed B0 trace; run validation to rebuild", flush=True)
                continue
            reference_ids = reference_trajectory_ids(trace)
            (input_ids, layout), setup_error = run_compute_unit(
                lambda: prepare_prompt(sample, runtime["tokenizer"], chunk_size=128),
                label=f"B1 prompt setup sample={sample['id']}",
            )
            if setup_error is not None:
                raise _compute_failure_exception(setup_error)
            input_ids = input_ids.to(runtime["target"].device).contiguous()
            if layout.prompt_hash != trace["prompt_hash"]:
                raise ValueError(f"prompt tokenization changed for sample {sample['id']}")
            if not _b0_trace_matches_sample(
                trace, sample, prompt_hash=layout.prompt_hash, gammas=params["gammas"],
            ):
                raise ValueError(f"B0 trace identity/protocol differs for sample {sample['id']}")
            prefixes = select_reference_prefixes(
                trace["adapter_rounds"],
                reference_ids,
                prompt_length=int(trace["prompt_tokens"]),
                max_new_tokens=args.max_new_tokens,
                eos_ids={int(token) for token in trace.get("eos_ids", [])},
                max_prefixes=args.max_prefixes,
            )
            reference_rounds = {
                int(row["round_index"]): row
                for row in trace["adapter_rounds"]
                if row.get("type") == "round" and int(row.get("gamma_executed", 0)) == 15
            }
        except Exception as exc:
            try:
                del state
            except UnboundLocalError:
                pass
            try:
                del input_ids
            except UnboundLocalError:
                pass
            sample_state_key = f"sample|{sample['dataset']}|{sample['id']}|setup"
            sample_cell_path = _state_cell_path(run_root, sample_state_key)
            _record_cell_error(
                sample_cell_path, phase="B1", campaign_signature=campaign_signature,
                state_key=sample_state_key, sample_id=str(sample["id"]), exc=exc,
            )
            continue
        _mark_sample_setup_recovered(run_root, "b1", sample)
        if not prefixes:
            print(f"[B1 {sample_index}/{len(samples)}] no regular states for {sample['dataset']}:{sample['id']}", flush=True)
            del input_ids
            continue
        for prefix_index, prefix in enumerate(prefixes, start=1):
            state_key = f"{sample['dataset']}|{sample['_cad_source_group_id']}|{prefix.round_index}"
            cell_path = _state_cell_path(run_root, state_key)
            if cell_path.exists():
                cell = _read_json_artifact(run_root, "b1", cell_path)
                if cell is not None and cell.get("state_key") != state_key:
                    _archive_stale_cell(run_root, "b1", cell_path, cell)
                elif cell is not None and (
                    cell.get("status", "complete") == "complete"
                    and _complete_b1_cell(cell, params["gammas"], args.repetitions)
                ):
                    continue
                elif cell is not None and cell.get("status", "complete") == "complete":
                    _archive_stale_cell(run_root, "b1", cell_path, cell)
            try:
                reference_round = reference_rounds.get(prefix.round_index)
                if reference_round is None:
                    raise ValueError(f"reference round missing: {prefix.round_index}")
                state, compute_error = run_compute_unit(
                    lambda: rebuild_replay_state(
                        runtime,
                        input_ids,
                        reference_ids,
                        prefix,
                        layout,
                        max_new_tokens=args.max_new_tokens,
                    ),
                    label=f"B1 state rebuild {state_key}",
                )
                if compute_error is not None:
                    raise _compute_failure_exception(compute_error)
                # Rebuild and verify gamma15 before measuring the other shapes.
                full, compute_error = run_compute_unit(
                    lambda: evaluate_replay_action(
                        runtime,
                        input_ids,
                        state,
                        layout,
                        [tuple(range(state.bank.length)) for _ in runtime["draft"].layers],
                        requested_budget="full",
                        selector_id="B_full_context_gate",
                        remaining_output_tokens=prefix.remaining_output_tokens,
                        gamma=15,
                        cleanup_after_action=False,
                    ),
                    label=f"B1 gamma15 validation {state_key}",
                    result_error=lambda result: (
                        None if result.get("status") == "ok" else str(result.get("reason", "replay failed"))
                    ),
                )
                if compute_error is not None:
                    raise _compute_failure_exception(compute_error)
                history = reference_history(trace["adapter_rounds"], current_round=prefix.round_index)
                result, compute_error = run_compute_unit(
                    lambda: run_same_prefix_sweep(
                        runtime,
                        input_ids,
                        state,
                        layout,
                        dataset=str(sample["dataset"]),
                        source_group_id=str(sample["_cad_source_group_id"]),
                        sample_id=str(sample["id"]),
                        prompt_token_ids=[int(token) for token in input_ids[0].detach().cpu().tolist()],
                        max_new_tokens=args.max_new_tokens,
                        round_index=prefix.round_index,
                        gammas=params["gammas"],
                        repetitions=args.repetitions,
                        selection_repetitions=args.selection_repetitions,
                        warmups=args.warmups,
                        seed=args.seed + prefix.round_index,
                        collect_uncertainty=True,
                        history=history,
                    ),
                    label=f"B1 timing sweep {state_key}",
                )
                if compute_error is not None:
                    raise _compute_failure_exception(compute_error)
                signal = result.get("signal")
                if signal is not None:
                    signal["history"] = history
                    signal["history_ema"] = history.get("ema_acceptance")
                    signal["previous_accepted_candidates"] = history.get("previous_accepted_candidates")
                    signal["previous_all_accepted"] = history.get("previous_all_accepted")
                    signal["history_missing"] = bool(history.get("missing", True))
                cell = {
                    "schema_version": "dflash-speculation-b.cell.v1",
                    "status": "complete",
                    "campaign_signature": campaign_signature,
                    "state_key": state_key,
                    "state_id": result["state_id"],
                    "actions": result["actions"],
                    "signal": signal,
                    "signal_error": result.get("signal_error"),
                    "schedule": result["schedule"],
                    "warmups": result["warmups"],
                    "accepted_by_gamma": result["accepted_by_gamma"],
                    "proposal_by_gamma": result["proposal_by_gamma"],
                    "bank_build_ms": state.bank_build_ms,
                    "reference_anchor_check": {
                        "reference_pending_anchor": int(state.pending_anchor),
                        "rebuilt_full_target_greedy_anchor": int(state.full_target_greedy_anchor),
                        "matches": bool(state.anchor_matches_full_target_greedy),
                        "role": "diagnostic_only",
                    },
                    "gamma15_replay_acceptance_matches_paired_sweep": (
                        int(full["accepted_candidates"]) == int(result["accepted_by_gamma"]["15"])
                    ),
                }
                atomic_json(cell_path, cell)
                print(
                    f"[B1 {sample_index}/{len(samples)} state {prefix_index}/{len(prefixes)}] "
                    f"{sample['dataset']}:{sample['id']} round={prefix.round_index} complete",
                    flush=True,
                )
                del state, full, result
            except Exception as exc:
                try:
                    del state
                except UnboundLocalError:
                    pass
                try:
                    del full
                except UnboundLocalError:
                    pass
                try:
                    del result
                except UnboundLocalError:
                    pass
                _record_cell_error(
                    cell_path, phase="B1", campaign_signature=campaign_signature,
                    state_key=state_key, sample_id=str(sample["id"]), exc=exc,
                )
        del input_ids
    cells = []
    for cell_path in sorted((run_root / "b1" / "cells").glob("*.json")):
        cell = _read_json_artifact(run_root, "b1", cell_path)
        if cell is None:
            continue
        if cell.get("status", "complete") == "complete" and _complete_b1_cell(cell, params["gammas"], args.repetitions):
            cells.append(cell)
    actions = [row for cell in cells for row in cell.get("actions", [])]
    signals = [cell["signal"] for cell in cells if cell.get("signal") is not None]
    failed_count = 0
    for path in sorted((run_root / "b1" / "cells").glob("*.json")):
        cell = _read_json_artifact(run_root, "b1", path)
        failed_count += int(cell is not None and cell.get("status") == "failed")
    _write_jsonl(run_root / "b1" / "draft_shape.jsonl", actions, {"type": "summary", "state_count": len(cells), "failed_state_count": failed_count, "action_count": len(actions)})
    signal_failure_count = sum(cell.get("signal") is None for cell in cells)
    _write_jsonl(run_root / "b2" / "signals.jsonl", signals, {
        "type": "summary",
        "state_count": len(cells),
        "available_signal_count": len(signals),
        "failed_signal_count": signal_failure_count,
        "failed_state_count": failed_count,
    })
    print(f"[B1] collected states={len(cells)} failed={failed_count}; continuing", flush=True)
    return {
        "status": "partial" if failed_count else "complete",
        "state_count": len(cells),
        "failed_state_count": failed_count,
        "action_count": len(actions),
        "signal_count": len(signals),
    }


def _b1v(args: argparse.Namespace, samples: Sequence[dict[str, Any]], run_root: Path, runtime: Mapping[str, Any]) -> None:
    from TrainingFree.context_adaptive.prompt import prepare_prompt
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.context_interventions.replay import rebuild_replay_state
    from TrainingFree.context_interventions.resilience import run_compute_unit
    from TrainingFree.speculation_experiments.reference import reference_trajectory_ids, select_reference_prefixes
    from TrainingFree.speculation_experiments.replay import run_verification_prefix_sweep

    parity_path = run_root / "evidence" / "parity.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8")) if parity_path.is_file() else {}
    runtime_sig = runtime["b_campaign_signature"]
    if not isinstance(parity, Mapping) or parity.get("status") not in {"complete", "partial"}:
        raise RuntimeError("B1-V requires at least one valid B0 reference trajectory")
    if parity.get("runtime") != runtime_sig:
        raise RuntimeError("B1-V runtime signature differs from the B0 runtime")
    b1_cells = []
    for path in sorted((run_root / "b1" / "cells").glob("*.json")):
        cell = _read_json_artifact(run_root, "b1", path)
        if cell is None:
            continue
        if (
            cell.get("status", "complete") == "complete"
            and _complete_b1_cell(cell, _parse_gammas(args.gammas), args.repetitions)
        ):
            b1_cells.append(cell)
    proposal_by_state = {
        str(cell["state_id"]): tuple(int(token) for token in cell["proposal_by_gamma"]["15"])
        for cell in b1_cells
    }
    accepted_by_state = {
        str(cell["state_id"]): int(cell["accepted_by_gamma"]["15"])
        for cell in b1_cells
    }
    print(f"[B1-V] compatible complete B1 cells={len(b1_cells)}; code hashes are not checked", flush=True)
    if not b1_cells:
        _write_jsonl(run_root / "b1v" / "verify_prefix.jsonl", [], {
            "type": "summary", "status": "skipped_no_complete_b1_cells",
            "state_count": 0, "failed_state_count": 0, "action_count": 0,
        })
        print("[B1-V] no complete protocol-compatible B1 cells; saved empty coverage artifact", flush=True)
        return {
            "status": "failed",
            "reason": "no_complete_b1_cells",
            "current_paired_state_count": 0,
        }
    params = {
        "verify_prefixes": (3, 7, 11, 15),
        "repetitions": args.repetitions,
        "selection_repetitions": args.selection_repetitions,
        "warmups": args.warmups,
        "max_prefixes": args.max_prefixes,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    signature = _hash_json({"runtime": runtime_sig, "campaign": params})
    _retain_compatible_cells(
        run_root,
        "b1v",
        is_complete=lambda cell: _complete_b1v_cell(cell, args.repetitions),
    )
    for sample_index, sample in enumerate(samples, start=1):
        try:
            trace_path = _sample_trace_path(run_root, sample)
            if not trace_path.is_file():
                print(f"[B1-V {sample_index}/{len(samples)}] skip {sample['dataset']}:{sample['id']}; B0 has no valid trace", flush=True)
                continue
            trace = _read_json_artifact(run_root, "traces", trace_path)
            if trace is None:
                print(f"[B1-V {sample_index}/{len(samples)}] no readable B0 trace; run validation to rebuild", flush=True)
                continue
            if trace.get("b0_status") != "complete":
                print(f"[B1-V {sample_index}/{len(samples)}] skip {sample['dataset']}:{sample['id']}; B0 trace incomplete", flush=True)
                continue
            if not _complete_b0_trace(trace, (3, 7, 11, 15)):
                print(f"[B1-V {sample_index}/{len(samples)}] skip malformed B0 trace; run validation to rebuild", flush=True)
                continue
            reference_ids = reference_trajectory_ids(trace)
            (ids, layout), setup_error = run_compute_unit(
                lambda: prepare_prompt(sample, runtime["tokenizer"], chunk_size=128),
                label=f"B1-V prompt setup sample={sample['id']}",
            )
            if setup_error is not None:
                raise _compute_failure_exception(setup_error)
            ids = ids.to(runtime["target"].device).contiguous()
            if layout.prompt_hash != trace["prompt_hash"]:
                raise ValueError(f"prompt tokenization changed for sample {sample['id']}")
            if not _b0_trace_matches_sample(
                trace, sample, prompt_hash=layout.prompt_hash, gammas=(3, 7, 11, 15),
            ):
                raise ValueError(f"B0 trace identity/protocol differs for sample {sample['id']}")
            prefixes = select_reference_prefixes(
                trace["adapter_rounds"], reference_ids,
                prompt_length=int(trace["prompt_tokens"]), max_new_tokens=args.max_new_tokens,
                eos_ids={int(token) for token in trace.get("eos_ids", [])}, max_prefixes=args.max_prefixes,
            )
            reference_rounds = {
                int(row["round_index"]): row for row in trace["adapter_rounds"]
                if row.get("type") == "round" and int(row.get("gamma_executed", 0)) == 15
            }
        except Exception as exc:
            try:
                del state
            except UnboundLocalError:
                pass
            try:
                del ids
            except UnboundLocalError:
                pass
            sample_state_key = f"sample|{sample['dataset']}|{sample['id']}|setup"
            sample_cell_path = _state_cell_path(run_root, sample_state_key, phase="b1v")
            _record_cell_error(
                sample_cell_path, phase="B1-V", campaign_signature=signature,
                state_key=sample_state_key, sample_id=str(sample["id"]), exc=exc,
            )
            continue
        _mark_sample_setup_recovered(run_root, "b1v", sample)
        for prefix_index, prefix in enumerate(prefixes, start=1):
            state_key = f"{sample['dataset']}|{sample['_cad_source_group_id']}|{prefix.round_index}"
            cell_path = run_root / "b1v" / "cells" / f"{hashlib.sha256(state_key.encode('utf-8')).hexdigest()}.json"
            if cell_path.exists():
                existing = _read_json_artifact(run_root, "b1v", cell_path)
                if existing is not None and existing.get("state_key") not in {None, state_key}:
                    _archive_stale_cell(run_root, "b1v", cell_path, existing)
                elif existing is not None and (
                    existing.get("status", "complete") == "complete"
                    and _complete_b1v_cell(existing, args.repetitions)
                ):
                    continue
                elif existing is not None and existing.get("status", "complete") == "complete":
                    _archive_stale_cell(run_root, "b1v", cell_path, existing)
            try:
                reference_round = reference_rounds.get(prefix.round_index)
                if reference_round is None:
                    raise ValueError(f"reference round missing: {prefix.round_index}")
                state, compute_error = run_compute_unit(
                    lambda: rebuild_replay_state(
                        runtime, ids, reference_ids, prefix, layout,
                        max_new_tokens=args.max_new_tokens,
                    ),
                    label=f"B1-V state rebuild {state_key}",
                )
                if compute_error is not None:
                    raise _compute_failure_exception(compute_error)
                prompt_ids = [int(token) for token in ids[0].detach().cpu().tolist()]
                # B1 and B1-V share an action-independent state key.
                from TrainingFree.speculation_experiments.protocol import state_identity

                state_identity_value = state_identity(
                    dataset=str(sample["dataset"]),
                    source_group_id=str(sample["_cad_source_group_id"]),
                    prompt_token_ids=prompt_ids,
                    output_prefix_ids=state.output_ids,
                    pending_anchor=int(state.pending_anchor),
                    anchor_position=len(prompt_ids) + len(state.output_ids),
                    round_index=prefix.round_index,
                )
                if state_identity_value not in proposal_by_state:
                    print(
                        f"[B1-V {sample_index}/{len(samples)}] skip state={state_key}; "
                        "paired B1 measurement unavailable",
                        flush=True,
                    )
                    del state
                    continue
                result, compute_error = run_compute_unit(
                    lambda: run_verification_prefix_sweep(
                        runtime,
                        ids,
                        state,
                        layout,
                        dataset=str(sample["dataset"]),
                        source_group_id=str(sample["_cad_source_group_id"]),
                        sample_id=str(sample["id"]),
                        prompt_token_ids=prompt_ids,
                        reference_output_ids=reference_ids,
                        max_new_tokens=args.max_new_tokens,
                        round_index=prefix.round_index,
                        full_accepted=accepted_by_state[state_identity_value],
                        repetitions=args.repetitions,
                        selection_repetitions=args.selection_repetitions,
                        warmups=args.warmups,
                        seed=args.seed + prefix.round_index,
                        expected_proposal_ids=proposal_by_state[state_identity_value],
                    ),
                    label=f"B1-V timing sweep {state_key}",
                )
                if compute_error is not None:
                    raise _compute_failure_exception(compute_error)
                gamma15_rows = [row for row in result["actions"] if int(row["k"]) == 15]
                gamma15_observed = sorted({int(row["accepted_candidates"]) for row in gamma15_rows})
                atomic_json(cell_path, {
                    "schema_version": "dflash-speculation-b.b1v-cell.v1",
                    "status": "complete",
                    "campaign_signature": signature,
                    "state_key": state_key,
                    "state_id": result["state_id"],
                    "actions": result["actions"],
                    "schedule": result["schedule"],
                    "proposal_token_ids": result["proposal_token_ids"],
                    "gamma15_acceptance_check": {
                        "reference_accepted_candidates": accepted_by_state[state_identity_value],
                        "observed_repetitions": gamma15_observed,
                        "matches": gamma15_observed == [accepted_by_state[state_identity_value]],
                        "role": "diagnostic_only",
                    },
                    "reference_anchor_check": {
                        "reference_pending_anchor": int(state.pending_anchor),
                        "rebuilt_full_target_greedy_anchor": int(state.full_target_greedy_anchor),
                        "matches": bool(state.anchor_matches_full_target_greedy),
                        "role": "diagnostic_only",
                    },
                })
                print(
                    f"[B1-V {sample_index}/{len(samples)} state {prefix_index}/{len(prefixes)}] "
                    f"{sample['dataset']}:{sample['id']} round={prefix.round_index} complete",
                    flush=True,
                )
                del state, result
            except Exception as exc:
                try:
                    del state
                except UnboundLocalError:
                    pass
                try:
                    del result
                except UnboundLocalError:
                    pass
                _record_cell_error(
                    cell_path, phase="B1-V", campaign_signature=signature,
                    state_key=state_key, sample_id=str(sample["id"]), exc=exc,
                )
        del ids
    valid_cells = []
    failed_count = 0
    for path in sorted((run_root / "b1v" / "cells").glob("*.json")):
        cell = _read_json_artifact(run_root, "b1v", path)
        if cell is None:
            failed_count += 1
            continue
        if cell.get("status", "complete") == "complete" and _complete_b1v_cell(cell, args.repetitions):
            valid_cells.append(cell)
        elif cell.get("status") == "failed":
            failed_count += 1
    actions = [row for cell in valid_cells for row in cell.get("actions", [])]
    _write_jsonl(run_root / "b1v" / "verify_prefix.jsonl", actions, {
        "type": "summary", "state_count": len(valid_cells), "failed_state_count": failed_count,
        "action_count": len(actions),
    })
    print(f"[B1-V] collected states={len(valid_cells)} failed={failed_count}; continuing", flush=True)
    return {
        "status": "partial" if failed_count else "complete",
        "state_count": len(valid_cells),
        "failed_state_count": failed_count,
        "action_count": len(actions),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    from common.io_util import JsonlWriter

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    writer = JsonlWriter(temporary)
    for row in rows:
        writer.add(dict(row))
    writer.finalize(dict(summary))
    temporary.replace(path)


def _sample_cell_path(root: Path, phase: str, sample: Mapping[str, Any]) -> Path:
    identity = f"{sample['dataset']}::{sample['id']}"
    name = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return root / "cells" / phase / f"{name}.json"


def _load_resume_cell(
    path: Path,
    *,
    resume: bool,
    run_root: Path,
    phase: str,
    is_compatible: Any,
) -> dict[str, Any] | None:
    """Reuse a complete protocol-compatible cell; source hashes are provenance only."""
    if not path.exists():
        return None
    del resume
    cell = _read_json_artifact(run_root, phase, path)
    if cell is None:
        return None
    if cell.get("status", "complete") != "complete":
        return None
    if not is_compatible(cell):
        _archive_stale_cell(run_root, phase, path, cell)
        return None
    return cell


def _request_cell_matches_protocol(
    cell: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    variants: Sequence[str],
    repetitions: int,
) -> bool:
    """Check a saved E2E cell by its actual request coverage, not source hash."""
    rows = cell.get("requests")
    if not isinstance(rows, list):
        return False
    expected = {(str(variant), repetition) for variant in variants for repetition in range(repetitions)}
    observed: set[tuple[str, int]] = set()
    try:
        for row in rows:
            pair = (str(row["variant"]), int(row["repetition"]))
            if pair in observed:
                return False
            if (
                str(row.get("sample_id")) != str(sample["id"])
                or str(row.get("dataset")) != str(sample["dataset"])
                or str(row.get("source_group_id")) != str(sample["_cad_source_group_id"])
            ):
                return False
            observed.add(pair)
    except (KeyError, TypeError, ValueError):
        return False
    return len(rows) == len(expected) and observed == expected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "summary":
                rows.append(row)
    return rows


def _analyze(args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.speculation_experiments.analysis import analyze_draft_shape
    from TrainingFree.speculation_experiments.report import make_decision_summary
    from TrainingFree.speculation_experiments.survival import cross_validated_survival

    action_path = run_root / "b1" / "draft_shape.jsonl"
    signal_path = run_root / "b2" / "signals.jsonl"
    if not action_path.is_file() or not signal_path.is_file():
        raise FileNotFoundError("B1/B2 action and signal streams are required; run --mode b1 first")
    b1_rows = _read_jsonl(action_path)
    signal_rows = _read_jsonl(signal_path)
    campaign_path = run_root / "campaign.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8")) if campaign_path.is_file() else {}
    n_splits = int(campaign.get("n_splits", args.n_splits))
    seed = int(campaign.get("seed", args.seed))
    bootstrap_replicates = int(campaign.get("bootstrap_replicates", args.bootstrap_replicates))
    thresholds = campaign.get("thresholds") or {}
    b1 = analyze_draft_shape(
        b1_rows,
        bootstrap_replicates=bootstrap_replicates,
        seed=20261013,
    )
    b2 = cross_validated_survival(
        signal_rows,
        n_splits=n_splits,
        seed=seed,
        variants=(
            "R",
            "R-H",
            "R-C",
            "R-A",
            "R-HA",
            "R-PreAll",
            "R-DH",
            "R-DC",
            "R-A-DH",
            "R-A-DC",
            "R-DAll",
            "R-DAll-noDH",
            "R-Hazard-Pos",
            "R-Hazard-Present",
            "R-Hazard-Prefix",
            "R-Hazard-Present-Future",
            "R-Hazard-Suffix-Margin",
            "R-Hazard-Suffix-Entropy",
        ),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=20261013,
    )
    decisions = make_decision_summary(
        b1,
        b2,
        b1_positive_gap=float(thresholds.get("b1_positive_gap", 0.05)),
        b1_practical_null=float(thresholds.get("b1_practical_null", 0.02)),
        b2_positive_skill=float(thresholds.get("b2_positive_skill", 0.05)),
        b2_incremental_skill=float(thresholds.get("b2_incremental_skill", 0.02)),
    )
    parity_path = run_root / "evidence" / "parity.json"
    parity = json.loads(parity_path.read_text(encoding="utf-8")) if parity_path.is_file() else {}

    def phase_cell_counts(phase: str) -> dict[str, int]:
        counts = {"complete": 0, "failed": 0}
        for path in sorted((run_root / phase / "cells").glob("*.json")):
            cell = json.loads(path.read_text(encoding="utf-8"))
            status = str(cell.get("status", "complete"))
            if status in counts:
                counts[status] += 1
        return counts

    b1_coverage = phase_cell_counts("b1")
    b1v_coverage = phase_cell_counts("b1v")
    coverage_status = (
        "partial"
        if parity.get("status") != "complete" or b1_coverage["failed"] or b1v_coverage["failed"]
        else "complete"
    )
    coverage = {
        "status": coverage_status,
        "b0_valid_samples": int(parity.get("valid_sample_count", 0)),
        "b0_sample_count": int(parity.get("sample_count", 0)),
        "b1_states": b1_coverage,
        "b1v_states": b1v_coverage,
    }
    b1["collection_coverage"] = coverage
    b2["collection_coverage"] = coverage
    if coverage_status == "partial":
        decisions["evidence_status"] = "partial_coverage_descriptive_only"
    else:
        decisions["evidence_status"] = "complete_coverage"
    reports = run_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    b1v_path = run_root / "b1v" / "verify_prefix.jsonl"
    b1v = None
    if b1v_path.is_file():
        b1v_rows = _read_jsonl(b1v_path)
        adapted_rows = [
            {
                **row,
                "type": "draft_shape_action",
                "gamma": int(row["k"]),
                "round_ms": float(row["total_round_ms"]),
            }
            for row in b1v_rows
        ]
        b1v = analyze_draft_shape(
            adapted_rows,
            bootstrap_replicates=bootstrap_replicates,
            seed=20261013,
        )
        atomic_json(reports / "b1v_metrics.json", b1v)
        decisions["b1v_verification_opportunity"] = (
            "supported"
            if b1v["evaluation"]["oracle_gap_j_ci95"]
            and b1v["evaluation"]["oracle_gap_j"] >= 0.05
            and b1v["evaluation"]["oracle_gap_j_ci95"][0] > 0.02
            else "low_opportunity"
            if b1v["evaluation"]["oracle_gap_j_ci95"]
            and b1v["evaluation"]["oracle_gap_j_ci95"][1] <= 0.02
            else "inconclusive"
        )
        if decisions["b1v_verification_opportunity"] == "supported":
            if decisions["b2_drafter_entropy"] == "predictive" or decisions["b2_drafter_confidence"] == "predictive":
                decisions["next_experiment"] = "evaluate_k_oracle_regret_before_B3"
            else:
                decisions["next_experiment"] = "keep_fixed_k_no_predictive_signal"
        elif decisions["b1v_verification_opportunity"] == "low_opportunity":
            decisions["next_experiment"] = "keep_fixed_k_report_low_opportunity"
        else:
            decisions["next_experiment"] = "collect_more_groups_for_B1V"
    atomic_json(reports / "b1_metrics.json", b1)
    atomic_json(reports / "b2_metrics.json", b2)
    atomic_json(reports / "decision.json", decisions)
    _write_jsonl(reports / "oof_predictions.jsonl", b2["oof_predictions"], {
        "type": "summary", "state_count": b2["state_count"], "variants": b2["variants"]
    })
    decision_lines = [
        "# Báo cáo quyết định B1/B2",
        "",
        f"- B1 opportunity: **{decisions['b1_opportunity']}**; oracle gap J={decisions['b1_oracle_gap_j']}; CI95={decisions['b1_oracle_gap_j_ci95']}.",
        f"- B2 DFlash entropy: **{decisions['b2_drafter_entropy']}**; Brier skill vs R={decisions['b2_entropy_skill_vs_R']}; CI95={decisions['b2_entropy_skill_ci95']}.",
        f"- Coverage: **{coverage['status']}**; B0={coverage['b0_valid_samples']}/{coverage['b0_sample_count']} samples, B1 states={coverage['b1_states']}, B1-V states={coverage['b1v_states']}.",
        f"- Entropy bổ sung sau acceptance history: **{decisions['b2_entropy_incremental_after_history']}**; skill={decisions['b2_entropy_incremental_skill']}; CI95={decisions['b2_entropy_incremental_ci95']}.",
        f"- DFlash confidence: **{decisions['b2_drafter_confidence']}**.",
        *(
            [f"- B1-V verification-prefix opportunity: **{decisions['b1v_verification_opportunity']}**; oracle gap J={b1v['evaluation']['oracle_gap_j']}; CI95={b1v['evaluation']['oracle_gap_j_ci95']}."]
            if b1v is not None else ["- B1-V verification-prefix opportunity: chưa chạy."]
        ),
        f"- Bước kế tiếp theo ngưỡng đã khóa: **{decisions['next_experiment']}**.",
        "",
        "Các CI bootstrap theo source group/document; B1 oracle và best-fixed được chọn từ selection repetitions rồi đánh giá trên evaluation repetitions. Đây là diagnostic replay, chưa phải E2E speedup.",
    ]
    (reports / "decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")
    print(json.dumps(decisions, ensure_ascii=False, indent=2), flush=True)
    return decisions


def _analyze_without_stopping_collection(args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    """Keep completed raw measurements when a downstream statistic is unavailable."""
    from TrainingFree.context_adaptive.schema import atomic_json

    try:
        result = _analyze(args, run_root)
    except Exception as exc:
        failure = {
            "status": "unavailable",
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-12000:],
            "raw_artifacts_retained": True,
        }
        report_root = run_root / "reports"
        report_root.mkdir(parents=True, exist_ok=True)
        try:
            atomic_json(report_root / "analysis_status.json", failure)
        except Exception as write_error:
            print(
                f"[analyze] status checkpoint unavailable "
                f"({type(write_error).__name__}: {write_error}); raw artifacts are retained",
                file=sys.stderr,
                flush=True,
            )
        print(
            f"[analyze] metrics unavailable ({type(exc).__name__}: {exc}); "
            "raw experiment artifacts are retained and collection can finish",
            file=sys.stderr,
            flush=True,
        )
        return failure
    try:
        atomic_json(run_root / "reports" / "analysis_status.json", {"status": "complete"})
    except Exception as exc:
        print(
            f"[analyze] completed analysis but could not save optional status "
            f"({type(exc).__name__}: {exc})",
            file=sys.stderr,
            flush=True,
        )
    return result


def _write_run_status(run_root: Path) -> dict[str, Any]:
    from TrainingFree.context_adaptive.schema import atomic_json

    parity_path = run_root / "evidence" / "parity.json"
    parity = _read_json_artifact(run_root, "evidence", parity_path) if parity_path.is_file() else {}
    parity = parity or {}

    def cell_counts(phase: str) -> dict[str, int]:
        counts = {"complete": 0, "failed": 0}
        for path in sorted((run_root / phase / "cells").glob("*.json")):
            cell = _read_json_artifact(run_root, phase, path)
            if cell is None:
                counts["failed"] += 1
                continue
            status = str(cell.get("status", "complete"))
            if status in counts:
                counts[status] += 1
        return counts

    b1 = cell_counts("b1")
    b1v = cell_counts("b1v")
    b2_signal_failures = 0
    for path in sorted((run_root / "b1" / "cells").glob("*.json")):
        cell = _read_json_artifact(run_root, "b1", path)
        if cell is not None and cell.get("status", "complete") == "complete" and cell.get("signal") is None:
            b2_signal_failures += 1
    analysis_path = run_root / "reports" / "analysis_status.json"
    analysis = _read_json_artifact(run_root, "reports", analysis_path) if analysis_path.is_file() else {"status": "not_run"}
    analysis = analysis or {"status": "unavailable", "error": {"message": "analysis status artifact is unreadable"}}
    partial = (
        parity.get("status") != "complete"
        or b1["failed"] > 0
        or b1v["failed"] > 0
        or b2_signal_failures > 0
    )
    summary = {
        "schema_version": "dflash-speculation-b.run-status.v1",
        "collection_status": "partial" if partial else "complete",
        "analysis_status": analysis.get("status", "not_run"),
        "b0": {
            "status": parity.get("status", "missing"),
            "valid_samples": int(parity.get("valid_sample_count", 0)),
            "failed_samples": int(parity.get("failed_sample_count", 0)),
            "sample_count": int(parity.get("sample_count", 0)),
        },
        "b1_states": b1,
        "b2_signal_failures": b2_signal_failures,
        "b1v_states": b1v,
        "analysis_error": analysis.get("error"),
    }
    atomic_json(run_root / "run_status.json", summary)
    print(f"[run] saved status: {run_root / 'run_status.json'} ({summary['collection_status']})", flush=True)
    return summary


def _write_run_status_best_effort(run_root: Path) -> dict[str, Any] | None:
    """Do not turn finished measurements into a failed process for a report tail."""
    try:
        return _write_run_status(run_root)
    except Exception as exc:
        print(
            f"[run] could not write optional final run_status.json "
            f"({type(exc).__name__}: {exc}); collected artifacts remain available in {run_root}",
            file=sys.stderr,
            flush=True,
        )
        return None


def _b3_eligible_models(
    run_root: Path,
    *,
    allow_gated_b3: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Apply the preregistered opportunity and held-out prediction gates."""
    decision_path = run_root / "reports" / "decision.json"
    b2_path = run_root / "reports" / "b2_metrics.json"
    if not decision_path.is_file() or not b2_path.is_file():
        raise FileNotFoundError("B3 dev requires --mode analyze to finish first")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    b2 = json.loads(b2_path.read_text(encoding="utf-8"))
    if decision.get("b1v_verification_opportunity") != "supported" and not allow_gated_b3:
        raise RuntimeError(
            "B3 gate closed: B1-V must have oracle gap >=5% and CI95 lower bound >2%; "
            f"observed {decision.get('b1v_verification_opportunity')}"
        )
    eligible = []
    candidates = (
        "R-DH",
        "R-DC",
        "R-Hazard-Pos",
        "R-Hazard-Present",
        "R-Hazard-Prefix",
        "R-Hazard-Present-Future",
        "R-Hazard-Suffix-Margin",
        "R-Hazard-Suffix-Entropy",
    )
    for variant in candidates:
        skill = (b2.get("brier_skill_vs_R") or {}).get(variant)
        interval = (b2.get("brier_skill_ci95") or {}).get(variant)
        if (
            skill is not None
            and interval
            and float(skill) >= 0.05
            and float(interval[0]) > 0.0
        ):
            eligible.append(variant)
    if allow_gated_b3 and "R-Hazard-Pos" not in eligible:
        eligible.insert(0, "R-Hazard-Pos")
    if not eligible:
        raise RuntimeError(
            "B3 gate closed: no candidate model has held-out Brier skill >=5% "
            "with CI95 improvement above zero"
        )
    return eligible, {"decision": decision, "b2": b2}


def _validate_b0_signature(
    run_root: Path,
    *,
    runtime_signature: Mapping[str, Any],
) -> None:
    parity_path = run_root / "evidence" / "parity.json"
    if not parity_path.is_file():
        raise FileNotFoundError("B0 validation evidence is missing; run --mode validate first")
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    if parity.get("status") != "complete":
        raise RuntimeError("phase requires completed B0 runtime validation")
    if parity.get("runtime") != runtime_signature:
        raise ValueError("B0 validation runtime signature differs from this run; rerun validate")


def _validate_replay_cells(
    args: argparse.Namespace,
    run_root: Path,
    *,
    require_b1v: bool,
) -> None:
    b1_cells = list((run_root / "b1" / "cells").glob("*.json"))
    if not b1_cells:
        raise FileNotFoundError("B1 replay cells are missing; run --mode b1 first")
    complete_b1 = 0
    for path in b1_cells:
        cell = _read_json_artifact(run_root, "b1", path)
        if cell is None:
            continue
        if cell.get("status", "complete") != "complete":
            continue
        if not _complete_b1_cell(cell, _parse_gammas(args.gammas), args.repetitions):
            raise ValueError(f"B1 cell has an incomplete measurement grid: {path.name}")
        complete_b1 += 1
    if complete_b1 == 0:
        raise ValueError("B1 has no complete cells for the configured gamma/repetition grid")
    if not require_b1v:
        return
    b1v_cells = list((run_root / "b1v" / "cells").glob("*.json"))
    if not b1v_cells:
        raise FileNotFoundError("B1-V replay cells are missing; run --mode b1v first")
    complete_b1v = 0
    for path in b1v_cells:
        cell = _read_json_artifact(run_root, "b1v", path)
        if cell is None:
            continue
        if cell.get("status", "complete") != "complete":
            continue
        if not _complete_b1v_cell(cell, args.repetitions):
            raise ValueError(f"B1-V cell has an incomplete verification-prefix grid: {path.name}")
        complete_b1v += 1
    if complete_b1v == 0:
        raise ValueError("B1-V has no complete cells for the configured prefix/repetition grid")


def _b3_models_and_costs(
    run_root: Path,
    *,
    eligible: Sequence[str],
    minimum_repetitions: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    from TrainingFree.speculation_experiments.b_controller import context_cost_profile
    from TrainingFree.speculation_experiments.survival import fit_survival_curve, serialize_survival_curve

    signal_path = run_root / "b2" / "signals.jsonl"
    cost_path = run_root / "b1v" / "verify_prefix.jsonl"
    if not signal_path.is_file() or not cost_path.is_file():
        raise FileNotFoundError("B3 dev requires saved B2 signals and B1-V costs")
    signals = _read_jsonl(signal_path)
    costs = _read_jsonl(cost_path)
    if not signals or not costs:
        raise ValueError("B3 dev cannot fit from empty B2 signals or B1-V cost rows")
    all_variants = list(dict.fromkeys(("R", "R-A", *eligible)))
    models = {
        variant: serialize_survival_curve(fit_survival_curve(signals, variant=variant))
        for variant in all_variants
    }
    profiles = context_cost_profile(costs, minimum_repetitions=minimum_repetitions)
    if not profiles:
        raise ValueError("B3 has no B1-V context cost profile with enough selection repetitions")
    return models, profiles


def _b3_oof_models_by_group(
    run_root: Path,
    *,
    variants: Sequence[str],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Fit leave-one-source-group-out survival curves to prevent in-sample leakage."""
    from TrainingFree.speculation_experiments.survival import fit_survival_curve

    signal_path = run_root / "b2" / "signals.jsonl"
    if not signal_path.is_file():
        return {}
    signals = [
        row for row in _read_jsonl(signal_path)
        if row.get("type") == "candidate_signals"
    ]
    groups = sorted({str(row.get("source_group_id", "")) for row in signals if row.get("source_group_id")})
    if len(groups) < 2:
        return {}
    oof: dict[str, dict[str, dict[str, Any]]] = {}
    for group_id in groups:
        train_rows = [row for row in signals if str(row.get("source_group_id")) != group_id]
        if not train_rows:
            continue
        oof[group_id] = {
            variant: fit_survival_curve(train_rows, variant=variant)
            for variant in variants
        }
    return oof


def _macro_geomean_latency(rows: Sequence[Mapping[str, Any]], variant: str) -> float:
    by_document: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row.get("variant") != variant or row.get("status") != "ok":
            continue
        latency = float(row["e2e_ms"])
        if latency <= 0:
            continue
        key = (str(row["dataset"]), str(row["source_group_id"]))
        by_document.setdefault(key, []).append(math.log(latency))
    by_dataset: dict[str, list[float]] = {}
    for (dataset, _), logs in by_document.items():
        by_dataset.setdefault(dataset, []).append(sum(logs) / len(logs))
    if not by_dataset:
        return math.inf
    return math.exp(sum(sum(values) / len(values) for values in by_dataset.values()) / len(by_dataset))


def _paired_macro_speedup(
    rows: Sequence[Mapping[str, Any]],
    *,
    control: str,
    treatment: str,
    bootstrap_replicates: int = 2000,
    seed: int = 20261013,
) -> dict[str, Any]:
    import random

    paired: dict[str, dict[tuple[str, str], list[float]]] = {control: {}, treatment: {}}
    for row in rows:
        name = str(row.get("variant"))
        if name not in paired or row.get("status") != "ok":
            continue
        key = (str(row["dataset"]), str(row["source_group_id"]))
        paired[name].setdefault(key, []).append(float(row["e2e_ms"]))
    common = sorted(set(paired[control]) & set(paired[treatment]))
    doc_log_ratios: dict[str, dict[str, float]] = {}
    for dataset, group in common:
        left = paired[control][(dataset, group)]
        right = paired[treatment][(dataset, group)]
        if len(left) != len(right) or not left or any(x <= 0 for x in (*left, *right)):
            continue
        doc_log_ratios.setdefault(dataset, {})[group] = sum(
            math.log(a / b) for a, b in zip(left, right)
        ) / len(left)
    point = (
        math.exp(sum(sum(values.values()) / len(values) for values in doc_log_ratios.values()) / len(doc_log_ratios))
        if doc_log_ratios and all(doc_log_ratios.values()) else None
    )
    draws: list[float] = []
    if point is not None and bootstrap_replicates > 0:
        rng = random.Random(seed)
        datasets = sorted(doc_log_ratios)
        if all(len(doc_log_ratios[name]) >= 2 for name in datasets):
            for _ in range(bootstrap_replicates):
                sampled_dataset_means = []
                for dataset in datasets:
                    groups = list(doc_log_ratios[dataset].values())
                    sampled_dataset_means.append(sum(rng.choice(groups) for _ in groups) / len(groups))
                draws.append(math.exp(sum(sampled_dataset_means) / len(sampled_dataset_means)))
    ci = None
    if draws:
        draws.sort()
        ci = [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]
    return {
        "control": control,
        "treatment": treatment,
        "macro_geometric_speedup": point,
        "ci95_document_cluster": ci,
        "paired_documents": sum(len(values) for values in doc_log_ratios.values()),
        "per_dataset": {
            dataset: {
                "geometric_speedup": math.exp(sum(values.values()) / len(values)),
                "paired_documents": len(values),
            }
            for dataset, values in doc_log_ratios.items()
        },
    }


def _b3_overhead_adjusted_opportunity(
    rows: Sequence[Mapping[str, Any]],
    *,
    overhead_ms: float,
    bootstrap_replicates: int = 2000,
    seed: int = 20261013,
) -> dict[str, Any]:
    """Re-select k on dev selection repeats after charging measured B3 overhead."""
    import random

    if not math.isfinite(overhead_ms) or overhead_ms < 0:
        raise ValueError("overhead_ms must be finite and nonnegative")

    by_state: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("acceptance_identity_valid") is not True:
            continue
        state_id = str(row.get("state_id", ""))
        if not state_id:
            continue
        state = by_state.setdefault(state_id, {
            "dataset": str(row["dataset"]),
            "source_group_id": str(row["source_group_id"]),
            "selection": {},
            "evaluation": {},
            "selection_baseline": {},
            "evaluation_baseline": {},
        })
        prefix = int(row["k"])
        committed = 1 + int(row["accepted_candidates"])
        cost = (float(row["total_round_ms"]) + float(overhead_ms)) / committed
        split = str(row.get("repetition_split"))
        if split in {"selection", "evaluation"} and math.isfinite(cost) and cost > 0:
            state[split].setdefault(prefix, []).append(cost)
            # Fixed-k does not extract signals or run the adaptive controller.
            state[f"{split}_baseline"].setdefault(prefix, []).append(
                float(row["total_round_ms"]) / committed
            )
    complete = {
        state_id: state for state_id, state in by_state.items()
        if all(
            prefix in state[split]
            for split in ("selection", "evaluation")
            for prefix in (3, 7, 11, 15)
        )
    }
    if not complete:
        return {"status": "inconclusive", "state_count": 0, "overhead_ms": overhead_ms}
    selection_means = {
        state_id: {
            prefix: sum(values) / len(values)
            for prefix, values in state["selection"].items()
        }
        for state_id, state in complete.items()
    }
    by_dataset_fixed: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for state_id, scores in selection_means.items():
        state = complete[state_id]
        by_dataset_fixed[state["dataset"]][state["source_group_id"]]
        for prefix, values in state["selection_baseline"].items():
            value = mean(values)
            by_dataset_fixed[state["dataset"]][state["source_group_id"]][prefix].append(value)
    fixed_means: dict[int, float] = {}
    for prefix in (3, 7, 11, 15):
        dataset_means = []
        for docs in by_dataset_fixed.values():
            document_means = [mean(costs[prefix]) for costs in docs.values() if prefix in costs]
            if document_means:
                dataset_means.append(mean(document_means))
        fixed_means[prefix] = mean(dataset_means) if dataset_means else math.inf
    best_fixed = min(fixed_means, key=lambda prefix: (fixed_means[prefix], prefix))
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for state_id, state in complete.items():
        chosen = min(selection_means[state_id], key=lambda prefix: (selection_means[state_id][prefix], prefix))
        evaluation = {
            prefix: sum(values) / len(values)
            for prefix, values in state["evaluation"].items()
        }
        baseline_cost = mean(state["evaluation_baseline"][best_fixed])
        grouped[state["dataset"]][state["source_group_id"]].append((evaluation[chosen], baseline_cost))

    def macro_ratio(documents: Mapping[str, Mapping[str, Sequence[tuple[float, float]]]]) -> float | None:
        numerator = []
        denominator = []
        for docs in documents.values():
            doc_num = [mean(pair[0] for pair in values) for values in docs.values() if values]
            doc_den = [mean(pair[1] for pair in values) for values in docs.values() if values]
            if doc_num and doc_den:
                numerator.append(mean(doc_num))
                denominator.append(mean(doc_den))
        if not numerator or not denominator or mean(denominator) <= 0:
            return None
        return 1.0 - mean(numerator) / mean(denominator)

    gap = macro_ratio(grouped)
    draws = []
    if gap is not None and bootstrap_replicates and all(len(docs) >= 2 for docs in grouped.values()):
        rng = random.Random(seed)
        for _ in range(bootstrap_replicates):
            sampled: dict[str, dict[str, Sequence[tuple[float, float]]]] = {}
            for dataset, docs in grouped.items():
                keys = list(docs)
                sampled[dataset] = {f"sample-{index}": docs[rng.choice(keys)] for index in range(len(keys))}
            value = macro_ratio(sampled)
            if value is not None:
                draws.append(value)
    interval = None
    if draws:
        draws.sort()
        interval = [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]
    status = (
        "supported" if gap is not None and gap >= 0.05 and interval and interval[0] > 0.02
        else "low_opportunity" if gap is not None and interval and interval[1] <= 0.02
        else "inconclusive"
    )
    return {
        "status": status,
        "overhead_ms": float(overhead_ms),
        "baseline_overhead_ms": 0.0,
        "best_fixed_prefix": best_fixed,
        "oracle_gap_j": gap,
        "oracle_gap_j_ci95": interval,
        "state_count": len(complete),
        "source_group_count": len({str(row["source_group_id"]) for row in complete.values()}),
    }


def _b3_run_one(
    *,
    runtime: Mapping[str, Any],
    sample: Mapping[str, Any],
    args: argparse.Namespace,
    variant: str,
    policy: Any = None,
    draft_policy: Any = None,
    fixed_gamma: int = 15,
) -> dict[str, Any]:
    import torch
    from TrainingFree.context_adaptive.config import AdaptiveConfig
    from TrainingFree.context_adaptive.generation import generate_adaptive, generate_target_only
    from TrainingFree.context_adaptive.prompt import prepare_prompt
    from TrainingFree.speculation_experiments.protocol import SUPPORTED_GAMMAS

    ids, layout = prepare_prompt(sample, runtime["tokenizer"], chunk_size=128)
    ids = ids.to(runtime["target"].device).contiguous()
    torch.cuda.synchronize(ids.device)
    started = time.perf_counter()
    if variant == "ar":
        result = generate_target_only(runtime["target"], ids, max_new_tokens=args.max_new_tokens, temperature=0.0)
    else:
        config = AdaptiveConfig(
            budgets=("full",),
            gammas=SUPPORTED_GAMMAS,
            selector="recent_only",
            timing_mode="wall",
            output_root=str(_run_root(args) / "b3" / "runtime"),
            seed=args.seed,
        )
        mode = "dflash_full_fixed" if variant == "dflash_fixed15" else "fixed"
        result = generate_adaptive(
            runtime["target"], runtime["draft"], ids, runtime["tokenizer"], layout, config,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            variant=mode,
            fixed_budget="full",
            fixed_gamma=int(fixed_gamma),
            gamma_reference=int(fixed_gamma),
            run_id=str(args.run_id),
            sample_id=str(sample["id"]),
            dataset=str(sample["dataset"]),
            split=str(sample.get("_cad_split", "unknown")),
            repetition=int(getattr(args, "_b3_repetition", 0)),
            verification_prefix_policy=policy,
            draft_length_policy=draft_policy,
        )
    torch.cuda.synchronize(ids.device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    output_ids = [int(value) for value in result.output_ids.detach().cpu().tolist()]
    return {
        "status": result.status,
        "error": result.error,
        "input_tokens": int(ids.shape[1]),
        "output_ids": output_ids,
        "output_tokens": int(result.output_tokens),
        "e2e_ms": wall_ms,
        "reported_timings": result.timings,
        "counters": result.counters,
        "rounds": result.rounds,
        "correctness_status": result.correctness_status,
    }


def _sample_reference_text(sample: Mapping[str, Any]) -> str:
    raw = sample.get("raw") or {}
    value = (
        sample.get("reference")
        or sample.get("reference_output")
        or raw.get("reference_output")
    )
    if not value:
        answers = raw.get("answers") or sample.get("answers") or ()
        if isinstance(answers, Sequence) and not isinstance(answers, (str, bytes)) and answers:
            value = answers[0]
    return str(value or "")


def _output_quality_fields(
    sample: Mapping[str, Any], output_ids: Sequence[int], tokenizer: Any
) -> dict[str, Any]:
    generated_text = tokenizer.decode(list(output_ids), skip_special_tokens=True)
    reference_text = _sample_reference_text(sample)
    fields: dict[str, Any] = {
        "generated_text": generated_text,
        "reference_output": reference_text or None,
    }
    try:
        from common.rouge import add_rouge

        scores = add_rouge(fields, generated_text, reference_text)
        fields["rouge_status"] = "scored" if scores is not None else "no_reference"
    except Exception as exc:
        fields["rouge_status"] = f"error:{type(exc).__name__}:{str(exc)[:160]}"
    return fields


def _request_quality_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from common.rouge import aggregate_rouge

    summary: dict[str, Any] = {}
    variants = sorted({str(row.get("variant")) for row in rows})
    for variant in variants:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        exact = [row for row in variant_rows if row.get("exact_match_ar") is not None]
        summary[variant] = {
            "request_count": len(variant_rows),
            "ar_exact_match_rate": (
                sum(bool(row.get("exact_match_ar")) for row in exact) / len(exact)
                if exact else None
            ),
            "ar_token_match_role": "diagnostic_only",
            "rouge_scored_request_count": sum(row.get("rouge_status") == "scored" for row in variant_rows),
            "reference_missing_request_count": sum(row.get("rouge_status") == "no_reference" for row in variant_rows),
            "rouge_error_request_count": sum(
                str(row.get("rouge_status", "")).startswith("error:") for row in variant_rows
            ),
            **aggregate_rouge(variant_rows),
        }
    return summary


def _run_e2e_with_retry(operation: Any, *, label: str) -> dict[str, Any]:
    from TrainingFree.context_interventions.resilience import run_compute_unit

    result, error = run_compute_unit(
        operation,
        label=label,
        result_error=lambda item: (
            None if item.get("status") == "ok" else str(item.get("error") or item.get("status"))
        ),
    )
    if error is not None:
        return {
            "status": "error",
            "error": error["reason"],
            "input_tokens": 0,
            "output_ids": [],
            "output_tokens": 0,
            "e2e_ms": None,
            "reported_timings": {},
            "counters": {},
            "rounds": [],
            "correctness_status": "unavailable",
        }
    return result


def _b3_campaign(
    args: argparse.Namespace,
    samples: Sequence[dict[str, Any]],
    run_root: Path,
    runtime: Mapping[str, Any],
    *,
    confirm: bool,
) -> dict[str, Any]:
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.speculation_experiments.b_controller import (
        FixedVerificationPrefixController,
        VerificationPrefixController,
    )
    from TrainingFree.speculation_experiments.survival import deserialize_survival_curve

    phase = "confirm" if confirm else "dev"
    allow_gated = bool(getattr(args, "allow_gated_b3", False))
    if args.cohort_phase != phase and not (not confirm and allow_gated and args.cohort_phase == "pilot"):
        raise ValueError(f"--mode b3-{phase} requires --cohort-phase {phase}")
    runtime_signature = runtime["b_campaign_signature"]
    _validate_b0_signature(run_root, runtime_signature=runtime_signature)
    campaign = json.loads((run_root / "campaign.json").read_text(encoding="utf-8"))
    source_groups = sorted({str(row["_cad_source_group_id"]) for row in samples})

    oof_models_by_group: dict[str, dict[str, dict[str, Any]]] = {}
    if confirm:
        if not args.policy_lock:
            raise ValueError("--mode b3-confirm requires --policy-lock from the completed dev run")
        lock_path = _resolve(args.policy_lock) or Path(args.policy_lock)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("schema_version") != "dflash-speculation-b.b3-lock.v2":
            raise ValueError("unsupported B3 policy-lock schema")
        if lock.get("runtime_signature") != runtime_signature:
            raise ValueError("B3 policy lock runtime signature differs from confirm run")
        _validate_policy_lock(lock, args=args, campaign=campaign, phase="B3")
        overlap = sorted(set(source_groups) & set(lock.get("dev_source_groups", [])))
        if overlap:
            raise ValueError(f"B3 confirm source groups overlap dev lock: {overlap[:5]}")
        model_payloads = lock["models"]
        cost_profiles = lock["cost_profiles"]
        selected_model = str(lock["selected_policy_variant"])
        fixed_gamma_best = int(lock["best_fixed_gamma"])
        fixed_k_best = int(lock["best_fixed_k"])
        learned_overheads = lock.get("policy_overhead_ms", {})
        candidate_variants = ["R", "R-A", selected_model]
    else:
        _validate_replay_cells(
            args, run_root, require_b1v=True,
        )
        _analyze(args, run_root)
        eligible, _ = _b3_eligible_models(run_root, allow_gated_b3=allow_gated)
        model_payloads, cost_profiles = _b3_models_and_costs(
            run_root,
            eligible=eligible,
            minimum_repetitions=int(campaign.get("selection_repetitions", args.selection_repetitions)),
        )
        selected_model = ""
        fixed_gamma_best = 15
        fixed_k_best = 15
        learned_overheads: dict[str, float] = {}
        candidate_variants = list(dict.fromkeys(["R", "R-A", *eligible]))
        oof_models_by_group = _b3_oof_models_by_group(run_root, variants=candidate_variants)

    models = {name: deserialize_survival_curve(value) for name, value in model_payloads.items()}
    policies: dict[str, Any] = {}
    for name in candidate_variants:
        policy = VerificationPrefixController(
            model=models[name],
            cost_profiles=cost_profiles,
            dataset="",
            fallback_prefix=15,
            overhead_ms=float(learned_overheads.get(name, 0.0)),
            opportunity_threshold_eta=float(getattr(args, "opportunity_threshold_eta", 0.0)),
        )
        policies[name] = policy

    def policy_for(name: str, sample: Mapping[str, Any]) -> Any:
        policy = policies[name]
        policy.dataset = str(sample["dataset"])
        group_id = str(sample.get("_cad_source_group_id", ""))
        if group_id and group_id in oof_models_by_group and name in oof_models_by_group[group_id]:
            policy.model = oof_models_by_group[group_id][name]
        else:
            policy.model = models[name]
        return policy

    if confirm:
        measured_variants = [
            "ar", "dflash_fixed15", f"best_fixed_gamma_{fixed_gamma_best}",
            f"best_fixed_k_{fixed_k_best}", f"adaptive_{selected_model}",
        ]
    else:
        default_variants = [
            "ar", "dflash_fixed15",
            *[f"fixed_gamma_{gamma}" for gamma in (3, 7, 11, 15)],
            *[f"fixed_k_{prefix}" for prefix in (3, 7, 11, 15)],
            *[f"adaptive_{name}" for name in candidate_variants],
        ]
        raw_b3_variants = getattr(args, "b3_variants", None)
        if raw_b3_variants:
            requested = [item.strip() for item in str(raw_b3_variants).split(",") if item.strip()]
            measured_variants = [item for item in requested if item in set(default_variants)]
            if "ar" not in measured_variants:
                measured_variants.insert(0, "ar")
            if "dflash_fixed15" not in measured_variants:
                measured_variants.insert(1, "dflash_fixed15")
            candidate_variants = [
                name for name in candidate_variants
                if f"adaptive_{name}" in set(measured_variants)
            ]
        else:
            measured_variants = default_variants
    b3_root = run_root / "b3"
    b3_root.mkdir(parents=True, exist_ok=True)
    profile_signature = _hash_json({
        "phase": phase,
        "runtime_signature": runtime_signature,
        "campaign": _hash_json(campaign),
        "models": model_payloads,
        "cost_profiles": cost_profiles,
        "variants": measured_variants,
        "e2e_repetitions": args.e2e_repetitions,
        "e2e_warmups": args.e2e_warmups,
        "max_new_tokens": args.max_new_tokens,
        "policy_lock": _hash_json(lock) if confirm else None,
    })
    profile_path = b3_root / f"{phase}_profile.json"
    saved_profile: dict[str, Any] | None = None
    warmup_total_overhead_ms: dict[str, float] = {}
    if not confirm and profile_path.exists():
        saved_profile = _read_json_artifact(run_root, "b3", profile_path)
        if saved_profile is not None:
            raw_overheads = saved_profile.get("policy_overhead_ms")
            try:
                normalized_overheads = {
                    str(key): float(value) for key, value in raw_overheads.items()
                } if isinstance(raw_overheads, Mapping) else {}
            except (TypeError, ValueError):
                normalized_overheads = {}
            if (
                not set(candidate_variants).issubset(normalized_overheads)
                or any(not math.isfinite(value) or value < 0 for value in normalized_overheads.values())
            ):
                _archive_stale_cell(run_root, "b3", profile_path, saved_profile)
                saved_profile = None
            else:
                learned_overheads = normalized_overheads
                warmup_total_overhead_ms = {
                    key: float(value) for key, value in saved_profile.get("total_overhead_ms", {}).items()
                }
                for name, value in learned_overheads.items():
                    if name in policies:
                        policies[name].set_policy_overhead(value)

    def run_variant(sample: Mapping[str, Any], variant: str, repetition: int) -> dict[str, Any]:
        args._b3_repetition = repetition
        if variant == "ar":
            generation_variant, gamma, policy = "ar", 15, None
        elif variant == "dflash_fixed15":
            generation_variant, gamma, policy = variant, 15, None
        elif variant.startswith("fixed_gamma_"):
            generation_variant, gamma, policy = variant, int(variant.rsplit("_", 1)[1]), None
        elif variant.startswith("fixed_k_"):
            generation_variant, gamma = variant, 15
            policy = FixedVerificationPrefixController(int(variant.rsplit("_", 1)[1]))
        elif variant.startswith("adaptive_"):
            name = variant.removeprefix("adaptive_")
            generation_variant, gamma, policy = variant, 15, policy_for(name, sample)
        elif variant.startswith("best_fixed_gamma_"):
            generation_variant, gamma, policy = f"fixed_gamma_{fixed_gamma_best}", fixed_gamma_best, None
        elif variant.startswith("best_fixed_k_"):
            generation_variant, gamma = f"fixed_k_{fixed_k_best}", 15
            policy = FixedVerificationPrefixController(fixed_k_best)
        else:
            raise ValueError(f"unknown B3 variant {variant}")
        return _run_e2e_with_retry(
            lambda: _b3_run_one(
                runtime=runtime, sample=sample, args=args, variant=generation_variant,
                policy=policy, fixed_gamma=gamma,
            ),
            label=f"B3 {variant} sample={sample['id']} repetition={repetition}",
        )

    warmup_policy_ms: dict[str, list[float]] = {name: [] for name in candidate_variants}
    warmup_total_policy_ms: dict[str, list[float]] = {name: [] for name in candidate_variants}
    warmup_samples = list(samples[: min(args.e2e_warmups, len(samples))])
    warmup_failed_variants: list[str] = []
    for variant in measured_variants:
        if variant == "ar":
            warmup_name, policy_name = "ar", None
        elif variant.startswith("adaptive_"):
            policy_name = variant.removeprefix("adaptive_")
            warmup_name = variant
        elif variant.startswith("fixed_k_"):
            warmup_name, policy_name = variant, None
        elif variant.startswith("best_fixed_k_"):
            warmup_name, policy_name = f"fixed_k_{fixed_k_best}", None
        elif variant.startswith("best_fixed_gamma_"):
            warmup_name, policy_name = f"fixed_gamma_{fixed_gamma_best}", None
        else:
            warmup_name, policy_name = variant, None
        warmup_success_count = 0
        for warm_index, sample in enumerate(warmup_samples):
            warmed = run_variant(sample, warmup_name, -warm_index - 1)
            if warmed["status"] != "ok":
                print(
                    f"[B3-{phase}] warmup failed after retry: {warmup_name} {sample['id']}: {warmed['error']}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            warmup_success_count += 1
            if policy_name is not None:
                warmup_policy_ms[policy_name].extend(
                    float(row.get("verification_policy_ms", 0.0))
                    for row in warmed["rounds"]
                    if int(row.get("gamma_executed", 0)) > 0
                )
                warmup_total_policy_ms[policy_name].extend(
                    float(row.get("draft_signal_ms", 0.0) or 0.0)
                    + float(row.get("verification_policy_ms", 0.0) or 0.0)
                    for row in warmed["rounds"]
                    if int(row.get("gamma_executed", 0)) > 0
                )
        if warmup_success_count == 0:
            warmup_failed_variants.append(variant)
    if warmup_failed_variants:
        failure = {
            "schema_version": "dflash-speculation-b.b3-summary.v1",
            "phase": phase,
            "status": "warmup_incomplete",
            "failed_variants": warmup_failed_variants,
            "reason": "No warmup completed for these variants after bounded retries; latency measurements would be unreliable.",
        }
        atomic_json(b3_root / f"{phase}_summary.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return failure
    if not confirm and saved_profile is None:
        for name, values in warmup_policy_ms.items():
            if values:
                learned_overheads[name] = sum(values) / len(values)
                policies[name].set_policy_overhead(float(learned_overheads[name]))
        warmup_total_overhead_ms = {
            name: (sum(values) / len(values) if values else float(learned_overheads.get(name, 0.0)))
            for name, values in warmup_total_policy_ms.items()
        }
        atomic_json(profile_path, {
            "schema_version": "dflash-speculation-b.b3-profile.v1",
            "profile_signature": profile_signature,
            "policy_overhead_ms": learned_overheads,
            "total_overhead_ms": warmup_total_overhead_ms,
        })

    if not confirm:
        eligible_models, _ = _b3_eligible_models(run_root, allow_gated_b3=allow_gated)
        overhead_ms = max(
            (float(warmup_total_overhead_ms.get(name, learned_overheads.get(name, 0.0)))
             for name in eligible_models),
            default=0.0,
        )
        adjusted_gate = _b3_overhead_adjusted_opportunity(
            _read_jsonl(run_root / "b1v" / "verify_prefix.jsonl"),
            overhead_ms=overhead_ms,
            bootstrap_replicates=int(campaign.get("bootstrap_replicates", args.bootstrap_replicates)),
        )
        atomic_json(b3_root / "dev_overhead_gate.json", adjusted_gate)
        if adjusted_gate.get("status") != "supported" and not allow_gated:
            blocked = {
                "schema_version": "dflash-speculation-b.b3-summary.v1",
                "phase": "dev",
                "status": "b1v_overhead_gate_closed",
                "overhead_adjusted_b1v": adjusted_gate,
                "reason": "B1-V opportunity did not remain supported after measured extraction/controller overhead",
            }
            atomic_json(b3_root / "dev_summary.json", blocked)
            print(json.dumps(blocked, ensure_ascii=False, indent=2), flush=True)
            return blocked

    records: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    generation_failure = False
    rng = random.Random(args.seed)
    cell_signature = _hash_json({"profile_signature": profile_signature, "policy_overhead_ms": learned_overheads})
    cells_root = b3_root / "cells" / phase
    cells_root.mkdir(parents=True, exist_ok=True)
    for sample_index, sample in enumerate(samples, start=1):
        cell_path = _sample_cell_path(b3_root, phase, sample)
        saved_cell = _load_resume_cell(
            cell_path, resume=args.resume, run_root=run_root, phase=f"b3/cells/{phase}",
            is_compatible=lambda cell: _request_cell_matches_protocol(
                cell, sample, variants=measured_variants, repetitions=args.e2e_repetitions,
            ),
        )
        if saved_cell is not None:
            records.extend(saved_cell.get("requests", []))
            round_rows.extend(saved_cell.get("rounds", []))
            generation_failure = generation_failure or any(
                row.get("status") != "ok"
                for row in saved_cell.get("requests", []) if row.get("variant") != "ar"
            )
            print(f"[B3-{phase} {sample_index}/{len(samples)}] reuse {sample['dataset']}:{sample['id']}", flush=True)
            continue
        request_start = len(records)
        round_start = len(round_rows)
        sample_ar_failed = False
        for repetition in range(args.e2e_repetitions):
            ar = run_variant(sample, "ar", repetition)
            if ar["status"] != "ok":
                generation_failure = True
                sample_ar_failed = True
                print(
                    f"[B3-{phase}] AR failed after retry for {sample['dataset']}:{sample['id']} "
                    f"rep={repetition}; skipping this sample and continuing: {ar['error']}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            ar_record = {
                "schema_version": "dflash-speculation-b.b3-request.v1",
                "type": "request",
                "run_id": str(args.run_id),
                "phase": phase,
                "variant": "ar",
                "dataset": str(sample["dataset"]),
                "sample_id": str(sample["id"]),
                "source_group_id": str(sample["_cad_source_group_id"]),
                "repetition": repetition,
                "status": ar["status"],
                "error": ar["error"],
                "input_tokens": ar["input_tokens"],
                "output_tokens": ar["output_tokens"],
                "e2e_ms": ar["e2e_ms"],
                "reported_timings": ar["reported_timings"],
                "counters": ar["counters"],
                "correctness_status": ar["correctness_status"],
                "exact_match_ar": True,
                "output_ids": ar["output_ids"],
                "output_ids_sha256": _hash_json(ar["output_ids"]),
                "ar_output_ids_sha256": _hash_json(ar["output_ids"]),
            }
            ar_record.update(_output_quality_fields(sample, ar["output_ids"], runtime["tokenizer"]))
            records.append(ar_record)
            for round_row in ar["rounds"]:
                round_rows.append({
                    **round_row,
                    "run_id": str(args.run_id),
                    "phase": phase,
                    "variant": "ar",
                    "dataset": str(sample["dataset"]),
                    "sample_id": str(sample["id"]),
                    "source_group_id": str(sample["_cad_source_group_id"]),
                    "repetition": repetition,
                })
            order = [value for value in measured_variants if value != "ar"]
            rng.shuffle(order)
            for variant in order:
                measured = run_variant(sample, variant, repetition)
                exact = measured["output_ids"] == ar["output_ids"]
                record = {
                    "schema_version": "dflash-speculation-b.b3-request.v1",
                    "type": "request",
                    "run_id": str(args.run_id),
                    "phase": phase,
                    "variant": variant,
                    "dataset": str(sample["dataset"]),
                    "sample_id": str(sample["id"]),
                    "source_group_id": str(sample["_cad_source_group_id"]),
                    "repetition": repetition,
                    "status": measured["status"],
                    "error": measured["error"],
                    "input_tokens": measured["input_tokens"],
                    "output_tokens": measured["output_tokens"],
                    "e2e_ms": measured["e2e_ms"],
                    "reported_timings": measured["reported_timings"],
                    "counters": measured["counters"],
                    "correctness_status": measured["correctness_status"],
                    "exact_match_ar": exact,
                    "output_ids": measured["output_ids"],
                    "output_ids_sha256": _hash_json(measured["output_ids"]),
                    "ar_output_ids_sha256": _hash_json(ar["output_ids"]),
                }
                record.update(_output_quality_fields(sample, measured["output_ids"], runtime["tokenizer"]))
                records.append(record)
                for round_row in measured["rounds"]:
                    round_rows.append({
                        **round_row,
                        "run_id": str(args.run_id),
                        "phase": phase,
                        "variant": variant,
                        "dataset": str(sample["dataset"]),
                        "sample_id": str(sample["id"]),
                        "source_group_id": str(sample["_cad_source_group_id"]),
                        "repetition": repetition,
                    })
                if measured["status"] != "ok":
                    generation_failure = True
                    print(
                        f"[B3-{phase}] generation failure: {variant} {sample['dataset']}:{sample['id']} "
                        f"rep={repetition}; status={measured['status']}",
                        file=sys.stderr,
                        flush=True,
                    )
                elif not exact:
                    print(
                        f"[B3-{phase}] AR token mismatch recorded as quality diagnostic: "
                        f"{variant} {sample['dataset']}:{sample['id']} rep={repetition}",
                        flush=True,
                    )
        if sample_ar_failed:
            del records[request_start:]
            del round_rows[round_start:]
            atomic_json(cell_path, {
                "schema_version": "dflash-speculation-b.b3-sample-cell.v1",
                "status": "failed",
                "campaign_signature": cell_signature,
                "requests": [],
                "rounds": [],
                "error": {"type": "target_only_ar_failed", "message": str(ar.get("error"))},
            })
            continue
        atomic_json(cell_path, {
            "schema_version": "dflash-speculation-b.b3-sample-cell.v1",
            "status": "complete",
            "campaign_signature": cell_signature,
            "requests": records[request_start:],
            "rounds": round_rows[round_start:],
        })
        print(f"[B3-{phase} {sample_index}/{len(samples)}] {sample['dataset']}:{sample['id']} complete", flush=True)

    quality = _request_quality_summary(records)
    _write_jsonl(b3_root / f"{phase}_requests.jsonl", records, {
        "type": "summary", "phase": phase, "request_count": len(records),
        "repetitions": args.e2e_repetitions, "variants": measured_variants,
        "quality_by_variant": quality,
    })
    _write_jsonl(b3_root / f"{phase}_rounds.jsonl", round_rows, {
        "type": "summary", "phase": phase, "round_count": len(round_rows),
    })
    if not records:
        summary = {
            "schema_version": "dflash-speculation-b.b3-summary.v1",
            "phase": phase,
            "status": "no_valid_requests",
            "request_count": 0,
            "all_generations_ok": False,
            "reason": "Every sample failed target-only AR; artifacts record the failed samples and can be resumed.",
        }
        atomic_json(b3_root / f"{phase}_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    by_variant = sorted({str(row["variant"]) for row in records})
    latency = {name: _macro_geomean_latency(records, name) for name in by_variant}
    if confirm:
        comparisons = {
            control: _paired_macro_speedup(records, control=control, treatment=f"adaptive_{selected_model}")
            for control in (
                "dflash_fixed15",
                f"best_fixed_gamma_{fixed_gamma_best}",
                f"best_fixed_k_{fixed_k_best}",
            )
        }
        summary = {
            "schema_version": "dflash-speculation-b.b3-summary.v1",
            "phase": phase,
            "status": "generation_error" if generation_failure else "complete",
            "request_count": len(records),
            "all_generations_ok": not generation_failure,
            "quality_by_variant": quality,
            "macro_geomean_latency_ms": latency,
            "selected_policy_variant": selected_model,
            "comparisons": comparisons,
            "lock_path": str((_resolve(args.policy_lock) or Path(args.policy_lock)).resolve()),
        }
    else:
        gamma_candidates = [g for g in (3, 7, 11, 15) if f"fixed_gamma_{g}" in latency]
        fixed_gamma_best = min(gamma_candidates, key=lambda value: latency[f"fixed_gamma_{value}"]) if gamma_candidates else 15
        k_candidates = [k for k in (3, 7, 11, 15) if f"fixed_k_{k}" in latency]
        fixed_k_best = min(k_candidates, key=lambda value: latency[f"fixed_k_{value}"]) if k_candidates else 15
        adaptive_names = [name for name in candidate_variants if name not in {"R", "R-A"} and f"adaptive_{name}" in latency]
        if not adaptive_names:
            adaptive_names = [name for name in candidate_variants if f"adaptive_{name}" in latency]
        selected_model = min(adaptive_names, key=lambda name: latency[f"adaptive_{name}"]) if adaptive_names else "R-Hazard-Suffix-Margin"
        dev_comparisons = {
            f"{variant}_vs_dflash_fixed15": _paired_macro_speedup(
                records, control="dflash_fixed15", treatment=variant
            )
            for variant in by_variant if variant not in {"ar", "dflash_fixed15"}
        }
        dev_comparisons["dflash_fixed15_vs_ar"] = _paired_macro_speedup(
            records, control="ar", treatment="dflash_fixed15"
        )
        if f"adaptive_{selected_model}" in by_variant:
            dev_comparisons[f"adaptive_{selected_model}_vs_ar"] = _paired_macro_speedup(
                records, control="ar", treatment=f"adaptive_{selected_model}"
            )
        lock = {
            "schema_version": "dflash-speculation-b.b3-lock.v2",
            "dev_run_id": str(args.run_id),
            "runtime_signature": runtime_signature,
            "benchmark_signature": _locked_benchmark_signature(campaign),
            "protocol_parameters": _locked_e2e_protocol(args),
            "dev_source_groups": source_groups,
            "dev_campaign_signature": _hash_json(campaign),
            "selected_policy_variant": selected_model,
            "best_fixed_gamma": fixed_gamma_best,
            "best_fixed_k": fixed_k_best,
            "models": model_payloads,
            "cost_profiles": cost_profiles,
            "policy_overhead_ms": learned_overheads,
            "policy_parameters": {
                "allowed_prefixes": [3, 7, 11, 15],
                "fallback_prefix": 15,
                "cost_objective": "measured_ms_per_expected_committed_token",
                "overhead_includes": ["draft_uncertainty_extraction", "policy_callback"],
                "history_policy": "not_used; dev-history labels are reference-rollout outcomes",
            },
            "dev_macro_geomean_latency_ms": latency,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        lock_path = b3_root / "locked_config.json"
        if not generation_failure:
            atomic_json(lock_path, lock)
        summary = {
            "schema_version": "dflash-speculation-b.b3-summary.v1",
            "phase": phase,
            "status": "generation_error" if generation_failure else "locked_for_confirm",
            "request_count": len(records),
            "all_generations_ok": not generation_failure,
            "quality_by_variant": quality,
            "macro_geomean_latency_ms": latency,
            "best_fixed_gamma": fixed_gamma_best,
            "best_fixed_k": fixed_k_best,
            "selected_policy_variant": selected_model,
            "eligible_signal_models": list(adaptive_names),
            "comparisons": dev_comparisons,
            "policy_lock": str(lock_path.resolve()) if not generation_failure else None,
        }
    atomic_json(b3_root / f"{phase}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def _b4_rows_from_artifacts(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    from TrainingFree.speculation_experiments.lagged import pair_adjacent_round_signals

    signal_path = run_root / "b2" / "signals.jsonl"
    if not signal_path.is_file():
        raise FileNotFoundError("B4 requires B1/B2 candidate signals")
    signal_rows = _read_jsonl(signal_path)
    paired = pair_adjacent_round_signals(signal_rows)
    pair_index = {
        (str(row["dataset"]), str(row["sample_id"]), int(row["round_index"])): row
        for row in paired
    }
    action_path = run_root / "b1" / "draft_shape.jsonl"
    if not action_path.is_file():
        raise FileNotFoundError("B4 requires B1 draft_shape.jsonl")
    actions = _read_jsonl(action_path)
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for action in actions:
        by_state[str(action["state_id"])].append(action)
    output: list[dict[str, Any]] = []
    skipped_unpaired = 0
    skipped_incomplete = 0
    for state_id, state_actions in by_state.items():
        first = state_actions[0]
        key = (str(first["dataset"]), str(first["sample_id"]), int(first["round_index"]))
        reference = pair_index.get(key)
        if reference is None:
            skipped_unpaired += 1
            continue
        selection: dict[int, list[float]] = defaultdict(list)
        evaluation: dict[int, list[float]] = defaultdict(list)
        for action in state_actions:
            gamma = int(action["gamma"])
            l_value = int(action["accepted_candidates"])
            cost = float(action["round_ms"]) / (1 + l_value)
            if not math.isfinite(cost) or cost <= 0:
                continue
            split = str(action.get("repetition_split"))
            if split == "selection":
                selection[gamma].append(cost)
            elif split == "evaluation":
                evaluation[gamma].append(cost)
        if any(gamma not in selection or gamma not in evaluation for gamma in (3, 7, 11, 15)):
            skipped_incomplete += 1
            continue
        output.append({
            "state_id": state_id,
            "dataset": str(first["dataset"]),
            "sample_id": str(first["sample_id"]),
            "source_group_id": str(first["source_group_id"]),
            "round_index": int(first["round_index"]),
            "prompt_tokens": int(first["prompt_tokens"]),
            "context_tokens": int(first["context_tokens"]),
            "processed_output_tokens": int(first["processed_output_tokens"]),
            "history_ema": reference.get("history_ema"),
            "lagged_signals": dict(reference["lagged_signals"]),
            "selection_costs_ms_per_commit_by_gamma": {
                str(gamma): sum(values) / len(values) for gamma, values in selection.items()
            },
            "evaluation_costs_ms_per_commit_by_gamma": {
                str(gamma): sum(values) / len(values) for gamma, values in evaluation.items()
            },
        })
    return output, {
        "gamma15_candidate_signal_states": len(signal_rows),
        "adjacent_round_pairs": len(paired),
        "matched_complete_b1_states": len(output),
        "b1_states_without_adjacent_pair": skipped_unpaired,
        "b1_states_without_complete_selection_evaluation": skipped_incomplete,
    }


def _b4_analyze(args: argparse.Namespace, run_root: Path, runtime_signature: Mapping[str, Any]) -> dict[str, Any]:
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.speculation_experiments.lagged import (
        _COST_VARIANTS,
        cross_validated_draft_cost_policy,
        fit_draft_cost_policy,
    )

    _validate_b0_signature(run_root, runtime_signature=runtime_signature)
    _validate_replay_cells(
        args, run_root, require_b1v=False,
    )
    rows, coverage = _b4_rows_from_artifacts(run_root)
    groups = {str(row["source_group_id"]) for row in rows}
    report: dict[str, Any]
    eligible: list[str] = []
    if len(groups) < args.n_splits:
        report = {
            "schema_version": "dflash-speculation-b.b4-analysis.v1",
            "status": "inconclusive",
            "state_count": len(rows),
            "source_group_count": len(groups),
            "coverage": coverage,
            "reason": f"need at least {args.n_splits} paired source groups",
        }
    else:
        report = cross_validated_draft_cost_policy(
            rows,
            variants=tuple(_COST_VARIANTS),
            n_splits=args.n_splits,
            seed=args.seed,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=20261013,
        )
        report["coverage"] = coverage
        for variant in ("R-A-LDH", "R-A-LDC", "R-A-LDHC"):
            skill = (report.get("cost_skill_vs_control") or {}).get(f"{variant}_vs_R-A")
            interval = (report.get("cost_skill_ci95_vs_control") or {}).get(f"{variant}_vs_R-A")
            if skill is not None and interval and float(skill) >= 0.02 and float(interval[0]) > 0.0:
                eligible.append(variant)
        report["decision"] = (
            "lagged_signal_supported"
            if eligible else "lagged_draft_length_signal_not_supported_or_inconclusive"
        )
        report["eligible_models"] = eligible
        if eligible:
            models = {
                "R": fit_draft_cost_policy(rows, variant="R"),
                "R-A": fit_draft_cost_policy(rows, variant="R-A"),
                **{variant: fit_draft_cost_policy(rows, variant=variant) for variant in eligible},
            }
            lock = {
                "schema_version": "dflash-speculation-b.b4-lock.v2",
                "dev_run_id": str(args.run_id),
                "runtime_signature": runtime_signature,
                "benchmark_signature": _locked_benchmark_signature(
                    json.loads((run_root / "campaign.json").read_text(encoding="utf-8"))
                ),
                "protocol_parameters": _locked_e2e_protocol(args),
                "dev_source_groups": sorted(groups),
                "selected_policy_variant": min(
                    eligible,
                    key=lambda value: float(report["macro_cost_ms_per_commit"][value]),
                ),
                "fixed_reference_gamma": 15,
                "models": models,
                "oof_metrics": {
                    "cost_skill_vs_control": report["cost_skill_vs_control"],
                    "cost_skill_ci95_vs_control": report["cost_skill_ci95_vs_control"],
                },
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            lock_path = run_root / "b4" / "locked_config.json"
            atomic_json(lock_path, lock)
            report["policy_lock"] = str(lock_path.resolve())
    b4_root = run_root / "b4"
    b4_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(b4_root / "lagged_training_rows.jsonl", rows, {
        "type": "summary", "row_count": len(rows), "coverage": coverage,
    })
    if "oof_predictions" in report:
        _write_jsonl(b4_root / "oof_predictions.jsonl", report["oof_predictions"], {
            "type": "summary", "state_count": len(report["oof_predictions"]),
            "variants": report.get("variants", []),
        })
    atomic_json(b4_root / "analysis.json", report)
    decision_text = [
        "# B4 — tín hiệu uncertainty round trước",
        "",
        f"- Trạng thái: **{report.get('decision', report.get('status'))}**.",
        f"- Reference gamma15 rounds: {coverage['gamma15_reference_states']}; cặp round liền kề: {coverage['adjacent_round_pairs']}; state B1 ghép đủ: {coverage['matched_complete_b1_states']}.",
        f"- Source groups trong đánh giá: {report.get('source_group_count', 0)}.",
        "- Metric là cost/commit cho gamma oracle trên held-out source groups; selection timing fit model, evaluation timing chấm điểm.",
        "- E2E draft-length policy chỉ được khóa khi CI skill so regime+history đạt ngưỡng; confirm phải dùng source groups riêng và B0 completed (AR-token equality chỉ là diagnostic).",
    ]
    (b4_root / "decision.md").write_text("\n".join(decision_text) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "oof_predictions"}, ensure_ascii=False, indent=2), flush=True)
    return report


def _b4_campaign(
    args: argparse.Namespace,
    samples: Sequence[dict[str, Any]],
    run_root: Path,
    runtime: Mapping[str, Any],
    *,
    confirm: bool,
) -> dict[str, Any]:
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.speculation_experiments.lagged import LaggedDraftLengthController

    phase = "confirm" if confirm else "dev"
    if args.cohort_phase != phase:
        raise ValueError(f"--mode b4-{phase} requires --cohort-phase {phase}")
    campaign = json.loads((run_root / "campaign.json").read_text(encoding="utf-8"))
    parity_path = run_root / "evidence" / "parity.json"
    if not parity_path.is_file() or json.loads(parity_path.read_text(encoding="utf-8")).get("status") != "complete":
        raise RuntimeError("B4 E2E requires completed B0 runtime validation in this cohort")
    runtime_signature = runtime["b_campaign_signature"]
    source_groups = sorted({str(row["_cad_source_group_id"]) for row in samples})
    if confirm:
        if not args.policy_lock:
            raise ValueError("--mode b4-confirm requires --policy-lock from b4-dev")
        lock_path = _resolve(args.policy_lock) or Path(args.policy_lock)
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("schema_version") != "dflash-speculation-b.b4-lock.v2":
            raise ValueError("unsupported B4 policy-lock schema")
        if lock.get("runtime_signature") != runtime_signature:
            raise ValueError("B4 policy lock runtime signature differs from confirm run")
        _validate_policy_lock(lock, args=args, campaign=campaign, phase="B4")
        overlap = sorted(set(source_groups) & set(lock.get("dev_source_groups", [])))
        if overlap:
            raise ValueError(f"B4 confirm source groups overlap dev lock: {overlap[:5]}")
        model_payloads = lock["models"]
        selected_model = str(lock["selected_policy_variant"])
        fixed_gamma_best = int(lock["best_fixed_gamma"])
        policy_models = ["R", "R-A", selected_model]
    else:
        analysis_path = run_root / "b4" / "analysis.json"
        if not analysis_path.is_file():
            raise FileNotFoundError("B4 dev requires --mode b4 analysis first")
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        eligible = [str(value) for value in analysis.get("eligible_models", [])]
        if analysis.get("decision") != "lagged_signal_supported" or not eligible:
            raise RuntimeError("B4 dev E2E gate closed: no lagged model passed held-out cost-skill screening")
        preliminary_path = run_root / "b4" / "locked_config.json"
        preliminary = json.loads(preliminary_path.read_text(encoding="utf-8"))
        if preliminary.get("runtime_signature") != runtime_signature:
            raise ValueError("B4 analysis lock runtime signature is stale; rerun --mode b4")
        _validate_policy_lock(preliminary, args=args, campaign=campaign, phase="B4 analysis")
        model_payloads = preliminary["models"]
        selected_model = ""
        fixed_gamma_best = 15
        policy_models = ["R", "R-A", *eligible]

    from TrainingFree.speculation_experiments.lagged import LaggedDraftLengthController
    models = {name: value for name, value in model_payloads.items()}
    policies = {
        name: LaggedDraftLengthController(models[name], fallback_gamma=15)
        for name in policy_models
    }
    if confirm:
        variants = ["ar", f"best_fixed_gamma_{fixed_gamma_best}", *[f"adaptive_{name}" for name in policy_models]]
    else:
        variants = [
            "ar",
            *[f"fixed_gamma_{gamma}" for gamma in (3, 7, 11, 15)],
            *[f"adaptive_{name}" for name in policy_models],
        ]
    b4_root = run_root / "b4"
    b4_root.mkdir(parents=True, exist_ok=True)
    cell_signature = _hash_json({
        "phase": phase,
        "runtime_signature": runtime_signature,
        "campaign": _hash_json(json.loads((run_root / "campaign.json").read_text(encoding="utf-8"))),
        "models": model_payloads,
        "variants": variants,
        "e2e_repetitions": args.e2e_repetitions,
        "e2e_warmups": args.e2e_warmups,
        "max_new_tokens": args.max_new_tokens,
        "policy_lock": _hash_json(lock) if confirm else None,
    })
    cells_root = b4_root / "cells" / phase
    cells_root.mkdir(parents=True, exist_ok=True)

    def run_variant(sample: Mapping[str, Any], variant: str, repetition: int) -> dict[str, Any]:
        args._b3_repetition = repetition
        draft_policy = None
        if variant == "ar":
            generation_variant, gamma = "ar", 15
        elif variant.startswith("fixed_gamma_"):
            generation_variant, gamma = variant, int(variant.rsplit("_", 1)[1])
        elif variant.startswith("best_fixed_gamma_"):
            generation_variant, gamma = f"fixed_gamma_{fixed_gamma_best}", fixed_gamma_best
        elif variant.startswith("adaptive_"):
            name = variant.removeprefix("adaptive_")
            generation_variant, gamma = variant, 15
            draft_policy = policies[name]
        else:
            raise ValueError(f"unknown B4 variant {variant}")
        return _run_e2e_with_retry(
            lambda: _b3_run_one(
                runtime=runtime, sample=sample, args=args, variant=generation_variant,
                fixed_gamma=gamma, draft_policy=draft_policy,
            ),
            label=f"B4 {variant} sample={sample['id']} repetition={repetition}",
        )

    warm_samples = list(samples[: min(args.e2e_warmups, len(samples))])
    warmup_failed_variants: list[str] = []
    for variant in variants:
        warm_variant = (
            f"fixed_gamma_{fixed_gamma_best}"
            if variant.startswith("best_fixed_gamma_")
            else variant
        )
        warmup_success_count = 0
        for index, sample in enumerate(warm_samples):
            warmed = run_variant(sample, warm_variant, -index - 1)
            if warmed["status"] != "ok":
                print(
                    f"[B4-{phase}] warmup failed after retry: {warm_variant} {sample['id']}: {warmed['error']}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            warmup_success_count += 1
        if warmup_success_count == 0:
            warmup_failed_variants.append(variant)
    if warmup_failed_variants:
        failure = {
            "schema_version": "dflash-speculation-b.b4-summary.v1",
            "phase": phase,
            "status": "warmup_incomplete",
            "failed_variants": warmup_failed_variants,
            "reason": "No warmup completed for these variants after bounded retries; latency measurements would be unreliable.",
        }
        atomic_json(b4_root / f"{phase}_summary.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), flush=True)
        return failure

    records: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    generation_failure = False
    rng = random.Random(args.seed)
    for sample_index, sample in enumerate(samples, start=1):
        cell_path = _sample_cell_path(b4_root, phase, sample)
        saved_cell = _load_resume_cell(
            cell_path, resume=args.resume, run_root=run_root, phase=f"b4/cells/{phase}",
            is_compatible=lambda cell: _request_cell_matches_protocol(
                cell, sample, variants=variants, repetitions=args.e2e_repetitions,
            ),
        )
        if saved_cell is not None:
            records.extend(saved_cell.get("requests", []))
            rounds.extend(saved_cell.get("rounds", []))
            generation_failure = generation_failure or any(
                row.get("status") != "ok"
                for row in saved_cell.get("requests", [])
            )
            print(f"[B4-{phase} {sample_index}/{len(samples)}] reuse {sample['dataset']}:{sample['id']}", flush=True)
            continue
        request_start = len(records)
        round_start = len(rounds)
        sample_ar_failed = False
        for repetition in range(args.e2e_repetitions):
            ar = run_variant(sample, "ar", repetition)
            if ar["status"] != "ok":
                generation_failure = True
                sample_ar_failed = True
                print(
                    f"[B4-{phase}] AR failed after retry for {sample['dataset']}:{sample['id']} "
                    f"rep={repetition}; skipping this sample and continuing: {ar['error']}",
                    file=sys.stderr,
                    flush=True,
                )
                break
            ar_record = {
                "schema_version": "dflash-speculation-b.b4-request.v1",
                "type": "request", "run_id": str(args.run_id), "phase": phase,
                "variant": "ar", "dataset": str(sample["dataset"]),
                "sample_id": str(sample["id"]), "source_group_id": str(sample["_cad_source_group_id"]),
                "repetition": repetition, "status": ar["status"], "error": ar["error"],
                "input_tokens": ar["input_tokens"], "output_tokens": ar["output_tokens"],
                "e2e_ms": ar["e2e_ms"], "reported_timings": ar["reported_timings"],
                "counters": ar["counters"], "exact_match_ar": True,
                "output_ids": ar["output_ids"], "output_ids_sha256": _hash_json(ar["output_ids"]),
                "ar_output_ids_sha256": _hash_json(ar["output_ids"]),
            }
            ar_record.update(_output_quality_fields(sample, ar["output_ids"], runtime["tokenizer"]))
            records.append(ar_record)
            for round_row in ar["rounds"]:
                rounds.append({
                    **round_row, "run_id": str(args.run_id), "phase": phase,
                    "variant": "ar", "dataset": str(sample["dataset"]),
                    "sample_id": str(sample["id"]), "source_group_id": str(sample["_cad_source_group_id"]),
                    "repetition": repetition,
                })
            order = [value for value in variants if value != "ar"]
            rng.shuffle(order)
            for variant in order:
                measured = run_variant(sample, variant, repetition)
                exact = measured["output_ids"] == ar["output_ids"]
                record = {
                    "schema_version": "dflash-speculation-b.b4-request.v1",
                    "type": "request", "run_id": str(args.run_id), "phase": phase,
                    "variant": variant, "dataset": str(sample["dataset"]),
                    "sample_id": str(sample["id"]), "source_group_id": str(sample["_cad_source_group_id"]),
                    "repetition": repetition, "status": measured["status"], "error": measured["error"],
                    "input_tokens": measured["input_tokens"], "output_tokens": measured["output_tokens"],
                    "e2e_ms": measured["e2e_ms"], "reported_timings": measured["reported_timings"],
                    "counters": measured["counters"], "exact_match_ar": exact,
                    "output_ids": measured["output_ids"], "output_ids_sha256": _hash_json(measured["output_ids"]),
                    "ar_output_ids_sha256": _hash_json(ar["output_ids"]),
                }
                record.update(_output_quality_fields(sample, measured["output_ids"], runtime["tokenizer"]))
                records.append(record)
                for round_row in measured["rounds"]:
                    rounds.append({
                        **round_row, "run_id": str(args.run_id), "phase": phase,
                        "variant": variant, "dataset": str(sample["dataset"]),
                        "sample_id": str(sample["id"]), "source_group_id": str(sample["_cad_source_group_id"]),
                        "repetition": repetition,
                    })
                if measured["status"] != "ok":
                    generation_failure = True
                    print(
                        f"[B4-{phase}] generation failure: {variant} {sample['dataset']}:{sample['id']} rep={repetition}",
                        file=sys.stderr, flush=True,
                    )
                elif not exact:
                    print(
                        f"[B4-{phase}] AR token mismatch recorded as quality diagnostic: "
                        f"{variant} {sample['dataset']}:{sample['id']} rep={repetition}",
                        flush=True,
                    )
        if sample_ar_failed:
            del records[request_start:]
            del rounds[round_start:]
            atomic_json(cell_path, {
                "schema_version": "dflash-speculation-b.b4-sample-cell.v1",
                "status": "failed",
                "campaign_signature": cell_signature,
                "requests": [],
                "rounds": [],
                "error": {"type": "target_only_ar_failed", "message": str(ar.get("error"))},
            })
            continue
        atomic_json(cell_path, {
            "schema_version": "dflash-speculation-b.b4-sample-cell.v1",
            "status": "complete",
            "campaign_signature": cell_signature,
            "requests": records[request_start:],
            "rounds": rounds[round_start:],
        })
        print(f"[B4-{phase} {sample_index}/{len(samples)}] {sample['dataset']}:{sample['id']} complete", flush=True)

    quality = _request_quality_summary(records)
    _write_jsonl(b4_root / f"{phase}_requests.jsonl", records, {
        "type": "summary", "phase": phase, "request_count": len(records), "variants": variants,
        "repetitions": args.e2e_repetitions, "quality_by_variant": quality,
    })
    _write_jsonl(b4_root / f"{phase}_rounds.jsonl", rounds, {
        "type": "summary", "phase": phase, "round_count": len(rounds),
    })
    if not records:
        summary = {
            "schema_version": "dflash-speculation-b.b4-summary.v1",
            "phase": phase,
            "status": "no_valid_requests",
            "request_count": 0,
            "all_generations_ok": False,
            "reason": "Every sample failed target-only AR; artifacts record the failed samples and can be resumed.",
        }
        atomic_json(b4_root / f"{phase}_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary
    latency = {str(row["variant"]): _macro_geomean_latency(records, str(row["variant"])) for row in records}
    if confirm:
        selected_model = str(lock["selected_policy_variant"])
        comparisons = {
            control: _paired_macro_speedup(
                records, control=control, treatment=f"adaptive_{selected_model}"
            )
            for control in (f"best_fixed_gamma_{fixed_gamma_best}", "adaptive_R", "adaptive_R-A")
        }
        summary = {
            "schema_version": "dflash-speculation-b.b4-summary.v1", "phase": phase,
            "status": "generation_error" if generation_failure else "complete",
            "request_count": len(records), "all_generations_ok": not generation_failure,
            "quality_by_variant": quality,
            "selected_policy_variant": selected_model, "comparisons": comparisons,
            "macro_geomean_latency_ms": latency,
        }
    else:
        fixed_gamma_best = min((3, 7, 11, 15), key=lambda gamma: latency[f"fixed_gamma_{gamma}"])
        eligible = [name for name in policy_models if name not in {"R", "R-A"}]
        selected_model = min(eligible, key=lambda name: latency[f"adaptive_{name}"])
        lock = {
            "schema_version": "dflash-speculation-b.b4-lock.v2",
            "dev_run_id": str(args.run_id),
            "runtime_signature": runtime_signature, "dev_source_groups": source_groups,
            "benchmark_signature": _locked_benchmark_signature(campaign),
            "protocol_parameters": _locked_e2e_protocol(args),
            "selected_policy_variant": selected_model, "best_fixed_gamma": fixed_gamma_best,
            "models": model_payloads, "dev_macro_geomean_latency_ms": latency,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        lock_path = b4_root / "locked_config.json"
        if not generation_failure:
            atomic_json(lock_path, lock)
        summary = {
            "schema_version": "dflash-speculation-b.b4-summary.v1", "phase": phase,
            "status": "generation_error" if generation_failure else "locked_for_confirm",
            "request_count": len(records), "all_generations_ok": not generation_failure,
            "quality_by_variant": quality,
            "selected_policy_variant": selected_model, "best_fixed_gamma": fixed_gamma_best,
            "macro_geomean_latency_ms": latency,
            "policy_lock": str(lock_path.resolve()) if not generation_failure else None,
        }
    atomic_json(b4_root / f"{phase}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def _run_all(args: argparse.Namespace) -> int:
    """Run discovery and eligible confirm branches on one loaded B200 runtime."""
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.context_interventions.resilience import (
        CampaignIntegrityError,
        release_compute_memory,
    )

    dev_args = argparse.Namespace(**vars(args))
    dev_args.cohort_phase = "dev"
    dev_root = _run_root(dev_args)
    dev_root.mkdir(parents=True, exist_ok=True)
    confirm_id = f"{args.run_id}_confirm"
    status_path = dev_root / "all_status.json"
    status: dict[str, Any] = {
        "schema_version": "dflash-speculation-b.all-status.v1",
        "run_id": args.run_id,
        "confirm_run_id": confirm_id,
        "workflow_status": "running",
        "collection_status": "pending",
        "stages": {},
    }

    def record(name: str, outcome: str, **details: Any) -> None:
        status["stages"][name] = {"status": outcome, **details}
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(status_path, status)
        print(f"[all] {name}: {outcome}", flush=True)

    def collect(name: str, operation: Any) -> Any:
        record(name, "running")
        try:
            result = operation()
        except Exception as exc:
            message = str(exc)
            # Model/protocol identity and cache integrity cannot be treated as
            # a missing optional statistic or a closed research gate.
            if _is_fatal_campaign_error(exc) or any(
                marker in message.lower()
                for marker in (
                    "runtime signature",
                    "protocol parameters differ",
                    "corpus/exposure signature",
                    "source groups overlap",
                    "unsupported b3 policy-lock schema",
                    "unsupported b4 policy-lock schema",
                )
            ):
                raise CampaignIntegrityError(f"{name}: {type(exc).__name__}: {exc}") from exc
            gate_closed = name == "B3-dev" and message.startswith("B3 gate closed:")
            record(
                name, "not_applicable" if gate_closed else "failed",
                error={"type": type(exc).__name__, "message": message},
                traceback="".join(traceback.format_exception(exc))[-12000:],
            )
            exc.__traceback__ = None
            release_compute_memory()
            return None
        reported = result.get("status") if isinstance(result, dict) else None
        outcome = (
            "not_applicable" if reported == "b1v_overhead_gate_closed"
            else "partial" if reported == "partial"
            else "failed" if reported in {"failed", "unavailable", "generation_error", "warmup_incomplete", "no_valid_requests"}
            else "complete"
        )
        record(name, outcome, reported_status=reported)
        return result

    def skip(name: str, reason: str, *, incomplete: bool = False) -> None:
        record(name, "prerequisites_incomplete" if incomplete else "not_applicable", reason=reason)

    try:
        if args.resume:
            samples, _, dev_root = _load_resume_campaign(dev_args)
            if dev_args.cohort_phase != "dev":
                raise ValueError("--mode all requires a dev campaign for its base --run-id")
        else:
            samples, _, dev_root = _prepare(dev_args)
        runtime = _runtime(dev_args)
        atomic_json(dev_root / "runtime.json", runtime["b_campaign_signature"])
        if not collect("B0", lambda: _validate(dev_args, samples, dev_root, runtime)):
            record("B0", "failed", reason="No usable reference trajectories")
            raise CampaignIntegrityError("B0 produced no usable reference trajectories; later measurements cannot be valid")
        collect("B1/B2", lambda: _b1(dev_args, samples, dev_root, runtime))
        collect("B1-V", lambda: _b1v(dev_args, samples, dev_root, runtime))
        analysis = collect("analysis", lambda: _analyze_without_stopping_collection(dev_args, dev_root))
        collection = _write_run_status_best_effort(dev_root)
        core_ready = (
            collection is not None
            and collection.get("collection_status") == "complete"
            and all(status["stages"][name]["status"] == "complete" for name in ("B0", "B1/B2", "B1-V"))
        )

        b3_result = None
        b4_result = None
        if core_ready:
            if analysis is not None and analysis.get("status") != "unavailable":
                b3_result = collect("B3-dev", lambda: _b3_campaign(dev_args, samples, dev_root, runtime, confirm=False))
            else:
                skip("B3-dev", "B1/B2 analysis is unavailable; raw measurements are retained", incomplete=True)
            b4_analysis = collect("B4-analysis", lambda: _b4_analyze(dev_args, dev_root, runtime["b_campaign_signature"]))
            if b4_analysis is not None and b4_analysis.get("decision") == "lagged_signal_supported":
                b4_result = collect("B4-dev", lambda: _b4_campaign(dev_args, samples, dev_root, runtime, confirm=False))
            else:
                skip("B4-dev", "Lagged signal did not pass the preregistered gate", incomplete=b4_analysis is None)
        else:
            for name in ("B3-dev", "B4-analysis", "B4-dev"):
                skip(name, "Core collection is incomplete; E2E policy selection requires complete evidence", incomplete=True)

        b3_ready = b3_result is not None and b3_result.get("status") == "locked_for_confirm"
        b4_ready = b4_result is not None and b4_result.get("status") == "locked_for_confirm"
        if b3_ready or b4_ready:
            confirm_args = argparse.Namespace(**vars(dev_args))
            confirm_args.run_id = confirm_id
            confirm_args.cohort_phase = "confirm"
            confirm_args.split_manifest = str(dev_root / "split_manifest.json")
            confirm_root = _run_root(confirm_args)

            def prepare_confirm() -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
                if args.resume and (confirm_root / "campaign.json").is_file():
                    prepared = _load_resume_campaign(confirm_args)
                    if confirm_args.cohort_phase != "confirm":
                        raise CampaignIntegrityError("Confirm run has a different cohort phase")
                    return prepared
                return _prepare(confirm_args)

            prepared = collect("confirm-prepare", prepare_confirm)
            confirm_ready = False
            if prepared is not None:
                confirm_samples, _, confirm_root = prepared
                atomic_json(confirm_root / "runtime.json", runtime["b_campaign_signature"])
                validated = collect("confirm-B0", lambda: _validate(confirm_args, confirm_samples, confirm_root, runtime))
                parity_path = confirm_root / "evidence" / "parity.json"
                parity = json.loads(parity_path.read_text(encoding="utf-8")) if parity_path.is_file() else {}
                confirm_ready = bool(validated) and parity.get("status") == "complete"
                if not confirm_ready and status["stages"]["confirm-B0"]["status"] == "complete":
                    record("confirm-B0", "failed", reason="Confirm runtime validation has incomplete coverage")
            else:
                skip("confirm-B0", "Confirm cohort could not be prepared", incomplete=True)
            for name, eligible, operation, lock_subdir in (
                ("B3-confirm", b3_ready, _b3_campaign, "b3"),
                ("B4-confirm", b4_ready, _b4_campaign, "b4"),
            ):
                if eligible and confirm_ready:
                    confirm_args.policy_lock = str(dev_root / lock_subdir / "locked_config.json")
                    collect(name, lambda operation=operation: operation(confirm_args, confirm_samples, confirm_root, runtime, confirm=True))
                elif eligible:
                    skip(name, "Confirm runtime validation is incomplete", incomplete=True)
                else:
                    skip(name, "Discovery did not produce a valid policy lock")
        else:
            for name in ("confirm-prepare", "confirm-B0", "B3-confirm", "B4-confirm"):
                skip(name, "No E2E policy was locked during discovery")

        partial = (
            not core_ready
            or any(stage["status"] in {"partial", "failed", "prerequisites_incomplete"} for stage in status["stages"].values())
        )
        status["workflow_status"] = "finished"
        status["collection_status"] = "partial" if partial else "complete"
        atomic_json(status_path, status)
        print(f"[all] finished; collection={status['collection_status']}; status={status_path}", flush=True)
        return 0
    except BaseException as exc:
        status["workflow_status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "stopped_fatal"
        status["collection_status"] = "partial"
        status["error"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            atomic_json(status_path, status)
        except Exception:
            pass
        raise


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    if args.run_id is None:
        if args.mode in {"prepare", "run", "all"}:
            args.run_id = _utc_run_id()
        else:
            parser.error("--run-id is required for modes other than prepare/run/all")
    if args.max_new_tokens < 16 or args.validation_tokens < 1:
        parser.error("--max-new-tokens must be >=16 and --validation-tokens must be positive")
    if args.repetitions < 2 or not 1 <= args.selection_repetitions < args.repetitions:
        parser.error("repetitions must exceed selection repetitions >=1")
    if args.warmups < 0 or args.max_prefixes < 1 or args.n_splits < 2:
        parser.error("warmups must be nonnegative; max-prefixes positive; n-splits at least two")
    if args.mode in {"b3-dev", "b3-confirm", "b4-dev", "b4-confirm", "all"} and (
        args.e2e_repetitions < 3 or args.e2e_warmups < 1
    ):
        parser.error("B3/B4 require at least three E2E repetitions and one warmup sample per variant")
    try:
        _parse_gammas(args.gammas)
        if args.mode == "all":
            return _run_all(args)
        if args.mode == "prepare":
            _prepare(args, allow_existing=True)
            return 0
        if args.mode == "analyze":
            _analyze_without_stopping_collection(args, _run_root(args))
            return 0
        if args.mode == "run":
            if args.resume:
                samples, campaign, run_root = _load_resume_campaign(args)
            else:
                samples, campaign, run_root = _prepare(args, allow_existing=True)
        else:
            samples, campaign, run_root = _load_prepared(args)
        if args.mode == "preflight":
            runtime = _runtime(args)
            from TrainingFree.context_adaptive.schema import atomic_json

            atomic_json(run_root / "runtime.json", runtime["b_campaign_signature"])
            print(f"[preflight] assets/backend ready; no experimental phase run", flush=True)
            return 0
        runtime = _runtime(args)
        from TrainingFree.context_adaptive.schema import atomic_json

        atomic_json(run_root / "runtime.json", runtime["b_campaign_signature"])
        if args.mode == "validate":
            return 0 if _validate(args, samples, run_root, runtime) else 2
        if args.mode == "b1":
            _b1(args, samples, run_root, runtime)
            return 0
        if args.mode == "b1v":
            _b1v(args, samples, run_root, runtime)
            return 0
        if args.mode == "run":
            if not _validate(args, samples, run_root, runtime):
                _write_run_status_best_effort(run_root)
                return 2
            _b1(args, samples, run_root, runtime)
            _b1v(args, samples, run_root, runtime)
            _analyze_without_stopping_collection(args, run_root)
            _write_run_status_best_effort(run_root)
            return 0
        if args.mode == "b3-dev":
            summary = _b3_campaign(args, samples, run_root, runtime, confirm=False)
            return 0 if summary.get("status") == "locked_for_confirm" else 2
        if args.mode == "b3-confirm":
            summary = _b3_campaign(args, samples, run_root, runtime, confirm=True)
            return 0 if summary.get("status") == "complete" else 2
        if args.mode == "b4":
            if args.cohort_phase != "dev":
                raise ValueError("B4 analysis is a dev phase; confirm uses a frozen lock")
            report = _b4_analyze(args, run_root, runtime["b_campaign_signature"])
            return 0 if report.get("decision") == "lagged_signal_supported" else 2
        if args.mode == "b4-dev":
            summary = _b4_campaign(args, samples, run_root, runtime, confirm=False)
            return 0 if summary.get("status") == "locked_for_confirm" else 2
        if args.mode == "b4-confirm":
            summary = _b4_campaign(args, samples, run_root, runtime, confirm=True)
            return 0 if summary.get("status") == "complete" else 2
        parser.error(f"unsupported mode {args.mode}")
    except KeyboardInterrupt:
        print("[interrupted] completed per-sample/state artifacts are retained; rerun with --resume", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[{type(exc).__name__}] {exc}", file=sys.stderr, flush=True)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
