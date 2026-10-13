#!/usr/bin/env python3
"""Run B0/B1/B2 DFlash speculation experiments on the shared B200 runtime.

This entrypoint is offline by design. It does not install packages or download
models/data. Run ``prepare`` on any machine, then use ``run`` on the B200 server
or execute ``preflight``, ``validate``, ``b1`` and ``analyze`` separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts", ROOT / "externals" / "dflash"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "preflight", "validate", "b1", "b1v", "analyze", "run"), default="prepare")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser


def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("b200_%Y%m%dT%H%M%SZ")


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


def _load_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[Path]]:
    from TrainingFree.context_adaptive.benchmark import load_corpus

    data_dir_value = args.data_dir or os.environ.get("DATA_INPUT") or os.environ.get("LONG_BENCH_DATA_DIR")
    data_file_value = args.data_file or os.environ.get("DATA_FILE") or os.environ.get("LONG_BENCH_DATA_FILE")
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
    run_root = _run_root(args)
    campaign_path = run_root / "campaign.json"
    cohort_path = run_root / "cohort_manifest.json"
    if not campaign_path.is_file() or not cohort_path.is_file():
        raise FileNotFoundError(f"run is not prepared; execute --mode prepare first: {run_root}")
    current, campaign, _ = _prepare(args, allow_existing=True)
    saved = json.loads(cohort_path.read_text(encoding="utf-8"))
    sample_by_key = {(str(row["dataset"]), str(row["id"])): row for row in current}
    selected = []
    for row in saved["groups"]:
        key = (str(row["dataset"]), str(row["sample_id"]))
        sample = sample_by_key.get(key)
        if sample is None:
            raise ValueError(f"prepared cohort sample missing from current corpus: {key}")
        selected.append(sample)
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


def _code_signature() -> str:
    from TrainingFree.context_adaptive.schema import sha256_file

    paths = [
        *sorted((ROOT / "src/TrainingFree/speculation_experiments").glob("*.py")),
        *sorted((ROOT / "src/TrainingFree/context_adaptive").glob("*.py")),
        *sorted((ROOT / "src/TrainingFree/context_interventions").glob("*.py")),
        ROOT / "scripts/run_dflash_speculation_B.py",
        ROOT / "externals/dflash/dflash/model.py",
    ]
    return _hash_json([
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
        for path in paths if path.is_file()
    ])


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
    code_signature: str,
) -> str:
    return _hash_json({
        "sample": f"{sample['dataset']}::{sample['id']}",
        "prompt_hash": prompt_hash,
        "gammas": _parse_gammas(args.gammas),
        "max_new_tokens": args.max_new_tokens,
        "validation_tokens": args.validation_tokens,
        "runtime": runtime_signature,
        "code": code_signature,
    })


def _validate(args: argparse.Namespace, samples: Sequence[dict[str, Any]], run_root: Path, runtime: Mapping[str, Any]) -> bool:
    from TrainingFree.context_adaptive.prompt import prepare_prompt
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.context_interventions.execution import run_reference_pair

    gammas = _parse_gammas(args.gammas)
    validation_rows = []
    code = _code_signature()
    runtime_signature = runtime["b_campaign_signature"]
    for sample_index, sample in enumerate(samples, start=1):
        path = _sample_trace_path(run_root, sample)
        ids, layout = prepare_prompt(sample, runtime["tokenizer"], chunk_size=128)
        expected_signature = _trace_signature(
            sample=sample,
            prompt_hash=layout.prompt_hash,
            args=args,
            runtime_signature=runtime_signature,
            code_signature=code,
        )
        if path.exists() and args.resume:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("trace_signature") != expected_signature:
                raise ValueError(f"resume trace signature mismatch: {path}")
            parity = saved["parity_by_gamma"]
            validation_rows.append({"sample_id": sample["id"], "parity_by_gamma": parity})
            print(f"[B0 {sample_index}/{len(samples)}] reuse {sample['dataset']}:{sample['id']}", flush=True)
            continue
        if path.exists():
            raise FileExistsError(f"trace already exists; pass --resume: {path}")

        parity_by_gamma: dict[str, Any] = {}
        reference_ar: list[int] | None = None
        reference_trace: dict[str, Any] | None = None
        gamma_order = (15, *[gamma for gamma in gammas if gamma != 15])
        for gamma in gamma_order:
            limit = args.max_new_tokens if gamma == 15 else min(args.validation_tokens, args.max_new_tokens)
            measured = run_reference_pair(
                runtime,
                sample,
                max_new_tokens=limit,
                gamma=gamma,
                reference_ar_ids=reference_ar,
            )
            if gamma == 15:
                reference_ar = measured["ar_ids"]
                reference_trace = measured
            parity_by_gamma[str(gamma)] = measured["parity"]
            print(
                f"[B0 {sample_index}/{len(samples)}] {sample['dataset']}:{sample['id']} "
                f"gamma={gamma} status={measured['parity']['status']}",
                flush=True,
            )
        parity_pass = all(row.get("status") == "pass" for row in parity_by_gamma.values())
        if reference_trace is None:
            raise RuntimeError("gamma15 reference trace was not produced")
        trace = {
            "schema_version": "dflash-speculation-b.reference.v1",
            "trace_signature": expected_signature,
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
            "parity_status": "pass" if parity_pass else "invalid_correctness",
        }
        atomic_json(path, trace)
        validation_rows.append({"sample_id": sample["id"], "parity_by_gamma": parity_by_gamma})
    passed = all(
        all(value.get("status") == "pass" for value in sample.get("parity_by_gamma", {}).values())
        for sample in validation_rows
    )
    summary = {
        "schema_version": "dflash-speculation-b.b0-parity.v1",
        "status": "pass" if passed else "invalid_correctness",
        "sample_count": len(validation_rows),
        "gamma_values": list(gammas),
        "parity": validation_rows,
        "runtime": runtime_signature,
        "code_signature": code,
    }
    atomic_json(run_root / "evidence" / "parity.json", summary)
    if not passed:
        print("[B0] parity failed; B1/B2 are gated", flush=True)
    return passed


def _state_cell_path(run_root: Path, state_key: str) -> Path:
    return run_root / "b1" / "cells" / f"{hashlib.sha256(state_key.encode('utf-8')).hexdigest()}.json"


def _b1(args: argparse.Namespace, samples: Sequence[dict[str, Any]], run_root: Path, runtime: Mapping[str, Any]) -> None:
    from TrainingFree.context_adaptive.prompt import prepare_prompt
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.context_interventions.replay import evaluate_replay_action, rebuild_replay_state
    from TrainingFree.speculation_experiments.reference import reference_history, select_reference_prefixes
    from TrainingFree.speculation_experiments.replay import run_same_prefix_sweep

    parity_path = run_root / "evidence" / "parity.json"
    if not parity_path.is_file() or json.loads(parity_path.read_text(encoding="utf-8")).get("status") != "pass":
        raise RuntimeError("B1 is gated on a complete B0 parity pass")
    code = _code_signature()
    runtime_sig = runtime["b_campaign_signature"]
    params = {
        "gammas": _parse_gammas(args.gammas),
        "repetitions": args.repetitions,
        "selection_repetitions": args.selection_repetitions,
        "warmups": args.warmups,
        "max_prefixes": args.max_prefixes,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    campaign_signature = _hash_json({"code": code, "runtime": runtime_sig, "campaign": params})
    for sample_index, sample in enumerate(samples, start=1):
        trace_path = _sample_trace_path(run_root, sample)
        if not trace_path.is_file():
            raise FileNotFoundError(f"missing gamma15 reference trace: {trace_path}")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        if trace.get("parity_status") != "pass":
            raise RuntimeError(f"sample {sample['id']} did not pass B0 parity")
        input_ids, layout = prepare_prompt(sample, runtime["tokenizer"], chunk_size=128)
        input_ids = input_ids.to(runtime["target"].device).contiguous()
        if layout.prompt_hash != trace["prompt_hash"]:
            raise ValueError(f"prompt tokenization changed for sample {sample['id']}")
        expected_trace_signature = _trace_signature(
            sample=sample,
            prompt_hash=layout.prompt_hash,
            args=args,
            runtime_signature=runtime_sig,
            code_signature=code,
        )
        if trace.get("trace_signature") != expected_trace_signature:
            raise ValueError(f"reference trace signature is stale for sample {sample['id']}; rerun B0 validate")
        prefixes = select_reference_prefixes(
            trace["adapter_rounds"],
            trace["ar_ids"],
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
        if not prefixes:
            print(f"[B1 {sample_index}/{len(samples)}] no regular states for {sample['dataset']}:{sample['id']}", flush=True)
            continue
        for prefix_index, prefix in enumerate(prefixes, start=1):
            state_key = f"{sample['dataset']}|{sample['_cad_source_group_id']}|{prefix.round_index}"
            cell_path = _state_cell_path(run_root, state_key)
            if cell_path.exists():
                if not args.resume:
                    raise FileExistsError(f"B1 state cell exists; pass --resume: {cell_path}")
                cell = json.loads(cell_path.read_text(encoding="utf-8"))
                if cell.get("campaign_signature") != campaign_signature or cell.get("state_key") != state_key:
                    raise ValueError(f"B1 cell signature mismatch: {cell_path}")
                continue
            reference_round = reference_rounds.get(prefix.round_index)
            if reference_round is None:
                raise ValueError(f"reference round missing: {prefix.round_index}")
            state = rebuild_replay_state(
                runtime,
                input_ids,
                trace["ar_ids"],
                prefix,
                layout,
                max_new_tokens=args.max_new_tokens,
            )
            # Rebuild and verify gamma15 before measuring the other shapes.
            full = evaluate_replay_action(
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
            )
            if full.get("status") != "ok" or int(full.get("accepted_candidates", -1)) != int(reference_round["accepted_candidates"]):
                raise RuntimeError(
                    f"gamma15 replay mismatch at {state_key}: replay={full.get('accepted_candidates')} "
                    f"reference={reference_round.get('accepted_candidates')}"
                )
            history = reference_history(trace["adapter_rounds"], current_round=prefix.round_index)
            result = run_same_prefix_sweep(
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
            )
            signal = result.get("signal")
            if signal is not None:
                signal["history"] = history
                signal["history_ema"] = history.get("ema_acceptance")
                signal["previous_accepted_candidates"] = history.get("previous_accepted_candidates")
                signal["previous_all_accepted"] = history.get("previous_all_accepted")
                signal["history_missing"] = bool(history.get("missing", True))
            cell = {
                "schema_version": "dflash-speculation-b.cell.v1",
                "campaign_signature": campaign_signature,
                "state_key": state_key,
                "state_id": result["state_id"],
                "actions": result["actions"],
                "signal": signal,
                "schedule": result["schedule"],
                "warmups": result["warmups"],
                "accepted_by_gamma": result["accepted_by_gamma"],
                "proposal_by_gamma": result["proposal_by_gamma"],
                "bank_build_ms": state.bank_build_ms,
            }
            atomic_json(cell_path, cell)
            print(
                f"[B1 {sample_index}/{len(samples)} state {prefix_index}/{len(prefixes)}] "
                f"{sample['dataset']}:{sample['id']} round={prefix.round_index} complete",
                flush=True,
            )
    cells = []
    for cell_path in sorted((run_root / "b1" / "cells").glob("*.json")):
        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        if cell.get("campaign_signature") != campaign_signature:
            raise ValueError(f"found incompatible B1 state cell {cell_path}")
        cells.append(cell)
    actions = [row for cell in cells for row in cell.get("actions", [])]
    signals = [cell["signal"] for cell in cells if cell.get("signal") is not None]
    _write_jsonl(run_root / "b1" / "draft_shape.jsonl", actions, {"type": "summary", "state_count": len(cells), "action_count": len(actions)})
    _write_jsonl(run_root / "b2" / "signals.jsonl", signals, {"type": "summary", "state_count": len(signals)})


def _b1v(args: argparse.Namespace, samples: Sequence[dict[str, Any]], run_root: Path, runtime: Mapping[str, Any]) -> None:
    from TrainingFree.context_adaptive.prompt import prepare_prompt
    from TrainingFree.context_adaptive.schema import atomic_json
    from TrainingFree.context_interventions.replay import rebuild_replay_state
    from TrainingFree.speculation_experiments.reference import select_reference_prefixes
    from TrainingFree.speculation_experiments.replay import run_verification_prefix_sweep

    parity_path = run_root / "evidence" / "parity.json"
    if not parity_path.is_file() or json.loads(parity_path.read_text(encoding="utf-8")).get("status") != "pass":
        raise RuntimeError("B1-V is gated on a complete B0 parity pass")
    b1_cells = []
    for path in sorted((run_root / "b1" / "cells").glob("*.json")):
        b1_cells.append(json.loads(path.read_text(encoding="utf-8")))
    proposal_by_state = {
        str(cell["state_id"]): tuple(int(token) for token in cell["proposal_by_gamma"]["15"])
        for cell in b1_cells
    }
    if not proposal_by_state:
        raise FileNotFoundError("B1 cells are required before B1-V")
    code = _code_signature()
    runtime_sig = runtime["b_campaign_signature"]
    b1_params = {
        "gammas": _parse_gammas(args.gammas),
        "repetitions": args.repetitions,
        "selection_repetitions": args.selection_repetitions,
        "warmups": args.warmups,
        "max_prefixes": args.max_prefixes,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    b1_signature = _hash_json({"code": code, "runtime": runtime_sig, "campaign": b1_params})
    if any(cell.get("campaign_signature") != b1_signature for cell in b1_cells):
        raise ValueError("B1 cells were produced with a different code/runtime/protocol signature")
    params = {
        "verify_prefixes": (3, 7, 11, 15),
        "repetitions": args.repetitions,
        "selection_repetitions": args.selection_repetitions,
        "warmups": args.warmups,
        "max_prefixes": args.max_prefixes,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    signature = _hash_json({"code": code, "runtime": runtime_sig, "campaign": params})
    for sample_index, sample in enumerate(samples, start=1):
        trace_path = _sample_trace_path(run_root, sample)
        if not trace_path.is_file():
            raise FileNotFoundError(f"missing gamma15 reference trace: {trace_path}")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        ids, layout = prepare_prompt(sample, runtime["tokenizer"], chunk_size=128)
        ids = ids.to(runtime["target"].device).contiguous()
        if layout.prompt_hash != trace["prompt_hash"]:
            raise ValueError(f"prompt tokenization changed for sample {sample['id']}")
        expected_trace_signature = _trace_signature(
            sample=sample,
            prompt_hash=layout.prompt_hash,
            args=args,
            runtime_signature=runtime_sig,
            code_signature=code,
        )
        if trace.get("trace_signature") != expected_trace_signature:
            raise ValueError(f"reference trace signature is stale for sample {sample['id']}; rerun B0 validate")
        prefixes = select_reference_prefixes(
            trace["adapter_rounds"], trace["ar_ids"],
            prompt_length=int(trace["prompt_tokens"]), max_new_tokens=args.max_new_tokens,
            eos_ids={int(token) for token in trace.get("eos_ids", [])}, max_prefixes=args.max_prefixes,
        )
        reference_rounds = {
            int(row["round_index"]): row for row in trace["adapter_rounds"]
            if row.get("type") == "round" and int(row.get("gamma_executed", 0)) == 15
        }
        for prefix_index, prefix in enumerate(prefixes, start=1):
            state_key = f"{sample['dataset']}|{sample['_cad_source_group_id']}|{prefix.round_index}"
            cell_path = run_root / "b1v" / "cells" / f"{hashlib.sha256(state_key.encode('utf-8')).hexdigest()}.json"
            if cell_path.exists():
                if not args.resume:
                    raise FileExistsError(f"B1-V state cell exists; pass --resume: {cell_path}")
                existing = json.loads(cell_path.read_text(encoding="utf-8"))
                if existing.get("campaign_signature") != signature:
                    raise ValueError(f"B1-V cell signature mismatch: {cell_path}")
                continue
            reference_round = reference_rounds.get(prefix.round_index)
            if reference_round is None:
                raise ValueError(f"reference round missing: {prefix.round_index}")
            state = rebuild_replay_state(
                runtime, ids, trace["ar_ids"], prefix, layout,
                max_new_tokens=args.max_new_tokens,
            )
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
                raise ValueError(f"B1 proposal missing for paired state {state_identity_value}")
            result = run_verification_prefix_sweep(
                runtime,
                ids,
                state,
                layout,
                dataset=str(sample["dataset"]),
                source_group_id=str(sample["_cad_source_group_id"]),
                sample_id=str(sample["id"]),
                prompt_token_ids=prompt_ids,
                reference_output_ids=trace["ar_ids"],
                max_new_tokens=args.max_new_tokens,
                round_index=prefix.round_index,
                full_accepted=int(reference_round["accepted_candidates"]),
                repetitions=args.repetitions,
                selection_repetitions=args.selection_repetitions,
                warmups=args.warmups,
                seed=args.seed + prefix.round_index,
                expected_proposal_ids=proposal_by_state[state_identity_value],
            )
            atomic_json(cell_path, {
                "schema_version": "dflash-speculation-b.b1v-cell.v1",
                "campaign_signature": signature,
                "state_key": state_key,
                "state_id": result["state_id"],
                "actions": result["actions"],
                "schedule": result["schedule"],
                "proposal_token_ids": result["proposal_token_ids"],
            })
            print(
                f"[B1-V {sample_index}/{len(samples)} state {prefix_index}/{len(prefixes)}] "
                f"{sample['dataset']}:{sample['id']} round={prefix.round_index} complete",
                flush=True,
            )
    cells = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((run_root / "b1v" / "cells").glob("*.json"))
    ]
    if any(cell.get("campaign_signature") != signature for cell in cells):
        raise ValueError("B1-V cells contain an incompatible campaign signature")
    actions = [row for cell in cells for row in cell.get("actions", [])]
    _write_jsonl(run_root / "b1v" / "verify_prefix.jsonl", actions, {
        "type": "summary", "state_count": len(cells), "action_count": len(actions),
    })


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


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
    if args.run_id is None:
        if args.mode in {"prepare", "run"}:
            args.run_id = _utc_run_id()
        else:
            parser.error("--run-id is required for modes other than prepare/run")
    if args.max_new_tokens < 16 or args.validation_tokens < 1:
        parser.error("--max-new-tokens must be >=16 and --validation-tokens must be positive")
    if args.repetitions < 2 or not 1 <= args.selection_repetitions < args.repetitions:
        parser.error("repetitions must exceed selection repetitions >=1")
    if args.warmups < 0 or args.max_prefixes < 1 or args.n_splits < 2:
        parser.error("warmups must be nonnegative; max-prefixes positive; n-splits at least two")
    try:
        _parse_gammas(args.gammas)
        if args.mode == "prepare":
            _prepare(args, allow_existing=True)
            return 0
        if args.mode == "analyze":
            _analyze(args, _run_root(args))
            return 0
        if args.mode == "run":
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
                return 2
            _b1(args, samples, run_root, runtime)
            _b1v(args, samples, run_root, runtime)
            _analyze(args, run_root)
            return 0
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
