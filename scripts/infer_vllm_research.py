#!/usr/bin/env python3
"""Điều phối ma trận nghiên cứu DFlash/A-B trên vLLM 0.30, chạy offline."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from Benchmark.common.benchmark_data import read_jsonl
from Benchmark.common.io_util import JsonlWriter
from Benchmark.common.prompt_format import format_chat_prompt
from Benchmark.vllm_all_baselines import _prepare_samples
from Benchmark.vllm_research.analysis import execute_analysis
from Benchmark.vllm_research.compat import inspect_capabilities
from Benchmark.vllm_research.engine import _model_identity, run_generation
from Benchmark.vllm_research.execution import atomic_json, collect_jobs, identity, summarize
from Benchmark.vllm_research.plan import ResearchJob, build_jobs
from Benchmark.vllm_research.replay import describe_replay_support


def _csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _int_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part) for part in _csv(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("danh sách phải gồm số nguyên phân tách bằng dấu phẩy") from exc
    if not values or any(number < 1 for number in values):
        raise argparse.ArgumentTypeError("các giá trị phải là số nguyên dương")
    return values


def _budget_csv(value: str) -> tuple[int | str, ...]:
    result: list[int | str] = []
    try:
        for item in _csv(value):
            result.append("full" if item.lower() == "full" else int(item))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("budget phải là số nguyên hoặc full") from exc
    if not result or any(item != "full" and item < 1 for item in result):
        raise argparse.ArgumentTypeError("budget phải là số nguyên dương hoặc full")
    return tuple(result)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Chạy ma trận A/B DFlash trên runtime vLLM dùng chung của server."
    )
    result.add_argument("--target-model", default=os.environ.get("VLLM_TARGET_MODEL", ""))
    result.add_argument("--draft-model", default=os.environ.get("VLLM_DFLASH_MODEL", ""))
    result.add_argument("--data-file", default=os.environ.get("VLLM_DATA_FILE", ""))
    result.add_argument("--output-root", default=os.environ.get("VLLM_RESEARCH_OUTPUT_ROOT", "outputs/vllm_research"))
    result.add_argument("--run-id", default=os.environ.get("VLLM_RESEARCH_RUN_ID", ""))
    result.add_argument("--resume", nargs="?", const="", metavar="RUN_ID",
                        help="tiếp tục campaign; có thể truyền RUN_ID")
    result.add_argument("--plan-only", action="store_true", help="ghi plan/capability, không nạp tokenizer hoặc GPU")
    result.add_argument("--full", action="store_true", help="dùng mọi mẫu đủ điều kiện; bỏ giới hạn mẫu")
    result.add_argument("--max-samples", type=int,
                        default=int(os.environ.get("VLLM_RESEARCH_MAX_SAMPLES", os.environ.get("RUN_SAMPLES", "2"))))
    result.add_argument("--experiments", default=os.environ.get("VLLM_RESEARCH_EXPERIMENTS", "ALL"))
    result.add_argument("--selectors", default=os.environ.get("VLLM_RESEARCH_SELECTORS", "previous_dflash"))
    result.add_argument("--budgets", type=_budget_csv, default=_budget_csv(os.environ.get("VLLM_RESEARCH_BUDGETS", "2048")))
    result.add_argument("--periods", type=_int_csv, default=_int_csv(os.environ.get("VLLM_RESEARCH_PERIODS", "2,4,8,16")))
    result.add_argument("--gammas", type=_int_csv, default=_int_csv(os.environ.get("VLLM_RESEARCH_GAMMAS", "3,7,11,15")))
    result.add_argument("--measurement", choices=("latency", "throughput"), default=os.environ.get("VLLM_RESEARCH_MEASUREMENT", "latency"))
    result.add_argument("--max-num-seqs", type=int, default=int(os.environ.get("VLLM_RESEARCH_MAX_NUM_SEQS", "16")))
    result.add_argument("--max-num-batched-tokens", type=int, default=int(os.environ["VLLM_MAX_NUM_BATCHED_TOKENS"]) if os.environ.get("VLLM_MAX_NUM_BATCHED_TOKENS") else None)
    result.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("VLLM_MAX_NEW_TOKENS", os.environ.get("RUN_MAX_NEW_TOKENS", "512"))))
    result.add_argument("--max-input-tokens", type=int, default=int(os.environ.get("VLLM_MAX_INPUT_TOKENS", os.environ.get("RUN_MAX_INPUT_TOKENS", "0"))))
    result.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "12288")))
    result.add_argument("--dtype", default=os.environ.get("VLLM_DTYPE", "bfloat16"))
    result.add_argument("--gpu-memory-utilization", type=float, default=float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85")))
    result.add_argument("--warmup-prompts", type=int, default=int(os.environ.get("VLLM_WARMUP_PROMPTS", "3")))
    result.add_argument("--warmup-tokens", type=int, default=int(os.environ.get("VLLM_WARMUP_TOKENS", "64")))
    result.add_argument("--repetitions", type=int, default=int(os.environ.get("VLLM_RESEARCH_REPETITIONS", "1")))
    result.add_argument("--seed", type=int, default=int(os.environ.get("VLLM_SEED", "42")))
    return result


def _input_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("*.jsonl"))
        if files:
            return files
    raise FileNotFoundError(f"dữ liệu phải là JSONL hoặc thư mục chứa JSONL: {path}")


def _load_rows(path: Path) -> tuple[list[dict[str, Any]], str, list[str]]:
    files = _input_files(path)
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.resolve()).encode("utf-8"))
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.extend(read_jsonl(file))
    return rows, digest.hexdigest(), [str(file.resolve()) for file in files]


def _source_layout(sample: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Tính vị trí nguồn từ đúng prompt/token IDs mà vLLM sẽ nhận."""
    prompt = sample["prompt"]
    token_ids = sample["prompt_token_ids"]
    raw = sample.get("source_record") or {}
    dataset = str(raw.get("dataset", sample.get("source_dataset", "unknown")))
    source = next((str(raw[key]) for key in ("context", "document", "text", "input")
                   if isinstance(raw.get(key), str) and raw[key]), None)
    prompt_hash = hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()
    base = {"prompt_tokens": len(token_ids), "prompt_hash": prompt_hash,
            "dataset": dataset, "sample_id": sample.get("sample_id"),
            "source_group_id": sample.get("source_group_id")}
    if not source:
        return {**base, "source_positions": [], "global_positions": [],
                "source_available": False, "source_reason": "source_text_unavailable"}
    first = prompt.find(source)
    if first < 0 or prompt.find(source, first + 1) >= 0:
        reason = "source_span_not_found" if first < 0 else "source_span_ambiguous"
        return {**base, "source_positions": [], "global_positions": [],
                "source_available": False, "source_reason": reason}
    try:
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if offsets and isinstance(offsets[0], list) and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
            offsets = offsets[0]
        offset_ids = encoded["input_ids"]
        if hasattr(offset_ids, "tolist"):
            offset_ids = offset_ids.tolist()
        if offset_ids and isinstance(offset_ids[0], list):
            offset_ids = offset_ids[0]
        if [int(value) for value in offset_ids] != token_ids:
            raise ValueError("offset token IDs differ from native prompt IDs")
    except Exception as exc:
        return {**base, "source_positions": [], "global_positions": [],
                "source_available": False,
                "source_reason": f"tokenizer_offsets_unavailable:{type(exc).__name__}"}
    last = first + len(source)
    source_positions = [index for index, pair in enumerate(offsets)
                        if int(pair[1]) > first and int(pair[0]) < last and int(pair[1]) > int(pair[0])]
    selected = set(source_positions)
    global_positions = [index for index in range(len(token_ids)) if index not in selected]
    return {**base, "source_positions": source_positions, "global_positions": global_positions,
            "source_available": bool(source_positions),
            "source_reason": None if source_positions else "source_span_has_no_tokens"}


def _prepare_input(path: Path, target: str, *, max_samples: int, max_input_tokens: int,
                   max_model_len: int, max_new_tokens: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, list[str], dict[str, Any]]:
    rows, data_hash, files = _load_rows(path)
    id_counts: dict[str, int] = {}
    for index, row in enumerate(rows):
        sample_id = str(row.get("id", index))
        id_counts[sample_id] = id_counts.get(sample_id, 0) + 1
    for index, row in enumerate(rows):
        original_id = str(row.get("id", index))
        if id_counts[original_id] > 1:
            dataset = str(row.get("dataset") or "unknown")
            row["source_record_id"] = original_id
            row["id"] = f"{dataset}:{original_id}"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True)
    samples, excluded = _prepare_samples(rows, tokenizer,
        max_input_tokens=max_input_tokens, max_total_tokens=max_model_len - max_new_tokens,
        max_samples=max_samples, length_spread_samples=True)
    for sample in samples:
        raw = sample.get("source_record") or {}
        dataset = str(raw.get("dataset") or sample.get("source_dataset") or "unknown")
        original_id = str(raw.get("source_record_id") or raw.get("id") or sample["sample_id"])
        group = raw.get("source_group_id") or raw.get("_cad_source_group_id") or f"{dataset}:{original_id}"
        sample["source_group_id"] = str(group)
        sample["layout"] = _source_layout(sample, tokenizer)
        sample["layout"]["source_group_id"] = sample["source_group_id"]
    layout_map = {sample["layout"]["prompt_hash"]: sample["layout"] for sample in samples}
    tokenizer_info = {"class": type(tokenizer).__name__, "name_or_path": getattr(tokenizer, "name_or_path", None),
                      "vocab_size": getattr(tokenizer, "vocab_size", None)}
    return samples, excluded, data_hash, files, {"layouts": layout_map, "tokenizer": tokenizer_info}


def _capability_map(jobs: list[ResearchJob], installed: dict[str, Any], replay: dict[str, Any], draft_model: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "native_generation": installed["native"],
        "trace": installed["trace"],
        "full_adapter": installed["full_adapter"],
        "a3": installed["a3"],
        "b3_fixed_k": installed["b3_fixed_k"],
        "adaptive_verification": {"available": False,
            "reason": "CLI chưa nối frozen survival fit và measured cost_profiles vào B3 worker; fixed-k vẫn chạy."},
        "adaptive_draft": {"available": False,
            "reason": "CLI chưa nối frozen lagged-cost fit vào B4 worker; fixed-gamma vẫn chạy."},
        "same_prefix_replay": {"available": False, "reason": replay["reason"],
                                "required_worker_primitives": replay["required_worker_primitives"]},
        "cpu_analysis": {"available": True, "reason": "analysis chạy trên CPU; phụ thuộc raw evidence thật"},
    }
    for job in jobs:
        if job.capability.startswith("missing_draft:"):
            result[job.capability] = {"available": False,
                "reason": f"không tìm thấy checkpoint DFlash local: {draft_model or '(chưa cấu hình)'}"}
    return result


def _job_artifact_rows(run_dir: Path, job_id: str) -> list[dict[str, Any]]:
    folder = run_dir / "jobs" / job_id
    collected: list[dict[str, Any]] = []
    trace_files = sorted((folder / "traces").glob("*.jsonl")) if (folder / "traces").is_dir() else []
    for path in (folder / "results.jsonl", *trace_files):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("record_type") != "summary" and row.get("type") != "summary":
                        collected.append(row)
    return collected


def _safe_run_id(value: str) -> str:
    value = value or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("run-id chỉ được gồm chữ, số, '.', '_' và '-'")
    return value


def _implementation_identity() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / "src/Benchmark/vllm_all_baselines.py"]
    paths.extend(root / "src/Benchmark/vllm_research" / name
                 for name in ("plan.py", "analysis.py", "compat.py", "adapter.py",
                              "context.py", "engine.py", "execution.py", "replay.py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.is_file()}


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("vllm", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.max_samples < 0 or args.max_new_tokens < 1 or args.max_input_tokens < 0:
        raise SystemExit("max-samples/input-tokens phải không âm; max-new-tokens phải dương")
    if args.max_model_len <= args.max_new_tokens or args.max_num_seqs < 1 or args.repetitions < 1:
        raise SystemExit("max-model-len, max-num-seqs hoặc repetitions không hợp lệ")
    if args.warmup_prompts < 0 or args.warmup_tokens < 1 or not 0 < args.gpu_memory_utilization < 1:
        raise SystemExit("warmup hoặc gpu-memory-utilization không hợp lệ")
    if args.max_num_batched_tokens is not None and args.max_num_batched_tokens < 1:
        raise SystemExit("max-num-batched-tokens phải dương")
    if args.resume is not None:
        if args.resume:
            args.run_id = args.resume
        elif not args.run_id:
            raise SystemExit("--resume cần RUN_ID hoặc --run-id/VLLM_RESEARCH_RUN_ID")
    run_id = _safe_run_id(args.run_id)
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = Path.cwd() / output_root
    run_dir = output_root / run_id
    resuming = args.resume is not None
    if resuming and not run_dir.is_dir():
        raise SystemExit(f"không tìm thấy campaign để resume: {run_dir}")
    if not resuming and run_dir.exists():
        precreated = run_dir.is_dir() and set(path.name for path in run_dir.iterdir()) <= {"console.log"}
        if not precreated:
            raise SystemExit(f"campaign đã tồn tại; dùng --resume {run_id}: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args.experiments, selectors=_csv(args.selectors), budgets=args.budgets,
                      periods=args.periods, gammas=args.gammas)
    installed = inspect_capabilities()
    replay = describe_replay_support(str(installed.get("native", {}).get("vllm_version") or "unknown"))
    if not installed.get("native", {}).get("available"):
        installed["native"] = {**installed["native"], "available": False,
                                "reason": "không tìm thấy package vLLM trong Python runtime đã chọn"}
    # Native worker intentionally pins the same internal runner profile as the adapter.
    if installed.get("native", {}).get("vllm_version") != "0.30.0":
        installed["native"] = {**installed["native"], "available": False,
                                "reason": f"worker yêu cầu vLLM 0.30.0; hiện có {installed['native'].get('vllm_version')}"}
    missing_target = not args.target_model
    missing_data = not args.data_file
    preflight_errors: list[str] = []
    if not args.plan_only and missing_target:
        preflight_errors.append("thiếu --target-model/VLLM_TARGET_MODEL")
    if not args.plan_only and missing_data:
        preflight_errors.append("thiếu --data-file/VLLM_DATA_FILE")
    samples: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    data_hash = None
    data_files: list[str] = []
    input_meta: dict[str, Any] = {"layouts": {}, "tokenizer": None}
    if not args.plan_only:
        if not missing_target:
            target_path = Path(args.target_model).expanduser()
            if not target_path.is_dir() or not (target_path / "config.json").is_file():
                preflight_errors.append(f"target model không phải thư mục local có config.json: {args.target_model}")
        if not missing_data:
            try:
                data_path = Path(args.data_file).expanduser().resolve()
                if preflight_errors:
                    _, data_hash, data_files = _load_rows(data_path)
                else:
                    samples, excluded, data_hash, data_files, input_meta = _prepare_input(
                        data_path, args.target_model, max_samples=0 if args.full else args.max_samples,
                        max_input_tokens=args.max_input_tokens, max_model_len=args.max_model_len,
                        max_new_tokens=args.max_new_tokens)
            except Exception as exc:
                preflight_errors.append(f"data/tokenizer preparation failed: {type(exc).__name__}: {exc}")
    elif args.data_file:
        data_path = Path(args.data_file).expanduser()
        if data_path.exists():
            _, data_hash, data_files = _load_rows(data_path)

    draft_checkpoint = Path(args.draft_model).expanduser() if args.draft_model else None
    draft_available = bool(draft_checkpoint and draft_checkpoint.is_dir() and (draft_checkpoint / "config.json").is_file())
    target_identity = _model_identity(args.target_model) if args.target_model and Path(args.target_model).is_dir() else None
    draft_identity = _model_identity(args.draft_model) if draft_available else None
    implementation = _implementation_identity()
    runtime_versions = _runtime_versions()
    planned_jobs = []
    for job in jobs:
        if job.variant != "ar" and not draft_available and job.capability not in {"same_prefix_replay", "cpu_analysis"}:
            job = replace(job, capability=f"missing_draft:{job.id}")
        planned_jobs.append(job)
    capabilities = _capability_map(planned_jobs, installed, replay, args.draft_model)
    settings = {
        "target_model": args.target_model, "draft_model": args.draft_model,
        "max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens,
        "max_model_len": args.max_model_len, "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "measurement": args.measurement, "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "warmup_prompts": args.warmup_prompts, "warmup_tokens": args.warmup_tokens,
        "seed": args.seed, "repetitions": args.repetitions, "resume": resuming,
        "prompt_layouts_file": str(run_dir / "prompt_layouts.json") if not args.plan_only else None,
    }
    identity_settings = {key: value for key, value in settings.items() if key not in {"resume", "prompt_layouts_file"}}
    layout_digest = identity(input_meta["layouts"])
    plan_signature = identity({"jobs": [job.to_dict() for job in planned_jobs],
                               "settings": identity_settings, "data_sha256": data_hash,
                               "data_files": data_files, "prompt_layouts_sha256": layout_digest,
                               "target_identity": target_identity, "draft_identity": draft_identity,
                               "implementation": implementation, "runtime_versions": runtime_versions})
    plan = {
        "schema_version": "vllm-research-plan.v1", "run_id": run_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "plan_signature": plan_signature, "experiments": args.experiments,
        "measurement": args.measurement, "sample_limit": "all" if args.full else args.max_samples,
        "sample_count": len(samples) if not args.plan_only and not preflight_errors else None,
        "planned_jobs": len(planned_jobs), "estimate_cells": len(samples) * len(planned_jobs) if samples else None,
        "preflight_errors": preflight_errors,
        "data_files": data_files, "data_sha256": data_hash,
        "prompt_layouts_sha256": layout_digest,
        "model_identities": {"target": target_identity, "draft": draft_identity},
        "implementation": implementation, "runtime_versions": runtime_versions,
        "target_model": args.target_model, "draft_model": args.draft_model,
        "settings": identity_settings,
        "capabilities": capabilities, "replay_support": replay,
        "jobs": [job.to_dict() for job in planned_jobs],
    }
    existing_plan = run_dir / "plan.json"
    if resuming and existing_plan.is_file():
        previous_plan = json.loads(existing_plan.read_text(encoding="utf-8"))
        if previous_plan.get("plan_signature") != plan_signature:
            raise SystemExit("plan/config/data khác campaign cũ; dùng run-id mới thay vì resume")
    if not args.plan_only:
        atomic_json(run_dir / "excluded_samples.json", excluded)
        atomic_json(run_dir / "prompt_layouts.json", input_meta["layouts"])
    atomic_json(existing_plan, plan)
    atomic_json(run_dir / "runtime.json", {"capabilities": capabilities,
        "versions": runtime_versions,
        "gpu_verified": False, "semantic_parity_verified": False,
        "execution_profile": {"measurement": args.measurement,
        "max_num_seqs": 1 if args.measurement == "latency" else args.max_num_seqs,
        "dtype": args.dtype, "target_model": args.target_model, "draft_model": args.draft_model}})
    if args.plan_only:
        print(json.dumps({"run_id": run_id, "plan": str(existing_plan), "planned_jobs": len(planned_jobs),
                          "capability_counts": {key: sum(job.capability == key for job in planned_jobs) for key in sorted({job.capability for job in planned_jobs})}}, ensure_ascii=False, indent=2))
        return 0

    print(f"[vLLM research] run={run_id} jobs={len(planned_jobs)} samples={len(samples)} measurement={args.measurement}", flush=True)
    for sample in samples:
        sample["source_group_id"] = sample.get("source_group_id") or sample["sample_id"]
    by_job = {job.id: job for job in planned_jobs}

    def execute(job: ResearchJob, job_dir: Path) -> dict[str, Any]:
        if job.kind in {"generation", "trace"}:
            if preflight_errors:
                return {"status": "failed", "reason": "; ".join(preflight_errors), "metrics": {}}
            if job.experiment == "A3":
                invalid = [sample["sample_id"] for sample in samples if not sample.get("layout", {}).get("source_available")]
                if invalid:
                    return {"status": "unsupported", "reason": "A3 cần source-token offsets chính xác; mẫu thiếu layout: " + ", ".join(invalid[:8]),
                            "unsupported_samples": len(invalid), "metrics": {}}
            outcome = run_generation(job.to_dict(), samples, settings, job_dir)
            outcome.pop("records", None)
            return outcome
        if job.kind == "analysis":
            artifacts = {dependency: _job_artifact_rows(run_dir, dependency) for dependency in job.dependencies}
            outcome = execute_analysis(job, artifacts)
            if job.parameters.get("analysis") == "candidate_signal_extract" and outcome.get("status") == "succeeded":
                writer = JsonlWriter(job_dir / "results.jsonl")
                for row in outcome.get("payload", []):
                    writer.add(row)
                writer.finalize({"type": "summary", "job_id": job.id,
                                 "candidate_signal_rows": len(outcome.get("payload", []))})
            return outcome
        if job.kind == "replay":
            return {"status": "unsupported", "reason": replay["reason"],
                    "required_worker_primitives": replay["required_worker_primitives"], "metrics": {}}
        return {"status": "unsupported", "reason": f"unknown job kind: {job.kind}"}

    results = collect_jobs(planned_jobs, run_dir, capabilities, execute, resume=resuming)
    summary = summarize(results)
    summary.update({"run_id": run_id, "plan_signature": plan_signature,
                    "measurement": args.measurement, "sample_count": len(samples),
                    "excluded_samples": len(excluded), "data_sha256": data_hash,
                    "preflight_errors": preflight_errors, "replay_support": replay})
    writer = JsonlWriter(run_dir / "results.jsonl")
    for row in results:
        job = by_job.get(row.get("job_id"), {})
        writer.add({"record_type": "job_result", "job_id": row.get("job_id"),
                    "experiment": job.experiment if isinstance(job, ResearchJob) else None,
                    "variant": job.variant if isinstance(job, ResearchJob) else None,
                    "status": row.get("status"), "reason": row.get("reason"),
                    "metrics": row.get("metrics"), "planned_cells": row.get("planned_cells"),
                    "succeeded_cells": row.get("succeeded_cells"), "failed_cells": row.get("failed_cells")})
    writer.finalize(summary)
    atomic_json(run_dir / "report.json", {**summary, "jobs": results})
    atomic_json(run_dir / "progress.json", {"status": "finished", **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    # Scientific diagnostics and unsupported cells are recorded, not used as a campaign gate.
    return 0


if __name__ == "__main__":
    sys.exit(main())
