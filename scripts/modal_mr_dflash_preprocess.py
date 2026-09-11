#!/usr/bin/env python3
"""Smoke test MR-DFlash preprocessing trên một GPU Modal.

Runner này cố ý không dùng path ``/workspace/storage-shared`` của server B200:
Modal không nhìn thấy filesystem đó. Nó tải target Qwen3-4B vào Volume Modal,
tạo một prompt tổng hợp gần 32K token, rồi chạy đúng các primitive production:

    parallel_stage(regenerate) -> validate -> tokenize -> parallel_stage(cache) -> audit

Mục tiêu là kiểm tra runtime/telemetry/cache contract và đo một sample dài;
đây không phải benchmark chất lượng hay benchmark speedup cuối cùng.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # local contract tests do not need the Modal SDK
    modal = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum")
VOLUME_MOUNT = Path("/mnt/mr-dflash")
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_GPU = "A100-80GB"
DEFAULT_VOLUME = "fast-infer-text-sum-mr-dflash"
FEATURE_LAYER_IDS = (1, 9, 17, 25, 33)
SOURCE_IGNORE = (".git", ".venv", "__pycache__", "*.pyc", "outputs", "checkpoints")


def build_smoke_plan(
    *,
    remote_root: Path,
    run_root: Path,
    target_model: str,
    max_length: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Dựng command plan deterministic cho test local và remote runner."""

    scripts = remote_root / "scripts" / "mr_dflash"
    regenerated = run_root / "regenerated" / "train.jsonl"
    regenerated_manifest = run_root / "manifests" / "regeneration.json"
    validated_report = run_root / "manifests" / "validation.json"
    tokenized_root = run_root / "tokenized" / "train"
    tokenized_manifest = tokenized_root / "manifest.json"
    cache_root = run_root / "target_features" / "train"
    cache_manifest = cache_root / "manifest.json"
    stop_file = run_root / ".stop_parallel"
    regenerate_work = run_root / "regenerated" / "parallel_regenerate_train"
    cache_work = run_root / "target_features" / "parallel_cache_train"
    progress_interval = "8"

    regenerate = [
        sys.executable,
        str(scripts / "parallel_stage.py"),
        "--mode",
        "regenerate",
        "--gpu-ids",
        "0",
        "--input",
        str(run_root / "normalized" / "train_prompts.jsonl"),
        "--output",
        str(regenerated),
        "--manifest",
        str(regenerated_manifest),
        "--work-root",
        str(regenerate_work),
        "--target-model-path",
        target_model,
        "--max-length",
        str(max_length),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        "0.0",
        "--seed",
        "42",
        "--torch-dtype",
        "bfloat16",
        "--overflow-policy",
        "error",
        "--sample-error-policy",
        "error",
        "--preserve-full-input",
        "--progress-interval-tokens",
        progress_interval,
        "--output-batch-size",
        "1",
        "--stall-timeout-seconds",
        "0",
        "--stop-file",
        str(stop_file),
        "--resume",
    ]
    validate = [
        sys.executable,
        str(scripts / "validate_pilot_dataset.py"),
        "--input",
        str(regenerated),
        "--tokenizer",
        target_model,
        "--max-length",
        str(max_length),
        "--expected-target-model",
        target_model,
        "--require-generated",
        "--report",
        str(validated_report),
    ]
    tokenize = [
        sys.executable,
        str(scripts / "tokenize_dataset.py"),
        "--input",
        str(regenerated),
        "--output",
        str(tokenized_root),
        "--provenance-manifest",
        str(run_root / "manifests" / "tokenization.json"),
        "--target-model-path",
        target_model,
        "--max-length",
        str(max_length),
        "--supervision-mode",
        "last_assistant",
        "--feature-layer-ids",
        *(str(value) for value in FEATURE_LAYER_IDS),
        "--shard-size",
        "1",
        "--resume",
    ]
    cache = [
        sys.executable,
        str(scripts / "parallel_stage.py"),
        "--mode",
        "cache",
        "--gpu-ids",
        "0",
        "--input",
        str(regenerated),
        "--output",
        str(cache_root),
        "--manifest",
        str(cache_manifest),
        "--work-root",
        str(cache_work),
        "--target-model-path",
        target_model,
        "--tokenized-path",
        str(tokenized_root),
        "--max-length",
        str(max_length),
        "--target-layer-ids",
        *(str(value) for value in FEATURE_LAYER_IDS),
        "--batch-size",
        "1",
        "--bucket-buffer-size",
        "1",
        "--shard-size",
        "1",
        "--torch-dtype",
        "bfloat16",
        "--supervision-mode",
        "last_assistant",
        "--attention-backend",
        "sdpa",
        "--io-threads",
        "2",
        "--io-queue-size",
        "4",
        "--progress-interval-tokens",
        progress_interval,
        "--stall-timeout-seconds",
        "0",
        "--stop-file",
        str(stop_file),
        "--resume",
    ]
    audit = [
        sys.executable,
        str(scripts / "verify_feature_cache.py"),
        "--data-path",
        str(regenerated),
        "--cache-path",
        str(cache_root),
        "--tokenizer",
        target_model,
        "--max-length",
        str(max_length),
        "--expected-target-model",
        target_model,
        "--expected-feature-layer-ids",
        *(str(value) for value in FEATURE_LAYER_IDS),
        "--report",
        str(run_root / "manifests" / "cache_audit.json"),
    ]
    return {
        "regenerate": regenerate,
        "validate": validate,
        "tokenize": tokenize,
        "cache": cache,
        "audit": audit,
        "regenerate_status": regenerate_work / "status.json",
        "cache_status": cache_work / "status.json",
        "regenerate_progress": regenerate_work / "rank_00" / "progress.json",
        "cache_progress": cache_work / "rank_00" / "progress.json",
        "regenerated": regenerated,
        "tokenized_root": tokenized_root,
        "tokenized_manifest": tokenized_manifest,
        "cache_root": cache_root,
        "stop_file": stop_file,
    }


def validate_smoke_report(report: dict[str, Any]) -> dict[str, Any]:
    """Fail closed nếu một phase hoặc tracker/audit không đạt."""

    for phase in ("regenerate", "cache"):
        value = report.get(phase) or {}
        if value.get("status") != "success":
            raise RuntimeError(f"smoke phase {phase} không thành công: {value}")
        if not value.get("tracker"):
            raise RuntimeError(f"smoke phase {phase} thiếu tracker artifact")
    if not (report.get("audit") or {}).get("valid"):
        raise RuntimeError(f"cache audit không valid: {report.get('audit')}")
    return {**report, "status": "success"}


def _build_image() -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(
        str(ROOT / "requirements.modal.txt")
    )
    image = image.add_local_dir(
        ROOT / "src",
        str(REMOTE_ROOT / "src"),
        copy=False,
        ignore=SOURCE_IGNORE,
    )
    return image.add_local_dir(
        ROOT / "scripts" / "mr_dflash",
        str(REMOTE_ROOT / "scripts" / "mr_dflash"),
        copy=False,
        ignore=SOURCE_IGNORE,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _status_line(path: Path) -> str:
    payload = _read_json(path)
    workers = payload.get("workers") or []
    fields = [f"status={payload.get('status', 'waiting')}"]
    for worker in workers:
        progress = worker.get("progress") or {}
        fields.append(
            " ".join(
                (
                    f"gpu={worker.get('gpu_id', '?')}",
                    f"pid={worker.get('pid', '?')}",
                    f"phase={progress.get('phase', '?')}",
                    f"sample={progress.get('sample_id', '-')}",
                    f"tokens={progress.get('generated_tokens', '-')}",
                    f"done={progress.get('completed_samples', '-')}",
                    f"vram={progress.get('cuda_memory_allocated_gb', '-')}",
                )
            )
        )
    return " | ".join(fields)


def _run_stage(command: list[str], *, cwd: Path, env: dict[str, str], status_path: Path, log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"[modal-smoke] START {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        last = None
        while process.poll() is None:
            current = _status_line(status_path)
            if current != last:
                print(f"[modal-smoke] {current}", flush=True)
                last = current
            time.sleep(1.0)
        return_code = process.wait()
    elapsed = time.monotonic() - started
    final_status = _read_json(status_path)
    print(f"[modal-smoke] DONE rc={return_code} elapsed_s={elapsed:.3f} {status_path}", flush=True)
    if return_code != 0:
        tail = "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-40:])
        raise RuntimeError(f"stage failed rc={return_code}; log={log_path}\n{tail}")
    return {
        "status": "success",
        "elapsed_seconds": round(elapsed, 3),
        "tracker": status_path.is_file(),
        "status_payload": final_status,
        "log": str(log_path),
    }


def _watcher_once(status_path: Path, env: dict[str, str], *, cwd: Path) -> None:
    command = [
        sys.executable,
        str(REMOTE_ROOT / "scripts" / "mr_dflash" / "watch_parallel_stage.py"),
        "--status",
        str(status_path),
        "--once",
        "--tqdm",
    ]
    result = subprocess.run(command, cwd=str(cwd), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"watcher thất bại rc={result.returncode}: {status_path}")


def _create_long_prompt(tokenizer: Any, *, target_tokens: int) -> tuple[list[dict[str, str]], int]:
    """Tạo prompt text hợp lệ gần target_tokens mà không cắt input."""

    unit = (
        "This synthetic scientific article discusses a method, its experiments, "
        "limitations, and conclusions. Summarize the article faithfully.\n"
    )

    def render(repeats: int) -> tuple[list[dict[str, str]], int]:
        messages = [{"role": "user", "content": unit * repeats}]
        value = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        )
        if isinstance(value, dict):
            value = value["input_ids"]
        if hasattr(value, "input_ids"):
            value = value.input_ids
        return messages, int(value.shape[-1])

    unit_tokens = len(tokenizer(unit, add_special_tokens=False)["input_ids"])
    estimate = max(1, int(target_tokens / max(1, unit_tokens)))
    lo, hi = max(1, estimate // 2), max(2, estimate * 2)
    best_messages, best_length = render(1)
    while lo <= hi:
        middle = (lo + hi) // 2
        messages, length = render(middle)
        if length <= target_tokens:
            best_messages, best_length = messages, length
            lo = middle + 1
        else:
            hi = middle - 1
    return best_messages, best_length


def _prepare_input(model_path: str, data_root: Path, *, target_tokens: int) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=False)
    messages, prompt_tokens = _create_long_prompt(tokenizer, target_tokens=target_tokens)
    path = data_root / "normalized" / "train_prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "id": "modal_32k_cache_smoke",
        "source": "modal_synthetic",
        "conversations": messages,
        "metadata": {"requested_prompt_tokens": target_tokens, "actual_prompt_tokens": prompt_tokens},
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[modal-smoke] input_prompt_tokens={prompt_tokens} path={path}", flush=True)
    return prompt_tokens


def _peak_from_status(status: dict[str, Any]) -> float | None:
    values = []
    for worker in status.get("workers") or []:
        progress = worker.get("progress") or {}
        for key in ("cuda_max_memory_allocated_gb", "cuda_memory_allocated_gb"):
            value = progress.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
                break
    return max(values) if values else None


VOLUME_NAME = os.environ.get("MODAL_MR_DFLASH_VOLUME", DEFAULT_VOLUME)
GPU_TYPE = os.environ.get("MODAL_GPU", DEFAULT_GPU)
SECRET_NAME = os.environ.get("MODAL_HF_SECRET", "").strip()
SECRETS = []


class _NoopApp:
    """Cho phép import các pure planning helpers khi SDK chưa cài local."""

    def function(self, **_kwargs):
        return lambda fn: fn

    def local_entrypoint(self):
        return lambda fn: fn


if modal is not None:
    SECRETS = [modal.Secret.from_name(SECRET_NAME, required_keys=["HF_TOKEN"])] if SECRET_NAME else []
    IMAGE = _build_image()
    CACHE_VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    app = modal.App("mr-dflash-preprocess-smoke")
else:
    IMAGE = None
    CACHE_VOLUME = None
    app = _NoopApp()


@app.function(
    image=IMAGE,
    gpu=GPU_TYPE,
    volumes={str(VOLUME_MOUNT): CACHE_VOLUME},
    secrets=SECRETS,
    timeout=6 * 60 * 60,
    env={
        "HF_HOME": str(VOLUME_MOUNT / "hf"),
        "HF_HUB_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
        "TRANSFORMERS_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
        "PYTHONUNBUFFERED": "1",
    },
)
def run_smoke(
    *,
    target_model: str = DEFAULT_MODEL,
    prompt_tokens: int = 30000,
    max_length: int = 32768,
    max_new_tokens: int = 64,
    run_id: str = "",
) -> dict[str, Any]:
    """Chạy smoke generate/validate/cache/audit trên một GPU Modal."""

    if prompt_tokens < 128 or prompt_tokens >= max_length:
        raise ValueError("prompt_tokens phải trong [128, max_length)")
    run_id = run_id or f"modal-smoke-{uuid.uuid4().hex[:10]}"
    run_root = VOLUME_MOUNT / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    data_root = run_root
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts" / "mr_dflash"))),
            "HF_HOME": str(VOLUME_MOUNT / "hf"),
            "HF_HUB_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
            "TRANSFORMERS_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    prompt_length = _prepare_input(target_model, data_root, target_tokens=prompt_tokens)
    plan = build_smoke_plan(
        remote_root=REMOTE_ROOT,
        run_root=run_root,
        target_model=target_model,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
    )

    regenerate_report = _run_stage(
        plan["regenerate"],
        cwd=REMOTE_ROOT,
        env=env,
        status_path=plan["regenerate_status"],
        log_path=run_root / "logs" / "regenerate.log",
    )
    _watcher_once(plan["regenerate_status"], env, cwd=REMOTE_ROOT)
    validate_report = _run_stage(
        plan["validate"],
        cwd=REMOTE_ROOT,
        env=env,
        status_path=run_root / "manifests" / "validation.json",
        log_path=run_root / "logs" / "validate.log",
    )
    tokenize_report = _run_stage(
        plan["tokenize"],
        cwd=REMOTE_ROOT,
        env=env,
        status_path=plan["tokenized_manifest"],
        log_path=run_root / "logs" / "tokenize.log",
    )
    cache_report = _run_stage(
        plan["cache"],
        cwd=REMOTE_ROOT,
        env=env,
        status_path=plan["cache_status"],
        log_path=run_root / "logs" / "cache.log",
    )
    _watcher_once(plan["cache_status"], env, cwd=REMOTE_ROOT)
    audit_report = _run_stage(
        plan["audit"],
        cwd=REMOTE_ROOT,
        env=env,
        status_path=run_root / "manifests" / "cache_audit.json",
        log_path=run_root / "logs" / "audit.log",
    )
    audit_payload = _read_json(run_root / "manifests" / "cache_audit.json")
    regenerate_report["peak_vram_gb"] = _peak_from_status(_read_json(plan["regenerate_status"]))
    cache_report["peak_vram_gb"] = _peak_from_status(_read_json(plan["cache_status"]))
    report = validate_smoke_report(
        {
            "regenerate": regenerate_report,
            "validate": validate_report,
            "tokenize": tokenize_report,
            "cache": cache_report,
            "audit": {**audit_report, **audit_payload},
            "run_id": run_id,
            "target_model": target_model,
            "prompt_tokens": prompt_length,
            "max_length": max_length,
            "max_new_tokens": max_new_tokens,
            "regenerated_path": str(plan["regenerated"]),
            "tokenized_path": str(plan["tokenized_root"]),
            "cache_path": str(plan["cache_root"]),
        }
    )
    report_path = run_root / "smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    CACHE_VOLUME.commit()
    print(f"[modal-smoke] REPORT {report_path}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return report


@app.local_entrypoint()
def main(
    target_model: str = DEFAULT_MODEL,
    prompt_tokens: int = 30000,
    max_length: int = 32768,
    max_new_tokens: int = 64,
    run_id: str = "",
) -> None:
    result = run_smoke.remote(
        target_model=target_model,
        prompt_tokens=prompt_tokens,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        run_id=run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result.get("status") != "success":
        raise SystemExit(1)
