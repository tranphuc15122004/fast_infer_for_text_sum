#!/usr/bin/env python3
"""Debug vLLM + MR-DFlash Phase 1 trên một GPU Modal.

Runner này cố ý chỉ tạo một input tổng hợp rất nhỏ.  Nó kiểm tra đúng đường
      vLLM server -> regenerate(vLLM) -> validate -> tokenize -> cache(HF) -> audit
trước khi đưa cấu hình tương tự lên B200 thật.

vLLM phải được dừng trước cache: cùng một GPU không thể vừa giữ KV pool của
server vừa nạp backbone HF cho target-feature cache.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # local .venv contract tests do not need Modal SDK
    modal = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum")
VOLUME_MOUNT = Path("/mnt/mr-dflash")
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_GPU = os.environ.get("MODAL_GPU", "A100-80GB")
DEFAULT_VOLUME = "fast-infer-text-sum-mr-dflash-vllm"
FEATURE_LAYER_IDS = (1, 9, 17, 25, 33)
SERVED_MODEL = "qwen3-4b"


def build_vllm_server_command(
    *,
    model: str,
    port: int,
    max_model_len: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
) -> list[str]:
    """Build the vLLM 0.24 command used by both Modal and the B200 guide."""

    return [
        "vllm",
        "serve",
        str(model),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(int(port)),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(int(max_model_len)),
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        str(int(max_num_seqs)),
        "--max-num-batched-tokens",
        str(int(max_num_batched_tokens)),
        # FlashInfer on the user's B200 image tries to JIT-compile a kernel
        # that includes nvrtc.h, which is absent there.  Force the installed
        # FlashAttention path for this smoke and production command.
        "--attention-config",
        '{"backend":"FLASH_ATTN"}',
        # vLLM 0.24 still runs a FlashInfer autotune warmup on SM100 unless
        # it is explicitly disabled.  That warmup is unrelated to the chosen
        # attention backend and was the source of the Modal B200 failure.
        "--kernel-config",
        '{"enable_flashinfer_autotune":false}',
        # Keep the smoke deterministic while validating the serving protocol;
        # the B200 production command can remove this after the kernel stack
        # passes the smoke.
        "--enforce-eager",
        "--kv-cache-metrics",
    ]


def build_vllm_phase1_plan(
    *,
    remote_root: Path,
    run_root: Path,
    target_model: str,
    server_address: str,
    max_length: int,
    max_new_tokens: int,
    request_concurrency: int,
    max_batched_tokens: int,
) -> dict[str, list[str]]:
    """Build production primitives with vLLM only in the regenerate phase."""

    scripts = remote_root / "scripts" / "mr_dflash"
    normalized = run_root / "normalized" / "train_prompts.jsonl"
    regenerated = run_root / "regenerated" / "train.jsonl"
    regenerated_manifest = run_root / "manifests" / "regeneration.json"
    tokenized_root = run_root / "tokenized" / "train"
    tokenized_manifest = run_root / "manifests" / "tokenization.json"
    cache_root = run_root / "target_features" / "train"
    cache_manifest = run_root / "manifests" / "cache.json"
    stop_file = run_root / "stop_parallel"

    regenerate = [
        sys.executable,
        str(scripts / "parallel_stage.py"),
        "--mode",
        "regenerate",
        "--scheduler",
        "shared_lease",
        "--queue-quantum-items",
        "2",
        "--gpu-ids",
        "0",
        "--input",
        str(normalized),
        "--output",
        str(regenerated),
        "--manifest",
        str(regenerated_manifest),
        "--work-root",
        str(run_root / "regenerated" / "parallel_regenerate_train"),
        "--target-model-path",
        str(target_model),
        "--regenerate-backend",
        "vllm",
        "--vllm-server-addresses",
        str(server_address),
        "--vllm-model",
        SERVED_MODEL,
        "--vllm-request-concurrency",
        str(int(request_concurrency)),
        "--vllm-request-concurrency-start",
        "1",
        "--vllm-max-batched-tokens",
        str(int(max_batched_tokens)),
        "--vllm-max-batched-tokens-start",
        str(min(2048, int(max_batched_tokens))),
        "--vllm-request-growth-factor",
        "2.0",
        "--vllm-request-timeout-seconds",
        "120",
        "--vllm-request-retries",
        "1",
        "--vllm-metrics-address",
        str(server_address).rstrip("/").removesuffix("/v1") + "/metrics",
        "--vllm-metrics-poll-interval-seconds",
        "0.25",
        "--vllm-gpu-cache-target",
        "0.85",
        "--vllm-gpu-cache-hard",
        "0.95",
        "--max-length",
        str(int(max_length)),
        "--max-new-tokens",
        str(int(max_new_tokens)),
        "--temperature",
        "0.0",
        "--seed",
        "42",
        "--overflow-policy",
        "error",
        "--sample-error-policy",
        "error",
        "--preserve-full-input",
        "--progress-interval-tokens",
        "8",
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
        str(target_model),
        "--max-length",
        str(int(max_length)),
        "--expected-target-model",
        SERVED_MODEL,
        "--require-generated",
        "--report",
        str(run_root / "manifests" / "validation.json"),
    ]
    tokenize = [
        sys.executable,
        str(scripts / "tokenize_dataset.py"),
        "--input",
        str(regenerated),
        "--output",
        str(tokenized_root),
        "--provenance-manifest",
        str(tokenized_manifest),
        "--target-model-path",
        str(target_model),
        "--max-length",
        str(int(max_length)),
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
        "--scheduler",
        "shared_lease",
        "--queue-quantum-items",
        "2",
        "--gpu-ids",
        "0",
        "--input",
        str(regenerated),
        "--tokenized-path",
        str(tokenized_root),
        "--output",
        str(cache_root),
        "--manifest",
        str(cache_manifest),
        "--work-root",
        str(run_root / "target_features" / "parallel_cache_train"),
        "--target-model-path",
        str(target_model),
        "--max-length",
        str(int(max_length)),
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
        "8",
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
        str(target_model),
        "--max-length",
        str(int(max_length)),
        "--expected-target-model",
        str(target_model),
        "--expected-feature-layer-ids",
        *(str(value) for value in FEATURE_LAYER_IDS),
        "--report",
        str(run_root / "manifests" / "cache_audit.json"),
    ]
    return {"regenerate": regenerate, "validate": validate, "tokenize": tokenize, "cache": cache, "audit": audit}


def validate_vllm_smoke_report(report: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless every phase and the final cache audit passed."""

    if (report.get("server") or {}).get("status") != "ready":
        raise RuntimeError(f"vLLM server chưa ready: {report.get('server')}")
    for phase in ("regenerate", "validate", "tokenize", "cache"):
        if (report.get(phase) or {}).get("status") != "success":
            raise RuntimeError(f"phase {phase} không thành công: {report.get(phase)}")
    if not (report.get("audit") or {}).get("valid"):
        raise RuntimeError(f"cache audit không valid: {report.get('audit')}")
    return {**report, "status": "success"}


def _build_image() -> Any:
    """Create a cu130 image close to the server stack, then add vLLM."""

    if modal is None:
        return None
    image = (
        modal.Image.from_registry(
            "nvidia/cuda:13.0.1-devel-ubuntu24.04",
            add_python="3.12",
        )
        .entrypoint([])
        .apt_install("libnuma1", "libnuma-dev")
        .pip_install(
            "torch==2.11.0+cu130",
            extra_options="--index-url https://download.pytorch.org/whl/cu130",
        )
        .pip_install_from_requirements(str(ROOT / "requirements.modal.txt"))
        .pip_install("vllm==0.24.0")
        .add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=False)
        .add_local_dir(ROOT / "scripts" / "mr_dflash", str(REMOTE_ROOT / "scripts" / "mr_dflash"), copy=False)
    )
    return image


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _tail(path: Path, lines: int = 160) -> str:
    try:
        return "".join(path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-lines:])
    except OSError:
        return ""


def _create_input(model: str, path: Path, *, prompt_tokens: int, count: int) -> int:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=False)
    unit = (
        "This synthetic article describes a reproducible scientific method, "
        "its experiments, limitations, and conclusions. Summarize it faithfully.\n"
    )
    unit_tokens = len(tokenizer(unit, add_special_tokens=False)["input_ids"])
    rows: list[dict[str, Any]] = []
    actual = 0
    for index in range(int(count)):
        repeats = max(1, int(prompt_tokens / max(1, unit_tokens)))
        messages = [{"role": "user", "content": unit * repeats + f"\nDocument id: {index}."}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        )
        if isinstance(rendered, dict):
            rendered = rendered["input_ids"]
        if hasattr(rendered, "input_ids"):
            rendered = rendered.input_ids
        actual = max(actual, int(rendered.shape[-1]))
        rows.append(
            {
                "id": f"modal_vllm_smoke_{index}",
                "source": "modal_synthetic",
                "conversations": messages,
                "metadata": {"requested_prompt_tokens": int(prompt_tokens)},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return actual


def _wait_for_server(process: subprocess.Popen[Any], *, port: int, log_path: Path, timeout: float = 900.0) -> None:
    url = f"http://127.0.0.1:{int(port)}/health"
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited rc={process.returncode}\n{_tail(log_path)}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= int(response.status) < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2.0)
    raise TimeoutError(f"vLLM server không ready sau {timeout}s\n{_tail(log_path)}")


def _stop_server(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def _run_stage(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"[modal-vllm-smoke] START {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    payload = {"status": "success" if result.returncode == 0 else "failed", "returncode": result.returncode, "elapsed_seconds": round(time.monotonic() - started, 3), "log": str(log_path)}
    if result.returncode != 0:
        raise RuntimeError(f"stage failed rc={result.returncode}; log={log_path}\n{_tail(log_path)}")
    return payload


VOLUME_NAME = os.environ.get("MODAL_MR_DFLASH_VLLM_VOLUME", DEFAULT_VOLUME)
GPU_TYPE = os.environ.get("MODAL_GPU", DEFAULT_GPU)
SECRET_NAME = os.environ.get("MODAL_HF_SECRET", "").strip()


class _NoopApp:
    def function(self, **_kwargs):
        return lambda fn: fn

    def local_entrypoint(self):
        return lambda fn: fn


if modal is not None:
    IMAGE = _build_image()
    VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    SECRETS = [modal.Secret.from_name(SECRET_NAME)] if SECRET_NAME else []
    app = modal.App("mr-dflash-vllm-phase1-smoke")
else:
    IMAGE = None
    VOLUME = None
    SECRETS = []
    app = _NoopApp()


@app.function(
    image=IMAGE,
    gpu=GPU_TYPE,
    volumes={str(VOLUME_MOUNT): VOLUME},
    secrets=SECRETS,
    timeout=6 * 60 * 60,
)
def run_smoke(
    *,
    target_model: str = DEFAULT_MODEL,
    prompt_tokens: int = 1024,
    max_length: int = 2048,
    max_new_tokens: int = 32,
    num_samples: int = 2,
    request_concurrency: int = 2,
    max_batched_tokens: int = 8192,
    run_id: str = "",
) -> dict[str, Any]:
    if prompt_tokens < 128 or prompt_tokens + max_new_tokens >= max_length:
        raise ValueError("prompt_tokens + max_new_tokens phải nhỏ hơn max_length")
    if num_samples < 1 or request_concurrency < 1 or max_batched_tokens < 1:
        raise ValueError("num_samples/request_concurrency/max_batched_tokens phải >= 1")
    run_id = run_id or f"modal-vllm-smoke-{uuid.uuid4().hex[:10]}"
    run_root = VOLUME_MOUNT / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    runtime_tmp = Path("/tmp/mr_dflash_vllm")
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    cache_root = run_root / "runtime_cache"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": os.pathsep.join((str(REMOTE_ROOT / "src"), str(REMOTE_ROOT / "scripts" / "mr_dflash"))),
            "HF_HOME": str(VOLUME_MOUNT / "hf"),
            "HF_HUB_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
            "TRANSFORMERS_CACHE": str(VOLUME_MOUNT / "hf" / "hub"),
            "TMPDIR": str(runtime_tmp),
            "TEMP": str(runtime_tmp),
            "TMP": str(runtime_tmp),
            "VLLM_CACHE_ROOT": str(cache_root / "vllm"),
            "TRITON_CACHE_DIR": str(cache_root / "triton"),
            "TORCH_EXTENSIONS_DIR": str(cache_root / "torch_extensions"),
            "FLASHINFER_WORKSPACE_BASE": str(cache_root / "flashinfer"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    for value in (cache_root / "vllm", cache_root / "triton", cache_root / "torch_extensions", cache_root / "flashinfer"):
        value.mkdir(parents=True, exist_ok=True)

    input_path = run_root / "normalized" / "train_prompts.jsonl"
    prompt_actual = _create_input(target_model, input_path, prompt_tokens=prompt_tokens, count=num_samples)
    port = 38147
    server_address = f"http://127.0.0.1:{port}/v1"
    plan = build_vllm_phase1_plan(
        remote_root=REMOTE_ROOT,
        run_root=run_root,
        target_model=target_model,
        server_address=server_address,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        request_concurrency=request_concurrency,
        max_batched_tokens=max_batched_tokens,
    )
    server_log = run_root / "logs" / "vllm.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    server_command = build_vllm_server_command(
        model=target_model,
        port=port,
        max_model_len=max_length,
        max_num_seqs=request_concurrency,
        max_num_batched_tokens=max_batched_tokens,
    )
    server_env = dict(env)
    server_env["VLLM_LOGGING_LEVEL"] = "INFO"
    server_process: subprocess.Popen[Any] | None = None
    report: dict[str, Any] = {"server": {"status": "starting", "port": port, "log": str(server_log)}}
    try:
        with server_log.open("w", encoding="utf-8") as handle:
            server_process = subprocess.Popen(
                server_command,
                cwd=str(REMOTE_ROOT),
                env=server_env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _wait_for_server(server_process, port=port, log_path=server_log)
        report["server"] = {"status": "ready", "pid": server_process.pid, "command": server_command, "log": str(server_log)}
        report["regenerate"] = _run_stage(plan["regenerate"], cwd=REMOTE_ROOT, env=env, log_path=run_root / "logs" / "regenerate.log")
    finally:
        _stop_server(server_process)
    report["validate"] = _run_stage(plan["validate"], cwd=REMOTE_ROOT, env=env, log_path=run_root / "logs" / "validate.log")
    report["tokenize"] = _run_stage(plan["tokenize"], cwd=REMOTE_ROOT, env=env, log_path=run_root / "logs" / "tokenize.log")
    report["cache"] = _run_stage(plan["cache"], cwd=REMOTE_ROOT, env=env, log_path=run_root / "logs" / "cache.log")
    audit = _run_stage(plan["audit"], cwd=REMOTE_ROOT, env=env, log_path=run_root / "logs" / "audit.log")
    audit.update(_read_json(run_root / "manifests" / "cache_audit.json"))
    report["audit"] = audit
    report.update({"run_id": run_id, "target_model": target_model, "prompt_tokens": prompt_actual, "max_length": max_length, "max_new_tokens": max_new_tokens, "artifacts_root": str(run_root)})
    report = validate_vllm_smoke_report(report)
    report_path = run_root / "smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    VOLUME.commit()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    return report


@app.local_entrypoint()
def main(
    target_model: str = DEFAULT_MODEL,
    prompt_tokens: int = 1024,
    max_length: int = 2048,
    max_new_tokens: int = 32,
    num_samples: int = 2,
    request_concurrency: int = 2,
    max_batched_tokens: int = 8192,
    run_id: str = "",
) -> None:
    result = run_smoke.remote(
        target_model=target_model,
        prompt_tokens=prompt_tokens,
        max_length=max_length,
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        request_concurrency=request_concurrency,
        max_batched_tokens=max_batched_tokens,
        run_id=run_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result.get("status") != "success":
        raise SystemExit(1)
