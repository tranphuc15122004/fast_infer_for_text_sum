#!/usr/bin/env python3
"""Run a small paired FA4/vLLM metric pilot on Modal B200.

Stages the local Qwen3-4B and DFlash snapshots plus three summarization rows
from length bin 3, runs the repository benchmark launchers, and downloads the
result artifacts under outputs/length_bins_metric_pilot/.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = Path("/workspace/fast_infer_text_sum")
VOLUME_MOUNT = Path("/mnt/lengthbins")
RUN_ID = "lengthbins_metricpilot_b200_20261012"
VLLM_RUN_ID = os.environ.get("LENGTH_BINS_VLLM_RUN_ID", f"{RUN_ID}-vllm")
VOLUME_NAME = "lengthbins-metric-pilot-20261012"
GPU_TYPE = os.environ.get("MODAL_GPU", "B200")

SOURCE_DATA = ROOT / "data/length_bins/bin3_04k_08k.jsonl"
TARGET_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
)
DFLASH_SNAPSHOT = (
    Path.home()
    / ".cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/61ab4992e5b5ec5913c7f8a9618367b4309533a3"
)
SAMPLE_IDS = (
    "multi_news_ba0f8e390417f2e0",
    "qmsum_a49c337a0657329f",
    "gov_report_d8f57a133bfc5558",
)
MAX_NEW_TOKENS = 256
MAX_MODEL_LEN = 12288

VOLUME = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App("length-bins-metric-pilot")


FA4_RUNTIME_REQUIREMENTS = (
    "accelerate==1.15.0",
    "apache-tvm-ffi==0.1.11",
    "cuda-bindings==13.4.1",
    "cuda-core==1.2.0",
    "cuda-pathfinder==1.8.1",
    "cuda-python==13.4.1",
    "cuda-toolkit==13.0.3.0",
    "einops==0.8.2",
    "flash-attn-4==4.0.0b19",
    "huggingface_hub==1.31.0",
    "ninja==1.13.0",
    "numpy==2.3.5",
    "nvidia-cublas==13.1.1.3",
    "nvidia-cuda-cccl==13.3.4.3.1",
    "nvidia-cuda-crt==13.4.59",
    "nvidia-cuda-cupti==13.0.85",
    "nvidia-cuda-nvcc==13.2.86",
    "nvidia-cuda-nvdisasm==13.4.92",
    "nvidia-cuda-nvrtc==13.0.88",
    "nvidia-cuda-runtime==13.0.96",
    "nvidia-cuda-tileiras==13.2.86",
    "nvidia-cudnn-cu13==9.20.0.48",
    "nvidia-cudnn-frontend==1.28.0",
    "nvidia-cufft==12.0.0.61",
    "nvidia-cufile==1.15.1.6",
    "nvidia-curand==10.4.0.35",
    "nvidia-cusolver==12.0.4.66",
    "nvidia-cusparse==12.6.3.3",
    "nvidia-cusparselt-cu13==0.8.1",
    "nvidia-cutlass-dsl==4.7.1",
    "nvidia-cutlass-dsl-libs-base==4.7.1",
    "nvidia-cutlass-dsl-libs-core==4.7.1",
    "nvidia-cutlass-dsl-libs-cu13==4.7.1",
    "nvidia-mathdx==25.6.0",
    "nvidia-nccl-cu13==2.29.7",
    "nvidia-nvjitlink==13.0.88",
    "nvidia-nvvm==13.2.86",
    "nvidia-nvtx==13.0.85",
    "packaging==26.3",
    "psutil==7.2.2",
    "PyYAML==6.0.3",
    "quack-kernels==0.6.5",
    "safetensors==0.8.0",
    "tokenizers==0.22.2",
    "torch==2.13.0",
    "tqdm==4.70.1",
    "transformers==5.12.1",
    "triton==3.7.1",
)


def _add_repo_code(image: modal.Image) -> modal.Image:
    return (
        image.add_local_dir(ROOT / "src", str(REMOTE_ROOT / "src"), copy=False)
        .add_local_dir(ROOT / "scripts", str(REMOTE_ROOT / "scripts"), copy=False)
        .add_local_dir(
            ROOT / "externals/dflash",
            str(REMOTE_ROOT / "externals/dflash"),
            copy=False,
        )
    )


FA4_IMAGE = modal.Image.debian_slim(python_version="3.12").pip_install(
    *FA4_RUNTIME_REQUIREMENTS
).env(
    {
        "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'scripts'}:{REMOTE_ROOT / 'externals/dflash'}",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRITON_CACHE_DIR": "/tmp/lengthbins/triton",
        "TORCH_EXTENSIONS_DIR": "/tmp/lengthbins/torch_extensions",
    }
)
FA4_IMAGE = _add_repo_code(FA4_IMAGE)

VLLM_IMAGE = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("curl", "libnuma1", "libnuma-dev")
    .pip_install(
        "torch==2.11.0+cu130",
        extra_options="--index-url https://download.pytorch.org/whl/cu130",
    )
    .pip_install_from_requirements(str(ROOT / "requirements.modal.txt"))
    .pip_install("vllm==0.24.0")
).env(
    {
        "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'scripts'}:{REMOTE_ROOT / 'externals/dflash'}",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "FLASHINFER_WORKSPACE_BASE": "/tmp/lengthbins/flashinfer",
        "TRITON_CACHE_DIR": "/tmp/lengthbins/triton",
        "TORCH_EXTENSIONS_DIR": "/tmp/lengthbins/torch_extensions",
    }
)
VLLM_IMAGE = _add_repo_code(VLLM_IMAGE)


def _make_master_config() -> Path:
    target = VOLUME_MOUNT / "models/Qwen3-4B"
    draft = VOLUME_MOUNT / "models/Qwen3-4B-DFlash-b16"
    config = Path("/tmp/lengthbins_metric_pilot.env")
    config.write_text(
        "\n".join(
            (
                "FI_PYTHON=python3",
                "FI_DEVICE=cuda",
                "FI_OFFLINE=1",
                f"MODEL_TARGET={target}",
                f"MODEL_DFLASH_DRAFT={draft}",
                f"LONG_BENCH_MODEL={target}",
                f"LONG_BENCH_DFLASH_MODEL={draft}",
                f"VLLM_TARGET_MODEL={target}",
                f"VLLM_DFLASH_MODEL={draft}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return config


def _gpu_snapshot() -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Modal worker did not expose CUDA")
    properties = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)
    if "B200" not in gpu_name or capability[0] < 10:
        raise RuntimeError(
            f"This pilot requires Modal B200 / SM100+, got {gpu_name}, capability={capability}"
        )
    return {
        "gpu": gpu_name,
        "compute_capability": list(capability),
        "total_vram_bytes": int(properties.total_memory),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }


def _run(command: list[str], *, env: dict[str, str], log_path: Path) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=str(REMOTE_ROOT),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    result = {
        "returncode": completed.returncode,
        "log": str(log_path),
        "tail": "\n".join(lines[-100:]),
    }
    if completed.returncode:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    config = _make_master_config()
    env.update(
        {
            "FAST_INFER_MASTER_CONFIG": str(config),
            "FAST_INFER_VENV": "",
            "FAST_INFER_PYTHON": "python3",
            "FI_PYTHON": "python3",
            "FI_DEVICE": "cuda",
            "FI_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "FAST_INFER_CACHE_ROOT": "/tmp/lengthbins/cache",
            "PYTHONPATH": f"{REMOTE_ROOT / 'src'}:{REMOTE_ROOT / 'scripts'}:{REMOTE_ROOT / 'externals/dflash'}",
        }
    )
    return env


@app.function(
    image=FA4_IMAGE,
    gpu=GPU_TYPE,
    cpu=16,
    memory=65536,
    timeout=7200,
    volumes={str(VOLUME_MOUNT): VOLUME},
)
def run_fa4() -> dict[str, object]:
    os.chdir(REMOTE_ROOT)
    hardware = _gpu_snapshot()
    data_dir = VOLUME_MOUNT / "runs" / RUN_ID / "input"
    output_dir = VOLUME_MOUNT / "runs" / RUN_ID / "results/fa4"
    env = _base_env()
    env.update(
        {
            "FA4_EXECUTION_BACKEND": "server",
            "FA4_DATA_DIR": str(data_dir),
            "FA4_OUTPUT_ROOT": str(output_dir),
            "LONG_BENCH_MODEL": str(VOLUME_MOUNT / "models/Qwen3-4B"),
            "LONG_BENCH_DFLASH_MODEL": str(
                VOLUME_MOUNT / "models/Qwen3-4B-DFlash-b16"
            ),
        }
    )
    command = [
        "bash",
        str(REMOTE_ROOT / "scripts/run_fa4_benchmark.sh"),
        "--mode",
        "representative",
        "--datasets",
        "bin3_04k_08k",
        "--samples-per-dataset",
        "3",
        "--methods",
        "vanilla_hf,dflash",
        "--max-input-tokens",
        "0",
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--warmup-tokens",
        "8",
        "--repetitions",
        "1",
        "--run-id",
        f"{RUN_ID}-fa4",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
    ]
    try:
        result = _run(
            command,
            env=env,
            log_path=VOLUME_MOUNT / "runs" / RUN_ID / "logs/fa4.log",
        )
    finally:
        # Preserve report/log artifacts even when a quality or parity gate fails.
        VOLUME.commit()
    return {"backend": "fa4_native", "hardware": hardware, **result}


@app.function(
    image=VLLM_IMAGE,
    gpu=GPU_TYPE,
    cpu=16,
    memory=65536,
    timeout=7200,
    volumes={str(VOLUME_MOUNT): VOLUME},
)
def run_vllm(run_id: str = f"{RUN_ID}-vllm") -> dict[str, object]:
    os.chdir(REMOTE_ROOT)
    hardware = _gpu_snapshot()
    data_file = VOLUME_MOUNT / "runs" / RUN_ID / "input/bin3_04k_08k.jsonl"
    output_dir = VOLUME_MOUNT / "runs" / RUN_ID / "results/vllm"
    env = _base_env()
    env.update(
        {
            "VLLM_TARGET_MODEL": str(VOLUME_MOUNT / "models/Qwen3-4B"),
            "VLLM_DFLASH_MODEL": str(
                VOLUME_MOUNT / "models/Qwen3-4B-DFlash-b16"
            ),
            "VLLM_DATA_FILE": str(data_file),
            "VLLM_OUTPUT_DIR": str(output_dir),
            "VLLM_RUN_ID": run_id,
            "VLLM_METHODS": "vanilla_vllm,dflash",
            "VLLM_MAX_INPUT_TOKENS": "0",
            "VLLM_MAX_NEW_TOKENS": str(MAX_NEW_TOKENS),
            "VLLM_MAX_MODEL_LEN": str(MAX_MODEL_LEN),
            "VLLM_GPU_MEMORY_UTILIZATION": "0.75",
        }
    )
    command = [
        "bash",
        str(REMOTE_ROOT / "scripts/run_vllm_all.sh"),
        "--length-spread-samples",
        "--require-blackwell",
        "--max-samples",
        "3",
        "--max-new-tokens",
        str(MAX_NEW_TOKENS),
        "--max-input-tokens",
        "0",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--methods",
        "vanilla_vllm,dflash",
    ]
    try:
        result = _run(
            command,
            env=env,
            log_path=VOLUME_MOUNT / "runs" / RUN_ID / "logs/vllm.log",
        )
    finally:
        VOLUME.commit()
    return {"backend": "vllm", "hardware": hardware, **result}


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    cpu=2,
    memory=4096,
    timeout=600,
    volumes={str(VOLUME_MOUNT): VOLUME},
)
def bundle_results() -> str:
    source = VOLUME_MOUNT / "runs" / RUN_ID / "results"
    source.mkdir(parents=True, exist_ok=True)
    archive = VOLUME_MOUNT / "runs" / RUN_ID / "results.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as output:
        output.add(source, arcname="results")
        for name in ("logs/fa4.log", "logs/vllm.log", "input/manifest.json", "input/bin3_04k_08k.jsonl"):
            path = VOLUME_MOUNT / "runs" / RUN_ID / name
            if path.is_file():
                output.add(path, arcname=name)
    VOLUME.commit()
    return str(archive)


def _prepare_local_inputs(output_root: Path) -> Path:
    if not SOURCE_DATA.is_file():
        raise FileNotFoundError(SOURCE_DATA)
    if not (TARGET_SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"Qwen3-4B snapshot missing: {TARGET_SNAPSHOT}")
    if not (DFLASH_SNAPSHOT / "config.json").is_file():
        raise FileNotFoundError(f"DFlash snapshot missing: {DFLASH_SNAPSHOT}")

    by_id = {}
    for line in SOURCE_DATA.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        by_id[str(row.get("id", ""))] = row
    rows = [by_id[sample_id] for sample_id in SAMPLE_IDS]
    for row in rows:
        if row.get("task_type") != "summarization" or not row.get("reference_output"):
            raise ValueError(f"Sample lacks a summarization reference: {row.get('id')}")
        if row.get("length_bin_name") != "bin3_04k_08k":
            raise ValueError(f"Sample is outside bin 3: {row.get('id')}")

    input_dir = output_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = input_dir / "bin3_04k_08k.jsonl"
    serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    dataset_path.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "length-bins-v1",
        "pilot": True,
        "source_file": "data/length_bins/bin3_04k_08k.jsonl",
        "bins": {
            "bin3_04k_08k": {
                "file": dataset_path.name,
                "count": len(rows),
                "sha256": digest,
                "range": [4000, 8000],
                "sample_ids": [row["id"] for row in rows],
                "dataset_distribution": {
                    str(name): sum(row.get("dataset") == name for row in rows)
                    for name in sorted({str(row.get("dataset")) for row in rows})
                },
            }
        },
    }
    (input_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_root / "protocol.json").write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "gpu_requested": GPU_TYPE,
                "volume_name": VOLUME_NAME,
                "bin": "bin3_04k_08k",
                "sample_ids": SAMPLE_IDS,
                "sample_input_tokens": [row.get("input_tokens") for row in rows],
                "methods": {
                    "fa4_native": ["vanilla_hf", "dflash"],
                    "vllm": ["vanilla_vllm", "dflash"],
                },
                "vllm_run_id": VLLM_RUN_ID,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_model_len": MAX_MODEL_LEN,
                "input_truncation": False,
                "temperature": 0.0,
                "note": "Three paired summarization samples for pipeline/metric validation; not a comparative result.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return input_dir


@app.local_entrypoint()
def main() -> None:
    output_root = ROOT / "outputs/length_bins_metric_pilot" / RUN_ID
    output_root.mkdir(parents=True, exist_ok=True)
    input_dir = _prepare_local_inputs(output_root)
    print(f"[pilot] run_id={RUN_ID} gpu={GPU_TYPE} samples={SAMPLE_IDS}", flush=True)
    if os.environ.get("SKIP_MODAL_STAGING") == "1":
        print("[pilot] using model snapshots and input rows already staged in the volume", flush=True)
    else:
        print("[pilot] staging local model snapshots and the three paired input rows", flush=True)
        with VOLUME.batch_upload() as upload:
            upload.put_directory(input_dir, f"/runs/{RUN_ID}/input")
            upload.put_directory(TARGET_SNAPSHOT, "/models/Qwen3-4B")
            upload.put_directory(DFLASH_SNAPSHOT, "/models/Qwen3-4B-DFlash-b16")

    if os.environ.get("SKIP_FA4") == "1":
        fa4_result = {
            "backend": "fa4_native",
            "status": "reused_existing_artifacts",
            "run_id": f"{RUN_ID}-fa4",
        }
    else:
        try:
            fa4_result = run_fa4.remote()
        except Exception as exc:
            fa4_result = {
                "backend": "fa4_native",
                "status": "failed",
                "error": f"{type(exc).__name__}: {str(exc)[:800]}",
            }
    try:
            vllm_result = run_vllm.remote(run_id=VLLM_RUN_ID)
    except Exception as exc:
        vllm_result = {
            "backend": "vllm",
            "status": "failed",
            "error": f"{type(exc).__name__}: {str(exc)[:800]}",
        }
    archive_path = bundle_results.remote()
    local_archive = output_root / "results.tar.gz"
    with local_archive.open("wb") as stream:
        VOLUME.read_file_into_fileobj(
            archive_path.removeprefix(str(VOLUME_MOUNT) + "/"), stream
        )
    with tarfile.open(local_archive, "r:gz") as archive:
        archive.extractall(output_root, filter="data")
    summary = {
        "run_id": RUN_ID,
        "gpu_requested": GPU_TYPE,
        "fa4": fa4_result,
        "vllm": vllm_result,
        "archive": str(local_archive),
    }
    (output_root / "modal_execution.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[pilot] local artifacts: {output_root}", flush=True)
