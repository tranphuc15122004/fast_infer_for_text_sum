#!/usr/bin/env python3
"""Rerun the Context-Adaptive DFlash length-bin pilot with FA4 on Modal H100.

The selected runtime pins are copied from the repository server manifest.
The H100 uses Hopper FA4 kernels while keeping the server's Python/cu130,
Torch, Transformers, CUTLASS DSL, and FlashAttention-4 versions.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
import tarfile

import modal


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / "outputs/cadflash_lengthbins_modal_20261009"
STAGE_ROOT = PREVIOUS / "assets"
OUTPUT_ROOT = ROOT / "outputs/cadflash_lengthbins_modal_fa4_h100_20261009"
ASSET_VOLUME_NAME = "cadflash-lengthbins-20261009"
TARGET_VOLUME_NAME = "fast-infer-text-sum-mr-dflash-timer"
RUN_ID = "cadflash_lengthbins_modal_fa4_h100_20261009_001"
ASSET_ROOT = Path("/assets/assets")
TARGET_ROOT = Path("/models/models/Qwen3-4B")
REPO_ROOT = Path("/workspace/fast_infer_text_sum")
RUN_ROOT = ASSET_ROOT / "runs" / RUN_ID
ARCHIVE_PATH = ASSET_ROOT / f"{RUN_ID}_results_bundle.tar.gz"
DATA_FILE = ASSET_ROOT / "inputs/lengthbin_pilot.jsonl"
SPLIT_FILE = ASSET_ROOT / "split_manifest.json"

VARIANTS = (
    "ar",
    "dflash_full_fixed",
    "a_only",
    "b_only_history",
    "b_only_entropy",
    "independent_ab",
    "joint_no_entropy",
    "joint_no_source_relevance",
    "joint",
)
VARIANT_ORDER = list(VARIANTS)
random.Random(42).shuffle(VARIANT_ORDER)


# Exact pins copied from requirements.txt for the Qwen + DFlash + FA4 path.
# The server-only file-URL wheels and unrelated baseline packages are omitted.

RUNTIME_REQUIREMENTS = (
    'accelerate==1.15.0',
    'apache-tvm-ffi==0.1.11',
    'cuda-bindings==13.4.1',
    'cuda-core==1.2.0',
    'cuda-pathfinder==1.8.1',
    'cuda-python==13.4.1',
    'cuda-toolkit==13.0.3.0',
    'einops==0.8.2',
    'flash-attn-4==4.0.0b19',
    'huggingface_hub==1.31.0',
    'ninja==1.13.0',
    'numpy==2.3.5',
    'nvidia-cublas==13.1.1.3',
    'nvidia-cuda-cccl==13.3.4.3.1',
    'nvidia-cuda-crt==13.4.59',
    'nvidia-cuda-cupti==13.0.85',
    'nvidia-cuda-nvdisasm==13.4.92',
    'nvidia-cuda-nvrtc==13.0.88',
    'nvidia-cuda-runtime==13.0.96',
    'nvidia-cuda-tileiras==13.2.86',
    'nvidia-cudnn-cu13==9.20.0.48',
    'nvidia-cudnn-frontend==1.28.0',
    'nvidia-cufft==12.0.0.61',
    'nvidia-cufile==1.15.1.6',
    'nvidia-curand==10.4.0.35',
    'nvidia-cusolver==12.0.4.66',
    'nvidia-cusparse==12.6.3.3',
    'nvidia-cusparselt-cu13==0.8.1',
    'nvidia-cuda-nvcc==13.2.86',
    'nvidia-cutlass-dsl==4.7.1',
    'nvidia-cutlass-dsl-libs-base==4.7.1',
    'nvidia-cutlass-dsl-libs-core==4.7.1',
    'nvidia-cutlass-dsl-libs-cu13==4.7.1',
    'nvidia-mathdx==25.6.0',
    'nvidia-nccl-cu13==2.29.7',
    'nvidia-nvjitlink==13.0.88',
    'nvidia-nvvm==13.2.86',
    'nvidia-nvtx==13.0.85',
    'packaging==26.3',
    'psutil==7.2.2',
    'PyYAML==6.0.3',
    'quack-kernels==0.6.5',
    'safetensors==0.8.0',
    'tokenizers==0.22.2',
    'torch==2.13.0',
    'tqdm==4.70.1',
    'transformers==5.12.1',
    'triton==3.7.1',
)
PREREGISTRATION_TEXT = """# Context-Adaptive DFlash: FA4/H100 rerun

- GPU: Modal H100 (Hopper), selected to run the same FlashAttention-4 backend at lower GPU cost than B200.
- Runtime: Python 3.12, exact method-path pins from `requirements.txt`; Torch 2.13.0 / CUDA 13.0, Transformers 5.12.1, FlashAttention-4 4.0.0b19, CUTLASS DSL 4.7.1.
- Dataset/split: reuse the previous source-group-disjoint seed-42 sample manifest, 30 calibration documents and 15 dev documents (3 dev documents per length bin; gov_report, multi_news, qmsum).
- Decoding: greedy, natural EOS, max-new-tokens 2048, 3 repetitions and 3 warmups; this replaces the previous fixed 128-token cap.
- Variants: AR, DFlash full-context gamma 15, A/B ablations, and joint policy.
- Throughput speedup: mean request-level `tok/s(candidate) / tok/s(baseline)`. Report E2E latency separately.
- Require a successful FA4 import probe, direct FA4 kernel probe, and AR/DFlash model smoke with runtime manifest backend exactly `flash_attention_4`; fail before calibration otherwise.
- GPU order: randomized fixed variant-block order recorded in `modal_hardware.json`; each variant keeps its three paired repetitions together.
- Limitations: three independent dev documents per bin, one GPU family, and the existing DFlash-vs-AR token parity issue remains a separate validity gate.
"""


app = modal.App("cadflash-lengthbin-fa4-h100-20261009")
asset_volume = modal.Volume.from_name(ASSET_VOLUME_NAME)
target_volume = modal.Volume.from_name(TARGET_VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*RUNTIME_REQUIREMENTS)
    .add_local_dir(ROOT / "src/TrainingFree", remote_path=str(REPO_ROOT / "src/TrainingFree"), copy=True)
    .add_local_dir(ROOT / "scripts", remote_path=str(REPO_ROOT / "scripts"), copy=True)
    .add_local_dir(ROOT / "externals/dflash", remote_path=str(REPO_ROOT / "externals/dflash"), copy=True)
    .env(
        {
            "PYTHONPATH": f"{REPO_ROOT / 'src'}:{REPO_ROOT / 'scripts'}:{REPO_ROOT / 'externals/dflash'}",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "CAD_ATTENTION_BACKEND": "flash_attention_4",
            "TRITON_CACHE_DIR": "/tmp/cadflash_fa4/triton",
            "TORCH_EXTENSIONS_DIR": "/tmp/cadflash_fa4/torch_extensions",
        }
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_manifest(root: Path) -> dict[str, object]:
    files = []
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_hash = _sha256(path)
        files.append({"path": relative, "bytes": size, "sha256": file_hash})
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
    return {"file_count": len(files), "content_sha256": digest.hexdigest(), "files": files}


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    cpu=4,
    memory=8192,
    timeout=900,
    volumes={"/assets": asset_volume, "/models": target_volume},
)
def verify_target_volume() -> dict[str, object]:
    expected_path = ASSET_ROOT / "expected_target_signature.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = _tree_manifest(TARGET_ROOT)
    actual_files = {entry["path"]: entry for entry in actual["files"]}
    mismatches = [entry for entry in expected["files"] if actual_files.get(entry["path"]) != entry]
    if mismatches:
        raise RuntimeError(f"Target checkpoint volume hash mismatch: {mismatches[:3]}")
    result = {
        "matches_local_snapshot": True,
        "actual": actual,
        "expected_content_sha256": expected.get("content_sha256"),
    }
    (ASSET_ROOT / "target_volume_verification_fa4_h100.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    asset_volume.commit()
    print(f"Verified target volume SHA-256: {actual['content_sha256']}", flush=True)
    return result


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=14400,
    volumes={"/assets": asset_volume, "/models": target_volume},
)
def run_experiment() -> str:
    import importlib.metadata as metadata
    import os
    import subprocess
    import sys

    import torch

    os.chdir(REPO_ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on the Modal H100 worker")
    gpu_name = torch.cuda.get_device_name(0)
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(0))
    if "H100" not in gpu_name or capability[0] != 9:
        raise RuntimeError(f"Expected Hopper H100 for FA4; got {gpu_name} capability={capability}")
    if str(torch.version.cuda) != "13.0":
        raise RuntimeError(f"Expected Torch CUDA 13.0 from server manifest; got {torch.version.cuda}")

    expected_versions = {
        "torch": "2.13.0",
        "transformers": "5.12.1",
        "accelerate": "1.15.0",
        "huggingface-hub": "1.31.0",
        "flash-attn-4": "4.0.0b19",
        "nvidia-cutlass-dsl": "4.7.1",
        "nvidia-cutlass-dsl-libs-cu13": "4.7.1",
        "numpy": "2.3.5",
        "safetensors": "0.8.0",
        "tokenizers": "0.22.2",
    }
    actual_versions = {name: metadata.version(name) for name in expected_versions}
    for name, expected in expected_versions.items():
        actual = actual_versions[name].split("+", 1)[0]
        if actual != expected:
            raise RuntimeError(f"Runtime version mismatch for {name}: expected {expected}, got {actual}")

    import sys as sys_module

    sys_module.path.insert(0, str(REPO_ROOT / "scripts"))
    from common.vanilla_inference import (
        _install_flash_attention_4_cutlass_compat,
        _probe_flash_attention_4,
    )

    _install_flash_attention_4_cutlass_compat()
    fa4_available, fa4_reason = _probe_flash_attention_4()
    if not fa4_available:
        raise RuntimeError(f"FlashAttention-4 import probe failed: {fa4_reason}")
    from flash_attn.cute import flash_attn_func

    q = torch.randn((1, 128, 8, 128), device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    fa4_start = torch.cuda.Event(enable_timing=True)
    fa4_end = torch.cuda.Event(enable_timing=True)
    fa4_start.record()
    with torch.inference_mode():
        fa4_result = flash_attn_func(q, q, q, causal=True)
    fa4_end.record()
    torch.cuda.synchronize()
    # FA4's CuTe API returns the output together with auxiliary data.
    fa4_out = fa4_result[0] if isinstance(fa4_result, tuple) else fa4_result
    if not isinstance(fa4_out, torch.Tensor):
        raise RuntimeError(f"FlashAttention-4 returned no output tensor: {type(fa4_result)!r}")
    if tuple(fa4_out.shape) != tuple(q.shape) or not torch.isfinite(fa4_out).all().item():
        raise RuntimeError("FlashAttention-4 kernel probe returned invalid output")

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "MODEL_TARGET": str(TARGET_ROOT),
            "MODEL_DFLASH_DRAFT": str(ASSET_ROOT / "models/Qwen3-4B-DFlash-b16"),
            "PYTHONPATH": ":".join(
                [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts"), str(REPO_ROOT / "externals/dflash")]
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CAD_ATTENTION_BACKEND": "flash_attention_4",
            "CAD_SELECTOR": "target_parent",
            "CAD_TARGET_LAYER": "middle",
            "CAD_BUDGETS": "1024,full",
            "CAD_GAMMAS": "3,7,15",
            "CAD_REFRESH_PERIOD": "4",
            "CAD_SIGNAL_UPDATE_PERIOD": "1",
            "CAD_SOURCE_CHUNK_SIZE": "128",
            "CAD_SOURCE_ANCHORS": "64",
            "CAD_RECENT_OUTPUT": "256",
            "CAD_PRIOR_STRENGTH": "16",
            "CAD_MIN_STATE_SUPPORT": "8",
            "CAD_STATISTICS_DECAY": "0.95",
            "CAD_HISTORY_ALPHA": "0.1",
            "CAD_LATENCY_EMA_ALPHA": "0.1",
            "CAD_CONTROLLER_MARGIN": "0.02",
            "CAD_ENTROPY_SIGNAL_TEMPERATURE": "1",
            "CAD_TIMING_MODE": "wall",
            "CAD_COST_UPDATE_MODE": "frozen_cost",
            "CAD_STATISTICS_UPDATE_MODE": "online",
            "CAD_LENGTH_MODE": "draft_shape",
        }
    )
    resuming_existing_run = RUN_ROOT.exists()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "run_id": RUN_ID,
        "gpu": "Modal H100 (Hopper, FA4-compatible)",
        "attention_backend": "flash_attention_4 (required; no fallback)",
        "runtime_manifest": "requirements.txt exact pins for the Qwen/DFlash/FA4 path",
        "max_new_tokens": 2048,
        "output_scope": "natural_eos",
        "dev_documents": 15,
        "calibration_documents": 30,
        "repetitions": 3,
        "warmup_runs": 3,
        "variant_order": list(VARIANT_ORDER),
        "seed": 42,
        "variants": list(VARIANTS),
        "throughput_speedup_definition": "mean tok/s of candidate divided by mean tok/s of baseline",
    }
    evidence_root = RUN_ROOT / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    preregistration_path = evidence_root / "preregistration.json"
    if preregistration_path.exists():
        old_preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
        for key in ("run_id", "gpu", "attention_backend", "max_new_tokens", "dev_documents", "repetitions", "variants"):
            if old_preregistration.get(key) != preregistration.get(key):
                raise RuntimeError(
                    f"Existing run preregistration mismatch for {key}: "
                    f"{old_preregistration.get(key)!r} != {preregistration.get(key)!r}"
                )
    else:
        preregistration_path.write_text(json.dumps(preregistration, indent=2) + "\n", encoding="utf-8")
    preregistration_md_path = evidence_root / "preregistration.md"
    if not preregistration_md_path.exists():
        preregistration_md_path.write_text(PREREGISTRATION_TEXT, encoding="utf-8")
    hardware = {
        "gpu_name": gpu_name,
        "gpu_capability": list(capability),
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version.split()[0],
        "runtime_versions": actual_versions,
        "attention_backend_requested": "flash_attention_4",
        "fa4_import_probe": {"ok": fa4_available, "reason": fa4_reason},
        "fa4_kernel_probe_ms": round(fa4_start.elapsed_time(fa4_end), 4),
        "gpu_driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "requirements_pins_used": list(RUNTIME_REQUIREMENTS),
        "requirements_pins_omitted": [
            "cuda-tile", "flashinfer-cubin", "flashinfer-python", "sglang", "vllm",
            "other baseline-only packages not imported by this runner",
        ],
        "run_id": RUN_ID,
        "data_sha256": _sha256(DATA_FILE),
        "split_sha256": _sha256(SPLIT_FILE),
        "variant_order": list(VARIANT_ORDER),
    }
    hardware_path = evidence_root / "modal_hardware.json"
    if hardware_path.exists():
        old_hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
        if old_hardware.get("attention_backend_requested") != "flash_attention_4":
            raise RuntimeError("Existing run evidence does not record FlashAttention-4")
        if old_hardware.get("data_sha256") != hardware["data_sha256"] or old_hardware.get("split_sha256") != hardware["split_sha256"]:
            raise RuntimeError("Existing run evidence points to different data or split artifacts")
    else:
        hardware_path.write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
    if resuming_existing_run:
        resume_log = evidence_root / "resume_events.jsonl"
        with resume_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"resumed_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(), "reason": "previous Modal function was canceled", "gpu_name": gpu_name, "attention_backend": "flash_attention_4"}) + "\n")
    asset_volume.commit()

    common = [
        "--target-model", str(TARGET_ROOT),
        "--draft-model", env["MODEL_DFLASH_DRAFT"],
        "--data-file", str(DATA_FILE),
        "--split-manifest", str(SPLIT_FILE),
        "--run-id", RUN_ID,
        "--output-root", str(RUN_ROOT),
        "--temperature", "0",
        "--max-input-tokens", "0",
        "--max-new-tokens", "2048",
        "--fixed-budget", "full",
        "--fixed-gamma", "15",
        "--gamma-reference", "15",
        "--budgets", "1024,full",
        "--gammas", "3,7,15",
        "--seed", "42",
    ]
    cli = str(REPO_ROOT / "scripts/infer_context_adaptive_dflash.py")

    def run_cli(args: list[str]) -> None:
        command = [sys.executable, cli, *args]
        print("[cadflash-fa4-h100] " + " ".join(command), flush=True)
        try:
            subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        except BaseException:
            asset_volume.commit()
            raise
        asset_volume.commit()

    preflight_path = RUN_ROOT / "preflight.json"
    if not preflight_path.exists():
        run_cli(["--phase", "preflight", *common, "--model-check"])
    for variant in ("ar", "dflash_full_fixed"):
        manifest = RUN_ROOT / "smoke/smoke" / variant / "online/manifest.json"
        if not manifest.exists():
            run_cli(
                [
                    "--phase", "smoke", "--variant", variant, *common,
                    "--output-root", str(RUN_ROOT / "smoke"),
                    "--max-samples", "1", "--max-new-tokens", "16",
                    "--fixed-output-tokens", "16", "--repetitions", "1", "--warmup-runs", "0",
                ]
            )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        actual_backend = payload.get("runtime", {}).get("attention_backend")
        if actual_backend != "flash_attention_4":
            raise RuntimeError(f"{variant} loaded backend {actual_backend!r}; expected flash_attention_4")

    calibration = str(RUN_ROOT / "calibration/calibration.json")
    calibration_path = Path(calibration)
    if not calibration_path.exists():
        observations_path = calibration_path.parent / "action_observations.jsonl"
        if observations_path.exists():
            raise RuntimeError("Calibration is incomplete and cannot be resumed safely from its current partial artifacts")
        run_cli(
            [
                "--phase", "calibrate", "--variant", "dflash_full_fixed", *common,
                "--calibration-checkpoints", "0,64", "--repetitions", "3", "--warmup-runs", "3",
            ]
        )

    def variant_is_complete(variant: str) -> bool:
        for repetition in range(3):
            request_path = RUN_ROOT / "dev" / variant / "online" / f"rep_{repetition}" / "requests.jsonl"
            if not request_path.is_file():
                return False
            try:
                records = [json.loads(line) for line in request_path.read_text(encoding="utf-8").splitlines() if line]
            except (OSError, json.JSONDecodeError):
                return False
            if not records or records[-1].get("type") != "summary":
                return False
            summary = records[-1]
            successes = sum(row.get("type") == "sample" and row.get("status") == "ok" for row in records[:-1])
            if summary.get("status") != "complete" or successes != 15 or summary.get("requested_samples") != 15:
                return False
        return True

    for variant in VARIANT_ORDER:
        if variant_is_complete(variant):
            print(f"[cadflash-fa4-h100] SKIP complete dev variant={variant}", flush=True)
            continue
        print(f"[cadflash-fa4-h100] START dev variant={variant}", flush=True)
        run_cli(
            [
                "--phase", "dev", "--variant", variant, *common,
                "--calibration-file", calibration,
                "--repetitions", "3", "--warmup-runs", "3", "--resume",
            ]
        )
    run_cli(
        [
            "--phase", "report", "--output-root", str(RUN_ROOT),
            "--report-phase", "dev", "--report-statistics-mode", "online",
            "--bootstrap-samples", "1000",
        ]
    )

    with tarfile.open(ARCHIVE_PATH, "w:gz") as bundle:
        bundle.add(RUN_ROOT, arcname=f"runs/{RUN_ID}")
        bundle.add(RUN_ROOT / "evidence/preregistration.md", arcname="preregistration.md")
        for filename in ("sample_manifest.json", "split_manifest.json"):
            path = STAGE_ROOT / filename
            if path.is_file():
                bundle.add(path, arcname=filename)
    asset_volume.commit()
    print(json.dumps(hardware, indent=2), flush=True)
    return str(ARCHIVE_PATH)


@app.local_entrypoint()
def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    local_assets = OUTPUT_ROOT / "assets"
    local_assets.mkdir(exist_ok=True)
    for filename in ("sample_manifest.json", "split_manifest.json", "preregistration.md"):
        source = STAGE_ROOT / filename
        if source.is_file() and filename != "preregistration.md":
            (local_assets / filename).write_bytes(source.read_bytes())
    (local_assets / "preregistration.md").write_text(PREREGISTRATION_TEXT, encoding="utf-8")
    (local_assets / "requirements_pins.json").write_text(
        json.dumps({"requirements_file": "requirements.txt", "pins_used": list(RUNTIME_REQUIREMENTS)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    verification = verify_target_volume.remote()
    if not verification.get("matches_local_snapshot"):
        raise RuntimeError("Modal target model volume differs from its local pinned signature")
    archive_path = run_experiment.remote()
    print(f"Experiment finished; retrieve archive from Modal volume: {archive_path}")
