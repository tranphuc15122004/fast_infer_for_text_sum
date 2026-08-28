"""Registry and adapters for the canonical LongBench 9-baseline matrix.

The upstream projects do not share one command line or one input format.  This
module keeps that translation in one place so the orchestrator can remain a
small, reproducible experiment runner.  Adapters only prepare files and
commands; they do not import CUDA kernels or load model weights during
preflight.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.paths import ROOT


BASELINES = (
    "vanilla_hf",
    "vanilla_fa",
    "magicdec",
    "longspec",
    "eagle3",
    "dflash",
    "specextend",
    "sssd",
    "fafo",
)

CUDA_BASELINES = set(BASELINES)


def _path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return path


def convert_records_for_baseline(
    baseline: str,
    records: Sequence[Mapping[str, Any]],
    output: Path,
) -> Path | None:
    """Convert canonical rows when an upstream runner needs another shape.

    ``None`` means the canonical JSONL can be passed directly.  Conversion
    files are intentionally kept under the run directory and therefore become
    part of the run artifact without modifying ``data/longbench_200``.
    """

    if baseline == "eagle3":
        return _write_jsonl(
            output,
            [
                {
                    "question_id": str(row["id"]),
                    "turns": [str(row.get("prompt", ""))],
                    "answer": row.get("reference") or "",
                    "task_type": row.get("raw", {}).get("task_type") or row.get("task_type"),
                }
                for row in records
            ],
        )

    if baseline == "specextend":
        # run_classic.py/run_eagle.py consume only ``text``.  Keep IDs and
        # references as extra fields so a future upstream JSON writer can
        # recover the source mapping without another conversion step.
        return _write_jsonl(
            output,
            [
                {
                    "id": row["id"],
                    "text": str(row.get("prompt", "")),
                    "reference": row.get("reference"),
                    "task_type": row.get("task_type"),
                }
                for row in records
            ],
        )

    # SSSD and FAFO have their own converters in infer_sssd.py/infer_fafo.py;
    # dflash, LongSpec and vanilla adapters consume the canonical loader.
    return None


def _local_requirement(value: str | None) -> tuple[bool, str | None]:
    """Return whether a configured path-like requirement exists.

    Hugging Face repo IDs (for example ``org/model``) are not rejected here:
    the offline cache may contain them even though the ID itself is not a
    filesystem path.  Absolute paths and explicit ``./``/``../`` paths are
    checked before a child process starts.
    """

    if not value:
        return False, "not configured"
    raw = str(value)
    candidate = Path(raw)
    path_like = candidate.is_absolute() or raw.startswith(("./", "../"))
    if not path_like:
        return True, None
    if candidate.exists():
        return True, None
    return False, f"path not found: {raw}"


def _model(config: Mapping[str, Any]) -> str | None:
    return str(config.get("model") or "") or None


def preflight_baseline(
    baseline: str,
    config: Mapping[str, Any] | None = None,
    *,
    device: str | None = None,
    cuda_available: bool | None = None,
) -> dict[str, Any]:
    """Check cheap, deterministic prerequisites without loading a model.

    The returned object is JSON-safe and is embedded in ``run_manifest.json``.
    Missing optional upstream dependencies are reported as data rather than
    raised exceptions, allowing a smoke matrix to explain every cell.
    """

    if baseline not in BASELINES:
        return {
            "status": "invalid_baseline",
            "reason": f"unknown baseline: {baseline}",
            "requires_cuda": None,
        }

    cfg = dict(config or {})
    if cuda_available is None and device is not None and str(device).lower().startswith("cpu"):
        cuda_available = False
    if cuda_available is None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False

    result: dict[str, Any] = {
        "baseline": baseline,
        "requires_cuda": baseline in CUDA_BASELINES,
        "cuda_available": bool(cuda_available),
        "status": "ready",
        "reason": None,
        "requirements": {},
    }

    if baseline in CUDA_BASELINES and not cuda_available:
        result.update(status="unsupported_cpu", reason="CUDA is unavailable")
        # Continue collecting path/dependency facts for a useful manifest.

    model = _model(cfg)
    model_ok, model_reason = _local_requirement(model)
    result["requirements"]["target_model"] = {
        "configured": bool(model),
        "available": model_ok,
        "reason": model_reason,
    }

    if baseline == "vanilla_fa":
        installed = importlib.util.find_spec("flash_attn") is not None
        result["requirements"]["flash_attn"] = {"available": installed}
        if not installed and result["status"] == "ready":
            result.update(
                status="missing_dependency",
                reason="flash_attn is required by vanilla_fa; no fallback is allowed",
            )

    if baseline == "eagle3":
        draft_ok, draft_reason = _local_requirement(
            str(cfg.get("eagle_model") or "") or None
        )
        result["requirements"]["eagle_model"] = {
            "configured": bool(cfg.get("eagle_model")),
            "available": draft_ok,
            "reason": draft_reason,
        }
        result["requirements"]["source"] = {
            "available": (ROOT / "externals" / "EAGLE").is_dir()
        }

    if baseline == "dflash":
        draft_ok, draft_reason = _local_requirement(
            str(cfg.get("dflash_model") or "") or None
        )
        result["requirements"]["dflash_model"] = {
            "configured": bool(cfg.get("dflash_model")),
            "available": draft_ok,
            "reason": draft_reason,
        }
        result["requirements"]["package"] = {
            "available": (ROOT / "externals" / "dflash").is_dir()
        }
        if not result["requirements"]["package"]["available"] and result["status"] == "ready":
            result.update(status="missing_dependency", reason="vendored dflash source is missing")

    if baseline == "longspec":
        for key, label in (
            ("longspec_target_model", "target_model"),
            ("longspec_draft_model", "draft_model"),
        ):
            ok, reason = _local_requirement(str(cfg.get(key) or "") or None)
            result["requirements"][label] = {
                "configured": bool(cfg.get(key)),
                "available": ok,
                "reason": reason,
            }
        result["requirements"]["source"] = {
            "available": (ROOT / "externals" / "LongSpec").is_dir()
        }
        if not result["requirements"]["source"]["available"] and result["status"] == "ready":
            result.update(status="missing_dependency", reason="vendored LongSpec source is missing")

    if baseline == "specextend":
        draft_ok, draft_reason = _local_requirement(
            str(cfg.get("specextend_draft_model") or "") or None
        )
        result["requirements"]["draft_model"] = {
            "configured": bool(cfg.get("specextend_draft_model")),
            "available": draft_ok,
            "reason": draft_reason,
        }
        result["requirements"]["source"] = {
            "available": (ROOT / "externals" / "SpecExtend").is_dir()
        }
        if not result["requirements"]["source"]["available"] and result["status"] == "ready":
            result.update(status="missing_dependency", reason="vendored SpecExtend source is missing")

    if baseline == "sssd":
        datastore = str(cfg.get("sssd_datastore_path") or "") or None
        if datastore:
            ok, reason = _local_requirement(datastore)
            result["requirements"]["datastore"] = {
                "configured": True,
                "available": ok,
                "reason": reason,
            }
            if not ok and result["status"] == "ready":
                result.update(status="missing_checkpoint", reason=reason)
        else:
            result["requirements"]["datastore"] = {
                "configured": False,
                "available": False,
                "reason": "empty datastore is allowed; using prompt/self-output-only retrieval",
            }
            if result["status"] == "ready":
                result.update(
                    status="aggregate_only",
                    reason="SSSD datastore is empty; using prompt/self-output-only retrieval",
                )

    if baseline == "magicdec":
        checkpoint = str(cfg.get("magicdec_model_pth") or "") or None
        ok, reason = _local_requirement(checkpoint)
        result["requirements"]["checkpoint"] = {
            "configured": bool(checkpoint),
            "available": ok,
            "reason": reason,
        }
        result["requirements"]["source"] = {
            "available": (ROOT / "externals" / "MagicDec").is_dir()
        }
        result["requirements"]["flashinfer"] = {
            "available": importlib.util.find_spec("flashinfer") is not None
        }
        if not result["requirements"]["source"]["available"] and result["status"] == "ready":
            result.update(status="missing_dependency", reason="vendored MagicDec source is missing")
        elif not result["requirements"]["flashinfer"]["available"] and result["status"] == "ready":
            result.update(status="missing_dependency", reason="flashinfer is required by MagicDec SnapKV")

    if baseline in {"sssd", "fafo"} and result["status"] == "ready":
        result.update(
            status="aggregate_only",
            reason="upstream runner emits aggregate timing; per-sample timing is unavailable",
        )

    # A configured path that is explicitly missing is more actionable than a
    # generic ready status.  Preserve unsupported_cpu as the primary cause on
    # a CPU host so smoke output clearly explains why no model was loaded.
    if result["status"] == "ready" and not model_ok:
        result.update(
            status="missing_checkpoint",
            reason=model_reason if model else "target model is not configured",
        )

    return result


def build_adapter_command(
    baseline: str,
    config: Mapping[str, Any] | None = None,
    data_file: Path | None = None,
    output: Path | None = None,
    mode: str | None = None,
    *,
    max_samples: int = 1,
    max_new_tokens: int = 64,
    converted_input: Path | None = None,
) -> list[str] | None:
    """Build the child-process command for one matrix cell.

    MagicDec uses the canonical-prompt branch in ``infer_magicdec.py``; its
    converted checkpoint is still supplied separately from the HF tokenizer.
    """

    if baseline not in BASELINES:
        raise ValueError(f"unknown baseline: {baseline}")
    cfg = dict(config or {})
    if data_file is None or output is None:
        raise ValueError("data_file and output are required")
    if mode is not None:
        cfg.setdefault("smoke", mode == "smoke")
    python = str(cfg.get("python") or sys.executable)
    input_file = converted_input or data_file
    smoke = bool(cfg.get("smoke"))
    max_input = int(cfg.get("max_input_tokens", 0) or 0)
    temperature = str(cfg.get("temperature", 0))
    warmup = str(cfg.get("warmup_runs", 3))
    seed = str(cfg.get("seed", 42))

    common = [
        "--max-samples",
        str(max_samples),
        "--max-new-tokens",
        str(max_new_tokens),
        "--output",
        str(output),
    ]

    if baseline == "vanilla_hf":
        return [
            python,
            str(ROOT / "scripts" / "infer_vanilla_hf.py"),
            "--model",
            str(cfg.get("model") or ""),
            "--data-file",
            str(data_file),
            "--temperature",
            temperature,
            "--seed",
            seed,
            "--warmup-runs",
            warmup,
            "--max-input-tokens",
            str(max_input),
            "--device",
            str(cfg.get("device", "cuda")),
            "--dtype",
            str(cfg.get("dtype", "bfloat16")),
            "--local-files-only" if bool(cfg.get("local_files_only", True)) else "--no-local-files-only",
            *common,
        ] + (["--smoke"] if smoke else [])

    if baseline == "vanilla_fa":
        command = build_adapter_command(
            "vanilla_hf",
            data_file=data_file,
            output=output,
            max_samples=max_samples,
            max_new_tokens=max_new_tokens,
            config={**cfg, "smoke": smoke},
        )
        assert command is not None
        command[1] = str(ROOT / "scripts" / "infer_vanilla_fa.py")
        # The parser enforces flash_attention_2 from the wrapper default.
        return command

    if baseline == "magicdec":
        return [
            python,
            str(ROOT / "scripts" / "infer_magicdec.py"),
            "--model-pth",
            str(cfg.get("magicdec_model_pth") or ""),
            "--model-name",
            str(cfg.get("magicdec_model_name") or cfg.get("model") or ""),
            "--data-file",
            str(data_file),
            "--max-samples",
            str(max_samples),
            "--max-new-tokens",
            str(max_new_tokens),
            "--max-input-tokens",
            str(max_input),
            "--temperature",
            temperature,
            "--seed",
            seed,
            "--warmup-runs",
            warmup,
            "--local-files-only" if bool(cfg.get("local_files_only", True)) else "--no-local-files-only",
            "--output",
            str(output),
        ] + (["--smoke"] if smoke else [])

    if baseline == "dflash":
        return [
            python,
            str(ROOT / "scripts" / "infer_dflash.py"),
            "--target-model",
            str(cfg.get("model") or ""),
            "--draft-model",
            str(cfg.get("dflash_model") or ""),
            "--data-file",
            str(data_file),
            "--max-samples",
            str(max_samples),
            "--max-new-tokens",
            str(max_new_tokens),
            "--max-input-tokens",
            str(max_input),
            "--temperature",
            temperature,
            "--output",
            str(output),
        ] + (["--smoke"] if smoke else [])

    if baseline == "longspec":
        command = [
            python,
            str(ROOT / "scripts" / "infer_longspec.py"),
            "--target-model",
            str(cfg.get("longspec_target_model") or cfg.get("model") or ""),
            "--draft-model",
            str(cfg.get("longspec_draft_model") or ""),
            "--model-name",
            str(cfg.get("longspec_model_name", "llama8b")),
            "--data-file",
            str(data_file),
            "--max-samples",
            str(max_samples),
            "--max-gen-len",
            str(max_new_tokens),
            "--output",
            str(output),
        ]
        if max_input > 0:
            command += ["--max-input-tokens", str(max_input)]
        if smoke:
            command.append("--smoke")
        return command

    if baseline == "eagle3":
        return [
            python,
            str(ROOT / "scripts" / "eagle3_infer_qwen3.py"),
            "--base-model",
            str(cfg.get("model") or ""),
            "--eagle-model",
            str(cfg.get("eagle_model") or ""),
            "--question-file",
            str(input_file),
            "--question-begin",
            "0",
            "--question-end",
            str(max_samples),
            "--max-new-tokens",
            str(max_new_tokens),
            "--max-input-tokens",
            str(max_input),
            "--total-token",
            str(cfg.get("eagle_total_token", 32)),
            "--depth",
            str(cfg.get("eagle_depth", 8)),
            "--top-k",
            str(cfg.get("eagle_top_k", 4)),
            "--temperature",
            temperature,
            "--output",
            str(output),
        ] + (["--smoke"] if smoke else [])

    if baseline == "specextend":
        return [
            python,
            str(ROOT / "scripts" / "infer_specextend.py"),
            "--script",
            str(cfg.get("specextend_script", "run_eagle.py")),
            "--model-name",
            str(cfg.get("specextend_model_name", "llama3_1_8b")),
            "--base-model",
            str(cfg.get("model") or ""),
            "--draft-model",
            str(cfg.get("specextend_draft_model") or ""),
            "--input-file",
            str(input_file),
            "--max-samples",
            str(max_samples),
            "--max-gen-len",
            str(max_new_tokens),
            "--max-input-tokens",
            str(max_input),
            "--warmup-runs",
            warmup,
            "--output",
            str(output),
            "--use-specextend",
        ] + (["--smoke"] if smoke else [])

    if baseline == "sssd":
        command = [
            python,
            str(ROOT / "scripts" / "infer_sssd.py"),
            "--model",
            str(cfg.get("model") or ""),
            "--data-file",
            str(data_file),
            "--max-samples",
            str(max_samples),
            "--max-new-tokens",
            str(max_new_tokens),
            "--datastore-path",
            str(cfg.get("sssd_datastore_path") or ""),
            "--num-draft-tokens",
            str(cfg.get("sssd_num_draft_tokens", 8)),
            "--num-steps",
            str(cfg.get("sssd_num_steps", 5)),
            "--topk",
            str(cfg.get("sssd_topk", 5)),
            "--output",
            str(output),
        ]
        if bool(cfg.get("sssd_adaptive")):
            command.append("--adaptive")
        if smoke:
            command.append("--smoke")
        return command

    if baseline == "fafo":
        command = [
            python,
            str(ROOT / "scripts" / "infer_fafo.py"),
            "--model",
            str(cfg.get("model") or ""),
            "--data-file",
            str(data_file),
            "--max-samples",
            str(max_samples),
            "--max-new-tokens",
            str(max_new_tokens),
            "--kv-method",
            str(cfg.get("fafo_kv_method", "stream-llm")),
            "--output",
            str(output),
        ]
        command.append("--use-flash" if bool(cfg.get("fafo_use_flash")) else "--no-use-flash")
        if smoke:
            command.append("--smoke")
        return command

    return None


def baseline_config_from_env(baseline: str, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build the adapter-specific config from master-loaded environment."""

    values = dict(env or os.environ)
    return {
        "python": values.get("FAST_INFER_PYTHON", sys.executable),
        "model": values.get("LONG_BENCH_MODEL") or values.get("MODEL_TARGET"),
        "device": values.get("LONG_BENCH_DEVICE", "cuda"),
        "dtype": values.get("LONG_BENCH_DTYPE", "bfloat16"),
        "local_files_only": values.get("LONG_BENCH_LOCAL_FILES_ONLY", "1") == "1",
        "temperature": float(values.get("LONG_BENCH_TEMPERATURE", "0")),
        "warmup_runs": int(values.get("LONG_BENCH_WARMUP_RUNS", "3")),
        "seed": int(values.get("LONG_BENCH_SEED", "42")),
        "max_input_tokens": int(values.get("LONG_BENCH_MAX_INPUT_TOKENS", "0")),
        "smoke": values.get("LONG_BENCH_MODE", "smoke") == "smoke",
        "eagle_model": values.get("LONG_BENCH_EAGLE_MODEL") or values.get("MODEL_EAGLE_DRAFT"),
        "dflash_model": values.get("LONG_BENCH_DFLASH_MODEL") or values.get("MODEL_DFLASH_DRAFT"),
        "longspec_target_model": values.get("LONG_BENCH_LONGSPEC_TARGET_MODEL") or values.get("MODEL_TARGET"),
        "longspec_draft_model": values.get("LONG_BENCH_LONGSPEC_DRAFT_MODEL") or values.get("MODEL_LONGSPEC_DRAFT"),
        "longspec_model_name": values.get("LONG_BENCH_LONGSPEC_MODEL_NAME", "llama8b"),
        "specextend_draft_model": values.get("LONG_BENCH_SPECEXTEND_DRAFT_MODEL") or values.get("MODEL_EAGLE_DRAFT"),
        "specextend_model_name": values.get("LONG_BENCH_SPECEXTEND_MODEL_NAME", "llama3_1_8b"),
        "specextend_script": values.get("SPECEXTEND_SCRIPT", "run_eagle.py"),
        "sssd_datastore_path": values.get("LONG_BENCH_SSSD_DATASTORE_PATH") or values.get("SSSD_DATASTORE_PATH"),
        "sssd_num_draft_tokens": int(values.get("SSSD_NUM_DRAFT_TOKENS", "8")),
        "sssd_num_steps": int(values.get("SSSD_NUM_STEPS", "5")),
        "sssd_topk": int(values.get("SSSD_TOPK", "5")),
        "sssd_adaptive": values.get("SSSD_ADAPTIVE", "0") == "1",
        "magicdec_model_pth": values.get("LONG_BENCH_MAGICDEC_MODEL_PTH") or values.get("CHECKPOINT_MAGICDEC"),
        "magicdec_model_name": values.get("LONG_BENCH_MAGICDEC_MODEL_NAME") or values.get("MODEL_MAGICDEC_NAME") or values.get("MODEL_TARGET"),
        "fafo_kv_method": values.get("LONG_BENCH_FAFO_KV_METHOD") or values.get("FAFO_KV_METHOD", "stream-llm"),
        "fafo_use_flash": values.get("FAFO_USE_FLASH", "0") == "1",
        "eagle_total_token": int(values.get("LONG_BENCH_EAGLE_TOTAL_TOKEN", "32")),
        "eagle_depth": int(values.get("LONG_BENCH_EAGLE_DEPTH", "8")),
        "eagle_top_k": int(values.get("LONG_BENCH_EAGLE_TOP_K", "4")),
    }


__all__ = [
    "BASELINES",
    "baseline_config_from_env",
    "build_adapter_command",
    "convert_records_for_baseline",
    "preflight_baseline",
]
