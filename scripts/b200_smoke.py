#!/usr/bin/env python3
"""Coordinator for the B200 one-sample smoke matrix."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from check_b200_env import _parse_baselines, build_report  # noqa: E402


def _path(value: str, root: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else root / candidate


def _first_record(data_file: str | None, root: Path) -> dict | None:
    if not data_file:
        return None
    path = _path(data_file, root)
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    return None


def _prepare_special_inputs(data_file: str | None, generated: Path, root: Path) -> dict[str, str]:
    row = _first_record(data_file, root)
    if row is None:
        return {}
    document = row.get("document") or row.get("text") or row.get("prompt") or ""
    reference = row.get("reference") or row.get("summary") or row.get("answer")
    sample_id = row.get("id", 0)
    generated.mkdir(parents=True, exist_ok=True)
    eagle_path = generated / "eagle3.jsonl"
    eagle_path.write_text(
        json.dumps(
            {
                "question_id": sample_id,
                "turns": ["Summarize the following document.\n\n" + document],
                "reference": reference,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    spec_path = generated / "specextend.jsonl"
    spec_path.write_text(
        json.dumps(
            {
                "id": sample_id,
                "text": "Summarize the following document.\n\n" + document,
                "reference": reference,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "EAGLE_DATA_FILE": str(eagle_path),
        "SPECEXTEND_INPUT_FILE": str(spec_path),
    }


def _overlay_values(
    baseline: str, output: Path, generated_inputs: dict[str, str]
) -> dict[str, str]:
    env = os.environ
    target = env.get("B200_TARGET_MODEL", "")
    data_file = env.get("B200_DATA_FILE", "")
    device = env.get("B200_DEVICE", "cuda")
    common = {
        "SMOKE": "1",
        "OUTPUT_FILE": str(output),
        "MAX_SAMPLES": env.get("B200_SMOKE_MAX_SAMPLES", "1"),
        "MAX_NEW_TOKENS": env.get("B200_SMOKE_MAX_NEW_TOKENS", "8"),
        "MAX_GEN_LEN": env.get("B200_SMOKE_MAX_NEW_TOKENS", "8"),
        "MAX_TOKENS": env.get("B200_SMOKE_MAX_NEW_TOKENS", "8"),
        "MAX_INPUT_TOKENS": "512",
        "WARMUP_RUNS": "0",
        "NUM_RUNS": "1",
    }
    values = dict(common)
    if baseline == "eagle3":
        values.update(
            {
                "BASE_MODEL": target,
                "EAGLE_MODEL": env.get("B200_EAGLE_MODEL", ""),
                "DATA_FILE": generated_inputs.get("EAGLE_DATA_FILE", data_file),
                "QUESTION_BEGIN": "0",
                "QUESTION_END": "1",
            }
        )
    elif baseline == "dflash":
        values.update(
            {
                "TARGET_MODEL": target,
                "DRAFT_MODEL": env.get("B200_DFLASH_MODEL", ""),
                "DATA_FILE": data_file,
                "ATTN_IMPLEMENTATION": "flash_attention_2",
            }
        )
    elif baseline == "llmlingua":
        values.update(
            {
                "COMPRESSOR_MODEL": env.get("B200_COMPRESSOR_MODEL", ""),
                "TARGET_MODEL": target,
                "DOC_FILE": data_file,
                "DEVICE": device,
            }
        )
    elif baseline == "fastkv":
        values.update(
            {
                "MODEL": target,
                "METHOD": "fastkv",
                "ATTN_IMPL": "flash_attention_2",
                "DATA_FILE": data_file,
            }
        )
    elif baseline == "rocketkv":
        values["NUM_RUNS"] = "2"
    elif baseline == "gemfilter":
        values.update({"MODEL": target, "DATA_FILE": data_file})
    elif baseline == "specprefill":
        values.update(
            {
                "TARGET_MODEL": target,
                "SPEC_MODEL": env.get("B200_SPEC_MODEL", ""),
                "DATA_FILE": data_file,
            }
        )
    elif baseline == "minference":
        values.update(
            {
                "MODEL": target,
                "DATA_FILE": data_file,
                "DEVICE": device,
                "ATTN_IMPLEMENTATION": "auto",
            }
        )
    elif baseline == "magicdec":
        values.update(
            {
                "MODEL_PTH": env.get("B200_MAGICDEC_MODEL_PTH", ""),
                "MODEL_NAME": env.get("B200_MAGICDEC_MODEL_NAME", ""),
            }
        )
    elif baseline == "longspec":
        values.update(
            {
                "MODEL_NAME": "vicuna7b",
                "TARGET_MODEL": env.get("B200_VICUNA_MODEL", ""),
                "DRAFT_MODEL": env.get("B200_LONGSPEC_DRAFT_MODEL", ""),
                "DATA_FILE": data_file,
            }
        )
    elif baseline == "specextend":
        values.update(
            {
                "SCRIPT": "run_eagle.py",
                "MODEL_NAME": "llama3_1_8b",
                "BASE_MODEL": target,
                "DRAFT_MODEL": env.get("B200_EAGLE_MODEL", ""),
                "INPUT_FILE": generated_inputs.get("SPECEXTEND_INPUT_FILE", data_file),
                "USE_SPECEXTEND": "1",
            }
        )
    elif baseline == "higoe":
        values["NUM_DOCS"] = "3"
    elif baseline == "semantic_selection":
        values.update(
            {
                "MODEL": target,
                "INPUT_FILE": data_file,
                "EMBEDDING_MODEL": env.get("B200_EMBEDDING_MODEL", ""),
                "DEVICE": device,
                "EMBEDDING_DEVICE": "cpu",
            }
        )
    elif baseline == "flexprefill":
        values.update({"MODEL": target, "DATA_FILE": data_file})
    return values


def _write_overlay(path: Path, values: dict[str, str]) -> None:
    path.write_text(
        "\n".join(
            f"{key}={shlex.quote(value)}" for key, value in sorted(values.items())
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines = _parse_baselines(args.baselines)
    preflight = build_report(baselines, os.environ.get("B200_TARGET_GPU", "B200"))
    preflight_path = output_dir / "preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "type": "b200_smoke_summary",
        "interpreter": {"path": sys.executable},
        "target_gpu": preflight["target_gpu"],
        "preflight": {
            "status": preflight["status"],
            "errors": preflight["errors"],
            "artifact": str(preflight_path),
        },
        "baselines": {
            name: {"status": "PENDING", "reason": "not_started"}
            for name in baselines
        },
    }
    generated_inputs = _prepare_special_inputs(
        os.environ.get("B200_DATA_FILE"), output_dir / "generated", root
    )

    if args.preflight_only:
        for baseline in baselines:
            info = preflight["baselines"][baseline]
            summary["baselines"][baseline] = {
                "status": info["status"],
                "reason": info["reason"],
            }
    else:
        for baseline in baselines:
            info = preflight["baselines"][baseline]
            if info["status"] != "PASS":
                summary["baselines"][baseline] = {
                    "status": info["status"],
                    "reason": info["reason"],
                }
                continue
            output = output_dir / f"{baseline}.jsonl"
            log_path = output_dir / f"{baseline}.log"
            overlay = output_dir / "generated" / f"{baseline}.env"
            overlay.parent.mkdir(parents=True, exist_ok=True)
            _write_overlay(overlay, _overlay_values(baseline, output, generated_inputs))
            child_env = dict(os.environ)
            child_env["FAST_INFER_CONFIG_OVERLAY"] = str(overlay)
            command = ["bash", str(root / "scripts" / "run.sh"), baseline, "--smoke"]
            start = time.perf_counter()
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    proc = subprocess.run(
                        command,
                        cwd=root,
                        env=child_env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        timeout=args.timeout,
                        check=False,
                    )
                reason = "completed" if proc.returncode == 0 else "launcher_failed"
                status = "PASS" if proc.returncode == 0 else "FAIL"
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                status, reason, exit_code = "FAIL", "timeout", 124
            summary["baselines"][baseline] = {
                "status": status,
                "reason": reason,
                "exit_code": exit_code,
                "duration_s": round(time.perf_counter() - start, 3),
                "output": str(output),
                "log": str(log_path),
            }

    statuses = [item["status"] for item in summary["baselines"].values()]
    summary["status"] = (
        "PASS"
        if preflight["status"] == "PASS"
        and statuses
        and all(s == "PASS" for s in statuses)
        else "BLOCKED"
    )
    summary_path = output_dir / "b200_smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
