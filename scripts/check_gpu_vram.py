#!/usr/bin/env python3
"""Kiểm tra GPU id và VRAM còn trống trước khi chạy job benchmark LongBench.

Script chạy bằng python3 hệ thống (không import torch, không load model, không
cần venv) nên an toàn chạy trên server B200. Dùng nvidia-smi để liệt kê GPU
**vật lý** của host, vì vậy kết quả không bị CUDA_VISIBLE_DEVICES che đi; kèm
theo tiến trình đang chiếm từng GPU.

Lựa chọn GPU mà job sẽ dùng được lấy theo cùng thứ tự ưu tiên của
``scripts/run_longbench_200.py``:
``--gpu-ids`` > ``LONG_BENCH_GPU_IDS`` > ``FI_GPU_IDS`` > ``CUDA_VISIBLE_DEVICES``.

Usage:
  python3 scripts/check_gpu_vram.py
  python3 scripts/check_gpu_vram.py --gpu-ids 2
  python3 scripts/check_gpu_vram.py --min-free-gb 120
  python3 scripts/check_gpu_vram.py --json gpu_report.json

Exit code:
  0  kiểm tra đạt (hoặc không yêu cầu --min-free-gb)
  2  GPU được chọn có VRAM trống < --min-free-gb (không nên chạy job)
  1  lỗi khác (thiếu nvidia-smi, không thấy GPU, tham số sai)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

GPU_QUERY = (
    "index,uuid,name,memory.total,memory.free,memory.used,"
    "utilization.gpu,temperature.gpu,power.draw,power.limit"
)
APP_QUERY = "gpu_uuid,pid,process_name,used_memory"
_SELECTION_ENV = ("LONG_BENCH_GPU_IDS", "FI_GPU_IDS", "CUDA_VISIBLE_DEVICES")


# ---------------------------------------------------------------------------
# nvidia-smi helpers (stdlib only)
# ---------------------------------------------------------------------------

def _nvidia_smi(query: str) -> tuple[bool, list[str], list[str]]:
    """Return (ok, csv_lines, errors) for one nvidia-smi query."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return False, [], ["nvidia-smi not found on PATH"]
    except subprocess.TimeoutExpired:
        return False, [], ["nvidia-smi timed out after 10s"]
    if out.returncode != 0:
        errors = [line.strip() for line in out.stderr.splitlines() if line.strip()]
        return False, [], errors or [f"nvidia-smi exited {out.returncode}"]
    lines = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return True, lines, []


def _csv_rows(lines: list[str]) -> list[list[str]]:
    return [[field.strip() for field in line.split(",")] for line in lines]


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gib(value: str | float | None) -> float | None:
    """Convert an nvidia-smi MiB value (str) into GiB; None on failure."""
    mib = _to_float(value) if isinstance(value, str) else value
    if mib is None:
        return None
    return round(mib / 1024.0, 1)


# ---------------------------------------------------------------------------
# Selection resolution (khớp run_longbench_200.py)
# ---------------------------------------------------------------------------

def _selection() -> tuple[str | None, str | None, bool]:
    """Return (raw_ids, source_var, cpu_mode) from CLI/env.

    ``cpu_mode`` is True when CUDA_VISIBLE_DEVICES is explicitly empty, which
    is how CPU smoke runs hide every GPU.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        return None, "CUDA_VISIBLE_DEVICES", True
    for name in _SELECTION_ENV:
        value = os.environ.get(name)
        if value:
            return value, name, False
    return None, None, False


def _parse_ids(raw: str | None) -> list[int] | None:
    if not raw or not raw.strip():
        return None
    parts = [part for part in raw.replace(",", " ").split() if part]
    try:
        ids = [int(part) for part in parts]
    except ValueError as exc:
        raise SystemExit(f"invalid GPU ids {raw!r}: expected indices such as '0', '2' or '0,1'") from exc
    if any(index < 0 for index in ids):
        raise SystemExit(f"invalid GPU ids {raw!r}: indices must be >= 0")
    return ids


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _collect_gpus() -> list[dict[str, Any]]:
    ok, lines, errors = _nvidia_smi(f"gpu={GPU_QUERY}")
    if not ok:
        raise SystemExit("nvidia-smi GPU query failed:\n  " + "\n  ".join(errors))
    gpus: list[dict[str, Any]] = []
    for fields in _csv_rows(lines):
        if len(fields) < 10:
            continue
        try:
            index = int(fields[0])
        except ValueError:
            continue
        total_mib = _to_float(fields[3])
        free_mib = _to_float(fields[4])
        used_mib = _to_float(fields[5])
        util = _to_float(fields[6])
        gpus.append(
            {
                "index": index,
                "uuid": fields[1] or None,
                "name": fields[2],
                "total_gb": _gib(total_mib),
                "used_gb": _gib(used_mib),
                "free_gb": _gib(free_mib),
                "free_percent": round(free_mib / total_mib * 100.0, 1)
                if free_mib is not None and total_mib
                else None,
                "utilization_percent": int(round(util)) if util is not None else None,
                "temperature_c": int(round(_to_float(fields[7])))
                if fields[7] not in ("", "[N/A]") and _to_float(fields[7]) is not None
                else None,
                "power_w": _to_float(fields[8]),
                "power_limit_w": _to_float(fields[9]),
                "selected": False,
                "processes": [],
            }
        )
    return gpus


def _collect_processes(gpus: list[dict[str, Any]]) -> None:
    ok, lines, errors = _nvidia_smi(f"compute-apps={APP_QUERY}")
    if not ok:
        return  # processes are informational; never fail on them
    by_uuid = {gpu["uuid"]: gpu for gpu in gpus if gpu.get("uuid")}
    for fields in _csv_rows(lines):
        if len(fields) < 4:
            continue
        uuid, pid, name, used = fields[0], fields[1], fields[2], fields[3]
        try:
            pid_int = int(pid)
        except ValueError:
            continue
        app = {
            "pid": pid_int,
            "name": name,
            "used_gb": _gib(used),
        }
        target = by_uuid.get(uuid)
        if target is None:
            continue
        target["processes"].append(app)


def _recommend(gpus: list[dict[str, Any]], min_free_gb: float | None) -> dict[str, Any] | None:
    candidates = [g for g in gpus if g["free_gb"] is not None]
    if not candidates:
        return None
    if min_free_gb is not None:
        candidates = [g for g in candidates if g["free_gb"] >= min_free_gb]
    if not candidates:
        return None
    best = max(candidates, key=lambda g: g["free_gb"])
    return {
        "index": best["index"],
        "name": best["name"],
        "free_gb": best["free_gb"],
        "total_gb": best["total_gb"],
    }


def build_report(
    *,
    cli_gpu_ids: str | None,
    min_free_gb: float | None,
) -> dict[str, Any]:
    env_raw, source, cpu_mode = _selection()
    raw = cli_gpu_ids if cli_gpu_ids is not None else env_raw
    if cli_gpu_ids is not None:
        source = "--gpu-ids"
    ids = _parse_ids(raw)

    gpus = _collect_gpus()
    _collect_processes(gpus)

    host_indexes = {gpu["index"] for gpu in gpus}
    selected_ids = ids if ids is not None else sorted(host_indexes)
    missing = [index for index in (ids or []) if index not in host_indexes]
    for gpu in gpus:
        gpu["selected"] = gpu["index"] in (selected_ids or [])

    # Gate: GPU(s) job sẽ dùng phải có đủ VRAM trống.
    verdict = "skip_cpu"
    failing: list[int] = []
    if not cpu_mode:
        if min_free_gb is None:
            verdict = "ok"
        else:
            failing = [
                index
                for index in selected_ids
                if index not in host_indexes
                or not (g := next((g for g in gpus if g["index"] == index), None))
                or g["free_gb"] is None
                or g["free_gb"] < min_free_gb
            ]
            if missing:
                failing = sorted(set(failing) | set(missing))
            verdict = "pass" if not failing else "fail"

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_count": len(gpus),
        "selection": {
            "raw": raw,
            "ids": ids,
            "source": source,
            "missing": missing,
            "cpu_mode": cpu_mode,
        },
        "min_free_gb": min_free_gb,
        "gpus": gpus,
        "gate": {
            "enabled": min_free_gb is not None and not cpu_mode,
            "verdict": verdict,
            "failing_gpu_ids": sorted(failing),
        },
        "recommended_gpu": _recommend(gpus, min_free_gb),
        "result": verdict,
    }


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------

def _fmt_gb(value: float | None) -> str:
    return "N/A" if value is None else f"{value:>7.1f}"


def _fmt_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value}%"


def _print_report(report: dict[str, Any]) -> None:
    gpus = report["gpus"]
    sel = report["selection"]
    print("GPU check trước khi chạy LongBench (nvidia-smi, GPU vật lý)")
    print(f"Thời điểm : {report['timestamp_utc']}")
    print(f"Số GPU    : {report['gpu_count']} trên host")
    if sel["cpu_mode"]:
        print("Chọn GPU  : (CUDA_VISIBLE_DEVICES rỗng -> job sẽ chạy CPU)")
    elif sel["raw"]:
        src = sel["source"] or "auto"
        print(f"Chọn GPU  : GPU {', '.join(map(str, sel['ids'] or []))}  (nguồn: {src})")
        if sel["missing"]:
            print(
                f"  WARNING: GPU id {sel['missing']} không tồn tại trên host -> "
                "CUDA sẽ không thấy device nào (job fail/OOM ngay khi load model)."
            )
    else:
        print("Chọn GPU  : (chưa chỉ định -> job mặc định thấy toàn bộ GPU host)")

    print("\nDanh sách GPU:")
    if not gpus:
        print("  (không có GPU nào - nvidia-smi trả về rỗng)")
    for gpu in gpus:
        marker = "*" if gpu["selected"] else " "
        util = gpu["utilization_percent"]
        temp = gpu["temperature_c"]
        pwr = gpu["power_w"]
        pwr_lim = gpu["power_limit_w"]
        extra = []
        if util is not None:
            extra.append(f"util {util}%")
        if temp is not None:
            extra.append(f"{temp}C")
        if pwr is not None:
            extra.append(
                f"pwr {pwr:.0f}W" + (f"/{pwr_lim:.0f}W" if pwr_lim is not None else "")
            )
        suffix = " | " + " | ".join(extra) if extra else ""
        print(
            f"{marker} GPU {gpu['index']}: {gpu['name']}"
            f"  total {_fmt_gb(gpu['total_gb'])} GB"
            f"  | used {_fmt_gb(gpu['used_gb'])} GB"
            f"  | free {_fmt_gb(gpu['free_gb'])} GB"
            f" ({_fmt_percent(gpu['free_percent'])})"
            f"{suffix}"
        )
        for app in gpu["processes"]:
            print(
                f"      - pid {app['pid']} {app['name']}"
                f"  dùng {_fmt_gb(app['used_gb'])} GB"
            )
    print("  (*) GPU mà job benchmark sẽ dùng")

    gate = report["gate"]
    if report["min_free_gb"] is not None:
        print(f"\nNgưỡng VRAM trống tối thiểu: {report['min_free_gb']:.1f} GB")
        if gate["verdict"] == "pass":
            print("Kết quả  : PASS - GPU được chọn đủ VRAM trống, sẵn sàng chạy job.")
        elif gate["verdict"] == "fail":
            print(
                f"Kết quả  : FAIL - GPU {gate['failing_gpu_ids']} không đủ VRAM trống "
                f"(< {report['min_free_gb']:.1f} GB). Dừng lại hoặc chọn GPU khác."
            )
        elif gate["verdict"] == "skip_cpu":
            print("Kết quả  : SKIP - job chạy CPU nên không áp dụng ngưỡng VRAM.")

    rec = report["recommended_gpu"]
    if rec is not None:
        print(
            f"\nGợi ý    : GPU trống nhiều nhất là GPU {rec['index']} "
            f"({rec['name']}, free {rec['free_gb']} GB). "
            f"Chạy job với: LONG_BENCH_GPU_IDS={rec['index']} "
            "bash scripts/run_longbench_200.sh <master> --mode full"
        )
    elif report["min_free_gb"] is not None and gpus:
        print(
            "\nGợi ý    : hiện không có GPU nào đạt ngưỡng "
            f"{report['min_free_gb']:.1f} GB trống."
        )


def _write_json(report: dict[str, Any], destination: str | None) -> None:
    if not destination:
        return
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if destination == "-":
        print(text, end="")
        return
    path = os.path.abspath(destination)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gpu-ids",
        default=None,
        help="GPU id(s) job sẽ dùng (vd '2' hoặc '0,1'); ghi đè LONG_BENCH_GPU_IDS/FI_GPU_IDS/CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="ngưỡng VRAM trống tối thiểu (GB) trên GPU được chọn; exit 2 nếu thiếu",
    )
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="ghi báo cáo JSON (stdout nếu dùng '-')",
    )
    args = parser.parse_args(argv)

    try:
        report = build_report(cli_gpu_ids=args.gpu_ids, min_free_gb=args.min_free_gb)
    except SystemExit as exc:
        _write_json({"error": str(exc)}, args.json)
        raise

    _write_json(report, args.json)
    _print_report(report)

    if report["gate"]["verdict"] == "fail":
        return 2
    if report["gpu_count"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
