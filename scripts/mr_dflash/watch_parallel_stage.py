"""Theo dõi tiến trình parallel regenerate/cache theo từng GPU.

Ví dụ:
``python3 scripts/mr_dflash/watch_parallel_stage.py \
  --status data/mr_dflash_pilot_full/regenerated_full/parallel_regenerate_train/status.json``

Script chỉ đọc status/heartbeat, không attach vào CUDA và không thay đổi job.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"status": "waiting", "workers": []}
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "unreadable", "error": repr(exc), "workers": []}
    return value if isinstance(value, dict) else {"status": "invalid", "workers": []}


def format_status(payload: dict[str, Any]) -> str:
    """Format một snapshot ngắn, luôn hiển thị đủ từng worker/GPU."""
    lines = [f"status={payload.get('status', 'unknown')}"]
    aggregate = payload.get("aggregate") or {}
    if aggregate:
        rate = aggregate.get("throughput_tokens_per_second")
        rate_text = "unknown" if rate is None else f"{float(rate):.1f}tok/s"
        total_tokens = aggregate.get("total_tokens")
        token_text = (
            f"{aggregate.get('completed_tokens', 0)}/{total_tokens}"
            if isinstance(total_tokens, (int, float)) and int(total_tokens) > 0
            else "unknown"
        )
        lines.append(
            "aggregate "
            f"samples={aggregate.get('completed_samples', 0)}/{aggregate.get('total_samples', 0)} "
            f"tokens={token_text} "
            f"rate={rate_text} eta={aggregate.get('eta_human', 'unknown')}"
        )
    workers = payload.get("workers") or []
    if not workers:
        if payload.get("error"):
            lines.append(f"error={payload['error']}")
        return "\n".join(lines)
    for worker in workers:
        progress = worker.get("progress") or {}
        alive = "alive" if worker.get("alive", worker.get("return_code") is None) else "exited"
        phase = progress.get("phase", "unknown")
        sample_id = progress.get("sample_id", "-")
        tokens = progress.get("generated_tokens")
        budget = progress.get("generation_budget")
        token_text = "-"
        if tokens is not None and budget is not None:
            token_text = f"{tokens}/{budget}"
        elif tokens is not None:
            token_text = str(tokens)
        completed = progress.get("completed_samples", "-")
        total_samples = progress.get("total_samples")
        sample_progress = f"{completed}/{total_samples}" if total_samples is not None else str(completed)
        eta = progress.get("eta_human", "unknown")
        rate = progress.get("throughput_tokens_per_second")
        rate_text = "unknown" if rate is None else f"{float(rate):.1f}tok/s"
        age = worker.get("progress_age_seconds")
        age_text = "unknown" if age is None else f"{float(age):.1f}s"
        memory = progress.get("cuda_memory_allocated_gb")
        memory_text = "-" if memory is None else f"{float(memory):.2f}GB"
        code = worker.get("return_code")
        lines.append(
            " ".join(
                [
                    f"rank={worker.get('rank', '?')}",
                    f"gpu={worker.get('gpu_id', '?')}",
                    f"pid={worker.get('pid', '?')}",
                    f"state={alive}",
                    f"code={code}",
                    f"phase={phase}",
                    f"sample={sample_id}",
                    f"tokens={token_text}",
                    f"completed={sample_progress}",
                    f"rate={rate_text}",
                    f"eta={eta}",
                    f"age={age_text}",
                    f"vram={memory_text}",
                ]
            )
        )
    return "\n".join(lines)


def _progress_bar_state(progress: dict[str, Any]) -> dict[str, Any]:
    """Chọn counter cho cache theo sample hoặc regenerate theo token."""
    total_samples = progress.get("total_samples")
    if isinstance(total_samples, (int, float)) and not isinstance(total_samples, bool):
        total = max(0, int(total_samples))
        completed = max(0, int(progress.get("completed_samples", 0) or 0))
        return {
            "mode": "sample",
            "total": total,
            "n": min(total, completed),
            "unit": "sample",
        }
    budget = progress.get("generation_budget")
    if isinstance(budget, (int, float)) and not isinstance(budget, bool) and int(budget) > 0:
        generated = max(0, int(progress.get("generated_tokens", 0) or 0))
        return {
            "mode": "token",
            "total": int(budget),
            "n": min(int(budget), generated),
            "unit": "token",
        }
    return {"mode": "unknown", "total": None, "n": 0, "unit": "item"}


def _stage_label(status_path: Path) -> str:
    """Lấy label ngắn từ parallel plan để watcher giống pipeline chính."""
    plan_path = status_path.parent / "parallel_plan.json"
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "parallel"
    if not isinstance(plan, dict):
        return "parallel"
    mode = str(plan.get("mode", "parallel"))
    input_name = Path(str(plan.get("input", ""))).stem
    split = input_name.rsplit("_", 1)[-1] if input_name else "stage"
    return f"{mode} {split}"


def _input_sample_count(status_path: Path) -> int:
    """Đọc total cố định từ nguồn mà parallel worker thực sự phân shard."""
    plan_path = status_path.parent / "parallel_plan.json"
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        source_value = plan.get("input")
        if plan.get("mode") == "cache" and plan.get("tokenized_path"):
            source_value = plan["tokenized_path"]
        input_path = Path(str(source_value))
        if input_path.is_dir():
            manifest = json.loads(
                (input_path / "manifest.json").read_text(encoding="utf-8")
            )
            if not isinstance(manifest, dict):
                return 0
            for key in ("num_samples", "total_samples", "valid_samples"):
                value = int(manifest.get(key, 0) or 0)
                if value >= 0:
                    return value
            return 0
        with input_path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return 0


def _fixed_total(*, aggregate_total: int, input_total: int) -> int:
    """Ưu tiên total toàn input, không dùng aggregate còn thiếu worker."""
    if input_total > 0:
        return input_total
    return max(0, aggregate_total)


def _watch_tqdm(status_path: Path, *, interval: float, once: bool) -> int:
    """Hiển thị một thanh tổng hợp, dữ liệu lấy từ heartbeat atomic."""
    try:
        from tqdm import tqdm
    except ImportError:
        print("[watch] cần cài tqdm để theo dõi progress", file=sys.stderr, flush=True)
        return 2

    total_from_input = _input_sample_count(status_path)
    bar = None
    try:
        while True:
            payload = _read_status(status_path)
            aggregate = payload.get("aggregate") or {}
            try:
                total = int(aggregate.get("total_samples", 0) or 0)
            except (TypeError, ValueError):
                total = 0
            total = _fixed_total(
                aggregate_total=total,
                input_total=total_from_input,
            )
            try:
                completed = int(aggregate.get("completed_samples", 0) or 0)
            except (TypeError, ValueError):
                completed = 0
            completed = min(max(0, completed), max(0, total))
            if bar is None:
                bar = tqdm(
                    total=total,
                    initial=completed,
                    leave=True,
                    dynamic_ncols=True,
                    unit="sample",
                    desc=_stage_label(status_path),
                )
            elif bar.total != total:
                bar.total = total
            if completed < bar.n:
                bar.n = completed
                bar.refresh()
            elif completed > bar.n:
                bar.update(completed - bar.n)
            rate = aggregate.get("throughput_tokens_per_second")
            eta = aggregate.get("eta_human", "unknown")
            bar.set_postfix_str(f"tok/s={rate if rate is not None else 'unknown'} eta={eta}")
            bar.refresh()
            if once or payload.get("status") in {"success", "failed"}:
                return 0 if payload.get("status") == "success" else (0 if once else 1)
            time.sleep(interval)
    finally:
        if bar is not None:
            bar.close()


def _watch_text(status_path: Path, *, interval: float, once: bool) -> int:
    while True:
        payload = _read_status(status_path)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"[{now}] {format_status(payload)}", flush=True)
        if once or payload.get("status") in {"success", "failed"}:
            return 0 if payload.get("status") == "success" else (0 if once else 1)
        time.sleep(interval)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Theo dõi parallel MR-DFlash stage")
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--tqdm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hiển thị một thanh tiến trình tổng hợp theo sample (mặc định bật)",
    )
    args = parser.parse_args(argv)
    if args.interval <= 0:
        raise ValueError("interval phải > 0")
    if args.tqdm:
        return _watch_tqdm(args.status, interval=args.interval, once=args.once)
    return _watch_text(args.status, interval=args.interval, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
