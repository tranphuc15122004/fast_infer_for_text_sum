"""Theo dõi tiến trình parallel regenerate/cache theo từng GPU.

Ví dụ:
``python3 scripts/mr_dflash/watch_parallel_stage.py \
  --status data/mr_dflash_pilot_full/regenerated_full/parallel_regenerate_train/status.json``

Script chỉ đọc status/heartbeat, không attach vào CUDA và không thay đổi job.
"""

from __future__ import annotations

import argparse
import json
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
        lines.append(
            "aggregate "
            f"samples={aggregate.get('completed_samples', 0)}/{aggregate.get('total_samples', 0)} "
            f"tokens={aggregate.get('completed_tokens', 0)}/{aggregate.get('total_tokens', 0)} "
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


def _watch_tqdm(status_path: Path, *, interval: float, once: bool) -> int:
    """Hiển thị một thanh/worker, dữ liệu vẫn lấy từ heartbeat atomic."""
    try:
        from tqdm import tqdm
    except ImportError:
        print("[watch] thiếu tqdm; chuyển sang chế độ text", flush=True)
        return _watch_text(status_path, interval=interval, once=once)

    bars: dict[int, Any] = {}
    last_phases: dict[int, str] = {}
    try:
        while True:
            payload = _read_status(status_path)
            workers = payload.get("workers") or []
            for position, worker in enumerate(workers):
                rank = int(worker.get("rank", position))
                progress = worker.get("progress") or {}
                phase = str(progress.get("phase", "unknown"))
                budget = progress.get("generation_budget")
                total = int(budget) if isinstance(budget, (int, float)) and int(budget) > 0 else None
                if rank not in bars:
                    bars[rank] = tqdm(
                        total=total,
                        position=position,
                        leave=True,
                        dynamic_ncols=True,
                        desc=f"GPU {worker.get('gpu_id', '?')} {phase}",
                    )
                bar = bars[rank]
                if total is not None and bar.total != total:
                    bar.total = total
                generated = progress.get("generated_tokens")
                completed = progress.get("completed_samples", "-")
                if phase != last_phases.get(rank):
                    bar.set_description(f"GPU {worker.get('gpu_id', '?')} {phase}")
                    last_phases[rank] = phase
                if isinstance(generated, (int, float)):
                    generated = int(generated)
                    if generated < bar.n:
                        bar.n = generated
                    else:
                        bar.update(generated - bar.n)
                memory = progress.get("cuda_memory_allocated_gb")
                memory_text = "-" if memory is None else f"{float(memory):.1f}GB"
                bar.set_postfix(sample=progress.get("sample_id", "-"), done=completed, vram=memory_text)
                bar.refresh()
            if once or payload.get("status") in {"success", "failed"}:
                return 0 if payload.get("status") == "success" else (0 if once else 1)
            time.sleep(interval)
    finally:
        for bar in bars.values():
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
    parser.add_argument("--tqdm", action="store_true", help="hiển thị thanh tiến trình riêng cho mỗi GPU")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        raise ValueError("interval phải > 0")
    if args.tqdm:
        return _watch_tqdm(args.status, interval=args.interval, once=args.once)
    return _watch_text(args.status, interval=args.interval, once=args.once)


if __name__ == "__main__":
    raise SystemExit(main())
