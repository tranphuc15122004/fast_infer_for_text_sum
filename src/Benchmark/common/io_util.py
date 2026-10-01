"""JSONL output + unified result schema."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from Benchmark.common.paths import ROOT

BASE_SCHEMA_KEYS = [
    "method",
    "dataset",
    "model",
    "input_tokens",
    "retained_tokens",
    "output_tokens",
    "batch_size",
    "selector_latency_ms",
    "server_startup_ms",
    "queue_wait_ms",
    "batch_wait_ms",
    "ttft_ms",
    "draft_latency_ms",
    "verification_latency_ms",
    "tpot_ms",
    "e2e_ms",
    "server_reported_e2e_ms",
    "throughput_tok_s",
    "qps",
    "peak_memory_gb",
]

SPEC_SCHEMA_KEYS = [
    "avg_accept_length",
    "acceptance_rate",
    "acceptance_rate_percent",
    "accepted_draft_tokens_per_step",
    "draft_tokens_accepted",
    "draft_tokens_proposed",
    "draft_proposal_unit",
    "draft_latency_ms",
    "verification_latency_ms",
    "rejected_draft_ratio",
    "verification_steps",
]


def _json_safe(obj: Any) -> Any:
    """Recursively convert non-JSON values (tensors, numpy, Path, torch.Size)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def validate_schema(record: dict[str, Any], spec: bool = False) -> list[str]:
    keys = BASE_SCHEMA_KEYS + (SPEC_SCHEMA_KEYS if spec else [])
    missing = [k for k in keys if k not in record]
    return [f"missing key: {k}" for k in missing]


class JsonlWriter:
    """Write run records one per line; write summary record at finalize."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = ROOT / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self.records: list[dict[str, Any]] = []

    def add(self, record: dict[str, Any]) -> None:
        safe = _json_safe(record)
        self.records.append(safe)
        self._file.write(json.dumps(safe, ensure_ascii=False) + "\n")
        self._file.flush()

    def finalize(self, extra_summary: dict[str, Any] | None = None) -> dict[str, Any]:
        """Compute aggregate statistics over non-warmup records and append summary line."""
        measured = [r for r in self.records if not r.get("is_warmup", False)]
        if not measured:
            measured = self.records

        def mean_of(key: str) -> float | None:
            vals = [float(r[key]) for r in measured if r.get(key) is not None]
            return statistics.mean(vals) if vals else None

        summary: dict[str, Any] = {
            "record_type": "summary",
            "count": len(measured),
            "mean_tpot_ms": mean_of("tpot_ms"),
            "mean_e2e_ms": mean_of("e2e_ms"),
            "mean_ttft_ms": mean_of("ttft_ms"),
            "mean_throughput_tok_s": mean_of("throughput_tok_s"),
            "peak_memory_gb": max(
                (float(r["peak_memory_gb"]) for r in measured if r.get("peak_memory_gb") is not None),
                default=None,
            ),
        }
        if extra_summary:
            summary.update(_json_safe(extra_summary))
        self._file.write(json.dumps(_json_safe(summary), ensure_ascii=False) + "\n")
        self._file.flush()
        self._file.close()
        return summary
