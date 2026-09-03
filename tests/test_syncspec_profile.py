from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.profile import (  # noqa: E402
    ProfileKey, RuntimeProfiler, ar_cost_from_profile, costs_from_profile,
)


def test_ar_cost_from_profile_normalizes_measured_token_count() -> None:
    payload = {
        "key": {"kv": 4},
        "measurements_ms": {"target_ar": {"mean": 12.0}},
    }
    assert ar_cost_from_profile(payload) == 3.0

    payload["target_ar_tokens"] = 1
    assert ar_cost_from_profile(payload) == 12.0


def test_profile_key_contains_hardware_and_budget_axes() -> None:
    key = ProfileKey("target", "draft", "NVIDIA B200", "bfloat16", "long", "batch1", 16, 8)
    raw = key.to_dict()
    assert raw["gpu"] == "NVIDIA B200"
    assert raw["kd"] == 16 and raw["kv"] == 8


def test_runtime_profiler_emits_measured_costs(tmp_path: Path) -> None:
    profiler = RuntimeProfiler(ProfileKey("toy", "toy", "cpu", "float32", "short", "batch1", 4, 2))
    profiler.measure("draft", lambda: sum(range(100)))
    profiler.measure("verify", lambda: sum(range(50)))
    payload = profiler.to_dict()
    assert payload["source"] == "measured"
    assert payload["measurements_ms"]["draft"]["count"] == 1
    path = tmp_path / "profile.json"
    profiler.save(path)
    assert json.loads(path.read_text())["key"]["kd"] == 4
    assert costs_from_profile(profiler.to_dict(), component="draft")[2] > 0.0


def test_runtime_profiler_preserves_diagnostic_source() -> None:
    profiler = RuntimeProfiler(
        ProfileKey("toy", "toy", "cpu", "float32", "short", "batch1", 4, 2),
        source="diagnostic",
    )
    assert profiler.to_dict()["source"] == "diagnostic"


def test_profile_cli_measures_multi_request_batch(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/profile_syncspec.py", "--backend", "synthetic",
         "--device", "cpu", "--batch-size", "2", "--repeats", "1",
         "--warmup-runs", "0", "--kd", "4", "--kv", "2",
         "--output", str(tmp_path / "profile.json")],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads((tmp_path / "profile.json").read_text())
    assert payload["key"]["batch_bin"] == "batch2"
    assert payload["key"]["context_bin"] == "short"
    assert payload["measurements_ms"]["verify"]["count"] == 1
    assert payload["measurements_ms"]["verify"]["p95"] >= payload["measurements_ms"]["verify"]["mean"]
    assert "target_ar" in payload["measurements_ms"]
    assert payload["target_ar_tokens"] == 2


def test_profile_cli_can_emit_multiple_finite_budget_records(tmp_path: Path) -> None:
    output = tmp_path / "adaptive-profile.json"
    proc = subprocess.run(
        [sys.executable, "scripts/profile_syncspec.py", "--backend", "synthetic",
         "--device", "cpu", "--batch-size", "1", "--repeats", "1",
         "--warmup-runs", "0", "--budget-profiles", "4:2,4:4",
         "--output", str(output)],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(output.read_text())
    assert isinstance(payload, list)
    assert [(item["key"]["kd"], item["key"]["kv"]) for item in payload] == [
        (4, 2), (4, 4),
    ]
    assert all(item["source"] == "measured" for item in payload)


def test_transformers_profile_requires_trained_components(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/profile_syncspec.py", "--backend", "transformers",
         "--target-model", str(tmp_path / "target"),
         "--drafter-checkpoint", str(tmp_path / "drafter"),
         "--device", "cpu", "--output", str(tmp_path / "profile.json")],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert proc.returncode != 0
    message = (proc.stdout + proc.stderr).lower()
    assert "selector-checkpoint" in message
    assert "survival-checkpoint" in message


def test_profile_cli_measures_ar_baseline_as_one_token(monkeypatch, tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import profile_syncspec
    from SyncSpec.synthetic import SyntheticTarget

    calls: list[int] = []
    original = SyntheticTarget.generate_greedy

    def recording_generate(self, source_ids, max_new_tokens):
        calls.append(int(max_new_tokens))
        return original(self, source_ids, max_new_tokens)

    monkeypatch.setattr(SyntheticTarget, "generate_greedy", recording_generate)
    monkeypatch.setattr(sys, "argv", [
        "profile_syncspec.py", "--backend", "synthetic", "--device", "cpu",
        "--repeats", "1", "--warmup-runs", "0", "--kd", "4", "--kv", "2",
        "--output", str(tmp_path / "profile.json"),
    ])
    assert profile_syncspec.main() == 0
    # Profiling should measure one decode step after prefill, not invoke the
    # full vanilla generation helper (which would include prefill in the
    # opportunity cost used by the per-round controller).
    assert calls == []


def test_target_ar_profile_closes_cuda_prefill_boundary_before_timer(monkeypatch) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import profile_syncspec

    events: list[str] = []
    monkeypatch.setattr(
        profile_syncspec, "_sync_if_cuda",
        lambda device: events.append(f"sync:{device}"),
    )

    class Target:
        def next_logits(self, state):
            events.append("next_logits")
            return state

        def commit(self, state, result):
            del state, result
            events.append("commit")

    class Profiler:
        def measure(self, name, function):
            events.append(f"measure:{name}")
            return function()

    target = Target()
    profile_syncspec._measure_target_ar(
        Profiler(), target, [__import__("torch").tensor([1.0, 0.0])], "cuda:0",
    )
    assert events[:2] == ["sync:cuda:0", "measure:target_ar"]
    assert events[-1] == "sync:cuda:0"
