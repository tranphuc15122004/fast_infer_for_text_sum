from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_syncspec_b200_train_preflight_does_not_require_drafter_checkpoint(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_syncspec_b200 import build_report

    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps({"vocab_size": 32, "hidden_size": 16}), encoding="utf-8"
    )
    (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "data.jsonl"
    data.write_text('{"id":"sample","document":"hello"}\n', encoding="utf-8")
    report = build_report(
        target_model=str(target), data_file=str(data), phase="train",
    )
    assert "drafter_checkpoint_not_set" not in report["errors"]
    assert "target_model_not_set" not in report["errors"]
    assert "data_file_not_set" not in report["errors"]


def test_syncspec_b200_preflight_rejects_tokenizer_config_without_tokenizer_artifact(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_syncspec_b200 import build_report

    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps({"vocab_size": 32, "hidden_size": 16}), encoding="utf-8",
    )
    (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    data = tmp_path / "data.jsonl"
    data.write_text('{"id":"sample","document":"hello"}\n', encoding="utf-8")
    report = build_report(target_model=str(target), data_file=str(data), phase="train")
    assert "target_tokenizer_missing" in report["errors"]


def test_syncspec_b200_preflight_reports_blocked_without_cuda(tmp_path: Path) -> None:
    report = tmp_path / "preflight.json"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "scripts/check_syncspec_b200.py", "--output", str(report), "--strict"],
        cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert proc.returncode == 2
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert payload["cuda"]["available"] is False
    assert payload["cuda"]["reason"] == "hardware_unavailable"


def test_syncspec_b200_infer_requires_trained_selector_and_survival(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_syncspec_b200 import build_report

    report = build_report(phase="infer")
    assert "selector_checkpoint_not_set" in report["errors"]
    assert "survival_checkpoint_not_set" in report["errors"]


def test_syncspec_b200_preflight_checks_component_artifacts(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_syncspec_b200 import build_report

    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps({"vocab_size": 32, "hidden_size": 16}), encoding="utf-8"
    )
    (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"placeholder")
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    (drafter / "config.json").write_text(
        json.dumps({"vocab_size": 32, "hidden_size": 16}), encoding="utf-8"
    )
    (drafter / "pytorch_model.bin").write_bytes(b"placeholder")
    selector = tmp_path / "selector"
    selector.mkdir()
    survival = tmp_path / "survival"
    survival.mkdir()
    data = tmp_path / "data.jsonl"
    data.write_text('{"id":"x","source_ids":[1,2]}\n', encoding="utf-8")
    profile = tmp_path / "profile.json"
    profile.write_text("{}", encoding="utf-8")
    report = build_report(
        target_model=str(target), drafter_checkpoint=str(drafter),
        selector_checkpoint=str(selector), survival_checkpoint=str(survival),
        data_file=str(data), runtime_profile=str(profile), phase="infer",
    )
    assert "selector_checkpoint_artifacts_missing" in report["errors"]
    assert "survival_checkpoint_artifacts_missing" in report["errors"]
    assert "runtime_profile_invalid" in report["errors"]


def test_syncspec_b200_preflight_rejects_unmeasured_runtime_profile(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_syncspec_b200 import _profile_is_valid

    profile = tmp_path / "synthetic-profile.json"
    profile.write_text(json.dumps({
        "source": "synthetic",
        "key": {
            "model": "target", "checkpoint": "drafter", "gpu": "B200",
            "precision": "bfloat16", "context_bin": "short",
            "batch_bin": "batch1", "kd": 16, "kv": 8,
        },
        "measurements_ms": {
            "target_ar": {"mean": 1.0}, "verify": {"mean": 1.0},
        },
    }), encoding="utf-8")
    assert not _profile_is_valid({"exists": True, "path": str(profile)})


def test_syncspec_b200_preflight_rejects_unknown_profile_schema(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_syncspec_b200 import _profile_is_valid

    profile = tmp_path / "bad-profile-schema.json"
    profile.write_text(json.dumps({
        "schema_version": 2,
        "source": "measured",
        "key": {
            "model": "target", "checkpoint": "drafter", "gpu": "B200",
            "precision": "bfloat16", "context_bin": "short",
            "batch_bin": "batch1", "kd": 16, "kv": 8,
        },
        "measurements_ms": {
            "target_ar": {"mean": 1.0}, "verify": {"mean": 1.0},
        },
    }), encoding="utf-8")
    assert not _profile_is_valid({"exists": True, "path": str(profile)})


def test_syncspec_b200_preflight_rejects_profile_from_wrong_runtime_regime(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_syncspec_b200 import _profile_is_valid

    profile = tmp_path / "cpu-profile.json"
    profile.write_text(json.dumps({
        "schema_version": 1,
        "source": "measured",
        "key": {
            "model": "target", "checkpoint": "drafter", "gpu": "cpu",
            "precision": "float32", "context_bin": "short",
            "batch_bin": "batch1", "kd": 16, "kv": 8,
        },
        "measurements_ms": {
            "target_ar": {"mean": 1.0}, "verify": {"mean": 1.0},
        },
    }), encoding="utf-8")
    assert not _profile_is_valid(
        {"exists": True, "path": str(profile)},
        expected_model="target", expected_checkpoint="drafter",
        expected_gpu="B200", expected_precision="bfloat16", expected_batch_size=2,
    )
    matching = json.loads(profile.read_text(encoding="utf-8"))
    matching["key"].update({
        "gpu": "NVIDIA B200", "precision": "bfloat16", "batch_bin": "batch2",
    })
    profile.write_text(json.dumps(matching), encoding="utf-8")
    assert _profile_is_valid(
        {"exists": True, "path": str(profile)},
        expected_model="target", expected_checkpoint="drafter",
        expected_gpu="B200", expected_precision="bfloat16", expected_batch_size=2,
    )


def test_syncspec_b200_preflight_binds_profile_to_trained_components(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_syncspec_b200 import _profile_is_valid

    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "schema_version": 1,
        "source": "measured",
        "key": {
            "model": "target", "checkpoint": "drafter", "gpu": "NVIDIA B200",
            "precision": "bfloat16", "context_bin": "short", "batch_bin": "batch2",
            "kd": 16, "kv": 8, "selector_checkpoint": "selector-a",
            "survival_checkpoint": "survival-a",
        },
        "measurements_ms": {"target_ar": {"mean": 1.0}, "verify": {"mean": 1.0}},
    }), encoding="utf-8")
    asset = {"exists": True, "path": str(profile)}
    assert not _profile_is_valid(
        asset, expected_model="target", expected_checkpoint="drafter",
        expected_gpu="B200", expected_precision="bfloat16", expected_batch_size=2,
        expected_selector_checkpoint="selector-b",
        expected_survival_checkpoint="survival-a",
    )
    assert _profile_is_valid(
        asset, expected_model="target", expected_checkpoint="drafter",
        expected_gpu="B200", expected_precision="bfloat16", expected_batch_size=2,
        expected_selector_checkpoint="selector-a",
        expected_survival_checkpoint="survival-a",
    )


def test_syncspec_b200_preflight_rejects_drafter_shorter_position_capacity(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_syncspec_b200 import build_report

    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps({
            "vocab_size": 32, "hidden_size": 16,
            "max_position_embeddings": 8192,
        }), encoding="utf-8",
    )
    (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"placeholder")
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    (drafter / "config.json").write_text(
        json.dumps({
            "vocab_size": 32, "hidden_size": 16,
            "max_positions": 4096,
        }), encoding="utf-8",
    )
    (drafter / "pytorch_model.bin").write_bytes(b"placeholder")
    data = tmp_path / "data.jsonl"
    data.write_text('{"id":"x","source_ids":[1,2]}\n', encoding="utf-8")
    report = build_report(
        target_model=str(target), drafter_checkpoint=str(drafter),
        data_file=str(data), phase="train",
    )
    assert any(
        item.startswith("max_positions_insufficient_drafter_4096_target_8192")
        for item in report["compatibility"]["mismatches"]
    )


def test_syncspec_b200_preflight_rejects_selector_width_or_vocab_mismatch(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from check_syncspec_b200 import build_report

    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps({"vocab_size": 32, "hidden_size": 16}), encoding="utf-8",
    )
    (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"placeholder")
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    (drafter / "config.json").write_text(
        json.dumps({"vocab_size": 32, "hidden_size": 16}), encoding="utf-8",
    )
    (drafter / "pytorch_model.bin").write_bytes(b"placeholder")
    selector = tmp_path / "selector"
    selector.mkdir()
    (selector / "selector.pt").write_bytes(b"placeholder")
    (selector / "selector_config.json").write_text(
        json.dumps({"vocab_size": 64, "hidden_size": 8}), encoding="utf-8",
    )
    data = tmp_path / "data.jsonl"
    data.write_text('{"id":"x","source_ids":[1,2]}\n', encoding="utf-8")
    report = build_report(
        target_model=str(target), drafter_checkpoint=str(drafter),
        selector_checkpoint=str(selector), data_file=str(data), phase="infer",
    )
    assert "selector_vocab_size_mismatch_target_32_selector_64" in report["compatibility"]["mismatches"]
    assert "selector_hidden_size_mismatch_drafter_16_selector_8" in report["compatibility"]["mismatches"]


def test_syncspec_b200_preflight_resolves_hf_snapshot_ids_offline(tmp_path: Path) -> None:
    snapshot = tmp_path / "hub" / "models--owner--model" / "snapshots" / "rev1"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"vocab_size": 32, "hidden_size": 16}), encoding="utf-8"
    )
    (snapshot / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    report = tmp_path / "preflight-hf.json"
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "HF_HOME": str(tmp_path)})
    proc = subprocess.run(
        [sys.executable, "scripts/check_syncspec_b200.py", "--target-model", "owner/model",
         "--output", str(report)], cwd=ROOT, env=env, text=True, capture_output=True,
    )
    assert proc.returncode == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["assets"]["target_model"]["exists"] is True
    assert payload["assets"]["target_model"]["path"] == str(snapshot)
    assert payload["compatibility"]["status"] == "PASS"


def test_syncspec_b200_preflight_resolves_repo_relative_checkpoint(tmp_path: Path, monkeypatch) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_syncspec_b200

    monkeypatch.setattr(check_syncspec_b200, "ROOT", tmp_path)
    checkpoint = tmp_path / "checkpoints" / "syncspec-preflight-test"
    checkpoint.mkdir(parents=True, exist_ok=True)
    report = check_syncspec_b200._resolve_asset("checkpoints/syncspec-preflight-test")
    assert report["exists"] is True
    assert report["path"] == str(checkpoint)


def test_syncspec_b200_runner_is_registered_and_strict() -> None:
    runner = (ROOT / "scripts/run_syncspec_b200_smoke.sh").read_text(encoding="utf-8")
    assert "check_syncspec_b200.py" in runner
    assert "--strict" in runner
    assert "infer_syncspec.py" in runner
    assert '--batch-size "${BATCH_SIZE:-1}"' in runner
    assert '--precision "${DTYPE:-bfloat16}"' in runner
    assert "--check-exactness" in runner


def test_syncspec_b200_runner_uses_normalized_budget_overrides() -> None:
    runner = (ROOT / "scripts/run_syncspec_b200_smoke.sh").read_text(encoding="utf-8")
    # config.sh resolves both SYNCSPEC_KD/KV and generic KD/KV into the latter;
    # the wrapper must branch on the normalized names or it can pass mutually
    # exclusive --budget-profiles and --kd/--kv arguments.
    assert '&& -z "${KD:-}" && -z "${KV:-}"' in runner
    assert 'ARGS+=(--kd "${KD:-16}" --kv "${KV:-8}")' in runner


def test_syncspec_b200_train_runner_is_registered_and_strict() -> None:
    runner = (ROOT / "scripts/run_syncspec_b200_train_smoke.sh").read_text(encoding="utf-8")
    assert "check_syncspec_b200.py" in runner
    assert "--phase train" in runner
    assert "--phase infer" in runner
    assert "train_syncspec.py" in runner
    assert "infer_syncspec.py" in runner
    assert "profile_syncspec.py" in runner
    assert "--profile \"$PROFILE\"" in runner
    infer = runner[runner.index("scripts/infer_syncspec.py"):]
    assert ('"${PROFILE_ARGS[@]}"' in infer or
            ('--kd "$PROFILE_KD"' in infer and '--kv "$PROFILE_KV"' in infer))
    assert 'trajectories.pt' in runner
    assert "--include-source-memory" in runner
    assert "--check-exactness" in runner
    assert '--precision "${DTYPE:-bfloat16}" --batch-size "$BATCH_SIZE"' in runner


def test_syncspec_b200_runners_guard_batch_sample_mismatch() -> None:
    infer_runner = (ROOT / "scripts/run_syncspec_b200_smoke.sh").read_text(encoding="utf-8")
    train_runner = (ROOT / "scripts/run_syncspec_b200_train_smoke.sh").read_text(encoding="utf-8")
    for runner in (infer_runner, train_runner):
        assert "MAX_SAMPLES" in runner
        assert "BATCH_SIZE" in runner
        assert "max_samples" in runner
        assert "batch size" in runner
        assert "must be >= batch size" in runner
