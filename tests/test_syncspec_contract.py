from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from SyncSpec.config import (  # noqa: E402
    BudgetProfile, SyncSpecConfig, parse_budget_profiles,
)


def test_v11_default_profiles_and_budget_invariant() -> None:
    cfg = SyncSpecConfig()
    assert [(p.kd, p.kv) for p in cfg.budget_profiles] == [
        (0, 0), (8, 4), (8, 8), (16, 4), (16, 8), (16, 12), (16, 16)
    ]
    assert all(p.kv <= p.kd for p in cfg.budget_profiles)
    assert cfg.source_ngram_min == 2
    assert cfg.source_ngram_max == 6
    assert cfg.source_chunk_size == 128
    assert cfg.source_top_r == 8
    assert cfg.top_m == 16


def test_config_rejects_invalid_profile_and_round_trips_json(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="K_v"):
        BudgetProfile(kd=4, kv=8)

    cfg = SyncSpecConfig(budget_profiles=(BudgetProfile(0, 0),))
    path = tmp_path / "syncspec.json"
    cfg.save(path)
    loaded = SyncSpecConfig.load(path)
    assert loaded == cfg


def test_parse_budget_profiles_supports_finite_adaptive_profile_set() -> None:
    profiles = parse_budget_profiles("8:4,8:8,16:4,16:8,16:12,16:16")
    assert [(profile.kd, profile.kv) for profile in profiles] == [
        (0, 0), (8, 4), (8, 8), (16, 4), (16, 8), (16, 12), (16, 16),
    ]


def test_config_offline_requires_local_model_paths() -> None:
    cfg = SyncSpecConfig(offline=True, target_model="/missing/target")
    with pytest.raises(FileNotFoundError):
        cfg.validate_model_paths()
