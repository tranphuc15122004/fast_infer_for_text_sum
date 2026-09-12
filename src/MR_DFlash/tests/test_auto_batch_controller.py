from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "mr_dflash"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def test_auto_batch_grows_until_vram_target_then_stops() -> None:
    from auto_batch import AdaptiveBatchController

    controller = AdaptiveBatchController(
        initial_batch_size=4,
        max_batch_size=128,
        growth_factor=2.0,
        target_vram_gb=170.0,
    )

    assert controller.batch_size("short") == 4
    assert controller.record_success("short", peak_vram_gb=60.0) == 8
    assert controller.record_success("short", peak_vram_gb=120.0) == 16
    assert controller.record_success("short", peak_vram_gb=170.0) == 16
    assert controller.batch_size("short") == 16


def test_auto_batch_oom_halves_batch_and_keeps_other_bucket() -> None:
    from auto_batch import AdaptiveBatchController

    controller = AdaptiveBatchController(
        initial_batch_size=8,
        max_batch_size=128,
        growth_factor=2.0,
    )
    controller.record_success("short", peak_vram_gb=None)
    controller.record_oom("short")

    assert controller.batch_size("short") == 8
    assert controller.batch_size("long") == 8


def test_auto_batch_can_grow_without_telemetry_for_static_pool() -> None:
    from auto_batch import AdaptiveBatchController

    controller = AdaptiveBatchController(
        initial_batch_size=4,
        max_batch_size=64,
        growth_factor=2.0,
    )

    assert controller.record_success("8k", peak_vram_gb=None) == 8
    assert controller.record_success("8k", peak_vram_gb=None) == 16


def test_target_vram_fraction_caps_requested_fraction() -> None:
    from auto_batch import target_memory_fraction

    assert target_memory_fraction(170.0, 180.0, requested_fraction=0.99) == pytest.approx(
        170.0 / 180.0
    )
    assert target_memory_fraction(170.0, 180.0, requested_fraction=0.80) == pytest.approx(0.80)


def test_oom_sets_search_bound_and_recovers_between_safe_and_failed_batch() -> None:
    from auto_batch import AdaptiveBatchController

    controller = AdaptiveBatchController(
        initial_batch_size=4,
        max_batch_size=128,
        growth_factor=2.0,
        target_vram_gb=170.0,
    )
    assert controller.record_success("short", peak_vram_gb=100.0) == 8
    assert controller.record_success("short", peak_vram_gb=120.0) == 16
    assert controller.record_oom("short", attempted_batch_size=16) == 8
    assert controller.record_success("short", peak_vram_gb=130.0) == 12
