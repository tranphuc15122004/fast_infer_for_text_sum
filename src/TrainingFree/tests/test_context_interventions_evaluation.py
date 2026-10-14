from __future__ import annotations

import json

import pytest

from TrainingFree.context_interventions.evaluation import paired_cluster_bootstrap
from TrainingFree.context_interventions.artifacts import CellStore


def test_cluster_bootstrap_resamples_source_groups_and_is_reproducible() -> None:
    rows = [
        {"dataset": dataset, "source_group_id": f"{dataset}-{index}",
         "full": base, "variant": base + delta}
        for dataset, base, deltas in (
            ("a", 10.0, (1.0, 2.0, 3.0)),
            ("b", 20.0, (2.0, 4.0, 6.0)),
        )
        for index, delta in enumerate(deltas)
    ]
    result = paired_cluster_bootstrap(
        rows,
        baseline_field="full",
        variant_field="variant",
        metric="difference",
        resamples=500,
        seed=42,
    )
    repeat = paired_cluster_bootstrap(
        rows,
        baseline_field="full",
        variant_field="variant",
        metric="difference",
        resamples=500,
        seed=42,
    )
    assert result == repeat
    assert result["n_source_groups"] == 6
    assert result["estimate"] == pytest.approx(3.0)


def test_speedup_uses_paired_geometric_mean_of_group_ratios() -> None:
    rows = [
        {"dataset": "a", "source_group_id": "a1", "full": 100.0, "variant": 50.0},
        {"dataset": "a", "source_group_id": "a2", "full": 100.0, "variant": 100.0},
        {"dataset": "b", "source_group_id": "b1", "full": 400.0, "variant": 100.0},
        {"dataset": "b", "source_group_id": "b2", "full": 100.0, "variant": 100.0},
    ]
    result = paired_cluster_bootstrap(
        rows,
        baseline_field="full",
        variant_field="variant",
        metric="speedup",
        resamples=500,
        seed=7,
    )
    expected = ((2.0 * 1.0) ** 0.5 * (4.0 * 1.0) ** 0.5) ** 0.5
    assert result["estimate"] == pytest.approx(expected)
    assert result["n_source_groups"] == 4


def test_duplicate_group_cells_are_aggregated_before_bootstrap() -> None:
    rows = [
        {"dataset": "a", "source_group_id": "same", "full": 10.0, "variant": 8.0},
        {"dataset": "a", "source_group_id": "same", "full": 14.0, "variant": 12.0},
        {"dataset": "a", "source_group_id": "other", "full": 10.0, "variant": 9.0},
    ]
    result = paired_cluster_bootstrap(
        rows,
        baseline_field="full",
        variant_field="variant",
        metric="difference",
        resamples=100,
        seed=1,
    )
    assert result["n_source_groups"] == 2
    assert result["estimate"] == pytest.approx(-1.5)


def test_cell_store_resume_requires_matching_run_signature(tmp_path) -> None:
    store = CellStore(tmp_path / "a1", signature="run-hash", resume=False)
    store.save("state/action", {"status": "ok", "cell_id": "state/action"})
    resumed = CellStore(tmp_path / "a1", signature="run-hash", resume=True)
    assert resumed.load("state/action")["status"] == "ok"
    incompatible = CellStore(tmp_path / "a1", signature="changed", resume=True)
    with pytest.raises(ValueError, match="signature"):
        incompatible.load("state/action")


def test_cell_store_resume_replaces_failed_cell_and_keeps_success_immutable(tmp_path) -> None:
    store = CellStore(tmp_path / "a1", signature="run-hash", resume=False)
    store.save("state/action", {"status": "error", "reason": "temporary"})

    resumed = CellStore(tmp_path / "a1", signature="run-hash", resume=True)
    resumed.save("state/action", {"status": "ok", "value": 7})
    assert resumed.load("state/action")["status"] == "ok"
    assert resumed.load("state/action")["value"] == 7

    with pytest.raises(ValueError, match="refusing to replace completed cell"):
        resumed.save("state/action", {"status": "ok", "value": 8})


def test_cell_store_reads_cells_written_by_current_non_resume_run(tmp_path) -> None:
    store = CellStore(tmp_path / "a1", signature="run-hash", resume=False)
    store.save("state/action", {"status": "ok", "value": 7})

    assert store.rows()[0]["value"] == 7


def test_checkpoint_identity_corruption_is_rejected(tmp_path) -> None:
    store = CellStore(tmp_path / "a1", signature="run-hash", resume=False)
    store.save("original", {"status": "ok"})
    path = store.path("original")
    row = json.loads(path.read_text())
    row["cell_id"] = "another-cell"
    path.write_text(json.dumps(row), encoding="utf-8")
    resumed = CellStore(tmp_path / "a1", signature="run-hash", resume=True)
    with pytest.raises(ValueError, match="signature"):
        resumed.rows()
