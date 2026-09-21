from __future__ import annotations

from src.TrainingFree.run import build_parser


def test_runner_exposes_hierarchy_experiment_and_thresholds() -> None:
    args = build_parser().parse_args(
        [
            "--experiment",
            "hierarchy",
            "--output",
            "/tmp/hierarchy-output",
            "--region-size",
            "1024",
            "--hierarchy-mass-budget",
            "0.01",
        ]
    )

    assert args.experiment == "hierarchy"
    assert args.region_size == 1024
    assert args.hierarchy_mass_budget == 0.01
