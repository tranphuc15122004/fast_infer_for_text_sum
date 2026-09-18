from __future__ import annotations

from src.TrainingFree.run import build_parser


def test_runner_exposes_lease_experiment_and_certificate_thresholds() -> None:
    args = build_parser().parse_args(
        [
            "--experiment",
            "lease",
            "--output",
            "/tmp/lease-output",
            "--delta-anchor",
            "0.05",
            "--delta-cert",
            "0.10",
        ]
    )

    assert args.experiment == "lease"
    assert args.delta_anchor == 0.05
    assert args.delta_cert == 0.10
