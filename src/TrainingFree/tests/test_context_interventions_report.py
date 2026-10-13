from __future__ import annotations

from TrainingFree.context_interventions.report import summarize_phase


def test_a2_report_pairs_all_random_seeds_with_previous_by_prefix() -> None:
    rows = []
    for dataset in ("gov_report", "multi_news", "qmsum"):
        for group in (f"{dataset}-0", f"{dataset}-1"):
            common = {"experiment": "A2_selector", "phase": "dev", "status": "ok",
                      "dataset": dataset, "source_group_id": group, "prefix_round_index": 3,
                      "nominal_budget": 2048}
            for seed in (17, 42, 123):
                rows.append({**common, "selector_id": f"random_B2048_seed{seed}", "L_action": 4})
            for selector, value in (("previous_dflash_B2048", 6), ("recency_B2048", 5), ("target_parent_B2048", 5)):
                rows.append({**common, "selector_id": selector, "L_action": value})
    report = summarize_phase(rows, phase="dev")
    random_ci = report["family_metrics"]["A2"]["comparisons"]["previous_vs_random"]
    assert random_ci is not None
    assert random_ci["estimate"] == 2.0
    assert random_ci["n_source_groups"] == 6


def test_empty_campaign_report_marks_missing_results_inconclusive() -> None:
    report = summarize_phase([], phase="pilot")
    assert report["family_metrics"]["A1"]["decision"] == "inconclusive"
    assert report["family_metrics"]["A2"]["decision"] == "inconclusive_missing_pairs"
