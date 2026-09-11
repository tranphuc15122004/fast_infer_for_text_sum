from __future__ import annotations

from analyze.safe_budget.e26_budget_oracle import analyze_dataset, conservative_safe_budget


def row(doc: str, selector: str, label: str, budget: int, quality: float, cost: float) -> dict:
    return {
        "example_id": doc,
        "selector": selector,
        "budget_label": label,
        "token_budget": budget,
        "original_tokens": 1000,
        "selected_tokens": budget,
        "rougeL": quality,
        "pipeline_e2e_ms": cost,
    }


def test_conservative_safe_budget_uses_all_higher_conditions() -> None:
    conditions = [
        row("d", "mmr", "tokens_250", 250, 0.70, 10),
        row("d", "mmr", "tokens_500", 500, 0.80, 20),
        row("d", "full", "full", 1000, 0.81, 50),
    ]
    safe = conservative_safe_budget(
        conditions, full_quality=0.81, epsilon=0.02, quality_metric="rougeL"
    )
    assert safe is not None
    assert safe["budget_label"] == "tokens_500"


def test_analyze_marks_missing_budget_grid_as_pilot() -> None:
    rows = [
        row("d1", "full", "full", 1000, 0.80, 100),
        row("d1", "mmr", "tokens_512", 512, 0.80, 60),
        row("d2", "full", "full", 1000, 0.80, 100),
        row("d2", "mmr", "tokens_512", 512, 0.70, 60),
    ]
    result = analyze_dataset(rows, selector="mmr", min_documents=2, required_levels=2)
    assert result["completeness"]["complete_for_full_e26"] is True
    item = result["epsilon_results"]["0.0200"]["oracle_adaptive"]
    assert item["documents_assigned"] == 2
    # d1 can use 512; d2 must conservatively use full.
    assert item["mean_budget_tokens"] == 756.0


def test_analyze_rejects_incomplete_documents_from_fixed_policy() -> None:
    rows = [row("d1", "full", "full", 1000, 0.8, 100)]
    result = analyze_dataset(rows, selector="mmr", min_documents=1, required_levels=1)
    assert result["documents_usable"] == 0
    assert "missing selector=mmr condition" in result["documents_missing_or_invalid"]["d1"]
