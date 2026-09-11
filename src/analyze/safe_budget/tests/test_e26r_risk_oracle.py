from analyze.safe_budget.e26r_risk_oracle import analyze_dataset


def _row(doc, selector, budget_label, quality, cost, bert=None):
    return {
        "example_id": doc,
        "selector": selector,
        "budget_label": budget_label,
        "original_tokens": 1000,
        "selected_tokens": 1000 if budget_label == "full" else int(budget_label.split("_")[-1]),
        "token_budget": 1000 if budget_label == "full" else int(budget_label.split("_")[-1]),
        "pipeline_e2e_ms": cost,
        "rougeL": quality,
        "bertscore_f1": quality if bert is None else bert,
    }


def test_fixed_uses_document_risk_not_mean_quality():
    rows = [
        _row("a", "full", "full", 0.80, 100.0),
        _row("a", "mmr", "tokens_512", 0.80, 10.0),
        _row("b", "full", "full", 0.80, 100.0),
        _row("b", "mmr", "tokens_512", 0.70, 10.0),
    ]
    result = analyze_dataset(
        rows,
        selector="mmr",
        quality_metrics=["rougeL", "bertscore_f1"],
        epsilons=[0.02],
        alphas=[0.0, 0.5],
        required_levels=1,
        min_documents=1,
    )
    alpha_zero = result["epsilon_results"]["0.0200"]["alpha_results"]["0.0000"]
    alpha_half = result["epsilon_results"]["0.0200"]["alpha_results"]["0.5000"]
    assert alpha_zero["fixed"]["label"] == "full"
    assert alpha_zero["fixed"]["mean_cost_ms"] == 100.0
    assert alpha_half["fixed"]["label"] == "tokens_512"
    assert alpha_half["fixed"]["risk_rate"] == 0.5


def test_adaptive_oracle_uses_same_risk_budget():
    rows = [
        _row("a", "full", "full", 0.80, 100.0),
        _row("a", "mmr", "tokens_512", 0.80, 10.0),
        _row("b", "full", "full", 0.80, 100.0),
        _row("b", "mmr", "tokens_512", 0.70, 10.0),
    ]
    result = analyze_dataset(
        rows,
        selector="mmr",
        quality_metrics=["rougeL", "bertscore_f1"],
        epsilons=[0.02],
        alphas=[0.0, 0.5],
        required_levels=1,
        min_documents=1,
    )
    alpha_zero = result["epsilon_results"]["0.0200"]["alpha_results"]["0.0000"]
    alpha_half = result["epsilon_results"]["0.0200"]["alpha_results"]["0.5000"]
    assert alpha_zero["adaptive_oracle"]["violating_documents"] == 0
    assert alpha_zero["adaptive_oracle"]["mean_cost_ms"] == 55.0
    assert alpha_half["adaptive_oracle"]["violating_documents"] == 1
    assert alpha_half["adaptive_oracle"]["mean_cost_ms"] == 10.0


def test_missing_requested_quality_metric_is_incomplete():
    rows = [
        {
            "example_id": "a",
            "selector": "full",
            "budget_label": "full",
            "original_tokens": 1000,
            "pipeline_e2e_ms": 100.0,
            "rougeL": 0.8,
        },
        {
            "example_id": "a",
            "selector": "mmr",
            "budget_label": "tokens_512",
            "original_tokens": 1000,
            "selected_tokens": 512,
            "token_budget": 512,
            "pipeline_e2e_ms": 10.0,
            "rougeL": 0.8,
        },
    ]
    result = analyze_dataset(
        rows,
        selector="mmr",
        quality_metrics=["rougeL", "bertscore_f1"],
        epsilons=[0.02],
        alphas=[0.0],
        required_levels=1,
        min_documents=1,
    )
    assert result["completeness"]["status"] == "pilot_incomplete"
    assert result["completeness"]["quality_requirement_met"] is False
    assert result["documents_usable"] == 0
    assert "bertscore_f1" in result["missing_quality_metrics"]
