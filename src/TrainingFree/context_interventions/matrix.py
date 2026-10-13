"""Pre-registered cell grids for the B200 campaign."""

from __future__ import annotations

from typing import Any


def a1_matrix(phase: str) -> dict[str, Any]:
    controls = ("full", "source_only", "recent_only", "source_recent")
    if phase == "pilot":
        return {
            "controls": controls,
            "attention_scoring": ("target_1", "target_9", "target_17", "target_25", "target_33", "target_mean5", "target_corresponding"),
            "attention_deletions": (1024, 2048),
            "matched_deletions": (128,),
            "matched_regions": ("source", "recent", "old_output"),
        }
    if phase == "dev":
        return {
            "controls": controls,
            "attention_scoring": ("target_1", "target_9", "target_17", "target_25", "target_33", "target_mean5", "target_corresponding"),
            "attention_deletions": (1024, 2048, 4096),
            "matched_deletions": (64, 128),
            "matched_regions": ("source", "recent", "old_output"),
        }
    if phase == "confirm":
        return {
            "controls": controls,
            "attention_scoring": ("target_mean5", "target_corresponding"),
            "attention_deletions": (2048,),
            "matched_deletions": (128,),
            "matched_regions": ("source", "recent"),
        }
    raise ValueError(f"unsupported A1 phase: {phase}")


def a2_matrix(phase: str) -> dict[str, Any]:
    budgets = {"pilot": (1024, 2048), "dev": (1024, 2048, 4096), "confirm": (2048,)}
    try:
        selected_budgets = budgets[phase]
    except KeyError as exc:
        raise ValueError(f"unsupported A2 phase: {phase}") from exc
    return {
        "budgets": selected_budgets,
        "selectors": ("random", "recency", "target_parent", "previous_dflash", "current_dflash_oracle"),
        "random_seeds": (17, 42, 123),
        "primary_budget": 2048,
    }


def a3_matrix(phase: str) -> dict[str, Any]:
    budgets = {
        "pilot": (1024, 2048),
        "dev": (512, 1024, 2048, 4096, 8192, "full"),
        "confirm": (2048,),
    }
    try:
        return {
            "budgets": budgets[phase],
            "refresh_periods": (2, 4, 8, 16) if phase == "dev" else (4,),
            "ranking_ages": (1, 2, 4, 8, 16) if phase == "dev" else (1,),
            "timing_repetitions": 1 if phase == "pilot" else 3,
        }
    except KeyError as exc:
        raise ValueError(f"unsupported A3 phase: {phase}") from exc
