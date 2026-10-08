"""Cost-per-commit action controllers for fixed and adaptive baselines."""

from __future__ import annotations

import time
from functools import cmp_to_key
from dataclasses import replace
from typing import Any, Iterable

from .config import AdaptiveConfig
from .statistics import AdaptiveStatistics, state_bucket
from .types import Action, RoundState


class AdaptiveController:
    def __init__(
        self,
        config: AdaptiveConfig,
        statistics: AdaptiveStatistics | None,
        *,
        variant: str = "joint",
        fixed_budget: int | str = "full",
        fixed_gamma: int = 15,
        gamma_reference: int = 15,
    ) -> None:
        self.config = config
        self.statistics = statistics
        self.variant = variant
        self.fixed_budget = fixed_budget
        self.fixed_gamma = fixed_gamma
        self.gamma_reference = gamma_reference
        self.history_acceptance: float | None = None
        self.choose_ms_total = 0.0
        self.choose_calls = 0
        self.last_reason = ""

    def state(self, **values: Any) -> RoundState:
        return RoundState(history_acceptance=self.history_acceptance, **values)

    def _utility(self, action: Action, state: RoundState) -> float | None:
        if self.statistics is None:
            return None
        stats_state = state
        if self.variant in {"b_only_history", "b_only_entropy"}:
            stats_state = replace(stats_state, source_concentration=None)
        if self.variant in {"b_only_history", "joint_no_entropy"}:
            stats_state = replace(stats_state, parent_entropy=None)
        if self.variant == "joint_no_source_relevance":
            stats_state = replace(stats_state, source_concentration=None)
        commits = self.statistics.expected_commits(action, stats_state)
        cost = self.statistics.cost_ms(action, stats_state, self.variant)
        if commits is None or cost is None or commits <= 0:
            return None
        return cost / commits

    def _best(self, actions: Iterable[Action], state: RoundState) -> Action | None:
        actions = tuple(actions)
        rows = [(self._utility(action, state), action) for action in actions]
        supported = [(float(cost), action) for cost, action in rows if cost is not None]
        if not supported:
            return None

        def compare(left: tuple[float, Action], right: tuple[float, Action]) -> int:
            if abs(left[0] - right[0]) > 1e-9:
                return -1 if left[0] < right[0] else 1
            left_budget = float("inf") if left[1].budget == "full" else int(left[1].budget)
            right_budget = float("inf") if right[1].budget == "full" else int(right[1].budget)
            if left_budget != right_budget:
                return -1 if left_budget > right_budget else 1
            return (left[1].gamma > right[1].gamma) - (left[1].gamma < right[1].gamma)

        supported.sort(key=cmp_to_key(compare))
        best_cost, best_action = supported[0]
        full_actions = [action for action in actions if action.budget == "full"]
        fallback_costs = [self._utility(action, state) for action in full_actions]
        fallback_costs = [float(value) for value in fallback_costs if value is not None]
        # A reduced action has no justified predicted benefit until it can be
        # compared with a measured full-context action at this state.
        if not fallback_costs and best_action.budget != "full":
            return None
        if fallback_costs and best_action.budget != "full":
            fallback_cost = min(fallback_costs)
            if best_cost > (1.0 - self.config.controller_margin) * fallback_cost:
                return next(action for action in full_actions if self._utility(action, state) == fallback_cost)
        return best_action

    def _entropy_gamma(self, state: RoundState) -> int | None:
        if state.parent_entropy is None or self.statistics is None:
            return None
        bucket = state_bucket(state, self.statistics.cutpoints)
        index = int(bucket["entropy"]) if bucket["entropy"] != "unknown" else 3
        return (15, 15, 7, 3)[min(index, 3)]

    def choose(self, state: RoundState, feasible_actions: Iterable[Action]) -> tuple[Action, str, float]:
        started = time.perf_counter()
        actions = tuple(feasible_actions)
        if not actions:
            raise ValueError("controller received no feasible action")
        selected: Action | None = None
        reason = self.variant

        if self.variant == "dflash_full_fixed":
            desired = Action("full", self.gamma_reference, self.config.length_mode)
            selected = next((action for action in actions if action == desired), None)
            reason = "full_context_fixed_gamma" if selected else "full_fixed_action_unsupported"
        elif self.variant in {"fixed", "best_fixed_pair"}:
            canonical_budget = (
                "full"
                if self.fixed_budget == "full"
                or int(self.fixed_budget) >= state.logical_length
                else int(self.fixed_budget)
            )
            desired = Action(canonical_budget, self.fixed_gamma, self.config.length_mode)
            selected = next((action for action in actions if action == desired), None)
            reason = "fixed_action" if selected else "fixed_action_unsupported"
        elif self.variant == "a_only":
            selected = self._best((a for a in actions if a.gamma == self.fixed_gamma), state)
        elif self.variant == "b_only_entropy":
            desired_gamma = self._entropy_gamma(state)
            selected = self._best(
                (a for a in actions if a.budget == "full" and (desired_gamma is None or a.gamma == desired_gamma)),
                state,
            )
            if selected is None:
                selected = self._best((a for a in actions if a.budget == "full"), state)
        elif self.variant == "b_only_history":
            history_state = RoundState(**{**state.__dict__, "parent_entropy": None, "source_concentration": None})
            selected = self._best((a for a in actions if a.budget == "full"), history_state)
        elif self.variant == "independent_ab":
            budget_action = self._best((a for a in actions if a.gamma == self.gamma_reference), state)
            gamma_action = self._best((a for a in actions if a.budget == "full"), state)
            if budget_action is not None and gamma_action is not None:
                selected = next(
                    (a for a in actions if a.budget == budget_action.budget and a.gamma == gamma_action.gamma),
                    None,
                )
                reason = "independent_reference_axes"
        elif self.variant in {"joint_no_entropy", "joint_no_source_relevance"}:
            if self.variant == "joint_no_entropy":
                state = RoundState(**{**state.__dict__, "parent_entropy": None})
            else:
                state = RoundState(**{**state.__dict__, "source_concentration": None})
            selected = self._best(actions, state)
        else:
            selected = self._best(actions, state)

        if selected is None:
            full = [action for action in actions if action.budget == "full"]
            preferred_gamma = self.gamma_reference if self.variant == "dflash_full_fixed" else self.fixed_gamma
            tail_shapes = [action for action in full if action.gamma <= preferred_gamma]
            selected = (
                max(tail_shapes, key=lambda action: action.gamma)
                if tail_shapes else (max(full, key=lambda action: action.gamma) if full else actions[-1])
            )
            if self.variant == "dflash_full_fixed":
                reason = "full_fixed_tail_shape"
            elif self.statistics is not None and full and not any(
                self._utility(action, state) is not None for action in full
            ):
                reason = "full_fallback_uncalibrated_cost"
            else:
                reason = "full_fallback_no_supported_action"
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.choose_ms_total += elapsed_ms
        self.choose_calls += 1
        self.last_reason = reason
        return selected, reason, elapsed_ms

    def observe(self, state: RoundState, action: Action, accepted_prefix: int, action_cost_ms: float | None = None) -> None:
        if self.statistics is not None:
            self.statistics.observe(action, state, accepted_prefix, action_cost_ms)
        ratio = accepted_prefix / action.gamma
        alpha = self.config.history_alpha
        self.history_acceptance = ratio if self.history_acceptance is None else (1.0 - alpha) * self.history_acceptance + alpha * ratio
