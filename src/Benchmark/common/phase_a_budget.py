"""Request-local adaptive budget search for native FA4 DFlash Phase A."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class _BudgetStats:
    weight: float = 0.0
    round_ms: float = 0.0
    committed_tokens: float = 0.0
    draft_ms: float = 0.0
    verification_ms: float = 0.0

    @property
    def ms_per_committed_token(self) -> float | None:
        if self.weight <= 0.0 or self.committed_tokens <= 0.0:
            return None
        return self.round_ms / self.committed_tokens


class AdaptiveBudgetController:
    """Hill-climb integer token budgets using measured round cost and commits.

    The controller tests one budget for a short dwell window before comparing
    its exponentially weighted milliseconds per committed token with the best
    budget observed in this request. It never fabricates outcomes for budgets
    that were not executed.
    """

    def __init__(
        self,
        initial_budget: int = 2048,
        min_budget: int = 512,
        max_budget: int = 8192,
        chunk_size: int = 128,
        window_rounds: int = 3,
        min_gain: float = 0.03,
        decay: float = 0.9,
        recheck_rounds: int = 12,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("Phase A chunk size must be positive")
        if min_budget < 1 or max_budget < min_budget:
            raise ValueError("Phase A adaptive budget bounds must be positive and ordered")
        if not min_budget <= initial_budget <= max_budget:
            raise ValueError("Phase A initial budget must be within adaptive bounds")
        if (initial_budget - min_budget) % chunk_size or (max_budget - initial_budget) % chunk_size:
            raise ValueError("Phase A adaptive budget bounds must align to the configured chunk size")
        if window_rounds < 1 or recheck_rounds < window_rounds:
            raise ValueError("Phase A adaptive windows must be positive and ordered")
        if not 0.0 <= min_gain < 1.0:
            raise ValueError("Phase A adaptive minimum gain must be in [0, 1)")
        if not 0.0 < decay < 1.0:
            raise ValueError("Phase A adaptive decay must be in (0, 1)")

        self.initial_budget = int(initial_budget)
        self.min_budget = int(min_budget)
        self.max_budget = int(max_budget)
        self.chunk_size = int(chunk_size)
        self.window_rounds = int(window_rounds)
        self.min_gain = float(min_gain)
        self.decay = float(decay)
        self.recheck_rounds = int(recheck_rounds)
        self.reset()

    def reset(self) -> None:
        self.active_budget = self.initial_budget
        self.incumbent_budget = self.initial_budget
        self.step_tokens = self.chunk_size
        self.direction = -1
        self.rounds = 0
        self.active_rounds = 0
        self.rounds_since_recheck = 0
        self.failed_min_step_sides: set[int] = set()
        self.settled = False
        self.stats: dict[int, _BudgetStats] = {}
        self.last_reason = "initial_budget"

    def _snap_budget(self, budget: int) -> int:
        steps = round((int(budget) - self.initial_budget) / self.chunk_size)
        snapped = self.initial_budget + steps * self.chunk_size
        return max(self.min_budget, min(self.max_budget, snapped))

    def _cost(self, budget: int) -> float | None:
        state = self.stats.get(budget)
        return state.ms_per_committed_token if state is not None else None

    def _candidate(self, direction: int, step: int, context_length: int) -> int:
        candidate = self._snap_budget(self.incumbent_budget + direction * step)
        upper = min(self.max_budget, int(context_length))
        if upper < self.min_budget:
            return self.min_budget
        if candidate <= upper:
            return candidate
        # Keep the candidate on the configured chunk lattice while respecting
        # the available context. Rounding the cap back to nearest could exceed it.
        lattice_steps = (upper - self.initial_budget) // self.chunk_size
        return max(self.min_budget, min(self.max_budget, self.initial_budget + lattice_steps * self.chunk_size))

    def observe(
        self,
        *,
        budget: int,
        round_ms: float,
        committed_tokens: int,
        context_length: int,
        draft_ms: float | None = None,
        verification_ms: float | None = None,
    ) -> None:
        """Update statistics for the budget actually used by this round."""
        del context_length  # Kept in the API for future context-bucketed priors.
        values = (float(round_ms), int(committed_tokens))
        if not math.isfinite(values[0]) or values[0] <= 0.0 or values[1] <= 0:
            return
        budget = self._snap_budget(int(budget))
        for state in self.stats.values():
            state.weight *= self.decay
            state.round_ms *= self.decay
            state.committed_tokens *= self.decay
            state.draft_ms *= self.decay
            state.verification_ms *= self.decay

        state = self.stats.setdefault(budget, _BudgetStats())
        state.weight += 1.0
        state.round_ms += values[0]
        state.committed_tokens += float(values[1])
        if draft_ms is not None and math.isfinite(float(draft_ms)) and float(draft_ms) >= 0:
            state.draft_ms += float(draft_ms)
        if (
            verification_ms is not None
            and math.isfinite(float(verification_ms))
            and float(verification_ms) >= 0
        ):
            state.verification_ms += float(verification_ms)
        self.rounds += 1
        self.rounds_since_recheck += 1
        if budget == self.active_budget:
            self.active_rounds += 1

    def choose(
        self,
        *,
        context_length: int,
        allow_exploration: bool = True,
    ) -> tuple[int, str]:
        """Choose the next action after a dwell window has completed."""
        context_length = int(context_length)
        if not allow_exploration:
            self.active_budget = self.incumbent_budget
            self.active_rounds = 0
            self.last_reason = "tail_hold_incumbent"
            return self.active_budget, self.last_reason
        if context_length <= self.min_budget:
            self.active_budget = self._snap_budget(context_length)
            self.incumbent_budget = self.active_budget
            self.last_reason = "context_at_or_below_minimum"
            return self.active_budget, self.last_reason

        if self.settled and self.rounds_since_recheck >= self.recheck_rounds:
            self.settled = False
            self.rounds_since_recheck = 0
            self.failed_min_step_sides.clear()
            self.direction = -1
            self.step_tokens = self.chunk_size
            self.active_budget = self.incumbent_budget
            self.active_rounds = self.window_rounds

        if self.active_rounds < self.window_rounds:
            self.last_reason = "hold_for_window"
            return self.active_budget, self.last_reason

        if self.settled:
            self.last_reason = "hold_incumbent"
            return self.incumbent_budget, self.last_reason

        active_cost = self._cost(self.active_budget)
        incumbent_cost = self._cost(self.incumbent_budget)
        if self.active_budget != self.incumbent_budget and active_cost is not None:
            if incumbent_cost is None or active_cost <= (1.0 - self.min_gain) * incumbent_cost:
                side = 1 if self.active_budget > self.incumbent_budget else -1
                self.incumbent_budget = self.active_budget
                self.direction = side
                self.step_tokens = min(self.step_tokens * 2, 1024)
                self.failed_min_step_sides.clear()
                reason = "accepted_improvement"
            else:
                side = 1 if self.active_budget > self.incumbent_budget else -1
                if self.step_tokens <= self.chunk_size:
                    self.failed_min_step_sides.add(side)
                self.direction = -side
                self.step_tokens = max(self.chunk_size, self.step_tokens // 2)
                if self.step_tokens <= self.chunk_size and len(self.failed_min_step_sides) == 2:
                    self.settled = True
                    self.rounds_since_recheck = 0
                    self.active_budget = self.incumbent_budget
                    self.active_rounds = 0
                    self.last_reason = "local_minimum_hold"
                    return self.active_budget, self.last_reason
                reason = "reversed_after_no_improvement"
        else:
            reason = "initial_local_probe"

        candidate = self._candidate(self.direction, self.step_tokens, context_length)
        if candidate == self.incumbent_budget:
            self.direction *= -1
            candidate = self._candidate(self.direction, self.step_tokens, context_length)
        if candidate == self.incumbent_budget:
            self.settled = True
            self.rounds_since_recheck = 0
            self.active_budget = self.incumbent_budget
            self.active_rounds = 0
            self.last_reason = "budget_bounds_hold"
            return self.active_budget, self.last_reason

        self.active_budget = candidate
        self.active_rounds = 0
        self.last_reason = reason
        return self.active_budget, reason

    def snapshot(self) -> dict[str, object]:
        return {
            "policy": "round_ms_per_committed_token_local_chunk_search",
            "active_budget": self.active_budget,
            "incumbent_budget": self.incumbent_budget,
            "step_tokens": self.step_tokens,
            "direction": self.direction,
            "rounds": self.rounds,
            "settled": self.settled,
            "last_reason": self.last_reason,
            "ms_per_committed_token_by_budget": {
                str(budget): state.ms_per_committed_token
                for budget, state in sorted(self.stats.items())
            },
            "draft_ms_by_budget": {
                str(budget): state.draft_ms / state.weight if state.weight else None
                for budget, state in sorted(self.stats.items())
            },
            "verification_ms_by_budget": {
                str(budget): state.verification_ms / state.weight if state.weight else None
                for budget, state in sorted(self.stats.items())
            },
        }
