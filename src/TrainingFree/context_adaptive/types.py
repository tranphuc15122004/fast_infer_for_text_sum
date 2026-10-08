"""Small shared contracts for the Context-Adaptive DFlash executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Budget = int | Literal["full"]


@dataclass(frozen=True, order=True)
class Action:
    budget: Budget
    gamma: int
    length_mode: str = "draft_shape"

    def __post_init__(self) -> None:
        if self.budget != "full" and (not isinstance(self.budget, int) or self.budget < 1):
            raise ValueError("budget must be a positive integer or 'full'")
        if self.gamma < 1:
            raise ValueError("draft actions require gamma >= 1")
        if self.length_mode not in {"draft_shape", "verify_prefix"}:
            raise ValueError(f"unsupported length mode: {self.length_mode}")

    @property
    def block_size(self) -> int:
        return self.gamma + 1

    def key(self) -> str:
        budget = "full" if self.budget == "full" else str(self.budget)
        return f"{self.length_mode}|{budget}|{self.gamma}"


@dataclass(frozen=True)
class PromptLayout:
    prompt_tokens: int
    source_positions: tuple[int, ...]
    global_positions: tuple[int, ...]
    source_chunks: tuple[tuple[int, ...], ...]
    prompt_hash: str
    source_available: bool
    source_reason: str | None = None
    dataset: str | None = None
    source_group_id: str | None = None


@dataclass(frozen=True)
class RoundState:
    round_index: int
    logical_length: int
    processed_output_tokens: int
    remaining_output_tokens: int
    parent_entropy: float | None
    source_concentration: float | None
    history_acceptance: float | None
    ranking_age: int
    refresh_required: bool
    source_origin_round: int | None = None


@dataclass(frozen=True)
class Selection:
    positions_by_layer: tuple[tuple[int, ...], ...]
    requested_budget: Budget
    selector_id: str
    ranking_origin_round: int | None = None
    unused_budget_by_layer: tuple[int, ...] = ()


@dataclass(frozen=True)
class LayerContext:
    key: Any
    value: Any
    positions: Any


@dataclass
class DraftProposal:
    token_ids: Any
    attention_scores_by_layer: dict[int, dict[int, float]] = field(default_factory=dict)
    draft_latency_ms: float = 0.0
    context_tokens_by_layer: tuple[int, ...] = ()
    gather_bytes: int = 0
    draft_tokens_proposed: int = 0


@dataclass
class VerificationOutcome:
    accepted_prefix: int
    pending_anchor: int
    target_logits: Any
    retained_features: Any
    logical_length_before: int
    logical_length_after: int
    executed_gamma: int
    eos_offset: int | None = None
    processed_commits: int = 0
    trimmed_commits: int = 0
    prefix_right_censored: bool = False


@dataclass
class GenerationResult:
    output_ids: Any
    output_tokens: int
    rounds: list[dict[str, Any]]
    counters: dict[str, Any]
    timings: dict[str, float | None]
    correctness_status: str
    status: str = "ok"
    error: str | None = None
