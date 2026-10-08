"""Validated runtime settings for the training-free controller."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .types import Action


def _csv_ints(value: str, name: str) -> tuple[int, ...]:
    try:
        result = tuple(dict.fromkeys(int(part.strip()) for part in value.split(",") if part.strip()))
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


@dataclass(frozen=True)
class AdaptiveConfig:
    budgets: tuple[int | str, ...] = (1024, 2048, 4096, "full")
    gammas: tuple[int, ...] = (3, 7, 11, 15)
    selector: str = "target_parent"
    target_layer: str | int = "middle"
    refresh_period: int = 4
    signal_update_period: int = 1
    source_chunk_size: int = 128
    source_anchors: int = 64
    recent_output: int = 256
    prior_strength: float = 16.0
    min_state_support: int = 8
    statistics_decay: float = 0.95
    history_alpha: float = 0.1
    latency_ema_alpha: float = 0.1
    controller_margin: float = 0.02
    entropy_signal_temperature: float = 1.0
    timing_mode: str = "wall"
    cost_update_mode: str = "frozen_cost"
    statistics_update_mode: str = "online"
    length_mode: str = "draft_shape"
    output_root: str = "outputs/context_adaptive_dflash"
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.budgets:
            raise ValueError("at least one draft context budget is required")
        if "full" not in self.budgets:
            raise ValueError("context budgets must include 'full' as the correctness/cost fallback")
        if not self.gammas or any(g < 1 for g in self.gammas):
            raise ValueError("gammas must be positive candidate counts; AR gamma 0 is a separate mode")
        if any(b != "full" and (not isinstance(b, int) or b < 1) for b in self.budgets):
            raise ValueError("budgets must be positive integers or 'full'")
        if self.selector not in {"target_parent", "draft_refresh", "recent_only", "random", "lexical", "feature_cosine"}:
            raise ValueError(f"unknown deployable selector: {self.selector}")
        if self.target_layer != "middle" and (not isinstance(self.target_layer, int) or self.target_layer < 0):
            raise ValueError("target_layer must be 'middle' or a nonnegative layer index")
        if self.refresh_period < 1 or self.signal_update_period < 1:
            raise ValueError("refresh and signal update periods must be positive")
        if self.source_chunk_size < 1 or self.source_anchors < 0 or self.recent_output < 0:
            raise ValueError("chunk/protection values are outside their valid ranges")
        if self.prior_strength < 0 or self.min_state_support < 1:
            raise ValueError("prior strength and support threshold must be nonnegative/positive")
        if not 0.0 < self.statistics_decay < 1.0:
            raise ValueError("statistics_decay must be in (0,1)")
        if not 0.0 < self.history_alpha <= 1.0 or not 0.0 < self.latency_ema_alpha <= 1.0:
            raise ValueError("EMA alphas must be in (0,1]")
        if not 0.0 <= self.controller_margin < 1.0:
            raise ValueError("controller_margin must be in [0,1)")
        if self.entropy_signal_temperature <= 0:
            raise ValueError("entropy signal temperature must be positive")
        if self.timing_mode not in {"diagnostic", "wall"}:
            raise ValueError("timing_mode must be diagnostic or wall")
        if self.cost_update_mode != "frozen_cost":
            raise ValueError(
                "Context-Adaptive DFlash V1 supports frozen_cost; async event accounting is not implemented"
            )
        if self.statistics_update_mode not in {"online", "frozen"}:
            raise ValueError("statistics_update_mode must be online or frozen")
        if self.statistics_update_mode == "frozen" and self.cost_update_mode != "frozen_cost":
            raise ValueError("frozen statistics require frozen_cost mode")
        if self.length_mode != "draft_shape":
            raise ValueError("Context-Adaptive DFlash V1 supports draft_shape; verify_prefix is not implemented")

    @property
    def actions(self) -> tuple[Action, ...]:
        return tuple(Action(budget, gamma, self.length_mode) for budget in self.budgets for gamma in self.gammas)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "AdaptiveConfig":
        env = os.environ if environ is None else environ
        budget_text = env.get("CAD_BUDGETS", "1024,2048,4096,full")
        budgets: list[int | str] = []
        for part in budget_text.split(","):
            item = part.strip()
            if item == "full":
                budgets.append(item)
            elif item:
                budgets.append(int(item))
        target_layer: str | int = env.get("CAD_TARGET_LAYER", "middle")
        if target_layer != "middle":
            target_layer = int(target_layer)
        return cls(
            budgets=tuple(budgets),
            gammas=_csv_ints(env.get("CAD_GAMMAS", "3,7,11,15"), "CAD_GAMMAS"),
            selector=env.get("CAD_SELECTOR", "target_parent"),
            target_layer=target_layer,
            refresh_period=int(env.get("CAD_REFRESH_PERIOD", "4")),
            signal_update_period=int(env.get("CAD_SIGNAL_UPDATE_PERIOD", "1")),
            source_chunk_size=int(env.get("CAD_SOURCE_CHUNK_SIZE", "128")),
            source_anchors=int(env.get("CAD_SOURCE_ANCHORS", "64")),
            recent_output=int(env.get("CAD_RECENT_OUTPUT", "256")),
            prior_strength=float(env.get("CAD_PRIOR_STRENGTH", "16")),
            min_state_support=int(env.get("CAD_MIN_STATE_SUPPORT", "8")),
            statistics_decay=float(env.get("CAD_STATISTICS_DECAY", "0.95")),
            history_alpha=float(env.get("CAD_HISTORY_ALPHA", "0.1")),
            latency_ema_alpha=float(env.get("CAD_LATENCY_EMA_ALPHA", "0.1")),
            controller_margin=float(env.get("CAD_CONTROLLER_MARGIN", "0.02")),
            entropy_signal_temperature=float(env.get("CAD_ENTROPY_SIGNAL_TEMPERATURE", "1")),
            timing_mode=env.get("CAD_TIMING_MODE", "wall"),
            cost_update_mode=env.get("CAD_COST_UPDATE_MODE", "frozen_cost"),
            statistics_update_mode=env.get("CAD_STATISTICS_UPDATE_MODE", "online"),
            length_mode=env.get("CAD_LENGTH_MODE", "draft_shape"),
            output_root=env.get("CAD_OUTPUT_ROOT", "outputs/context_adaptive_dflash"),
            seed=int(env.get("LONG_BENCH_SEED", "42")),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["actions"] = [asdict(action) for action in self.actions]
        return result

    @classmethod
    def from_locked(cls, path: Path) -> tuple["AdaptiveConfig", dict[str, Any]]:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if obj.get("schema_version") != "cadflash.locked.v1":
            raise ValueError("unsupported locked-config schema")
        values = dict(obj["config"])
        values.pop("actions", None)
        values["budgets"] = tuple(values["budgets"])
        values["gammas"] = tuple(int(value) for value in values["gammas"])
        config = cls(**values)
        return config, obj
