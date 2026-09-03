"""Configuration and immutable budget contracts for SyncSpec-v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BudgetProfile:
    """One serving action; ``kd=kv=0`` is the AR fallback."""

    kd: int
    kv: int

    def __post_init__(self) -> None:
        if self.kd < 0 or self.kv < 0:
            raise ValueError("K_d and K_v must be non-negative")
        if self.kv > self.kd:
            raise ValueError("K_v must be <= K_d")

    @property
    def name(self) -> str:
        return "ar" if self.kd == 0 else f"kd{self.kd}_kv{self.kv}"


DEFAULT_BUDGET_PROFILES = (
    BudgetProfile(0, 0),
    BudgetProfile(8, 4),
    BudgetProfile(8, 8),
    BudgetProfile(16, 4),
    BudgetProfile(16, 8),
    BudgetProfile(16, 12),
    BudgetProfile(16, 16),
)


def parse_budget_profiles(value: str | None) -> tuple[BudgetProfile, ...]:
    """Parse ``K_d:K_v`` pairs into the finite serving profile contract.

    The AR profile is always inserted first.  Keeping parsing here lets the
    profile and inference CLIs consume exactly the same budget topology while
    retaining the legacy ``--kd/--kv`` fixed-profile flags.
    """
    if value is None or not str(value).strip():
        return DEFAULT_BUDGET_PROFILES
    profiles: list[BudgetProfile] = [BudgetProfile(0, 0)]
    seen = {(0, 0)}
    for raw_pair in str(value).split(","):
        pair = raw_pair.strip()
        if not pair:
            continue
        fields = pair.split(":")
        if len(fields) != 2:
            raise ValueError(
                f"invalid budget profile {pair!r}; expected K_d:K_v"
            )
        try:
            kd, kv = (int(field.strip()) for field in fields)
        except ValueError as exc:
            raise ValueError(
                f"invalid budget profile {pair!r}; K_d and K_v must be integers"
            ) from exc
        profile = BudgetProfile(kd, kv)
        if profile.kd == 0 and profile.kv == 0:
            continue
        if profile.kd == 0 or profile.kv == 0:
            raise ValueError("speculative budget profiles require K_d >= K_v >= 1")
        key = (profile.kd, profile.kv)
        if key not in seen:
            profiles.append(profile)
            seen.add(key)
    if len(profiles) == 1:
        raise ValueError("at least one speculative budget profile is required")
    return tuple(profiles)


@dataclass(frozen=True)
class SyncSpecConfig:
    """Serializable v1.1 defaults shared by training and inference."""

    vocab_size: int = 0
    hidden_size: int = 0
    top_m: int = 16
    source_ngram_min: int = 2
    source_ngram_max: int = 6
    source_chunk_size: int = 128
    source_top_r: int = 8
    recent_window: int = 128
    diffusion_layers: int = 3
    convolution_kernel: int = 2
    convolution_groups: int = 16
    selector_rank: int = 128
    selector_temperature: float = 1.0
    predicted_spec_gain: float = 0.15
    gate_epsilon: float = 0.03
    feedback_alpha: float = 0.2
    gate_table: dict[str, float] = field(default_factory=dict)
    max_new_tokens: int = 64
    dtype: str = "float32"
    device: str = "cpu"
    offline: bool = True
    target_model: str | None = None
    drafter_checkpoint: str | None = None
    selector_checkpoint: str | None = None
    survival_checkpoint: str | None = None
    runtime_profile: str | None = None
    require_measured_profile: bool = False
    budget_profiles: tuple[BudgetProfile, ...] = field(
        default_factory=lambda: DEFAULT_BUDGET_PROFILES
    )

    def __post_init__(self) -> None:
        profiles: list[BudgetProfile] = []
        for value in self.budget_profiles:
            if isinstance(value, BudgetProfile):
                profiles.append(value)
            elif isinstance(value, dict):
                profiles.append(BudgetProfile(int(value["kd"]), int(value["kv"])))
            else:
                profiles.append(BudgetProfile(int(value[0]), int(value[1])))
        if not profiles or profiles[0].kd != 0:
            raise ValueError("budget_profiles must start with the AR profile")
        object.__setattr__(self, "budget_profiles", tuple(profiles))
        if self.top_m <= 0:
            raise ValueError("top_m must be positive")
        if not 2 <= self.source_ngram_min <= self.source_ngram_max:
            raise ValueError("source n-gram range must satisfy 2 <= min <= max")
        if self.source_chunk_size <= 0 or self.source_top_r <= 0:
            raise ValueError("source chunk size and top-R must be positive")
        if self.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if not 0.0 < float(self.feedback_alpha) <= 1.0:
            raise ValueError("feedback_alpha must satisfy 0 < alpha <= 1")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["budget_profiles"] = [asdict(p) for p in self.budget_profiles]
        return value

    def save(self, path: Path | str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path | str) -> "SyncSpecConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**raw)

    def validate_model_paths(self) -> None:
        """Fail early in offline mode instead of triggering a network fetch."""
        if not self.offline:
            return
        for label, value in (
            ("target_model", self.target_model),
            ("drafter_checkpoint", self.drafter_checkpoint),
            ("selector_checkpoint", self.selector_checkpoint),
            ("survival_checkpoint", self.survival_checkpoint),
        ):
            if not value or Path(value).exists():
                continue
            if "/" in value and not Path(value).is_absolute():
                try:
                    from common.paths import snapshot_dir
                    if snapshot_dir(value) is not None:
                        continue
                except Exception:
                    pass
            raise FileNotFoundError(f"offline {label} not found: {value}")
