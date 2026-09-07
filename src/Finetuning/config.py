"""Typed configuration for offline Qwen3 DFlash training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


_DTYPES = {"float32", "float16", "bfloat16"}
_BACKENDS = {"eager", "sdpa", "flex_attention"}


@dataclass
class ModelConfig:
    target_model_path: str | None = None
    torch_dtype: str = "float32"
    mask_token_id: int | None = None
    num_draft_layers: int = 2
    draft_intermediate_size: int | None = None
    block_size: int = 16
    target_layer_ids: list[int] | None = None
    layer_types: list[str] | None = None
    trust_remote_code: bool = False


@dataclass
class DataConfig:
    train_data_path: str | None = None
    hidden_states_path: str | None = None
    eval_data_path: str | None = None
    eval_hidden_states_path: str | None = None
    max_length: int = 2048
    chat_template: str = "qwen3"
    feature_dtype: str = "float32"


@dataclass
class TrainingConfig:
    strategy: str = "dflash"
    num_epochs: int = 1
    max_steps: int | None = 1000
    batch_size: int = 1
    accumulation_steps: int = 1
    learning_rate: float = 6e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.04
    scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    num_anchors: int = 512
    loss_decay_gamma: float | None = 7.0
    objective_chunk_blocks: int = 128
    attention_backend: str = "eager"
    loss_type: str = "dflash"
    dpace_alpha: float = 0.5
    save_interval: int = 100
    eval_interval: int = 100
    log_interval: int = 10
    seed: int = 42
    hardware_peak_tflops: float | None = None


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str | Path = "outputs/finetuning-dflash"
    run_id: str = "dflash"
    device: str = "auto"
    offline: bool = True
    synthetic: bool = False

    def validate(self) -> "RunConfig":
        if self.model.torch_dtype not in _DTYPES:
            raise ValueError(f"model.torch_dtype must be one of {sorted(_DTYPES)}")
        if self.data.feature_dtype not in _DTYPES:
            raise ValueError(f"data.feature_dtype must be one of {sorted(_DTYPES)}")
        if self.training.attention_backend not in _BACKENDS:
            raise ValueError(
                "training.attention_backend must be one of "
                f"{sorted(_BACKENDS)}"
            )
        if self.training.strategy != "dflash":
            raise ValueError("only training.strategy='dflash' is supported")
        if self.model.num_draft_layers <= 0 or self.model.block_size < 2:
            raise ValueError("num_draft_layers must be positive and block_size >= 2")
        if self.data.max_length < self.model.block_size:
            raise ValueError("data.max_length must be at least model.block_size")
        if self.training.batch_size <= 0 or self.training.accumulation_steps <= 0:
            raise ValueError("batch_size and accumulation_steps must be positive")
        if self.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.training.max_steps is not None and self.training.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if not 0.0 <= self.training.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if not self.run_id or Path(self.run_id).name != self.run_id:
            raise ValueError("run_id must be a simple directory-safe name")
        if self.synthetic:
            return self
        sources = [self.data.train_data_path, self.data.hidden_states_path]
        if sum(value is not None for value in sources) != 1:
            raise ValueError(
                "exactly one of data.train_data_path or "
                "data.hidden_states_path must be configured"
            )
        if not self.model.target_model_path:
            raise ValueError("model.target_model_path is required for real training")
        eval_sources = [self.data.eval_data_path, self.data.eval_hidden_states_path]
        if sum(value is not None for value in eval_sources) > 1:
            raise ValueError(
                "at most one of data.eval_data_path or "
                "data.eval_hidden_states_path may be configured"
            )
        if self.model.target_layer_ids is not None:
            if not self.model.target_layer_ids or any(
                layer_id < 0 for layer_id in self.model.target_layer_ids
            ):
                raise ValueError("model.target_layer_ids must be non-negative and non-empty")
        return self

    @property
    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload

    @classmethod
    def from_synthetic(
        cls,
        *,
        output_dir: str | Path,
        max_steps: int = 4,
        batch_size: int = 1,
        eval_interval: int = 1,
        attention_backend: str = "eager",
    ) -> "RunConfig":
        config = cls(
            model=ModelConfig(
                torch_dtype="float32",
                mask_token_id=0,
                num_draft_layers=2,
                block_size=4,
            ),
            data=DataConfig(max_length=16, feature_dtype="float32"),
            training=TrainingConfig(
                max_steps=max_steps,
                batch_size=batch_size,
                num_anchors=2,
                learning_rate=0.03,
                warmup_ratio=0.0,
                attention_backend=attention_backend,
                save_interval=1,
                eval_interval=eval_interval,
                log_interval=1,
                seed=42,
            ),
            output_dir=output_dir,
            run_id="synthetic",
            device="cpu",
            synthetic=True,
        )
        return config.validate()

    @classmethod
    def from_local_qwen3(
        cls,
        *,
        target_model_path: str,
        output_dir: str | Path,
        max_steps: int = 1,
        device: str = "auto",
    ) -> "RunConfig":
        config = cls(
            model=ModelConfig(target_model_path=target_model_path, torch_dtype="float32"),
            data=DataConfig(hidden_states_path=str(Path(output_dir) / "features")),
            training=TrainingConfig(max_steps=max_steps),
            output_dir=output_dir,
            device=device,
        )
        return config


def _construct(cls: type[Any], values: Mapping[str, Any], section: str) -> Any:
    allowed = set(cls.__dataclass_fields__)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown {section} config fields: {sorted(unknown)}")
    return cls(**dict(values))


def load_run_config(path: str | Path) -> RunConfig:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"config file not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("run config must be a YAML mapping")
    model = _construct(ModelConfig, payload.get("model", {}) or {}, "model")
    data = _construct(DataConfig, payload.get("data", {}) or {}, "data")
    training = _construct(TrainingConfig, payload.get("training", {}) or {}, "training")
    top_allowed = {"model", "data", "training", "output_dir", "run_id", "device", "offline", "synthetic"}
    unknown = set(payload) - top_allowed
    if unknown:
        raise ValueError(f"unknown run config fields: {sorted(unknown)}")
    config = RunConfig(
        model=model,
        data=data,
        training=training,
        output_dir=payload.get("output_dir", "outputs/finetuning-dflash"),
        run_id=str(payload.get("run_id", "dflash")),
        device=str(payload.get("device", "auto")),
        offline=bool(payload.get("offline", True)),
        synthetic=bool(payload.get("synthetic", False)),
    )
    return config.validate()


def apply_cli_overrides(config: RunConfig, **overrides: Any) -> RunConfig:
    """Apply non-None CLI values without mutating nested dataclass identity."""

    for key, value in overrides.items():
        if value is None:
            continue
        if key in {"output_dir", "run_id", "device"}:
            setattr(config, key, value)
        elif key in {"target_model_path"}:
            config.model.target_model_path = value
        elif key in {"train_data_path", "hidden_states_path"}:
            setattr(config.data, key, value)
        elif key in {"max_steps", "batch_size", "attention_backend"}:
            if key == "attention_backend":
                config.training.attention_backend = value
            else:
                setattr(config.training, key, value)
        else:
            raise ValueError(f"unsupported CLI override: {key}")
    return config.validate()


__all__ = [
    "DataConfig",
    "ModelConfig",
    "RunConfig",
    "TrainingConfig",
    "apply_cli_overrides",
    "load_run_config",
]
