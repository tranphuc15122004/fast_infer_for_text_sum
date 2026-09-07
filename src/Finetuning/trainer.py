"""Single-process optimizer lifecycle for the self-contained DFlash port."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from .checkpoint import CheckpointManager, restore_rng_state
from .evaluation import Evaluator
from .schedule import build_scheduler, current_lr, resolve_total_steps, validate_fixed_accumulation_plan
from .strategy import StepContext, StepOutput, TrainBatch


def _as_batch(value: Any) -> TrainBatch:
    if isinstance(value, TrainBatch):
        return value
    if isinstance(value, dict):
        return TrainBatch(tensors=value)
    raise TypeError(f"dataloader item must be a mapping or TrainBatch, got {type(value)!r}")


class Trainer:
    """Train a strategy with fixed gradient accumulation and checkpoints."""

    def __init__(
        self,
        strategy: Any,
        train_dataloader: Iterable[Any],
        output_dir: str | Path,
        run_id: str,
        *,
        validation_dataloader: Iterable[Any] | None = None,
        batch_size: int = 1,
        accumulation_steps: int = 1,
        num_epochs: int = 1,
        max_steps: int | None = None,
        learning_rate: float = 6e-4,
        weight_decay: float = 0.0,
        warmup_steps: int | None = None,
        warmup_ratio: float | None = None,
        scheduler_type: str = "cosine",
        max_grad_norm: float = 1.0,
        save_interval: int = 1,
        log_interval: int = 1,
        eval_interval: int = 0,
        hardware_peak_tflops: float | None = None,
        device: torch.device | str | None = None,
        extra_checkpoint_state: dict[str, Any] | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        if batch_size <= 0 or accumulation_steps <= 0 or num_epochs <= 0:
            raise ValueError("batch_size, accumulation_steps and num_epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self.strategy = strategy
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.max_grad_norm = max_grad_norm
        self.save_interval = max(1, save_interval)
        self.log_interval = max(1, log_interval)
        self.eval_interval = max(0, eval_interval)
        self.hardware_peak_tflops = hardware_peak_tflops
        self.extra_checkpoint_state = extra_checkpoint_state or {}
        self._train_items = self._materialize_if_needed(train_dataloader)
        if not self._train_items:
            raise ValueError("empty training loader")
        self._validation_items = (
            self._materialize_if_needed(validation_dataloader)
            if validation_dataloader is not None
            else None
        )
        num_samples = len(self._train_items) * batch_size
        validate_fixed_accumulation_plan(
            num_samples=num_samples,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            num_epochs=num_epochs,
            max_steps=max_steps,
        )
        self.total_steps = resolve_total_steps(
            total_steps=None,
            max_steps=max_steps,
            num_samples=num_samples,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
            num_epochs=num_epochs,
        )
        module = strategy.trainable_module()
        module.to(self.device)
        self._trainable_parameter_count = sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        )
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in module.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = build_scheduler(
            self.optimizer,
            total_steps=self.total_steps,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            scheduler_type=scheduler_type,
        )
        self.checkpoint_manager = CheckpointManager(self.output_dir, run_id)
        self.evaluator = Evaluator()
        self.global_step = 0
        self.micro_step = 0
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.log_path = self.output_dir / "train.log"
        self._last_eval_step: int | None = None
        if resume_from is not None:
            self._resume(resume_from)

    @staticmethod
    def _materialize_if_needed(value: Iterable[Any] | None) -> list[Any] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return value
        return list(value)

    def _write_record(self, record: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"step={record.get('step', '?')} loss={record.get('loss', '?')} "
                f"lr={record.get('lr', '?')}\n"
            )

    def _save(self) -> Path:
        return self.checkpoint_manager.save(
            self.global_step,
            self.strategy,
            self.optimizer,
            self.scheduler,
            {
                "global_step": self.global_step,
                "micro_step": self.micro_step,
            },
            {
                "strategy": getattr(self.strategy, "name", type(self.strategy).__name__),
                "total_steps": self.total_steps,
                **self.extra_checkpoint_state,
            },
        )

    def _resume(self, path: str | Path) -> None:
        state = self.checkpoint_manager.load(path, map_location=self.device)
        module = self.strategy.trainable_module()
        load_module = module
        if hasattr(self.strategy, "dflash_model"):
            load_module = self.strategy.dflash_model.draft_model
        load_module.load_state_dict(state["draft_state_dict"], strict=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        trainer_state = state.get("trainer_state", {})
        self.global_step = int(trainer_state.get("global_step", 0))
        self.micro_step = int(trainer_state.get("micro_step", 0))
        restore_rng_state(state["rng_state"])

    def save_checkpoint(self) -> Path:
        if self.global_step <= 0:
            raise ValueError("cannot save a checkpoint before the first optimizer step")
        return self._save()

    def evaluate(self) -> dict[str, float]:
        if self._validation_items is None:
            raise ValueError("validation loader is not configured")
        return self.evaluator.evaluate_in_memory(
            self.strategy,
            self._validation_items,
            self.device,
        )

    def fit(self) -> int:
        if self._validation_items is not None and not self._validation_items:
            raise ValueError("empty validation loader")
        module = self.strategy.trainable_module()
        module.train()
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(self.num_epochs):
            for item_index, raw_batch in enumerate(self._train_items):
                if self.global_step >= self.total_steps:
                    break
                started = time.perf_counter()
                output: StepOutput = self.strategy.forward_loss(
                    _as_batch(raw_batch),
                    StepContext(self.global_step, self.total_steps),
                )
                if not torch.isfinite(output.loss.detach()).all():
                    raise ValueError("non-finite loss encountered")
                (output.loss / self.accumulation_steps).backward()
                self.micro_step += 1
                at_boundary = self.micro_step % self.accumulation_steps == 0
                if not at_boundary:
                    continue
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in module.parameters() if parameter.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                elapsed = max(time.perf_counter() - started, 1e-9)
                loss = float(output.loss.detach().cpu())
                tokens = 0
                if isinstance(raw_batch, dict) and "input_ids" in raw_batch:
                    tokens = int(torch.as_tensor(raw_batch["input_ids"]).numel())
                mfu = None
                if self.hardware_peak_tflops is not None:
                    # A transparent 6*N*token estimate; report null unless a
                    # hardware peak was explicitly supplied by the config.
                    mfu = (
                        0.0
                        if not tokens
                        else (6.0 * self._trainable_parameter_count * tokens / elapsed)
                        / (self.hardware_peak_tflops * 1e12)
                    )
                record = {
                    "type": "step",
                    "step": self.global_step,
                    "epoch": epoch,
                    "loss": loss,
                    "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                    "lr": current_lr(self.optimizer),
                    "step_time_s": elapsed,
                    "tokens_per_s": tokens / elapsed if tokens else 0.0,
                    "mfu": mfu,
                }
                for name, value in output.metrics.items():
                    if isinstance(value, torch.Tensor) and value.numel() == 1:
                        record[name] = float(value.detach().cpu())
                    elif isinstance(value, (int, float)):
                        record[name] = value
                self._write_record(record)
                if self.global_step % self.save_interval == 0:
                    self._save()
                if (
                    self._validation_items is not None
                    and self.eval_interval > 0
                    and self.global_step % self.eval_interval == 0
                ):
                    eval_metrics = self.evaluate()
                    self._write_record(
                        {"type": "evaluation", "step": self.global_step, **eval_metrics}
                    )
                    self._last_eval_step = self.global_step
                if self.global_step >= self.total_steps:
                    break
            if self.global_step >= self.total_steps:
                break
        if self.global_step <= 0:
            raise ValueError("training completed without an optimizer step")
        try:
            latest = self.checkpoint_manager.latest_dir()
        except FileNotFoundError:
            latest = None
        if latest is None or not latest.exists():
            self._save()
        return self.global_step


__all__ = ["Trainer"]
