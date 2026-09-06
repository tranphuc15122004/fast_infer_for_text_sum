"""Trainer spine cho DFlash — tương ứng Trainer/Controller/Core của SpecForge.

Quy ước giữ nguyên SpecForge:
- ``global_step``, LR, loss horizon, logging và checkpoint đều tính theo
  *completed optimizer updates* (không trộn micro-batch).
- Gradient accumulation: loss mỗi micro chia cho ``accumulation_steps``;
  optimizer step xảy ra đúng ở biên ``micro % accumulation_steps == 0``.
- Optimizer AdamW (fp32 master), cosine LR + linear warmup, grad clip.
- Chỉ draft model được tối ưu; target lm_head/embed_tokens frozen.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn

from .checkpoint import (
    load_training_checkpoint,
    save_draft_weights,
    save_training_checkpoint,
)
from .config import RunConfig
from .data import DFlashFeatureDataset
from .distributed import (
    all_reduce_sum,
    barrier,
    current_context,
    rank_shard_indices,
)
from .schedule import (
    build_cosine_with_warmup,
    current_lr,
    resolve_total_steps,
    validate_fixed_accumulation_plan,
)
from .training import DFlashTrainStrategy, StepContext, StepOutput, TrainBatch


class Trainer:
    """Huấn luyện OnlineDFlashModel trên dataset feature offline."""

    def __init__(
        self,
        run_cfg: RunConfig,
        strategy: DFlashTrainStrategy,
        dataset: DFlashFeatureDataset,
        *,
        device: Optional[torch.device] = None,
        resume_from: Optional[str] = None,
    ) -> None:
        self.run_cfg = run_cfg
        self.strategy = strategy
        self.dataset = dataset
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dist = current_context(self.device)
        self.base_model: nn.Module = strategy.model
        self.base_model.to(self.device)
        self.model: nn.Module = strategy.trainable_module()

        tcfg = run_cfg.training
        if tcfg.dp_world_size > 1 and tcfg.dp_world_size != self.dist.world_size:
            raise NotImplementedError(
                "training.dp_world_size phải bằng WORLD_SIZE thực tế khi dùng DDP: "
                f"config={tcfg.dp_world_size}, actual={self.dist.world_size}"
            )
        if self.dist.world_size > 1 and tcfg.dp_world_size not in {1, self.dist.world_size}:
            raise ValueError(
                "WORLD_SIZE hiện tại không khớp training.dp_world_size: "
                f"{self.dist.world_size} vs {tcfg.dp_world_size}"
            )
        self.global_batch_size = tcfg.batch_size * self.dist.world_size
        self.distributed_model: Optional[nn.Module] = None
        if self.dist.is_distributed:
            from torch.nn.parallel import DistributedDataParallel

            device_ids = [self.dist.device.index] if self.dist.device.type == "cuda" else None
            self.distributed_model = DistributedDataParallel(
                self.base_model,
                device_ids=device_ids,
                output_device=self.dist.device.index if device_ids else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            self.strategy.set_forward_model(self.distributed_model)
        validate_fixed_accumulation_plan(
            num_samples=len(dataset),
            batch_size=self.global_batch_size,
            accumulation_steps=tcfg.accumulation_steps,
            num_epochs=tcfg.num_epochs,
            max_steps=tcfg.max_steps,
        )
        self.total_steps = resolve_total_steps(
            total_steps=None,
            max_steps=tcfg.max_steps,
            num_samples=len(dataset),
            batch_size=self.global_batch_size,
            accumulation_steps=tcfg.accumulation_steps,
            num_epochs=tcfg.num_epochs,
        )
        self.warmup_steps = int(self.total_steps * tcfg.warmup_ratio)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=tcfg.learning_rate,
            weight_decay=tcfg.weight_decay,
        )
        self.scheduler = build_cosine_with_warmup(
            self.optimizer,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
        )

        self.global_step = 0
        self.micro = 0
        self._train_rng = random.Random(tcfg.seed)
        self.output_dir = run_cfg.resolved_output_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

        if resume_from:
            self._resume(resume_from)

    # ------------------------------------------------------------------ #
    # Resume
    # ------------------------------------------------------------------ #

    def _resume(self, path: str) -> None:
        state = load_training_checkpoint(path)
        draft_state = state["draft_state_dict"]
        if draft_state and all(key.startswith("draft_model.") for key in draft_state):
            draft_state = {
                key[len("draft_model.") :]: value
                for key, value in draft_state.items()
            }
        self.model.load_state_dict(draft_state, strict=False)
        if "optimizer_state" in state:
            self.optimizer.load_state_dict(state["optimizer_state"])
        if "scheduler_state" in state:
            self.scheduler.load_state_dict(state["scheduler_state"])
        self.global_step = int(state.get("global_step", 0))
        # Đảm bảo LR đúng tại step hiện tại (kể cả khi checkpoint không có
        # optimizer/scheduler state — warm-start weights-only).
        for group in self.optimizer.param_groups:
            base = group.get("initial_lr", self.run_cfg.training.learning_rate)
            from .schedule import _warmup_cosine_factor

            group["lr"] = base * _warmup_cosine_factor(
                self.global_step, self.total_steps, self.warmup_steps
            )
        print(
            f"[trainer] resume từ {path}: global_step={self.global_step}, "
            f"lr={current_lr(self.optimizer):.6f}"
        )

    # ------------------------------------------------------------------ #
    # Batch lặp
    # ------------------------------------------------------------------ #

    def _iter_batches(
        self,
        dataset: DFlashFeatureDataset,
        *,
        shuffle: bool,
        repeat: bool = True,
    ) -> Iterable[TrainBatch]:
        """Lặp batch theo rank, bỏ global tail để mọi rank cùng số update."""
        tcfg = self.run_cfg.training
        batch_size = tcfg.batch_size
        n = len(dataset)
        micros_per_epoch = n // self.global_batch_size
        if micros_per_epoch == 0:
            raise ValueError(
                f"dataset ({n}) nhỏ hơn global_batch_size ({self.global_batch_size}); "
                "không có micro-batch nào"
            )
        while True:
            indices = list(range(n))
            if shuffle:
                self._train_rng.shuffle(indices)
            usable = micros_per_epoch * self.global_batch_size
            rank_indices = indices[:usable][self.dist.rank:usable:self.dist.world_size]
            batches = [
                rank_indices[start : start + batch_size]
                for start in range(0, len(rank_indices), batch_size)
            ]
            from torch.utils.data import DataLoader

            data_cfg = self.run_cfg.data
            loader_kwargs = {
                "batch_sampler": batches,
                "collate_fn": dataset.collate,
                "num_workers": data_cfg.num_workers,
                "pin_memory": bool(data_cfg.pin_memory and self.device.type == "cuda"),
            }
            if data_cfg.num_workers > 0:
                loader_kwargs["prefetch_factor"] = data_cfg.prefetch_factor
                loader_kwargs["persistent_workers"] = data_cfg.persistent_workers
            loader = DataLoader(dataset, **loader_kwargs)
            for tensors in loader:
                yield TrainBatch(tensors=tensors)
            if not repeat:
                break

    # ------------------------------------------------------------------ #
    # Vòng huấn luyện
    # ------------------------------------------------------------------ #

    def fit(
        self,
        *,
        eval_dataset: Optional[DFlashFeatureDataset] = None,
    ) -> Dict[str, object]:
        """Chạy toàn bộ lịch train; có thể evaluate split offline sau/định kỳ."""
        tcfg = self.run_cfg.training
        acc_steps = tcfg.accumulation_steps
        log_every = max(1, tcfg.log_interval)
        save_every = tcfg.save_interval

        self.base_model.to(self.device)
        self.base_model.train()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        window_loss = 0.0
        window_acc_num = 0.0
        window_acc_den = 0.0
        window_micros = 0
        window_tokens = 0
        start_wall = time.time()

        ctx = StepContext(global_step=self.global_step, total_steps=self.total_steps)
        batches = self._iter_batches(self.dataset, shuffle=True)
        summary: Dict[str, object] = {}

        for batch in batches:
            if self.global_step >= self.total_steps:
                break
            out: StepOutput = self.strategy.forward_loss(batch, ctx)
            scalar = out.loss
            acc_num, acc_den = out.ratio_metrics.get(
                "acc", (torch.zeros(()), torch.zeros(()))
            )
            loss_for_backward = scalar / acc_steps
            loss_for_backward.backward()
            self.micro += 1
            window_loss += float(scalar.detach())
            window_acc_num += float(torch.as_tensor(acc_num).float().sum())
            window_acc_den += float(torch.as_tensor(acc_den).float().sum())
            window_micros += 1
            window_tokens += int(batch.tensors["input_ids"].numel())

            stepped = self.micro % acc_steps == 0
            if stepped:
                # Thứ tự chuẩn: optimizer.step() rồi scheduler.step() (torch
                # cảnh báo nếu ngược). Logged lr là giá trị cho step kế tiếp.
                if tcfg.max_grad_norm is not None and tcfg.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), tcfg.max_grad_norm
                    )
                else:
                    grad_norm = torch.tensor(float("nan"))
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                ctx = StepContext(
                    global_step=self.global_step, total_steps=self.total_steps
                )

                # DDP gradient đã được average, nên log cũng cần aggregate
                # để rank 0 phản ánh global batch thay vì local batch.
                metric_totals = torch.tensor(
                    [window_loss, window_acc_num, window_acc_den, window_tokens],
                    device=self.device,
                    dtype=torch.float64,
                )
                all_reduce_sum(metric_totals)
                global_micros = max(1, window_micros * self.dist.world_size)
                global_acc_den = float(metric_totals[2].item())
                metrics = {
                    "global_step": self.global_step,
                    "loss": float(metric_totals[0].item()) / global_micros,
                    "acc": (
                        float(metric_totals[1].item()) / global_acc_den
                        if global_acc_den > 0
                        else float("nan")
                    ),
                    "lr": current_lr(self.optimizer),
                    "grad_norm": float(grad_norm.detach()),
                    "tokens_per_step": int(round(metric_totals[3].item())),
                    "elapsed_s": round(time.time() - start_wall, 2),
                }
                self._log(metrics)
                window_loss = 0.0
                window_acc_num = 0.0
                window_acc_den = 0.0
                window_micros = 0
                window_tokens = 0

                if save_every and self.global_step % save_every == 0:
                    self._save_checkpoint(f"step_{self.global_step}")
                if (
                    eval_dataset is not None
                    and tcfg.eval_interval > 0
                    and self.global_step % tcfg.eval_interval == 0
                ):
                    self.evaluate(eval_dataset)
                if self.global_step >= self.total_steps:
                    break

        # Dừng với accumulation lẻ → cảnh báo (bình thường khi max_steps cắt sớm).
        if self.micro % acc_steps != 0:
            print(
                f"[trainer] kết thúc với {self.micro % acc_steps} micro chưa "
                "đủ accumulation (gradient bị bỏ)."
            )

        self._save_checkpoint("final")
        summary["global_step"] = self.global_step
        summary["world_size"] = self.dist.world_size
        summary["elapsed_s"] = round(time.time() - start_wall, 2)
        summary["output_dir"] = str(self.output_dir)
        if eval_dataset is not None:
            summary["eval"] = self.evaluate(eval_dataset)
        print(
            f"[trainer] xong: global_step={self.global_step}, "
            f"world_size={self.dist.world_size}, elapsed={summary['elapsed_s']}s"
        )
        return summary

    @torch.no_grad()
    def evaluate(
        self,
        dataset: DFlashFeatureDataset,
        *,
        max_batches: Optional[int] = None,
    ) -> Dict[str, object]:
        """Đánh giá DFlash loss/accuracy trên feature validation set."""
        self.base_model.eval()
        self.model.eval()
        loss_num = torch.zeros((), device=self.device, dtype=torch.float64)
        loss_den = torch.zeros((), device=self.device, dtype=torch.float64)
        acc_num = torch.zeros((), device=self.device, dtype=torch.float64)
        acc_den = torch.zeros((), device=self.device, dtype=torch.float64)
        batches = 0
        for batch in self._iter_batches(dataset, shuffle=False, repeat=False):
            out: StepOutput = self.strategy.forward_loss(
                batch,
                StepContext(global_step=self.global_step, total_steps=self.total_steps),
            )
            if out.loss_terms is not None:
                num, den = out.loss_terms
                loss_num += num.detach().to(dtype=torch.float64)
                loss_den += den.detach().to(dtype=torch.float64)
            else:
                loss_num += out.loss.detach().to(dtype=torch.float64)
                loss_den += 1.0
            acc = out.ratio_metrics.get("acc")
            if acc is not None:
                acc_num += torch.as_tensor(acc[0], device=self.device).sum().double()
                acc_den += torch.as_tensor(acc[1], device=self.device).sum().double()
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

        all_reduce_sum(loss_num)
        all_reduce_sum(loss_den)
        all_reduce_sum(acc_num)
        all_reduce_sum(acc_den)
        result: Dict[str, object] = {
            "eval_loss": float((loss_num / loss_den.clamp_min(1e-12)).cpu()),
            "eval_acc": float((acc_num / acc_den.clamp_min(1e-12)).cpu()),
            "eval_batches": batches,
            "global_step": self.global_step,
        }
        if self.dist.is_main:
            eval_path = self.output_dir / "eval_metrics.json"
            eval_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"[eval] step={self.global_step} loss={result['eval_loss']:.4f} "
                f"acc={result['eval_acc']:.4f}"
            )
        self.base_model.train()
        self.model.train()
        return result

    # ------------------------------------------------------------------ #
    # Checkpoint + logging
    # ------------------------------------------------------------------ #

    def _draft_state_dict(self) -> Dict[str, torch.Tensor]:
        """Persist toàn bộ state của draft model (strategy filter)."""
        full = self.model.state_dict()
        return {f"draft_model.{k}": v for k, v in full.items()}

    def _save_checkpoint(self, tag: str) -> str:
        path = str(self.output_dir / f"checkpoint_{tag}.pt")
        if not self.dist.is_main:
            barrier()
            return path
        save_training_checkpoint(
            path,
            draft_state_dict=self._draft_state_dict(),
            global_step=self.global_step,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            run_id=self.run_cfg.run_id,
            config_yaml=self.run_cfg.dump_yaml(),
        )
        # Bản weights-only tiện cho warm start/export.
        save_draft_weights(
            str(self.output_dir / f"draft_{tag}.pt"),
            self._draft_state_dict(),
        )
        print(f"[trainer] đã lưu checkpoint tại {path} (step {self.global_step})")
        barrier()
        return path

    def _log(self, metrics: Dict[str, object]) -> None:
        if not self.dist.is_main:
            return
        line = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
        with open(self.metrics_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.global_step % self.run_cfg.training.log_interval == 0:
            print(
                f"[train] step={metrics['global_step']} loss={metrics['loss']:.4f} "
                f"acc={metrics['acc']:.4f} lr={metrics['lr']:.2e} "
                f"gn={metrics['grad_norm']:.3f}"
            )


__all__ = ["Trainer"]
