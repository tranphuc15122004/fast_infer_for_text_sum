"""Lazy adapter around SpecForge's offline SGLang DFlash capture.

The cache worker owns batching and persistence.  This module only bridges the
SpecForge capture boundary and converts its flattened hidden-state output into
one tensor per real (unpadded) sample.  SGLang is imported only when the
``specforge_sglang`` backend is selected so CPU tests and the HF fallback stay
dependency-safe.
"""

from __future__ import annotations

from array import array
import os
from pathlib import Path
import sys
from typing import Any, List, Optional, Type

import torch


def _specforge_root() -> Path:
    return Path(__file__).resolve().parents[2] / "externals" / "SpecForge"


def _ensure_specforge_importable() -> None:
    # SpecForge itself is vendored separately from the SGLang fork it was
    # pinned and tested with. Put both on the worker path so a server that
    # does not install them as editable packages still uses the compatible
    # repository implementation.
    roots = (
        _specforge_root().parent / "SSSD" / "python",
        _specforge_root(),
    )
    for root_path in roots:
        root = str(root_path)
        if root_path.is_dir() and root not in sys.path:
            sys.path.insert(0, root)


def _dtype(name: str) -> torch.dtype:
    try:
        return {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported torch dtype for SpecForge capture: {name!r}") from exc


def _ensure_single_process_distributed() -> bool:
    """Initialize SpecForge's process groups for one independent GPU worker."""

    import torch.distributed as dist

    if dist.is_initialized():
        return False
    _ensure_specforge_importable()
    from specforge.distributed import init_distributed

    rank = int(os.environ.get("MR_DFLASH_WORKER_RANK", "0"))
    base_port = int(os.environ.get("MR_DFLASH_DIST_PORT", "29600"))
    # Worker này là process-group một GPU độc lập. Không thừa hưởng RANK/
    # WORLD_SIZE của một launcher torchrun bên ngoài, nếu không các worker có
    # thể chờ nhau vô hạn hoặc join nhầm rendezvous.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(base_port + rank)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    init_distributed(timeout=60, tp_size=1)
    return True


class SpecForgeTargetCapture:
    """SpecForge SGLang target with MR-DFlash's per-sample capture surface."""

    def __init__(
        self,
        target: Any,
        *,
        layer_ids: List[int],
        request_cls: Optional[Type[Any]] = None,
        sampling_params_cls: Optional[Type[Any]] = None,
        owns_distributed: bool = False,
    ) -> None:
        self._target = target
        self._backend = getattr(target, "_backend", target)
        self.layer_ids = [int(value) for value in layer_ids]
        self.request_cls = request_cls
        self.sampling_params_cls = sampling_params_cls
        self._owns_distributed = bool(owns_distributed)
        runner = getattr(self._backend, "model_runner", None)
        self.model = getattr(runner, "model", None)
        config = getattr(self.model, "config", None)
        self.hidden_size = int(getattr(config, "hidden_size", 0) or 0)
        self.context_feature_dim = (
            len(self.layer_ids) * self.hidden_size if self.hidden_size else None
        )
        self.device = torch.device(
            "cuda", torch.cuda.current_device()
        ) if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = getattr(target, "tokenizer", None)

    @classmethod
    def from_pretrained(
        cls,
        target_model_path: str,
        layer_ids: List[int],
        *,
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = False,
        attention_backend: str = "flashinfer",
        mem_fraction_static: float = 0.99,
        context_length: Optional[int] = None,
        max_running_requests: int = 64,
        max_total_tokens: int = 262144,
        cache_dir: str = "./cache",
        local_files_only: Optional[bool] = None,
        target_revision: Optional[str] = None,
        **kwargs: Any,
    ) -> "SpecForgeTargetCapture":
        try:
            owns_distributed = _ensure_single_process_distributed()
            _ensure_specforge_importable()
            from specforge.offline_capture import load_offline_capture
            from sglang.srt.managers.schedule_batch import Req
            from sglang.srt.sampling.sampling_params import SamplingParams
        except Exception as exc:
            raise RuntimeError(
                "SpecForge SGLang cache backend requires the local SpecForge "
                "package and compatible SGLang installation"
            ) from exc

        load_kwargs: dict[str, Any] = {
            "attention_backend": attention_backend,
            "mem_fraction_static": float(mem_fraction_static),
            "max_running_requests": int(max_running_requests),
            "max_total_tokens": int(max_total_tokens),
            "download_dir": str(cache_dir),
        }
        if context_length is not None:
            load_kwargs["context_length"] = int(context_length)
        if target_revision is not None:
            load_kwargs["revision"] = str(target_revision)
        # SGLang ServerArgs does not expose local_files_only. Its HF/model
        # loader paths honor HF_HUB_OFFLINE, so set it before initialization
        # when the pipeline requested offline-only loading.
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        load_kwargs.update(kwargs)
        try:
            target = load_offline_capture(
                target_model_path,
                torch_dtype=_dtype(torch_dtype),
                trust_remote_code=trust_remote_code,
                **load_kwargs,
            )
            target.set_capture_layers(
                [int(value) for value in layer_ids],
                capture_method="dflash",
            )
        except Exception:
            if owns_distributed:
                cls._destroy_distributed()
            raise
        return cls(
            target,
            layer_ids=layer_ids,
            request_cls=Req,
            sampling_params_cls=SamplingParams,
            owns_distributed=owns_distributed,
        )

    @staticmethod
    def _destroy_distributed() -> None:
        try:
            _ensure_specforge_importable()
            from specforge.distributed import destroy_distributed

            destroy_distributed()
        except Exception:
            # Process teardown must not hide the capture error that caused it.
            pass

    def _make_request(self, request_id: int, input_ids: List[int]) -> Any:
        if self.request_cls is None or self.sampling_params_cls is None:
            raise RuntimeError("SpecForge request classes are not initialized")
        sampling_params = self.sampling_params_cls(
            temperature=0,
            max_new_tokens=1,
            top_k=1,
        )
        request = self.request_cls(
            rid=str(request_id),
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
        )
        request.full_untruncated_fill_ids = array("q", input_ids)
        request.fill_len = len(input_ids)
        prefix_indices = getattr(request, "prefix_indices", ())
        request.extend_input_len = request.fill_len - len(prefix_indices)
        request.logprob_start_len = len(input_ids) - 1
        return request

    @staticmethod
    def _split_hidden_states(hidden_states: Any, lengths: List[int]) -> List[torch.Tensor]:
        if isinstance(hidden_states, (list, tuple)):
            rows = [torch.as_tensor(value) for value in hidden_states]
            if len(rows) != len(lengths):
                raise RuntimeError(
                    f"SpecForge trả {len(rows)} hidden rows cho {len(lengths)} sample"
                )
            return [row[:length].detach() for row, length in zip(rows, lengths)]
        tensor = torch.as_tensor(hidden_states)
        if tensor.dim() == 2:
            rows = list(torch.split(tensor, lengths, dim=0))
        elif tensor.dim() == 3 and tensor.shape[0] == len(lengths):
            rows = [tensor[index, :length] for index, length in enumerate(lengths)]
        else:
            raise RuntimeError(
                "SpecForge hidden_states phải có dạng [tokens,width], "
                f"[batch,tokens,width] hoặc list; got {tuple(tensor.shape)}"
            )
        return [row.detach() for row in rows]

    def capture_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Capture real, unpadded rows in one SGLang extend forward."""

        if input_ids.dim() != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids và attention_mask phải cùng shape [batch,seq]")
        if loss_mask is not None and loss_mask.shape != input_ids.shape:
            raise ValueError("loss_mask phải cùng shape input_ids")
        lengths = [int(value) for value in attention_mask.sum(dim=-1).cpu().tolist()]
        if any(length < 1 or length > input_ids.shape[1] for length in lengths):
            raise ValueError(f"attention_mask có độ dài không hợp lệ: {lengths}")
        requests = [
            self._make_request(index, input_ids[index, :length].detach().cpu().tolist())
            for index, length in enumerate(lengths)
        ]
        try:
            output = self._backend._forward_extend(requests)
            hidden_states = getattr(output, "aux_hidden_states", None)
            if hidden_states is None:
                raise RuntimeError("SpecForge không trả aux_hidden_states cho DFlash capture")
            rows = self._split_hidden_states(hidden_states, lengths)
            if any(row.dim() != 2 or row.shape[0] != length for row, length in zip(rows, lengths)):
                raise RuntimeError("SpecForge hidden row length không khớp attention_mask")
            return rows
        finally:
            clear_pools = getattr(self._backend, "_clear_pools", None)
            if callable(clear_pools):
                clear_pools()

    def close(self) -> None:
        close = getattr(self._target, "close", None)
        if callable(close):
            close()
        self._target = None
        self._backend = None
        if self._owns_distributed:
            self._destroy_distributed()


__all__ = ["SpecForgeTargetCapture"]
