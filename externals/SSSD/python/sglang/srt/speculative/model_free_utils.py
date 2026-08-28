from __future__ import annotations

import atexit
import bisect
import hashlib
import json
import logging
import os
import time
from collections import deque
from contextlib import contextmanager
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
import torch
from filelock import FileLock
from transformers import AutoTokenizer

from sglang.srt.distributed import (
    GroupCoordinator,
    broadcast_tensor_dict,
    patch_tensor_parallel_group,
)
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)

POSSIBLE_SPEC_LENS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]

_COMPUTE_BW_RATIO = None

COMMUNICATION_METHOD = "base"  # "base", "broadcast_small_mask"


@contextmanager
def draft_tp_context(tp_group: GroupCoordinator):
    # Draft model doesn't use dp and has its own tp group.
    # We disable mscclpp now because it doesn't support 2 comm groups.
    with patch_tensor_parallel_group(tp_group):
        yield


class ModelFreeSpeculator:
    "Python interface between SGLang and the C++ speculator code."

    def __init__(
        self,
        server_args: ServerArgs,
        device: str,
        captured_batch_sizes: List[int],
        num_tokens_per_bs_map: dict,
        tp_rank: int,
        tp_group: GroupCoordinator,
    ):

        # For tp
        self.tp_size = server_args.tp_size
        self.tp_rank = tp_rank
        self.group = tp_group
        self.owner = 0

        self.request_converter = RequestIdToIntegerConverter()
        self.device = torch.device(device)
        # If the speculation is not adaptive, these values are fixed, otherwise they are updated at each iteration
        self.num_draft_tokens = server_args.speculative_num_draft_tokens
        self.num_steps = server_args.speculative_num_steps
        self.topk = server_args.speculative_eagle_topk

        # If speculative_adaptive is true the previous 3 values coming from server_args are not used
        self.speculative_adaptive = server_args.speculative_adaptive
        self.captured_batch_sizes = captured_batch_sizes
        self.num_tokens_per_bs_map = num_tokens_per_bs_map
        self.last_pad_bs = None

        self.tokenizer = AutoTokenizer.from_pretrained(server_args.tokenizer_path)
        pad_tok = self.tokenizer.pad_token_id
        self.pad_token_id = (
            pad_tok if pad_tok is not None else self.tokenizer.eos_token_id
        )

        self.max_query_length = 4  # Default: override in each method

        # Do the dispatching of method only once at creation
        self._get_draft_method = self.get_candidates_mask

        if COMMUNICATION_METHOD == "base":
            self.get_draft = self._get_draft_base
        elif COMMUNICATION_METHOD == "broadcast_small_mask":
            self.get_draft = self._get_draft_broadcast_small_mask
        else:
            raise NotImplementedError(
                "Drafting method should be 'base', 'broadcast_small_mask'."
            )

    # Implementation based on different methods

    def add_new_sequence(self, req: Req) -> None:
        pass

    def stream_put(self, new_tokens: List[int], req: Req) -> None:
        pass

    def get_candidates_mask(
        self, batch: ScheduleBatch
    ) -> tuple[list[int], list[int], list[int], list[int], list[torch.BoolTensor]]:
        pass

    def clear_seq_cache(self, req_id: str) -> None:
        pass

    ### Shared code across speculators ###

    def update_speculate_params_adaptive(self, bs: int):
        index = bisect.bisect_left(self.captured_batch_sizes, bs)
        if index >= len(self.captured_batch_sizes):
            # Beyond captured graphs, should not happen. Don't speculate
            self.num_draft_tokens, self.num_steps, self.topk = 1, 0, 0
            return self.num_draft_tokens, self.num_steps, self.topk

        pad_bs = self.captured_batch_sizes[index]
        if self.last_pad_bs == pad_bs:
            # Not updated since last iteration
            return self.num_draft_tokens, self.num_steps, self.topk

        self.last_pad_bs = bs
        self.num_draft_tokens = self.num_tokens_per_bs_map[pad_bs]
        self.num_steps, self.topk = default_branch_func(self.num_draft_tokens)

        return self.num_draft_tokens, self.num_steps, self.topk

    """Single uncompressed broadcast solution"""

    def _get_draft_base(self, batch: ScheduleBatch):
        # TODO (mmarzollo): For now assume all speculate_lens are the same (SGLang doesn't support variable spec_len out of the box)
        candidates, position_ids, next_child_idx, next_sibling_idx, decoding_masks = (
            self._get_draft_method(batch)
        )

        device = self.device
        seq_lens = batch.seq_lens
        seq_lens_cpu = seq_lens.to("cpu", copy=False)
        spec_len = self.num_draft_tokens

        # If some request doesn't have all the candidates requested, pad it
        to_pad = [spec_len - l for l in [len(c) for c in candidates]]
        if sum(to_pad) > 0:
            self._pad_speculate_outputs(
                candidates, position_ids, next_child_idx, next_sibling_idx, to_pad
            )
            full_tree_mask = merge_masks_to_flat_with_padding(
                decoding_masks,
                seq_lens_cpu,
                to_pad,
                spec_len,
                device,
            )
        else:
            full_tree_mask = merge_masks_to_flat(decoding_masks, seq_lens_cpu, device)

        bs = batch.batch_size()
        flattened_positions = torch.tensor(
            list(chain.from_iterable(position_ids)), device=device, dtype=torch.long
        )
        spec_lens = torch.full((bs,), spec_len, device=device, dtype=torch.long)
        expanded_offsets = seq_lens.repeat_interleave(spec_lens)
        draft_token_positions = flattened_positions + expanded_offsets

        retrive_index = torch.arange(bs * spec_len, device=device).view(bs, spec_len)
        retrive_next_token = torch.tensor(
            next_child_idx, device=device, dtype=torch.long
        )
        retrive_next_sibling = torch.tensor(
            next_sibling_idx, device=device, dtype=torch.long
        )
        draft_tokens = torch.tensor(
            list(chain.from_iterable(candidates)), device=device
        )

        if self.tp_size > 1:
            # You are rank 0, otherwise you shouldn't even exist
            assert self.tp_rank == 0

            tensors = {
                "tree_mask": full_tree_mask,
                "position": draft_token_positions,
                "retrive_index": retrive_index,
                "retrive_next_token": retrive_next_token,
                "retrive_next_sibling": retrive_next_sibling,
                "draft_tokens": draft_tokens,
            }

            with draft_tp_context(self.group):
                broadcast_tensor_dict(tensors, src=self.owner)

        return (
            full_tree_mask,
            draft_token_positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        )

    """Only broadcast the small submasks, then let each worker reconstruct the full mask on its device"""

    def _get_draft_broadcast_small_mask(self, batch: ScheduleBatch):
        candidates, position_ids, next_child_idx, next_sibling_idx, decoding_masks = (
            self._get_draft_method(batch)
        )

        device = self.device
        seq_lens = batch.seq_lens
        spec_len = self.num_draft_tokens

        # If some request doesn't have all the candidates requested, pad it
        to_pad = [spec_len - l for l in [len(c) for c in candidates]]
        need_padding = sum(to_pad) > 0
        if need_padding:
            self._pad_speculate_outputs(
                candidates, position_ids, next_child_idx, next_sibling_idx, to_pad
            )

        bs = batch.batch_size()
        flattened_positions = torch.tensor(
            list(chain.from_iterable(position_ids)), device=device, dtype=torch.long
        )
        spec_lens = torch.full((bs,), spec_len, device=device, dtype=torch.long)
        expanded_offsets = seq_lens.repeat_interleave(spec_lens)
        draft_token_positions = flattened_positions + expanded_offsets

        retrive_index = torch.arange(bs * spec_len, device=device).view(bs, spec_len)
        retrive_next_token = torch.tensor(
            next_child_idx, device=device, dtype=torch.long
        )
        retrive_next_sibling = torch.tensor(
            next_sibling_idx, device=device, dtype=torch.long
        )
        draft_tokens = torch.tensor(
            list(chain.from_iterable(candidates)), device=device
        )

        if self.tp_size > 1:
            packed_masks = stack_masks_on_device(decoding_masks, spec_len, device)

            # You are rank 0, otherwise you shouldn't even exist
            assert self.tp_rank == 0

            tensors = {
                "packed_masks": packed_masks,  # stay on device
                "position": draft_token_positions,
                "retrive_index": retrive_index,
                "retrive_next_token": retrive_next_token,
                "retrive_next_sibling": retrive_next_sibling,
                "draft_tokens": draft_tokens,
                "seq_lens": seq_lens.to(device, non_blocking=True),
            }

            with draft_tp_context(self.group):
                broadcast_tensor_dict(tensors, src=self.owner)

            full_tree_mask = merge_packed_masks_flat_device(
                tensors["packed_masks"], tensors["seq_lens"], device
            )

        else:  # No broadcasting
            seq_lens_cpu = seq_lens.to("cpu", copy=False)
            if need_padding:
                full_tree_mask = merge_masks_to_flat_with_padding(
                    decoding_masks,
                    seq_lens_cpu,
                    to_pad,
                    spec_len,
                    device,
                )
            else:
                full_tree_mask = merge_masks_to_flat(
                    decoding_masks, seq_lens_cpu, device
                )

        return (
            full_tree_mask,
            draft_token_positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        )

    def _pad_speculate_outputs(
        self,
        candidates: List[List[int]],
        position_ids: List[List[int]],
        next_child_idx: List[List[int]],
        next_sibling_idx: List[List[int]],
        to_pad: List[int],
    ) -> None:
        pad_id = self.pad_token_id
        neg1 = -1

        for cands, pos, child, sib, pad in zip(
            candidates, position_ids, next_child_idx, next_sibling_idx, to_pad
        ):
            if pad:
                cands.extend([pad_id] * pad)
                pos.extend([neg1] * pad)
                child.extend([neg1] * pad)
                sib.extend([neg1] * pad)


class ProxySpeculator:
    def __init__(
        self,
        server_args: ServerArgs,
        device: str,
        captured_batch_sizes: List[int],
        num_tokens_per_bs_map: dict,
        tp_rank: int,
        tp_group: GroupCoordinator,
        owner_rank=0,
    ):
        self.device = torch.device(device)
        self.group = tp_group
        self.owner = owner_rank
        self.rank = tp_rank

        # If the speculation is not adaptive, these values are fixed, otherwise they are updated at each iteration
        self.num_draft_tokens = server_args.speculative_num_draft_tokens
        self.num_steps = server_args.speculative_num_steps
        self.topk = server_args.speculative_eagle_topk

        # If speculative_adaptive is true the previous 3 values coming from server_args are not used
        self.speculative_adaptive = server_args.speculative_adaptive
        self.captured_batch_sizes = captured_batch_sizes
        self.num_tokens_per_bs_map = num_tokens_per_bs_map
        self.last_pad_bs = None

        if COMMUNICATION_METHOD == "base":
            self.get_draft = self._get_draft_base
        elif COMMUNICATION_METHOD == "broadcast_small_mask":
            self.get_draft = self._get_draft_broadcast_small_mask
        else:
            raise NotImplementedError(
                "Drafting method should be 'base', 'broadcast_small_mask'."
            )

    # No-ops on non-owner for stateful mutators
    def add_new_sequence(self, req):
        pass

    def stream_put(self, tokens, req):
        pass

    def clear_seq_cache(self, req_id):
        pass

    def update_speculate_params_adaptive(self, bs: int):
        index = bisect.bisect_left(self.captured_batch_sizes, bs)
        if index >= len(self.captured_batch_sizes):
            return 1, 0, 0  # Beyond captured graphs, should not happen. Don't speculate
        pad_bs = self.captured_batch_sizes[index]
        if self.last_pad_bs == pad_bs:
            return self.num_draft_tokens, self.num_steps, self.topk
        self.last_pad_bs = bs
        self.num_draft_tokens = self.num_tokens_per_bs_map[pad_bs]
        self.num_steps, self.topk = default_branch_func(self.num_draft_tokens)

        return self.num_draft_tokens, self.num_steps, self.topk

    """Single uncompressed broadcast solution"""

    def _get_draft_base(self, batch: ScheduleBatch):
        with draft_tp_context(self.group):
            bufs = broadcast_tensor_dict(None, src=self.owner)

        # Unpack locals for return
        tree_mask = bufs["tree_mask"]
        position = bufs["position"]
        retrive_index = bufs["retrive_index"]
        retrive_next_token = bufs["retrive_next_token"]
        retrive_next_sibling = bufs["retrive_next_sibling"]
        draft_tokens = bufs["draft_tokens"]

        return (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        )

    def _get_draft_broadcast_small_mask(self, batch: ScheduleBatch):
        with draft_tp_context(self.group):
            bufs = broadcast_tensor_dict(None, src=self.owner)

        # Unpack locals for return
        position = bufs["position"]
        retrive_index = bufs["retrive_index"]
        retrive_next_token = bufs["retrive_next_token"]
        retrive_next_sibling = bufs["retrive_next_sibling"]
        draft_tokens = bufs["draft_tokens"]

        full_tree_mask = merge_packed_masks_flat_device(
            bufs["packed_masks"], bufs["seq_lens"], self.device
        )

        return (
            full_tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        )


### UTILS ###

CACHE_FILE = "/tmp/compute_bw_ratio.json"
_COMPUTE_BW_RATIO = None


def _get_compute_bw_device_type() -> Optional[str]:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "npu"
    return None


def _get_compute_bw_ratio_cache_file(device_type: str) -> str:
    if device_type == "cuda":
        return CACHE_FILE

    root, ext = os.path.splitext(CACHE_FILE)
    return f"{root}_{device_type}{ext}"


def _get_compute_bw_ratio_lock_file(device_type: str) -> str:
    return _get_compute_bw_ratio_cache_file(device_type) + ".lock"


def get_compute_bw_ratio():
    global _COMPUTE_BW_RATIO
    if _COMPUTE_BW_RATIO is None:
        device_type = _get_compute_bw_device_type()
        if device_type is None:
            _COMPUTE_BW_RATIO = float(
                os.getenv("SGLANG_SPECULATIVE_COMPUTE_BW_RATIO", "64")
            )
            logger.warning(
                "CUDA/NPU benchmark is unavailable on this device; using "
                f"SGLANG_SPECULATIVE_COMPUTE_BW_RATIO={_COMPUTE_BW_RATIO}."
            )
            return _COMPUTE_BW_RATIO

        cache_file = _get_compute_bw_ratio_cache_file(device_type)
        lock = FileLock(_get_compute_bw_ratio_lock_file(device_type))
        with lock:
            if os.path.exists(cache_file):
                with open(cache_file, "r") as f:
                    _COMPUTE_BW_RATIO = json.load(f)["ratio"]
            else:
                _COMPUTE_BW_RATIO = flops_to_bandwidth_ratio(device_type=device_type)
                with open(cache_file, "w") as f:
                    json.dump({"ratio": _COMPUTE_BW_RATIO}, f)
    logger.info(f"COMPUTE BW RATIO: {_COMPUTE_BW_RATIO}")
    return _COMPUTE_BW_RATIO


@atexit.register
def cleanup_cache():
    for device_type in ("cuda", "npu"):
        cache_file = _get_compute_bw_ratio_cache_file(device_type)
        if os.path.exists(cache_file):
            os.remove(cache_file)


def get_max_speclen_per_bs(bs):
    """The roofline model does not apply well to small batch sizes, when the free budget is a lot and speculating
    too much is not worth it. So we put some additional manual constraints obtained by experience.
    """
    if bs <= 1:
        return 32
    if bs >= 2 and bs < 4:
        return 64 // bs
    if bs >= 4 and bs <= 6:
        return 96 // bs
    # From here you can start trusting the roofline estimate
    return 512 // bs


def get_bs_speclen(capture_bs: List[int], spec_algorithm: SpeculativeAlgorithm):
    num_tokens_per_bs_map = {}

    compute_bw_ratio = int(get_compute_bw_ratio())
    # Make it a nice multiple of 16 for operator efficiency
    possible_multiples_of_16 = [16 * val for val in [1, 2, 3, 4, 6, 8, 10, 12, 16, 24]]
    index = bisect.bisect_right(possible_multiples_of_16, compute_bw_ratio)
    compute_bw_ratio = possible_multiples_of_16[index - 1] if index > 0 else 1
    if spec_algorithm.is_pld():
        # For prompt lookup decoding, don't speculate more than 10 tokens
        i = bisect.bisect_right(POSSIBLE_SPEC_LENS, 8)
        possible_spec_lens = POSSIBLE_SPEC_LENS[:i]
    else:
        possible_spec_lens = POSSIBLE_SPEC_LENS

    for bs in capture_bs:
        max_spec_len = min(compute_bw_ratio // bs, get_max_speclen_per_bs(bs))
        index = bisect.bisect_right(possible_spec_lens, max_spec_len)
        num_toks = possible_spec_lens[index - 1] if index > 0 else 1
        num_tokens_per_bs_map[bs] = num_toks

    return num_tokens_per_bs_map


class RequestIdToIntegerConverter:
    """
    Maps arbitrary strings to free 32-bit signed integers (0 … 2_147_483_647).

    • Same string ⇒ same number until `release()` is called.
    • Linear probing finds a gap if the first hash slot is taken.
    • Raises RuntimeError only when every int32 value is exhausted.
    • No thread-safety (single-thread use assumed).

    This is needed to map request ids (unsigned int128) to the speculator request ids (int32).
    """

    _MAX_INT32 = 2**31 - 1

    def __init__(self) -> None:
        self._str2int: dict[str, int] = {}
        self._used: set[int] = set()

    def acquire(self, key: str) -> Tuple[int, bool]:
        """
        Return an available int32 for `key`, remembering the choice.
        """
        if key in self._str2int:  # already allocated
            return self._str2int[key], True

        start = self._hash32(key)
        n = start
        while n in self._used:
            n = (n + 1) & self._MAX_INT32
            if n == start:
                raise RuntimeError("All 32-bit ints are in use")

        self._str2int[key] = n
        self._used.add(n)
        return n, False

    def release(self, ref) -> None:
        """
        Free the int32 previously assigned to `key` (no-op if unknown), or directly the int
        """
        val = self._str2int.pop(ref, None)
        if val is not None:
            self._used.discard(val)

    def __contains__(self, key: str) -> bool:
        return key in self._str2int

    def __len__(self) -> int:
        return len(self._used)

    @staticmethod
    def _hash32(s: str) -> int:
        """
        Deterministic 32-bit hash of the string `s`.
        (MD5 → take the first 4 bytes, big-endian, mask to 31 bits.)
        """
        digest = hashlib.md5(s.encode("utf-8")).digest()
        return (
            int.from_bytes(digest[:4], "big") & RequestIdToIntegerConverter._MAX_INT32
        )


def merge_masks_to_flat(
    spec_masks: List[torch.Tensor],  # CPU Bool (sLᵢ, sLᵢ)
    seq_lens: torch.Tensor,  # 1-D Long (batch,)
    device: torch.device,
) -> torch.Tensor:
    """
    Optimized: build one pinned host buffer (prefilled with True), write all right halves on CPU,
    then one async H->D copy to 'device'.
    """
    # assert len(spec_masks) == len(seq_lens), "batch mismatch or empty batch"

    # CPU bookkeeping
    spec_lens_list = [m.size(0) for m in spec_masks]
    spec_lens = torch.as_tensor(spec_lens_list, dtype=torch.long)
    seq_lens_cpu = seq_lens.to(dtype=torch.long, device="cpu", copy=False)

    blk_sizes = spec_lens * (spec_lens + seq_lens_cpu)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(blk_sizes[:-1], dim=0)]
    )
    total_elems = int(blk_sizes.sum().item())

    # One pinned host buffer (all True)
    pin = device.type == "cuda"
    host_buf = torch.ones(total_elems, dtype=torch.bool, pin_memory=pin)

    # CPU writes: set right halves from each cpu_mask
    for cpu_mask, seq_len, spec_len, start in zip(
        spec_masks, seq_lens_cpu.tolist(), spec_lens_list, offsets.tolist()
    ):
        blk = host_buf.narrow(0, start, spec_len * (seq_len + spec_len)).view(
            spec_len, seq_len + spec_len
        )
        src = cpu_mask if (not pin or cpu_mask.is_pinned()) else cpu_mask.pin_memory()
        blk[:, seq_len:].copy_(src, non_blocking=False)  # CPU->CPU

    # One H->D copy
    out = torch.empty(total_elems, dtype=torch.bool, device=device)
    if device.type == "cuda":
        stream = torch.cuda.current_stream(device)
        with torch.cuda.stream(stream):
            out.copy_(host_buf, non_blocking=True)
    else:
        out.copy_(host_buf)
    return out


def merge_masks_to_flat_with_padding(
    spec_masks: List[torch.Tensor],  # CPU Bool  (real_sLᵢ, real_sLᵢ) (can be pinned)
    seq_lens: torch.Tensor,  # 1-D Long  (batch,)
    pad_dims: List[int],
    max_spec_len: int,
    device: torch.device,
    *,
    # TODO (mmarzollo): possibly allocate on host. Take care of devices for multi-device
    reuse_host: Optional[
        torch.Tensor
    ] = None,  # optional preallocated pinned host buffer (bool, >= total_elems)
    reuse_out: Optional[
        torch.Tensor
    ] = None,  # optional preallocated device buffer (bool, >= total_elems)
) -> torch.Tensor:
    """Optimized: build (or reuse) one pinned host buffer (prefilled True), write masks + padding on CPU,
    then one async H2D."""
    # assert len(spec_masks) == len(seq_lens) == len(pad_dims), "batch mismatch or empty batch"

    bsz = len(spec_masks)
    real_spec_lens = torch.tensor([m.size(0) for m in spec_masks], dtype=torch.long)
    spec_lens = torch.full((bsz,), max_spec_len, dtype=torch.long)
    seq_lens_cpu = seq_lens.to(dtype=torch.long, device="cpu", copy=False)

    blk_sizes = spec_lens * (spec_lens + seq_lens_cpu)
    offsets = torch.cat(
        [torch.zeros(1, dtype=torch.long), torch.cumsum(blk_sizes[:-1], dim=0)]
    )
    total_elems = int(blk_sizes.sum().item())

    # Host buffer (reuse if large enough)
    pin = device.type == "cuda"
    if (
        reuse_host is not None
        and reuse_host.numel() >= total_elems
        and reuse_host.dtype == torch.bool
        and (not pin or reuse_host.is_pinned())
    ):
        host_buf = reuse_host[:total_elems]
        host_buf.fill_(True)
    else:
        host_buf = torch.ones(total_elems, dtype=torch.bool, pin_memory=pin)

    # CPU writes into host buffer
    for cpu_mask, seq_len, spec_len, real_spec_len, pad_size, start in zip(
        spec_masks,
        seq_lens_cpu.tolist(),
        spec_lens.tolist(),
        real_spec_lens.tolist(),
        pad_dims,
        offsets.tolist(),
    ):
        blk = host_buf.narrow(0, start, spec_len * (seq_len + spec_len)).view(
            spec_len, seq_len + spec_len
        )
        if pad_size == 0:
            blk[:, seq_len:].copy_(
                (
                    cpu_mask
                    if (not pin or cpu_mask.is_pinned())
                    else cpu_mask.pin_memory()
                )
            )
        else:
            blk[:real_spec_len, seq_len : seq_len + real_spec_len].copy_(
                cpu_mask if (not pin or cpu_mask.is_pinned()) else cpu_mask.pin_memory()
            )
            blk[real_spec_len:, :].fill_(False)
            blk[:real_spec_len, seq_len + real_spec_len :].fill_(False)

    # Device buffer (reuse if possible)
    if (
        reuse_out is not None
        and reuse_out.numel() >= total_elems
        and reuse_out.dtype == torch.bool
        and reuse_out.device == device
    ):
        out = reuse_out[:total_elems]
    else:
        out = torch.empty(total_elems, dtype=torch.bool, device=device)

    if device.type == "cuda":
        stream = torch.cuda.current_stream(device)
        with torch.cuda.stream(stream):
            out.copy_(host_buf, non_blocking=True)
    else:
        out.copy_(host_buf)
    return out


@torch.no_grad()
def stack_masks_on_device(spec_masks, spec_len, device):
    B = len(spec_masks)

    # Build on host in pinned memory for async H→D
    out_cpu = torch.zeros((B, spec_len, spec_len), dtype=torch.bool, pin_memory=True)

    for i, m in enumerate(spec_masks):
        # keep on CPU while filling; avoid per-sample H→D
        if m.is_cuda:
            m = m.to("cpu", dtype=torch.bool, non_blocking=True)
        else:
            m = m.to(dtype=torch.bool)
        s = m.size(0)
        out_cpu[i, :s, :s].copy_(m)  # CPU→CPU copy

    # Single H→D copy (can be overlapped with compute if you use streams)
    return out_cpu.to(device, non_blocking=True)


@torch.no_grad()
def merge_packed_masks_flat_device(
    packed_masks: torch.Tensor,  # (B, S, S) bool on `device`
    seq_lens: torch.Tensor,  # (B,) long on `device`
    device: torch.device,
) -> torch.Tensor:
    """
    Builds the flat output on device, prefilled with True, then writes all SxS
    blocks into the right halves of each sample in ONE index_copy_.

    For sample i:
      block has shape (S, seq_i+S), flattened row-major in the global output.
      We write packed_masks[i] into columns [seq_i : seq_i+S] of each of the S rows.
    """
    # assert packed_masks.device == device, f"{packed_masks.device}, {device}"
    # assert seq_lens.device == device
    B, S, S2 = packed_masks.shape
    # assert S == S2, "masks must be square and share the same spec_len"

    widths = seq_lens + S  # (B,)
    blk_sz = S * widths  # (B,)
    offsets = torch.cat(
        [torch.zeros(1, device=device, dtype=torch.long), blk_sz.cumsum(0)[:-1]]
    )  # (B,)
    total = int(blk_sz.sum().item())

    # prefill with True, then overwrite the right halves
    out = torch.ones(total, dtype=torch.bool, device=device)

    # Build target indices for all (B,S,S) positions in one go:
    # pos[i, r, c] = offsets[i] + (r * widths[i]) + seq_lens[i] + c
    r = torch.arange(S, device=device, dtype=torch.long).view(1, S, 1)
    c = torch.arange(S, device=device, dtype=torch.long).view(1, 1, S)
    pos = (
        offsets.view(B, 1, 1) + seq_lens.view(B, 1, 1) + r * widths.view(B, 1, 1) + c
    )  # (B, S, S) long

    out.index_copy_(0, pos.reshape(-1), packed_masks.reshape(-1))
    return out


def default_branch_func(speculate_len: int):
    if speculate_len <= 5:
        branch_length = speculate_len - 1
    elif speculate_len <= 8:
        branch_length = 5
    elif speculate_len <= 32:
        branch_length = 6
    elif speculate_len <= 48:
        branch_length = 8
    else:
        branch_length = 10

    return branch_length, min(5, speculate_len - 1)


def _synchronize_compute_bw_device(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "npu":
        torch.npu.synchronize()
    else:
        raise ValueError(f"Unsupported device type for synchronization: {device_type}")


def _measure_compute_bw_op_ms(device_type: str, repeats: int, fn) -> float:
    if device_type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats

    _synchronize_compute_bw_device(device_type)
    start_time = time.perf_counter()
    for _ in range(repeats):
        fn()
    _synchronize_compute_bw_device(device_type)
    return (time.perf_counter() - start_time) * 1e3 / repeats


def flops_to_bandwidth_ratio(
    copy_bytes: int = 1 * 1024 * 1024 * 1024,
    mm_size: int = 8192,
    repeats: int = 100,
    device_type: Optional[str] = None,
) -> float:
    """
    Estimate accelerator balance point (GFLOPs per GB/s) using roofline-style benchmarking.

    - Bandwidth: measured with streaming AXPY-like kernel (2 reads + 1 write).
    - FLOPs: measured with large GEMM on tensor cores (BF16).
    - Ratio = GFLOPs / GB/s, i.e. arithmetic intensity required to be compute-bound.

    Args:
        copy_bytes (int): Size of the buffer for bandwidth test (default: 1 GiB).
        mm_size (int): Matrix size for GEMM test (default: 8192).
        repeats (int): Number of iterations to average (default: 100).
        device_type (Optional[str]): Accelerator to benchmark (`cuda` or `npu`).

    Returns:
        float: FLOPs-to-bandwidth ratio (GFLOPs per GB/s).
    """
    if device_type is None:
        device_type = _get_compute_bw_device_type()

    if device_type not in {"cuda", "npu"}:
        raise RuntimeError(
            "No CUDA or NPU device available for compute/bandwidth benchmarking."
        )

    # Bandwidth measurement
    elems = copy_bytes // 2  # bf16 = 2 bytes
    a = torch.randn(elems, dtype=torch.bfloat16, device=device_type)
    b = torch.randn_like(a)

    # warm-up
    for _ in range(3):
        torch.add(a, b, out=b)
    _synchronize_compute_bw_device(device_type)

    avg_ms_bw = _measure_compute_bw_op_ms(
        device_type, repeats, lambda: torch.add(a, b, out=b)
    )
    bandwidth_gbps = (3 * copy_bytes * 1e-9) / (avg_ms_bw * 1e-3)

    # FLOPs measurement
    A = torch.randn(mm_size, mm_size, device=device_type, dtype=torch.bfloat16)
    B = torch.randn_like(A)

    for _ in range(3):
        torch.matmul(A, B)
    _synchronize_compute_bw_device(device_type)

    avg_ms_flops = _measure_compute_bw_op_ms(
        device_type, repeats, lambda: torch.matmul(A, B)
    )
    flops_gflops = (2.0 * mm_size**3 / 1e9) / (avg_ms_flops * 1e-3)

    return min(
        flops_gflops / bandwidth_gbps, 192
    )  # cap it to 192, otherwise we go too high


def bfs_reorder(candidates, mask: np.ndarray):
    """
    Reorder candidates and mask into BFS order, with children sorted
    by subtree depth (longest first). Returns:
      new_candidates, new_mask, depth, parent, first_child, next_sibling
    Assumes the mask is a valid verification mask, although not bfs: children
    rows must appear after the parent row is already added.
    """
    mask = mask != 0
    R, M = mask.shape
    cols = np.arange(M)

    # Extract parent info from mask
    has_any = mask.any(axis=0)
    first_row = np.where(has_any, mask.argmax(axis=0), -1)
    row_counts = mask.sum(axis=1)
    depth = np.full(M, -1, dtype=int)
    valid = first_row >= 0
    depth[valid] = row_counts[first_row[valid]] - 1

    prev_idx = np.maximum.accumulate(np.where(mask, np.arange(M), -1), axis=1)
    parent = np.full(M, -1, dtype=int)
    mask_valid_parent = valid & (cols > 0)
    parent[mask_valid_parent] = prev_idx[
        first_row[mask_valid_parent], cols[mask_valid_parent] - 1
    ]

    # Build children adjacency
    children = [[] for _ in range(M)]
    for c, p in enumerate(parent):
        if p >= 0:
            children[p].append(c)

    # Compute subtree depths (longest path below)
    subtree_depth = np.zeros(M, dtype=int)
    order = sorted(range(M), key=lambda x: depth[x], reverse=True)
    for u in order:
        if parent[u] >= 0:
            subtree_depth[parent[u]] = max(
                subtree_depth[parent[u]], subtree_depth[u] + 1
            )

    # BFS traversal to produce new order
    bfs_order = []
    q = deque([0])  # root = col 0
    while q:
        u = q.popleft()
        bfs_order.append(u)
        # sort children by subtree depth (descending), tie-breaker by token id
        sorted_children = sorted(children[u], key=lambda v: (-subtree_depth[v], v))
        q.extend(sorted_children)

    # Remap ids
    old_to_new = {old: new for new, old in enumerate(bfs_order)}
    N = len(bfs_order)

    # Rebuild arrays in BFS order
    new_candidates = [candidates[old] for old in bfs_order]

    new_depth = np.zeros(N, dtype=int)
    new_parent = np.full(N, -1, dtype=int)
    new_first_child = np.full(N, -1, dtype=int)
    new_next_sibling = np.full(N, -1, dtype=int)

    for new_idx, old_idx in enumerate(bfs_order):
        new_depth[new_idx] = depth[old_idx]
        if parent[old_idx] >= 0:
            new_parent[new_idx] = old_to_new[parent[old_idx]]

        # children in BFS order (same sorting rule)
        ch = sorted(children[old_idx], key=lambda v: (-subtree_depth[v], v))
        if ch:
            new_first_child[new_idx] = old_to_new[ch[0]]

        # siblings in same sorted order
        if parent[old_idx] >= 0:
            sibs = sorted(
                children[parent[old_idx]], key=lambda v: (-subtree_depth[v], v)
            )
            sibs_new = [old_to_new[s] for s in sibs]
            pos = sibs_new.index(new_idx)
            if pos < len(sibs_new) - 1:
                new_next_sibling[new_idx] = sibs_new[pos + 1]

    # Rebuild BFS–mask
    new_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        new_mask[i, i] = 1
        p = new_parent[i]
        while p >= 0:
            new_mask[i, p] = 1
            p = new_parent[p]

    return (
        new_candidates,
        new_mask,
        new_depth.tolist(),
        new_first_child.tolist(),
        new_next_sibling.tolist(),
    )


def bfs_reorder_and_prune(candidates, mask: np.ndarray, top_k: int = None):
    """
    Reorder candidates and mask into BFS order, with children sorted
    by subtree depth (longest first). Returns:
      new_candidates, new_mask, depth, parent, first_child, next_sibling
    Assumes the mask is a valid verification mask, although not bfs: children
    rows must appear after the parent row is already added.

    Args:
        candidates: list of candidate objects
        mask (np.ndarray): verification mask (R x M)
        top_k (int or None): maximum number of children per node. If set,
            prune excess children (removing smallest subtrees first).
    """
    mask = mask != 0
    R, M = mask.shape
    cols = np.arange(M)

    # Extract parent info from mask
    has_any = mask.any(axis=0)
    first_row = np.where(has_any, mask.argmax(axis=0), -1)
    row_counts = mask.sum(axis=1)
    depth = np.full(M, -1, dtype=int)
    valid = first_row >= 0
    depth[valid] = row_counts[first_row[valid]] - 1

    prev_idx = np.maximum.accumulate(np.where(mask, np.arange(M), -1), axis=1)
    parent = np.full(M, -1, dtype=int)
    mask_valid_parent = valid & (cols > 0)
    parent[mask_valid_parent] = prev_idx[
        first_row[mask_valid_parent], cols[mask_valid_parent] - 1
    ]

    # Build children adjacency
    children = [[] for _ in range(M)]
    for c, p in enumerate(parent):
        if p >= 0:
            children[p].append(c)

    # Compute subtree depths (longest path below)
    subtree_depth = np.zeros(M, dtype=int)
    order = sorted(range(M), key=lambda x: depth[x], reverse=True)
    for u in order:
        if parent[u] >= 0:
            subtree_depth[parent[u]] = max(
                subtree_depth[parent[u]], subtree_depth[u] + 1
            )

    # Apply pruning if top_k is set
    if top_k is not None:
        for u in range(M):
            if len(children[u]) > top_k:
                # Sort by (depth desc, id asc), then keep first top_k
                sorted_ch = sorted(children[u], key=lambda v: (-subtree_depth[v], v))
                keep = set(sorted_ch[:top_k])
                children[u] = [c for c in sorted_ch if c in keep]

    # BFS traversal to produce new order
    bfs_order = []
    q = deque([0])  # root = col 0
    while q:
        u = q.popleft()
        bfs_order.append(u)
        sorted_children = sorted(children[u], key=lambda v: (-subtree_depth[v], v))
        q.extend(sorted_children)

    # Remap ids
    old_to_new = {old: new for new, old in enumerate(bfs_order)}
    N = len(bfs_order)

    # Rebuild arrays in BFS order
    new_candidates = [candidates[old] for old in bfs_order]

    new_depth = np.zeros(N, dtype=int)
    new_parent = np.full(N, -1, dtype=int)
    new_first_child = np.full(N, -1, dtype=int)
    new_next_sibling = np.full(N, -1, dtype=int)

    for new_idx, old_idx in enumerate(bfs_order):
        new_depth[new_idx] = depth[old_idx]
        if parent[old_idx] >= 0 and parent[old_idx] in old_to_new:
            new_parent[new_idx] = old_to_new[parent[old_idx]]

        # children in BFS order (same sorting rule, respecting pruning)
        ch = sorted(children[old_idx], key=lambda v: (-subtree_depth[v], v))
        ch = [c for c in ch if c in old_to_new]  # filter pruned
        if ch:
            new_first_child[new_idx] = old_to_new[ch[0]]

        # siblings in same sorted order
        if parent[old_idx] >= 0 and parent[old_idx] in old_to_new:
            sibs = sorted(
                children[parent[old_idx]], key=lambda v: (-subtree_depth[v], v)
            )
            sibs_new = [old_to_new[s] for s in sibs if s in old_to_new]
            pos = sibs_new.index(new_idx)
            if pos < len(sibs_new) - 1:
                new_next_sibling[new_idx] = sibs_new[pos + 1]

    # Rebuild BFS–mask
    new_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        new_mask[i, i] = 1
        p = new_parent[i]
        while p >= 0:
            new_mask[i, p] = 1
            p = new_parent[p]

    return (
        new_candidates,
        new_mask,
        new_depth.tolist(),
        new_first_child.tolist(),
        new_next_sibling.tolist(),
    )
