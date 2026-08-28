from __future__ import annotations

import bisect
import logging
import time
from typing import List

import numpy as np
import torch
from lookahead.common.lookahead_cache import LookaheadCache

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.model_free_utils import ModelFreeSpeculator, bfs_reorder

logger = logging.getLogger(__name__)


class PIASpeculator(ModelFreeSpeculator):
    "Python interface between SGLang and the PIA speculator code."

    def __init__(
        self,
        server_args: ServerArgs,
        device: str,
        captured_batch_sizes: List[int],
        num_tokens_per_bs_map: dict,
        tp_rank: int,
        tp_group: GroupCoordinator,
    ):

        if LookaheadCache is None:
            raise ImportError(
                "PIA speculator is not installed. Please install speculator first."
            )

        super().__init__(
            server_args,
            device,
            captured_batch_sizes,
            num_tokens_per_bs_map,
            tp_rank,
            tp_group,
        )

        self.max_query_length = 2

        self._init_pia_cache(server_args)

    def _init_pia_cache(self, server_args):
        self.pia_cache = LookaheadCache()
        cache_path = server_args.speculative_draft_model_path
        if cache_path is not None:
            logger.info("Loading the Lookahead cache...")
            start_time = time.time()
            self.pia_cache.load_mem(cache_path)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(
                f"Finished loading the Lookahead cache. Time taken: {elapsed_time:.2f} seconds"
            )
        set_topk(self.pia_cache, self.topk)

    def add_new_sequence(self, req: Req) -> None:
        speculator_req_id, already_present = self.request_converter.acquire(req.rid)
        if not already_present:
            self.pia_cache.put(
                req.origin_input_ids, mode="input", idx=speculator_req_id
            )
            req.sssd_id = speculator_req_id
        # else it didn't finish prefilling with a single chunk

    def stream_put(self, new_tokens: List[int], req: Req) -> None:
        self.pia_cache.stream_put(
            new_tokens, final=False, mode="output", idx=req.sssd_id
        )

    def get_candidates_mask(self, batch: ScheduleBatch):
        candidates = []
        decoding_masks = []
        position_ids = []
        next_child_idx = []
        next_sibling_idx = []

        for req in batch.reqs:
            if len(req.output_ids) >= self.max_query_length:
                prefix = req.output_ids[-self.max_query_length :]
            else:
                tokens_from_prompt = self.max_query_length - len(req.output_ids)
                prefix = req.origin_input_ids[-tokens_from_prompt:] + req.output_ids

            # For some reason (e.g. warmup) the prompt might have not be added, and the request might not have an id
            if req.sssd_id is None:
                logger.warning(
                    f"The prompt of request {req.rid} has not been inserted."
                )
                speculator_req_id, already_present = self.request_converter.acquire(
                    req.rid
                )
                assert (
                    not already_present
                ), f"Request {req.rid} already added to the speculator with id {req.sssd_id}"
                req.sssd_id = speculator_req_id

            min_input_size = 0
            min_output_size = max(self.num_draft_tokens // 2, 1)
            spec_token_ids, decoding_mask, _ = self.pia_cache.hier_get(
                prefix,
                decoding_length=self.num_draft_tokens,
                branch_length=self.num_steps,
                min_input_size=min_input_size,
                min_output_size=min_output_size,
                mode="mix",
                idx=req.sssd_id,
            )
            # Convert to sglang format
            if len(spec_token_ids) > 0:
                cand, mask, pos_ids, next_tok_ids, next_sib_ids = bfs_reorder(
                    spec_token_ids, np.array(decoding_mask, dtype=bool)
                )
                candidates.append(cand)
                decoding_masks.append(torch.from_numpy(mask))
                position_ids.append(pos_ids)
                next_child_idx.append(next_tok_ids)
                next_sibling_idx.append(next_sib_ids)

        return (
            candidates,
            position_ids,
            next_child_idx,
            next_sibling_idx,
            decoding_masks,
        )

    def clear_seq_cache(self, req_id: str) -> None:
        speculator_req_id, already_present = self.request_converter.acquire(req_id)
        if not already_present:
            logger.warning("Sequence not present in the speculator")
        self.pia_cache.stream_put(
            [], branch_length=0, final=True, mode="output", idx=speculator_req_id
        )
        self.request_converter.release(req_id)

    def update_speculate_params_adaptive(self, bs: int):
        index = bisect.bisect_left(self.captured_batch_sizes, bs)
        if index >= len(self.captured_batch_sizes):
            return 1, 0, 0  # Beyond captured graphs, should not happen. Don't speculate
        pad_bs = self.captured_batch_sizes[index]
        if self.last_pad_bs == pad_bs:
            return self.num_draft_tokens, self.num_steps, self.topk

        self.last_pad_bs = bs
        self.num_draft_tokens = self.num_tokens_per_bs_map[pad_bs]
        self.num_steps = min(self.num_draft_tokens - 1, 8)
        self.topk = min(self.num_draft_tokens - 1, 5)

        return self.num_draft_tokens, self.num_steps, self.topk


# Monkey-path the _ravel() method to set a top-k


def _ravel_with_topk(
    self,
    nodes,
    ids,
    mask,
    pid,
    max_size=64,
    max_length=8,
    min_output_freq=1.0,
    min_input_freq=1.0,
    min_mix_freq=1.0,
    output_weight=1e-4,
    sizes=None,
    mode="mix",
    idx=0,
):
    if len(ids) >= max_size or max_length <= 0:
        return

    sorts = [
        (
            k,
            v,
            (1.0 - output_weight) * v.freqs.get(idx, 0.0)
            + output_weight * v.freqs.get(-1, 0.0),
        )
        for k, v in nodes.items()
    ]
    sorts = sorted(sorts, key=lambda x: x[2], reverse=True)

    # Cap the number of children to expand
    if self.topk is not None:
        sorts = sorts[: self.topk]

    for tid, node, fm in sorts:
        if len(ids) >= max_size:
            return
        f_i = node.freqs.get(idx, 0.0)
        f_o = node.freqs.get(-1, 0.0)
        if mode == "mix":
            if f_i < min_input_freq and f_o < min_output_freq and fm < min_mix_freq:
                continue
        elif mode == "input":
            if f_i < min_input_freq:
                continue
        else:
            if f_o < min_output_freq:
                continue
        if f_i > 0.0:
            sizes[0] += 1
        if f_o > 0.0:
            sizes[1] += 1
        ids.append(tid)
        rid = len(ids) - 1

        if pid > -1:
            mask[rid] = mask[pid]
        mask[rid, rid] = 1
        if len(node.children) > 0:
            self._ravel(
                node.children,
                ids,
                mask,
                rid,
                max_size=max_size,
                max_length=max_length - 1,
                min_output_freq=min_output_freq,
                min_input_freq=min_input_freq,
                min_mix_freq=min_mix_freq,
                output_weight=output_weight,
                sizes=sizes,
                mode=mode,
                idx=idx,
            )


LookaheadCache._ravel = _ravel_with_topk
LookaheadCache.topk = None


def set_topk(cache: LookaheadCache, topk):
    cache.topk = topk
