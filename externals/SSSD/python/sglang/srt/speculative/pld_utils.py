from __future__ import annotations

import bisect
import logging
from typing import List

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.model_free_utils import ModelFreeSpeculator, ProxySpeculator

logger = logging.getLogger(__name__)


class PLDSpeculator(ModelFreeSpeculator):
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

        super().__init__(
            server_args,
            device,
            captured_batch_sizes,
            num_tokens_per_bs_map,
            tp_rank,
            tp_group,
        )

        self.max_query_length = 3

        self.device = torch.device(device)
        # If the speculation is not adaptive, these values are fixes, otherwise they are updated at each iteration
        if not self.speculative_adaptive:
            assert (
                self.num_steps == self.num_draft_tokens - 1
            ), f"{self.num_steps}, {self.num_draft_tokens}"
            assert self.topk == 1, f"{self.topk}"

    def get_candidates_mask(self, batch: ScheduleBatch):
        candidates = []
        decoding_masks = []
        position_ids = []
        next_child_idx = []
        next_sibling_idx = []

        for req in batch.reqs:
            input_ids = torch.tensor(req.origin_input_ids + req.output_ids).unsqueeze(0)
            input_length = input_ids.size(1)

            cand = None
            for ngram_size in range(min(input_length, self.max_query_length), 0, -1):
                # Extract the last n tokens as our search ngram
                ngram = input_ids[0, -ngram_size:].tolist()

                # Create sliding windows of size ngram_size
                windows = input_ids.unfold(dimension=1, size=ngram_size, step=1)

                # Convert ngram to a tensor for comparison
                ngram_tensor = torch.tensor(ngram, device=input_ids.device).unsqueeze(0)

                # Find where the windows match the ngram
                matches = (windows == ngram_tensor).all(dim=2)

                # Get the indices of matches
                match_indices = matches.nonzero(as_tuple=True)[1]

                # Iterate through match indices to find a valid continuation
                for idx in match_indices:
                    start_idx = idx + ngram_size
                    end_idx = start_idx + self.num_steps
                    # Ensure we don't go beyond the length of input_ids and avoid self-match
                    if (
                        end_idx <= input_length
                        and start_idx < input_length - ngram_size
                    ):
                        cand = input_ids[0, start_idx - 1 : end_idx].tolist()
                        break

            if cand is None:
                cand = input_ids[0, -1:].tolist()

            # Linear structure (no trees): always fixed
            candidates.append(cand)
            num_toks = len(cand)
            decoding_masks.append(
                torch.tril(torch.ones((num_toks, num_toks), dtype=torch.bool))
            )
            position_ids.append(list(range(num_toks)))
            next_child_idx.append(list(range(1, num_toks)) + [-1])
            next_sibling_idx.append([-1] * num_toks)

        return (
            candidates,
            position_ids,
            next_child_idx,
            next_sibling_idx,
            decoding_masks,
        )

    def update_speculate_params_adaptive(self, bs: int):
        index = bisect.bisect_left(self.captured_batch_sizes, bs)
        if index >= len(self.captured_batch_sizes):
            return 1, 0, 0  # Beyond captured graphs, should not happen. Don't speculate
        pad_bs = self.captured_batch_sizes[index]
        if self.last_pad_bs == pad_bs:
            return self.num_draft_tokens, self.num_steps, self.topk

        self.last_pad_bs = bs
        self.num_draft_tokens = self.num_tokens_per_bs_map[pad_bs]
        self.num_steps = self.num_draft_tokens - 1
        self.topk = 1

        return self.num_draft_tokens, self.num_steps, self.topk


class ProxySpeculatorPld(ProxySpeculator):
    # Override choice of mask to match broadcasting
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
