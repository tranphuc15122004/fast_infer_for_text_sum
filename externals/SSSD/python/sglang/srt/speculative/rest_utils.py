from __future__ import annotations

import logging
import time
from typing import List

import draftretriever_adapted
import numpy as np
import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.model_free_utils import (
    ModelFreeSpeculator,
    bfs_reorder_and_prune,
)

logger = logging.getLogger(__name__)


class RESTSpeculator(ModelFreeSpeculator):
    "Python interface between SGLang and the REST speculator code."

    def __init__(
        self,
        server_args: ServerArgs,
        device: str,
        captured_batch_sizes: List[int],
        num_tokens_per_bs_map: dict,
        tp_rank: int,
        tp_group: GroupCoordinator,
    ):

        if draftretriever_adapted is None:
            raise ImportError(
                "REST speculator is not installed. Please install speculator first."
            )

        super().__init__(
            server_args,
            device,
            captured_batch_sizes,
            num_tokens_per_bs_map,
            tp_rank,
            tp_group,
        )

        self.max_query_length = 3

        self._init_draftretriever(server_args)

    def _init_draftretriever(self, server_args):
        datastore_path = server_args.speculative_draft_model_path
        s_time = time.time()
        print("Starting to load the datastore...")
        self.datastore = draftretriever_adapted.Reader(
            index_file_path=datastore_path,
        )
        print(f"Datastore loaded. Time taken: {int(time.time()-s_time)} seconds.")

    def get_candidates_mask(self, batch: ScheduleBatch):
        # TODO: For now assume all speculate_lens are the same (SGLang doesn't support variable spec_len out of the box)
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

            try:
                spec_token_ids, mask, _ = self.datastore.search(
                    prefix,
                    k=5000,
                    choices=self.num_draft_tokens
                    - 1,  # rest uses different convention (not including last token)
                    long=self.num_steps,
                )

                if not spec_token_ids:
                    # No match in the datastore
                    candidates.append([prefix[-1]])
                    decoding_masks.append(torch.ones((1, 1), dtype=torch.bool))
                    position_ids.append([0])
                    next_child_idx.append([-1])
                    next_sibling_idx.append([-1])
                else:
                    # Convert to sglang format
                    cand, mask, pos_ids, next_tok_ids, next_sib_ids = (
                        bfs_reorder_and_prune(
                            spec_token_ids, np.array(mask, dtype=bool), self.topk
                        )
                    )
                    candidates.append(cand)
                    decoding_masks.append(torch.from_numpy(mask))
                    position_ids.append(pos_ids)
                    next_child_idx.append(next_tok_ids)
                    next_sibling_idx.append(next_sib_ids)

            except ValueError as e:
                logger.debug("Error retrieving data from the REST speculator: ", e)
                candidates.append([prefix[-1]])
                decoding_masks.append(torch.ones((1, 1), dtype=torch.bool))
                position_ids.append([0])
                next_child_idx.append([-1])
                next_sibling_idx.append([-1])

        return (
            candidates,
            position_ids,
            next_child_idx,
            next_sibling_idx,
            decoding_masks,
        )
