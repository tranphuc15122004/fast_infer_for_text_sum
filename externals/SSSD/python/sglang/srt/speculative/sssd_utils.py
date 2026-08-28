from __future__ import annotations

import logging
import time
from typing import List

import torch

import sssd_speculator
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.model_free_utils import ModelFreeSpeculator

logger = logging.getLogger(__name__)


class SSSDSpeculator(ModelFreeSpeculator):
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

        if sssd_speculator is None:
            raise ImportError(
                "SSSD speculator is not installed. Please install speculator first."
            )

        super().__init__(
            server_args,
            device,
            captured_batch_sizes,
            num_tokens_per_bs_map,
            tp_rank,
            tp_group,
        )

        self.max_query_length = 4

        self._init_speculator(server_args)

    def _init_speculator(self, server_args):
        datastore_path = server_args.speculative_draft_model_path
        datastore_path = "" if datastore_path is None else datastore_path
        logger.info("Loading the SSSD speculator...")
        start_time = time.time()
        self.speculator = sssd_speculator.Reader(
            index_file_path=datastore_path,
            vocab_size=self.tokenizer.vocab_size + 100,
            stop_token=(
                -1
                if self.tokenizer.bos_token_id is None
                else self.tokenizer.bos_token_id
            ),
            max_search_entries=100,
            prompt_branch_length=8,
            prompt_prefix_length=self.max_query_length,
            max_output_size=server_args.num_reserved_decode_tokens,
            live_datastore=False,
            update_interval_ms=60 * 1000,
            max_update_chunk_size=512 * 1024 * 1024,
            max_indices=8,
            max_batch_size=server_args.max_running_requests,
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Finished loading the SSSD speculator. Time taken: {elapsed_time:.2f} seconds"
        )

    def add_new_sequence(self, req: Req) -> None:
        speculator_req_id, already_present = self.request_converter.acquire(req.rid)
        if not already_present:
            self.speculator.put(req.origin_input_ids, seq_id=speculator_req_id)
            req.sssd_id = speculator_req_id
        # else it didn't finish prefilling with a single chunk

    def stream_put(self, new_tokens: List[int], req: Req) -> None:
        assert (
            req.sssd_id is not None
        ), f"Request {req.rid} has no prompt inserted for speculation."
        self.speculator.stream_put(new_tokens=new_tokens, seq_id=req.sssd_id)

    def get_candidates_mask(self, batch: ScheduleBatch):
        prefixes = []
        speculator_req_ids = []
        for req in batch.reqs:
            if len(req.output_ids) >= self.max_query_length:
                prefixes.append(req.output_ids[-self.max_query_length :])
            else:
                tokens_from_prompt = self.max_query_length - len(req.output_ids)
                prefixes.append(
                    req.origin_input_ids[-tokens_from_prompt:] + req.output_ids
                )

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

            speculator_req_ids.append(req.sssd_id)

        bs = batch.batch_size()
        speculate_lens = [self.num_draft_tokens] * bs
        branch_lens = [self.num_steps] * bs
        max_topks = [self.topk] * bs

        candidates, position_ids, next_child_idx, next_sibling_idx, decoding_masks = (
            self.speculator.get_candidates_sglang(
                prefixes=prefixes,
                decoding_lengths=speculate_lens,
                branch_lengths=branch_lens,
                max_topks=max_topks,
                seq_ids=speculator_req_ids,
            )
        )

        # Convert numpy masks to torch (shares same memory)
        decoding_masks = [torch.from_numpy(m) for m in decoding_masks]

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
        self.speculator.finish_sequence(seq_id=speculator_req_id)
        self.request_converter.release(req_id)
