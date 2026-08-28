import logging
from typing import Optional, Tuple

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker import (
    get_last_loc_large_page_size_large_top_k,
    get_last_loc_large_page_size_top_k_1,
)
from sglang.srt.speculative.model_free_info import (
    ModelFreeVerifyInput,
    ModelFreeVerifyOutput,
)
from sglang.srt.speculative.model_free_utils import ProxySpeculator
from sglang.srt.speculative.pld_utils import ProxySpeculatorPld
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_draft_cache_locs,
    generate_token_bitmask,
    load_token_map,
    maybe_detect_nan,
)
from sglang.srt.utils import next_power_of_2

logger = logging.getLogger(__name__)


class ModelFreeWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args

        self.tp_rank = tp_rank
        self.tp_group = target_worker.model_runner.tp_group

        # If the speculation is not adaptive, these values are fixes, otherwise they are updated at each iteration
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens

        # For adaptive speculation
        self.adaptive_speculation = server_args.speculative_adaptive
        self.last_bs = None

        self.enable_nan_detection = server_args.enable_nan_detection
        self.gpu_id = gpu_id
        self.device = server_args.device
        if self.device == "cuda":
            self.device = torch.device(f"cuda:{gpu_id}")
        self.target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Share the allocator with a target worker.
        # Draft and target worker own their own KV cache pools.
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )
        # For adaptive speculation. Quite ugly way of getting this information, but the closest connection I could find.
        # it's needed for the speculative adaptive case to know what are the real batch sizes after padding, and get
        # the corresponding speculation length.
        # TODO: the captured batch sizes could change (see graph_runner->recapture_if_needed). It shouldn't
        # normally happen, and if it does the code might crash.
        if target_worker.model_runner.graph_runner is not None:
            self.captured_batch_sizes = (
                target_worker.model_runner.graph_runner.capture_bs
            )
            print(self.captured_batch_sizes)
            self.num_tokens_per_bs_map = (
                target_worker.model_runner.graph_runner.num_tokens_per_bs_map
            )
        else:
            # # No cuda graphs, any batch size is ok
            # self.captured_batch_sizes = list(range(1, server_args.max_running_requests + 1))
            # self.num_tokens_per_bs_map = get_bs_speclen(self.captured_batch_sizes, self.speculative_algorithm)
            raise ValueError(
                "Model-free speculation does not support this attention backend (probably because also EAGLE doesn't."
            )

        # Load hot token ids
        if server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(server_args.speculative_token_map)
            server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        if self.tp_rank != 0:
            if self.speculative_algorithm.is_pld():
                # TODO (mmarzollo): Should probably treat other cases too
                self.speculator = ProxySpeculatorPld(
                    server_args,
                    gpu_id,
                    self.captured_batch_sizes,
                    self.num_tokens_per_bs_map,
                    self.tp_rank,
                    self.tp_group,
                    owner_rank=0,
                )
            else:
                self.speculator = ProxySpeculator(
                    server_args,
                    gpu_id,
                    self.captured_batch_sizes,
                    self.num_tokens_per_bs_map,
                    self.tp_rank,
                    self.tp_group,
                    owner_rank=0,
                )

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Run speculative decoding forward.

        NOTE: Many states of batch is modified as you go through. It is not guaranteed that
        the final output batch have the same state as the input.

        Args:
            batch: The batch to run forward. The state of the batch is modified as it runs.
        Returns:
            A tuple of the final logit output of the target model, next tokens accepted,
            the batch id (used for overlap schedule), and number of accepted tokens.
        """
        # Prefill
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            # Insert prompts in the speculator
            for req in batch.reqs:
                self.speculator.add_new_sequence(req)
            # Run prefill
            logits_output, next_token_ids, _ = self.forward_target_extend(batch)
            # Add single generated token from prefill to the speculator
            for req, next_tok_ids in zip(batch.reqs, next_token_ids):
                self.speculator.stream_put([next_tok_ids.item()], req)

            batch.spec_info = None
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        # Decode
        else:
            spec_info = self.draft(batch)
            # Verification with target model
            logits_output, verify_output, _, can_run_cuda_graph = self.verify(
                batch, spec_info
            )
            # Update the speculator
            for req, next_tok_ids in zip(batch.reqs, verify_output.new_accepted_tokens):
                self.speculator.stream_put(next_tok_ids, req)
            for req_id in verify_output.finished_requests:
                self.speculator.clear_seq_cache(req_id)

            batch.spec_info = None  # No DraftInput
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )

    def forward_target_extend(
        self, batch: ScheduleBatch
    ) -> Tuple[LogitsProcessorOutput, torch.Tensor, int, Optional[torch.Tensor]]:
        """Run the target extend.

        Args:
            batch: The batch to run. States could be modified.

        Returns:
            logits_output: The output of logits.
            next_token_ids: Next token ids generated.
            bid: The model batch ID. Used for overlap schedule.
        """
        # Forward with the target model and get hidden states.
        # We need the full hidden states to prefill the KV cache of the draft model.
        model_worker_batch = batch.get_model_worker_batch()
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
        logits_output, next_token_ids = (
            batch_result.logits_output,
            batch_result.next_token_ids,
        )
        return (
            logits_output,
            next_token_ids,
            model_worker_batch.seq_lens_cpu,
        )

    def _draft_preprocess_decode(self, batch: ScheduleBatch):
        batch.maybe_evict_swa()
        for req in batch.reqs:
            req.decode_batch_idx += 1

        # Parse args
        num_seqs = batch.batch_size()

        # Allocate cache locations
        # Layout of the out_cache_loc
        # [       topk 0         ] [       topk 1         ]
        # [iter=0, iter=1, iter=2] [iter=0, iter=1, iter=2]
        if self.page_size == 1:
            alloc_len_per_decode = self.speculative_num_steps * self.topk
            # TODO: We only need self.speculative_num_steps - 1 * topk cache loc
            out_cache_loc, token_to_kv_pool_state_backup = alloc_token_slots(
                batch.tree_cache,
                num_seqs * alloc_len_per_decode,
                backup_state=True,
            )
        else:
            if self.topk == 1:
                prefix_lens, seq_lens, last_loc = get_last_loc_large_page_size_top_k_1(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                )
                prefix_lens_cpu = batch.seq_lens_cpu
                seq_lens_cpu = batch.seq_lens_cpu + self.speculative_num_steps
                extend_num_tokens = num_seqs * self.speculative_num_steps
            else:
                # In this case, the last partial page needs to be duplicated.
                # KV cache layout in batch.req_to_token_pool.req_to_token:
                #
                # | -------- | -- xxxx .. | -- xxxx .. | -- xxxx .. |
                #    prefix     top-k = 0    tok-k = 1    top-k = 2
                #
                #  "-" means prefix tokens
                #  "x" means speculative draft tokens
                #  "." means padded tokens

                (
                    prefix_lens,
                    seq_lens,
                    last_loc,
                    self.num_new_pages_per_topk,
                    self.extend_lens,
                    last_page_lens,
                ) = get_last_loc_large_page_size_large_top_k(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    batch.seq_lens,
                    self.speculative_num_steps,
                    self.topk,
                    self.page_size,
                )

                prefix_lens_cpu = batch.seq_lens_cpu
                last_page_lens_cpu = prefix_lens_cpu % self.page_size
                num_new_pages_per_topk = (
                    last_page_lens_cpu + self.speculative_num_steps + self.page_size - 1
                ) // self.page_size
                seq_lens_cpu = (
                    prefix_lens_cpu // self.page_size * self.page_size
                    + num_new_pages_per_topk * (self.page_size * self.topk)
                )
                extend_num_tokens = torch.sum((seq_lens_cpu - prefix_lens_cpu)).item()

            out_cache_loc, token_to_kv_pool_state_backup = (
                alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    prefix_lens,
                    prefix_lens_cpu,
                    seq_lens,
                    seq_lens_cpu,
                    last_loc,
                    extend_num_tokens,
                    backup_state=True,
                )
            )

        if self.page_size > 1 and self.topk > 1:
            last_page_lens_cumsum = torch.cumsum(last_page_lens, dim=0)
            duplicate_cache_len = torch.sum(last_page_lens_cpu).item() * (self.topk - 1)
            target_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
            source_cache_loc = torch.zeros(
                duplicate_cache_len, dtype=torch.int32, device=self.device
            )
        else:
            # When source_cache_loc is not needed, simply skip
            duplicate_cache_len = 0
            source_cache_loc, target_cache_loc, last_page_lens_cumsum = None, None, None

        assign_draft_cache_locs[(num_seqs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens,
            self.num_new_pages_per_topk,
            out_cache_loc,
            source_cache_loc,
            target_cache_loc,
            last_page_lens_cumsum,
            duplicate_cache_len,
            batch.req_to_token_pool.req_to_token.shape[1],
            self.topk,
            self.speculative_num_steps,
            self.page_size,
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps + self.page_size),
        )

        if self.page_size > 1 and self.topk > 1:
            if duplicate_cache_len > 0:
                batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    target_cache_loc, source_cache_loc
                )
            # Remove padded slots
            # TODO: We only need self.speculative_num_steps - 1 cache loc
            out_cache_loc = out_cache_loc[
                : num_seqs * self.topk * self.speculative_num_steps
            ]

        batch.out_cache_loc = out_cache_loc
        batch.seq_lens_sum = torch.sum(batch.seq_lens).item()
        batch.return_hidden_states = False
        self.token_to_kv_pool_allocator.restore_state(token_to_kv_pool_state_backup)

    def draft(self, batch: ScheduleBatch):
        # Parse args
        if batch.forward_mode.is_idle():
            batch.spec_info = None

            return ModelFreeVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                device=self.device,
            )
        else:
            if self.adaptive_speculation:
                bs = batch.batch_size()
                if bs != self.last_bs:
                    # Batch size changed: update arguments
                    (
                        self.speculative_num_draft_tokens,
                        self.speculative_num_steps,
                        self.topk,
                    ) = self.speculator.update_speculate_params_adaptive(bs)
                    self.last_bs = bs

            self._draft_preprocess_decode(batch)

            (
                tree_mask,
                position,
                retrive_index,
                retrive_next_token,
                retrive_next_sibling,
                draft_tokens,
            ) = self.speculator.get_draft(batch)

            return ModelFreeVerifyInput(
                draft_token=draft_tokens,
                custom_mask=tree_mask,
                positions=position,
                retrive_index=retrive_index,
                retrive_next_token=retrive_next_token,
                retrive_next_sibling=retrive_next_sibling,
                retrive_cum_len=None,
                spec_steps=self.speculative_num_steps,
                topk=self.topk,
                draft_token_num=self.speculative_num_draft_tokens,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                seq_lens_sum=batch.seq_lens_sum,
                seq_lens_cpu=None,
            )

    def verify(self, batch: ScheduleBatch, spec_info: ModelFreeVerifyInput):
        batch.token_to_kv_pool_allocator.get_kvcache()
        spec_info.prepare_for_verify(batch, self.page_size)
        spec_info.num_tokens_per_req = self.speculative_num_steps + 1
        batch.return_hidden_states = False
        batch.forward_mode = (
            ForwardMode.TARGET_VERIFY
            if not batch.forward_mode.is_idle()
            else ForwardMode.IDLE
        )
        batch.spec_info = spec_info

        model_worker_batch = batch.get_model_worker_batch(
            seq_lens_cpu_cache=spec_info.seq_lens_cpu
        )
        assert model_worker_batch.capture_hidden_mode == spec_info.capture_hidden_mode

        if batch.has_grammar:
            retrieve_next_token_cpu = spec_info.retrive_next_token.cpu()
            retrieve_next_sibling_cpu = spec_info.retrive_next_sibling.cpu()
            draft_tokens_cpu = spec_info.draft_token.view(
                spec_info.retrive_next_token.shape
            ).cpu()

        # Forward
        batch_result = self.target_worker.forward_batch_generation(
            model_worker_batch, is_verify=True
        )
        logits_output, can_run_cuda_graph = (
            batch_result.logits_output,
            batch_result.can_run_cuda_graph,
        )

        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            # Overlap the CPU operations for bitmask generation with the forward pass.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                spec_info,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert spec_info.grammar is not None
                vocab_mask = vocab_mask.to(spec_info.retrive_next_token.device)
                # NOTE (sk): otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        if self.enable_nan_detection:
            maybe_detect_nan(logits_output)
        res: ModelFreeVerifyOutput = spec_info.verify(
            batch,
            logits_output,
            self.token_to_kv_pool_allocator,
            self.page_size,
            vocab_mask,
        )

        # Post process based on verified outputs.
        # Pick indices that we care (accepted)
        logits_output.next_token_logits = logits_output.next_token_logits[
            res.accepted_indices
        ]

        if batch.return_logprob:
            add_output_logprobs_for_spec_v1(batch, res, logits_output)

        # Prepare the batch for the next draft forwards.
        batch.forward_mode = (
            ForwardMode.DECODE if not batch.forward_mode.is_idle() else ForwardMode.IDLE
        )

        return logits_output, res, model_worker_batch, can_run_cuda_graph

    # Methods used by SSSD and PIA to add a request when in goes in the prefill (only for PD)
    def add_request(self, req: Req):
        if self.tp_rank == 0:
            self.speculator.add_new_sequence(req)

    def remove_request(self, req_id: str):
        if self.tp_rank == 0:
            self.speculator.clear_seq_cache(req_id)
