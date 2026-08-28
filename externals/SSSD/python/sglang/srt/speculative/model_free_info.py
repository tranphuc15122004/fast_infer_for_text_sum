import copy
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.speculative.spec_utils import (
    TREE_SPEC_KERNEL_AVAILABLE,
    align_evict_mask_to_page_size,
    assign_req_to_token_pool_func,
    create_accept_length_filter,
    filter_finished_cache_loc_kernel,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
    softmax_topk_then_topp,
)
from sglang.srt.utils import is_cuda, next_power_of_2

logger = logging.getLogger(__name__)

if is_cuda():
    from sgl_kernel import tree_speculative_sampling_target_only


@dataclass
class ModelFreeVerifyInput(SpecInput):
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrive_index: torch.Tensor
    retrive_next_token: torch.Tensor
    retrive_next_sibling: torch.Tensor
    retrive_cum_len: torch.Tensor
    spec_steps: int
    topk: int
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: torch.Tensor
    grammar: BaseGrammarObject = None
    fused: bool = True
    naive_sampling: bool = False

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                len(batch.input_ids),
            )
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
            )
            self.last_loc = last_loc

        bs = batch.batch_size()
        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            bs,
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=device
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * batch_size,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask_numel = (
            paged_kernel_lens_sum * self.draft_token_num
            + (self.draft_token_num**2) * batch_size
        )
        if self.custom_mask.numel() < mask_numel:
            # FIXME(attn): temporary fix for custom mask padding with cuda graph
            self.custom_mask = torch.cat(
                [
                    self.custom_mask,
                    torch.full(
                        (mask_numel - self.custom_mask.numel(),),
                        True,
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=0,
            )

        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    @classmethod
    def create_idle_input(
        cls, topk: int, spec_steps: int, num_verify_tokens: int, device: str = "cuda"
    ):
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device=device),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device=device),
            positions=torch.empty((0,), dtype=torch.int64, device=device),
            retrive_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            retrive_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            retrive_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device=device
            ),
            retrive_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
        )

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,  # For grammar
    ) -> torch.Tensor:
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).

        WARNING: This API in-place modifies the states of logits_output

        This API updates values inside logits_output based on the accepted
        tokens. I.e., logits_output.next_token_logits only contains
        accepted token logits.
        """
        if batch.forward_mode.is_idle():
            return ModelFreeVerifyInput(
                logits_output=logits_output,
                verified_id=torch.empty(0, dtype=torch.long, device=batch.device),
                accept_length_per_req_cpu=[],
                accepted_indices=torch.full(
                    (0, self.spec_steps + 1),
                    -1,
                    dtype=torch.int32,
                    device=batch.device,
                ),
            )

        bs = self.retrive_index.shape[0]
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        sampling_info = batch.sampling_info

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        predict = torch.empty(predict_shape, dtype=torch.int32, device=batch.device)
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=batch.device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=batch.device)

        if bs != len(sampling_info):
            sampling_info = copy.deepcopy(sampling_info)
            # NOTE: retrive_index are the indices of the requests that are kept.
            sampling_info.filter_batch(self.retrive_index.tolist(), self.retrive_index)

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        # Apply penalty
        if (
            sampling_info.penalizer_orchestrator.is_required
            or sampling_info.logit_bias is not None
        ):
            # This is a relaxed version of penalties for speculative decoding.
            linear_penalty = torch.zeros(
                (bs, logits_output.next_token_logits.shape[1]),
                dtype=torch.float32,
                device=batch.device,
            )
            sampling_info.apply_logits_bias(linear_penalty)
            logits_output.next_token_logits.add_(
                torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
            )

        # Apply grammar mask
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits, vocab_mask=vocab_mask
            )

        if batch.sampling_info.is_all_greedy:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)

            predict, accept_index, accept_length = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
        else:
            flat_logits = logits_output.next_token_logits
            flat_ks = torch.repeat_interleave(
                sampling_info.top_ks, self.draft_token_num, dim=0
            ).view(-1)
            flat_ps = torch.repeat_interleave(
                sampling_info.top_ps, self.draft_token_num, dim=0
            ).view(-1)
            flat_T = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            ).view(-1)

            use_greedy_fallback = self.naive_sampling or not TREE_SPEC_KERNEL_AVAILABLE
            target_probs, sampled_idx = softmax_topk_then_topp(
                logits=flat_logits,
                ks=flat_ks,
                ps=flat_ps,
                temperatures=flat_T,
                return_sample=use_greedy_fallback,
                generator=None,
            )

            if use_greedy_fallback:
                target_predict = sampled_idx.reshape(bs, self.draft_token_num)

                predict, accept_index, accept_length = verify_tree_greedy_func(
                    predicts=predict,  # mutable
                    accept_index=accept_index,  # mutable
                    accept_token_num=accept_length,  # mutable
                    candidates=candidates,
                    retrive_index=self.retrive_index,
                    retrive_next_token=self.retrive_next_token,
                    retrive_next_sibling=self.retrive_next_sibling,
                    target_predict=target_predict,
                    topk=self.topk,
                )
            else:
                target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

                draft_probs = torch.zeros(
                    target_probs.shape, dtype=torch.float32, device=batch.device
                )

                # coins for rejection sampling
                coins = torch.rand_like(
                    candidates, dtype=torch.float32, device=batch.device
                )
                # coins for final sampling
                coins_for_final_sampling = torch.rand(
                    (bs,), dtype=torch.float32, device=batch.device
                )
                tree_speculative_sampling_target_only(
                    predicts=predict,  # mutable
                    accept_index=accept_index,  # mutable
                    accept_token_num=accept_length,  # mutable
                    candidates=candidates,
                    retrive_index=self.retrive_index,
                    retrive_next_token=self.retrive_next_token,
                    retrive_next_sibling=self.retrive_next_sibling,
                    uniform_samples=coins,
                    uniform_samples_for_final_sampling=coins_for_final_sampling,
                    target_probs=target_probs,
                    draft_probs=draft_probs,
                    threshold_single=get_global_server_args().speculative_accept_threshold_single,
                    threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                    deterministic=True,
                )

        unfinished_index = []
        unfinished_accept_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        finished_requests = []
        has_finished = False

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            num_accepted = 0
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                num_accepted += 1
                id = predict_cpu[idx]
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    finished_requests.append(req.rid)
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break
                else:
                    if req.grammar is not None:
                        try:
                            req.grammar.accept_token(id)
                        except ValueError as e:
                            logger.info(
                                f"{i=}, {req=}\n" f"{accept_index=}\n" f"{predict=}\n"
                            )
                            raise e
            # Update KV cache tracking for the accepted tokens
            req.kv_committed_len += num_accepted
            req.kv_allocated_len = req.kv_committed_len
            if not req.finished():
                unfinished_index.append(i)
                if idx == -1:
                    unfinished_accept_index.append(accept_index[i, :j])
                else:
                    unfinished_accept_index.append(accept_index[i])
            req.spec_verify_ct += 1
            accepted_draft_tokens = max(0, num_accepted - 1)
            req.spec_accepted_tokens += accepted_draft_tokens
            req.update_spec_acceptance_histogram(accepted_draft_tokens)

        # Recompute from finalized accept_index for consistency.
        accept_length = torch.clamp((accept_index != -1).sum(dim=1) - 1, min=0)

        # Free the KV cache for unaccepted tokens
        # TODO: fuse them
        accept_index = accept_index[accept_index != -1]
        verified_id = predict[accept_index]
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False
        accept_length_cpu = accept_length.cpu()
        # FIXME: this `tolist()` fixes the numerical calculation consistency
        # try to unify the tensor representation and list representation
        accept_length_list = accept_length_cpu.tolist()

        if page_size == 1:
            # TODO: boolean array index leads to a device sync. Remove it.
            token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        else:
            if self.topk == 1:
                # Only evict full empty page. Do not evict partial empty page
                align_evict_mask_to_page_size[len(batch.seq_lens),](
                    batch.seq_lens,
                    evict_mask,
                    page_size,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                )
                token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            else:
                # Shift the accepted tokens to the beginning.
                # Only evict the last part
                src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                    batch.seq_lens,
                    batch.out_cache_loc,
                    accept_index,
                    accept_length,
                    self.draft_token_num,
                    page_size,
                )
                to_free_slots = torch.empty(
                    (to_free_num_slots.sum().item(),),
                    dtype=torch.int64,
                    device=to_free_num_slots.device,
                )

                # out_cache_loc: [0  1  2,  3  4  5,  6  7  8]
                # accept_index:  [0 -1  2,  3  4 -1,  6 -1 -1]
                # tgt_cache_loc: [0  1   ,  3  4   ,  6      ]
                # to_free_slots: [      2,        5,     7  8]
                # to_free_slots also needs to be page-aligned without the first partial page
                #
                # split each row of out_cache_loc into two parts.
                # 1. the first part goes to tgt_cache_loc. length = accept_length[i] + 1
                # 2. the second part goes to to_free_slots.
                get_target_cache_loc[(bs,)](
                    tgt_cache_loc,
                    to_free_slots,
                    accept_length,
                    to_free_num_slots,
                    batch.out_cache_loc,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                    next_power_of_2(bs),
                )

                # Free the kv cache
                token_to_kv_pool_allocator.free(to_free_slots)

                # Copy the kv cache
                batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_cache_loc, src_cache_loc
                )

        accepted_lenghts_plus_bonus = accept_length + 1  # with bonus token
        accepted_by_req_gpu = verified_id.split(
            accepted_lenghts_plus_bonus.tolist()
        )  # tuple of bs tensors
        new_tokens_per_req = [t.cpu().tolist() for t in accepted_by_req_gpu]

        # Construct ModelFreeVerifyOutput
        if not has_finished:
            if page_size == 1 or self.topk == 1:
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc,
                    bs,
                )
            else:
                batch.out_cache_loc = tgt_cache_loc

            batch.seq_lens.add_(accepted_lenghts_plus_bonus)
            batch.seq_lens_cpu.add_(accept_length_cpu + 1)

            return ModelFreeVerifyOutput(
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=accept_length_list,
                accepted_indices=accept_index,
                new_accepted_tokens=new_tokens_per_req,
                finished_requests=finished_requests,
            )
        else:
            if page_size == 1 or self.topk == 1:
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc[accept_index],
                    bs,
                )
                batch.seq_lens.add_(accepted_lenghts_plus_bonus)
                batch.seq_lens_cpu.add_(accept_length_cpu + 1)

            if len(unfinished_accept_index) > 0:
                unfinished_accept_index = torch.cat(unfinished_accept_index)
                unfinished_index_device = torch.tensor(
                    unfinished_index, dtype=torch.int64, device=predict.device
                )
                draft_input_accept_length_cpu = [
                    accept_length_list[i] for i in unfinished_index
                ]
                if page_size == 1 or self.topk == 1:
                    batch.out_cache_loc = batch.out_cache_loc[unfinished_accept_index]
                else:
                    batch.out_cache_loc = torch.empty(
                        len(unfinished_index) + sum(draft_input_accept_length_cpu),
                        dtype=torch.int64,
                        device=predict.device,
                    )
                    accept_length_filter = create_accept_length_filter(
                        accept_length,
                        unfinished_index_device,
                        batch.seq_lens,
                    )
                    filter_finished_cache_loc_kernel[(bs,)](
                        batch.out_cache_loc,
                        tgt_cache_loc,
                        accept_length,
                        accept_length_filter,
                        next_power_of_2(bs),
                        next_power_of_2(self.draft_token_num),
                    )

            return ModelFreeVerifyOutput(
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=accept_length_list,
                accepted_indices=accept_index,
                new_accepted_tokens=new_tokens_per_req,
                finished_requests=finished_requests,
            )


@dataclass
class ModelFreeVerifyOutput:
    # Logit outputs from target worker
    logits_output: LogitsProcessorOutput
    # Accepted token ids including the bonus token
    verified_id: torch.Tensor
    # Accepted token length per sequence in a batch in CPU.
    accept_length_per_req_cpu: List[int]
    # Accepted indices from logits_output.next_token_logits
    accepted_indices: torch.Tensor
    # Tokens to add to speculator
    new_accepted_tokens: List[List[int]]
    # Finished requests, can be removed from speculator
    finished_requests: List[str]

    @classmethod
    def create_idle_input(cls, topk: int, spec_steps: int, num_verify_tokens: int):
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device="cuda"),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device="cuda"),
            positions=torch.empty((0,), dtype=torch.int64, device="cuda"),
            retrive_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrive_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrive_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrive_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
        )
