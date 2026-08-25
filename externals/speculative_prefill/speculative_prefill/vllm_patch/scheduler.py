import time
from collections import deque
from typing import Deque, List, Optional, Set, Tuple

from vllm.core import scheduler
from vllm.core.interfaces import AllocStatus
from vllm.core.scheduler import (ScheduledSequenceGroup,
                                 SchedulerPrefillOutputs, SchedulingBudget)
from vllm.logger import init_logger
from vllm.sequence import SequenceGroup, SequenceStatus

from speculative_prefill.vllm_patch.config import get_spec_config

logger = init_logger(__name__)

def _schedule_prefills(
        self: scheduler.Scheduler,
        budget: SchedulingBudget,
        curr_loras: Optional[Set[int]],
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        """Schedule sequence groups that are in prefill stage.

        Note that the current scheduler treats PREEMPTED_FOR_RECOMPUTE
        as a new prefill (that starts from beginning -> most recently generated
        tokens).

        It schedules waiting requests as long as it fits `budget` and
        curr_loras <= max_lora from the scheduling config. The input arguments
        `budget` and `curr_loras` are updated based on scheduled seq_groups.

        Args:
            budget: The scheduling budget. The argument is in-place updated
                when any requests are scheduled.
            curr_loras: Currently batched lora request ids. The argument is
                in-place updated when any requests are scheduled.
            enable_chunking: If True, seq group can be chunked and only a
                chunked number of tokens are scheduled  if
                `budget.num_batched_tokens` has not enough capacity to schedule
                all tokens.

        Returns:
            SchedulerPrefillOutputs.
        """
        ignored_seq_groups: List[SequenceGroup] = []
        seq_groups: List[ScheduledSequenceGroup] = []

        waiting_queue = self.waiting

        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self._get_prompt_limit(seq_group)
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            # num_lookahead_slots: int = 0
            # if self.scheduler_config.is_multi_step and enable_chunking:
            #     num_lookahead_slots = self._get_num_lookahead_slots(
            #         True, enable_chunking)

            num_lookahead_slots = self._get_num_lookahead_slots(
                True, enable_chunking)
            
            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(
                seq_group, num_lookahead_slots=num_lookahead_slots)
            
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) + lookahead slots (%d) is "
                    "too long and exceeds the capacity of block_manager",
                    num_new_tokens, num_lookahead_slots)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                waiting_queue.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    waiting_queue.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            blocks_to_copy: List[Tuple[int, int]] = []
            self._append_slots(seq_group, blocks_to_copy, enable_chunking)

            # if enable_chunking and self.scheduler_config.is_multi_step:
            #     blocks_to_copy: List[Tuple[int, int]] = []
            #     # init_multi_step_from_lookahead_slots happens in append_slots
            #     self._append_slots(seq_group, blocks_to_copy, enable_chunking)
            #     # This assert will trip when a copy-on-write happens. This is
            #     # not a concern as the very first sequence-group block
            #     # allocation happens above. Still, we have the assert to
            #     # catch any edge-cases.
            #     assert not blocks_to_copy
            # else:
            #     seq_group.init_multi_step_from_lookahead_slots(
            #         num_lookahead_slots,
            #         num_scheduler_steps=self.scheduler_config.
            #         num_scheduler_steps,
            #         is_multi_step=self.scheduler_config.is_multi_step,
            #         enable_chunking=enable_chunking)

            seq_groups.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Queue requests that couldn't be scheduled.
        waiting_queue.extendleft(leftover_waiting_sequences)
        if len(seq_groups) > 0:
            self.prev_prompt = True

        return SchedulerPrefillOutputs(
            seq_groups=seq_groups,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=self._get_num_lookahead_slots(
                is_prefill=True, enable_chunking=enable_chunking))


def _get_num_lookahead_slots(
    self: scheduler.Scheduler,
    is_prefill: bool,
    enable_chunking: bool
) -> int:
    assert not enable_chunking
    assert not self.scheduler_config.is_multi_step

    if is_prefill:
        return get_spec_config().look_ahead_cnt
    else:
        return self.scheduler_config.num_lookahead_slots


def patch_scheduler():
    scheduler.Scheduler._get_num_lookahead_slots = _get_num_lookahead_slots
    scheduler.Scheduler._schedule_prefills = _schedule_prefills