import os
from typing import Dict, List, Tuple

import torch
from vllm.config import ModelConfig
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.worker.model_runner import ModelRunner
from vllm.worker.worker import Worker
from vllm.worker.worker_base import LoraNotSupportedWorkerBase

from speculative_prefill.vllm_patch.config import get_spec_config
from speculative_prefill.vllm_patch.data.input_builder import \
    AugmentedModelInputForGPUBuilder
from speculative_prefill.vllm_patch.data.sequence import AugmentedSequenceData
from speculative_prefill.vllm_patch.worker.look_ahead_spec_worker import \
    LookAheadSpecWorker


def split_num_cache_blocks_evenly(
    base_model_cache_block_size_bytes: int,
    spec_model_cache_block_size_bytes: int,
    total_num_gpu_blocks: int
) -> int:
    new_num_gpu_blocks = int(
        total_num_gpu_blocks * base_model_cache_block_size_bytes /
        (spec_model_cache_block_size_bytes + base_model_cache_block_size_bytes))

    return new_num_gpu_blocks


def create_spec_worker(*args, **kwargs) -> "SpecPrefillWorker":
    spec_model_name = os.environ.get("SPEC_MODEL", None)
    assert spec_model_name is not None

    assert kwargs["scheduler_config"].chunked_prefill_enabled == False, \
        "Please set --enable-chunked-prefill=False or enable_chunked_prefill=False. "

    # create the base model
    kwargs["model_runner_cls"] = ModelRunner
    model_config: ModelConfig = kwargs['model_config']
    base_model_worker = Worker(*args, **kwargs) 

    # create the spec prefill model
    # TODO: deal with this later in a proper way
    spec_kwargs = kwargs.copy()
    spec_kwargs["model_config"] = ModelConfig(
        model=spec_model_name, 
        tokenizer=model_config.tokenizer,
        tokenizer_mode=model_config.tokenizer_mode,
        trust_remote_code=model_config.trust_remote_code,
        dtype=model_config.dtype,
        seed=model_config.seed,
        revision=model_config.revision,
        code_revision=model_config.code_revision,
        rope_scaling=model_config.rope_scaling,
        rope_theta=model_config.rope_theta,
        tokenizer_revision=model_config.tokenizer_revision,
        max_model_len=model_config.max_model_len,
        # explicitly set quantization to null
        quantization=None,
        quantization_param_path=None,
        enforce_eager=model_config.enforce_eager,
        max_seq_len_to_capture=model_config.max_seq_len_to_capture,
        max_logprobs=model_config.max_logprobs,
        disable_sliding_window=model_config.disable_sliding_window,
        skip_tokenizer_init=model_config.skip_tokenizer_init,
        served_model_name=model_config.served_model_name,
        use_async_output_proc=model_config.use_async_output_proc,
        override_neuron_config=model_config.override_neuron_config
    )
    spec_model_worker = LookAheadSpecWorker(*args, **spec_kwargs)

    return SpecPrefillWorker(
        base_model_worker=base_model_worker, 
        spec_model_worker=spec_model_worker
    )


class SpecPrefillWorker(LoraNotSupportedWorkerBase):
    def __init__(
        self,
        base_model_worker: Worker, 
        spec_model_worker: LookAheadSpecWorker, 
    ):
        self.base_model_worker = base_model_worker
        self.spec_model_worker = spec_model_worker
        self.spec_config = get_spec_config()

        self.base_model_worker.model_runner._builder_cls = \
            AugmentedModelInputForGPUBuilder
        
        self.id_to_context_len: Dict[str, int] = {}      
    
    def init_device(self) -> None:
        # The base worker model is initialized first in case the spec
        # model has a smaller TP degree than the base worker.
        self.base_model_worker.init_device()
        self.spec_model_worker.init_device()

        self.base_model_worker.load_model()
        self.spec_model_worker.load_model()

    def load_model(self, *args, **kwargs):
        pass

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        num_gpu_blocks, num_cpu_blocks = (
            self.base_model_worker.determine_num_available_blocks())

        base_model_cache_block_size_bytes = (
            self.base_model_worker.get_cache_block_size_bytes())
        spec_model_cache_block_size_bytes = (
            self.spec_model_worker.get_cache_block_size_bytes())

        new_num_gpu_blocks = split_num_cache_blocks_evenly(
            base_model_cache_block_size_bytes, 
            spec_model_cache_block_size_bytes, 
            num_gpu_blocks
        )
        
        return new_num_gpu_blocks, num_cpu_blocks

    def initialize_cache(
        self, num_gpu_blocks: int,
        num_cpu_blocks: int
    ) -> None:
        self.base_model_worker.initialize_cache(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks)
        self.spec_model_worker.initialize_cache(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks)
    
    @torch.inference_mode
    def execute_model(
        self, 
        execute_model_req: ExecuteModelRequest | None = None
    ) -> List[SamplerOutput] | None:
        if self.rank != self._driver_rank:
            self._run_non_driver_rank()
            return []
        
        if execute_model_req is None:
            # we finished all things
            broadcast_tensor_dict({}, src=self._driver_rank)
            return []

        has_prefill = any([sgm.is_prompt for sgm in execute_model_req.seq_group_metadata_list])

        broadcast_tensor_dict({
            "has_prefill": has_prefill
        }, src=self._driver_rank)

        if has_prefill:
            execute_model_req = self.spec_model_worker.speculate(execute_model_req)
        
        execute_model_req = self._record_and_update_requests(execute_model_req)

        return self.base_model_worker.execute_model(execute_model_req)

    @torch.inference_mode()
    def start_worker_execution_loop(self) -> None:
        while self._run_non_driver_rank():
            pass

    def _run_non_driver_rank(self) -> bool:
        assert self.rank != self._driver_rank

        data = broadcast_tensor_dict(src=self._driver_rank)
        if not data:
            # finished everything
            return False

        if data['has_prefill']:
            self.spec_model_worker.speculate()

        self.base_model_worker.execute_model()

        return True

    def _record_and_update_requests(
        self, 
        execute_model_req: ExecuteModelRequest
    ) -> ExecuteModelRequest:
        for metadata in execute_model_req.seq_group_metadata_list:
            assert len(metadata.seq_data) == 1
            request_id = metadata.request_id
            seq_id = metadata.get_first_seq_id()
            id = f"{request_id}_{seq_id}"
            seq_data: AugmentedSequenceData = metadata.seq_data[seq_id]

            if metadata.is_prompt:    
                self.id_to_context_len[id] = seq_data.get_prompt_len()
            else:
                seq_data._context_len = self.id_to_context_len[id]
                metadata.seq_data[seq_id] = seq_data
                # we decode every time
                self.id_to_context_len[id] += 1

        return execute_model_req

    def get_cache_block_size_bytes(self) -> int:
        raise NotImplementedError

    @property
    def rank(self):
        return self.base_model_worker.rank

    @property
    def device(self):
        return self.base_model_worker.device

    @property
    def _driver_rank(self) -> int:
        return 0