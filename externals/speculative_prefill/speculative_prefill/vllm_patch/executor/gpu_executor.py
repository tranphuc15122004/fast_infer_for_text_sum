import os
from typing import Callable, List, Optional, Tuple, Type, Union

from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutor
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.utils import make_async
from vllm.worker.worker_base import WorkerBase


class PatchedGPUExecutor(GPUExecutor):
    def _get_worker_module_and_class(self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        """
            Patching this logic to consider using prefill speculator
        """
        worker_class_fn = None
        if os.environ.get("SPEC_MODEL", None):
            worker_module_name = "speculative_prefill.vllm_patch.worker.spec_prefill_worker"
            worker_class_name = "create_spec_worker"
        elif self.scheduler_config.is_multi_step:
            worker_module_name = "vllm.worker.multi_step_worker"
            worker_class_name = "MultiStepWorker"
        elif self.speculative_config:
            worker_module_name = "vllm.spec_decode.spec_decode_worker"
            worker_class_name = "create_spec_worker"
        else:
            worker_module_name = "vllm.worker.worker"
            worker_class_name = "Worker"
        return (worker_module_name, worker_class_name, worker_class_fn)
    

class PatchedGPUExecutorAsync(PatchedGPUExecutor, ExecutorAsyncBase):
    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req)
        return output