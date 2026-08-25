from speculative_prefill.vllm_patch.executor.gpu_executor import (
    PatchedGPUExecutor, PatchedGPUExecutorAsync)


def patch_executor():
    from vllm.executor import gpu_executor
    gpu_executor.GPUExecutor = PatchedGPUExecutor
    gpu_executor.GPUExecutorAsync = PatchedGPUExecutorAsync