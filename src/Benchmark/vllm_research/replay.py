"""Báo khả năng replay cùng state của adapter vLLM hiện tại.

Stock vLLM 0.30 không cung cấp giao dịch fork/rollback worker. Việc gọi lại
LLM.generate hoặc collective_rpc không tự tạo snapshot hợp lệ. Module này
không giả lập replay bằng trajectory khác và không nhập CUDA trên CPU.
"""
from __future__ import annotations

from typing import Any


REQUIRED_REPLAY_PRIMITIVES = (
    'pause_before_draft_on_live_reference_request',
    'fork_target_and_draft_kv_including_partial_tail_page',
    'snapshot_request_input_batch_rng_and_proposer_scratch',
    'build_isolated_target_attention_metadata_and_slot_mapping',
    'verify_proposal_without_scheduler_commit',
    'rollback_and_repeat_local_full_integrity_check',
)


def describe_replay_support(version: str = '0.30.0') -> dict[str, Any]:
    """Không quảng cáo capability trước khi có implementation worker an toàn."""
    return {
        'status': 'unsupported',
        'vllm_version': version,
        'capabilities': ['same_prefix_replay', 'verification_prefix_replay'],
        'reason': (
            'Adapter hiện chưa có fork/rollback target và draft KV theo request. '
            'vLLM GPUModelRunner.execute_model cập nhật input_batch và ghi target KV; '
            'sample_tokens xóa ExecuteModelState, cập nhật request rồi gọi proposer; '
            'DFlashProposer.build_model_inputs_first_pass ghi context KV và forward ghi query KV. '
            'collective_rpc không cô lập hoặc hoàn tác các thay đổi ấy. '
            'A1/A2/A3 same-prefix replay, B1 gamma sweep và B1-V chưa thể được đánh dấu đã thực hiện. '
            'Native B0 full-generation trace là job riêng và vẫn chạy qua trace adapter.'
        ),
        'required_worker_primitives': list(REQUIRED_REPLAY_PRIMITIVES),
        'source_api': {
            'target_execution': 'vllm.v1.worker.gpu_model_runner.GPUModelRunner.execute_model',
            'sampling_commit': 'vllm.v1.worker.gpu_model_runner.GPUModelRunner.sample_tokens',
            'target_forward': 'vllm.v1.worker.gpu_model_runner.GPUModelRunner._model_forward',
            'draft_execution': 'vllm.v1.spec_decode.llm_base_proposer.SpecDecodeBaseProposer.propose',
            'draft_cache_write': 'vllm.v1.spec_decode.dflash.DFlashProposer.build_model_inputs_first_pass',
        },
        'generation_capability_unaffected': True,
    }
