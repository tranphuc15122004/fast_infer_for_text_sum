from speculative_prefill.vllm_patch.data.sequence import AugmentedSequenceData


def patch_data():
    from vllm import sequence
    sequence.SequenceData = AugmentedSequenceData