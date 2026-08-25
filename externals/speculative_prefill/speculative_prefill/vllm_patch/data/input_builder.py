from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.sequence import SequenceGroupMetadata
from vllm.worker.model_runner import ModelInputForGPUBuilder

from speculative_prefill.vllm_patch.data.sequence import AugmentedSequenceData


class AugmentedModelInputForGPUBuilder(ModelInputForGPUBuilder):
    def _compute_lens(
        self, 
        inter_data: ModelInputForGPUBuilder.InterDataForSeqGroup, 
        seq_idx: int, 
        seq_group_metadata: SequenceGroupMetadata
    ):
        """Compute context length, sequence length and tokens
        for the given sequence data.
        """
        seq_data = seq_group_metadata.seq_data[inter_data.seq_ids[seq_idx]]
        token_chunk_size = seq_group_metadata.token_chunk_size

        # Compute context length (the number of tokens that are
        # already computed) and sequence length (total number of tokens).
        assert isinstance(seq_data, AugmentedSequenceData)
        assert not self.runner.scheduler_config.is_multi_step
        assert not self.runner.model_config.is_encoder_decoder_model

        seq_len = seq_data.get_len()
        if inter_data.is_prompt:
            context_len = seq_data.get_num_computed_tokens()
            seq_len = min(seq_len, context_len + token_chunk_size)
        else:
            context_len = seq_data.get_num_computed_tokens()

        # Compute tokens.
        tokens = seq_data.get_token_ids()[context_len:seq_len]
        inter_data.input_tokens[seq_idx].extend(tokens)

        # first calculate pos ids
        if seq_data._position_ids is not None:
            inter_data.input_positions[seq_idx].extend(seq_data._position_ids) 
        else:
            inter_data.input_positions[seq_idx].extend(range(context_len, seq_len))

        # correct seq_len and context_len for decode
        if not inter_data.is_prompt:
            context_len = seq_data._context_len
            seq_len = context_len + 1

        inter_data.seq_lens[seq_idx] = seq_len
        inter_data.orig_seq_lens[seq_idx] = seq_len
        inter_data.context_lens[seq_idx] = context_len
        inter_data.query_lens[seq_idx] = seq_len - context_len

        if seq_data.mrope_position_delta is not None:
            if inter_data.mrope_input_positions is None:
                inter_data.mrope_input_positions = [None] * inter_data.n_seqs

            inter_data.mrope_input_positions[
                seq_idx] = MRotaryEmbedding.get_next_input_positions(
                    seq_data.mrope_position_delta,
                    context_len,
                    seq_len,
                )