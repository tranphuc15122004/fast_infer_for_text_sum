from array import array
from typing import Optional
from typing import Sequence as GenericSequence

from vllm.sequence import SequenceData

VLLM_TOKEN_ID_ARRAY_TYPE = "l"

class AugmentedSequenceData(SequenceData):
    _position_ids: GenericSequence[int] = None
    _context_len: int = -1

    @staticmethod
    def from_seqs_and_pos_ids(
        prompt_token_ids: GenericSequence[int], 
        position_ids: GenericSequence[int], 
        output_token_ids: Optional[GenericSequence[int]] = None
    ) -> "AugmentedSequenceData":
        """
        Construct a :class:`SequenceData` instance from prompt and output
        token sequences.
        """
        prompt_token_ids_arr = array(VLLM_TOKEN_ID_ARRAY_TYPE, 
                                     prompt_token_ids)
                
        assert len(prompt_token_ids) == len(position_ids)

        if output_token_ids is None:
            return AugmentedSequenceData(
                prompt_token_ids_arr, 
                _position_ids=position_ids
            )

        output_token_ids_arr = array(VLLM_TOKEN_ID_ARRAY_TYPE,
                                     output_token_ids)
        return AugmentedSequenceData(
            prompt_token_ids_arr, 
            _position_ids=position_ids, 
            _output_token_ids=output_token_ids_arr
        )