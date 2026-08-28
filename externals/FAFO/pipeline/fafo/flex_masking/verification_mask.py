from torch.nn.attention.flex_attention import _mask_mod_signature

def generating_verification_mask(seq_len, num_input, llookahead, lguess, guess_len) -> _mask_mod_signature:

    def verification_mask(b, h, q_idx, kv_idx):
        is_verification = (num_input <= q_idx) & (q_idx < num_input + lguess)
        is_verification_kv = (num_input <= kv_idx) & (kv_idx < num_input + lguess)
        is_prefill = kv_idx < num_input
        clipped_q_idx = q_idx - num_input
        clipped_kv_idx = kv_idx - num_input

        start_seq_idx = clipped_q_idx // guess_len * guess_len
        is_causal = (start_seq_idx <= clipped_kv_idx) & (clipped_kv_idx <= clipped_q_idx)

        return (is_verification & ((is_causal & is_verification_kv) | is_prefill))

    return verification_mask
