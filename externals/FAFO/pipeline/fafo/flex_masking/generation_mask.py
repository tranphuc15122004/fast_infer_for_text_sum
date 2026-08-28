from torch.nn.attention.flex_attention import _mask_mod_signature

def generating_gen_mask(seq_len, num_input, llookahead, lguess, window, gen_len) -> _mask_mod_signature:

    def gen_mask(b, h, q_idx, kv_idx):
        is_generation = (num_input + lguess <= q_idx) & (q_idx < seq_len) 
        is_gen_kv = (num_input + lguess <= kv_idx) & (kv_idx < seq_len)
        is_prefill = kv_idx < num_input

        distance = q_idx - kv_idx
        is_same_subsequence = (distance >= 0) & (distance % window == 0) # also accounts for diagonal masking

        subsequence_idx = (q_idx - lguess) % window + lguess
        beginning_causal = (num_input + lguess <= kv_idx) & (kv_idx < num_input + lguess + window - 1) & (kv_idx <= subsequence_idx)


        return (is_generation & (is_prefill | (is_gen_kv & (is_same_subsequence | beginning_causal))))

    return gen_mask
