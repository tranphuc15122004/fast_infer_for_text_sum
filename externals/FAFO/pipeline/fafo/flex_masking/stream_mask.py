from torch.nn.attention.flex_attention import _mask_mod_signature

def generating_stream_mask(seq_len, num_input, llookahead, lguess, num_init, num_local) -> _mask_mod_signature:

    def stream(b, h, q_idx, kv_idx):
        is_kv = kv_idx >= seq_len
        is_generation = (num_input + lguess <= q_idx) & (q_idx < seq_len)
        is_stream = (kv_idx < seq_len + num_init + num_local)
        is_verification = (num_input <= q_idx) & (q_idx < num_input + lguess)
        is_prefill = (q_idx < num_input)

        return  (is_kv & (is_verification | (is_generation & is_stream) | is_prefill))

    return stream
