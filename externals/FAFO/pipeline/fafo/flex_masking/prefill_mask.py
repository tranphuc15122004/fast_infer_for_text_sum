from torch.nn.attention.flex_attention import _mask_mod_signature

def generating_prefill_mask(num_input) -> _mask_mod_signature:

    def prefill_mask(b, h, q_idx, kv_idx):
        is_prefill = (q_idx < num_input) & (kv_idx < num_input)
        is_causal = kv_idx <= q_idx

        return (is_prefill & is_causal)

    return prefill_mask
