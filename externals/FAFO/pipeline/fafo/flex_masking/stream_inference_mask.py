import torch
from torch.nn.attention.flex_attention import (
    create_block_mask,
    or_masks
)

torch._dynamo.config.cache_size_limit = 2000
create_block_mask = torch.compile(create_block_mask)

from pipeline.fafo.flex_masking.generation_mask import generating_gen_mask
from pipeline.fafo.flex_masking.stream_mask import generating_stream_mask
from pipeline.fafo.flex_masking.verification_mask import generating_verification_mask
from pipeline.fafo.flex_masking.prefill_mask import generating_prefill_mask


def generate_stream_inference_mask(CONFIG_MAP):
    seq_len = (CONFIG_MAP['WINDOW_SIZE'] + CONFIG_MAP['GUESS_SET_SIZE']) * (CONFIG_MAP['LEVEL'] - 1)
    prefill_size = 1
    attn_size = CONFIG_MAP['WINDOW_SIZE'] * (CONFIG_MAP['LEVEL'] - 1) - 1
    lguess = CONFIG_MAP['GUESS_SET_SIZE'] * (CONFIG_MAP['LEVEL'] - 1)
    WINDOWS_SIZE = CONFIG_MAP['WINDOW_SIZE']
    gen_mask = generating_gen_mask(
        seq_len,
        prefill_size,
        attn_size,
        lguess,
        WINDOWS_SIZE,
        CONFIG_MAP['LEVEL'] - 1
    )
    verification_mask = generating_verification_mask(
        seq_len,
        prefill_size,
        attn_size,
        lguess,
        CONFIG_MAP['LEVEL'] - 1
    )
    num_init = CONFIG_MAP['config']['num_init']
    num_local = CONFIG_MAP['config']['num_local']
    stream_mask = generating_stream_mask(
        seq_len,
        prefill_size,
        attn_size,
        lguess,
        num_init,
        num_local
    )
    prefill_mask = generating_prefill_mask(
        prefill_size
    )
    decoding_mask = or_masks(gen_mask, verification_mask, stream_mask, prefill_mask)
    masks = []

    for kv_len in range(128, 6580, 128):
        mask = create_block_mask(decoding_mask, 1, 1, seq_len, kv_len, _compile=True)
        masks.append(mask)

    return masks
