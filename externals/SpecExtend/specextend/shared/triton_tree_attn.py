import math
import torch

__all__ = ["tree_part_fwd_target"]


def _tree_part_fwd_target(query_states, key_states, value_states, tree_mask,
                  cache_lens, prefix_lse, bsz, q_len, num_heads, hidden_dim):
    tree_mask = tree_mask[:, :, -q_len:, -q_len:]
    tree_mask = (tree_mask == 0).to(torch.int8) # convert to 1 and 0

    softmax_scale = 1. / math.sqrt(hidden_dim)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.permute(0, 2, 3, 1)
    value_states = value_states.transpose(1, 2)
    
    attn_score = torch.matmul(query_states, key_states) * softmax_scale

    attn_score = attn_score.to(torch.float32)
    attn_score_tree_mask = tree_mask.expand(-1, num_heads, -1, -1)
    attn_score = attn_score.masked_fill(attn_score_tree_mask == 0, -float('inf'))
    attn_weight = torch.softmax(attn_score, dim=-1).to(query_states.dtype)
    current_out = torch.matmul(attn_weight, value_states).permute(0, 2, 1, 3)
    current_lse = attn_score.logsumexp(dim=-1, keepdim=True).transpose(1, 2)

    prefix_lse = prefix_lse.view(bsz, num_heads, q_len, -1).transpose(1, 2)

    weight = torch.nn.functional.sigmoid(prefix_lse - current_lse).to(query_states.dtype)
    return current_out, weight


def _supports_compiled_tree_attention() -> bool:
    """Torch.compile/Triton path is only valid on Ampere+ GPUs.

    The upstream decorator compiles unconditionally.  On a Tesla T4 this
    either selects an unsupported Triton kernel or fails while compiling the
    Python 3.11 extension.  Keep the exact eager implementation available for
    sm75 and CPU; use compilation only where the upstream kernel is supported.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability()[0] >= 8
    except (RuntimeError, AssertionError, IndexError):
        return False


if _supports_compiled_tree_attention():
    tree_part_fwd_target = torch.compile(_tree_part_fwd_target)
else:
    tree_part_fwd_target = _tree_part_fwd_target
