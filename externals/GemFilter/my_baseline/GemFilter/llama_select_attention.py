# Compatible with both the legacy (4.43) and modern (4.57+) attention APIs.
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .gem_filter_utils import find_context


class LlamaSelectAttention(LlamaAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Transformers 4.57+ keeps these values on the config rather than on
        # each attention module; the vendored GemFilter forward still uses the
        # legacy attribute names.
        self.hidden_size = getattr(self, "hidden_size", self.config.hidden_size)
        self.num_heads = getattr(
            self, "num_heads", self.config.num_attention_heads
        )
        self.num_key_value_heads = getattr(
            self, "num_key_value_heads", self.config.num_key_value_heads
        )
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        self.reset()
        self.topk = 1024
        self.select_layer_idx = 13
        self.select_mode = False

    def reset(self):
        self.indecies = None
        return

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        # will become mandatory in v4.45
        position_embeddings: Optional[Tuple[torch.Tensor,
                                            torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if past_key_values is not None:
            past_key_value = past_key_values
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        # [GemFilter] update below
        if self.select_mode:
            self.reset()
            find_context(self, query_states, key_states)

        if not self.select_mode and past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        # [GemFilter] update above
        
        attn_output, attn_weights = self.flash_softmax(
            query_states, key_states, value_states, attention_mask, q_len, position_ids)
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None

        if getattr(self, "_modern_transformers_attention_api", False):
            return attn_output, attn_weights
        return attn_output, attn_weights, past_key_value

    def flash_softmax(self, query_states, key_states, value_states, attention_mask, q_len, position_ids):
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            query_states = query_states.to(torch.float16)
            key_states = key_states.to(torch.float16)
            value_states = value_states.to(torch.float16)

        try:
            import flash_attn  # noqa: F401
            has_flash_attn = True
        except Exception:
            has_flash_attn = False

        if has_flash_attn:
            try:
                attn_output = _flash_attention_forward(
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    q_len,
                    dropout=0.0,
                    sliding_window=getattr(self, "sliding_window", None),
                    use_top_left_mask=self._flash_attn_uses_top_left_mask,
                    is_causal=True,
                )
            except (TypeError, NameError, ImportError, RuntimeError):
                has_flash_attn = False

        if not has_flash_attn:
            # Fallback without flash-attn: torch SDPA.  This is also used by
            # CPU smoke runs and by the server when flash-attn is unavailable.
            mask = attention_mask
            if mask is not None:
                mask = mask[:, :, :, : key_states.shape[1]]  # seq dim (S in B,S,H,D)
            # GQA: SDPA needs kv heads == query heads (flash-attn broadcasts).
            if key_states.shape[2] != query_states.shape[2]:
                kv = key_states.repeat_interleave(
                    query_states.shape[2] // key_states.shape[2], dim=2)
                vv = value_states.repeat_interleave(
                    query_states.shape[2] // value_states.shape[2], dim=2)
            else:
                kv, vv = key_states, value_states
            attn_output = F.scaled_dot_product_attention(
                query_states.transpose(1, 2),
                kv.transpose(1, 2),
                vv.transpose(1, 2),
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=mask is None,
            )
            attn_output = attn_output.transpose(1, 2)
        if input_dtype == torch.float32:
            attn_output = attn_output.to(torch.float32)
        return attn_output, None

    
