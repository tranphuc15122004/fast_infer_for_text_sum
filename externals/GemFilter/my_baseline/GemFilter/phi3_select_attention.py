# Compatible with both the legacy (4.43) and modern (4.57+) attention APIs.
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.phi3.modeling_phi3 import Phi3Attention, apply_rotary_pos_emb, repeat_kv
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .gem_filter_utils import find_context


class Phi3SelectAttention(Phi3Attention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        self.select_layer_idx = 19
        self.select_mode = False

    def reset(self):
        self.indecies = None
        return

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if past_key_values is not None:
            past_key_value = past_key_values
        # Phi3FlashAttention2 attention does not support output_attentions

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos: query_pos +
                         self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos +
                           self.num_key_value_heads * self.head_dim:]

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if hasattr(past_key_value, "get_usable_length"):
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
            else:
                kv_seq_len += past_key_value.get_seq_length(self.layer_idx)

        if position_embeddings is None:
            # Legacy Phi-3 rotary embeddings accepted an explicit sequence
            # length.  Modern Transformers computes and passes the pair from
            # Phi3Model, whose rotary embedding no longer accepts seq_len.
            rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
            cos, sin = self.rotary_emb(
                value_states, position_ids, seq_len=rotary_seq_len)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids)
        
        # [GemFilter] update below
        if self.select_mode:
            self.reset()
            find_context(self, query_states, key_states)

        if not self.select_mode and past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(
                self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            # Specific to RoPE models
            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)
        # [GemFilter] update above
        
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output, attn_weights = self.flash_softmax(
            query_states, key_states, value_states, attention_mask, q_len, position_ids)
        attn_output = attn_output.reshape(
            bsz, q_len, self.hidden_size).contiguous()

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
            except (TypeError, ImportError, RuntimeError):
                has_flash_attn = False

        if not has_flash_attn:
            mask = attention_mask
            if mask is not None:
                mask = mask[:, :, :, : key_states.shape[1]]
            attn_output = F.scaled_dot_product_attention(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=mask is None,
            ).transpose(1, 2)
        if input_dtype == torch.float32:
            attn_output = attn_output.to(torch.float32)
        return attn_output, None
