# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Qwen2 model."""
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import Qwen2Config
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import (
    PreTrainedModel,
)
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
)

from transformers.utils import (
    add_start_docstrings,
    is_flash_attn_2_available,
    logging,
)

from pipeline.fafo.kv_cache_managers.utils import KV_CACHE_MANAGERS

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )


logger = logging.get_logger(__name__)


_CONFIG_FOR_DOC = "Qwen2Config"


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    warnings.warn(
        "Calling `transformers.models.llama.modeling_llama._prepare_4d_attention_mask` is deprecated and will be removed in v4.37. Use `transformers.modeling_attn_mask_utils._prepare_4d_attention_mask"
    )
    return _prepare_4d_attention_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

def j_make_causal_mask_multilevel(
    level_sizes: list,is_prefill:bool, WINDOW_SIZE: int, guess : list, guess_size: int, not_seq:bool, continue_all:bool,input_ids_shape: torch.Size, dtype: torch.dtype, la_mask_offset:int, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)

    if is_prefill:
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        assert past_key_values_length == 0
        assert guess is None
        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    
    tiny_mask_size = level_sizes[-1] # + 1
    mask_cond = torch.arange(tiny_mask_size, device=device)
    hm = mask_cond < (mask_cond + 1).view(tiny_mask_size, 1)
    
    level_offset = tgt_len - (sum(level_sizes) + 1) - (len(guess) if guess is not None else 0) #offset when you guess multiple tokens and not copy kv-cache 
    dist_offset = (1 + level_sizes[0] - level_sizes[-1]) #offset for distributed inference 
    assert dist_offset>=0 and level_offset>=0
    if guess is not None:
        
        lguess = len(guess)
        if guess_size == 2:
            small_m = torch.tensor([0, torch.finfo(dtype).min]).repeat(lguess // 2)[:-1]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m, -1)
        elif guess_size == 3:
            small_m1 = torch.tensor([0, 0, torch.finfo(dtype).min]).repeat(lguess // 3)[:-1]
            small_m2 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 3)[:-2]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2)
        elif guess_size == 4:
            small_m1 = torch.tensor([0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 4)[:-1]
            small_m2 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 4)[:-2]
            small_m3 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 4)[:-3]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3)
        elif guess_size == 5:
            small_m1 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 5)[:-1]
            small_m2 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-2]
            small_m3 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-3]
            small_m4 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 5)[:-4]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4)
        elif guess_size == 6:
            small_m1 = torch.tensor([0, 0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 6)[:-1]
            small_m2 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-2]
            small_m3 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-3]
            small_m4 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-4]
            small_m5 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 6)[:-5]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4).diagonal_scatter(small_m5, -5)
        elif guess_size == 7:
            small_m1 = torch.tensor([0, 0, 0, 0, 0, 0, torch.finfo(dtype).min]).repeat(lguess // 7)[:-1]
            small_m2 = torch.tensor([0, 0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-2]
            small_m3 = torch.tensor([0, 0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-3]
            small_m4 = torch.tensor([0, 0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-4]
            small_m5 = torch.tensor([0, 0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-5]
            small_m6 = torch.tensor([0, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min, torch.finfo(dtype).min]).repeat(lguess // 7)[:-6]
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0).diagonal_scatter(small_m1, -1).diagonal_scatter(small_m2, -2).diagonal_scatter(small_m3, -3).diagonal_scatter(small_m4, -4).diagonal_scatter(small_m5, -5).diagonal_scatter(small_m6, -6)

        else:
            mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].fill_diagonal_(0)
            for i in range(guess_size - 1): #7 : 0 - 5
                small_l = [0] * (guess_size - i - 1)  + [torch.finfo(dtype).min] * (i + 1)
                small_m = torch.tensor(small_l).repeat(lguess // guess_size)[:-1 - i]
                mask[-lguess:,-lguess:] = mask[-lguess:,-lguess:].diagonal_scatter(small_m, -1 - i)

        mask[-lguess:,:level_offset + 1] = 0   
    else:
        lguess = 0

    all_offset = level_offset + dist_offset
    if all_offset > 0:
        mask_offset = torch.arange(all_offset, device=device)
        moff = mask_offset < (mask_offset + 1).view(all_offset, 1)
        mask[:all_offset, :all_offset].masked_fill_(moff, 0) #fill level_offset + dist_offset with causal mask

        #
        mask[all_offset:-lguess+mask.size(0),:all_offset] = 0

    for ll in range(len(level_sizes)):
        if ll > 0:
            assert level_sizes[ll] == tiny_mask_size
        mask[all_offset+tiny_mask_size*ll:all_offset+tiny_mask_size*(ll+1),all_offset+tiny_mask_size*max(0,ll-1):all_offset+tiny_mask_size*max(1,ll)].masked_fill_(hm, 0)
        for row in range(0, ll + 1):
            mask[all_offset+tiny_mask_size*ll:all_offset+tiny_mask_size*(ll+1),all_offset+tiny_mask_size*row:all_offset+tiny_mask_size*(row+1)].fill_diagonal_(0)

    if past_key_values_length > 0 or la_mask_offset > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length + la_mask_offset, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length + la_mask_offset)


class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()

    def init_kv_cache(self, config):
        self.kv_cache_manager = KV_CACHE_MANAGERS[config['kv_cache_method']](config)

    def _init_rope(self):
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
        _use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
        decoding_mask=None,
        flex_attention=None,
        lookahead: List = [0, 0, 0, 0, 0, 0, 0], # This is not useful
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            if decoding_mask is not None:
                n_new_tokens = self.kv_cache_manager.update(
                    kv=past_key_value,
                    query=query_states
                )
                self.kv_cache_manager.advance_ptr(
                    n_new_tokens=n_new_tokens,
                    kv_end=past_key_value[0].shape[-2]
                )

                B, H, D = key_states.shape[0], key_states.shape[1], key_states.shape[3]
                PAD = 10
                zero_padding = torch.zeros((B, H, PAD, D), dtype=key_states.dtype, device=key_states.device)
                key_states = torch.cat([key_states, past_key_value[0], zero_padding], dim=2)
                value_states = torch.cat([value_states, past_key_value[1], zero_padding], dim=2)
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            self.kv_cache_manager.reset()

        past_key_value = (key_states, value_states) if _use_cache else None

        sliding_window = None
        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if decoding_mask is not None:
            mask = decoding_mask[key_states.size(2) // 128]
            attn_output = flex_attention(query_states, key_states, value_states, block_mask = mask)

            if hasattr(self.kv_cache_manager, 'accumulate_verify_logits'):
                self.kv_cache_manager.accumulate_verify_logits(
                    query_states, key_states, query_states.shape[-2]
                )

        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self._flash_attn_2_enabled = config._attn_implementation == "flash_attention_2"
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

    def init_kv_cache(self, config):
        self.self_attn.init_kv_cache(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        _use_cache: Optional[bool] = False,
        use_flash: bool=False,
        cache_position: Optional[torch.LongTensor] = None,
        lookahead: List=[0,0,0,0,0,0,0],
        is_prefill: bool=False,
        decoding_mask=None,
        flex_attention=None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        if not use_flash and not self._flash_attn_2_enabled:
            hidden_states, self_attn_weights, present_key_value = self.self_attn.forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                _use_cache=_use_cache,
                decoding_mask=decoding_mask,
                flex_attention=flex_attention,
                lookahead=None,
            )
        else:
            hidden_states, self_attn_weights, present_key_value = self.self_attn.forward_flash(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                _use_cache=_use_cache,
                lookahead=lookahead,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if _use_cache:
            outputs += (present_key_value,)


        return outputs


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen2RMSNorm):
            module.weight.data.fill_(1.0)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len + 200].to(dtype=x.dtype),
            self.sin_cached[:seq_len + 200].to(dtype=x.dtype),
        )


QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length) or `BlockMask`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If the model is configured to use flex_attention, it will attempt to convert the mask Tensor into a BlockMask,
            but you can also pass a `BlockMask` object directly here.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `_use_cache=True` or `config._use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        _use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def init_kv_cache(self, config):
        for layer in self.layers:
            layer.init_kv_cache(config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask


    def j_prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length, others):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        WINDOW_SIZE, is_prefill, guess, guess_size, not_seq, continue_all, la_mask_offset, level_sizes = others
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = j_make_causal_mask_multilevel(
                level_sizes,
                is_prefill,            
                WINDOW_SIZE,
                guess,
                guess_size,
                not_seq,
                continue_all,
                input_shape,
                inputs_embeds.dtype,
                la_mask_offset=la_mask_offset,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def Qwen2Modeljforward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        _use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        WINDOWS_SIZE: int=0,
        is_prefill: bool=False,
        level_sizes: Optional[List[int]] =None,
        guess_size: int=2,
        la_mask_offset: int=0,
        not_seq: bool=False,
        continue_all: bool=False,
        guess: Optional[torch.Tensor] = None,
        decoding_mask = None,
        flex_attention = None,
        use_flash: bool=False,
        **flash_attn_kwargs,
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        _use_cache = _use_cache if _use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if self.gradient_checkpointing and self.training and _use_cache:
            logger.warning_once(
                "`_use_cache=True` is incompatible with gradient checkpointing. Setting `_use_cache=False`."
            )
            _use_cache = False


        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
            padding_mask = None
        else:
            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None

        if use_flash:
            attention_mask = None
        else:
            if decoding_mask is None:
                attention_mask = self.j_prepare_decoder_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length, (WINDOWS_SIZE, is_prefill, guess, guess_size, not_seq, continue_all, la_mask_offset, level_sizes), 
                )
            else:
                attention_mask = None

        level_offset = seq_length - (sum(level_sizes) + 1) - (len(guess) if guess is not None else 0) #offset when you guess multiple tokens and not copy kv-cache 
        dist_offset = (1 + level_sizes[0] - level_sizes[-1]) #offset for distributed inference 

        lookahead = [level_sizes[-1], len(level_sizes) + 1, len(guess) // guess_size if guess is not None else 0,0,level_offset+dist_offset,level_offset,0]


        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if _use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    _use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    _use_cache=_use_cache,
                    use_flash=use_flash,
                    lookahead=lookahead,
                    is_prefill=is_prefill,
                    decoding_mask=decoding_mask,
                    flex_attention=flex_attention
                )

            hidden_states = layer_outputs[0]

            if _use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if _use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def init_kv_cache(self, config):
        self.model.init_kv_cache(config)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        _use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            _use_cache=_use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def jforward_multilevel(
        self,
        input_ids: torch.LongTensor = None,
        past_tokens: Optional[List[int]] = None,
        guess_tokens: Optional[List[int]] = None,
        guess_size: int = 2,
        not_seq: bool = False,
        continue_all: bool=False,
        level: int = 3,
        fill_level: int=-1,
        WINDOWS_SIZE: int=-1,
        dist_workers: int=-1,
        local_rank: int=-1,
        la_mask_offset: int=0,
        use_flash: bool=False, 
        decoding_mask=None,
        flex_attention=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        _use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if use_flash:
            assert flash_attn_lookahead_available
            swap_axis_for_flash = use_flash  and past_tokens[1] is not None
        else:
            swap_axis_for_flash = False 
        
        assert labels is None, " Inference Mode "
        assert input_ids.size(0) == 1, " single batch only "
        if level is not None:
            assert level == len(past_tokens) + 1
        
        if past_key_values is not None:
            past_size = past_key_values[0][0].size(2)
        else:
            past_size = 0
        seq_len_with_past = input_ids.size(1) + past_size

        prefill_size = input_ids.size(1) 
        for layer in self.model.layers:
            layer.self_attn.cur_len = prefill_size
        
        level_sizes = []

        assert continue_all == False
        lst_id = position_ids[0][-1].item()

        all_past = []
        ids_list = []
        attn_size = 0
        if swap_axis_for_flash:
            for ll in range(fill_level + 1):
                all_past.append(past_tokens[ll])

                attn_size += len(past_tokens[ll])
                level_sizes.append(len(past_tokens[ll]))

                if ll == 0:
                    ids_list.append(list(range(lst_id + 1, lst_id + 1 + len(past_tokens[ll]))))
                else:
                    offset = len(past_tokens[0]) + 1 - len(past_tokens[ll])
                    ids_list.append(list(range(lst_id + ll + offset, lst_id + ll + offset + len(past_tokens[ll]))))

            all_past = np.append(np.array(all_past[0]), np.array(all_past[1:]).transpose().flatten()).tolist()
            ids_list = np.append(np.array(ids_list[0]), np.array(ids_list[1:]).transpose().flatten()).tolist()
            
        # Manipulating positional id
        else:
            pos_offset = 0
            for ll in range(fill_level + 1):
                all_past += past_tokens[ll]
                attn_size += len(past_tokens[ll])
                level_sizes.append(len(past_tokens[ll]))

                if ll == 0:
                    ids_list += list(range(lst_id + 1 + pos_offset, lst_id + 1 + pos_offset + 1*len(past_tokens[ll]), 1))
                else:
                    offset = len(past_tokens[0]) + 1 - len(past_tokens[ll]) + pos_offset
                    ids_list += list(range(lst_id + ll + offset, lst_id + ll + offset + 1*len(past_tokens[ll]), 1))

        if guess_tokens is not None:
            if decoding_mask is None:
                input_ids = torch.cat((input_ids, torch.tensor(all_past + guess_tokens, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
                guess_ids = list(range(lst_id + 1, lst_id + 1 + guess_size)) * (len(guess_tokens) // guess_size)
                position_ids = torch.cat((position_ids, torch.tensor(ids_list + guess_ids, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
            else:
                input_ids = torch.cat((input_ids, torch.tensor(guess_tokens + all_past, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
                guess_ids = list(range(lst_id + 1, lst_id + 1 + guess_size)) * (len(guess_tokens) // guess_size)
                position_ids = torch.cat((position_ids, torch.tensor(guess_ids + ids_list, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
            
            attention_mask = torch.cat((attention_mask, torch.ones(1, attn_size + len(guess_tokens), \
                    device=input_ids.device, dtype=input_ids.dtype)), dim=1)

        else:
            input_ids = torch.cat((input_ids, torch.tensor(all_past, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
            position_ids = torch.cat((position_ids, torch.tensor(ids_list, device=input_ids.device, dtype=input_ids.dtype).unsqueeze(0)), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones(1, attn_size, \
                    device=input_ids.device, dtype=input_ids.dtype)), dim=1)

        step_len = attention_mask.size(1)
        
        outputs = self.model.Qwen2Modeljforward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            _use_cache=_use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            WINDOWS_SIZE=WINDOWS_SIZE,
            la_mask_offset=la_mask_offset,
            is_prefill=past_tokens[1] is None,
            level_sizes=level_sizes,
            guess_size=guess_size,
            not_seq=not_seq,
            guess=guess_tokens,
            decoding_mask=decoding_mask,
            flex_attention=flex_attention,
            use_flash=use_flash
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        
        logits = logits.float()

        loss = None
        if labels is not None: #train
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        ret = CausalLMOutputWithPast(
            loss=loss,
            logits=logits.to(input_ids.device),
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        ret.kvcache_len = prefill_size + past_size
        ret.step_len = step_len

        if guess_tokens is not None:
            lguess = len(guess_tokens)
        else:
            lguess = 0

        ret.out_logits = ret.logits[:,prefill_size - 1,:].to(input_ids.device) #decode logits
        assert fill_level != -1
        if lguess > 0:
            window = len(past_tokens[fill_level])
            llookahead = sum(level_sizes)
            start = ret.logits.size(1)-llookahead-lguess
            end = ret.logits.size(1)-llookahead
            if swap_axis_for_flash:
                level = len(level_sizes) + 1
                assert window == level_sizes[-1]
                base = start - (level - 3) * window
                tmp = einops.rearrange(ret.logits[:,base:end], "b ( w l) h->b (l w) h", b=ret.logits.size(0), w=window, l=level-2,h=ret.logits.size(2)) #.clone()
                ret.inp_logits = tmp[:,start - base:end - base,:].to(input_ids.device)         
            else:
                ret.inp_logits = ret.logits[:,-window:,:].to(input_ids.device) #lookahead branch logits
            ret.guess_logits = ret.logits[:,start:end,:].to(input_ids.device) #verification branch logits
        else:
            window = len(past_tokens[fill_level])
            start = ret.logits.size(1)-window
            end = ret.logits.size(1)
            if swap_axis_for_flash:
                level = len(level_sizes) + 1
                assert window == level_sizes[-1]
                base = start - (level - 3) * window
                tmp = einops.rearrange(ret.logits[:,base:end], "b ( w l) h->b (l w) h", b=ret.logits.size(0), w=window, l=level-2,h=ret.logits.size(2)) #.clone()
                ret.inp_logits = tmp[:,start - base:end - base,:].to(input_ids.device)
            else:
                ret.inp_logits = ret.logits[:,start:end,:].to(input_ids.device) #lookahead branch logits
        
        return ret

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "_use_cache": kwargs.get("_use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def j_update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs,
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ):
        # update past_key_values
        model_kwargs["past_key_values"] = outputs.past_key_values
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )

        return model_kwargs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
