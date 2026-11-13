# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Self/cross attention, GQA, mask và KV cache; không tải hay tạo model."""
import torch
from torch import nn
from transformers import DynamicCache

from .embeddings import apply_rope


def block_mask(pad, boundaries):
    """Bool [B,L,L]: True allow query reading key; supporting block causal."""
    if pad.ndim != 2 or boundaries.shape != pad.shape or pad.dtype != torch.bool:
        raise ValueError("pad must be bool[B,L] and boundaries must have the same shape")
    groups = torch.cumsum(boundaries, dim=1)
    """
    Tokens are allowed to read the same-group tokens or previous tokens (group_key <= group_query)
    & valid key
    & valid query
    """
    return (groups[:, None, :] <= groups[:, :, None]) & pad[:, None, :] & pad[:, :, None]


class AttentionEngine:
    # TODO: Change to English
    """Chỉ thực thi phép toán; Q/K/V/o_proj vẫn thuộc VLM và expert.

    Không phải nn.Module vì không có parameter/buffer hay trạng thái train/eval.
    Không giữ tham chiếu đến toàn bộ policy, tránh vòng phụ thuộc giữa các file.
    """
    def __init__(self, num_attention_heads, num_key_value_heads):
        if min(num_attention_heads, num_key_value_heads) <= 0 or num_attention_heads % num_key_value_heads:
            raise ValueError("query heads must be a positive multiple of KV heads")
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        past_key_values: "DynamicCache | None" = None,
    ) -> "tuple[list[torch.Tensor], DynamicCache | None]":
        query_states = []
        key_states = []
        value_states = []
        # Prefix : Layers of VLM, Suffix: Layers of Expert 
        for i, hidden_states in enumerate(inputs_embeds):  # input_embeds = [prefix, suffix]
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            # Normalize the hidden_states
            hidden_states = layer.input_layernorm(hidden_states) # [B, L, vlm_width]
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim) # [B, L, -1, head_dim]

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)

            # Create Q/K/V for VLM and Expert
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape) # [B, L, Hq, head_dim]
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape) # [B, L, Hkv, head_dim]
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape) # [B, L, Hkv, head_dim]
 
            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        # concatenate on the number of embeddings/tokens
        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]

        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len] # [B, L]
            _attention_mask = attention_mask[:, :seq_len, :seq_len] # [B, L, L]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        attention_mask_ = _attention_mask
        position_ids_ = _position_ids

        query_states = apply_rope(query_states, position_ids_) # [B, L, Hq, head_dim]
        key_states = apply_rope(key_states, position_ids_) # [B, L, Hkv, head_dim]

        if use_cache:
            if past_key_values is None:
                raise ValueError(
                    "`use_cache=True` requires a `past_key_values` cache (forward() creates one on prefill)."
                )
            # `DynamicCache` stores tensors as [batch, heads, seq, head_dim]; this module works with
            # [batch, seq, heads, head_dim]. During prefix prefill this stores the (post-RoPE) K/V and
            # returns them unchanged; during denoising it appends the suffix K/V and returns
            # [prefix; suffix], exactly like the previous hand-rolled dict cache.
            key_states, value_states = past_key_values.update(
                key_states.transpose(1, 2), value_states.transpose(1, 2), layer_idx
            )
            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)

        attention_interface = self.eager_attention_forward

        att_output = attention_interface(
            attention_mask_, batch_size, head_dim, query_states, key_states, value_states
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        past_key_values: "DynamicCache | None" = None,
    ) -> "tuple[list[torch.Tensor], DynamicCache | None]":
        attention_interface = self.eager_attention_forward

        att_outputs = []
        assert len(inputs_embeds) == 2 or (use_cache and past_key_values is not None), (
            f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
        )

        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            # [prefix position] and [suffix position]
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            """VLM Layer: Taking K/V from Prefix"""
            layer = model_layers[0][layer_idx]

            hidden_states = layer.input_layernorm(inputs_embeds[0]) # [B, Prefix, vlm_width]

            input_shape = hidden_states.shape[:-1] # [B, P]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim) # [B, P, -1, head_dim]

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape) # [B, P, Hq, head_dim]
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape) # [B, P, Hkv, head_dim]
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape) # [B, P, Hkv, head_dim]

            # B,L,H,D with L sequence length, H number of heads, D head dim
            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            att_output = attention_interface(
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_outputs.append(att_output)
        else:
            # During denoising the posistion_ids is exactly the suffix positions, dont need to separate 
            expert_position_id = position_ids

        if use_cache and past_key_values is not None:
            # Cross-attention layers never fill the cache themselves: during the prefix prefill every
            # layer goes through `forward_attn_layer`, which stores the (post-RoPE) VLM K/V for this
            # layer index. Here we only read them back (no concatenation: the expert cross-attends to
            # the fixed prefix). `DynamicCache` stores [batch, heads, seq, head_dim]; transpose back.
            key_states = past_key_values.layers[layer_idx].keys.transpose(1, 2)
            value_states = past_key_values.layers[layer_idx].values.transpose(1, 2)

        # Expert
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1]) # [B, Suffix, expert_width]

            expert_input_shape = expert_hidden_states.shape[:-1] # [B, Suffix]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim) # [B, Suffix, -1, head_dim]

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape) # [B, Suffix, Hq, head_dim]

            # reshape (not view): K/V read back from the cache are transposed, hence non-contiguous
            """
            Taking K/V expert by taking back the K/V Prefix
            """
            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).reshape(
                *key_states.shape[:2], -1
            ) # [B, P, -1] Combine two last dim
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )  # k_proj should have same dim as kv [B, P, Hkv, head_dim]

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).reshape(
                *value_states.shape[:2], -1
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            ) # [B, P, Hkv, head_dim]

            expert_position_id = (
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            )  # start from 0
            expert_attention_mask = attention_mask[
                :, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :
            ]  # [B, S, P]

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )
            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        # att_output = att_output.to(dtype=models[i].dtype)
        return att_outputs, past_key_values

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):

        """
        Calculate the attention score:
        Attention(Q, K, V) = softmax(dot(Q,K)/sqrtDh) + maks) * V
        """
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads
        if query_states.shape[2:] != (num_att_heads, head_dim):
            raise ValueError("Query head geometry differs from VLM")
        if key_states.shape[2:] != (num_key_value_heads, head_dim) or value_states.shape != key_states.shape:
            raise ValueError("K/V head geometry differs from VLM")

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        ) 
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min  # -2.3819763e38  # See gemma/modules.py
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        # Q/K decide the rate of obtaining information
        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        # and After that combining the heads
        att_output = att_output.permute(0, 2, 1, 3) # [B, Hq, Lq, head_dim] -> [B, Lq, Hq, head_dim]
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output


def crop_prefix_cache(cache: DynamicCache, prefix_length: int) -> DynamicCache:
    """Restore ONLY prefix K/V after every denoising step (inference only). (Remove the suffix)

    Self-attention appends suffix K/V in place; cross-attention just reads prefix.
    Transformers 5.22 accepts a NEGATIVE number of tokens to remove. Each
    layer needs its own count because cross-attention does not append a suffix.
    Skip unchanged layers, so the meaning of crop(0) is irrelevant.
    """
    if prefix_length <= 0:
        raise ValueError("prefix_length must be positive")
    if any(layer.get_seq_length() < prefix_length for layer in cache.layers):
        raise ValueError("cache is shorter than prefix_length")
    for layer in cache.layers:
        extra_tokens = layer.get_seq_length() - prefix_length
        if extra_tokens > 0:
            layer.crop(-extra_tokens)
    return cache
