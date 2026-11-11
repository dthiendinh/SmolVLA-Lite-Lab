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

"""Tạo action transformer nhỏ, sửa cross K/V và căn layer với VLM."""
import copy

from torch import nn
from transformers import AutoModel


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    if hidden_dim <= 0 or ffn_dim_multiplier <= 0 or multiple_of <= 0:
        raise ValueError("FFN dimensions and multiplier must be positive")
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def build_action_expert(text_config, num_vlm_layers, *, num_expert_layers=-1,
                        expert_width_multiplier=0.75, attention_mode="cross_attn",
                        self_attn_every_n_layers=2):
    """Expert mới từ text config; VLM weights được quản lý ở backbone.py."""
    if attention_mode not in {"self_attn", "cross_attn"} or expert_width_multiplier <= 0:
        raise ValueError("Invalid attention_mode or expert_width_multiplier")
    if num_vlm_layers <= 0 or num_expert_layers > num_vlm_layers or (
        num_expert_layers > 0 and num_vlm_layers % num_expert_layers
    ):
        raise ValueError("num_expert_layers must divide the retained VLM depth")

    # Smaller lm expert
    lm_expert_config = copy.deepcopy(text_config)
    hidden_size = lm_expert_config.hidden_size
    lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
    if lm_expert_config.hidden_size < 1:
        raise ValueError("expert width rounds to zero")

    # Keep projection head geometry even when the residual-stream width shrinks.
    lm_expert_config.head_dim = text_config.head_dim
    lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
    lm_expert_config.num_hidden_layers = num_vlm_layers
    if num_expert_layers > 0:
        lm_expert_config.num_hidden_layers = num_expert_layers
    
    # Initialize the action expert with the initialized architecture above
    expert = AutoModel.from_config(lm_expert_config)

    actual_expert_layers = len(expert.layers)
    if "cross" in attention_mode:
        # Reshape qkv projections to have the same input dimension as the vlm
        for layer_idx in range(len(expert.layers)):
            # Experts with fewer layers run at global VLM indices 0, stride, ... .
            # Choose projection type using that GLOBAL index, not the local expert index.
            vlm_layer_idx = layer_idx * (num_vlm_layers // actual_expert_layers)
            if self_attn_every_n_layers > 0 and vlm_layer_idx % self_attn_every_n_layers == 0:
                continue

            # K/V VLM's width -> K/V Expert's width
            expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                text_config.num_key_value_heads * text_config.head_dim,
                lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                bias=lm_expert_config.attention_bias,
            )
            expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                text_config.num_key_value_heads * text_config.head_dim,
                lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                bias=lm_expert_config.attention_bias,
            )
    # Remove unused embed_tokens
    expert.embed_tokens = None

    # AutoModel normally has no language head; retain the original freeze guard.
    for name, parameter in expert.named_parameters():
        if "lm_head" in name:
            parameter.requires_grad = False
    return expert


def align_model_layers(vlm_layers, expert_layers):
    """Ví dụ VLM 4/expert 2 -> [E0,None,E1,None], không nhân đôi expert."""
    if not expert_layers or len(vlm_layers) % len(expert_layers):
        raise ValueError("expert depth must divide VLM depth")
    stride = len(vlm_layers) // len(expert_layers)
    aligned = [expert_layers[i // stride] if i % stride == 0 else None
               for i in range(len(vlm_layers))]
    return [list(vlm_layers), aligned]
