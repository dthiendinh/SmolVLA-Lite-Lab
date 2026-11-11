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

"""Mã hóa vị trí và tạo prefix/suffix; không tải VLM, không tính attention.

RoPE mã hóa vị trí token. Sin/cos time embedding mã hóa mức nhiễu flow matching.
PrefixSuffixEmbeddings sở hữu các projection của state, action và timestep.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("RoPE expects [B,L,H,D] with even D")
    if positions.shape != x.shape[:2] or max_wavelength <= 0:
        raise ValueError("positions must be [B,L] and max_wavelength must be positive")
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def create_sinusoidal_pos_embedding(time, dimension, min_period=4e-3, max_period=4.0, device=None):
    """Flow time [B] hoặc [B,S] -> [..., dimension], theo recipe SmolVLA."""
    if dimension <= 0 or dimension % 2:
        raise ValueError("time embedding dimension must be positive and even")
    if time.ndim not in (1, 2) or not 0 < min_period <= max_period:
        raise ValueError("Require time [B] or [B,S] and 0 < min_period <= max_period")
    target = time.device if device is None else torch.device(device)
    dtype = torch.float32 if target.type in {"mps", "xpu"} else torch.float64
    fraction = torch.linspace(0, 1, dimension // 2, dtype=dtype, device=target)
    period = min_period * (max_period / min_period) ** fraction
    angles = time.to(target)[..., None] * (1.0 / period * 2 * math.pi)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class PrefixSuffixEmbeddings(nn.Module):
    """Biến input đã normalize thành token cho hai nhánh transformer.

    Baseline một camera; caller chuẩn bị resize và normalize trước khi gọi.
    Backbone được truyền vào embed_prefix, không đăng ký thêm một bản VLM ở đây.
    """
    def __init__(self, vlm_width, expert_width, state_dim=32, action_dim=32,
                 min_period=4e-3, max_period=4.0):
        super().__init__()
        if min(vlm_width, expert_width, state_dim, action_dim) <= 0 or expert_width % 2:
            raise ValueError("dimensions must be positive; expert_width must be even")
        if not 0 < min_period <= max_period:
            raise ValueError("Require 0 < min_period <= max_period")
        self.padded_dim = state_dim
        self.min_period, self.max_period = min_period, max_period
        self.state_proj = nn.Linear(state_dim, vlm_width)
        self.action_in_proj = nn.Linear(action_dim, expert_width)
        self.action_time_mlp_in = nn.Linear(2 * expert_width, expert_width)
        self.action_time_mlp_out = nn.Linear(expert_width, expert_width)

    def embed_prefix(self, core, image, tokens, language_pad, state):
        if state.ndim != 2 or state.shape[-1] > self.padded_dim:
            raise ValueError("state must be [B,state_dim] with state_dim <= padded state width")
        if tokens.ndim != 2 or language_pad.shape != tokens.shape or language_pad.dtype != torch.bool:
            raise ValueError("tokens and bool language_pad must have shape [B,L]")
        # The sqrt(width) scaling is outside embed_image/embed_language_tokens.
        image_emb = core.embed_image(image)
        image_emb = image_emb * image_emb.shape[-1] ** 0.5
        language_emb = core.embed_language_tokens(tokens)
        language_emb = language_emb * math.sqrt(language_emb.shape[-1])
        state_emb = self.state_proj(F.pad(state, (0, self.padded_dim - state.shape[-1])))[:, None, :]
        prefix = torch.cat([image_emb, language_emb, state_emb], dim=1)
        pad = torch.cat([
            torch.ones(image_emb.shape[:2], dtype=torch.bool, device=image.device),
            language_pad,
            torch.ones(state_emb.shape[:2], dtype=torch.bool, device=image.device),
        ], dim=1)
        boundaries = torch.zeros_like(pad)
        boundaries[:, -1] = True  # image/language cannot read state
        return prefix, pad, boundaries

    def embed_suffix(self, x_t, time):
        action = self.action_in_proj(x_t)
        time_emb = create_sinusoidal_pos_embedding(
            time, action.shape[-1], self.min_period, self.max_period, device=action.device,
        ).to(action.dtype)
        if time.ndim == 1:
            time_emb = time_emb[:, None, :].expand_as(action)
        elif time.shape != x_t.shape[:2]:
            raise ValueError("per-action time must match [B,S]")
        return self.action_time_mlp_out(F.silu(self.action_time_mlp_in(torch.cat([action, time_emb], -1))))
