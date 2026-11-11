# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
# Adapted for educational use under Apache-2.0; see reference/LICENSE.lerobot.
"""Runnable bridge from image/language/state to FM velocity, with a tiny REAL VLM.

This demonstrates the interface; it is not a dataset trainer or a complete robot
policy. Inputs here are already normalized. Run from the lesson directory:
    .venv-smolvlm/bin/python -B -m examples.smolvlm_expert_steps
"""
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from transformers import SmolVLMConfig, SmolVLMForConditionalGeneration

from smolvla_scratch.models.smolvla import SmolVLMWithExpertModel, SmolVLA, positions
from smolvla_scratch.models.attention import block_mask
from smolvla_scratch.models.flow_matching import make_flow_matching_inputs, FlowConvention, euler_integrate


def tiny_vlm_config():
    """Small random model for tests; never allocate the default multi-billion model."""
    return SmolVLMConfig(
        text_config=dict(model_type="llama", vocab_size=64, hidden_size=32,
                         intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                         attention_dropout=0.0, pad_token_id=0, tie_word_embeddings=False),
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, image_size=16, patch_size=4,
                           attention_dropout=0.0),
        scale_factor=2, image_token_id=63, tie_word_embeddings=False,
    )


def tiny_model(**kwargs):
    """No tokenizer/download needed: the example supplies token IDs directly."""
    return SmolVLMWithExpertModel(
        vlm=SmolVLMForConditionalGeneration(tiny_vlm_config()),
        processor=SimpleNamespace(tokenizer=None), **kwargs,
    )


def from_project_config(config):
    """Construction mapping for Lite-Lab; downloads the chosen VLM on first use.

    Set config.load_vlm_weights=True when training a NEW expert over a pretrained
    VLM. Set it False only for a full checkpoint load, or deliberate random-VLM tests.
    """
    return SmolVLMWithExpertModel(
        model_id=config.vlm_model_name,
        load_vlm_weights=config.load_vlm_weights,
        train_expert_only=config.train_expert_only,
        freeze_vision_encoder=config.freeze_vision_encoder,
        attention_mode=config.attention_mode,
        num_expert_layers=config.num_expert_layers,
        num_vlm_layers=config.num_vlm_layers,
        self_attn_every_n_layers=config.self_attn_every_n_layers,
        expert_width_multiplier=config.expert_width_multiplier,
        device=getattr(config, "device", None) or "auto",
    )


# Backward-compatible example name; model implementation lives in models/smolvla.py.
VelocityExample = SmolVLA


def demo():
    torch.manual_seed(2026)
    policy = VelocityExample(tiny_model(
        attention_mode="cross_attn", self_attn_every_n_layers=2,
        expert_width_multiplier=0.75, train_expert_only=True, freeze_vision_encoder=True,
    ))
    image = torch.randn(2, 3, 16, 16)
    tokens = torch.tensor([[3, 4, 5, 6, 7], [8, 9, 10, 0, 0]])
    state = torch.randn(2, 6)
    policy.train()
    prefix, pad, boundaries = policy.prefix(image, tokens, tokens != 0, state)
    actions = F.pad(torch.randn(2, 50, 6), (0, 26))
    noise, time = torch.randn_like(actions), torch.tensor([0.2, 0.8])
    x_t, target, _ = make_flow_matching_inputs(actions, noise, time, convention=FlowConvention.NOISE_AT_ONE)
    prediction = policy(prefix, pad, boundaries, x_t, time)
    loss = F.mse_loss(prediction[..., :6], target[..., :6])
    loss.backward()
    print("prefix:", tuple(prefix.shape), "velocity:", tuple(prediction.shape), "loss:", loss.item())
    policy.eval()
    prefix, pad, boundaries = policy.prefix(image, tokens, tokens != 0, state)
    result = policy.sample(prefix, pad, boundaries, noise)
    print("normalized action chunk:", tuple(result.shape), "finite:", bool(result.isfinite().all()))


if __name__ == "__main__":
    demo()
