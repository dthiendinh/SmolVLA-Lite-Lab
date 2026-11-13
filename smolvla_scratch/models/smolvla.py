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

"""Điều phối VLM + expert và ghép thành baseline policy flow matching.

Chi tiết tải VLM, tạo expert, attention và embedding nằm ở bốn module riêng.
"""
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from transformers import DynamicCache

from .action_expert import build_action_expert, align_model_layers
from .attention import AttentionEngine, block_mask, crop_prefix_cache
from .backbone import (load_backbone, set_backbone_requires_grad,
                       keep_frozen_backbone_in_eval, embed_image, embed_language_tokens)
from .embeddings import PrefixSuffixEmbeddings
from .flow_matching import (FlowConvention, make_flow_matching_inputs,
                            sample_noise, sample_time_beta, euler_integrate)
from ..utils.vla_utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
)
from .rtc_processor import RTCProcessor, RTCActionSelectKwargs
from ..configuration.configuration_smolvla import SmolVLAConfig
import math
from typing import Unpack


def positions(pad):
    """Vị trí token; padding không tăng chỉ số."""
    return torch.cumsum(pad, dim=1) - 1


class SmolVLMWithExpertModel(nn.Module):
    """ Combine backbone and expert, receive [prefix,suffix] was embeded.

        hidden state
            ├─────────────────────┐
            ↓                     │
        Input RMSNorm             │
            ↓                     │
        Q/K/V + attention         │
            ↓                     │
        o_proj                    │
            ↓                     │
        Residual ◄───────────────
            │
            ├─────────────────────┐
            ↓                     │
        Post-attention RMSNorm    │
            ↓                     │
        FFN                       │
            ↓                     │
        Residual ◄─────-----──────┘
            │
        hidden for the next layer
    """
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        device: str = "auto",
        *,
        vlm: nn.Module | None = None,
        processor=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.vlm, self.processor = load_backbone(
            model_id, load_vlm_weights=load_vlm_weights, num_vlm_layers=num_vlm_layers,
            vlm=vlm, processor=processor, dtype=dtype,
        )
        self.config = self.vlm.config
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.lm_expert = build_action_expert(
            self.config.text_config, self.num_vlm_layers, num_expert_layers=num_expert_layers,
            expert_width_multiplier=expert_width_multiplier, attention_mode=attention_mode,
            self_attn_every_n_layers=self_attn_every_n_layers,
        )
        self.num_expert_layers = len(self.lm_expert.layers)
        self.expert_hidden_size = self.lm_expert.config.hidden_size
        # TODO: Check text_config here
        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads
        self.attention = AttentionEngine(self.num_attention_heads, self.num_key_value_heads)
        self.attention_mode = attention_mode
        self.self_attn_every_n_layers = self_attn_every_n_layers
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.set_requires_grad()
        if device != "auto":
            self.to(device=device)
                                 
    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        set_backbone_requires_grad(
            self.vlm, freeze_vision_encoder=self.freeze_vision_encoder,
            train_expert_only=self.train_expert_only, num_expert_layers=self.num_expert_layers,
        )

    def train(self, mode=True):
        super().train(mode)
        keep_frozen_backbone_in_eval(
            self.vlm, freeze_vision_encoder=self.freeze_vision_encoder,
            train_expert_only=self.train_expert_only,
        )
        return self

    def embed_image(self, image):
        return embed_image(self.vlm, image)

    def embed_language_tokens(self, tokens):
        return embed_language_tokens(self.vlm, tokens)

    def get_model_layers(self, models):
        return align_model_layers(models[0].layers, models[1].layers)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: "DynamicCache | None" = None,
        inputs_embeds: list[torch.Tensor | None] | None = None,
        use_cache: bool | None = None,
    ):
        if inputs_embeds is None or len(inputs_embeds) != 2:
            raise ValueError("inputs_embeds must be [prefix, suffix]")
        prefix, suffix = inputs_embeds
        active = [h for h in inputs_embeds if h is not None]
        if not active or any(h.ndim != 3 or h.shape[1] == 0 for h in active):
            raise ValueError("Provide at least one nonempty [B,L,hidden] stream")
        if len({(h.shape[0], h.device) for h in active}) != 1:
            raise ValueError("Embedding streams must share batch size and device")
        for h, width in zip(inputs_embeds, (self.config.text_config.hidden_size, self.expert_hidden_size)):
            if h is not None and h.shape[-1] != width:
                raise ValueError(f"Embedding width must be {width}, got {h.shape[-1]}")
        use_cache = bool(use_cache)
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        if past_key_values is not None:
            if past_key_values.get_seq_length() == 0 or prefix is not None or suffix is None:
                raise ValueError("A filled prefix cache accepts [None, suffix] only")
        if use_cache and past_key_values is None and (prefix is None or suffix is not None):
            raise ValueError("Prefill requires [prefix, None]; train with use_cache=False")
        if prefix is None and past_key_values is None and self.attention_mode == "cross_attn":
            raise ValueError("Cross-attention suffix needs a prefix or a filled cache")
        total = sum(h.shape[1] for h in active)
        key_len = total + (past_key_values.get_seq_length() if past_key_values is not None else 0)
        batch_size = active[0].shape[0]
        if position_ids is None or position_ids.shape != (batch_size, total):
            raise ValueError("position_ids must be [B, number of current tokens]")
        if attention_mask is None or attention_mask.dtype != torch.bool or attention_mask.shape != (batch_size, total, key_len):
            raise ValueError("attention_mask must be bool [B,current_tokens,total_keys]; True means visible")
        
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        # Prefix prefill: no cache was passed, so create one and fill it (every layer runs
        # self-attention over the prefix). When a filled cache is passed (denoising), layers
        # read from it instead.
        fill_kv_cache = use_cache and past_key_values is None
        if fill_kv_cache:
            past_key_values = DynamicCache()

        # RMSNorm
        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        # Run each global transformer layer
        for layer_idx in range(num_layers):
            if (inputs_embeds[0] is None and model_layers[1][layer_idx] is None):
                continue

            if (
                fill_kv_cache
                or inputs_embeds[1] is None
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = self.attention.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                ) # [B, S + P, Hq x head_dim]
            else:
                att_outputs, past_key_values = self.attention.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                )
                #  att_outputs = [
                #       prefix_attention_output, [B, P, Hq x head_dim]
                #       suffix_attention_output, [B, S, Hq x head_dim]
                #  ]
            outputs_embeds = []
            start = 0

            # inputs_embeds = [prefix (VLM Layer), suffix (Action )]
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )  # in case of self_attn
                if hidden_states is not None:
                    """ 
                        If having hidden_states but dont have expert layer,
                        keep the hidden_states, skip the attention update and FFN
                    """
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue

                    # return the suffix or prefix token index
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    
                    att_out = att_output[:, start:end]

                    # [B, P or S, Hq x head_dim] -> [B, P or S, vlm_width or expert_width]
                    out_emb = layer.self_attn.o_proj(att_out)

                    """ Residual Layer h_att = h_input + o_proj(attention_output) """
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()


                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual # [B, P or S, vlm_width or expert_width]

                    outputs_embeds.append(out_emb) 

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            # Current output is the input for the next layer
            inputs_embeds = outputs_embeds

        # final norm for each part
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        return self.attention.eager_attention_forward

    @property
    def eager_attention_forward(self):
        return self.attention.eager_attention_forward



def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor

class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    
    **Model Architecture: VLAFlowMatching combine backbone, action expert and flow matching to train by loss and form action from noise. 

    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )

        # Projection to hidden_size (vlm_width or expert_width)
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )


        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor

        # Compile model if requested
        if config.compile_model:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)  # type: ignore[method-assign]
            self.forward = torch.compile(self.forward, mode=config.compile_mode)  # type: ignore[method-assign]

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        return sample_noise(shape, device)

    def sample_time(self, bsize, device):
        return sample_time_beta(bsize, device, alpha=1.5, beta=1.0, scale=0.999, offset=0.001)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        for _img_idx, (
            img,
            img_mask,
        ) in enumerate(zip(images, img_masks, strict=False)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            # output img_emb from Vision Encoder
            img_emb = self.vlm_with_expert.embed_image(img)
            img_emb = img_emb

            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        
        # Ouput lang_emb from Text Encoder
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        # embs [B, P, vlm_width]
        # pad_masks [B, P]
        # att_masks [B, P]

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using a linear
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""

        # Create noise and timestep
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # noise ation , target velocity
        x_t, u_t, _ = make_flow_matching_inputs(actions, noise, time, convention=FlowConvention.NOISE_AT_ONE)

        # Prefix Embedding
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )

        # Suffix Embedding
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Call the model 
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
        ) 
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        # Compute loss
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    # Inference
    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs: Unpack[RTCActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            # Start with noise
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        else:
            noise = noise.to(device=device, dtype=torch.float32)
        if noise.shape != (bsize, self.config.chunk_size, self.config.max_action_dim):
            raise ValueError("Noise shape differs from configured batch/chunk/action dimensions")

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        if self.config.use_cache:
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            # Compute image, language and state key/value cache once.
            _, past_key_values = self.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

            def velocity(x_t, timestep):
                return self.denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)
        else:
            # Without a cache every step must receive the full prefix again.
            def velocity(x_t, timestep):
                suffix, suffix_pad, suffix_att = self.embed_suffix(x_t, timestep)
                pad = torch.cat([prefix_pad_masks, suffix_pad], dim=1)
                boundary = torch.cat([prefix_att_masks, suffix_att], dim=1)
                (_, suffix_out), _ = self.vlm_with_expert.forward(
                    attention_mask=make_att_2d_masks(pad, boundary),
                    position_ids=positions(pad),
                    inputs_embeds=[prefix_embs, suffix],
                    use_cache=False,
                )
                return self.action_out_proj(suffix_out[:, -self.config.chunk_size:].float())

        return euler_integrate(
            velocity,
            noise,
            self.config.num_steps,
            rtc_processor=self.rtc_processor,
            rtc_enabled=self._rtc_enabled(),
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        try:
            outputs_embeds, _ = self.vlm_with_expert.forward(
                attention_mask=full_att_2d_masks,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=self.config.use_cache,
            )
        finally:
            if past_key_values is not None:
                crop_prefix_cache(past_key_values, prefix_len)
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t
