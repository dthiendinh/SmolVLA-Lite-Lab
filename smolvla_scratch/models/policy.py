# Adapted from Hugging Face LeRobot modeling_smolvla.py (Apache-2.0).
from collections import deque

import torch
from torch import nn

from ..utils.vla_utils import pad_vector, resize_with_pad
from .smolvla import VLAFlowMatching
from .rtc_processor import RTCProcessor

STATE = "observation.state"
ACTION = "action"
TOKENS = "observation.language.tokens"
LANG_MASK = "observation.language.attention_mask"


class SmolVLAPolicy(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        if config.adapt_to_pi_aloha or config.use_delta_joint_actions_aloha:
            raise NotImplementedError("Aloha mappings are not implemented")

        if not 1 <= config.n_action_steps <= config.chunk_size:
            raise ValueError("Require 1 <= n_action_steps <= chunk_size")

        self.action_dim = int(config.output_features[ACTION]["shape"][0])
        if not 1 <= self.action_dim <= config.max_action_dim:
            raise ValueError("Actual action width must fit max_action_dim")

        self.image_keys = [
            key for key, feature in config.input_features.items()
            if feature["type"] == "VISUAL"
        ]
        if not self.image_keys:
            raise ValueError("Configure at least one VISUAL camera")

        if config.empty_cameras < 0:
            raise ValueError("empty_cameras must be nonnegative")

        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor = self.rtc_processor)
        self.model.rtc_processor = self.rtc_processor
        self.reset()

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def supports_rtc(self):
        return True  # Guided RTC; trained mode is rejected below.

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def reset(self):
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps)
        }

    # Return trainable params
    def get_optim_params(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )


    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.image_keys if key in batch]
        missing_img_keys = [key for key in self.image_keys if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        for key in present_img_keys:
            # Take the last observation (newest frame)
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if img.dtype == torch.uint8:
                img = img.to(torch.float32) / 255.0 # range [0,1]
            if self.config.resize_imgs_with_padding is not None:
                # SmolVLA stores the target as (width, height); the shared helper expects (height, width).
                img = resize_with_pad(
                    img,
                    self.config.resize_imgs_with_padding[1],
                    self.config.resize_imgs_with_padding[0],
                    pad_value=0,
                )

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device

            # valid mask for each image
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        return images, img_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[STATE][:, -1, :] if batch[STATE].ndim > 2 else batch[STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def forward(self, batch, noise=None, time=None, reduction="mean"):
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if reduction not in ("mean", "none"):
            raise ValueError("reduction must be mean or none")

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[TOKENS]
        lang_masks = batch[LANG_MASK]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict: dict[str, float] = {}

        losses = self.model(images, img_masks, lang_tokens, lang_masks, state, actions, noise=noise, time=time,)
        # Only supervise actual robot action dimensions.
        losses = losses[..., :self.action_dim]

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        if reduction == "none":
            # Return per-sample losses (B,) by averaging over valid (time, action) entries
            if actions_is_pad is None:
                per_sample_loss = losses.mean(dim=(1, 2))
            else:
                num_valid = ((~actions_is_pad).sum(dim=1) * losses.shape[-1]).clamp_min(1)
                per_sample_loss = losses.sum(dim=(1, 2)) / num_valid
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            # Default: return scalar mean loss over valid (time, action) entries
            if actions_is_pad is None:
                loss = losses.mean()
            else:
                num_valid = ((~actions_is_pad).sum() * losses.shape[-1]).clamp_min(1)
                loss = losses.sum() / num_valid
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    def _get_action_chunk(self, batch, *, noise=None, **rtc_kwargs):
        images, masks = self.prepare_images(batch)

        actions = self.model.sample_actions(
            images,
            masks,
            batch[TOKENS],
            batch[LANG_MASK],
            self.prepare_state(batch),
            noise=noise,
            **rtc_kwargs,
        )

        return actions[..., :self.action_dim]

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch,
        *,
        noise=None,
        inference_delay=None,
        prev_chunk_left_over=None,
        execution_horizon=None,
    ):
        self.eval()

        return self._get_action_chunk(
            batch,
            noise=noise,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )

    def _check_get_actions_condition(self):
        return len(self._queues[ACTION]) == 0

    @torch.no_grad()
    def select_action(self, batch, *, noise=None):
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        if self._rtc_enabled():
            raise RuntimeError("For RTC, use predict_action_chunk")

        self.eval()

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(batch,noise=noise,)
            self._queues[ACTION].extend(actions.transpose(0, 1)[:self.config.n_action_steps])
        return self._queues[ACTION].popleft()