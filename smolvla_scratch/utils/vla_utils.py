"""
Core VLA utilities for SmolVLA From Scratch.

Reference:
https://github.com/huggingface/lerobot

Includes:
- Sinusoidal timestep embedding
- Block-causal attention masks
- Additive attention masks
- State/action feature padding
- SmolVLA-style image resizing
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

# This func is used for positional embedding inside Attention
def create_sinusoidal_pos_embedding( 
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device: torch.device | str | None = None
) -> Tensor:
    """
    Convert Flow Matching timesteps into sinusoidal embeddings.

    Args:
        time: [batch_size] or [batch_size, action_horizon]
        dimension: Even embedding dimension
        min_period: Smallest sinusoidal period
        max_period: Largest sinusoidal period
        device: Optional computation device

    Returns:
        [batch_size, dimension] or [batch_size, horizon, dimension]
    """

    """Compute sine-cosine embeddings for scalar or per-action positions."""
    if time.ndim not in (1, 2):
        raise ValueError("time must have shape [batch_size] or [batch_size, horizon]")

    if dimension <= 0 or dimension % 2 != 0:
        raise ValueError("dimension must be a positive even integer")

    if min_period <= 0 or max_period < min_period:
        raise ValueError("Require 0 < min_period <= max_period")

    # Resolve device and computation precision.
    target_device = (time.device if device is None else torch.device(device))
    compute_dtype = (torch.float32 if target_device.type in {"mps", "xpu"} else torch.float64)
    t = time.to(device=target_device, dtype=compute_dtype,)

    # D/2 frequencies spaced geometrically.
    fraction = torch.linspace(0.0, 1.0, dimension // 2, device=target_device, dtype=compute_dtype,)
    periods = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / periods * 2 * math.pi
    # [B, 1] * [D/2] -> [B, D/2]
    # Also supports [B, H, 1].
    sin_input = t[..., None] * scaling_factor
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)


# This func is used for defining the type of Causal Attention
def make_att_2d_masks(pad_masks: Tensor,att_masks: Tensor,) -> Tensor:
    """
    Build a block-causal boolean attention mask.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.

    Returns:
        [B, N, N] boolean attention mask

    Matrix convention:
        mask[b, query_position, key_position]
    """
    if pad_masks.ndim != 2 or att_masks.ndim != 2:
        raise ValueError("Both inputs must have shape [B, N]")
    if pad_masks.shape != att_masks.shape:
        raise ValueError("pad_masks and att_masks must have the same shape")
    if pad_masks.device != att_masks.device:
        raise ValueError("Both masks must be on the same device")

    cumsum = torch.cumsum(att_masks, dim=1)
    # key_group <= query_group 
    # A query can see a key if key is in the same group or in the previous group as query (Causal Attention)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def prepare_attention_masks_4d(att_2d_masks: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """Expand boolean 2D attention masks to the additive 4D layout expected by transformers.
    Input: [B, Q, K]
    Output: [B, 1, Q, K]
    Valid positions become 0.0 and masked positions the large negative openpi constant.
    """
    OPENPI_ATTENTION_MASK_VALUE = -2.3819763e38  # TODO(pepijn): Modify this when extending support to fp8 models

    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    result = torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result



# Padding the vector to satisfy the needed size for neural network
def pad_vector(vector: Tensor, new_dim: int, *, truncate: bool = False) -> Tensor:
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)

    With ``truncate=False`` (openpi behavior), vectors whose last dimension is already
    >= new_dim are returned unchanged. With ``truncate=True`` (xVLA behavior), the last
    dimension is truncated to exactly ``new_dim`` (which may be 0).
    """
    if vector.shape[-1] == new_dim:
        return vector
    if not truncate:
        if vector.shape[-1] >= new_dim:
            return vector
        return F.pad(vector, (0, new_dim - vector.shape[-1]))
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = vector.new_zeros(*shape)
    length = min(current_dim, new_dim)
    new_vector[..., :length] = vector[..., :length]
    return new_vector


# Padding and Resize the image
def resize_with_pad(img: torch.Tensor, height: int, width: int, *, pad_value: float) -> torch.Tensor:
    """Resize a (b, c, h, w) image without distortion, padding on the LEFT and TOP.

    This is the smolvla/xvla convention. For the centered-padding openpi variant, see
    :func:`resize_with_pad_torch`. ``pad_value`` is keyword-only on purpose: callers
    historically used different values (0, -1) and must state their choice explicitly.
    """
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but got {img.shape}")

    current_height, current_width = img.shape[2:]
    if current_height == height and current_width == width:
        return img

    ratio = max(current_width / width, current_height / height)
    resized_height = int(current_height / ratio)
    resized_width = int(current_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, height - resized_height)
    pad_width = max(0, width - resized_width)
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img