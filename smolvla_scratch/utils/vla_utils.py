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
import json
from pathlib import Path

import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from bisect import bisect_right

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


JOINT_NAMES = [
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
]
STATE = "observation.state"
ACTION = "action"

def read_episode(path, legacy_joint_order=False):
    """Đưa cột state/action về cùng thứ tự khớp vật lý của ACT-Rebuild."""
    with np.load(path / "data.npz", allow_pickle=False) as data:
        states = np.asarray(data["states"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        if "joint_names" in data.files:
            names = data["joint_names"].tolist()
        elif legacy_joint_order:
            # Recorder ACT cũ lưu sorted(action.keys()). Chỉ bật khi đúng nguồn này.
            names = sorted(JOINT_NAMES)
        else:
            raise ValueError(f"{path}: missing joint_names; use --legacy_joint_order only for old ACT recordings")
        if len(names) != 6 or set(names) != set(JOINT_NAMES):
            raise ValueError(f"{path}: unsupported joint_names: {names}")
        if states.ndim != 2 or states.shape[1] != 6 or actions.shape != states.shape:
            raise ValueError(f"{path}: states/actions must both be (T, 6)")
        if len(states) == 0 or not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError(f"{path}: empty episode or non-finite states/actions")
        if "timestamps" in data.files:
            times = data["timestamps"]
            if (times.shape != (len(states),) or not np.isfinite(times).all()
                    or np.any(np.diff(times) <= 0)):
                raise ValueError(f"{path}: timestamps must match samples and increase strictly")
    order = [names.index(name) for name in JOINT_NAMES]
    return states[:, order], actions[:, order]

class EpisodicDataset(Dataset):
    """Mỗi index là một timestep: ảnh/state tại t và action[t:t+chunk].

    Giữ state/action trong RAM; chỉ giải mã frame video đang cần.
    Dữ liệu trả về còn thô: processor sẽ normalize đúng một lần.
    """

    def __init__(self, dataset_dir, episodes, camera_names, chunk_size, task=None,
                 legacy_joint_order=False, episode_tasks=None):
        self.root = Path(dataset_dir)
        self.episodes = list(episodes)
        self.camera_names = list(camera_names)
        self.chunk_size = int(chunk_size)
        self.task = task
        self.episode_tasks = {}
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.arrays = []
        self.ends = []
        self.image_shapes = {}
        total = 0
        for name in self.episodes:
            if Path(name).name != name:
                raise ValueError(f"Invalid episode directory name: {name}")
            path = self.root / name
            instruction = task
            if instruction is None and episode_tasks is not None:
                instruction = episode_tasks.get(name)
            if instruction is None:
                metadata_path = path / "metadata.json"
                if not metadata_path.is_file():
                    raise ValueError(f"{path}: missing task; provide --task or metadata.json")
                instruction = json.loads(metadata_path.read_text(encoding="utf-8")).get("task")
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"{path}: task must be a nonempty instruction")
            self.episode_tasks[name] = instruction.strip()

            # Read the states and actions
            states, actions = read_episode(path, legacy_joint_order)
            # keep the states and actions in arrays to use again
            self.arrays.append((states, actions))
            total += len(states)
            # self.ends help seeing the whole dataset as N samples
            self.ends.append(total)

            # Check whether the camera is readable and the number of frame is equal to the num of timesteps or not
            for camera in self.camera_names:
                video = path / f"{camera}.mp4"
                cap = cv2.VideoCapture(str(video))
                try:
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    ok, frame = cap.read()
                    if not cap.isOpened() or not ok or count != len(states):
                        raise ValueError(f"{video}: unreadable video or frame/sample count mismatch")
                    shape = (3, *frame.shape[:2])
                    if camera in self.image_shapes and self.image_shapes[camera] != shape:
                        raise ValueError(f"{video}: inconsistent image resolution")
                    self.image_shapes[camera] = shape
                finally:
                    cap.release()
        if not self.ends:
            raise ValueError("Dataset split has no episodes")

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Episode Index
        episode_index = bisect_right(self.ends, index)
        # The timestep in that episode
        start = index - (self.ends[episode_index - 1] if episode_index else 0)
        states, actions = self.arrays[episode_index]
        # Take action chunk
        valid = min(self.chunk_size, len(actions) - start)
        chunk = np.zeros((self.chunk_size, 6), dtype=np.float32)
        chunk[:valid] = actions[start:start + valid]

        # Create the sample 
        sample = {
            STATE: torch.from_numpy(states[start].copy()),
            ACTION: torch.from_numpy(chunk),
            "action_is_pad": torch.arange(self.chunk_size) >= valid,
            "task": self.episode_tasks[self.episodes[episode_index]],
        }
        for camera in self.camera_names:
            video = self.root / self.episodes[episode_index] / f"{camera}.mp4"
            cap = cv2.VideoCapture(str(video))
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Cannot decode frame {start}: {video}")
            finally:
                cap.release()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # uint8 [0,255]; policy.prepare_images automatically convert to [-1,1].
            sample[f"observation.images.{camera}"] = torch.from_numpy(rgb).permute(2, 0, 1)
        return sample
