"""
    Inference-time guided RTC configuration
"""
from dataclasses import dataclass
from enum import Enum
import math
from collections import deque

import torch
from torch.nn import functional as F

# TODO: Có thể thay đổi processor sau này cho inference 

class RTCAttentionSchedule(str, Enum):
    ZEROS = "ZEROS"
    ONES = "ONES"
    LINEAR = "LINEAR"
    EXP = "EXP"

@dataclass
class RTCConfig:
    """Configuration for Real Time Chunking (RTC) inference.

    RTC improves real-time inference by treating chunk generation as an inpainting problem,
    strategically handling overlapping timesteps between action chunks using prefix attention.
    """

    # Infrastructure
    enabled: bool = True

    # ``guided`` is the original inference-time Jacobian guidance. ``trained``
    # hard-inpaints a prefix and requires a compatible training-time RTC checkpoint.
    mode: str = "guided"

    # Core RTC settings
    # Todo change to exp
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.LINEAR
    max_guidance_weight: float = 10.0
    execution_horizon: int = 10

    # Debug settings
    debug: bool = False
    debug_maxlen: int = 100

    def __post_init__(self):
        """Validate RTC configuration parameters."""
        if self.mode not in {"guided", "trained"}:
            raise ValueError(f"mode must be 'guided' or 'trained', got {self.mode!r}")
        if self.max_guidance_weight <= 0:
            raise ValueError(f"max_guidance_weight must be positive, got {self.max_guidance_weight}")
        if self.debug_maxlen <= 0:
            raise ValueError(f"debug_maxlen must be positive, got {self.debug_maxlen}")




class RTCProcessor:
    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config
        self._steps = deque(maxlen=rtc_config.debug_maxlen)

    def is_debug_enabled(self):
        return self.rtc_config.debug

    def track(self, **values):
        if self.is_debug_enabled():
            self._steps.append({
                key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                for key, value in values.items()
            })

    def get_all_debug_steps(self):
        return list(self._steps)

    def reset_tracker(self):
        self._steps.clear()

    def _linweights(self, start, end, total):
        skip_steps_at_end = max(total - end, 0)

        linspace_steps = total - skip_steps_at_end - start

        if end <= start or linspace_steps <= 0:
            return torch.tensor([])

        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights, total, end):
        zeros_len = total - end

        if zeros_len <= 0:
            return weights

        zeros = torch.zeros(zeros_len)
        return torch.cat([weights, zeros])

    def _add_leading_ones(self, weights, start, total):
        ones_len = min(start, total)

        if ones_len <= 0:
            return weights

        ones = torch.ones(ones_len)
        return torch.cat([ones, weights])

    def get_prefix_weights(self, start, end, total):
        start = min(start, end)

        if self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ZEROS:
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.ONES:
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.LINEAR:
            lin_weights = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif self.rtc_config.prefix_attention_schedule == RTCAttentionSchedule.EXP:
            lin_weights = self._linweights(start, end, total)
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)

    def denoise_step(self, x_t, prev_chunk_left_over, inference_delay, time,
                     original_denoise_step_partial, execution_horizon=None):

        """RTC guidance wrapper around an existing denoiser.

        This method wraps an original denoising callable that only takes ``x_t`` and
        returns a base denoised velocity ``v_t``. It then applies Real-Time Chunking
        (RTC) prefix guidance using the leftover prefix from the previous chunk.

        Args:
            x_t (Tensor): Current latent/state to denoise. Shape ``(B, T, A)`` or ``(T, A)``.
            prev_chunk_left_over (Tensor | None): Unexecuted prefix from the previous
                chunk. Shape ``(B, T_prev, A)`` or ``(T_prev, A)``. If ``None``, no guidance
                is applied and the method returns ``v_t`` from the original denoiser.
            inference_delay (int): Number of timesteps from the prefix to use for guidance.
            time (float | Tensor): Scalar in [0, 1] indicating normalized time. Must be
                broadcastable with ``x_t``.
            original_denoise_step_partial (Callable[[Tensor], Tensor]): Callable that
                computes the base denoised velocity given only ``x_t``.
            execution_horizon (int | None): Horizon used to build prefix weights. If
                ``None``, defaults to ``self.rtc_config.execution_horizon``.

        Returns:
            Tensor: Guided velocity with the same shape as ``v_t``.

        Notes:
            - If inputs are 2D, a batch dimension is temporarily added and removed at the end.
            - If ``prev_chunk_left_over`` is shorter than the current chunk length ``T``, it is
              right-padded with zeros to match ``T``.
            - Prefix weights are constructed via ``get_prefix_weights(inference_delay, execution_horizon, T)``
              and broadcast to ``(B, T, A)``.
            - Guidance correction is computed via autograd using ``x1_t = x_t + time * v_t`` and
              ``error = (prev_chunk_left_over - x1_t) * weights``.
            - The final guidance weight is clamped by ``max_guidance_weight`` from the config.

        Reference:
            https://www.physicalintelligence.company/download/real_time_chunking.pdf
        """

        # If dont have the previous chunk, run the normal denoiser
        if prev_chunk_left_over is None or not self.rtc_config.enabled:
            return original_denoise_step_partial(x_t)
        if torch.is_inference_mode_enabled():
            raise RuntimeError("Guided RTC requires no_grad + enable_grad; use no_grad, not inference_mode")
        
        # Add batch dimension
        squeezed = x_t.ndim == 2
        x = x_t.unsqueeze(0) if squeezed else x_t 
        prev = prev_chunk_left_over
        prev = prev.unsqueeze(0) if prev.ndim == 2 else prev

        if x.ndim != 3 or prev.ndim != 3 or prev.shape[0] != x.shape[0]:
            raise ValueError("Current/previous chunks must have matching batch dimensions")
        if prev.shape[1] > x.shape[1] or prev.shape[2] > x.shape[2]:
            raise ValueError("Previous chunk cannot exceed the model horizon/action width")
        horizon = self.rtc_config.execution_horizon if execution_horizon is None else execution_horizon
        if not isinstance(horizon, int) or horizon < 0:
            raise ValueError("execution_horizon must be a nonnegative integer")
        
        horizon = min(horizon, prev.shape[1], x.shape[1])
        if not isinstance(inference_delay, int) or not 0 <= inference_delay <= horizon:
            raise ValueError("inference_delay must fit in the available guided prefix")
        
        # Check the flow time and pad the prev actions
        t = float(time)
        if not 0 <= t <= 1:
            raise ValueError("time must lie in [0,1]")
        prev = prev.detach().to(device=x.device, dtype=x.dtype)
        prev = F.pad(prev, (0, x.shape[2] - prev.shape[2], 0, x.shape[1] - prev.shape[1]))
        
        weights = self.get_prefix_weights(inference_delay, horizon, x.shape[1]).to(x)[None, :, None]
        with torch.enable_grad():
            # Set requires_grad BEFORE evaluating v(x): retain the denoiser Jacobian.
            x = x.detach().clone().requires_grad_(True)
            velocity = original_denoise_step_partial(x)
            clean_prediction = x - t * velocity
            error = (prev - clean_prediction) * weights
            correction = torch.autograd.grad(
                clean_prediction, x, grad_outputs=error.detach()
            )[0]
        strength = (0.0 if t == 0 else self.rtc_config.max_guidance_weight if t == 1
                    else min((t*t + (1-t)**2)/(t*(1-t)), self.rtc_config.max_guidance_weight))
        guided = (velocity - strength * correction).detach()
        self.track(time=t, correction=correction, err=error, weights=weights,
                   guidance_weight=strength, inference_delay=inference_delay,
                   execution_horizon=horizon)
        return guided.squeeze(0) if squeezed else guided
