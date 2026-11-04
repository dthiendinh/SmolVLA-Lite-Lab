from dataclasses import dataclass
import torch


@dataclass
class AdamWConfig:
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10.0
    optimizer_fused: bool | None = None

    def build(self, parameters):
        return torch.optim.AdamW(
            parameters,
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            fused=self.optimizer_fused,
        )
