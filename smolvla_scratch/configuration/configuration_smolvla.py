
from dataclasses import dataclass, field


@dataclass
class SmolVLAConfig:
    # 1. Input / Output
    n_obs_steps: int = 1
    chunk_size: int = 50
    # The number of steps using before needing a new chunk
    n_action_steps: int = 50

    # 2. Tensor dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # 3. Image & language
    image_size: tuple[int, int] = (512, 512)
    tokenizer_max_length: int = 48

    # 4. Action generation (steps in FLow Matching)
    num_steps: int = 10

    # 5. Architecture
    num_vlm_layers: int = 16
    expert_width_multiplier: float = 0.75

    # 6. Training
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    learning_rate: float = 1e-4

    # 7. Normalization
    normalization_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
    )

    def __post_init__(self):
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )

        if self.n_obs_steps <= 0:
            raise ValueError("n_obs_steps must be positive")

        if self.max_state_dim <= 0:
            raise ValueError("max_state_dim must be positive")

        if self.max_action_dim <= 0:
            raise ValueError("max_action_dim must be positive")

    @property
    def observation_delta_indices(self):
        return [0]

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None



