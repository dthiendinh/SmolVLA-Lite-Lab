from dataclasses import dataclass, field
from .optimizers import AdamWConfig
from .schedulers import CosineDecayWithWarmupSchedulerConfig
from low_control.hardware_constant import ROBOT_STATE
from .rtc_processor import RTCConfig

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

    input_features: dict = field(
        default_factory=lambda: {
            "observation.state": {
                "shape": (ROBOT_STATE,),
                "type": "STATE",
            }
        }
    )

    output_features: dict = field(
        default_factory=lambda: {
            "action": {
                "shape": (ROBOT_STATE,),
                "type": "ACTION",
            }
        }
    )

    # 3. Image & language
    resize_imgs_with_padding: tuple[int, int] = (512, 512)
    tokenizer_max_length: int = 48

    """This will be used for sim"""
    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False
    """"""

    # 4. Action generation (steps in FLow Matching)
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

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

    # 8. Optimizer Config
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10
    optimizer_fused: bool | None = None

    # 9. Scheduler Config
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # 10. Model Config
    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

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

    def get_optimizer_preset(self):
        return AdamWConfig(
            optimizer_lr=self.optimizer_lr,
            optimizer_betas=self.optimizer_betas,
            optimizer_eps=self.optimizer_eps,
            optimizer_weight_decay=self.optimizer_weight_decay,
            optimizer_grad_clip_norm=self.optimizer_grad_clip_norm,
            optimizer_fused=self.optimizer_fused,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self):
        return [0]

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None



