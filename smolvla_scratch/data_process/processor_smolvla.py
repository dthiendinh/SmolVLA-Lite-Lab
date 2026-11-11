"""
Raw obs -> _rename (normalize feature's names) -> 
_ensure_batch (verify tensor shapes) -> 
_prepare_tasks (normalize instructions) -> 
_tokenize (create token IDs and masks) -> 
_to_device() (transform to CPU/GPU) ->
_transform (Normalize state and action)
"""

from typing import Any 
import torch 

STATE_KEY = "observation.state"
ACTION_KEY = "action"

LANGUAGE_TOKENS = "observation.language.tokens"
LANGUAGE_MASK = "observation.language.attention_mask"

IMAGE_PREFIX = "observation.images."



def _enum_name(value: Any) -> str:
    """Convert an Enum or string to a standard uppercase name."""
    name = getattr(value, "name", str(value))
    return str(name).split(".")[-1].upper()


class SmolVLAProcessor():
    """
    SmolVLA preprocessing/postprocessing.

    Input: 
        - observation.state: (D_s,), (B, D_s), or (B, O, D_s)
        - observation.images.*: (C,H,W), (B,C,H,W),
          or (B,O,C,H,W)
        - task: str or list[str]
        - action (optional): (B,D_a) or (B,T,D_a)

    Output:
        - Batched state and images
        - Language token IDs and attention mask
        - Normalized state and training actions

    Image resize and [-1,1] conversion are handled
    later by the model's prepare_images function.
    """

    def __init__(self, config: Any, dataset_stats: dict | None = None, tokenizer: Any | None = None, rename_map: dict[str, str] | None = None):
        self.config = config
        self.stats = dataset_stats or {}
        self.rename_map = rename_map or {}

        self.device = torch.device(getattr(config, "device", None) or "cpu")
        self.eps = 1e-8

        # Use the tokenizer belonging to the VLM backbone
        if tokenizer is None: 
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name)
        
        self.tokenizer = tokenizer 
        self.tokenizer.padding_side = "right"

        self.padding = config.pad_language_to
        self.max_length = config.tokenizer_max_length

        # Real action dimensionality, before model padding
        output_features = (getattr(config, "output_features", None) or {})

        action_feature = config.output_features["action"]

        self.action_dim = (int(action_feature["shape"][0]) if action_feature is not None else None)

    def _mode(self, feature_type: str) -> str:
        """Find normalization mode for STATE/ACTION/VISUAL."""
        mapping = getattr(self.config, "normalization_mapping", {})

        for key, mode in mapping.items():
            if _enum_name(key) == feature_type:
                return _enum_name(mode)
            
        return "IDENTITY"

    def _stat(self, feature_key: str, stat_name: str, reference: torch.Tensor) -> torch.Tensor:
        """Load a statistic on the input tensor's device/dtype"""
        feature_stats = self.stats.get(feature_key)

        if (feature_stats is None or stat_name not in feature_stats):
            raise ValueError(f"Missing {feature_key}.{stat_name} statistics")

        return torch.as_tensor(feature_stats[stat_name], dtype=reference.dtype, device=reference.device)


    # Data Processing
    def _rename(self, data: dict) -> dict:
        """
        Rename configured keys, including padding metadata.
        For example: 
        data = {
            "robot_state": state,
            "front_camera": image
        }
        self.rename_map = {"robot_state": "observation.state", "front_camera": "observation.images.front"}
        Result: new_key = "observation_state" 
        """
        renamed = {}

        for key, value in data.items():
            new_key = self.rename_map.get(key, key)

            if new_key == key:
                # Check whether the key endded with the two suffixes
                for suffix in ("_is_pad", "_padding_mask"):
                    if key.endswith(suffix):
                        # Delete the suffix
                        base = key[: -len(suffix)]
                        if base in self.rename_map:
                            # Add the suffix back
                            new_key = (self.rename_map[base] + suffix)
                        break
            
            if new_key in renamed:
                raise ValueError(f"Duplicate key after rename: {new_key}")
            
            renamed[new_key] = value
        return renamed


    def _ensure_batch(self, data: dict) -> dict:
        """Check and normalize the tensor dimension."""
        # Shallow copy the dictionary
        batch = dict(data)

        # Check whether the state exists or not
        if STATE_KEY not in batch:
            raise KeyError(f"Missing required key: {STATE_KEY}")
        state = torch.as_tensor(batch[STATE_KEY])

        if state.ndim == 1: # [state_dim,]
            state = state.unsqueeze(0) # [batch_size, state_dim]

        if state.ndim not in (2, 3):
            raise ValueError("State must have shape (state_dim), (batch_size,state_dim), or (batch_size,timesteps,state_dim)")
        batch[STATE_KEY] = state
        batch_size = state.shape[0]

        # Find the types of cameras
        image_keys = [
            key for key in batch
            if (key == "observation.image" or key.startswith(IMAGE_PREFIX)) and not key.endswith(("_padding_mask", "_is_pad"))
        ]

        if not image_keys:
            raise ValueError("SmolVLA needs at least one camera")
        for key in image_keys:
            image = torch.as_tensor(batch[key])

            if image.ndim == 3:   # [Channel, H, W]
                image = image.unsqueeze(0) # [1, Channel, H, W]

            if image.ndim not in (4, 5): # [batch_size, Channel, H, W] or [batch_size, timesteps, channel, h, w]
                raise ValueError(f"Invalid image shape: {image.shape}")
            # The channel's index
            channel_axis = 1 if image.ndim == 4 else 2
            if image.shape[channel_axis] != 3:
                raise ValueError(f"{key} must be channel-first RGB")
            if image.shape[0] != batch_size:
                raise ValueError(f"Batch mismatch for camera {key}")
            batch[key] = image
        
        # Check the action if existing
        if ACTION_KEY in batch:
            action = torch.as_tensor(batch[ACTION_KEY])
            if action.ndim == 1: # [action_dim,]
                action = action.unsqueeze(0) # [1, action_dim]
            if action.ndim not in (2, 3):
                raise ValueError("Action must be (batch_size, action_dim) or (batch_size, timesteps, action_dim)")
            if action.shape[0] != batch_size:
                raise ValueError("Action batch dimension mismatch")
            if (self.action_dim is not None and action.shape[-1] != self.action_dim):
                raise ValueError("Training action dimension mismatch")
            batch[ACTION_KEY] = action
        return batch

    def _prepare_tasks(self, task: str | list[str], batch_size: int,) -> list[str]:
        """
        Validate task's instructions and append newline once.
        Beside images and states, SmolVLA also need instruction so we need this func .
        """
        if isinstance(task, str):
            tasks = [task]
        elif (isinstance(task, (list, tuple)) and all(isinstance(t, str) for t in task)):
            tasks = list(task)
        else:
            raise TypeError("task must be a string or a list of strings")
        if len(tasks) != batch_size:
            raise ValueError(f"Got {len(tasks)} tasks for batch size {batch_size}")
        return [t if t.endswith("\n") else t + "\n" for t in tasks]

    def _tokenize(self, tasks: list[str]) -> dict:
        """Tokenize language instructions."""
        encoded = self.tokenizer(tasks, max_length=self.max_length, padding=self.padding, truncation=True, return_tensors="pt",)

        return {
            LANGUAGE_TOKENS: encoded["input_ids"].long(),
            LANGUAGE_MASK: (encoded["attention_mask"].bool()),
        }

    def _to_device(self, data: dict) -> dict:
        """Move tensors without converting integer token IDs."""
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }

    def _transform(self, x: torch.Tensor, feature_key: str, feature_type: str, inverse: bool = False,) -> torch.Tensor:
        """Apply normalization or its inverse."""
        mode = self._mode(feature_type)
        if mode == "IDENTITY":
            return x
        if not x.is_floating_point():
            x = x.float()
        if mode == "MEAN_STD":
            mean = self._stat(feature_key, "mean", x)
            std = self._stat(feature_key, "std", x)
            if inverse:
                return x * std + mean
            return (x - mean) / (std + self.eps)
        if mode == "MIN_MAX":
            low_name, high_name = "min", "max"
        elif mode == "QUANTILES":
            low_name, high_name = "q01", "q99"
        elif mode == "QUANTILE10":
            low_name, high_name = "q10", "q90"
        else:
            raise ValueError(f"Unsupported normalization mode: {mode}")

        low = self._stat(feature_key, low_name, x)
        high = self._stat(feature_key, high_name, x)
        delta = high - low
        delta = torch.where(delta == 0, torch.full_like(delta, self.eps), delta,)

        if inverse:
            return (x + 1.0) * delta / 2.0 + low
        return 2.0 * (x - low) / delta - 1.0

    def preprocess(self, data: dict) -> dict:
        """
        Raw observation -> model-ready observation.
        Training action, if present, is also normalized.
        """

        # Validate the raw observation
        # Rename the data
        batch = self._rename(data)
        # validate the dimension
        batch = self._ensure_batch(batch)
        batch_size = batch[STATE_KEY].shape[0]
        if "task" not in batch:
            raise KeyError("Missing language instruction: task")
        # Instruction
        tasks = self._prepare_tasks(
            batch["task"], batch_size
        )

        batch["task"] = tasks
        tokenized = self._tokenize(tasks)
        batch.update(tokenized)
        # Match the original ordering: device then normalize.
        batch = self._to_device(batch)
        batch[STATE_KEY] = self._transform(
            batch[STATE_KEY],
            STATE_KEY,
            "STATE",
        )
        if ACTION_KEY in batch:
            batch[ACTION_KEY] = self._transform(batch[ACTION_KEY], ACTION_KEY, "ACTION",)
        return batch

    def postprocess(self, action: torch.Tensor,) -> torch.Tensor:
        """Normalized model action -> raw CPU action."""
        if not isinstance(action, torch.Tensor):
            raise TypeError("Action must be a torch.Tensor")
        # Remove model's zero-padded action dimensions.
        if self.action_dim is not None:
            if action.shape[-1] < self.action_dim:
                raise ValueError("Predicted action has too few dimensions")

            action = action[..., :self.action_dim]

        # Normalized tensor to normal-scaled tensor
        action = self._transform(action, ACTION_KEY, "ACTION", inverse=True,)

        return action.to("cpu")

def make_smolvla_pre_post_processors(config: Any, dataset_stats: dict | None = None, *, tokenizer: Any | None = None, rename_map: dict[str, str] | None = None,):
    """Create a simple preprocessor/postprocessor pair."""
    processor = SmolVLAProcessor(
        config=config,
        dataset_stats=dataset_stats,
        tokenizer=tokenizer,
        rename_map=rename_map,
    )
    return processor.preprocess, processor.postprocess