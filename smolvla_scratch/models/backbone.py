"""Download VLM/processor, freeze VLM and run vision/text embedding lookup."""
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, SmolVLMForConditionalGeneration


def load_backbone(model_id, *, load_vlm_weights=True, num_vlm_layers=-1,
                  vlm=None, processor=None, dtype=torch.float32):
    """return (VLM, processor). Expert will be always created inside action_expert.py.

    load_vlm_weights=False create the random VLM weights.
    """
    if vlm is None:
        if load_vlm_weights:
            vlm = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=dtype, low_cpu_mem_usage=True,
            )
        else:
            vlm = SmolVLMForConditionalGeneration(config=AutoConfig.from_pretrained(model_id))
    processor = processor if processor is not None else AutoProcessor.from_pretrained(model_id)
    if num_vlm_layers > 0:
        # Cut the first N text layers 
        vlm.model.text_model.layers = vlm.model.text_model.layers[:num_vlm_layers]
    if not vlm.model.text_model.layers:
        raise ValueError("VLM needs at least one text layer")
    return vlm, processor


def set_backbone_requires_grad(vlm, *, freeze_vision_encoder, train_expert_only, num_expert_layers):
    """Keep the lerobot's policy"""
    num_vlm_layers = len(vlm.model.text_model.layers)
    # Freeze the vision encoder 
    if freeze_vision_encoder:
        vlm.model.vision_model.eval()
        for params in vlm.model.vision_model.parameters():
            params.requires_grad = False
    # Freeze the VLM
    if train_expert_only:
        vlm.eval()
        for params in vlm.parameters():
            params.requires_grad = False
    else:
        # To avoid unused params issue with distributed training
        last_layers = [num_vlm_layers - 1]
        if (
            num_vlm_layers != num_expert_layers
            and num_vlm_layers % num_expert_layers == 0
        ):
            last_layers.append(num_vlm_layers - 2)
        frozen_layers = [
            "lm_head",
            "text_model.norm.weight",
        ]
        for layer in last_layers:
            frozen_layers.append(f"text_model.layers.{layer}.")

        unmatched_patterns = set(frozen_layers)
        for name, params in vlm.named_parameters():
            matched_patterns = [k for k in frozen_layers if k in name]
            if matched_patterns:
                params.requires_grad = False
                unmatched_patterns.difference_update(matched_patterns)
        if unmatched_patterns:
            raise RuntimeError(
                "Some frozen layer patterns matched no VLM parameters, so the corresponding layers "
                "would silently remain trainable (parameter naming may have changed in transformers): "
                f"{sorted(unmatched_patterns)}"
            )


def embed_image(vlm, image: torch.Tensor):
    patch_attention_mask = None
    # Get sequence from the vision encoder
    image_hidden_states = (
        vlm.model
        .vision_model(
            pixel_values=image.to(dtype=vlm.model.vision_model.dtype),
            patch_attention_mask=patch_attention_mask,
        )
        .last_hidden_state
    )
    # Modality projection & resampling
    image_hidden_states = vlm.model.connector(image_hidden_states)
    return image_hidden_states


def embed_language_tokens(vlm, tokens: torch.Tensor):
    return vlm.model.text_model.get_input_embeddings()(tokens)


def keep_frozen_backbone_in_eval(vlm, *, freeze_vision_encoder, train_expert_only):
    """Calling after policy.train(mode)."""
    if freeze_vision_encoder:
        vlm.model.vision_model.eval()
    if train_expert_only:
        vlm.eval()
