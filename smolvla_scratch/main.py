import argparse
import random
from pathlib import Path
import numpy as np
from smolvla_scratch.configuration.configuration_smolvla import SmolVLAConfig
from smolvla_scratch.data_process.processor_smolvla import SmolVLAProcessor
from smolvla_scratch.models.policy import SmolVLAPolicy

import torch


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed):
    """Seed cho split, shuffle, khởi tạo model và noise khi train."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()

    # Train
    parser.add_argument("--dataset_dir")
    parser.add_argument("--ckpt_dir")
    parser.add_argument("--task", help="Override the instruction for all episodes; otherwise read each episode's metadata.json")
    parser.add_argument("--camera_names", nargs="+", default=["front", "wrist"])
    parser.add_argument("--num_episodes", type=int)
    parser.add_argument("--legacy_joint_order", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--n_action_steps", type=int, help="Default: chunk_size")
    parser.add_argument("--num_euler_steps", type=int, default=10, help="Euler denoising steps at inference")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--vlm_model_name", default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct")

    # Eval 
    mode.add_argument("--eval", action="store_true", help="Offline validation + sampled action error")
    mode.add_argument("--rollout", action="store_true", help="Run an explicitly supplied environment")
    parser.add_argument("--episode_len", type=int, default=1000)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--checkpoint", help="Checkpoint created by this script; required for eval/rollout")
    # Dont need
    parser.add_argument("--env_factory", help="Optional custom factory; default low_control.low_level_control:make_real_env")
    parser.add_argument("--env_kwargs", help="JSON file containing environment factory keyword arguments")
    
    return parser

def make_policy(config):
    """Dùng implementation trong repo, không chuyển sang policy của LeRobot."""
    return SmolVLAPolicy(config).to(config.device)


def make_optimizer(policy):
    return policy.config.get_optimizer_preset().build(policy.get_optim_params())


def make_processor(policy, stats):
    # Tái dùng tokenizer đã tải cùng backbone, tránh tải bản thứ hai.
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    return SmolVLAProcessor(policy.config, stats, tokenizer=tokenizer)

