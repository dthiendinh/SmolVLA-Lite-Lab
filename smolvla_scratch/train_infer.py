"""Train/evaluate local SmolVLA on ACT-Rebuild recordings.
"""

from __future__ import annotations

from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
import random
import sys

# Cho phép cả python file.py lẫn python -m package.module.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from smolvla_scratch.main import (
    build_parser, choose_device, make_policy, make_optimizer, make_processor,
)
from smolvla_scratch.configuration.configuration_smolvla import SmolVLAConfig
from smolvla_scratch.models.rtc_processor import RTCAttentionSchedule
from smolvla_scratch.utils.vla_utils import (
    ACTION, STATE, JOINT_NAMES, EpisodicDataset, read_episode,
)

def set_seed(seed):
    """Seed cho split, shuffle, khởi tạo model và noise khi train."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_norm_stats(dataset):
    """Chỉ tính trên các timestep thật của TRAIN, không dùng validation/padding."""
    stats = {}
    for key, column in ((STATE, 0), (ACTION, 1)):
        values = np.concatenate([pair[column] for pair in dataset.arrays]).astype(np.float64)
        stats[key] = {
            "mean": values.mean(axis=0).tolist(),
            "std": np.maximum(values.std(axis=0), 1e-2).tolist(),
        }
    return stats


def load_data(args, metadata=None):
    """Split theo episode; eval dùng lại chính split đã lưu trong checkpoint."""
    root = Path(args.dataset_dir).expanduser().resolve()
    if metadata is None:
        names = sorted(path.name for path in root.glob("episode_*") if path.is_dir())
        if args.num_episodes is not None:
            names = names[:args.num_episodes]
        if len(names) < 2:
            raise ValueError("Need at least two episodes for separate train/validation splits")
        shuffled = np.random.default_rng(args.seed).permutation(names).tolist()
        n_val = max(1, min(len(names) - 1, math.ceil(len(names) * args.val_ratio)))
        metadata = {
            "train_episodes": shuffled[n_val:], "val_episodes": shuffled[:n_val],
            "camera_names": args.camera_names, "task": args.task,
            "chunk_size": args.chunk_size, "legacy_joint_order": args.legacy_joint_order,
            "joint_names": JOINT_NAMES, "validation_batch_size": args.batch_size,
        }
    common = dict(
        dataset_dir=root, camera_names=metadata["camera_names"],
        chunk_size=metadata["chunk_size"], task=metadata["task"],
        legacy_joint_order=metadata["legacy_joint_order"],
        episode_tasks=metadata.get("episode_tasks"),
    )
    val = EpisodicDataset(episodes=metadata["val_episodes"], **common)
    batch_size = metadata.get("validation_batch_size", args.batch_size) if args.eval else args.batch_size
    options = dict(batch_size=batch_size, num_workers=args.num_workers,
                   pin_memory=str(args.device).startswith("cuda"))
    val_loader = DataLoader(val, shuffle=False, **options)
    if args.eval:
        return None, val_loader, None, metadata
    train = EpisodicDataset(episodes=metadata["train_episodes"], **common)
    if train.image_shapes != val.image_shapes:
        raise ValueError("Train/validation camera resolutions differ")
    metadata["episode_tasks"] = {**train.episode_tasks, **val.episode_tasks}
    tasks = set(metadata["episode_tasks"].values())
    if metadata["task"] is None and len(tasks) == 1:
        metadata["task"] = next(iter(tasks))
    return DataLoader(train, shuffle=True, **options), val_loader, get_norm_stats(train), metadata


# Put one batch into model to get loss
def forward_pass(data, policy, processor, *, generator=None):
    """raw dict -> normalize/tokenize/device -> (scalar loss, metrics)."""
    # Pre process the data
    batch = processor.preprocess(data)
    # Giá trị pad ở raw space có thể normalize thành khác 0: xóa lại sau normalize.
    batch[ACTION] = batch[ACTION].masked_fill(batch["action_is_pad"].unsqueeze(-1), 0)
    kwargs = {}

    # When training, dont use generator: model take the arbitary noise/ time 
    # When validation, use fixed-seed generator, noise/time will be controled to compare losses between epoches
    if generator is not None:
        # Sinh noise/time cố định trên CPU để validation lặp lại được, kể cả MPS.
        b, t = batch[ACTION].shape[:2]
        noise = torch.randn(b, t, policy.config.max_action_dim, generator=generator)
        # Beta(1.5, 1) inverse CDF, cùng phân phối model dùng khi train.
        time = torch.rand(b, generator=generator).pow(1 / 1.5) * 0.999 + 0.001
        kwargs = dict(noise=noise.to(policy.config.device), time=time.to(policy.config.device))
    return policy(batch, **kwargs) # loss, metrics


@torch.no_grad()
def validate(policy, loader, processor, seed, *, action_metrics=False):
    """Mean batch flow loss, matching train; optionally measure sampled action error."""
    policy.eval()
    # noise/time to compute flow loss
    generator = torch.Generator().manual_seed(seed)
    # noise to denoise action chunk
    sampling_generator = torch.Generator().manual_seed(seed + 1)
    losses = []
    squared_error = absolute_error = action_count = 0
    for data in loader:
        loss, _ = forward_pass(data, policy, processor, generator=generator)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")
        losses.append(loss.item())
        # compute action_MSE/MAE 
        if action_metrics:
            batch = processor.preprocess(data)
            shape = (*data[ACTION].shape[:2], policy.config.max_action_dim)
            noise = torch.randn(shape, generator=sampling_generator).to(policy.config.device)
            predicted = processor.postprocess(policy.predict_action_chunk(batch, noise=noise))
            valid = (~data["action_is_pad"]).unsqueeze(-1).expand_as(predicted)
            error = (predicted - data[ACTION])[valid]
            if not torch.isfinite(error).all():
                raise RuntimeError("Non-finite sampled action error")
            squared_error += error.square().sum().item()
            absolute_error += error.abs().sum().item()
            action_count += error.numel()
    if not losses:
        raise ValueError("Validation loader has no batches")
    result = {"flow_loss": sum(losses) / len(losses)}
    if action_metrics:
        if not action_count:
            raise ValueError("No valid action targets to evaluate")
        result.update(action_mse=squared_error / action_count,
                      action_mae=absolute_error / action_count)
    return result


def save_checkpoint(path, policy, stats, metadata, epoch, best_loss):
    """Store weights together with config/stats/split so eval can restore them."""
    payload = {
        # Store nested enums as plain strings, compatible with weights_only=True.
        "format_version": 1, "policy_config": json.loads(json.dumps(asdict(policy.config))),
        "model": policy.state_dict(), "dataset_stats": stats,
        "data": metadata, "epoch": epoch, "best_val_loss": best_loss,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    # Older local RTC checkpoints stored this known Enum directly.
    with torch.serialization.safe_globals([RTCAttentionSchedule]):
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Checkpoint needs model/config/stats/data; a weights-only file is insufficient")
    if payload["data"]["joint_names"] != JOINT_NAMES:
        raise ValueError("Unsupported checkpoint joint schema")
    policy_config = SmolVLAConfig(**{**payload["policy_config"], "device": str(device),
                                   "load_vlm_weights": False})
    policy = make_policy(policy_config)
    policy.load_state_dict(payload["model"], strict=True)
    policy.eval()
    return policy, make_processor(policy, payload["dataset_stats"]), payload


def train_bc(train_dataloader, val_dataloader, stats, config):
    """ACT-style trainer. config: num_epochs, ckpt_dir, seed, policy_config, data.

    Return (best_epoch, best_val_loss, best_state_dict), as in ACT.
    Both train and validation report the arithmetic mean of batch losses.
    """
    num_epochs = config['num_epochs']
    ckpt_dir = Path(config['ckpt_dir'])
    seed = config['seed']
    policy_config = config['policy_config']
    metadata = {**config['data'], "validation_seed": seed}
    save_every = config.get('save_every', 100)
    if num_epochs < 1 or save_every < 1 or not len(train_dataloader) or not len(val_dataloader):
        raise ValueError("Positive epochs/save_every and nonempty loaders are required")
    if isinstance(policy_config, dict):
        policy_config = SmolVLAConfig(**policy_config)
    if policy_config.device in (None, "auto"):
        policy_config.device = str(choose_device())
    set_seed(seed)
    # Checkpoint dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Policy
    policy = make_policy(policy_config)
    # Optimizer
    optimizer = make_optimizer(policy)
    # Scheduler
    scheduler = policy.config.get_scheduler_preset().build(
        optimizer, num_training_steps=num_epochs * len(train_dataloader))
    # Processor
    processor = make_processor(policy, stats)

    train_history = []
    validation_history = []
    history = []
    min_val_loss = math.inf
    best_ckpt_info = None
    with tqdm(total=num_epochs, desc="Training", dynamic_ncols=True) as progress:
        for epoch in range(num_epochs):

            # Train
            policy.train()
            batch_losses = []  # Reset each epoch, not once for the entire run.
            for data in train_dataloader:
                optimizer.zero_grad(set_to_none=True)
                loss, _ = forward_pass(data, policy, processor)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss; inspect data and learning rate")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad],
                    policy.config.optimizer_grad_clip_norm, error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                batch_losses.append(loss.item())
            epoch_train_loss = sum(batch_losses) / len(batch_losses)
            train_history.append({"loss": epoch_train_loss})

            # Validation
            # Reuse exactly this validation procedure during offline eval.
            val = validate(policy, val_dataloader, processor, seed)
            epoch_val_loss = val["flow_loss"]
            validation_history.append({"loss": epoch_val_loss})
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                # Keep the best snapshot on CPU rather than occupying extra GPU memory.
                best_state = {name: value.detach().cpu().clone()
                              for name, value in policy.state_dict().items()}
                best_ckpt_info = (epoch, min_val_loss, best_state)
                save_checkpoint(ckpt_dir / "policy_best.ckpt", policy, stats, metadata, epoch, min_val_loss)

            # Save the checkpoint
            save_checkpoint(ckpt_dir / "policy_last.ckpt", policy, stats, metadata, epoch, min_val_loss)
            if (epoch + 1) % save_every == 0:
                save_checkpoint(ckpt_dir / f"policy_epoch_{epoch}_seed_{seed}.ckpt",
                                policy, stats, metadata, epoch, min_val_loss)
            history.append(dict(epoch=epoch, train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                lr=optimizer.param_groups[0]["lr"]))
            (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            progress.set_postfix(train=f"{epoch_train_loss:.4f}", val=f"{epoch_val_loss:.4f}",
                                 best=f"{min_val_loss:.4f}", refresh=False)
            progress.update(1)

    best_epoch, min_val_loss, _ = best_ckpt_info
    print(f"Training finished: seed={seed}, best val={min_val_loss:.6f} at epoch {best_epoch}")
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    """History entries contain Python floats; plotting is optional."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install matplotlib to generate loss plots; history.json is available.")
        return
    for key in train_history[0]:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(train_history) + 1),
                [float(row[key]) for row in train_history], label="train")
        ax.plot(range(1, len(validation_history) + 1),
                [float(row[key]) for row in validation_history], label="validation")
        ax.set(xlabel="Epoch", ylabel=key, title=key)
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(ckpt_dir) / f"train_val_{key}_seed_{seed}.png")
        plt.close(fig)


def eval_bc(policy, processor, payload, args):
    """Offline: dùng validation split và normalization đã lưu khi train."""
    _, loader, _, _ = load_data(args, payload["data"])
    result = validate(policy, loader, processor, payload["data"].get("validation_seed", args.seed), action_metrics=True)
    destination = Path(args.checkpoint).with_suffix(".eval.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Saved: {destination}")
    return result


def make_rollout_env(payload, args):
    """Validate policy/environment compatibility before any robot connection."""
    if payload['data']['joint_names'] != JOINT_NAMES:
        raise ValueError("Unsupported rollout joint order")
    cameras = payload['data']['camera_names']
    expected = [f"observation.images.{name}" for name in cameras]
    policy_cameras = {key for key, feature in payload['policy_config']['input_features'].items()
                      if feature['type'] == 'VISUAL'}
    if not cameras or len(set(cameras)) != len(cameras) or set(expected) != policy_cameras:
        raise ValueError("Checkpoint camera metadata does not match policy input features")
    module_name = args.env_factory or 'low_control.low_level_control:make_real_env'
    module, separator, name = module_name.partition(':')
    if not separator or not module or not name:
        raise ValueError("--env_factory must have the form package.module:function")
    kwargs = json.loads(Path(args.env_kwargs).read_text()) if args.env_kwargs else {}
    if not isinstance(kwargs, dict):
        raise ValueError("--env_kwargs must contain a JSON object")
    environment = importlib.import_module(module)
    if module == 'low_control.low_level_control':
        if list(environment.JOINT_KEYS) != JOINT_NAMES:
            raise ValueError("Environment joint order differs from checkpoint")
        if 'camera_names' in kwargs and list(kwargs['camera_names']) != list(cameras):
            raise ValueError("Environment camera_names must match checkpoint camera_names")
        kwargs['camera_names'] = cameras
    return getattr(environment, name)(**kwargs)


def rollout_bc(policy, processor, payload, args):
    """Adapter cho env kiểu ACT: reset/step -> ts.observation[qpos, images].
    """
    if args.num_rollouts < 1 or args.episode_len < 1:
        raise ValueError("num_rollouts and episode_len must be positive")
    if policy.config.rtc_config is not None and policy.config.rtc_config.enabled:
        raise ValueError("Synchronous rollout uses select_action; disable RTC for this mode")
    task = args.task if args.task is not None else payload['data']['task']
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Rollout needs a nonempty language instruction")
    policy.eval()
    env = make_rollout_env(payload, args)
    summaries = []
    try:
        for rollout in range(args.num_rollouts):
            policy.reset()  # Delete previous episode's action queue
            ts = env.reset()    # Reset robot's position
            rewards = []
            steps = 0
            with torch.no_grad():
                for _ in range(args.episode_len):
                    obs = ts.observation

                    # QPOS
                    qpos = np.asarray(obs["qpos"], dtype=np.float32)
                    if qpos.shape != (6,) or not np.isfinite(qpos).all():
                        raise ValueError("Environment qpos must contain six finite canonical joint values")
                    data = {STATE: torch.from_numpy(qpos.copy()), "task": task}

                    # Camera
                    for camera in payload["data"]["camera_names"]:
                        rgb = np.asarray(obs["images"][camera])
                        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                            raise ValueError("Environment images must be HWC RGB uint8")
                        data[f"observation.images.{camera}"] = torch.from_numpy(rgb.copy()).permute(2, 0, 1)

                    batch = processor.preprocess(data)

                    # Ouput action
                    normalized_action = policy.select_action(batch)
                    action = processor.postprocess(normalized_action)[0].numpy()
                    if action.shape != (6,) or not np.isfinite(action).all():
                        raise RuntimeError("Predicted action must contain six finite values")

                    
                    ts = env.step(action)
                    steps += 1

                    if ts.reward is not None:
                        reward = float(ts.reward)
                        if not math.isfinite(reward):
                            raise ValueError("Environment returned non-finite reward")
                        rewards.append(reward)
                    if hasattr(ts, "last") and ts.last():
                        break

            summary = {"rollout": rollout, "steps": steps,
                       "return": sum(rewards) if rewards else None}
                       
            summaries.append(summary)
            print(json.dumps(summary))
    finally:
        env.close()
    # Không tự suy ra success rate khi env chưa định nghĩa tiêu chí thành công.
    destination = Path(args.checkpoint).with_suffix('.rollout.json')
    destination.write_text(json.dumps(summaries, indent=2) + '\n')
    return summaries


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # Check the args param
    for name in ("batch_size", "num_epochs", "chunk_size", "num_euler_steps", "save_every", "episode_len", "num_rollouts"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.num_workers < 0 or not 0 < args.val_ratio < 1:
        parser.error("num_workers must be nonnegative; val_ratio must be between 0 and 1")
    if not math.isfinite(args.lr) or args.lr <= 0 or args.lr < 2.5e-6:
        parser.error("--lr must be finite and >= the scheduler floor (2.5e-6)")
    if args.num_episodes is not None and args.num_episodes < 2:
        parser.error("--num_episodes must be at least 2")
    if args.n_action_steps is None:
        args.n_action_steps = args.chunk_size
    if not 1 <= args.n_action_steps <= args.chunk_size:
        parser.error("Require 1 <= n_action_steps <= chunk_size")
    if len(set(args.camera_names)) != len(args.camera_names) or any(
        not name or Path(name).name != name or name in (".", "..") for name in args.camera_names
    ):
        parser.error("camera_names must be unique file stems")
    if args.eval or args.rollout:
        if not args.checkpoint or (args.eval and not args.dataset_dir):
            parser.error("eval needs checkpoint/dataset_dir; rollout needs checkpoint")
    elif not args.dataset_dir or not args.ckpt_dir:
        parser.error("Training needs --dataset_dir and --ckpt_dir")
    elif args.checkpoint:
        parser.error("--checkpoint is for eval/rollout; training initializes a new action expert")
    if args.task is not None and not args.task.strip():
        parser.error("--task must be nonempty when supplied")
    
    # Choose device
    args.device = str(choose_device(args.device))
    set_seed(args.seed)
    print(f"Using device: {args.device}", flush=True)

    # Eval 
    if args.eval or args.rollout:
        policy, processor, payload = load_checkpoint(args.checkpoint, args.device)
        if args.rollout:
            return rollout_bc(policy, processor, payload, args)
        return eval_bc(policy, processor, payload, args)

    output = Path(args.ckpt_dir).expanduser().resolve()
    # Tránh vô tình ghi đè một lần train cũ.
    if output.exists() and any(output.iterdir()):
        parser.error("--ckpt_dir must be empty or new")
    args.ckpt_dir = str(output)
    train, val, stats, metadata = load_data(args)

    # Model config
    policy_config = SmolVLAConfig(
        device=args.device, chunk_size=args.chunk_size, n_action_steps=args.n_action_steps,
        num_steps=args.num_euler_steps, vlm_model_name=args.vlm_model_name,
        optimizer_lr=args.lr, learning_rate=args.lr,
        input_features={STATE: {"shape": (len(JOINT_NAMES),), "type": "STATE"}, **{
            f"observation.images.{camera}": {"shape": shape, "type": "VISUAL"}
            for camera, shape in train.dataset.image_shapes.items()
        }},
        load_vlm_weights=True, train_expert_only=True, freeze_vision_encoder=True,
    )
    metadata["validation_seed"] = args.seed

    # Training's Configs
    config = dict(num_epochs=args.num_epochs, 
                    ckpt_dir=args.ckpt_dir, 
                    seed=args.seed,
                    policy_config=policy_config, 
                    data=metadata, 
                    save_every=args.save_every)

    output.mkdir(parents=True, exist_ok=True)

    (output / "training_config.json").write_text(json.dumps({
        "arguments": vars(args), "policy_config": asdict(policy_config),
        "dataset_stats": stats, "data": metadata,
    }, indent=2) + "\n")

    print(f"Train: {len(train.dataset)} frames; validation: {len(val.dataset)} frames", flush=True)
    return train_bc(train, val, stats, config)



if __name__ == "__main__":
    main()
