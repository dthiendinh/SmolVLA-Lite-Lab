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
from smolvla_scratch.checkpointing import (
    capture_rng_state, restore_rng_state, dataset_manifest, read_checkpoint, save_checkpoint,
)
from smolvla_scratch.utils.vla_utils import (
    ACTION, STATE, JOINT_NAMES, EpisodicDataset,
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


def load_data(args, metadata=None, stats=None):
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
    return DataLoader(train, shuffle=True, **options), val_loader, (get_norm_stats(train) if stats is None else stats), metadata


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
    per_joint_squared = torch.zeros(len(JOINT_NAMES), dtype=torch.float64)
    per_joint_absolute = torch.zeros_like(per_joint_squared)
    valid_timesteps = 0
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
            # Các khớp góc và gripper có thang đo khác nhau: lưu thêm lỗi
            # từng khớp để một số trung bình không che mất khớp bị dự đoán kém.
            joint_error = (predicted - data[ACTION])[~data["action_is_pad"]].double()
            per_joint_squared += joint_error.square().sum(dim=0)
            per_joint_absolute += joint_error.abs().sum(dim=0)
            valid_timesteps += len(joint_error)
    if not losses:
        raise ValueError("Validation loader has no batches")
    result = {"flow_loss": sum(losses) / len(losses)}
    if action_metrics:
        if not action_count:
            raise ValueError("No valid action targets to evaluate")
        result.update(action_mse=squared_error / action_count,
                      action_mae=absolute_error / action_count,
                      num_samples=len(loader.dataset), valid_action_timesteps=valid_timesteps,
                      per_joint={name: {"mse": per_joint_squared[index].item() / valid_timesteps,
                                        "mae": per_joint_absolute[index].item() / valid_timesteps}
                                 for index, name in enumerate(JOINT_NAMES)})
    return result


def load_checkpoint(path, device):
    """Eval/rollout chấp nhận cả model checkpoint và training checkpoint."""
    payload = read_checkpoint(path)
    policy_config = SmolVLAConfig(**{**payload["policy_config"], "device": str(device),
                                   "load_vlm_weights": False})
    policy = make_policy(policy_config)
    policy.load_state_dict(payload["model"], strict=True)
    policy.eval()
    return policy, make_processor(policy, payload["dataset_stats"]), payload


def train_bc(train_dataloader, val_dataloader, stats, config):
    """Train mới, fine-tuning hoặc resume tại ranh giới epoch.

    config['initial_checkpoint'] là payload đã đọc, không phải đường dẫn Hub.
    Resume khôi phục toàn bộ run; fine-tuning chỉ nạp weights. Trả về tuple
    (best_epoch, best_val_loss, best_state_dict), giữ API của trainer cũ.
    """
    num_epochs = config['num_epochs']
    ckpt_dir = Path(config['ckpt_dir'])
    seed = config['seed']
    metadata = {**config['data'], "validation_seed": seed}
    save_every = config.get('save_every', 100)
    if num_epochs < 1 or save_every < 1 or not len(train_dataloader) or not len(val_dataloader):
        raise ValueError("Positive epochs/save_every and nonempty loaders are required")
    initial = config.get('initial_checkpoint')
    resumed = initial['training_state'] if config.get('mode') == 'resume' else None
    start_epoch = initial['epoch'] + 1 if resumed else 0
    if start_epoch >= num_epochs:
        raise ValueError(f"Checkpoint completed {start_epoch} epochs; --num_epochs must be greater than {start_epoch}")
    if resumed and resumed['steps_per_epoch'] != len(train_dataloader):
        raise ValueError("Resume requires the original batch size and number of train batches")

    policy_config = config['policy_config']
    if isinstance(policy_config, dict):
        policy_config = SmolVLAConfig(**policy_config)
    if policy_config.device in (None, "auto"):
        policy_config.device = str(choose_device())
    set_seed(seed)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    policy = make_policy(policy_config)
    if initial is not None:
        # strict=True ngăn bỏ sót layer hoặc trộn hai kiến trúc khác nhau.
        policy.load_state_dict(initial['model'], strict=True)
    optimizer = make_optimizer(policy)

    # ===== MODEL PARAMETERS =====
    total_params = sum(p.numel() for p in policy.parameters())

    trainable_params = sum(
        p.numel() for p in policy.parameters()
        if p.requires_grad
    )

    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
    print(f"Frozen params: {(total_params - trainable_params) / 1e6:.2f}M")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    # LambdaLR không lưu closure lr_lambda trong state_dict. Phải dựng lại
    # bằng horizon GỐC, sau đó mới nạp scheduler và optimizer. Nếu kéo dài
    # num_epochs khi resume, LR vẫn theo lịch cũ và giữ floor khi lịch kết thúc.
    scheduler_total_steps = (resumed['scheduler_total_steps'] if resumed
                             else num_epochs * len(train_dataloader))
    scheduler = policy.config.get_scheduler_preset().build(
        optimizer, num_training_steps=scheduler_total_steps)
    processor = make_processor(policy, stats)
    history = list(resumed['history']) if resumed else []
    global_step = resumed['global_step'] if resumed else 0
    min_val_loss = initial['best_val_loss'] if resumed else math.inf
    best_epoch = resumed['best_epoch'] if resumed else None
    best_state = resumed['best_model'] if resumed else None
    if resumed:
        scheduler.load_state_dict(resumed['scheduler'])
        optimizer.load_state_dict(resumed['optimizer'])
        # Khôi phục RNG cuối cùng: dựng model/processor đã tiêu thụ random.
        restore_rng_state(resumed['rng'])
        # Best snapshot được nhúng trong full checkpoint để resume vào thư
        # mục mới vẫn giữ model tốt nhất, kể cả khi source best file đã mất.
        policy.load_state_dict(best_state, strict=True)
        save_checkpoint(ckpt_dir / "policy_best.ckpt", policy, stats, metadata, best_epoch, min_val_loss)
        # Nạp lại weights cuối epoch từ payload sau khi export best snapshot.
        policy.load_state_dict(initial['model'], strict=True)

    # Model nguồn chỉ cần trong lúc khởi tạo; tránh giữ thừa một backbone CPU.
    # Payload best_state vẫn được giữ riêng để lưu full checkpoint và trả về.
    config.pop('initial_checkpoint', None)
    del initial, resumed
    with tqdm(total=num_epochs, initial=start_epoch, desc="Training", dynamic_ncols=True) as progress:
        for epoch in range(start_epoch, num_epochs):
            policy.train()
            batch_losses = []
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
                global_step += 1
                batch_losses.append(loss.item())
            epoch_train_loss = sum(batch_losses) / len(batch_losses)

            # Noise/time validation dùng generator riêng và seed cố định,
            # cùng quy trình với đánh giá offline để so sánh checkpoint.
            val = validate(policy, val_dataloader, processor, seed)
            epoch_val_loss = val["flow_loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_epoch = epoch
                best_state = {name: value.detach().cpu().clone()
                              for name, value in policy.state_dict().items()}
                # File model nhẹ hơn: không có optimizer/scheduler/RNG/history.
                save_checkpoint(ckpt_dir / "policy_best.ckpt", policy, stats, metadata, epoch, min_val_loss)

            history.append(dict(epoch=epoch, train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                lr=optimizer.param_groups[0]["lr"], global_step=global_step))
            training_state = {
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": capture_rng_state(), "history": history,
                "best_epoch": best_epoch, "best_model": best_state,
                "global_step": global_step, "scheduler_total_steps": scheduler_total_steps,
                "steps_per_epoch": len(train_dataloader),
                "arguments": config.get('arguments', {}),
                "dataset_manifest": config.get('dataset_manifest', {}),
            }
            # Checkpoint full luôn ở cuối epoch: resume lặp lại phần epoch
            # chưa hoàn thành nếu tiến trình bị ngắt giữa epoch tiếp theo.
            save_checkpoint(ckpt_dir / "policy_last.ckpt", policy, stats, metadata, epoch,
                            min_val_loss, training_state=training_state)
            if (epoch + 1) % save_every == 0:
                save_checkpoint(ckpt_dir / f"policy_epoch_{epoch}_seed_{seed}.ckpt",
                                policy, stats, metadata, epoch, min_val_loss,
                                training_state=training_state)
            (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            progress.set_postfix(train=f"{epoch_train_loss:.4f}", val=f"{epoch_val_loss:.4f}",
                                 best=f"{min_val_loss:.4f}", refresh=False)
            progress.update(1)

    # Export model cuối cùng riêng để fine-tuning/eval không phải mang theo
    # optimizer và best snapshot. Full policy_last.ckpt vẫn dùng để resume.
    save_checkpoint(ckpt_dir / "policy_last_model.ckpt", policy, stats, metadata,
                    num_epochs - 1, min_val_loss)
    print(f"Training finished: seed={seed}, best val={min_val_loss:.6f} at epoch {best_epoch}")
    plot_history([{"loss": row['train_loss']} for row in history],
                 [{"loss": row['val_loss']} for row in history], num_epochs, ckpt_dir, seed)
    return best_epoch, min_val_loss, best_state


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
    if "dataset_manifest" in payload["data"]:
        # Eval chỉ cần các episode validation, không bắt buộc giữ train data.
        validation_data = {**payload["data"], "train_episodes": []}
        manifest = dataset_manifest(Path(args.dataset_dir).expanduser(), validation_data)
        expected = {name: digest for name, digest in payload["data"]["dataset_manifest"].items()
                    if name.split('/', 1)[0] in payload["data"]["val_episodes"]}
        if manifest != expected:
            raise ValueError("Validation data changed since checkpoint; cannot compare on the saved holdout")
    _, loader, _, _ = load_data(args, payload["data"])
    result = validate(policy, loader, processor, payload["data"].get("validation_seed", args.seed), action_metrics=True)
    result['validation_seed'] = payload["data"].get("validation_seed", args.seed)
    result['checkpoint_epoch'] = payload['epoch']
    destination = Path(args.checkpoint).expanduser().with_suffix(".eval.json")
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


def inherit_training_arguments(args, payload, supplied, parser):
    """Không để default CLI âm thầm thay cấu hình của checkpoint.

    Resume giữ các tham số ảnh hưởng phép train. Fine-tuning giữ kiến trúc,
    còn split/seed/batch size/LR là của run mới. Chỉ các option được người
    dùng gõ rõ mới được coi là yêu cầu override.
    """
    if args.resume:
        previous = payload['training_state']['arguments']
        protected = ("batch_size", "seed", "val_ratio", "num_episodes", "task",
                     "legacy_joint_order", "camera_names", "chunk_size", "n_action_steps",
                     "num_euler_steps", "vlm_model_name", "lr", "finetune_stats")
        for name in protected:
            if name not in previous:
                parser.error(f"Resume checkpoint is missing the original argument: {name}")
            if name in supplied and getattr(args, name) != previous[name]:
                parser.error(f"Resume must preserve --{name}; use --finetune to change training settings")
            setattr(args, name, previous[name])
        if args.dataset_dir is None:
            args.dataset_dir = previous.get('dataset_dir')
        if args.ckpt_dir is None:
            args.ckpt_dir = str(Path(args.resume).expanduser().resolve().parent)
        if 'num_epochs' not in supplied:
            args.num_epochs = previous['num_epochs']
    elif args.finetune:
        source = payload['policy_config']
        architecture = {
            "chunk_size": source['chunk_size'], "vlm_model_name": source['vlm_model_name'],
            "camera_names": payload['data']['camera_names'],
        }
        for name, value in architecture.items():
            if name in supplied and getattr(args, name) != value:
                parser.error(f"Fine-tuning must preserve --{name} to match source policy")
            setattr(args, name, value)
        for argument, field in (("n_action_steps", "n_action_steps"), ("num_euler_steps", "num_steps")):
            if argument not in supplied:
                setattr(args, argument, source[field])


def validate_dataset_policy(dataset, policy_config):
    """Chặn sai tên camera/state/action trước khi bước train đầu tiên chạy."""
    camera_keys = {key for key, feature in policy_config.input_features.items()
                   if feature['type'] == 'VISUAL'}
    expected = {f"observation.images.{name}" for name in dataset.camera_names}
    if camera_keys != expected:
        raise ValueError("Dataset cameras do not match source policy input features")
    if (list(policy_config.input_features[STATE]['shape']) != [len(JOINT_NAMES)]
            or list(policy_config.output_features[ACTION]['shape']) != [len(JOINT_NAMES)]):
        raise ValueError("Source policy must use the six canonical robot joints")


def main(argv=None):
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv)
    # Hỗ trợ cả --lr 1e-5 và --lr=1e-5 khi nhận diện override tường minh.
    supplied = {parser._option_string_actions[token.split('=', 1)[0]].dest
                for token in argv if token.split('=', 1)[0] in parser._option_string_actions}
    initial = None
    if args.resume or args.finetune:
        initial = read_checkpoint(args.resume or args.finetune, for_resume=bool(args.resume))
        inherit_training_arguments(args, initial, supplied, parser)

    for name in ("batch_size", "num_epochs", "chunk_size", "num_euler_steps", "save_every", "episode_len", "num_rollouts"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be positive")
    if args.num_workers < 0 or not 0 < args.val_ratio < 1:
        parser.error("num_workers must be nonnegative; val_ratio must be between 0 and 1")
    if not math.isfinite(args.lr) or args.lr < 2.5e-6:
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
        parser.error("Training needs --dataset_dir and --ckpt_dir (resume can inherit both)")
    elif args.checkpoint:
        parser.error("--checkpoint is for eval/rollout; use --finetune or --resume for training")
    if args.task is not None and not args.task.strip():
        parser.error("--task must be nonempty when supplied")
    if 'finetune_stats' in supplied and not args.finetune:
        parser.error("--finetune_stats is only valid with --finetune")

    args.device = str(choose_device(args.device))
    set_seed(args.seed)
    print(f"Using device: {args.device}", flush=True)
    if args.eval or args.rollout:
        policy, processor, payload = load_checkpoint(args.checkpoint, args.device)
        if args.rollout:
            return rollout_bc(policy, processor, payload, args)
        return eval_bc(policy, processor, payload, args)

    output = Path(args.ckpt_dir).expanduser().resolve()
    # Chỉ resume từ last của chính thư mục đó mới được ghi tiếp. Khi dùng
    # checkpoint epoch cũ, phải tạo thư mục mới để không ghi đè lịch sử mới hơn.
    same_run = bool(args.resume and Path(args.resume).expanduser().resolve() == output / 'policy_last.ckpt')
    if output.exists() and any(output.iterdir()) and not same_run:
        parser.error("--ckpt_dir must be empty/new; only --resume from its policy_last.ckpt may reuse it")
    args.ckpt_dir = str(output)
    args.dataset_dir = str(Path(args.dataset_dir).expanduser().resolve())
    if args.resume and args.num_epochs <= initial['epoch'] + 1:
        parser.error(f"Checkpoint already completed {initial['epoch'] + 1} epochs; increase --num_epochs")

    if args.resume:
        # Không split lại hay tính stats mới. Kiểm tra byte dữ liệu trước khi
        # mở loader để tránh vô tình gọi dataset đã sửa là cùng một run.
        manifest = dataset_manifest(args.dataset_dir, initial['data'])
        if manifest != initial['training_state']['dataset_manifest']:
            parser.error("Dataset content changed since checkpoint; use --finetune for a new dataset")
        train, val, stats, metadata = load_data(args, initial['data'], initial['dataset_stats'])
    else:
        # Fine-tuning có split/task mới; mặc định giữ normalization nguồn để
        # ánh xạ state/action mà weights đã học không đột ngột đổi ý nghĩa.
        source_stats = (initial['dataset_stats'] if args.finetune and args.finetune_stats == 'checkpoint' else None)
        train, val, stats, metadata = load_data(args, stats=source_stats)
        print("Fingerprinting dataset for future resume...", flush=True)
        manifest = dataset_manifest(args.dataset_dir, metadata)

    if initial is not None:
        source_config = {**initial['policy_config'], 'device': args.device, 'load_vlm_weights': False}
        if args.finetune:
            source_config.update(optimizer_lr=args.lr, learning_rate=args.lr,
                                 n_action_steps=args.n_action_steps, num_steps=args.num_euler_steps)
        policy_config = SmolVLAConfig(**source_config)
        validate_dataset_policy(train.dataset, policy_config)
        # Resize preprocessing hỗ trợ độ phân giải mới khi fine-tuning.
        # Resume đã kiểm tra manifest nên độ phân giải vẫn phải là dữ liệu cũ.
        for camera, shape in train.dataset.image_shapes.items():
            policy_config.input_features[f'observation.images.{camera}']['shape'] = shape
    else:
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
    metadata['validation_seed'] = args.seed
    metadata['dataset_manifest'] = manifest
    run_mode = 'resume' if args.resume else ('finetune' if args.finetune else 'new')
    config = dict(num_epochs=args.num_epochs, ckpt_dir=args.ckpt_dir, seed=args.seed,
                  policy_config=policy_config, data=metadata, save_every=args.save_every,
                  mode=run_mode, initial_checkpoint=initial,
                  arguments=vars(args).copy(), dataset_manifest=manifest)
    del initial
    output.mkdir(parents=True, exist_ok=True)
    # Giữ nguyên mô tả run gốc khi resume vào cùng thư mục.
    filename = 'resume_config.json' if same_run else 'training_config.json'
    (output / filename).write_text(json.dumps({
        'mode': run_mode, 'arguments': vars(args), 'policy_config': asdict(policy_config),
        'dataset_stats': stats, 'data': metadata,
    }, indent=2) + "\n")
    print(f"Mode: {run_mode}; train: {len(train.dataset)} frames; validation: {len(val.dataset)} frames", flush=True)
    return train_bc(train, val, stats, config)


if __name__ == "__main__":
    main()
