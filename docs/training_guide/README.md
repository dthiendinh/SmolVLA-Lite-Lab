# Training and Evaluation Guide

[Back to the project README](../../README.md)

This guide covers the complete SO-ARM101 workflow: record demonstrations, verify a dataset, run a smoke test, train an action expert, inspect metrics, resume or fine-tune a checkpoint, evaluate actions offline, and run the policy on the robot.

Run every command from the **repository root** with the project environment activated. Training and offline evaluation do not require a connected robot.

## 1. Install the environment

Clone the repository and create a Python 3.12 environment:

```bash
git clone https://github.com/dthiendinh/SmolVLA-Lite-Lab.git
cd SmolVLA-Lite-Lab
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If the repository is already present, start at `cd SmolVLA-Lite-Lab`. On subsequent terminal sessions, activate the existing environment:

```bash
source .venv/bin/activate
```

The supplied requirements pin PyTorch `2.11.0`, Transformers `5.22.2`, and LeRobot `0.6.1`. Check imports and CLI options before starting a run:

```bash
python -c 'import sys, torch, lerobot; print(sys.executable); print(torch.__version__); print(lerobot.__file__); print("CUDA available:", torch.cuda.is_available())'
python -B -m smolvla_scratch.train_infer --help
```

`--device auto` selects CUDA, then MPS, then CPU. Use `--device cuda` on an NVIDIA training server, `--device mps` on a compatible Mac, or `--device cpu` explicitly. GPU training requires a working PyTorch installation for that backend.

A fresh run downloads the `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` backbone, configuration, and tokenizer. Resume, fine-tuning, and evaluation load weights from the local checkpoint, but still require the VLM configuration/tokenizer in the cache or accessible from the Hub.

### OpenCV for the recording dashboard

The teleoperation dashboard uses `cv2.imshow` and needs a desktop environment and GUI-enabled OpenCV. Inspect installed variants if the dashboard fails:

```bash
python -m pip list | python -c 'import sys; print("".join(line for line in sys.stdin if "opencv" in line.lower()))'
```

If GUI and headless variants conflict, reinstall one GUI variant in the recording environment:

```bash
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install 'opencv-python>=4.10,<4.14'
```

LeRobot also declares headless OpenCV in its dependencies. After this replacement, `pip check` may report that dependency as missing; reinstalling the project requirements can restore the headless variant. Offline server training does not need the recording dashboard.

## 2. Record or prepare demonstrations

For new recordings, configure ports, calibration IDs, camera indices, and the robot reset pose in [`low_control/hardware_constant.py`](../../low_control/hardware_constant.py). Use the same calibration, units, and camera arrangement for recording and rollout.

```bash
python -B -m data_recording.record_teleop_data \
  --data-dir pick_place_smolvla_v1 \
  --task "Pick up an orange marker and put it in the cup"
```

The recorder's `--data-dir` names a subdirectory under `data_recording/data/`. Training uses the separate `--dataset_dir` option with the dataset's actual path.

| Dashboard key | Action |
| --- | --- |
| `t`, then Enter | Edit and confirm the English task instruction before recording |
| `r` | Start an episode |
| `s` | Save the episode |
| `x` | Discard the current episode |
| `q` | Exit |

Record one complete task execution per episode. Inspect both camera streams and the final placement. Save after completing the task, then reset the scene for the next recording; include a return-to-reset motion in demonstrations only if it is part of the behavior you want the policy to learn.

### Dataset layout and schema

Each `episode_*` directory contains:

| File | Required content |
| --- | --- |
| `data.npz` | `states` and `actions` with shape `(T, 6)`, plus `joint_names` |
| `front.mp4` | Exactly `T` front-camera frames |
| `wrist.mp4` | Exactly `T` wrist-camera frames |
| `metadata.json` | A nonempty task, such as `{"task": "Pick up an orange marker and put it in the cup"}` |

Example episode path: `data_recording/data/pick_place_smolvla_v1/episode_0000/`.

At least two valid episodes are needed to form separate training and validation splits; this is a code minimum, not evidence of sufficient training data. If `timestamps` are present, they must contain `T` strictly increasing values. Each camera's resolution must be consistent across episodes and splits.

The canonical joint order is:

```text
shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
wrist_flex.pos, wrist_roll.pos, gripper.pos
```

The loader reorders columns using `joint_names`. The five arm joints use degrees; the gripper uses its LeRobot position scale. Targets are absolute joint positions in this configuration.

By default, the trainer reads each episode's instruction. `--task "..."` overrides the instruction for **every episode** and is appropriate only when the entire dataset represents that instruction. Use `--legacy_joint_order` only for legacy ACT recordings that lack `joint_names` and actually store columns in `sorted(joint_keys)` order.

### Sampling and normalization

The split is made by episode, so observations from one demonstration do not appear in both splits. A sample uses images and state at timestep `t`, the episode instruction, and actions starting at `t` for `chunk_size` steps. Action chunks overlap between neighboring samples. Targets beyond the episode end are padded and excluded from loss.

State/action mean and standard deviation are computed only from actual training timesteps. The processor normalizes these values and pads model dimensions; action postprocessing converts predictions back to the original six-joint scales. Split membership, per-episode tasks, stats, and dataset fingerprints are saved with the checkpoint.

## 3. Run a one-epoch smoke test

This checks data loading, model initialization, loss computation, validation, and checkpoint writing:

```bash
python -B -m smolvla_scratch.train_infer \
  --dataset_dir data_recording/data/pick_place_smolvla_v1 \
  --ckpt_dir outputs/pick_place_smoke \
  --camera_names front wrist \
  --batch_size 4 \
  --num_epochs 1 \
  --chunk_size 50 \
  --lr 1e-4 \
  --seed 0 \
  --num_workers 0 \
  --device auto
```

Use a new or empty output directory. A successful smoke test completes one epoch and writes `outputs/pick_place_smoke/policy_last.ckpt`. Use a separate directory for the main experiment. A successful smoke test establishes that the workflow runs; robot performance requires further evaluation.

## 4. Train a new action expert

The following example uses the batch size, epoch budget, split, and LR of the reported experiment, with zero loader workers as a portable starting point:

```bash
python -B -m smolvla_scratch.train_infer \
  --dataset_dir data_recording/data/pick_place_smolvla_v1 \
  --ckpt_dir outputs/pick_place_run1 \
  --camera_names front wrist \
  --batch_size 32 \
  --num_epochs 30 \
  --chunk_size 50 \
  --n_action_steps 50 \
  --num_euler_steps 10 \
  --lr 1e-4 \
  --val_ratio 0.2 \
  --seed 0 \
  --num_workers 0 \
  --save_every 10 \
  --log_every 10 \
  --device auto
```

These values define an example experiment, not a guarantee of good generalization. The reported run's validation loss was lowest after two epochs. Inspect validation rather than assuming the final epoch is preferable. Reduce batch size if memory is insufficient; use a new run directory when changing training settings. Increase loader workers only after confirming that data loading works in your environment.

The VLM is pretrained and frozen. The action expert and action/state projection components are trained by flow matching. This is a new action expert on a pretrained VLM; it does not load an already action-trained official SmolVLA policy.

At startup, the trainer validates data, fingerprints NPZ/video files, and prints total/trainable/frozen parameter counts. Fingerprinting can take time before the first update.

| Option | Meaning |
| --- | --- |
| `--dataset_dir` | Directory containing the `episode_*` folders |
| `--ckpt_dir` | Output directory; new training and fine-tuning require a new/empty directory |
| `--batch_size` | Number of observation-timestep samples per batch |
| `--num_epochs` | Total target epochs; includes completed epochs during resume |
| `--chunk_size` | Predicted action horizon |
| `--n_action_steps` | Actions executed before predicting another chunk; defaults to `chunk_size` |
| `--num_euler_steps` | Denoising integration steps saved in the policy configuration |
| `--lr` | Peak LR; must be finite and at least the scheduler floor `2.5e-6` |
| `--val_ratio` | Validation episode fraction; rounded up with at least one episode in each split |
| `--seed` | Seed for split, initialization, shuffling, and training randomness |
| `--num_workers` | Data-loader worker processes; start with 0 |
| `--save_every` | Full-checkpoint interval in epochs; `policy_last.ckpt` is still written every epoch |
| `--log_every` | TensorBoard step-logging interval in optimizer updates |
| `--num_episodes N` | Optional subset: first N episode directories in sorted name order |

One epoch visits all observation timesteps in the training split. With 23,516 training timesteps and batch size 32, this is `ceil(23516 / 32) = 735` optimizer updates.

### Learning-rate schedule

The default scheduler configuration has 1,000 warmup steps, a 30,000-step cosine horizon, and an LR floor of `2.5e-6`. If the planned run has fewer than 30,000 updates, the scheduler scales the warmup and horizon down to that budget. For 30 epochs × 735 updates, the budget is 22,050 updates and warmup is scaled to 735 steps.

For runs longer than 30,000 updates, the default horizon remains 30,000; LR then stays at the floor. Extending a run with `--resume` preserves its original schedule instead of creating a new warmup or stretching the decay.

The current trainer validates after each epoch and does not implement automatic early stopping.

## 5. Monitor TensorBoard and select a checkpoint

Open a second terminal, activate the same environment, and run:

```bash
tensorboard --logdir outputs --reload_interval 5 --port 6006
```

Open [http://localhost:6006](http://localhost:6006). If runs are stored elsewhere, change `--logdir` to their parent directory.

| TensorBoard tag | Interpretation |
| --- | --- |
| `loss/train_step` | Training batch flow loss at logged updates |
| `loss/train_epoch` | Mean batch training loss over an epoch |
| `loss/val_epoch` | Mean batch validation flow loss at the end of an epoch |
| `loss/best_val` | Lowest validation flow loss seen so far |
| `learning_rate/step` | LR used for the logged optimizer update |
| `learning_rate/epoch` | LR ready for the next update at the epoch boundary |
| `grad_norm/before_clip` | Gradient norm before clipping |
| `progress/epoch` | Completed epoch count, starting at 1 |

The horizontal axis is global optimizer updates. `Custom Scalars` → `Loss` → `Train vs validation` overlays epoch-level losses. Validation points appear after an epoch finishes. Logs flush approximately every 10 seconds and after each epoch. Add `--log_every 1` to log every update.

For a remote server, keep TensorBoard running on that server and forward its port from your local machine. Replace the SSH placeholders with your server connection details:

```bash
ssh -N -L 6006:127.0.0.1:6006 -p SSH_PORT USER@HOST
```

Then open the same local TensorBoard URL. Run the SSH tunnel in a separate terminal.

### Output files

| File in the run directory | Contents and use |
| --- | --- |
| `training_config.json` | Initial arguments, policy configuration, split, tasks, and normalization stats |
| `history.json` | Epoch-level train/validation loss, next-step LR, and global step |
| `tensorboard/` | Event files for live visualization |
| `policy_best.ckpt` | Model with the lowest validation flow loss; evaluate this checkpoint first |
| `policy_last.ckpt` | Latest full training state, written after every completed epoch; use for resume |
| `policy_last_model.ckpt` | Final model-only export, written after normal training completion |
| `policy_epoch_<index>_seed_<seed>.ckpt` | Periodic full training checkpoint |
| `resume_config.json` | Latest resume arguments when continuing in the same run directory |

Epoch indices in checkpoint metadata and periodic filenames start at **0**. `epoch=1` means two completed epochs; `policy_epoch_9_seed_0.ckpt` means ten completed epochs.

Model checkpoints include the entire policy, including the VLM, plus configuration, stats, and data metadata. Full checkpoints additionally save optimizer/scheduler state, RNG state, history, global step, the original schedule horizon, and the best model snapshot. They can be substantially larger. Resume from a completed `.ckpt` file, not a `.tmp` file.

## 6. Resume an interrupted run

Resume restores the same optimization process from the last saved epoch:

```bash
python -B -m smolvla_scratch.train_infer \
  --resume outputs/pick_place_run1/policy_last.ckpt \
  --device auto
```

Dataset path, batch size, split, LR, and the original epoch target are inherited. If interruption occurs mid-epoch, that incomplete epoch is rerun. No recoverable epoch checkpoint exists until at least one epoch has completed.

To extend a completed 30-epoch run to **40 total epochs**:

```bash
python -B -m smolvla_scratch.train_infer \
  --resume outputs/pick_place_run1/policy_last.ckpt \
  --num_epochs 40 \
  --device auto
```

This adds ten epochs and preserves the original LR schedule. Use fine-tuning for a new LR schedule, changed batch size, task, or dataset.

Resume allows operational changes such as device, worker count, save/log intervals, and dataset location. Dataset bytes and training settings must remain compatible with the saved run. If files have moved, pass their new root with `--dataset_dir`; content fingerprints must still match.

To branch from an older periodic checkpoint, use a new/empty destination:

```bash
python -B -m smolvla_scratch.train_infer \
  --resume outputs/pick_place_run1/policy_epoch_9_seed_0.ckpt \
  --ckpt_dir outputs/pick_place_from_epoch10 \
  --device auto
```

Only resuming a destination's own `policy_last.ckpt` permits reuse of a nonempty run directory. Full checkpoints retain the best model snapshot, so a separate best-checkpoint file is not required to recover it. Exact numerical equivalence across different hardware/backends is not guaranteed.

## 7. Fine-tune a local policy

Fine-tuning loads policy weights into a **new run** and resets optimizer, scheduler, epoch count, history, and best-checkpoint selection:

```bash
python -B -m smolvla_scratch.train_infer \
  --finetune outputs/pick_place_run1/policy_best.ckpt \
  --dataset_dir data_recording/data/pick_place_smolvla_v2 \
  --ckpt_dir outputs/pick_place_finetune1 \
  --batch_size 32 \
  --num_epochs 30 \
  --lr 1e-5 \
  --num_workers 0 \
  --device auto
```

Prepare the new dataset before running the command. Camera names, backbone, and chunk size must match the source checkpoint. Instructions come from the new dataset unless explicitly overridden with `--task`.

By default, `--finetune_stats checkpoint` preserves source normalization. To intentionally recompute stats from the new training split, add `--finetune_stats dataset`. Recomputing changes both the model's state inputs and the mapping of outputs to robot joint units, so evaluate the resulting policy again.

| Mode | Starting weights | Optimizer / scheduler / RNG | History and checkpoint selection |
| --- | --- | --- | --- |
| New training | Pretrained VLM, new action expert/projections | New | Start from epoch 0 |
| `--resume` | Saved full training checkpoint | Restored | Continue the same run |
| `--finetune` | Saved local policy | New | Start a new run at epoch 0 |

Do not pass the Hub ID `lerobot/smolvla_base` to `--finetune`; this CLI does not convert official policy weights into its local format.

## 8. Evaluate actions offline

```bash
python -B -m smolvla_scratch.train_infer --eval \
  --dataset_dir data_recording/data/pick_place_smolvla_v1 \
  --checkpoint outputs/pick_place_run1/policy_best.ckpt \
  --device auto
```

Evaluation reuses the saved validation episodes, instructions, normalization, validation seed, and validation batch size. For current checkpoints, validation NPZ/video fingerprints must match the saved files. Only validation episodes are needed on the evaluation machine.

Results are printed and written beside the checkpoint as `policy_best.eval.json`.

| Field | Meaning |
| --- | --- |
| `flow_loss` | Mean batch velocity-prediction loss with seeded noise/time |
| `action_mse`, `action_mae` | Sampled action errors after denoising and denormalization; exclude padded targets |
| `per_joint` | MSE and MAE for each canonical joint |
| `num_samples` | Validation observation timesteps used as chunk starting points |
| `valid_action_timesteps` | Valid positions across predicted chunks, including overlap between neighboring chunks |
| `validation_seed` | Saved seed used for evaluation |
| `checkpoint_epoch` | Zero-based epoch index of the evaluated weights |

Flow loss is a mean of batch losses. Action metrics sum errors over all valid action values before dividing by the corresponding count. The arm joints and gripper have different scales; use per-joint metrics to interpret physical errors.

Compare best and final checkpoints on the same saved split:

```bash
python -B -m smolvla_scratch.train_infer --eval \
  --dataset_dir data_recording/data/pick_place_smolvla_v1 \
  --checkpoint outputs/pick_place_run1/policy_last_model.ckpt \
  --device auto
```

If training did not finish normally, `policy_last.ckpt` can be evaluated instead. For fine-tuning runs, point to the new dataset and the fine-tuning checkpoint.

Offline metrics measure imitation on demonstration observations. Task completion, motion initiation, recovery, and failure modes require real rollouts. The current `--eval` command evaluates the checkpoint's saved validation split; it does not provide a separate arbitrary test-set mode.

## 9. Run the policy on SO-ARM101

The rollout command connects and moves the follower robot, including an initial move to `RESET_POSE`. Confirm the configured reset pose, ports, calibration, and cameras match your setup. Stop the recorder before rollout to release the camera and serial devices.

```bash
python -B -m smolvla_scratch.train_infer --rollout \
  --checkpoint outputs/pick_place_run1/policy_best.ckpt \
  --task "Pick up an orange marker and put it in the cup" \
  --episode_len 300 \
  --num_rollouts 1 \
  --device auto
```

Use the task instruction corresponding to the learned behavior. Multi-task checkpoints require an explicit task choice when no single shared instruction is saved.

`--episode_len` is a maximum number of environment steps, not a time limit in seconds. The runner uses the checkpoint's saved `n_action_steps` and Euler-step configuration. It executes a queue of predicted actions synchronously, waits for inference when another chunk is needed, and does not guarantee a sustained 30 Hz loop. Guided RTC checkpoints are rejected by this runner.

Use `--env_kwargs path/to/env.json` for a JSON object overriding environment settings such as `port`, `robot_id`, `reset_pose`, `max_relative_target`, or `dt`. The default factory is `low_control.low_level_control:make_real_env`; camera names must match the checkpoint.

The CLI writes `policy_best.rollout.json` beside the checkpoint. It records step counts and returns when rewards exist. The supplied environment has no task-success detector, so `return` may be `null`. Video capture, trial outcomes, task-success labels, and completion times need to be recorded separately. Repeated rollout commands use the same summary filename; preserve each trial's summary before another command overwrites it.

## 10. Troubleshooting

| Symptom | Action |
| --- | --- |
| Module import fails | Activate the intended environment, install requirements, and run from the repository root |
| CUDA is unavailable | Check the installed PyTorch build and GPU environment, or choose a supported device |
| `Need at least two episodes` | Supply at least two valid episode directories |
| Missing task | Add the task to episode metadata, or override only if all episodes share one task |
| Missing `joint_names` | Restore the recording schema; use legacy ordering only for verified legacy ACT data |
| Video decode/frame-count error | Check each camera video has the same number of frames as NPZ timesteps |
| Output directory must be empty/new | Choose a new output directory, or resume the existing run's `policy_last.ckpt` |
| Out of memory | Start a new run with a smaller batch size |
| Worker process fails | Retry with `--num_workers 0` |
| Dataset fingerprint mismatch | Restore the original bytes, or use fine-tuning for a changed dataset |
| Model/tokenizer download fails | Check network access and the Hugging Face cache |
| Dashboard `cv2.imshow` fails | Use GUI-enabled OpenCV and a desktop recording environment |
| Validation loss rises while train loss falls | Inspect the unsmoothed validation curve; evaluate the best and final checkpoints separately |
| Robot success rate is absent from JSON | Score trials separately; the environment does not infer task success |

For published results, preserve the training configuration, history, checkpoint identity, and evaluation JSON alongside the figures. Report the episode split, seed, budget, action horizon, metric units, and rollout denominators so results remain interpretable.
