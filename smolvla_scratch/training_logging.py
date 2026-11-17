"""Live training curves, with checkpoint-aligned TensorBoard resume."""
from pathlib import Path


def require_tensorboard():
    """Check the training dependency without requiring it for eval/rollout."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is required for training. Install it with: "
            "python -m pip install 'tensorboard>=2.20,<3'"
        ) from exc
    return SummaryWriter


class TensorBoardLogger:
    def __init__(self, ckpt_dir, *, history=(), global_step=0):
        SummaryWriter = require_tensorboard()

        self.log_dir = Path(ckpt_dir) / "tensorboard"
        has_events = any(self.log_dir.glob("events.out.tfevents.*"))
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir), flush_secs=10,
            # A partially completed epoch is replayed after resume. Hide its
            # old events, retaining points belonging to completed updates.
            purge_step=global_step + 1 if history else None,
        )
        self.writer.add_custom_scalars({
            "Loss": {"Train vs validation": [
                "Multiline", ["loss/train_epoch", "loss/val_epoch"],
            ]},
        })
        # A copied checkpoint or an older run may have no event files. Its
        # authoritative epoch history is enough to recover the epoch curves.
        if not has_events:
            best = float("inf")
            for row in history:
                best = min(best, row["val_loss"])
                self.log_epoch(row, best)
        self.writer.flush()

    def log_step(self, loss, lr, grad_norm, global_step):
        self.writer.add_scalar("loss/train_step", loss, global_step)
        # LR used by THIS optimizer update, before scheduler.step().
        self.writer.add_scalar("learning_rate/step", lr, global_step)
        self.writer.add_scalar("grad_norm/before_clip", grad_norm, global_step)

    def log_epoch(self, row, best_val_loss):
        step = row["global_step"]
        self.writer.add_scalar("loss/train_epoch", row["train_loss"], step)
        self.writer.add_scalar("loss/val_epoch", row["val_loss"], step)
        self.writer.add_scalar("loss/best_val", best_val_loss, step)
        self.writer.add_scalar("learning_rate/epoch", row["lr"], step)
        self.writer.add_scalar("progress/epoch", row["epoch"] + 1, step)
        self.writer.flush()

    def close(self):
        self.writer.close()
