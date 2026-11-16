"""Checkpoint cục bộ: model để dùng lại, training để tiếp tục đúng một run.

Không dùng pickle tùy ý: payload chỉ có tensor và kiểu Python cơ bản để có
thể đọc bằng ``weights_only=True``. Format v1 cũ vẫn đọc được nhưng thiếu
optimizer/scheduler/RNG, vì vậy chỉ dùng cho eval, rollout và fine-tuning.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from .models.rtc_processor import RTCAttentionSchedule
from .utils.vla_utils import JOINT_NAMES


def capture_rng_state():
    """Lưu RNG sau validation, trước khi bắt đầu epoch kế tiếp.

    Shuffle DataLoader, flow noise/time và dropout dùng RNG của PyTorch.
    NumPy chứa ndarray trong get_state(); đổi sang list để safe-load được.
    """
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state):
    """Gọi SAU khi dựng model/optimizer: khởi tạo model cũng tiêu thụ RNG."""
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32),
                         position, has_gauss, cached_gaussian))
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same CUDA device count for RNG restoration")
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"])


def dataset_manifest(root, metadata):
    """Băm dữ liệu một lần khi mở run; resume từ chối dữ liệu bị sửa.

    Tên episode giống nhau chưa đủ: nội dung state/action hoặc video có thể
    đã bị ghi đè. Đường dẫn tuyệt đối không nằm trong manifest nên có thể
    chuyển nguyên dataset sang máy/thư mục khác và vẫn resume được.
    """
    manifest = {}
    files = ["data.npz", *[f"{camera}.mp4" for camera in metadata["camera_names"]]]
    for episode in metadata["train_episodes"] + metadata["val_episodes"]:
        for filename in files:
            relative = f"{episode}/{filename}"
            digest = hashlib.sha256()
            with (Path(root) / relative).open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            manifest[relative] = digest.hexdigest()
    return manifest


def save_checkpoint(path, policy, stats, metadata, epoch, best_loss, *, training_state=None):
    """Ghi atomically: file đích chỉ thay thế sau khi torch.save hoàn tất.

    Model checkpoint chứa đủ config, weights, stats và split để eval hoặc
    fine-tuning. Training checkpoint có thêm trạng thái toàn bộ run.
    Cả hai đều chứa backbone đã nạp, không phải chỉ riêng action expert.
    """
    path = Path(path)
    payload = {
        "format_version": 2,
        "checkpoint_kind": "training" if training_state is not None else "model",
        # Enum/dataclass lồng nhau được chuyển thành JSON-compatible values.
        "policy_config": json.loads(json.dumps(asdict(policy.config))),
        "model": policy.state_dict(), "dataset_stats": stats,
        "data": metadata, "epoch": epoch, "best_val_loss": best_loss,
    }
    if training_state is not None:
        payload["training_state"] = training_state
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def read_checkpoint(path, *, for_resume=False):
    """Đọc ở CPU trước để tránh đưa cả optimizer/best snapshot lên GPU."""
    # Một số checkpoint v1 lưu Enum RTC trực tiếp; chỉ allowlist đúng lớp đó.
    with torch.serialization.safe_globals([RTCAttentionSchedule]):
        payload = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
    required = {"model", "policy_config", "dataset_stats", "data", "epoch", "best_val_loss"}
    if not isinstance(payload, dict) or payload.get("format_version") not in (1, 2) or not required <= payload.keys():
        raise ValueError("Checkpoint needs model/config/stats/data; a weights-only file is insufficient")
    if payload["data"]["joint_names"] != JOINT_NAMES:
        raise ValueError("Unsupported checkpoint joint schema")
    if for_resume:
        training = payload.get("training_state", {})
        keys = {"optimizer", "scheduler", "rng", "history", "best_epoch", "best_model",
                "global_step", "scheduler_total_steps", "steps_per_epoch", "arguments", "dataset_manifest"}
        if payload.get("checkpoint_kind") != "training" or not keys <= training.keys():
            raise ValueError("Resume needs a full training checkpoint (policy_last.ckpt or policy_epoch_*.ckpt); "
                             "use --finetune for v1/model checkpoints")
        if len(training["history"]) != payload["epoch"] + 1:
            raise ValueError("Checkpoint epoch/history mismatch")
    return payload
