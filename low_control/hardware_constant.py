# Discover each arm with lerobot-find-port; use the same IDs for calibration.
LEADER_PORT = "/dev/tty.usbmodem5B8E1128291"
FOLLOWER_PORT = "/dev/tty.usbmodem5B8E1131141"
LEADER_ID = "so101_leader"
FOLLOWER_ID = "so101_follower"

ROBOT_STATE = 6

CAMERA_FPS = 30
CAMERA_CONFIGS = {
    "front": dict(index_or_path=1, width=640, height=480, fps=CAMERA_FPS),
    "wrist": dict(index_or_path=0, width=640, height=480, fps=CAMERA_FPS),
}


def make_camera_configs(camera_names):
    """Build LeRobot configs only for rollout; importing constants needs no hardware packages."""
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    return {name: OpenCVCameraConfig(**CAMERA_CONFIGS[name]) for name in camera_names}
CONTROL_HZ = 30.0
CONTROL_DT = 1.0 / CONTROL_HZ

# Joint order shared by the recorder, dataset loader and robot environment.
MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
JOINT_KEYS = [f"{name}.pos" for name in MOTOR_NAMES]

# Reset pose during inference
RESET_POSE = [-4.571428571428571, -104.26373626373626, 95.78021978021978, -99.47252747252747, -0.8351648351648352, 0.6293706293706294]
