from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
import numpy as np
import time
import dm_env
import math
from .hardware_constant import (
    FOLLOWER_PORT, FOLLOWER_ID, CONTROL_DT, CAMERA_CONFIGS, MOTOR_NAMES,
    JOINT_KEYS, RESET_POSE, make_camera_configs,
)

ROBOT_PORT = FOLLOWER_PORT
ROBOT_ID = FOLLOWER_ID
RESET_POSE = RESET_POSE
DT = CONTROL_DT
MAX_RELATIVE_TARGET = 5.0  # Per-command bound in LeRobot joint units (gripper uses its own scale).
CAMERAS = CAMERA_CONFIGS

def disconnect_arm(arm):
    # SOFollower.is_connected is False if even one camera failed to connect.
    # Disconnect bus and cameras independently so one failure cannot skip others.
    bus = getattr(arm, "bus", None)
    if bus is not None and bus.is_connected:
        try:
            if hasattr(arm.config, "disable_torque_on_disconnect"):
                bus.disconnect(arm.config.disable_torque_on_disconnect)
            else:
                bus.disconnect()
        except Exception as exc:
            print(f"[WARN] Motor bus disconnect failed: {exc}")
    for name, camera in getattr(arm, "cameras", {}).items():
        try:
            if camera.is_connected:
                camera.disconnect()
        except Exception as exc:
            print(f"[WARN] Camera {name} disconnect failed: {exc}")


class RealEnv:
    # Create the real robot
    def __init__(self, port, robot_id, cameras=None, dt=CONTROL_DT, reset_pose=None, max_relative_target=None):
        # Validate before constructing/connecting the robot.
        if not port or not robot_id:
            raise ValueError("Robot port and ID must be nonempty")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        if max_relative_target is None or not math.isfinite(max_relative_target) or max_relative_target <= 0:
            raise ValueError("max_relative_target must be finite and positive")
        pose = np.asarray(reset_pose, dtype=np.float32)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("reset_pose must contain six finite canonical joint values")
        self.dt = dt
        self.reset_pose = pose

        # SO-ARM configuration
        config = SO101FollowerConfig(
            port = port,
            id = robot_id,
            cameras = cameras or {},
            max_relative_target = max_relative_target,
            use_degrees = True,
        )
        
        self.robot = SO101Follower(config)
        try:
            self.robot.connect()
        except BaseException:
            disconnect_arm(self.robot)
            raise

    
    def get_qpos(self, raw_obs):
        qpos = np.array(
            [
                raw_obs[f"{name}.pos"]
                for name in MOTOR_NAMES
            ],
            dtype=np.float32
        )

        if not np.isfinite(qpos).all():
            raise ValueError("Robot state contains NaN or infinity")
        return qpos

    def get_images(self, raw_obs):

        images = {}

        for camera_name in self.robot.cameras:
            images[camera_name] = raw_obs[camera_name]

        return images
    
    def get_observation(self):
        """
        Read SO-ARM observation once; return raw qpos and RGB images for SmolVLA.
        """

        raw_obs = self.robot.get_observation()

        obs = {
            "qpos": self.get_qpos(raw_obs),
            "images": self.get_images(raw_obs),
        }

        return obs

    # Action
    def _vector_to_action_dict(self, action):
        """
        Convert np.ndarray(6,) into 
        { 
            "shoulder_pan.pos": ...,
            ...
            "gripper.pos":....
        }
        """
        action = np.asarray(
            action,
            dtype = np.float32
        )

        if action.shape != (6,):
            raise ValueError(
                "SO-ARM101 action must have "
                f"shape (6,), got {action.shape}"
            )

        if not np.all(np.isfinite(action)):
            raise ValueError("SO-ARM101 action contains NaN or infinity")
        
        robot_action = {
            f"{name}.pos": float(action[i])
            for i, name in enumerate(MOTOR_NAMES)
        }

        return robot_action

    # Reset
    def _move_to_pose(self, target_pose, move_time=1.5):
        "Smoothly interpolate from current position to target position."

        target_pose = np.asarray(target_pose,dtype=np.float32)
        if target_pose.shape != (6,) or not np.all(np.isfinite(target_pose)):
            raise ValueError(
                "reset pose must contain six finite values"
            )
        
        raw_obs = self.robot.get_observation()

        start_pos = self.get_qpos(raw_obs)

        num_steps = max(1, int(move_time/self.dt))

        for i in range(1, num_steps + 1):
            alpha = i / num_steps

            pose = (1.0 - alpha) * start_pos + alpha * target_pose

            command = self._vector_to_action_dict(pose)

            self.robot.send_action(command)

            time.sleep(self.dt)
        
    def _reset_joints(self):
        # Move robot to configured reset pose
        if self.reset_pose is None:
            raise RuntimeError(
                "A six-joint reset_pose is required before a real-robot rollout. "
                "Refusing to start from an unknown pose."
            )
        
        self._move_to_pose(self.reset_pose)

    # Reward to evaluate
    def get_reward(self):
        # No automatic real-world task reward yet.
        # None explicitly means "not measured"; returning 0 would make every
        # rollout look successful when env_max_reward is also zero.
        return None
    
    def reset(self, fake=False):
        """
        The rollout calls this at the beginning
        of an episode.

        fake=True:
            do not physically move robot.

        fake=False:
            move robot to reset_pose if provided.
        """

        if not fake:
            self._reset_joints()

        # container TimeStep
        return dm_env.TimeStep(
            step_type=dm_env.StepType.FIRST,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )
    
    def step(self, action):
        # Receive an already denormalized six-joint target from SmolVLA.
        robot_action = self._vector_to_action_dict(action)
        self.robot.send_action(robot_action)
        time.sleep(self.dt)

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=self.get_observation(),
        )

    def close(self):
        disconnect_arm(self.robot)


def make_real_env(port=ROBOT_PORT, robot_id=ROBOT_ID, cameras=None, dt=CONTROL_DT,
                  reset_pose=RESET_POSE, max_relative_target=MAX_RELATIVE_TARGET,
                  camera_names=None):
    available = CAMERA_CONFIGS if cameras is None else cameras
    names = list(available) if camera_names is None else list(camera_names)
    if not names or len(names) != len(set(names)) or set(names) - set(available):
        raise ValueError(f"Requested cameras are missing or duplicated: {names}")
    selected = (make_camera_configs(names) if cameras is None
                else {name: cameras[name] for name in names})
    env = RealEnv(
        port=port,
        robot_id=robot_id,
        cameras=selected,
        dt=dt,
        reset_pose=reset_pose,
        max_relative_target=max_relative_target,
    )

    return env

