"""NumPy deployment contract, independent of torch, mjlab and training data.

All quaternions are wxyz; joints are named MuJoCo/Unitree PR order. Histories
are term-major, oldest first. The tokenizer's unusual chop is intentional.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


JOINT_NAMES = [
    *[f"{side}_{part}_joint" for side in ("left", "right")
      for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")],
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    *[f"{side}_{part}_joint" for side in ("left", "right")
      for part in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                   "wrist_roll", "wrist_pitch", "wrist_yaw")],
]


def checked(value, shape, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"Invalid {name}: shape={value.shape}, expected={shape}")
    return value


def rotation(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1)
    if quat.shape[-1] != 4 or not np.isfinite(quat).all() or np.any(abs(norm - 1) > .02):
        raise ValueError("Expected normalized wxyz quaternion")
    return Rotation.from_quat(quat[..., [1, 2, 3, 0]])


@dataclass
class RobotState:
    q: np.ndarray
    dq: np.ndarray
    quat: np.ndarray
    gyro: np.ndarray
    received_at: float
    tick: int = 0
    mode_machine: int = 0
    motor_errors: tuple = ()
    temperature: float = 0.
    remote_stop: bool = False


@dataclass
class Reference:
    joint_pos: np.ndarray           # ten samples at 0, .1, ..., .9 seconds
    joint_vel: np.ndarray
    root_quat: np.ndarray
    root_lin_vel: np.ndarray        # current WORLD-frame reference twist
    root_ang_vel: np.ndarray


class ObservationBuilder:
    def __init__(self, contract, patch_shape=(21, 192)):
        if contract["joint_names"] != JOINT_NAMES:
            raise ValueError("Joint order is not the supported G1 29 DOF PR order")
        self.default = checked(contract["default_joint_pos"], (29,), "default pose")
        self.patch_shape = tuple(patch_shape)
        self.reset()

    def reset(self):
        self.history = None
        self.previous_action = np.zeros(29, np.float32)
        self.heading = None

    def align_heading(self, quat):
        # Establish +X once on engage. Do NOT re-zero heading every frame.
        r = rotation(quat).as_matrix()
        self.heading = Rotation.from_euler("z", np.arctan2(r[1, 0], r[0, 0]))

    def accept_action(self, action):
        self.previous_action = checked(action, (29,), "applied action").copy()

    def build(self, state, reference, patches, cls, speed, *, velocity=True, group="online",
              heading=0., yaw_rate=0.):
        q = checked(state.q, (29,), "joint positions")
        dq = checked(state.dq, (29,), "joint velocities")
        gyro = checked(state.gyro, (3,), "pelvis gyro")
        live = rotation(state.quat)
        if self.heading is None:
            self.align_heading(state.quat)
        relative = self.heading.inv() * live
        gravity = live.inv().apply([0., 0., -1.]).astype(np.float32)
        terms = [gyro, q - self.default, dq, self.previous_action, gravity]
        if self.history is None:
            self.history = [np.repeat(x[None], 10, axis=0) for x in terms]
        else:
            for history, value in zip(self.history, terms, strict=True):
                history[:-1] = history[1:]
                history[-1] = value
        jp = checked(reference.joint_pos, (10, 29), "reference joint positions")
        jv = checked(reference.joint_vel, (10, 29), "reference joint velocities")
        rq = rotation(checked(reference.root_quat, (10, 4), "reference quaternions"))
        # Ref is in a canonical world whose +X is the heading at engage.
        orientation = (relative.inv() * rq).as_matrix()[..., :2].reshape(10, 6)
        chop = np.concatenate([jp.ravel(), jv.ravel()]).reshape(10, 58)
        twist = np.concatenate([
            rq[0].inv().apply(checked(reference.root_lin_vel, (3,), "reference linear velocity")),
            rq[0].inv().apply(checked(reference.root_ang_vel, (3,), "reference angular velocity")),
        ])
        out = {
            "policy": np.concatenate([h.ravel() for h in self.history]),
            "tokenizer": np.concatenate([chop, orientation], axis=1).ravel(),
            "augmentation": twist,
            "q_task_cmd": twist.copy(),
            "q_proprio": np.concatenate([gravity, gyro, q - self.default, dq]),
            "kv_tokens__img_tokens": checked(patches, self.patch_shape, "image patches"),
            "q_cls": checked(cls, (192,), "image CLS"),
        }
        if velocity:
            if group not in ("online", "paired", "flat"):
                raise ValueError("Unknown task group")
            if not np.isfinite([speed, heading, yaw_rate]).all() or not -.3 <= speed <= .7:
                raise ValueError("Invalid directional command")
            r = relative.as_matrix()
            yaw = np.arctan2(r[1, 0], r[0, 0]) - heading
            online = group == "online"
            out["velocity_command"] = np.array([
                speed * np.cos(yaw) if online else twist[0],
                -speed * np.sin(yaw) if online else twist[1],
                yaw_rate if online else twist[5],
                1 - np.cos(yaw) if online else 0., -np.sin(yaw) if online else 0.,
                float(online), float(group == "paired"), float(group == "flat"),
            ])
        return {name: np.ascontiguousarray(value[None], dtype=np.float32) for name, value in out.items()}
