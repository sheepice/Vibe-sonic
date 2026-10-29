"""CPU conversion into the frozen SONIC model's named MuJoCo layout."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

CHANNELS = {
    "joint_pos": (29,),
    "joint_vel": (29,),
    "body_pos_w": (14, 3),
    "body_quat_w": (14, 4),
    "body_lin_vel_w": (14, 3),
    "body_ang_vel_w": (14, 3),
}


def resample_qpos(qpos: np.ndarray, source_dt: float, target_dt: float) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2 or not np.isfinite(qpos).all():
        raise ValueError("Invalid source qpos")
    if source_dt <= 0 or target_dt <= 0:
        raise ValueError("Timesteps must be positive")
    norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    if np.any(norms < 1e-8) or np.max(np.abs(norms - 1)) > 0.02:
        raise ValueError("Non-unit root quaternion")
    times = np.arange(len(qpos)) * source_dt
    samples = np.arange(int(np.floor(times[-1] / target_dt + 1e-7)) + 1) * target_dt
    samples = np.minimum(samples, times[-1])
    if len(samples) < 2:
        raise ValueError("Motion is shorter than one control interval")
    upper = np.clip(np.searchsorted(times, samples, side="right"), 1, len(times)-1)
    lower = upper-1
    weight = ((samples-times[lower]) / (times[upper]-times[lower]))[:, None]
    result = qpos[lower] + (qpos[upper]-qpos[lower]) * weight
    quat = qpos[:, [4, 5, 6, 3]] / norms[:, None]
    result[:, 3:7] = Slerp(times, Rotation.from_quat(quat))(samples).as_quat()[:, [3, 0, 1, 2]]
    return result.astype(np.float32)


def angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity: q(t+1) * inverse(q(t)), not Euler differences."""
    shape = quat_wxyz.shape
    q = quat_wxyz[..., [1, 2, 3, 0]]
    current = Rotation.from_quat(q[1:].reshape(-1, 4))
    previous = Rotation.from_quat(q[:-1].reshape(-1, 4))
    delta = (current * previous.inv()).as_rotvec().reshape(shape[0] - 1, *shape[1:-1], 3) / dt
    output = np.empty((*shape[:-1], 3), dtype=np.float32)
    output[0], output[-1] = delta[0], delta[-1]
    if len(output) > 2:
        output[1:-1] = 0.5 * (delta[:-1] + delta[1:])
    return output


def convert_motion(row: dict, source: dict, model, body_names: tuple[str, ...], hz: int) -> dict:
    import mujoco

    with np.load(row["path"], allow_pickle=False) as data:
        qpos = resample_qpos(data["qpos"], row["dt"], 1.0 / hz)
    names = source["qpos_names"]
    joint_names = [
        model.joint(i).name for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
    ]
    if len(joint_names) != 29 or model.nq != 36:
        raise ValueError("The SONIC base requires a floating G1 with 29 joints")
    columns = [names.index(name) for name in joint_names]
    joints = qpos[:, columns]
    body_ids = np.array([model.body(name).id for name in body_names])
    body_pos = np.empty((len(qpos), len(body_ids), 3), np.float32)
    body_quat = np.empty((len(qpos), len(body_ids), 4), np.float32)
    state = mujoco.MjData(model)
    addresses = np.array([model.joint(name).qposadr[0] for name in joint_names])
    for index, pose in enumerate(qpos):
        state.qpos[:7] = pose[:7]
        state.qpos[addresses] = joints[index]
        mujoco.mj_kinematics(model, state)
        body_pos[index] = state.xpos[body_ids]
        body_quat[index] = state.xquat[body_ids]
    dt = 1.0 / hz
    return {
        "joint_pos": joints.astype(np.float32),
        "joint_vel": np.gradient(joints, dt, axis=0).astype(np.float32),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_lin_vel_w": np.gradient(body_pos, dt, axis=0).astype(np.float32),
        "body_ang_vel_w": angular_velocity(body_quat, dt),
    }


def windows(length: int, maximum: int):
    """Cover long flat motions with bounded episodes; every source frame remains eligible."""
    if maximum < 2 or length < 2:
        raise ValueError("Need at least two frames")
    start = 0
    while start < length - 1:
        end = min(start + maximum, length)
        yield start, end
        if end == length:
            break
        start = end - 1
