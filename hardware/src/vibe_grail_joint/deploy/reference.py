"""Continuous SONIC reference, including standing/start/stop, without a dataset.

Planner runs at 5 Hz, tracker at 50 Hz. A rolling future window, never a padded
walking clip, is sampled by the tracker. CPU planning may run in a worker.
"""
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from ..kinematics import angular_velocity, resample_qpos
from .observations import Reference, checked
from .runtime import session


def linear_rows(values, locations):
    """Vectorized np.interp on a uniform grid, preserving float64 arithmetic."""
    locations = np.asarray(locations, dtype=np.float64)
    if (not np.isfinite(locations).all() or locations.min() < 0
            or locations.max() > len(values) - 1):
        raise RuntimeError("Planner future horizon exhausted")
    lower = np.floor(locations).astype(np.intp)
    upper = np.minimum(lower + 1, len(values) - 1)
    weight = (locations - lower)[..., None]
    left = values[lower].astype(np.float64)
    right = values[upper].astype(np.float64)
    return left + (right - left) * weight


def interpolate(poses, locations):
    locations = np.asarray(locations)
    out = linear_rows(poses, locations)
    indices = np.arange(len(poses))
    out[:, 3:7] = Slerp(indices, Rotation.from_quat(poses[:, [4, 5, 6, 3]]))(locations).as_quat()[:, [3, 0, 1, 2]]
    return out.astype(np.float32)


class ReferenceWindow:
    """Immutable snapshot: compute derivatives and quaternion setup once per plan."""
    def __init__(self, poses):
        self.poses = np.array(poses, dtype=np.float32, copy=True)
        self.joint_vel = np.gradient(self.poses[:, 7:], .02, axis=0)
        self.lin_vel = np.gradient(self.poses[:, :3], .02, axis=0)
        self.ang_vel = angular_velocity(self.poses[:, 3:7], .02)
        self.slerp = Slerp(np.arange(len(self.poses)), Rotation.from_quat(self.poses[:, [4, 5, 6, 3]]))
        for array in (self.poses, self.joint_vel, self.lin_vel, self.ang_vel):
            array.flags.writeable = False

    def sample(self, at):
        positions = at + np.arange(10) * 5
        qpos = linear_rows(self.poses, positions)
        qpos[:, 3:7] = self.slerp(positions).as_quat()[:, [3, 0, 1, 2]]
        qpos = qpos.astype(np.float32)
        joint_vel = linear_rows(self.joint_vel, positions).astype(np.float32)
        return Reference(qpos[:, 7:], joint_vel, qpos[:, 3:7],
                         linear_rows(self.lin_vel, at), linear_rows(self.ang_vel, at))


def blend_prefix(previous, prediction):
    """Batch the same shortest-arc quaternion interpolation as pairwise Slerp."""
    n = len(previous)
    weight = np.linspace(0, 1, n)[:, None]
    old = Rotation.from_quat(previous[:, [4, 5, 6, 3]])
    new = Rotation.from_quat(prediction[:n, [4, 5, 6, 3]])
    delta = (old.inv() * new).as_rotvec()
    quat = (old * Rotation.from_rotvec(delta * weight)).as_quat()[:, [3, 0, 1, 2]]
    prediction[:n] = previous * (1-weight) + prediction[:n] * weight
    prediction[:n, 3:7] = quat


class OnlineReference:
    def __init__(self, planner, contract, seed=42, session_override=None, *, threads=2):
        options = {"threads":threads} if threads != 2 else {}
        self.session = session_override or session(planner, "cpu", **options)
        seed_pose = np.r_[0., 0., contract["root_height"], 1., 0., 0., 0., contract["default_joint_pos"]]
        self.poses = np.repeat(seed_pose[None], 256, axis=0).astype(np.float32)
        self.start = 0
        self.end = 256
        self.updated_at = None
        self.seed = seed
        self.lock = threading.Lock()
        self.seconds = 0.
        self.window = None

    def replan(self, elapsed, speed, heading=0.):
        if not np.isfinite([speed,heading]).all() or not -.3 <= speed <= .7:
            raise ValueError("Unsupported movement command")
        tick = int(round(elapsed * 50))
        with self.lock:
            poses, start, end = self.poses.copy(), self.start, self.end
        cut = tick + 2
        local = cut - start
        if not 0 <= local < end - start - 6:
            raise RuntimeError("Planner missed its rolling-buffer deadline")
        context = interpolate(poses[:end-start], local + np.arange(4) * 50 / 30)
        mask = np.zeros((1, 11), np.int64)
        mask[0, 5] = 1
        facing = np.array([np.cos(heading),np.sin(heading),0.],np.float32)
        movement = facing*(-1 if speed<0 else 1)
        inputs = {
            "context_mujoco_qpos": context[None],
            "target_vel": np.array([abs(speed) if speed != 0 else -1.], np.float32),
            "mode": np.array([2 if speed != 0 else 0], np.int64),
            "movement_direction": movement[None],
            "facing_direction": facing[None],
            "random_seed": np.array([self.seed], np.int64),
            "height": np.array([-1.], np.float32),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": mask,
        }
        before = time.monotonic()
        outputs = dict(zip([o.name for o in self.session.get_outputs()], self.session.run(None, inputs), strict=True))
        self.seconds = time.monotonic() - before
        count = int(outputs["num_pred_frames"].reshape(-1)[0])
        if count != 44:
            raise RuntimeError(f"Unexpected planner prediction length {count}")
        pred = resample_qpos(outputs["mujoco_qpos"][0, :count], 1/30, .02)
        # Initial state is stationary; do not warm-start by jumping 0.8 s into walking.
        n = min(8, end - cut, len(pred))
        previous = poses[local:local+n].copy()
        blend_prefix(previous, pred)
        # Keep enough past for centered derivatives. Absolute ticks avoid clock resets.
        new_start = max(start, tick - 4)
        prefix = poses[new_start-start:local]
        combined = np.concatenate([prefix, pred])
        window = ReferenceWindow(combined)
        with self.lock:
            self.poses = combined
            self.start = new_start
            self.end = new_start + len(combined)
            self.updated_at = elapsed
            self.window = window

    def sample(self, elapsed):
        with self.lock:
            window, start, updated = self.window, self.start, self.updated_at
        if updated is None or elapsed - updated > .45:
            age = None if updated is None else elapsed - updated
            raise RuntimeError(f"Planner is stale or not initialized: age_s={age}, last_inference_s={self.seconds:.4f}")
        at = elapsed * 50 - start
        return window.sample(at)


class PlannerWorker:
    def __init__(self, planner, contract, *, threads=2):
        self.reference = OnlineReference(planner, contract, threads=threads)
        self.origin = time.monotonic()
        self.speed = 0.
        self.command = None
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="sonic-reference", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                begin = time.monotonic()
                speed,heading = self.command if self.command is not None else (self.speed,0.)
                self.reference.replan(begin - self.origin, speed, heading)
                self.stop.wait(max(0., .2 - (time.monotonic() - begin)))
        except Exception as exc:
            self.error = exc

    def sample(self, now):
        if self.error:
            raise RuntimeError(f"Planner worker failed: {self.error}") from self.error
        return self.reference.sample(now - self.origin)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.)


class ClipReference:
    """Prepared, named MJ-order reference; never loops a stair clip at its end."""
    def __init__(self, path, contract):
        with np.load(path, allow_pickle=False) as data:
            if list(data["joint_names"]) != contract["joint_names"] or float(data["fps"]) != 50:
                raise ValueError("Reference must use named MJ-order joints at 50 Hz")
            self.qpos = data["qpos"].copy()
        if self.qpos.ndim != 2 or self.qpos.shape[1] != 36 or len(self.qpos) < 47:
            raise ValueError("Reference must contain at least 47 frames of qpos[36]")
        if not np.isfinite(self.qpos).all():
            raise ValueError("Non-finite reference")
        # Canonical heading, matching the initial heading alignment in the runtime.
        first = Rotation.from_quat(self.qpos[0, [4, 5, 6, 3]])
        yaw = first.as_euler("xyz")[2]
        heading = Rotation.from_euler("z", yaw)
        self.qpos[:, :3] = heading.inv().apply(self.qpos[:, :3] - self.qpos[0, :3])
        self.qpos[:, 3:7] = (heading.inv() * Rotation.from_quat(self.qpos[:, [4, 5, 6, 3]])).as_quat()[:, [3, 0, 1, 2]]
        self.dq = np.gradient(self.qpos[:, 7:], .02, axis=0)
        self.lin = np.gradient(self.qpos[:, :3], .02, axis=0)
        self.ang = angular_velocity(self.qpos[:, 3:7], .02)

    def sample(self, elapsed):
        at = elapsed * 50
        idx = at + np.arange(10) * 5
        if idx[-1] >= len(self.qpos):
            raise RuntimeError("Reference ended; no automatic restart")
        pose = interpolate(self.qpos, idx)
        t = np.arange(len(self.qpos))
        dq = np.stack([np.interp(idx, t, self.dq[:, j]) for j in range(29)], -1)
        return Reference(pose[:, 7:], dq, pose[:, 3:7],
                         np.array([np.interp(at, t, self.lin[:, j]) for j in range(3)]),
                         np.array([np.interp(at, t, self.ang[:, j]) for j in range(3)]))
