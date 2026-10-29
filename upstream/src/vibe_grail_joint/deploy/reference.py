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


def interpolate(poses, locations):
    locations = np.asarray(locations)
    if locations.min() < 0 or locations.max() > len(poses) - 1:
        raise RuntimeError("Planner future horizon exhausted")
    indices = np.arange(len(poses))
    out = np.stack([np.interp(locations, indices, poses[:, j]) for j in range(36)], axis=-1)
    out[:, 3:7] = Slerp(indices, Rotation.from_quat(poses[:, [4, 5, 6, 3]]))(locations).as_quat()[:, [3, 0, 1, 2]]
    return out.astype(np.float32)


class OnlineReference:
    def __init__(self, planner, contract, seed=42, session_override=None):
        self.session = session_override or session(planner, "cpu")
        seed_pose = np.r_[0., 0., contract["root_height"], 1., 0., 0., 0., contract["default_joint_pos"]]
        self.poses = np.repeat(seed_pose[None], 256, axis=0).astype(np.float32)
        self.start = 0
        self.end = 256
        self.updated_at = None
        self.seed = seed
        self.lock = threading.Lock()
        self.seconds = 0.

    def replan(self, elapsed, speed):
        if not np.isfinite(speed) or not 0 <= speed <= .7:
            raise ValueError("Unsupported forward speed")
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
        inputs = {
            "context_mujoco_qpos": context[None],
            "target_vel": np.array([speed if speed > 0 else -1.], np.float32),
            "mode": np.array([2 if speed > 0 else 0], np.int64),
            "movement_direction": np.array([[1., 0., 0.]], np.float32),
            "facing_direction": np.array([[1., 0., 0.]], np.float32),
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
        for i, weight in enumerate(np.linspace(0, 1, n)):
            old_q, new_q = previous[i, [4, 5, 6, 3]], pred[i, [4, 5, 6, 3]].copy()
            pred[i] = previous[i] * (1-weight) + pred[i] * weight
            pred[i, 3:7] = Slerp([0, 1], Rotation.from_quat([old_q, new_q]))([weight]).as_quat()[0, [3, 0, 1, 2]]
        # Keep enough past for centered derivatives. Absolute ticks avoid clock resets.
        new_start = max(start, tick - 4)
        prefix = poses[new_start-start:local]
        combined = np.concatenate([prefix, pred])
        with self.lock:
            self.poses = combined
            self.start = new_start
            self.end = new_start + len(combined)
            self.updated_at = elapsed

    def sample(self, elapsed):
        with self.lock:
            poses, start, updated = self.poses.copy(), self.start, self.updated_at
        if updated is None or elapsed - updated > .45:
            age = None if updated is None else elapsed - updated
            raise RuntimeError(f"Planner is stale or not initialized: age_s={age}, last_inference_s={self.seconds:.4f}")
        at = elapsed * 50 - start
        positions = at + np.arange(10) * 5
        qpos = interpolate(poses, positions)
        qvel = np.gradient(poses[:, 7:], .02, axis=0)
        joint_vel = np.stack([np.interp(positions, np.arange(len(poses)), qvel[:, j]) for j in range(29)], axis=-1)
        lin = np.gradient(poses[:, :3], .02, axis=0)
        ang = angular_velocity(poses[:, 3:7], .02)
        twist = [np.array([np.interp(at, np.arange(len(poses)), data[:, j]) for j in range(3)]) for data in (lin, ang)]
        return Reference(qpos[:, 7:], joint_vel.astype(np.float32), qpos[:, 3:7], *twist)


class PlannerWorker:
    def __init__(self, planner, contract):
        self.reference = OnlineReference(planner, contract)
        self.origin = time.monotonic()
        self.speed = 0.
        self.error = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="sonic-reference", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                begin = time.monotonic()
                self.reference.replan(begin - self.origin, self.speed)
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
