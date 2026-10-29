"""Isolated W-command probe: online SONIC references, unchanged ViBe actor.

World 0 is the recorded-reference control. World 1 only receives a constant
walking command and online predictions. The stair recording seeds its initial
state, but no future stair poses or terrain samples enter the planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import time

import mjlab  # noqa: F401
import mujoco
import numpy as np
import onnxruntime as ort
import torch
from scipy.spatial.transform import Rotation, Slerp
from orcs.core.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg

from .kinematics import CHANNELS, angular_velocity, resample_qpos
from .prepare import build_model, validate_bank
from .terrain import atlas_layout


def interpolate_poses(poses, times):
    grid = np.arange(len(poses), dtype=float)
    out = np.stack([np.interp(times, grid, poses[:, i]) for i in range(36)], axis=1)
    out[:, 3:7] = Slerp(grid, Rotation.from_quat(poses[:, [4, 5, 6, 3]]))(times).as_quat()[:, [3, 0, 1, 2]]
    return out.astype(np.float32)


class OnlinePlanner:
    def __init__(self, path, seed_pose, capacity, *, speed, heading_deg, seed):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])
        self.poses = np.repeat(seed_pose[None], capacity, axis=0).astype(np.float32)
        self.seed_pose = seed_pose.copy()
        angle = np.deg2rad(heading_deg)
        self.direction = np.array([[np.cos(angle), np.sin(angle), 0]], np.float32)
        self.speed, self.seed = speed, seed
        self.calls = []
        self.last_step = -100
        self.valid_end = 0

    def reset(self):
        self.poses[:] = self.seed_pose
        self.last_step, self.valid_end = -100, 0

    def update(self, step):
        # SONIC's tokenizer consumes 0.9 seconds of future motion. Replan every
        # 0.2 seconds with a 44-frame/30 Hz horizon so this window never pads.
        if step - self.last_step < 10:
            assert self.valid_end > step + 46
            return None
        start = 0 if self.valid_end == 0 else step + 2
        context = (np.repeat(self.seed_pose[None], 4, axis=0) if not self.valid_end else
                   interpolate_poses(self.poses[:self.valid_end], start + np.arange(4) * 50 / 30))
        mask = np.zeros((1, 11), np.int64)
        mask[0, 5] = 1  # 11 tokens = 44 frames: documented supported horizon.
        inputs = {
            "context_mujoco_qpos": context[None].astype(np.float32),
            "target_vel": np.array([self.speed], np.float32),
            "mode": np.array([2], np.int64),
            "movement_direction": self.direction,
            "facing_direction": self.direction,
            "random_seed": np.array([self.seed], np.int64),
            "height": np.array([-1], np.float32),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": mask,
        }
        before = time.perf_counter()
        outputs = dict(zip([o.name for o in self.session.get_outputs()], self.session.run(None, inputs)))
        elapsed = time.perf_counter() - before
        count = int(outputs["num_pred_frames"].reshape(-1)[0])
        raw = outputs["mujoco_qpos"][0, :count]
        assert count == 44 and np.isfinite(raw).all()
        predicted = resample_qpos(raw, 1 / 30, 1 / 50)
        if self.valid_end:
            # Match the deployment's eight-frame cross-fade, including SLERP.
            n = min(8, len(predicted), self.valid_end - start)
            old = self.poses[start:start + n].copy()
            new_quaternions = predicted[:n, [4, 5, 6, 3]].copy()
            weights = np.linspace(0, 1, n)
            predicted[:n] = old * (1 - weights[:, None]) + predicted[:n] * weights[:, None]
            for i, weight in enumerate(weights):
                # Use the pre-blend resampled orientation, not a linear quaternion.
                rots = Rotation.from_quat(np.stack([old[i, [4, 5, 6, 3]], new_quaternions[i]]))
                predicted[i, 3:7] = Slerp([0, 1], rots)([weight]).as_quat()[0, [3, 0, 1, 2]]
        stop = start + len(predicted)
        if stop >= len(self.poses):
            raise RuntimeError("Online reference buffer exhausted")
        self.poses[start:stop] = predicted
        self.poses[stop:] = predicted[-1]
        self.last_step, self.valid_end = step, stop
        self.calls.append({"control_step": step, "start": start, "valid_end": stop,
                           "predicted_frames_30hz": count, "inference_seconds": elapsed,
                           "reference_root_z_min": float(predicted[:, 2].min()),
                           "reference_root_z_max": float(predicted[:, 2].max())})
        return max(0, start - 2), stop


class PlannerProbeCommand(MultiClipMotionCommand):
    def _build_loader(self):
        self.bank = validate_bank(Path(self.cfg.bank_path))
        clip = next(c for c in self.bank["clips"] if c["id"] == self.cfg.clip_id)
        self.source_clip = clip
        self.capacity = max(round(self.cfg.duration_s * 50) + 200, clip["length"] + 200)
        self.source = {}
        loader = SimpleNamespace(n_clips=2, time_step_total=self.capacity * 2,
                                 max_clip_length=self.capacity, metadata=self.bank)
        for name, tail in CHANNELS.items():
            mapped = np.memmap(Path(self.cfg.bank_path) / (name + ".bin"), mode="r", dtype=np.float32,
                               shape=(self.bank["total_frames"], *tail))
            original = np.array(mapped[clip["offset"]:clip["offset"] + clip["length"]])
            self.source[name] = original
            control = np.repeat(original[-1:], self.capacity, axis=0)
            control[:len(original)] = original
            online = np.repeat(original[:1], self.capacity, axis=0)
            setattr(loader, name, torch.from_numpy(np.concatenate([control, online])).to(self.device))
        loader.clip_offsets = torch.tensor([0, self.capacity], device=self.device)
        loader.clip_lengths = torch.full((2,), self.capacity, device=self.device, dtype=torch.long)
        loader.clip_ends = loader.clip_offsets + loader.clip_lengths
        loader.terrain_ids = torch.full((2,), clip["terrain_id"], device=self.device, dtype=torch.long)
        return loader

    def _init_task(self):
        assert self.num_envs == 2
        assert list(self.robot.joint_names) == self.bank["joint_names"]
        self.kin_model, bodies = build_model()
        assert list(bodies) == list(self.cfg.body_names)
        self.kin_data = mujoco.MjData(self.kin_model)
        model_names = [self.kin_model.joint(i).name for i in range(self.kin_model.njnt)
                       if self.kin_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE]
        assert model_names == self.bank["joint_names"]
        self.body_ids = [self.kin_model.body(n).id for n in bodies]
        seed_pose = np.concatenate([self.source["body_pos_w"][0, 0],
                                    self.source["body_quat_w"][0, 0], self.source["joint_pos"][0]])
        self.planner = OnlinePlanner(self.cfg.planner_path, seed_pose, self.capacity,
                                    speed=self.cfg.speed, heading_deg=self.cfg.heading_deg, seed=self.cfg.seed)
        origins, _ = atlas_layout(self.bank["terrain_rows"])
        self.origin = torch.tensor(origins[self.source_clip["terrain_id"]], device=self.device)
        self.origin += self._env.scene.terrain.terrain_origins[0, 0]
        self.metrics["terrain_fraction"] = torch.ones(2, device=self.device)
        # Independently check FK/layout against the source conversion at frame 0.
        check = self._channels(np.repeat(seed_pose[None], 3, axis=0))
        self.fk_initial_error = float(np.abs(check["body_pos_w"][0] - self.source["body_pos_w"][0]).max())
        assert self.fk_initial_error < 1e-5, self.fk_initial_error

    def _sample_init_frame(self, env_ids):
        self._env.scene.env_origins[env_ids] = self.origin
        return env_ids.clone(), self.motion.clip_offsets[env_ids]

    def _resample_command(self, env_ids):
        # Wrappers may reset once at construction and once before evaluation.
        # Restore the measured seed before RSI, never initialize from a prior
        # planner prediction or its finite-difference velocity.
        if 1 in env_ids.tolist():
            for name, original in self.source.items():
                getattr(self.motion, name)[self.capacity] = torch.as_tensor(original[0], device=self.device)
        super()._resample_command(env_ids)

    def _reset_task(self, env_ids, clip_ids, time_steps, origins):
        if 1 in env_ids.tolist():
            self.planner.reset()
        self._update_online()

    def _channels(self, qpos):
        positions = np.empty((len(qpos), 14, 3), np.float32)
        rotations = np.empty((len(qpos), 14, 4), np.float32)
        for i, pose in enumerate(qpos):
            self.kin_data.qpos[:] = pose
            mujoco.mj_kinematics(self.kin_model, self.kin_data)
            positions[i] = self.kin_data.xpos[self.body_ids]
            rotations[i] = self.kin_data.xquat[self.body_ids]
        return {"joint_pos": qpos[:, 7:], "joint_vel": np.gradient(qpos[:, 7:], .02, axis=0),
                "body_pos_w": positions, "body_quat_w": rotations,
                "body_lin_vel_w": np.gradient(positions, .02, axis=0),
                "body_ang_vel_w": angular_velocity(rotations, .02)}

    def _update_online(self):
        step = int(self.time_steps[1]) - self.capacity
        if step < 0:
            return
        span = self.planner.update(step)
        if span is not None:
            first, last = span
            channels = self._channels(self.planner.poses[first:last])
            for name, values in channels.items():
                array = getattr(self.motion, name)
                array[self.capacity + first:self.capacity + last] = torch.as_tensor(values, device=self.device)
                array[self.capacity + last:] = array[self.capacity + last - 1]
                if name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                    array[self.capacity + last:] = 0

    def _update_task(self):
        self._update_online()


@dataclass(kw_only=True)
class PlannerProbeCommandCfg(MultiClipMotionCommandCfg):
    bank_path: str = ""
    planner_path: str = ""
    clip_id: str = ""
    duration_s: float = 15.
    speed: float = .5
    heading_deg: float = -90.
    seed: int = 42

    def build(self, env):
        return PlannerProbeCommand(self, env)
