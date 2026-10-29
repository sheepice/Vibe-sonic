"""Bounded, online SONIC planner streams shared by independent physics worlds.

Each stream is replanned from its previous generated context, never from a stair
demonstration or privileged terrain. Environments independently select stream,
phase, terrain and spawn. Sharing amortizes the fixed-batch ONNX inference; it
does not share robot state or rewards. The 0.9 s SONIC lookahead stays valid.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation, Slerp

from ..kinematics import angular_velocity, resample_qpos
from ..online_planner_probe import interpolate_poses
from ..prepare import build_model


class PlannerPool:
    capacity = 256
    period = 10

    def __init__(self, path, seed_pose, speeds, seed=42):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.speeds = np.asarray(speeds, dtype=np.float32)
        self.seed = seed
        self.poses = np.repeat(np.repeat(seed_pose[None, None], len(speeds), 0), self.capacity, 1)
        self.valid_end = np.zeros(len(speeds), dtype=int)
        self.last_step = np.full(len(speeds), -100, dtype=int)
        self.cursor = 0
        self.calls = 0
        self.inference_seconds = 0.0
        self.model, bodies = build_model()
        self.data = mujoco.MjData(self.model)
        self.body_ids = [self.model.body(n).id for n in bodies]
        # Warm up generated walking, avoiding the standing-to-walk transient at RSI.
        for step in range(0, 41, self.period):
            for stream in range(len(speeds)):
                self._replan(stream, step)
        self.cursor = 40

    def _replan(self, stream, step):
        poses = self.poses[stream]
        valid = int(self.valid_end[stream])
        start = 0 if not valid else step + 2
        context = (
            np.repeat(poses[:1], 4, axis=0)
            if not valid
            else interpolate_poses(poses[:valid], start + np.arange(4) * 50 / 30)
        )
        mask = np.zeros((1, 11), np.int64)
        mask[0, 5] = 1  # 44 frames at 30 Hz; > 0.9 s tokenizer lookahead.
        speed = float(self.speeds[stream])
        inputs = {
            "context_mujoco_qpos": context[None].astype(np.float32),
            "target_vel": np.array([speed if speed > 0 else -1], np.float32),
            # SONIC interprets target_vel <= 0 as mode default, NOT stop.
            "mode": np.array([2 if speed > 0 else 0], np.int64),
            "movement_direction": np.array([[1, 0, 0]], np.float32),
            "facing_direction": np.array([[1, 0, 0]], np.float32),
            "random_seed": np.array([self.seed + stream], np.int64),
            "height": np.array([-1], np.float32),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            "allowed_pred_num_tokens": mask,
        }
        before = time.perf_counter()
        output = dict(zip([o.name for o in self.session.get_outputs()], self.session.run(None, inputs)))
        self.inference_seconds += time.perf_counter() - before
        self.calls += 1
        count = int(output["num_pred_frames"].reshape(-1)[0])
        if count != 44 or not np.isfinite(output["mujoco_qpos"]).all():
            raise RuntimeError("Invalid SONIC planner prediction")
        predicted = resample_qpos(output["mujoco_qpos"][0, :count], 1 / 30, 0.02)
        if valid:
            n = min(8, valid - start)
            old = poses[start : start + n].copy()
            new = predicted[:n].copy()
            weight = np.linspace(0, 1, n)
            predicted[:n] = old * (1 - weight[:, None]) + new * weight[:, None]
            for i, w in enumerate(weight):
                rotations = Rotation.from_quat(np.stack([old[i, [4, 5, 6, 3]], new[i, [4, 5, 6, 3]]]))
                predicted[i, 3:7] = Slerp([0, 1], rotations)([w]).as_quat()[0, [3, 0, 1, 2]]
        end = start + len(predicted)
        if end >= self.capacity:
            raise RuntimeError("Planner rolling buffer exhausted")
        poses[start:end] = predicted
        poses[end:] = predicted[-1]
        self.valid_end[stream], self.last_step[stream] = end, step

    def advance(self):
        """Return (changed, per-stream XY recentering), advance exactly one control step."""
        self.cursor += 1
        recenter = np.zeros((len(self.speeds), 2), np.float32)
        changed = False
        if self.cursor >= 128:
            shift = 64
            self.poses[:, :-shift] = self.poses[:, shift:].copy()
            self.poses[:, -shift:] = self.poses[:, -shift - 1 : -shift]
            self.cursor -= shift
            self.valid_end -= shift
            self.last_step -= shift
            recenter[:] = self.poses[:, self.cursor, :2]
            self.poses[:, :, :2] -= recenter[:, None, :]
            changed = True
        for i in range(len(self.speeds)):
            if self.cursor - self.last_step[i] >= self.period:
                self._replan(i, self.cursor)
                changed = True
        if np.any(self.valid_end <= self.cursor + 46):
            raise RuntimeError("Online reference would pad the SONIC future window")
        return changed, recenter

    def channels(self):
        result = {
            name: []
            for name in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            )
        }
        for poses in self.poses:
            pos = np.empty((self.capacity, 14, 3), np.float32)
            quat = np.empty((self.capacity, 14, 4), np.float32)
            for i, pose in enumerate(poses):
                self.data.qpos[:] = pose
                mujoco.mj_kinematics(self.model, self.data)
                pos[i] = self.data.xpos[self.body_ids]
                quat[i] = self.data.xquat[self.body_ids]
            values = {
                "joint_pos": poses[:, 7:],
                "joint_vel": np.gradient(poses[:, 7:], 0.02, axis=0),
                "body_pos_w": pos,
                "body_quat_w": quat,
                "body_lin_vel_w": np.gradient(pos, 0.02, axis=0),
                "body_ang_vel_w": angular_velocity(quat, 0.02),
            }
            for name, value in values.items():
                result[name].append(value)
        return {name: np.concatenate(value).astype(np.float32) for name, value in result.items()}
