"""One SONIC reference interface for online locomotion and original motion replay."""

from dataclasses import dataclass

import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul
from orcs.core.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg

from ..kinematics import CHANNELS
from ..reference import BankMotionLoader
from ..terrain import atlas_layout
from .planner import PlannerPool


class VelocityMotionCommand(MultiClipMotionCommand):
    def _build_loader(self):
        motion = BankMotionLoader(self.cfg.bank_path, self.device)
        if list(self.robot.joint_names) != motion.metadata["joint_names"]:
            raise ValueError("SONIC joint order does not match the bank")
        if list(self.cfg.body_names) != motion.metadata["body_names"]:
            raise ValueError("Tracked body order does not match the bank")
        self.replay_frames, self.replay_clips = motion.time_step_total, motion.n_clips
        count = len(self.cfg.planner_speeds)
        for name, tail in CHANNELS.items():
            setattr(
                motion,
                name,
                torch.cat(
                    [
                        getattr(motion, name),
                        torch.zeros(count * PlannerPool.capacity, *tail, device=self.device),
                    ]
                ),
            )
        extra_offsets = self.replay_frames + torch.arange(count, device=self.device) * PlannerPool.capacity
        motion.clip_offsets = torch.cat([motion.clip_offsets, extra_offsets])
        motion.clip_lengths = torch.cat(
            [motion.clip_lengths, torch.full((count,), PlannerPool.capacity, device=self.device)]
        )
        motion.clip_ends = motion.clip_offsets + motion.clip_lengths
        motion.terrain_ids = torch.cat([motion.terrain_ids, torch.full((count,), -1, device=self.device)])
        motion.n_clips += count
        motion.time_step_total += count * PlannerPool.capacity
        return motion

    def _init_task(self):
        # Fixed groups make per-task curves and replay fractions unambiguous.
        slot = torch.arange(self.num_envs, device=self.device) % 10
        self.online = slot < 6
        self.paired = (slot >= 6) & (slot < 8)
        self.flat = slot >= 8
        self.online_ids = self.online.nonzero().flatten()
        self.stream = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.phase = torch.zeros_like(self.stream)
        self.tile = torch.zeros_like(self.stream)
        self.speed = torch.zeros(self.num_envs, device=self.device)
        self.success_hold = torch.zeros_like(self.stream)
        self.success_step = -1
        self._last_pool_step = int(self._env.common_step_counter)
        self._reset_count = 0
        self.metrics["terrain_fraction"] = (~self.flat).float()
        atlas, _ = atlas_layout(self.motion.metadata["terrain_rows"])
        offset = self._env.scene.terrain.terrain_origins[0, 0]
        self.atlas_origins = torch.as_tensor(atlas, device=self.device) + offset
        self.tile_origins = (
            torch.tensor([t["origin"] for t in self.cfg.random_tiles], device=self.device) + offset
        )
        self.tile_data = {
            key: torch.tensor([t[key] for t in self.cfg.random_tiles], device=self.device)
            for key in ("n", "tread", "height", "start", "plateau", "goal", "level")
        }
        self.tile_data["kind"] = torch.tensor(
            [("up", "down", "up_down", "flat").index(t["kind"]) for t in self.cfg.random_tiles],
            device=self.device,
        )
        clips = self.motion.metadata["clips"]
        self.source_groups = {
            source: torch.tensor(
                [i for i, c in enumerate(clips) if c["source"] == source],
                device=self.device,
                dtype=torch.long,
            )
            for source in ("grail", *self.motion.metadata["config"]["flat_sources"])
        }
        self.flat_sources = list(self.motion.metadata["config"]["flat_sources"])
        self.flat_weights = torch.tensor(
            [self.motion.metadata["config"]["flat_sources"][s]["weight"] for s in self.flat_sources],
            device=self.device,
        )
        if any(len(v) == 0 for v in self.source_groups.values()):
            raise ValueError("Every rank must retain GRAIL and each original flat source")
        seed_pose = np.concatenate(
            [
                self.robot.data.default_root_state[0, :7].cpu().numpy(),
                self.robot.data.default_joint_pos[0].cpu().numpy(),
            ]
        ).astype(np.float32)
        seed_pose[:2], seed_pose[3:7] = 0, [1, 0, 0, 0]
        self.pool = PlannerPool(self.cfg.planner_path, seed_pose, self.cfg.planner_speeds, self.cfg.seed)
        self._upload_pool()

    def _upload_pool(self):
        for name, array in self.pool.channels().items():
            getattr(self.motion, name)[self.replay_frames :] = torch.as_tensor(array, device=self.device)

    def _sample_init_frame(self, env_ids):
        chosen = torch.empty(len(env_ids), device=self.device, dtype=torch.long)
        for task in ("paired", "flat"):
            mask = getattr(self, task)[env_ids]
            local = mask.nonzero().flatten()
            if not len(local):
                continue
            if task == "paired":
                groups = [(local, self.source_groups["grail"])]
            else:
                sources = torch.multinomial(self.flat_weights, len(local), replacement=True)
                groups = [
                    (local[sources == i], self.source_groups[s]) for i, s in enumerate(self.flat_sources)
                ]
            for idx, group in groups:
                if len(idx):
                    chosen[idx] = group[
                        torch.multinomial(self.motion.clip_lengths[group].float(), len(idx), replacement=True)
                    ]
        mask = self.online[env_ids]
        ids = env_ids[mask]
        if len(ids):
            # Idle is a valid zero-speed task, sampled less often than walking.
            weights = torch.tensor(
                [0.06] + [0.94 / (len(self.cfg.planner_speeds) - 1)] * (len(self.cfg.planner_speeds) - 1),
                device=self.device,
            )
            self.stream[ids] = torch.multinomial(weights, len(ids), replacement=True)
            self.phase[ids] = torch.randint(0, 20, (len(ids),), device=self.device)
            chosen[mask] = self.replay_clips + self.stream[ids]
            iteration = int(getattr(self._env, "policy_update_count", 0))
            max_level = min(7, 1 + iteration // self.cfg.curriculum_interval)
            levels = torch.randint(0, max_level + 1, (len(ids),), device=self.device)
            kinds = torch.multinomial(
                torch.tensor([0.35, 0.25, 0.25, 0.15], device=self.device), len(ids), replacement=True
            )
            self.tile[ids] = levels * 4 + kinds
            speeds = torch.tensor(self.cfg.planner_speeds, device=self.device)[self.stream[ids]]
            self.speed[ids] = torch.where(
                speeds > 0, (speeds + torch.empty_like(speeds).uniform_(-0.04, 0.04)).clamp(min=0.15), 0.0
            )
        lengths = (self.motion.clip_lengths[chosen].float() * self._init_phase_max).long().clamp(min=1)
        local_frames = (torch.rand(len(env_ids), device=self.device) * lengths).long()
        if self.cfg.start_from_zero:
            local_frames.zero_()
        local_frames[mask] = self.pool.cursor - self.phase[ids]
        frames = self.motion.clip_offsets[chosen] + local_frames
        replay_ids = env_ids[~mask]
        terrain_ids = self.motion.terrain_ids[chosen[~mask]]
        self._env.scene.env_origins[replay_ids] = self.atlas_origins[terrain_ids]
        if len(ids):
            spawn = self.tile_origins[self.tile[ids]].clone()
            spawn[:, 0] += torch.empty(len(ids), device=self.device).uniform_(-0.3, 0.3)
            spawn[:, 1] += torch.empty(len(ids), device=self.device).uniform_(-0.25, 0.25)
            spawn[:, 2] += self.ground_height(spawn[:, :2], ids)
            ref = self.motion.body_pos_w[frames[mask], 0]
            spawn[:, :2] -= ref[:, :2]
            self._env.scene.env_origins[ids] = spawn
        return chosen, frames

    def _reset_task(self, env_ids, clip_ids, time_steps, origins):
        self.success_hold[env_ids] = 0
        ids = env_ids[self.online[env_ids]]
        if len(ids):
            t = self.time_steps[ids]
            pos = self.motion.body_pos_w[t, 0] + self._env.scene.env_origins[ids]
            yaw = torch.empty(len(ids), device=self.device).uniform_(-0.15, 0.15)
            ori = quat_mul(quat_from_euler_xyz(yaw * 0, yaw * 0, yaw), self.motion.body_quat_w[t, 0])
            self._write_reference_state_to_sim(
                ids,
                pos,
                ori,
                self.motion.body_lin_vel_w[t, 0],
                self.motion.body_ang_vel_w[t, 0],
                self.motion.joint_pos[t],
                self.motion.joint_vel[t],
            )

    def _update_command(self, env_ids=None):
        iteration = int(getattr(self._env, "policy_update_count", 0))
        self._init_phase_max = max(0.0, 1 - iteration / max(1, self.cfg.init_phase_anneal_iterations))
        selected = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        replay = selected[~self.online[selected]]
        self.time_steps[replay] += 1
        last = self.motion.clip_ends[self._clip_ids[replay]] - 1
        self._steps_past_end[replay] += (self.time_steps[replay] > last).long()
        self.time_steps[replay] = torch.minimum(self.time_steps[replay], last)
        step = int(self._env.common_step_counter)
        if env_ids is None and step != self._last_pool_step:
            changed, recenter = self.pool.advance()
            self._last_pool_step = step
            ids = self.online_ids
            self._env.scene.env_origins[ids, :2] += torch.as_tensor(recenter, device=self.device)[
                self.stream[ids]
            ]
            if changed:
                self._upload_pool()
            # Mild within-episode speed changes keep command supervision separate
            # from the reference's measured velocity. No discontinuous pose switch.
            if step % 200 == 0:
                nominal = torch.tensor(self.cfg.planner_speeds, device=self.device)[self.stream[ids]]
                self.speed[ids] = torch.where(
                    nominal > 0,
                    (nominal + torch.empty_like(nominal).uniform_(-0.05, 0.05)).clamp(min=0.15),
                    0.0,
                )
        ids = self.online_ids
        self.time_steps[ids] = (
            self.replay_frames + self.stream[ids] * self.pool.capacity + self.pool.cursor - self.phase[ids]
        )
        self.update_relative_body_poses()

    def ground_height(self, xy, env_ids=None):
        """Privileged reward/reset geometry only; never included in actor observations."""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        tile = self.tile[ids]
        expand = (slice(None),) + (None,) * (xy.ndim - 2)
        x = xy[..., 0] - self.tile_origins[tile, 0][expand]
        y = xy[..., 1] - self.tile_origins[tile, 1][expand]
        p = {k: v[tile][expand] for k, v in self.tile_data.items()}
        up = ((x - p["start"]) / p["tread"]).floor().add(1).clamp(min=0)
        up = torch.minimum(up, p["n"])
        down = (
            ((x - p["start"] - p["n"] * p["tread"] - p["plateau"]) / p["tread"]).floor().add(1).clamp(min=0)
        )
        down = torch.minimum(down, p["n"])
        h = torch.where(p["kind"] == 1, p["n"] - up, up)
        h = torch.where(p["kind"] == 2, up - down, h) * p["height"]
        return torch.where((p["kind"] == 3) | (x < -2) | (x > 12) | (y.abs() > 2), 0.0, h)


@dataclass(kw_only=True)
class VelocityMotionCommandCfg(MultiClipMotionCommandCfg):
    bank_path: str = ""
    planner_path: str = ""
    planner_speeds: tuple = (0.0, 0.25, 0.4, 0.55, 0.7)
    random_tiles: tuple = ()
    curriculum_interval: int = 1500
    seed: int = 42

    def build(self, env):
        return VelocityMotionCommand(self, env)
