"""Pair every reference with its own stair geometry; preserve ViBe's RSI schedule."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from orcs.core.mdp.commands import MultiClipMotionCommand, MultiClipMotionCommandCfg

from .reference import BankMotionLoader, JointSampler
from .terrain import atlas_layout


class JointMotionCommand(MultiClipMotionCommand):
    def _build_loader(self):
        motion = BankMotionLoader(self.cfg.bank_path, self.device)
        if list(self.robot.joint_names) != motion.metadata["joint_names"]:
            raise ValueError("Reference bank and robot joint orders differ")
        if list(self.cfg.body_names) != motion.metadata["body_names"]:
            raise ValueError("Reference bank and tracked body orders differ")
        return motion

    def _init_task(self):
        self.sampler = JointSampler(self.motion, terrain_only=self.cfg.terrain_only)
        origins, _ = atlas_layout(self.motion.metadata["terrain_rows"])
        self.atlas_origins = torch.tensor(origins, device=self.device)
        self.atlas_origins += self._env.scene.terrain.terrain_origins[0, 0]
        self.forced_clip_ids = None  # deterministic evaluation supplies one clip per env
        self.metrics["terrain_fraction"] = torch.zeros(self.num_envs, device=self.device)

    def _sample_init_frame(self, env_ids):
        if self.forced_clip_ids is None:
            clips, frames = self.sampler.sample(env_ids, self._init_phase_max, self.cfg.start_from_zero)
        else:
            clips = self.forced_clip_ids[env_ids]
            frames = self.motion.clip_offsets[clips]
        terrain_ids = self.motion.terrain_ids[clips]  # -1 intentionally selects the final flat origin
        self._env.scene.env_origins[env_ids] = self.atlas_origins[terrain_ids]
        self.metrics["terrain_fraction"][env_ids] = (terrain_ids >= 0).float()
        return clips, frames


@dataclass(kw_only=True)
class JointMotionCommandCfg(MultiClipMotionCommandCfg):
    bank_path: str = ""
    terrain_only: bool = False

    def build(self, env):
        return JointMotionCommand(self, env)
