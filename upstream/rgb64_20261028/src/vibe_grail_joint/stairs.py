"""Stair-only ViBe task, reusing the joint task's model, rewards and geometry.

The original mixed-data task is kept unchanged. This module replaces only the
reference sampler and logging, so every environment follows a paired stair clip.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import mjlab  # noqa: F401 -- initialize task discovery before ORCS
import torch
from mjlab.rl import RslRlVecEnvWrapper

from .commands import JointMotionCommand, JointMotionCommandCfg
from .env_cfg import build_env_cfg
from .prepare import validate_bank
from .terrain import atlas_layout


def validate_stairs_bank(path: Path):
    bank = validate_bank(path)
    settings = bank["config"]
    if settings.get("training_mode") != "stairs_only":
        raise ValueError("Use a bank prepared with configs/stairs_only.json")
    if settings["flat_sources"] or settings["flat_to_terrain_ratio"] != 0:
        raise ValueError("Stair-only training must not configure flat replay")
    if set(bank["source_motion_counts"]) != {"grail"}:
        raise ValueError("Stair-only bank contains a non-GRAIL source")
    if any(c["source"] != "grail" or c["terrain_id"] < 0 for c in bank["clips"]):
        raise ValueError("Every clip needs its paired stair terrain")
    return bank


class StairsSampler:
    """Sample all stair references, retaining the existing length-weighted RSI."""

    def __init__(self, motion):
        self.motion, self.device = motion, motion.device
        if not motion.n_clips or bool((motion.terrain_ids < 0).any()):
            raise ValueError("StairsSampler requires nonempty, terrain-paired clips")
        if any(c["source"] != "grail" for c in motion.metadata["clips"]):
            raise ValueError("StairsSampler refuses flat-motion clips")
        self.weights = motion.clip_lengths.float()

    def sample(self, env_ids, phase: float, from_zero: bool):
        n = len(env_ids)
        if n == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty
        chosen = torch.multinomial(self.weights, n, replacement=True)
        lengths = (
            self.motion.clip_lengths[chosen].float() * min(1.0, max(0.0, phase))
        ).long().clamp(min=1)
        local = (
            torch.zeros(n, dtype=torch.long, device=self.device)
            if from_zero
            else (torch.rand(n, device=self.device) * lengths).long()
        )
        return chosen, self.motion.clip_offsets[chosen] + local


class StairsMotionCommand(JointMotionCommand):
    def _init_task(self):
        self.sampler = StairsSampler(self.motion)
        origins, _ = atlas_layout(self.motion.metadata["terrain_rows"])
        self.atlas_origins = torch.tensor(origins, device=self.device)
        self.atlas_origins += self._env.scene.terrain.terrain_origins[0, 0]
        self.forced_clip_ids = None
        self.metrics["terrain_fraction"] = torch.zeros(self.num_envs, device=self.device)


@dataclass(kw_only=True)
class StairsMotionCommandCfg(JointMotionCommandCfg):
    terrain_only: bool = True

    def build(self, env):
        return StairsMotionCommand(self, env)


def build_stairs_env_cfg(bank_path: str, *, num_envs: int, play: bool = False):
    validate_stairs_bank(Path(bank_path))
    cfg = build_env_cfg(bank_path, num_envs=num_envs, play=play)
    old = cfg.commands["motion"]
    values = {field.name: getattr(old, field.name) for field in fields(StairsMotionCommandCfg)}
    values["terrain_only"] = True
    cfg.commands["motion"] = StairsMotionCommandCfg(**values)
    return cfg


class StairsLoggingWrapper(RslRlVecEnvWrapper):
    """Keep comparable stair metrics without taking means over an empty flat group."""

    def __init__(self, env):
        super().__init__(env)
        self._episode_returns = torch.zeros(self.num_envs, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, device=self.device)

    def step(self, actions):
        obs, reward, dones, extras = super().step(actions)
        self._episode_returns += reward
        self._episode_steps += 1
        log = extras.setdefault("log", {})
        log["Joint/stairs_step_reward"] = reward.mean()
        ended = dones.bool()
        if bool(ended.any()):
            log["Joint/stairs_episode_return"] = self._episode_returns[ended].mean()
            log["Joint/stairs_episode_steps"] = self._episode_steps[ended].mean()
        self._episode_returns[ended] = 0
        self._episode_steps[ended] = 0
        return obs, reward, dones, extras
