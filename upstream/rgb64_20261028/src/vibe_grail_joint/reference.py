"""Memory-mapped motion bank and bounded-memory, source-balanced RSI sampler."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .kinematics import CHANNELS
from .prepare import validate_bank


class BankMotionLoader:
    def __init__(self, path: str, device):
        path = Path(path)
        self.metadata = validate_bank(path)
        self.device = device
        self.time_step_total = self.metadata["total_frames"]
        self._maps = []
        for name, tail in CHANNELS.items():
            # Copy-on-write mapping: safe torch view without allocating a second CPU bank.
            array = np.memmap(
                path / f"{name}.bin", dtype=np.float32, mode="c", shape=(self.time_step_total, *tail)
            )
            self._maps.append(array)
            setattr(self, name, torch.from_numpy(array).to(device))
        clips = self.metadata["clips"]
        self.clip_lengths = torch.tensor([c["length"] for c in clips], device=device)
        self.clip_offsets = torch.tensor([c["offset"] for c in clips], device=device)
        self.clip_ends = self.clip_offsets + self.clip_lengths
        self.terrain_ids = torch.tensor([c["terrain_id"] for c in clips], device=device)
        self.n_clips = len(clips)
        self.max_clip_length = max(c["length"] for c in clips)


class JointSampler:
    def __init__(self, motion: BankMotionLoader, *, terrain_only: bool = False):
        self.motion, self.device = motion, motion.device
        self.config = motion.metadata["config"]
        self.terrain_only = terrain_only
        self.sources = list(self.config["flat_sources"])
        self.groups = {
            source: torch.tensor(
                [i for i, clip in enumerate(motion.metadata["clips"]) if clip["source"] == source],
                device=self.device,
                dtype=torch.long,
            )
            for source in ["grail", *self.sources]
        }
        required = ["grail"] if terrain_only else list(self.groups)
        if any(len(self.groups[source]) == 0 for source in required):
            raise ValueError(f"Each rank must contain all required sources: {required}")
        self.weights = torch.tensor(
            [self.config["flat_sources"][s]["weight"] for s in self.sources], device=self.device
        )
        if (self.weights <= 0).any():
            raise ValueError("Every flat source needs positive weight")

    def sample(self, env_ids, phase: float, from_zero: bool):
        n = len(env_ids)
        terrain = (
            torch.ones(n, dtype=torch.bool, device=self.device)
            if self.terrain_only
            else env_ids % (self.config["flat_to_terrain_ratio"] + 1) == 0
        )
        source_ids = torch.multinomial(self.weights, n, replacement=True)
        source_ids[terrain] = len(self.sources)
        chosen = torch.empty(n, dtype=torch.long, device=self.device)
        for i, source in enumerate([*self.sources, "grail"]):
            mask = source_ids == i
            count = int(mask.sum())
            if count == 0:
                continue
            group = self.groups[source]
            # Length weighting avoids over-sampling the short final window of a flat clip.
            weights = self.motion.clip_lengths[group].float()
            chosen[mask] = group[torch.multinomial(weights, count, replacement=True)]
        lengths = (self.motion.clip_lengths[chosen].float() * min(1.0, max(0.0, phase))).long().clamp(min=1)
        local = (
            torch.zeros(n, dtype=torch.long, device=self.device)
            if from_zero
            else (torch.rand(n, device=self.device) * lengths).long()
        )
        return chosen, self.motion.clip_offsets[chosen] + local
