"""Flat DAgger MSE is accumulated in the same optimizer step as PPO."""

import hashlib
import inspect
import math
from pathlib import Path

import torch
from rsl_rl.models.sonic_base_model import SonicBaseModel

from .posture_policy import UpperBodyPreservationPPO


def build_teacher(obs, actor_cfg, obs_groups, checkpoint, device):
    """An independent original SONIC: no LoRA, visual or command branch."""
    allowed = inspect.signature(SonicBaseModel.__init__).parameters
    kwargs = {k: v for k, v in actor_cfg.items() if k in allowed}
    kwargs.update(base_checkpoint=checkpoint, freeze_base=True, cache_tokens=False,
                  distribution_cfg=None)
    teacher = SonicBaseModel(obs, obs_groups, "actor", 29, **kwargs).to(device)
    teacher.requires_grad_(False)
    teacher.eval()
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    return teacher, digest


class DaggerPPO(UpperBodyPreservationPPO):
    def __init__(self, *args, dagger_coef=0.8, dagger_batch_size=256, **kwargs):
        if not math.isfinite(dagger_coef) or dagger_coef < 0 or dagger_batch_size < 1:
            raise ValueError("Invalid DAgger coefficient or batch size")
        if kwargs.get("preservation_coef", 0) != 0:
            raise ValueError("DAgger replaces the old flat preservation prior")
        kwargs["preservation_coef"] = 0.0
        super().__init__(*args, **kwargs)
        self.dagger_coef, self.dagger_batch_size = dagger_coef, int(dagger_batch_size)
        self.dagger_buffer = None
        self.teacher_sha256 = None

    def _extra_backward(self):
        # Preserve the pre-existing online waist/arm prior. Its inherited flat
        # prior has exactly zero coefficient; flat examples never enter PPO.
        super()._extra_backward()
        obs, label = self.dagger_buffer.sample(self.dagger_batch_size, self.device)
        student = self._raw_actor(obs, stochastic_output=False)
        # Ordinary action units; mean over examples AND all 29 joints. No std
        # normalization or hidden 1/2 factor changes the configured weight.
        mse = (student - label).square().mean()
        (self.dagger_coef * mse).backward()
        self._dagger_sum += float(mse.detach())
        self._dagger_count += 1

    def update(self):
        self._dagger_sum, self._dagger_count = 0.0, 0
        losses, info = super().update()
        losses.pop("flat_sonic_preservation", None)
        mse = self._dagger_sum / max(self._dagger_count, 1)
        losses.update(dagger_action_mse=mse, dagger_weighted=self.dagger_coef * mse)
        info["dagger_action_rmse"] = math.sqrt(mse)
        self.dagger_buffer.commit()
        info["dagger_history_samples"] = len(self.dagger_buffer)
        for key, count in enumerate(self.dagger_buffer.seen.tolist()):
            info[f"dagger_stratum_{key}_collected"] = count
        return losses, info

    def save(self):
        state = super().save()
        state["dagger"] = dict(version=1, coefficient=self.dagger_coef,
                               teacher_sha256=self.teacher_sha256)
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        expected = dict(version=1, coefficient=self.dagger_coef, teacher_sha256=self.teacher_sha256)
        if loaded_dict.get("dagger") != expected:
            raise ValueError("Resume requires a matching DAgger run and the same frozen teacher")
        return super().load(loaded_dict, load_cfg, strict)
