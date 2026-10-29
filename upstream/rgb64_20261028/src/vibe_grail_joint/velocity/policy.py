"""Original SONIC + ViBe with a zero-initialized explicit-command conditioning path."""

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models.extractor_model import ExtractorSonicAdapterModel
from torch import nn


class CommandSonicActor(ExtractorSonicAdapterModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.decoder.adapters[0].down.in_features
        self.command_encoder = nn.Sequential(nn.Linear(8, 64), nn.ELU(), nn.Linear(64, width))
        nn.init.zeros_(self.command_encoder[-1].weight)
        nn.init.zeros_(self.command_encoder[-1].bias)
        self._latest_update_obs = None

    def _adapter_cond(self, obs):
        base = super()._adapter_cond(obs)
        return base + self.command_encoder(obs["velocity_command"])

    def forward(self, obs, *args, **kwargs):
        if torch.is_grad_enabled():
            self._latest_update_obs = obs
        return super().forward(obs, *args, **kwargs)

    @torch.no_grad()
    def sonic_action(self, obs):
        x = torch.cat([self.encode_tokens(obs), self._get_proprio(obs)], -1)
        for i, layer in enumerate(self.decoder.base_linears):
            if i:
                x = self.decoder.activation(x)
            x = layer(x)
        return x

    def as_onnx(self, *args, **kwargs):
        raise NotImplementedError("Use a command-aware export; the legacy ViBe export omits velocity_command")


class PreservationPPO(PPO):
    def __init__(self, *args, preservation_coef=0.02, **kwargs):
        super().__init__(*args, **kwargs)
        self.preservation_coef = preservation_coef
        self._preservation_sum = 0.0
        self._preservation_count = 0

    def _extra_backward(self):
        obs = self._raw_actor._latest_update_obs
        ids = (obs["velocity_command"][:, 7] > 0.5).nonzero().flatten()[:128]
        # All ranks execute the same trainable graph, even when a batch has no flat data.
        if not len(ids):
            ids = torch.zeros(1, device=self.device, dtype=torch.long)
            weight = 0.0
        else:
            weight = self.preservation_coef
        sample = obs[ids]
        std = self._raw_actor.output_std.detach()
        if std.ndim > 1:
            std = std[ids]
        std = std.clamp(min=0.05)
        teacher = self._raw_actor.sonic_action(sample)
        student = self._raw_actor(sample)
        loss = 0.5 * ((student - teacher) / std).square().mean()
        (weight * loss).backward()
        self._preservation_sum += float(loss.detach())
        self._preservation_count += 1

    def update(self):
        self._preservation_sum, self._preservation_count = 0.0, 0
        losses, infos = super().update()
        losses["flat_sonic_preservation"] = self._preservation_sum / max(1, self._preservation_count)
        self._raw_actor._latest_update_obs = None
        return losses, infos
