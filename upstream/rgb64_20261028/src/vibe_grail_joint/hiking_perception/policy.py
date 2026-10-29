"""Additive geometry conditioning; original SONIC and command policy unchanged."""
import torch
from torch import nn
from ..velocity.policy import CommandSonicActor


class HikingGeometryActor(CommandSonicActor):
    def __init__(self, *args, geometry_dim=128, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.decoder.adapters[0].down.in_features
        self.geometry_encoder = nn.Sequential(nn.Linear(geometry_dim, 128), nn.ELU(), nn.Linear(128, width))
        nn.init.zeros_(self.geometry_encoder[-1].weight)
        nn.init.zeros_(self.geometry_encoder[-1].bias)

    def _adapter_cond(self, obs):
        return super()._adapter_cond(obs) + self.geometry_encoder(obs['hiking_latent'])


def configure_agent(agent, dimension):
    agent['actor']['class_name'] = 'vibe_grail_joint.hiking_perception.policy.HikingGeometryActor'
    agent['actor']['geometry_dim'] = dimension
    return agent
