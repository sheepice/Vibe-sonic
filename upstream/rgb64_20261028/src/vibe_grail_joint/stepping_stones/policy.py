"""Deployable actor variants: RGB/proprio/command only, no simulator imports."""
import torch
from torch import nn
from ..hiking_gaussgym.policy import GaussGymActor

class StoneActor(GaussGymActor):
    def __init__(self,*args,stone_variant='dual',**kwargs):
        super().__init__(*args,**kwargs)
        if stone_variant not in ('dual','geometry_only','moe'):raise ValueError(stone_variant)
        self.stone_variant=stone_variant
        if stone_variant=='geometry_only':
            for p in self.extractors.parameters():p.requires_grad_(False)
        if stone_variant=='moe':
            # Small condition residual experts, not hard oracle terrain selection.
            self.experts=nn.ModuleList([nn.Sequential(nn.Linear(744,96),nn.ELU(),nn.Linear(96,134)) for _ in range(3)])
            self.router=nn.Sequential(nn.Linear(744,64),nn.ELU(),nn.Linear(64,3))
            for expert in self.experts:
                nn.init.zeros_(expert[-1].weight);nn.init.zeros_(expert[-1].bias)
            self.last_routes=None
    def update_normalization(self,obs):
        # Fine-tune using checkpoint normalization on all ranks and deployment.
        return
    def _get_adapter_latent(self,obs):
        if self.stone_variant!='geometry_only':return super()._get_adapter_latent(obs)
        plain=torch.cat([obs[g] for g in self.plain_adapter_groups],-1)
        return torch.cat((self.adapter_normalizer(plain),torch.zeros(len(plain),128,device=plain.device,dtype=plain.dtype)),-1)
    def _adapter_cond(self,obs):
        base=super()._adapter_cond(obs)
        if self.stone_variant!='moe':return base
        x=torch.cat((obs['hiking_latent'],obs['q_proprio'],obs['velocity_command']),-1)
        # Clip only the extra conditioning path; preserve checkpoint branch exactly.
        x=x.sign()*torch.log1p(x.abs())
        weights=self.router(x).softmax(-1);self.last_routes=weights
        residual=torch.stack([e(x) for e in self.experts],-2)
        return base+(weights[...,None]*residual).sum(-2)

def configure_agent(agent,dimension,variant='dual'):
    agent['actor'].update(class_name='vibe_grail_joint.stepping_stones.policy.StoneActor',geometry_dim=dimension,stone_variant=variant)
    if variant=='moe':agent['algorithm']['class_name']='vibe_grail_joint.stepping_stones.ppo.StoneMoEPPO'
    return agent
