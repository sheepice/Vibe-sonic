"""Regularize deterministic PD targets using only actor-observable quantities."""
import math

import torch

from ..deploy.safety import Limits
from ..velocity.dagger_policy import DaggerPPO
from .ppo import StoneMoEPPO


class DeploymentTargetMixin:
    def __init__(self, *args, deployment_contract, target_training_options, **kwargs):
        super().__init__(*args, **kwargs)
        self.deployment_contract = deployment_contract
        self.target_options = target_training_options
        self.target_coefficient = float(target_training_options.get('loss_coefficient', .1))
        self.target_std = float(target_training_options.get('penalty_std_rad', .2))
        self.target_samples = int(target_training_options.get('max_samples', 256))
        if not math.isfinite(self.target_coefficient) or self.target_coefficient < 0 or not self.target_std > 0 or self.target_samples < 1:
            raise ValueError('Invalid deployment target regularizer')
        self.target_default = torch.tensor(deployment_contract['default_joint_pos'], device=self.device)
        self.target_scale = torch.tensor(deployment_contract['scale'], device=self.device)
        self.target_bounds = torch.tensor(deployment_contract['joint_limits'], device=self.device)

    def _extra_backward(self):
        actor = self._raw_actor
        obs = actor._latest_update_obs
        if obs is None:
            raise RuntimeError('Target regularizer requires the PPO observation minibatch')
        try:
            super()._extra_backward()
            ids = (obs['velocity_command'][:, 5] > .5).nonzero().flatten()[:self.target_samples]
            count = len(ids)
            if not count:
                ids = torch.zeros(1, dtype=torch.long, device=self.device)
            sample = obs[ids]
            target = self.target_default+self.target_scale*actor(sample, stochastic_output=False)
            # Existing 930-D term-major history: gyro, q, dq, applied action,
            # gravity, ten frames each. Parity with ObservationBuilder is tested.
            previous_action = sample['policy'][:, 871:900]
            if sample['policy'].shape[-1] != 930 or sample['q_proprio'].shape[-1] != 64:
                raise ValueError('Unexpected actor history/proprio layout')
            previous = self.target_default+self.target_scale*previous_action
            measured = self.target_default+sample['q_proprio'][:, 6:35]
            limits = Limits()
            bounded = target.clamp(self.target_bounds[:, 0], self.target_bounds[:, 1])
            limited = bounded.clamp(previous-limits.target_slew_rad_per_step,
                                    previous+limits.target_slew_rad_per_step)
            projection = (target-bounded).square().sum(-1).mean()
            slew = (bounded-limited).square().sum(-1).mean()
            tracking = ((limited-measured).abs()-limits.max_target_error_rad).clamp_min(0).square().sum(-1).mean()
            loss = (projection+slew+tracking)/self.target_std**2
            coefficient = self.target_coefficient if count else 0.
            (coefficient*loss).backward()
            self.target_loss_sum += coefficient*float(loss.detach())
            self.target_batch_count += 1
            self.target_projection_max = max(self.target_projection_max, float((target-bounded).detach().abs().max()))
        finally:
            actor._latest_update_obs = obs

    def update(self):
        self.target_loss_sum = 0.
        self.target_projection_max = 0.
        self.target_batch_count = 0
        losses, info = super().update()
        losses['deployment_target_feasibility'] = self.target_loss_sum/max(self.target_batch_count, 1)
        info['deployment_mean_target_max_projection_rad'] = self.target_projection_max
        return losses, info

    def save(self):
        state = super().save()
        state['deployment_target_training'] = dict(contract=self.deployment_contract, options=self.target_options)
        return state

    def load(self, state, load_cfg=None, strict=True):
        expected = dict(contract=self.deployment_contract, options=self.target_options)
        if state.get('deployment_target_training') != expected:
            raise ValueError('Exact resume requires the same deployment target contract and loss')
        return super().load(state, load_cfg, strict)


class DeploymentTargetPPO(DeploymentTargetMixin, DaggerPPO):
    pass


class DeploymentTargetMoEPPO(DeploymentTargetMixin, StoneMoEPPO):
    pass
