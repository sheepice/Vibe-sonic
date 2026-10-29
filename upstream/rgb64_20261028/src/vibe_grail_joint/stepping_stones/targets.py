"""Measured-state actuator conditioning shared by training and evaluation.

No terrain, contact, root position, or base velocity is read here. Training
projects exploratory actions instead of terminating every noisy sample. Final
evaluation additionally rejects the same excessive projections as real control.
"""
from dataclasses import asdict
import math

import torch

from ..deploy.safety import Limits


def condition_batch(target, previous, measured, bounds, limits=None, projection_bounds=None):
    """Torch equivalent of deploy.condition_target + target_fault, per world."""
    limits = limits or Limits()
    # Round clipping endpoints inward, while preserving the physical JSON
    # bounds for the subsequent check. Do not relax any rejection threshold.
    if projection_bounds is None:
        projection_bounds = bounds.to(dtype=target.dtype).clone()
        low, high = projection_bounds[:,0], projection_bounds[:,1]
        projection_bounds[:,0] = torch.where(low.double()<bounds[:,0],
            torch.nextafter(low,torch.full_like(low,float('inf'))),low)
        projection_bounds[:,1] = torch.where(high.double()>bounds[:,1],
            torch.nextafter(high,torch.full_like(high,float('-inf'))),high)
    bounded = target.clamp(projection_bounds[:, 0], projection_bounds[:, 1])
    projected = (target-bounded).abs().amax(-1)
    limited = bounded.clamp(previous-limits.target_slew_rad_per_step,
                            previous+limits.target_slew_rad_per_step)
    slew = (bounded-limited).abs().amax(-1)
    tracking = (limited-measured).abs().amax(-1)
    step = (limited-previous).abs().amax(-1)
    finite = torch.isfinite(target).all(-1) & torch.isfinite(previous).all(-1) & torch.isfinite(measured).all(-1)
    fault = (~finite | (projected > limits.max_limit_projection_rad)
             | (slew > limits.max_slew_projection_rad)
             | (tracking > limits.max_target_error_rad)
             | (step > limits.max_target_step_rad)
             | (limited.double() < bounds[:, 0]).any(-1) | (limited.double() > bounds[:, 1]).any(-1))
    return limited, dict(projection=projected, slew=slew, tracking=tracking, step=step, fault=fault)


def enabled(settings):
    return bool(settings['stones'].get('deployment_targets', {}).get('enabled', False))


class DeploymentTargets:
    def __init__(self, env, contract, *, strict):
        self.env, self.contract, self.strict = env, contract, strict
        if abs(env.step_dt-contract['control_dt']) > 1e-8 or abs(env.step_dt-.02) > 1e-8:
            raise ValueError('Deployment target conditioning requires the same 50 Hz control period')
        self.robot = env.scene['robot']
        action = env.action_manager.get_term('joint_pos')
        if list(action.target_names) != contract['joint_names']:
            raise ValueError('Deployment target joint order mismatch')
        self.default = torch.tensor(contract['default_joint_pos'], device=env.device)
        self.scale = torch.tensor(contract['scale'], device=env.device)
        self.bounds = torch.tensor(contract['joint_limits'], device=env.device, dtype=torch.float64)
        from .safety import inward_bounds
        self.projection_bounds = torch.tensor(inward_bounds(contract),device=env.device)
        actual_scale = torch.as_tensor(action.scale, device=env.device)
        if not torch.allclose(actual_scale, self.scale.expand_as(actual_scale), atol=1e-6):
            raise ValueError('Deployment target action scale mismatch')
        actual_offset = torch.as_tensor(action.offset, device=env.device)
        if not torch.allclose(actual_offset, self.default.expand_as(actual_offset), atol=1e-6):
            raise ValueError('Deployment target offset mismatch')
        self.previous = self.robot.data.joint_pos.clone()
        self.fault = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.stats = {}
        self.last_diagnostic = None

    def reset(self, ids=None):
        if ids is None:
            self.previous.copy_(self.robot.data.joint_pos)
            self.fault.zero_()
        else:
            self.previous[ids] = self.robot.data.joint_pos[ids]
            self.fault[ids] = False

    @torch.no_grad()
    def apply(self, action):
        if not torch.isfinite(action).all():
            raise RuntimeError('Nonfinite action before deployment conditioning')
        target = self.default+self.scale*action
        limited, stats = condition_batch(target, self.previous, self.robot.data.joint_pos, self.bounds,
                                        projection_bounds=self.projection_bounds)
        self.fault.copy_(stats['fault'])
        self.fault |= self.robot.data.projected_gravity_b[:, 2] > -math.cos(Limits().max_tilt_rad)
        if self.strict:
            self.last_diagnostic = dict(target=target.clone(), previous=self.previous.clone(),
                limited=limited.clone(), measured=self.robot.data.joint_pos.clone(),
                gravity_z=self.robot.data.projected_gravity_b[:,2].clone(),
                **{k:v.clone() for k,v in stats.items()})
            # A failed world is scored as a guard failure by the evaluator. Do
            # not execute its rejected target while other worlds finish.
            limited = torch.where(self.fault[:, None], self.previous, limited)
        self.previous.copy_(limited)
        self.stats = {
            'DeploymentTarget/projection_world_fraction': (stats['projection'] > 1e-6).float().mean(),
            'DeploymentTarget/max_projection_rad': stats['projection'].max(),
            'DeploymentTarget/max_slew_projection_rad': stats['slew'].max(),
            'DeploymentTarget/max_tracking_error_rad': stats['tracking'].max(),
            'DeploymentTarget/sample_guard_violation_fraction': self.fault.float().mean(),
        }
        return (limited-self.default)/self.scale

    def diagnostic_row(self, index):
        if self.last_diagnostic is None:
            return None
        d = self.last_diagnostic
        limits = Limits()
        low, high = self.bounds[:,0], self.bounds[:,1]
        bounded_error = torch.maximum(low-d['limited'][index].double(),d['limited'][index].double()-high).clamp_min(0)
        causes = []
        for name, triggered in [
            ('excessive_joint_limit_projection', d['projection'][index]>limits.max_limit_projection_rad),
            ('excessive_target_slew', d['slew'][index]>limits.max_slew_projection_rad),
            ('joint_limit', bounded_error.max()>0),
            ('target_discontinuity', d['step'][index]>limits.max_target_step_rad),
            ('target_tracking_error', d['tracking'][index]>limits.max_target_error_rad),
            ('excessive_tilt', d['gravity_z'][index]>-math.cos(limits.max_tilt_rad)),
        ]:
            if bool(triggered):causes.append(name)
        error = (d['target'][index]-d['measured'][index]).abs()
        joint = int(error.argmax())
        return dict(causes=causes,projection_rad=float(d['projection'][index]),
                    slew_projection_rad=float(d['slew'][index]),tracking_error_rad=float(d['tracking'][index]),
                    bound_roundoff_rad=float(bounded_error.max()),
                    tracking_joint=self.contract['joint_names'][joint],
                    raw_target=d['target'][index].cpu().tolist(),
                    applied_target=d['limited'][index].cpu().tolist(),
                    previous_target=d['previous'][index].cpu().tolist(),
                    measured_q=d['measured'][index].cpu().tolist())


def install(env, settings, *, strict=False):
    if not enabled(settings):
        return None
    if hasattr(env, 'deployment_targets'):
        raise RuntimeError('Deployment conditioning already installed')
    from ..deploy.export import robot_contract
    conditioner = DeploymentTargets(env, robot_contract(settings['bank_root']+'/rank_0'), strict=strict)
    previous_step, previous_reset = env.step, env._reset_idx
    checks_remaining = 10

    def step(actions):
        nonlocal checks_remaining
        result = previous_step(conditioner.apply(actions))
        if checks_remaining:
            newest_action = result[0]['policy'][:, 871:900]
            error = (newest_action-env.action_manager.action).abs().max()
            if error > 1e-6:
                raise RuntimeError(f'Actor history is not the applied action: {float(error)}')
            conditioner.stats['DeploymentTarget/applied_action_history_max_error'] = error
            checks_remaining -= 1
        result[-1].setdefault('log', {}).update(conditioner.stats)
        return result

    def reset(ids=None):
        result = previous_reset(ids)
        conditioner.reset(ids)
        return result

    env.deployment_targets = conditioner
    env.step, env._reset_idx = step, reset
    return conditioner


def guard_failed(env):
    if hasattr(env, 'deployment_targets'):
        return env.deployment_targets.fault
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def guard_diagnostic(env, index):
    return env.deployment_targets.diagnostic_row(index) if hasattr(env,'deployment_targets') else None


def configure_algorithm(agent, env, settings):
    if not enabled(settings):
        return
    variant = settings['stones']['variant']
    klass = 'DeploymentTargetMoEPPO' if variant == 'moe' else 'DeploymentTargetPPO'
    agent['algorithm'].update(
        class_name='vibe_grail_joint.stepping_stones.target_ppo.'+klass,
        deployment_contract=env.unwrapped.deployment_targets.contract,
        target_training_options=settings['stones']['deployment_targets'],
    )


def contract_summary(settings):
    active = enabled(settings)
    return dict(enabled=active, limits=asdict(Limits()),
                sampled_training_actions=(
                    'joint limits and slew conditioned; noisy overshoot logged, not a reset' if active else
                    'original SONIC PD interface; no additional deployment target conditioning during training'),
                deployment_actions=(
                    'same conditioning plus strict guard rejection' if active else
                    'portable guarded evaluation adds the deployment conditioning and strict rejection separately'),
                actor_inputs_added=False)
