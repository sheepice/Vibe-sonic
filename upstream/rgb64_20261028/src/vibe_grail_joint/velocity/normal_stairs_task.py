"""Matched four-GPU SSR / Hiking experiments, preserving original trainers."""
import copy
import math
from dataclasses import replace
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg

from ..prepare import validate_bank
from .config import env_config as original_env_config, agent_config
from .normal_stairs_geometry import normal_tiles
from .normal_stairs_runtime import FootholdGuidance
from .ssr_task import REMOVED_REWARDS, options as ssr_options


def options(settings):
    experiment=settings['foothold_experiment']
    if experiment['version']!=1 or experiment['method'] not in ('ssr_normal','hiking_edges'):
        raise ValueError('Unknown foothold comparison version or method')
    if settings['expected_world_size']!=4 or settings['dagger'] is None:
        raise ValueError('Both experiments require four ranks and flat DAgger preservation')
    normal=settings['normal_stairs']
    if not .25<=normal['minimum_tread_m']<=normal['maximum_tread_m']<=.5:
        raise ValueError('Invalid normal tread depth')
    pose=settings['foot_pose']
    for key in ('yaw_tolerance_deg','yaw_scale_deg','tilt_tolerance_deg','tilt_scale_deg',
                'tilt_velocity_scale_m_s','minimum_contact_steps','warmup_updates'):
        if not math.isfinite(pose[key]) or pose[key]<=0:raise ValueError(f'Invalid foot pose option {key}')
    if pose['yaw_weight']>=0 or pose['tilt_weight']>=0:raise ValueError('Pose costs need negative weights')
    if experiment['method']=='ssr_normal':
        ssr_options(settings)
        a=settings['normal_ssr']
        if not 0<=a['allowed_support_deficiency']<=.2 or not 0<=a['planning_margin_m']<=.03:
            raise ValueError('Invalid SSR normal stair adaptation')
    else:
        h=settings['hiking']
        if not 0<h['sharpness_deg']<90 or not 0<h['edge_radius_m']<.06 or h['weight']>=0:
            raise ValueError('Invalid Hiking edge parameters')
        if h['speed_epsilon']!=.001 or h['geometry_batch_envs']<1:
            raise ValueError('Invalid Hiking speed epsilon or geometry batching')
    return copy.deepcopy({key:settings[key] for key in
                          ('foothold_experiment','normal_stairs','foot_pose','normal_ssr','hiking','ssr_foothold',
                           'seed','profile_levels','profile_variants')
                          if key in settings})


def guidance_reward(env,name):
    return env.guidance.reward(name)


def env_config(bank_path,settings,num_envs):
    options(settings)
    adapted=copy.deepcopy(settings)
    adapted['foothold']=dict(clearance_weight=0.,collision_weight=0.,support_weight=0.)
    cfg=original_env_config(bank_path,adapted,num_envs)
    meta=validate_bank(Path(bank_path))
    if not meta['smoke_only'] and meta['world_size']!=4:
        raise ValueError('Use the complete merged four-rank bank')
    terrain=settings['normal_stairs']
    tiles=tuple(normal_tiles(settings['seed']+meta['rank']*7919,settings['profile_levels'],
                             settings['profile_variants'],terrain['minimum_tread_m'],terrain['maximum_tread_m']))
    cfg.commands['motion'].random_tiles=tiles
    old=cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']=replace(old,random_tiles=tiles)
    for name in REMOVED_REWARDS:cfg.rewards.pop(name,None)
    method=settings['foothold_experiment']['method']
    main_name='online_ssr_foothold' if method=='ssr_normal' else 'online_hiking_edge_volume'
    weight=settings['ssr_foothold']['weight'] if method=='ssr_normal' else settings['hiking']['weight']
    cfg.rewards[main_name]=RewardTermCfg(func=guidance_reward,weight=weight,params={'name':'main'})
    for name in ('yaw','tilt'):
        cfg.rewards[f'online_foot_{name}']=RewardTermCfg(func=guidance_reward,
                weight=settings['foot_pose'][f'{name}_weight'],params={'name':name})
    return cfg


class NormalStairsEnv(ManagerBasedRlEnv):
    def __init__(self,cfg,*,device,settings):
        super().__init__(cfg,device=device)
        self.guidance=FootholdGuidance(self,settings)

    def step(self,action):
        self.guidance.before_step(action)
        return super().step(action)

    def _reset_idx(self,env_ids=None):
        if hasattr(self,'guidance'):self.guidance.reset(env_ids)
        return super()._reset_idx(env_ids)
