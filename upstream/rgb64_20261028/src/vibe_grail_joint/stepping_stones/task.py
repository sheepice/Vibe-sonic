"""New task only. Stair rewards retained; sparse-support rewards depend on contacts."""
import copy,hashlib,json
from dataclasses import fields
import torch
from mjlab.managers import RewardTermCfg,TerminationTermCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply
from ..hiking_gaussgym.task import env_config as stairs_config,options as stairs_options,REMOVED_REWARDS
from ..velocity.config import agent_config
from ..velocity.quality_runtime import QualityGuidance
from ..velocity.posture_task import posture_online_fall
from ..velocity import mdp
from .terrain import make_tiles,atlas_class,FAMILIES,key
from .command import StoneCommandCfg

def options(settings):
    adapted=copy.deepcopy(settings);adapted['expected_world_size']=1
    return dict(stairs=stairs_options(adapted),stones=copy.deepcopy(settings['stones']),seed=settings['seed'])

def env_config(bank_path,settings,num_envs):
    adapted=copy.deepcopy(settings);adapted['expected_world_size']=1
    cfg=stairs_config(bank_path,adapted,num_envs)
    from ..prepare import validate_bank
    from pathlib import Path
    rank=validate_bank(Path(bank_path))['rank']
    tiles=tuple(make_tiles(settings['seed']+7919*rank,settings['stones']['levels'],settings['profile_variants']))
    if settings['stones'].get('cylinder_collision')=='convex64':
        for tile in tiles:
            for solid in tile.get('solids',[]):
                if solid['shape']=='cylinder':solid['collision_sides']=64
    old=cfg.commands['motion']
    params={f.name:getattr(old,f.name) for f in fields(StoneCommandCfg) if hasattr(old,f.name)}
    params.update(random_tiles=tiles,stone_levels=settings['stones']['levels'],family_weights=tuple(settings['stones']['family_weights']))
    params['recent_curriculum']=copy.deepcopy(settings['stones'].get('recent_curriculum'))
    cfg.commands['motion']=StoneCommandCfg(**params)
    old=cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']=atlas_class()(root=old.root,rows=old.rows,random_tiles=tiles)
    cfg.rewards['command_velocity'].func=velocity_tracking
    cfg.rewards['stone_low_support']=RewardTermCfg(func=guidance_reward,weight=-.35,params={'name':'low_support'})
    cfg.rewards['stone_safe_support']=RewardTermCfg(func=guidance_reward,weight=.25,params={'name':'safe_support'})
    cfg.rewards['stone_floor_contact']=RewardTermCfg(func=floor_contact,weight=-12.)
    cfg.terminations['stone_floor_contact']=TerminationTermCfg(func=floor_contact)
    cfg.episode_length_s=45.
    return cfg

def guidance_reward(env,name):return env.guidance.reward(name)

def floor_contact(env):
    c=env.command_manager.get_term('motion');ids=torch.arange(env.num_envs,device=env.device)
    family=c.family[c.tile];sparse=(family>0)&c.ppo_online
    feet=c.robot_body_pos_w[:,[3,6]]
    local=feet-c.tile_origins[c.tile,None]
    ground=c.ground_height(feet[:,:,:2])
    support=[]
    for side in ('left','right'):
        d=env.scene[f'foothold_{side}'].data
        support.append(((d.found>0)&(d.normal[...,2].abs()>.7)&(d.force[...,2].abs()>15.)).any(-1))
    support=torch.stack(support,-1)
    # For stair-gap courses, ordinary approach/final floor remains valid.
    bad=torch.zeros_like(support)
    pillars=(family==1)|(family==2)
    bad[pillars]=(ground[pillars]<.10)&(local[pillars,:,2]<.10)&support[pillars]
    gap_ids=(family==3).nonzero().flatten()
    if len(gap_ids):
        bounds=c.gap_bounds[c.tile[gap_ids]]
        x=local[gap_ids,:,0]
        ingap=(x>=bounds[:,0,None])&(x<=bounds[:,1,None])
        bad[gap_ids]=ingap&(ground[gap_ids]<.10)&(local[gap_ids,:,2]<.10)&support[gap_ids]
    result=bad.any(-1)&sparse
    c.terminal_fall=getattr(c,'terminal_fall',result.clone())|result
    return result

def velocity_tracking(env):
    c=env.command_manager.get_term('motion')
    original=mdp.velocity_tracking(env)
    sparse=(c.family[c.tile]>0)&c.ppo_online
    if hasattr(env,'guidance'):
        # A slower moving average allows settling both feet on one stone.
        v=env.guidance.average_velocity
        error=(v[:,0]-c.speed).square()+v[:,1].square()
        softened=torch.exp(-error/.30**2)*(-c.robot.data.projected_gravity_b[:,2]).clamp(0,1)
        original=torch.where(sparse,.65*softened,original)
    return original

class StoneGuidance(QualityGuidance):
    def __init__(self,env,settings):
        super().__init__(env,settings)
        self.edge_hash=hashlib.sha256(json.dumps([key(t) for t in self.c.cfg.random_tiles]).encode()).hexdigest()
        self.average_velocity=torch.zeros(env.num_envs,2,device=env.device)
        self.reward_ids=self.ids
        self.rewards.update({k:torch.zeros(env.num_envs,device=env.device) for k in ('low_support','safe_support')})
        self.sparse_last_step=-1
        self.sole_samples=torch.cartesian_prod(torch.linspace(-.05,.13,9,device=env.device),torch.linspace(-.026,.026,3,device=env.device))
        self.sole_samples=torch.cat((self.sole_samples,torch.full((len(self.sole_samples),1),-.035,device=env.device)),-1)
    def before_step(self,actions):
        self.average_velocity.lerp_(self.c.robot.data.root_link_lin_vel_w[:,:2],.04)
        super().before_step(actions)
    @torch.no_grad()
    def measure(self):
        step=int(self.env.common_step_counter)
        if step==self.sparse_last_step:return
        try:super().measure()
        except RuntimeError:
            bad=torch.zeros(self.env.num_envs,dtype=torch.bool,device=self.env.device)
            for value in self.rewards.values():bad|=~torch.isfinite(value)
            ids=bad.nonzero().flatten()
            if len(ids):
                from pathlib import Path
                import os
                robot=self.c.robot
                state=dict(step=step,ids=ids.cpu(),family=self.c.family[self.c.tile[ids]].cpu(),tile=self.c.tile[ids].cpu(),
                    rewards={k:v[ids].cpu() for k,v in self.rewards.items()},
                    root=robot.data.root_link_pos_w[ids].cpu(),qpos=robot.data.joint_pos[ids].cpu(),qvel=robot.data.joint_vel[ids].cpu(),
                    body_pos=self.c.robot_body_pos_w[ids].cpu(),body_vel=self.c.robot_body_lin_vel_w[ids].cpu(),
                    body_quat=self.c.robot_body_quat_w[ids].cpu(),body_angvel=self.c.robot_body_ang_vel_w[ids].cpu(),
                    actions=self.env.action_manager.action[ids].cpu(),reset_buf=self.env.reset_buf[ids].cpu())
                folder=Path(getattr(self.env,'diagnostic_dir','records/stones_20261025'))
                torch.save(state,folder/f'nonfinite_rank_{os.environ.get("RANK","0")}_step_{step}.pt')
            raise
        self.sparse_last_step=step
        c=self.c;ids=self.ids;family=c.family[c.tile[ids]];sparse=family>0
        if not sparse.any():return
        # No center-of-stone target, no left/right sequence target, no compulsory
        # two-foot stance. Partial supported soles are allowed down to 35%.
        pos=c.robot_body_pos_w[ids][:,self.feet];quat=c.robot_body_quat_w[ids][:,self.feet]
        q=quat[:,:,None].expand(-1,-1,len(self.sole_samples),-1)
        points=pos[:,:,None]+quat_apply(q.reshape(-1,4),self.sole_samples.expand(*q.shape[:-1],3).reshape(-1,3)).reshape(*q.shape[:-1],3)
        height=c.ground_height(points[...,:2],ids)
        fraction=((points[...,2]-height).abs()<.045).float().mean(-1)
        support=self.quality.sensor_support
        deficit=((.35-fraction).clamp_min(0)/.35).square()*support
        self.rewards['low_support'][ids]=deficit.mean(-1)*sparse
        safe=((fraction>=.35)&support).any(-1).float()
        self.rewards['safe_support'][ids]=safe*sparse
        # Remove stair-specific edge/margin/swing-riser shaping on sparse courses.
        for name in ('main','support_margin','swing_path'):
            self.rewards[name][ids[sparse]]=0.
        # Foot contact yaw is not forced into a narrow direction on small tops.
        self.rewards['yaw'][ids[sparse]]*=.25
        both= support.all(-1)
        sid=c.surface_ids(pos[:,:,:2],ids)
        same=both&(sid[:,0]==sid[:,1])&(sid[:,0]>=0)
        for f,name in enumerate(FAMILIES[1:],1):
            mask=family==f;den=mask.sum().clamp_min(1)
            self.metrics[f'Stones/{name}/safe_support_fraction']=(safe*mask).sum()/den
            self.metrics[f'Stones/{name}/both_feet_same_top_fraction']=(same*mask).sum()/den
            self.metrics[f'Stones/{name}/low_support_cost']=(deficit.mean(-1)*mask).sum()/den
        self.env.extras.setdefault('log',{}).update(self.metrics)
    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'average_velocity'):
            if env_ids is None:self.average_velocity.zero_()
            else:self.average_velocity[env_ids]=0.

class QualityStairsEnv(ManagerBasedRlEnv):
    def __init__(self,cfg,*,device,settings):
        super().__init__(cfg,device=device)
        c=self.command_manager.get_term('motion')
        c.gap_bounds=torch.tensor([t['gaps'][0] if t.get('family')=='stair_gap' else [-1e5,-1e5] for t in c.cfg.random_tiles],device=device)
        self.guidance=StoneGuidance(self,settings)
        previous=self.metrics_manager.compute_substep
        def substep():previous();self.guidance.quality.observe_substep()
        self.metrics_manager.compute_substep=substep
    def step(self,actions):
        self.guidance.before_step(actions)
        c=self.command_manager.get_term('motion');f=c.family[c.tile].clone();level=c.course_level[c.tile].clone()
        result=super().step(actions)
        done=self.reset_buf.bool()
        c.observe_course(f,level,done,c.terminal_success)
        return result
    def _reset_idx(self,env_ids=None):
        if hasattr(self,'guidance'):self.guidance.reset(env_ids)
        return super()._reset_idx(env_ids)
