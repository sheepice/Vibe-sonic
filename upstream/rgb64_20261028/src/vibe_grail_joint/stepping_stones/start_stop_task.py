"""Optional stationary-start episodes with causal deployment-reference windows.

Enabled only by the separate train_start_stop entrypoint. Terrain remains a
training sampler/reward variable; no terrain data is passed to the reference
generator or the actor. Existing moving-start worlds and DAgger are preserved.
"""
import copy
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch

from ..kinematics import CHANNELS
from .start_stop_references import CausalReferenceCache, CURRENT_OFFSET, FRAME_STRIDE
from .curriculum import validate_resume_options as base_validate_resume_options


def start_stop_mask(num_envs, device='cpu'):
    if num_envs < 80 or num_envs%80:
        raise ValueError('Matched-start training requires a multiple of 80 worlds per rank')
    slots = torch.arange(num_envs,device=device)
    return (slots%20 < 10) & ((slots//20)%4 == 0)


def validate_start_stop_options(value):
    expected = {'kind','cache_path','cache_manifest_sha256','online_fraction'}
    if (not isinstance(value,dict) or set(value)!=expected
            or value['kind']!='causal_start_stop_v1' or value['online_fraction']!=.25):
        raise ValueError('Require the explicit 25% causal start/stop training extension')
    from .deployment_suite import digest
    path = Path(value['cache_path']).resolve()
    if digest(path/'references.json')!=value['cache_manifest_sha256']:
        raise ValueError('Start/stop cache manifest changed')
    cache = CausalReferenceCache(path)
    if cache.manifest.get('smoke'):
        raise ValueError('Short smoke references cannot be used for training')
    return cache


def validate_start_stop_resume(saved, current, *, allow_revision=False, allow_start_stop=False):
    old = saved.get('stones',{}).get('start_stop') if saved else None
    new = current.get('stones',{}).get('start_stop')
    if old == new:
        return base_validate_resume_options(saved,current,allow_revision=allow_revision)
    if not allow_start_stop or old is not None or new is None or saved is None or allow_revision:
        raise ValueError('Explicit --revise-start-stop is required for the isolated start/stop migration')
    before,after=copy.deepcopy(saved),copy.deepcopy(current)
    after['stones'].pop('start_stop')
    if before!=after:
        raise ValueError('Start/stop migration cannot also change rewards, terrain, curriculum, actor or optimizers')
    return dict(start_stop_changed=True,old=None,new=new,restores_optimizer_and_curriculum=True,
                actor_inputs_changed=False,terrain_or_rewards_changed=False,
                episode_sampler_change='25% of online PPO worlds use stationary start and causal deployment references')


# mjlab task discovery must precede importing the existing command inheritance.
import mjlab  # noqa: E402,F401
from mjlab.managers import TerminationTermCfg  # noqa: E402
from .command import StoneCommand, StoneCommandCfg  # noqa: E402
from .task import env_config as base_env_config  # noqa: E402
from .task import QualityStairsEnv as BaseQualityStairsEnv  # noqa: E402


class StartStopStoneCommand(StoneCommand):
    def _build_loader(self):
        motion=super()._build_loader()
        self.start_stop_cache=validate_start_stop_options(self.cfg.start_stop)
        manifest=self.start_stop_cache.manifest
        if (list(self.cfg.body_names)!=manifest['body_names']
                or motion.metadata['joint_names']!=manifest['robot']['joint_names']):
            raise ValueError('Causal reference joint/body ordering mismatch')
        self.start_stop_clip_base=motion.n_clips
        self.start_stop_frame_base=motion.time_step_total
        self.start_stop_lengths=torch.tensor([c['control_steps'] for c in manifest['clips']],device=self.device)
        frame_lengths=self.start_stop_lengths*FRAME_STRIDE
        offsets=self.start_stop_frame_base+torch.cat([torch.zeros(1,device=self.device,dtype=torch.long),frame_lengths.cumsum(0)[:-1]])
        self.start_stop_offsets=offsets
        for name in CHANNELS:
            extra=torch.cat([torch.as_tensor(np.array(clip[name],copy=True),device=self.device)
                             for clip in self.start_stop_cache.clips])
            setattr(motion,name,torch.cat([getattr(motion,name),extra]))
        motion.clip_offsets=torch.cat([motion.clip_offsets,offsets])
        motion.clip_lengths=torch.cat([motion.clip_lengths,frame_lengths])
        motion.clip_ends=motion.clip_offsets+motion.clip_lengths
        motion.terrain_ids=torch.cat([motion.terrain_ids,torch.full_like(frame_lengths,-1)])
        motion.n_clips+=len(frame_lengths)
        motion.time_step_total+=int(frame_lengths.sum())
        motion.max_clip_length=max(motion.max_clip_length,int(frame_lengths.max()))
        self.start_stop_commands=torch.zeros(len(frame_lengths),int(self.start_stop_lengths.max()),device=self.device)
        for index,clip in enumerate(self.start_stop_cache.clips):
            self.start_stop_commands[index,:len(clip['command'])]=torch.as_tensor(np.array(clip['command'],copy=True),device=self.device)
        return motion

    def _upload_pool(self):
        # The original pool owns only its five 256-frame segments. Appended
        # causal windows must never be overwritten by its periodic upload.
        end=self.replay_frames+len(self.cfg.planner_speeds)*self.pool.capacity
        for name,array in self.pool.channels().items():
            getattr(self.motion,name)[self.replay_frames:end]=torch.as_tensor(array,device=self.device)

    def _init_task(self):
        super()._init_task()
        self.start_stop=start_stop_mask(self.num_envs,self.device)
        if (self.start_stop & ~self.ppo_online).any():
            raise ValueError('Start/stop worlds must belong only to online PPO')
        self.start_stop_ids=self.start_stop.nonzero().flatten()
        self.reference_clip=torch.zeros(self.num_envs,dtype=torch.long,device=self.device)
        self.reference_tick=torch.zeros_like(self.reference_clip)
        self.reference_route_seen=torch.zeros_like(self.start_stop)
        self.metrics['start_stop_world_fraction']=self.start_stop.float()

    def _sample_init_frame(self,env_ids):
        chosen,frames=super()._sample_init_frame(env_ids)
        selected=self.start_stop[env_ids]
        ids=env_ids[selected]
        if len(ids):
            clip=torch.randint(len(self.start_stop_lengths),(len(ids),),device=self.device)
            self.reference_clip[ids]=clip
            self.reference_tick[ids]=0
            self.reference_route_seen[ids]=False
            chosen[selected]=self.start_stop_clip_base+clip
            frames[selected]=self.start_stop_offsets[clip]+CURRENT_OFFSET
            self.speed[ids]=self.start_stop_commands[clip,0]
            spawn=self.tile_origins[self.tile[ids]].clone()
            spawn[:,0]+=torch.empty(len(ids),device=self.device).uniform_(-.2,.2)
            spawn[:,1]+=torch.empty(len(ids),device=self.device).uniform_(-.10,.10)
            spawn[:,2]=self.ground_height(spawn[:,:2],ids)
            spawn[:,:2]-=self.motion.body_pos_w[frames[selected],0,:2]
            self._env.scene.env_origins[ids]=spawn
        return chosen,frames

    def _reset_task(self,env_ids,clip_ids,time_steps,origins):
        super()._reset_task(env_ids,clip_ids,time_steps,origins)
        ids=env_ids[self.start_stop[env_ids]]
        if len(ids):
            t=self.time_steps[ids]
            pos=self.motion.body_pos_w[t,0]+self._env.scene.env_origins[ids]
            self._write_reference_state_to_sim(ids,pos,self.motion.body_quat_w[t,0],
                torch.zeros(len(ids),3,device=self.device),torch.zeros(len(ids),3,device=self.device),
                self.motion.joint_pos[t],torch.zeros(len(ids),29,device=self.device))

    def _update_command(self,env_ids=None):
        all_online=self.online_ids
        # Exclude only the causal-cache worlds from shared stream recentering
        # and phase advancement. They remain ONLINE in all actor task flags.
        self.online_ids=all_online[~self.start_stop[all_online]]
        try:
            super()._update_command(env_ids)
        finally:
            self.online_ids=all_online
        ids=self.start_stop_ids if env_ids is None else env_ids[self.start_stop[env_ids]]
        clip=self.reference_clip[ids]
        # episode_length_buf is zero after auto-reset, avoiding a one-tick skip
        # on the reset step; no reference can wrap to the next cached clip.
        tick=self._env.episode_length_buf[ids].long()
        if (tick>=self.start_stop_lengths[clip]).any():
            raise RuntimeError('Causal start/stop reference exhausted before episode termination')
        self.reference_tick[ids]=tick
        self.time_steps[ids]=self.start_stop_offsets[clip]+tick*FRAME_STRIDE+CURRENT_OFFSET
        self.speed[ids]=self.start_stop_commands[clip,tick]
        self.update_relative_body_poses()


@dataclass(kw_only=True)
class StartStopStoneCommandCfg(StoneCommandCfg):
    start_stop:dict|None=None
    def build(self,env):return StartStopStoneCommand(self,env)


def reference_end(env):
    c=env.command_manager.get_term('motion')
    return c.start_stop & (env.episode_length_buf>=c.start_stop_lengths[c.reference_clip])


def course_and_stop_success(env):
    from ..velocity.recovery_terrain import profile_success
    c=env.command_manager.get_term('motion')
    moving_success=profile_success(env)
    c.reference_route_seen|=moving_success & c.start_stop
    root=c.robot.data.root_link_pos_w
    feet=c.robot_body_pos_w[:,[3,6]]
    origin=c.tile_origins[c.tile]
    goal=c.tile_data['goal'][c.tile]
    stable=(c.robot.data.root_link_lin_vel_w[:,:2].norm(dim=-1)<.12) & (c.speed.abs()<1e-6)
    stable &= (root[:,0]-origin[:,0]>=goal)
    stable &= (feet[:,:,0]-origin[:,None,0]).min(-1).values>=goal-.45
    stable &= (feet[:,:,2]-c.ground_height(feet[:,:,:2])).abs().max(-1).values<.18
    stable &= (c.robot.data.projected_gravity_b[:,2]<-.85)
    stable &= (root[:,2]-c.ground_height(root[:,:2])>.45)
    c.terminal_success=torch.where(c.start_stop,reference_end(env)&c.reference_route_seen&stable,moving_success)
    return c.terminal_success


class StartStopStairsEnv(BaseQualityStairsEnv):
    def step(self,actions):
        c=self.command_manager.get_term('motion')
        ids=c.start_stop_ids
        lengths=self.episode_length_buf[ids].clone()+1
        command_error=(c.speed[ids]-c.start_stop_commands[c.reference_clip[ids],c.reference_tick[ids]]).abs().max()
        if command_error>1e-6:
            raise RuntimeError('Start/stop actor velocity command differs from the deployment schedule')
        expected=c.start_stop_offsets[c.reference_clip[ids]]+c.reference_tick[ids]*FRAME_STRIDE+CURRENT_OFFSET
        if not torch.equal(c.time_steps[ids],expected):
            raise RuntimeError('Start/stop reference cursor lost its causal window')
        result=super().step(actions)
        done=self.reset_buf[ids].bool()
        logs=result[-1].setdefault('log',{})
        logs.update({'Stones/StartStop/world_fraction':c.start_stop.float().mean(),
                     'Stones/StartStop/online_fraction':c.start_stop.sum()/c.ppo_online.sum(),
                     'Stones/StartStop/command_error_max':command_error,
                     'Stones/StartStop/ended_episodes':done.sum().float(),
                     'Stones/StartStop/successful_episodes':(done&c.terminal_success[ids]).sum().float()})
        if done.any():
            logs['Stones/StartStop/ended_episode_seconds']=lengths[done].float().mean()*self.step_dt
        return result


def env_config(bank_path,settings,num_envs):
    start_stop_mask(num_envs)
    if not settings['stones'].get('deployment_targets',{}).get('enabled'):
        raise ValueError('Matched start/stop experiment requires existing deployment target training')
    cfg=base_env_config(bank_path,settings,num_envs)
    old=cfg.commands['motion']
    params={f.name:getattr(old,f.name) for f in fields(StoneCommandCfg)}
    cfg.commands['motion']=StartStopStoneCommandCfg(**params,start_stop=copy.deepcopy(settings['stones']['start_stop']))
    cfg.terminations['online_success'].func=course_and_stop_success
    cfg.terminations['start_stop_reference_end']=TerminationTermCfg(func=reference_end,time_out=True)
    return cfg
