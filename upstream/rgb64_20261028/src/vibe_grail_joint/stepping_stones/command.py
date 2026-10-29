"""Curriculum/reset logic is privileged training machinery, absent from actor inputs."""
from dataclasses import dataclass,fields
import torch
from ..velocity.dagger_task import DaggerMotionCommand,DaggerMotionCommandCfg
from .terrain import PrimitiveHeights,FAMILIES

class StoneCommand(DaggerMotionCommand):
    def _init_task(self):
        super()._init_task()
        self.primitives=PrimitiveHeights(self.cfg.random_tiles,self.device)
        self.family=torch.tensor([FAMILIES.index(t.get('family','stairs')) for t in self.cfg.random_tiles],device=self.device)
        self.course_level=torch.tensor([t.get('course_level',t['level']) for t in self.cfg.random_tiles],device=self.device)
        self.frontier=torch.tensor([10,0,0,0],device=self.device,dtype=torch.long)
        self.course_counts=torch.zeros(4,11,2,device=self.device)
        self.course_total=torch.zeros_like(self.course_counts)
        self.last_promoted=torch.zeros(4,device=self.device,dtype=torch.long)
        from .curriculum import initialize_recent
        initialize_recent(self, self.cfg.recent_curriculum)
        self.geometry_last_step=-1

    def ground_height(self,xy,env_ids=None):
        base=super().ground_height(xy,env_ids)
        if not hasattr(self,'primitives'):return base
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        tile=self.tile[ids];mask=self.family[tile]>0
        if mask.any():
            height,_=self.primitives.query(xy[mask],tile[mask],self.tile_origins)
            base[mask]=height
        return base

    def surface_ids(self,xy,env_ids):
        return self.primitives.query(xy,self.tile[env_ids],self.tile_origins)[1]

    def _sample_init_frame(self,env_ids):
        chosen,frames=super()._sample_init_frame(env_ids)
        ids=env_ids[self.ppo_online[env_ids]]
        if len(ids):
            family=torch.multinomial(torch.tensor(self.cfg.family_weights,device=self.device),len(ids),replacement=True)
            for f in range(4):
                selected=ids[family==f]
                if not len(selected):continue
                maximum=int(self.frontier[f])
                levels=torch.randint(maximum+1,(len(selected),),device=self.device)
                frontier=torch.rand(len(selected),device=self.device)<.5
                levels[frontier]=maximum
                for level in range(maximum+1):
                    selected_level=selected[levels==level]
                    options=((self.family==f)&(self.course_level==level)).nonzero().flatten()
                    if f==0:options=options[self.tile_data['kind'][options]!=3]
                    if len(selected_level):self.tile[selected_level]=options[torch.randint(len(options),(len(selected_level),),device=self.device)]
            # Conservative commands for new surfaces; no terrain input reaches planner.
            custom=self.family[self.tile[ids]]>0;new=ids[custom]
            if len(new):
                self.stream[new]=torch.randint(1,3,(len(new),),device=self.device) # .25/.4
                self.speed[new]=torch.tensor(self.cfg.planner_speeds,device=self.device)[self.stream[new]]
                local=self.ppo_online[env_ids].nonzero().flatten()[custom]
                chosen[local]=self.replay_clips+self.stream[new]
                frames[local]=self.motion.clip_offsets[chosen[local]]+self.pool.cursor-self.phase[new]
            mask=self.ppo_online[env_ids]
            spawn=self.tile_origins[self.tile[ids]].clone()
            spawn[:,0]+=torch.empty(len(ids),device=self.device).uniform_(-.2,.2)
            spawn[:,1]+=torch.empty(len(ids),device=self.device).uniform_(-.10,.10)
            spawn[:,2]=self.ground_height(spawn[:,:2],ids)
            spawn[:,:2]-=self.motion.body_pos_w[frames[mask],0,:2]
            self._env.scene.env_origins[ids]=spawn
        return chosen,frames

    def observe_course(self,family,level,done,success):
        ids=(done&self.ppo_online).nonzero().flatten()
        index=family[ids]*11+level[ids]
        self.course_counts[:,:,0].view(-1).index_add_(0,index,torch.ones(len(ids),device=self.device))
        self.course_counts[:,:,1].view(-1).index_add_(0,index,success[ids].float())

    def update_course(self,update):
        if update%50:return {}
        counts=self.course_counts.clone()
        if torch.distributed.is_initialized():torch.distributed.all_reduce(counts)
        self.course_counts.zero_();self.course_total+=counts
        if self.recent_curriculum is not None:
            from .curriculum import update_recent
            return update_recent(self, counts, update, FAMILIES)
        result={}
        for f,name in enumerate(FAMILIES):
            level=int(self.frontier[f]);episodes,success=self.course_total[f,level]
            rate=float(success/episodes.clamp_min(1))
            if f and level<self.cfg.stone_levels-1 and episodes>=80 and rate>=.72 and update-int(self.last_promoted[f])>=150:
                self.frontier[f]+=1;self.last_promoted[f]=update
            result[f'Course/{name}/frontier']=int(self.frontier[f])
            result[f'Course/{name}/frontier_success_rate']=rate
            result[f'Course/{name}/frontier_episodes']=float(episodes)
        return result

    def course_state(self):
        from .curriculum import RECENT_STATE
        keys=('frontier','course_total','last_promoted')
        if self.recent_curriculum is not None:keys+=RECENT_STATE
        return {k:getattr(self,k).clone() for k in keys}

@dataclass(kw_only=True)
class StoneCommandCfg(DaggerMotionCommandCfg):
    family_weights:tuple=(.35,.25,.25,.15)
    stone_levels:int=6
    recent_curriculum:dict|None=None
    def build(self,env):return StoneCommand(self,env)
