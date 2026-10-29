"""Online supervised perception between PPO updates; no privileged actor inputs."""
from collections import deque
import torch
from ..hiking_gaussgym.model import load_model,ARCHITECTURE
from .labels import PhysicsLabels
from .learning import supervised_update
from ..hiking_perception.contracts import scene_split


class PerceptionWrapper:
    def __init__(self,env,*,mode,checkpoint=None,chunk=256,settings=None):
        if mode!='geometry':raise ValueError('This experiment permits RGB geometry only, no oracle policy')
        self.env=env;self.settings=settings or {};self.chunk=chunk
        state=torch.load(checkpoint,map_location=env.device,weights_only=False)
        state=state.get('infos',{}).get('gaussgym',state)
        self.model=load_model(state,env.device).eval()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.settings.get('learning_rate',5e-5))
        if 'online_optimizer' in state:self.optimizer.load_state_dict(state['online_optimizer'])
        # Preserve Adam moments on fine-tuning while honoring the new task LR.
        for group in self.optimizer.param_groups:group['lr']=self.settings.get('learning_rate',5e-5)
        self.updates=state.get('online_updates',0)
        self.dimension=self.model.latent_dim
        self.metadata=dict(mode='geometry',architecture=ARCHITECTURE,contract=self.model.contract(),
                           frozen=False,depth_input=False,update='supervised optimizer between PPO rollouts; no PPO gradient into perception')
        self.h,self.c=self.model.initial(env.num_envs,env.device)
        self.pre_h=self.h.clone();self.pre_c=self.c.clone()
        self._reset=torch.ones(env.num_envs,dtype=torch.bool,device=env.device)
        self.last_reset=self._reset.clone();self._cached=None
        self.labels=PhysicsLabels(env.unwrapped,self.model.geometry)
        command=env.unwrapped.command_manager.get_term('motion')
        available=command.ppo_online.nonzero().flatten()
        # Fixed physical worlds, independently resampled terrains at episode resets.
        order=torch.randperm(len(available),device=env.device)
        self.ids=available[order[:min(len(available),self.settings.get('sample_worlds',64))]]
        self.allowed_tiles=torch.ones(len(self.labels.keys),device=env.device,dtype=torch.bool) # separate held-out seeds in evaluation
        self.rows=deque(maxlen=self.settings.get('sequence',64))
        self.update_interval=self.settings.get('update_interval',10)
        if self.update_interval<1:raise ValueError('Positive update interval required')

    def __getattr__(self,name):return getattr(self.env,name)

    @torch.no_grad()
    def _encode(self,obs,h,c,reset):
        values=[]
        for start in range(0,self.num_envs,self.chunk):
            sl=slice(start,start+self.chunk)
            z,nh,nc=self.model.step(obs['kv_tokens']['img_tokens'][sl],obs['q_proprio'][sl],h[sl],c[sl],reset[sl])
            self.h[sl].copy_(nh);self.c[sl].copy_(nc);values.append(z)
        return torch.cat(values).clone()

    @torch.no_grad()
    def _augment(self,obs):
        obs=obs.clone(False)
        self.pre_h.copy_(self.h);self.pre_c.copy_(self.c);self.last_reset.copy_(self._reset)
        obs['hiking_latent']=self._encode(obs,self.pre_h,self.pre_c,self._reset)
        target=self.labels.sample(self.ids)
        valid=self.allowed_tiles[self.labels.c.tile[self.ids]]
        # Inference-mode rollout tensors must be copied to ordinary storage before backward.
        with torch.inference_mode(False):
            row={k:target[k].detach().clone() for k in ('height','surface_valid','base_lin_vel_b','base_ang_vel_b')}
            row.update(tokens=obs['kv_tokens']['img_tokens'][self.ids].detach().clone().half(),
                       proprio=obs['q_proprio'][self.ids].detach().clone(),reset=self._reset[self.ids].detach().clone(),
                       valid=valid.detach().clone(),h=self.pre_h[self.ids].detach().clone(),c=self.pre_c[self.ids].detach().clone())
            self.rows.append(row)
        self._reset.zero_();self._cached=obs
        return obs

    def get_observations(self):
        return self._cached if self._cached is not None else self._augment(self.env.get_observations())

    def reset(self):
        obs,extras=self.env.reset();self._reset.fill_(True);self.rows.clear()
        return self._augment(obs),extras

    def step(self,actions):
        obs,reward,done,extras=self.env.step(actions);self._reset.copy_(done.bool())
        return self._augment(obs),reward,done,extras

    def update_perception(self,policy_update,dagger_env):
        if (policy_update+1)%self.update_interval or len(self.rows)<self.rows.maxlen:return {}
        stacked={k:torch.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        metrics={};steps=self.settings.get('updates_per_interval',10)
        for _ in range(steps):
            eligible=stacked['valid'].any(0).nonzero().flatten()
            if not len(eligible):raise RuntimeError('No eligible geometry worlds; all ranks must execute identical collectives')
            ids=eligible[torch.randint(len(eligible),(self.settings.get('batch_size',4),),device=self.device)]
            batch={k:v[:,ids].float() if k in ('tokens','proprio') else v[:,ids]
                   for k,v in stacked.items() if k not in ('h','c')}
            initial=(stacked['h'][0,ids],stacked['c'][0,ids])
            result=supervised_update(self.model,batch,self.optimizer,self.settings.get('burn_in',16),
                                     self.settings.get('recon_stride',8),self.settings.get('microbatch',2),initial,
                                     occupancy_objective=self.settings.get('occupancy_objective'))
            self.updates+=1
            for k,v in result.items():metrics[k]=metrics.get(k,0.)+v/steps
        if not metrics:return {}
        # PPO storage has already been consumed. Recompute the current observation
        # once from its PRE-step state; do not accidentally advance time twice.
        with torch.inference_mode():
            latent=self._encode(self._cached,self.pre_h,self.pre_c,self.last_reset)
            self._cached['hiking_latent'].copy_(latent)
            dagger_env.refresh_latent(latent)
        # Cached replay contains old recurrent latents, so discard it on encoder
        # version changes. Teacher/base weights remain unchanged; recollect next rollout.
        discarded=len(dagger_env.buffer)
        dagger_env.buffer.fresh.clear();dagger_env.buffer.history.clear()
        metrics.update(online_updates=self.updates,replay_samples_discarded=discarded)
        result={'GaussGym/'+k:v for k,v in metrics.items()}
        result['GaussGym/distributed_parameter_max_error']=self.check_sync()
        result['GaussGym/learning_rate']=self.optimizer.param_groups[0]['lr']
        return result

    @torch.no_grad()
    def synchronize(self):
        if torch.distributed.is_initialized():
            for value in self.model.state_dict().values():torch.distributed.broadcast(value,0)

    @torch.no_grad()
    def check_sync(self):
        if not torch.distributed.is_initialized():return 0.
        # Full tensor equality audit, not a sampled checksum.
        error=torch.zeros((),device=self.device)
        for p in self.model.parameters():
            other=p.detach().clone();torch.distributed.broadcast(other,0)
            error=torch.maximum(error,(other-p).abs().max())
        torch.distributed.all_reduce(error,op=torch.distributed.ReduceOp.MAX)
        if error>1e-6:raise RuntimeError(f'Geometry ranks diverged: {float(error)}')
        return float(error)

    def snapshot(self):
        return dict(architecture=ARCHITECTURE,contract=self.model.contract(),encoder=self.model.state_dict(),
                    online_optimizer=self.optimizer.state_dict(),online_updates=self.updates,settings=self.settings,
                    depth_input=False,supervision='finite height classes including saturation; no privileged actor input')


from ..velocity.dagger_env import DaggerVecEnv as BaseDaggerEnv

class OnlineDaggerVecEnv(BaseDaggerEnv):
    def get_observations(self):
        value=super().get_observations();self.last_ppo_obs=value;return value
    def reset(self):
        value,extra=super().reset();self.last_ppo_obs=value;return value,extra
    def step(self,actions):
        value,reward,done,extra=super().step(actions);self.last_ppo_obs=value
        return value,reward,done,extra
    def refresh_latent(self,latent):
        self._full_obs['hiking_latent'].copy_(latent)
        self.last_ppo_obs['hiking_latent'].copy_(latent[self.ppo_ids])
