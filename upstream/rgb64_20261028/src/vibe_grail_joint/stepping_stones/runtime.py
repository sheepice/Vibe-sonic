"""No simulator imports: explicit-state policy runtime for the RGB stone variants."""

def load_torch_control(checkpoint,obs,settings,device):
    import copy,torch
    from .policy import StoneActor
    from ..hiking_gaussgym.runtime import RGBMemory
    from ..velocity.config import agent_config
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    trained=state.get('infos',{}).get('stepping_stones',{}).get('settings',{}).get('deployment_targets',{}).get('enabled',False)
    if trained and not settings.get('stones',{}).get('deployment_targets',{}).get('enabled',False):
        raise ValueError('This checkpoint requires deployment target conditioning in the evaluation config')
    memory=RGBMemory(state['infos']['gaussgym'],len(obs),device)
    cfg=agent_config(settings['bank_root']+'/rank_0',settings,20)
    acfg=copy.deepcopy(cfg['actor']);acfg.pop('class_name')
    variant=state.get('infos',{}).get('stepping_stones',{}).get('settings',{}).get('variant','dual')
    obs=obs.clone(False);obs['hiking_latent']=torch.zeros(len(obs),672,device=device)
    actor=StoneActor(obs=obs,obs_groups=cfg['obs_groups'],obs_set='actor',output_dim=29,
                    geometry_dim=672,stone_variant=variant,**acfg).to(device).eval()
    actor.load_state_dict(state['actor_state_dict'],strict=True)
    return actor,memory

# ONNX runtime has no torch/mujoco/mjlab/terrain dependency. This helper's torch
# loader above is for simulation; deployed bundles carry all shapes/weights.
import time
import numpy as np
from ..deploy.runtime import Runtime,session

class StoneRuntime(Runtime):
    def __init__(self,bundle,provider='cpu'):
        super().__init__(bundle,provider)
        if self.manifest.get('required_runtime')!='vibe_grail_joint.stepping_stones.runtime.StoneRuntime':raise ValueError('Wrong RGB memory bundle')
        self.memory=session(self.root/'memory.onnx',provider)
        from .input_audit import validate_ports
        validate_ports(self)
        self.reset()
    def reset(self):
        self.hidden=np.zeros((1,512),np.float32);self.cell=self.hidden.copy();self.pending_reset=True
    def warmup(self,steps=5):
        start=time.monotonic()
        feed={k:np.zeros(shape,np.float32) for k,shape in self.inputs.items()}
        feed['q_proprio']=np.zeros((1,64),np.float32)
        try:
            for _ in range(steps):
                tok,cls=self.encode(np.zeros((63,112,3),np.uint8))
                feed['kv_tokens__img_tokens']=tok[None];feed['q_cls']=cls[None]
                self.act(feed)
            return time.monotonic()-start
        finally:self.reset()
    def act(self,observations):
        start=time.monotonic()
        feed=dict(tokens=observations['kv_tokens__img_tokens'],proprio=observations['q_proprio'],
                  hidden=self.hidden,cell=self.cell,reset=np.array([self.pending_reset],bool))
        for port in self.memory.get_inputs():
            value=feed[port.name]
            if list(value.shape)!=port.shape or not np.isfinite(value).all():raise ValueError(port.name)
        latent,hidden,cell=self.memory.run(None,feed)
        if not all(np.isfinite(x).all() for x in (latent,hidden,cell)):raise ValueError('Nonfinite recurrent state')
        action,target,_=super().act(dict(observations,hiking_latent=latent))
        self.hidden,self.cell=hidden,cell;self.pending_reset=False
        return action,target,time.monotonic()-start
