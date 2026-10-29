"""Explicit recurrent state, checksum-checked; no torch or simulator dependency."""
import time
import numpy as np
from ..deploy.runtime import Runtime,session


class HikingRuntime(Runtime):
    def __init__(self,bundle,provider='cpu'):
        super().__init__(bundle,provider)
        metadata=self.manifest.get('hiking_perception',{})
        if metadata.get('mode')!='geometry': raise ValueError('RGB Hiking bundle required')
        self.memory=session(self.root/'memory.onnx',provider)
        self.hidden_dim=metadata['contract']['model']['hidden_dim']
        self.control_dt=self.contract['control_dt']
        self.reset()

    def reset(self):
        self.hidden=np.zeros((1,self.hidden_dim),np.float32); self.cell=self.hidden.copy()
        self.pending_reset=True

    def warmup(self,steps=5):
        try: return super().warmup(steps)
        finally: self.reset()

    def act(self,observations,elapsed=None):
        elapsed=self.control_dt if elapsed is None else float(elapsed)
        if not np.isfinite(elapsed) or not 0 < elapsed <= .1: raise ValueError('Invalid sensor/control interval')
        start=time.monotonic()
        feed=dict(tokens=observations['kv_tokens__img_tokens'],proprio=observations['q_proprio'],elapsed=np.array([elapsed],np.float32),
                  hidden=self.hidden,cell=self.cell,reset=np.array([self.pending_reset],bool))
        for port in self.memory.get_inputs():
            value=feed[port.name]
            if list(value.shape)!=port.shape or not np.isfinite(value).all(): raise ValueError(f'Invalid recurrent input {port.name}')
        hidden,cell=self.memory.run(None,feed)
        if not np.isfinite(hidden).all() or not np.isfinite(cell).all(): raise ValueError('Nonfinite recurrent output')
        action,target,_=super().act(dict(observations,hiking_latent=hidden))
        self.hidden,self.cell=hidden,cell; self.pending_reset=False
        return action,target,time.monotonic()-start
