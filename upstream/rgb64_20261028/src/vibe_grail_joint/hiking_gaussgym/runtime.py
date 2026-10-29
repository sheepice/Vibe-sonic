"""Simulator-independent RGB/proprio memory loader; no physics or depth dependency."""
import torch
from .model import load_model

class RGBMemory:
    def __init__(self,state,batch,device,chunk=256):
        self.model=load_model(state,device).eval().requires_grad_(False)
        self.h,self.c=self.model.initial(batch,device);self.chunk=chunk;self.latent=None

    @torch.no_grad()
    def step(self,obs,reset):
        values=[]
        for start in range(0,len(self.h),self.chunk):
            sl=slice(start,start+self.chunk)
            z,h,c=self.model.step(obs['kv_tokens']['img_tokens'][sl],obs['q_proprio'][sl],self.h[sl],self.c[sl],reset[sl])
            self.h[sl].copy_(h);self.c[sl].copy_(c);values.append(z)
        self.latent=torch.cat(values)
        return self.latent.clone()
