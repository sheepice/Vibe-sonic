"""Encode physically black RGB pixels with the SAME pinned Theia preprocessing."""
import torch
from ..vision import FrozenTheiaFeature

class BlackRGB:
    def __init__(self,raw):
        term=raw.cfg.observations['kv_tokens'].terms['img_tokens']
        self.encoder=FrozenTheiaFeature(term,raw)
        rgb=raw.scene['head_cam'].data.rgb
        with torch.inference_mode():
            # Exactly the original RGB -> [0,1] -> ImageNet normalization path.
            self.pixels=torch.zeros_like(rgb)
            x=self.pixels.to(self.encoder._dev).permute(0,3,1,2).to(self.encoder._dtype)/255.
            x=(x-self.encoder._mean)/self.encoder._std
            dense,glob=self.encoder._encode(x)
            self.tokens=dense.detach().float().to(raw.device)
            self.cls=glob.detach().float().to(raw.device)
        if not torch.isfinite(self.tokens).all() or not torch.isfinite(self.cls).all():raise RuntimeError('Nonfinite black RGB encoding')
        if not self.tokens.abs().any():raise RuntimeError('Black RGB is not zero-feature ablation')
        self.calls=0
        self.audit=dict(mask='all RGB pixels are 0 BEFORE normalization and frozen Theia',pixel_max=int(self.pixels.max()),
                        tokens_abs_mean=float(self.tokens.abs().mean()),cls_abs_mean=float(self.cls.abs().mean()),
                        token_shape=list(self.tokens.shape),cls_shape=list(self.cls.shape),
                        replaced=['kv_tokens.img_tokens','q_cls'],geometry_memory='reset at start; black RGB features from first action',
                        retained=['proprioception','velocity command','online planner reference'],depth_input=False)

    def __call__(self,obs):
        out=obs.clone()
        out['kv_tokens']['img_tokens']=self.tokens.clone()
        out['q_cls']=self.cls.clone()
        self.calls+=1
        return out
