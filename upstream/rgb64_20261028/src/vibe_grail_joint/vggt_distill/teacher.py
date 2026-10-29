"""Frozen, current-frame VGGT-Omega patch targets. No cross-world attention."""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from .model import position_encoding
from .visual_contract import validate_patch_coverage


class GeometryTeacher:
    def __init__(self, settings, device):
        source = Path(settings['teacher_source']).resolve()
        sys.path.insert(0, str(source))
        from vggt_omega.models.aggregator import Aggregator
        # mmap + meta construction avoids eight simultaneous 4GB CPU copies and
        # expensive random initialization of parameters immediately overwritten.
        state = torch.load(settings['teacher_checkpoint'], map_location='cpu', weights_only=True, mmap=True)
        weights = {k.removeprefix('aggregator.'): v for k, v in state.items() if k.startswith('aggregator.')}
        with torch.device('meta'):
            model = Aggregator(cached_layer_indices=(11, 17, 23))
        model.load_state_dict(weights, strict=True, assign=True)
        self.model = model.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        del weights, state
        self.device = device
        self.shape = tuple(settings['teacher_shape'])
        self.grid = tuple(settings['patch_grid'])
        self.batch_size = settings['teacher_batch_size']
        self.calls = self.images = 0
        if any(n % 16 for n in self.shape):
            raise ValueError('Teacher image dimensions must be divisible by patch size 16')

    @torch.no_grad()
    def __call__(self, images):
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError('Geometry teacher requires RGB images shaped [B,H,W,3]')
        validate_patch_coverage(images.shape[1], images.shape[2], self.grid)
        results = []
        for start in range(0, len(images), self.batch_size):
            rgb = images[start:start + self.batch_size].to(self.device).permute(0, 3, 1, 2).float() / 255.
            rgb = F.interpolate(rgb, size=self.shape, mode='bilinear', align_corners=False, antialias=True)
            # B separate scenes, S=1 current frame. Never use [1,B,...] here.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                layers, patch_start = self.model(rgb[:, None])
            targets = []
            for level in (11, 17, 23):
                tokens = layers[level][:, 0, patch_start:]
                if tokens.shape[1] != self.shape[0] // 16 * (self.shape[1] // 16):
                    raise RuntimeError('Teacher patch grid mismatch')
                feature_map = tokens.transpose(1, 2).reshape(len(rgb), 2048, self.shape[0] // 16, self.shape[1] // 16)
                aligned = F.interpolate(feature_map.float(), self.grid, mode='bilinear', align_corners=False)
                aligned = aligned.flatten(2).transpose(1, 2)
                # Explicit position guidance, with normalized content to make the
                # relative position weight independent of teacher feature scale.
                aligned = F.layer_norm(aligned, (2048,))
                aligned = aligned + .1 * position_encoding(*self.grid, 2048, aligned.device)
                targets.append(F.normalize(aligned, dim=-1))
            result = torch.stack(targets, 1)
            if not torch.isfinite(result).all():
                raise RuntimeError('Nonfinite VGGT-Omega target')
            results.append(result.to(device='cpu', dtype=torch.float16))
            self.calls += 1; self.images += len(rgb)
        return torch.cat(results)

    def assert_frozen(self):
        if self.model.training or any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError('Geometry teacher must stay frozen')
