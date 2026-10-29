"""Full-resolution learned terrain heads; frozen-SONIC interfaces are untouched."""
import torch
from torch import nn
from torch.nn import functional as F
from ..hiking_perception.models import GeometryMemory
from ..hiking_perception.geometry import height_bins, decode_height, masked_mean


class SpatialGeometry(GeometryMemory):
    architecture = 'hiking_spatial_v2'

    def __init__(self, geometry, spec):
        if (geometry.nx, geometry.ny) != (64, 32):
            raise ValueError('V2 requires measured 64x32 labels, not interpolated old targets')
        super().__init__(geometry, spec)
        del self.seed, self.decoder, self.edge
        self.spatial_seed = nn.Linear(spec.hidden_dim, 64*8*4)
        self.spatial_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ELU(),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.GroupNorm(8, 32), nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ELU())
        self.spatial_head = nn.Conv2d(32, 2*geometry.height_bins+1, 1)

    def heads(self, hidden):
        spatial = self.spatial_decoder(self.spatial_seed(hidden).reshape(-1, 64, 8, 4))
        output = self.spatial_head(spatial).flatten(2).transpose(1, 2)
        bins = self.geometry.height_bins
        return output[..., :bins], output[..., bins:2*bins], output[..., -1]

    def warm_start_backbone(self, checkpoint):
        prefixes = ('patch.', 'visual.', 'proprio.', 'fuse.', 'memory.')
        selected = {k:v for k,v in checkpoint['encoder'].items() if k.startswith(prefixes)}
        expected = {k for k in self.state_dict() if k.startswith(prefixes)}
        if set(selected) != expected:
            raise ValueError('Incomplete perception backbone initialization')
        self.load_state_dict(selected, strict=False)
        return sorted(selected)


def terrain_loss(model, hidden, targets):
    logits, offsets, edge_logits = model.heads(hidden)
    finite = torch.isfinite(targets['height'])
    height = torch.nan_to_num(targets['height'])
    index, fraction, bounded = height_bins(height, model.geometry)
    mask = targets['visible'].bool() & bounded & finite
    edge = targets['edge'].float()
    # The height bins are mutually exclusive. Do not reward many positive bins.
    ce = F.cross_entropy(logits.transpose(1,2), index, reduction='none')
    weights = 1.+2.*edge
    def weighted(value):
        return torch.where(mask,value*weights,0.).sum()/torch.where(mask,weights,0.).sum().clamp_min(1.)
    offset = offsets.gather(-1,index[...,None]).squeeze(-1).sigmoid()
    offset_loss = weighted(F.smooth_l1_loss(offset, fraction, reduction='none', beta=.1))
    bins = torch.arange(model.geometry.height_bins,device=logits.device)
    width = (model.geometry.z_max-model.geometry.z_min)/model.geometry.height_bins
    expected = (logits.softmax(-1)*(model.geometry.z_min+(bins+offsets.sigmoid())*width)).sum(-1)
    height_loss = weighted(F.smooth_l1_loss(expected,height,reduction='none',beta=.05))/.05
    # Unweighted binary likelihood avoids the previous 4x positive-class bias.
    edge_loss = masked_mean(F.binary_cross_entropy_with_logits(edge_logits,edge,reduction='none'),mask)
    shape = (-1,model.geometry.nx,model.geometry.ny)
    pred_grid, truth_grid, mask_grid = expected.reshape(shape), height.reshape(shape), mask.reshape(shape)
    gradients = []
    for dim in (1,2):
        valid_pair = mask_grid.narrow(dim,0,mask_grid.shape[dim]-1) & mask_grid.narrow(dim,1,mask_grid.shape[dim]-1)
        delta = F.smooth_l1_loss(pred_grid.diff(dim=dim),truth_grid.diff(dim=dim),reduction='none',beta=.05)
        gradients.append(masked_mean(delta,valid_pair)/.05)
    loss = weighted(ce)+.5*offset_loss+.25*height_loss+.5*edge_loss+.1*sum(gradients)
    with torch.no_grad():
        mae = masked_mean((decode_height(logits,offsets,model.geometry)-height).abs(),mask)
    return loss, dict(loss=float(loss.detach()),height_mae_m=float(mae),edge_bce=float(edge_loss.detach()),
                      visible_cells=int(mask.sum()),height_ce=float(weighted(ce).detach()))
