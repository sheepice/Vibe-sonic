"""Optional RGB geometry supervision for mutually exclusive height classes.

Labels affect the loss only. Encoder, actor and exported inputs are unchanged.
Keep the original BCE objective available as the comparison experiment.
"""
import torch
from torch.nn import functional as F

from ..hiking_gaussgym.model import mask_mean, decode_height
from ..hiking_gaussgym.voxel import heightmap_to_voxels

KIND = 'saturation_balanced_ce_v1'


def occupancy_loss(logits, occupancy, valid, *, scale=.1, region_mix=.5):
    if not 0 < scale <= 1 or not 0 <= region_mix <= 1:
        raise ValueError('Positive CE scale <= 1 and region mixture in [0, 1] required')
    index = occupancy.long().argmax(-1)
    ce = F.cross_entropy(logits.float().movedim(-1, 1), index, reduction='none')
    global_mean = mask_mean(ce, valid)
    # Give lower saturation, interior surfaces, and upper saturation equal shares
    # of the regional component WHEN PRESENT. Absent regions contribute nothing.
    # This avoids treating 23 empty bins as 23 independent negative examples.
    masks = (valid & (index == 0), valid & (index > 0) & (index < logits.shape[-1]-1),
             valid & (index == logits.shape[-1]-1))
    present = torch.stack([m.any() for m in masks]).to(ce.dtype)
    regions = torch.stack([mask_mean(ce, m) for m in masks])
    regional = (regions*present).sum()/present.sum().clamp_min(1)
    loss = scale*((1-region_mix)*global_mean + region_mix*regional)
    return loss, dict(occupancy_ce_global=float(global_mean.detach()),
                     occupancy_ce_regional=float(regional.detach()))


def reconstruction_loss(model, latent, target, *, scale=.1, region_mix=.5):
    g = model.geometry
    logits, centroid, linvel, angvel = model.heads(latent)
    height = target['height'].reshape(-1, g.nx, g.ny)
    valid = (target['surface_valid'].reshape_as(height).bool() & torch.isfinite(height)
             & target['valid'].bool().reshape(-1, 1, 1))
    interior = valid & (height > g.z_min) & (height < g.z_max)
    occupancy, offset = heightmap_to_voxels(torch.nan_to_num(height), g.height_bins, g.z_min, g.z_max)
    occ, values = occupancy_loss(logits, occupancy, valid, scale=scale, region_mix=region_mix)
    off = mask_mean((centroid.float()-offset).square(), interior[..., None] & occupancy)
    lv = mask_mean((linvel.float()-target['base_lin_vel_b']).square().mean(-1), target['valid'].bool())
    av = mask_mean((angvel.float()-target['base_ang_vel_b']).square().mean(-1), target['valid'].bool())
    with torch.no_grad():
        predicted = decode_height(logits, centroid, g)
        correct = (logits.argmax(-1) == occupancy.long().argmax(-1)).float()
        saturated = valid & ~interior
        lower = valid & (height <= g.z_min)
        values.update(
            height_mae_m=float(mask_mean((predicted-height).abs(), interior)),
            clipped_height_mae_all_m=float(mask_mean((predicted-height.clamp(g.z_min, g.z_max)).abs(), valid)),
            saturated_accuracy=float(mask_mean(correct, saturated)),
            lower_saturated_accuracy=float(mask_mean(correct, lower)),
            all_bin_accuracy=float(mask_mean(correct, valid)),
            saturated_cells=int(saturated.sum()), lower_saturated_cells=int(lower.sum()),
            valid_cells=int(valid.sum()))
    values.update(occupancy_loss=float(occ.detach()), offset_loss=float(off.detach()),
                  linvel_mse=float(lv.detach()), angvel_mse=float(av.detach()))
    return occ+off+lv+av, values
