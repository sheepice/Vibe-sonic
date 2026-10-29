"""Extra supervision in a fixed egocentric forefoot region; no actor inputs.

Keep the whole-map objective, add the same saturation-balanced objective locally,
and normalize the mixture. Regions are fixed coordinates, never oracle footholds.
"""
import math
import torch

from .balanced_loss import occupancy_loss as balanced_occupancy_loss
from ..hiking_gaussgym.model import mask_mean, decode_height
from ..hiking_gaussgym.voxel import heightmap_to_voxels

KIND = 'front_saturation_balanced_ce_v1'
OPTIONS = {'scale', 'region_mix', 'roi_mix', 'x_min', 'x_max', 'y_half_width'}


def front_mask(g, device, *, x_min=.1, x_max=.9, y_half_width=.28):
    if (not all(math.isfinite(x) for x in (x_min, x_max, y_half_width))
            or not g.x_min <= x_min < x_max <= g.x_max
            or not 0 < y_half_width <= min(-g.y_min, g.y_max)):
        raise ValueError('Forefoot region must lie within the geometry grid')
    xs = torch.linspace(g.x_min, g.x_max, g.nx, device=device)
    ys = torch.linspace(g.y_min, g.y_max, g.ny, device=device)
    mask = ((xs[:, None] >= x_min) & (xs[:, None] <= x_max)
            & (ys[None, :].abs() <= y_half_width))
    if not mask.any():
        raise ValueError('Forefoot region contains no grid cells')
    return mask


def occupancy_loss(logits, occupancy, valid, roi, *, scale=.1, region_mix=.5, roi_mix=.5):
    if not math.isfinite(roi_mix) or not 0 <= roi_mix <= 1:
        raise ValueError('ROI mixture must be in [0, 1]')
    whole, values = balanced_occupancy_loss(logits, occupancy, valid,
                                           scale=scale, region_mix=region_mix)
    local_valid = valid & roi
    local, local_values = balanced_occupancy_loss(logits, occupancy, local_valid,
                                                scale=scale, region_mix=region_mix)
    # A missing valid ROI must not dilute the original supervision.
    weight = roi_mix * local_valid.any().to(whole.dtype)
    values.update({'front_'+k: v for k, v in local_values.items()})
    values['front_valid_cells'] = int(local_valid.sum())
    values['front_loss_mix'] = float(weight)
    return (1-weight)*whole + weight*local, values


def reconstruction_loss(model, latent, target, *, scale=.1, region_mix=.5,
                        roi_mix=.5, x_min=.1, x_max=.9, y_half_width=.28):
    g = model.geometry
    roi = front_mask(g, latent.device, x_min=x_min, x_max=x_max,
                     y_half_width=y_half_width)
    logits, centroid, linvel, angvel = model.heads(latent)
    height = target['height'].reshape(-1, g.nx, g.ny)
    valid = (target['surface_valid'].reshape_as(height).bool() & torch.isfinite(height)
             & target['valid'].bool().reshape(-1, 1, 1))
    interior = valid & (height > g.z_min) & (height < g.z_max)
    occupancy, offset = heightmap_to_voxels(torch.nan_to_num(height), g.height_bins, g.z_min, g.z_max)
    occ, values = occupancy_loss(logits, occupancy, valid, roi, scale=scale,
                                region_mix=region_mix, roi_mix=roi_mix)
    offset_error = (centroid.float()-offset).square()
    whole_off = mask_mean(offset_error, interior[..., None] & occupancy)
    front_interior = interior & roi
    front_off = mask_mean(offset_error, front_interior[..., None] & occupancy)
    weight = roi_mix * front_interior.any().to(whole_off.dtype)
    off = (1-weight)*whole_off + weight*front_off
    lv = mask_mean((linvel.float()-target['base_lin_vel_b']).square().mean(-1), target['valid'].bool())
    av = mask_mean((angvel.float()-target['base_ang_vel_b']).square().mean(-1), target['valid'].bool())
    with torch.no_grad():
        predicted = decode_height(logits, centroid, g)
        correct = (logits.argmax(-1) == occupancy.long().argmax(-1)).float()
        clipped_error = (predicted-height.clamp(g.z_min, g.z_max)).abs()
        lower = valid & (height <= g.z_min)
        values.update(
            height_mae_m=float(mask_mean((predicted-height).abs(), interior)),
            clipped_height_mae_all_m=float(mask_mean(clipped_error, valid)),
            saturated_accuracy=float(mask_mean(correct, valid & ~interior)),
            lower_saturated_accuracy=float(mask_mean(correct, lower)),
            all_bin_accuracy=float(mask_mean(correct, valid)),
            saturated_cells=int((valid & ~interior).sum()),
            lower_saturated_cells=int(lower.sum()), valid_cells=int(valid.sum()),
            front_clipped_height_mae_m=float(mask_mean(clipped_error, valid & roi)),
            front_lower_saturated_accuracy=float(mask_mean(correct, lower & roi)),
            front_lower_saturated_cells=int((lower & roi).sum()),
            front_all_bin_accuracy=float(mask_mean(correct, valid & roi)))
    values.update(occupancy_loss=float(occ.detach()), offset_loss=float(off.detach()),
                  front_offset_loss=float(front_off.detach()), linvel_mse=float(lv.detach()),
                  angvel_mse=float(av.detach()))
    return occ+off+lv+av, values
