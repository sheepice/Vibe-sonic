"""Train below-grid void category too; centroid regression stays unsaturated."""
import torch
from torch.nn import functional as F
from ..hiking_gaussgym.voxel import heightmap_to_voxels
from ..hiking_gaussgym.model import mask_mean,decode_height

def reconstruction_loss(model,latent,target):
    g=model.geometry
    logits,centroid,linvel,angvel=model.heads(latent)
    height=target['height'].reshape(-1,g.nx,g.ny)
    valid=target['surface_valid'].reshape_as(height).bool()&torch.isfinite(height)&target['valid'].bool().reshape(-1,1,1)
    interior=valid&(height>g.z_min)&(height<g.z_max)
    occupancy,offset=heightmap_to_voxels(torch.nan_to_num(height),g.height_bins,g.z_min,g.z_max)
    occ=mask_mean(F.binary_cross_entropy_with_logits(logits.float(),occupancy.float(),reduction='none'),valid[...,None].expand_as(logits))
    off=mask_mean((centroid.float()-offset).square(),interior[...,None]&occupancy)
    lv=mask_mean((linvel.float()-target['base_lin_vel_b']).square().mean(-1),target['valid'].bool())
    av=mask_mean((angvel.float()-target['base_ang_vel_b']).square().mean(-1),target['valid'].bool())
    with torch.no_grad():
        mae=mask_mean((decode_height(logits,centroid,g)-height).abs(),interior)
        saturated=valid&~interior
        accuracy=mask_mean((logits.argmax(-1)==occupancy.long().argmax(-1)).float(),saturated)
    return occ+off+lv+av,dict(occupancy_loss=float(occ.detach()),offset_loss=float(off.detach()),
        linvel_mse=float(lv.detach()),angvel_mse=float(av.detach()),height_mae_m=float(mae),
        saturated_accuracy=float(accuracy),saturated_cells=int(saturated.sum()),valid_cells=int(valid.sum()))
