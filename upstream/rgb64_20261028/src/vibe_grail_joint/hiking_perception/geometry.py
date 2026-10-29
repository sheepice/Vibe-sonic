"""CPU/GPU geometry labels; all functions are independent of the actor."""
import math
import torch
from .contracts import GeometrySpec


def yaw_from_quat(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))


def rotate_yaw(points, yaw):
    while yaw.ndim < points.ndim-1:
        yaw = yaw.unsqueeze(-1)
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c*points[..., 0]-s*points[..., 1],
                        s*points[..., 0]+c*points[..., 1], points[..., 2]), -1)


def grid_world(root, yaw, spec=GeometrySpec()):
    x = torch.linspace(spec.x_min, spec.x_max, spec.nx, device=root.device)
    y = torch.linspace(spec.y_min, spec.y_max, spec.ny, device=root.device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    local = torch.stack((xx, yy, torch.zeros_like(xx)), -1).flatten(0, 1)
    return root[:, None] + rotate_yaw(local[None].expand(len(root), -1, -1), yaw)


def profile_heights(x, edges, jumps, base):
    """x [B,...], edges/jumps [B,E]. Padded edges are +inf."""
    shape = [len(x)] + [1]*(x.ndim-1) + [edges.shape[-1]]
    value = ((x[..., None] >= edges.reshape(shape))*jumps.reshape(shape)).sum(-1)
    return value + base.reshape([len(x)]+[1]*(x.ndim-1))


def height_labels(world, root, origins, edges, jumps, base, spec=GeometrySpec()):
    x = world[..., 0]-origins[:, None, 0]
    height_w = profile_heights(x, edges, jumps, base)+origins[:, None, 2]
    relative = height_w-root[:, None, 2]
    distance = (x[..., None]-edges[:, None]).abs()
    valid_edge = torch.isfinite(edges) & (jumps.abs() > 1e-6)
    distance = torch.where(valid_edge[:, None], distance, 1e4).amin(-1)
    valid = ((world[..., 1]-origins[:, None, 1]).abs() < 2.) & (x >= -2.) & (x <= 12.)
    points = world.clone(); points[..., 2] = height_w
    return {'height': relative, 'edge': distance < spec.edge_radius,
            'edge_distance': distance.clamp_max(2.), 'surface_valid': valid, 'points_w': points}


def project(points_w, camera_position, camera_rotation, width, height, fovy):
    """MuJoCo camera: rotation columns are world axes, looks down local -Z."""
    local = torch.matmul(points_w-camera_position[:, None], camera_rotation)
    depth = -local[..., 2]
    focal = height/(2*math.tan(math.radians(fovy)/2))
    u = width/2 + focal*local[..., 0]/depth.clamp_min(1e-6)
    v = height/2 - focal*local[..., 1]/depth.clamp_min(1e-6)
    valid = (depth > 1e-4) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return torch.stack((u, v), -1), depth, valid


def visible_from_depth(uv, camera_depth, in_fov, depth_image, tolerance):
    """Metric optical-axis depth. Renderer depth, never actor input."""
    if depth_image.ndim == 4:
        depth_image = depth_image[..., 0]
    b, h, w = depth_image.shape
    xy = uv.round().long()
    sample = depth_image[torch.arange(b, device=uv.device)[:, None],
                         xy[..., 1].clamp(0, h-1), xy[..., 0].clamp(0, w-1)]
    return in_fov & torch.isfinite(sample) & (sample > 0) & ((sample-camera_depth).abs() <= tolerance)


def height_bins(height, spec=GeometrySpec()):
    """Saturated values are masked, not silently supervised as valid heights."""
    normalized = (height-spec.z_min)/(spec.z_max-spec.z_min)*spec.height_bins
    index = normalized.floor().long().clamp(0, spec.height_bins-1)
    offset = (normalized-index).clamp(0, 1)
    valid = (height >= spec.z_min) & (height < spec.z_max) & torch.isfinite(height)
    return index, offset, valid


def decode_height(logits, offsets, spec=GeometrySpec()):
    index = logits.argmax(-1)
    offset = offsets.gather(-1, index[..., None]).squeeze(-1).sigmoid()
    return spec.z_min + (index+offset)*(spec.z_max-spec.z_min)/spec.height_bins


def masked_mean(value, mask):
    # Mask before the sum: NaNs in invalid labels must not contaminate gradients.
    return torch.where(mask, value, 0.).sum()/mask.sum().clamp_min(1)
