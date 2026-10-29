"""A stationary world camera framing one full reference and its paired stairs."""

from itertools import product
from pathlib import Path

import numpy as np


def attach_fixed_observer(cfg, bank, bank_path, clip_index):
    from mjlab.sensor import CameraSensorCfg
    from scipy.spatial.transform import Rotation
    from .terrain import atlas_layout

    clip = bank["clips"][clip_index]
    rows = bank["terrain_rows"]
    terrain = rows[clip["terrain_id"]]
    poses = np.memmap(Path(bank_path) / "body_pos_w.bin", mode="r", dtype=np.float32,
                      shape=(bank["total_frames"], 14, 3))
    motion = np.array(poses[clip["offset"]:clip["offset"] + clip["length"]])
    root = motion[:, 0]
    low, high = np.argmin(root[:, 2]), np.argmax(root[:, 2])
    upslope = root[high] - root[low]
    upslope[2] = 0
    if np.linalg.norm(upslope) < .1:
        raise ValueError("Cannot determine stair direction from this reference")
    upslope /= np.linalg.norm(upslope)
    side = np.cross([0., 0., 1.], upslope)
    points = np.concatenate([motion.reshape(-1, 3), np.asarray(terrain["mesh_bounds"])])
    lower = points.min(0) - [.35, .35, .2]
    upper = points.max(0) + [.35, .35, .3]
    center = (lower + upper) / 2
    # View the risers from the lower landing at an oblique, fixed angle.
    z = -.65 * upslope - side + np.array([0., 0., .65])
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z], axis=1)
    corners = np.array(list(product(*zip(lower, upper))))
    projected = (corners - center) @ rotation
    fovy, width, height = 43., 640, 360
    tan_y = np.tan(np.deg2rad(fovy / 2))
    tan_x = tan_y * width / height
    distance = max(np.max(np.abs(projected[:, 0]) / tan_x + projected[:, 2]),
                   np.max(np.abs(projected[:, 1]) / tan_y + projected[:, 2])) * 1.06
    atlas, size = atlas_layout(rows)
    generator = cfg.scene.terrain.terrain_generator
    if (generator.num_rows, generator.num_cols) != (1, 1):
        raise ValueError("This camera placement expects the existing single-tile atlas")
    origin = atlas[clip["terrain_id"]] + np.array([-size[0] / 2, -size[1] / 2, 0.])
    position = center + origin + distance * z
    quaternion = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    cfg.scene.sensors = (*cfg.scene.sensors, CameraSensorCfg(
        name="review_camera", parent_body=None, pos=tuple(position), quat=tuple(quaternion),
        fovy=fovy, width=width, height=height, data_types=("rgb",),
        enabled_geom_groups=(0, 2), use_shadows=False, use_textures=True,
    ))
    return {"parent": "world", "position": position.tolist(), "quaternion_wxyz": quaternion.tolist(),
            "target": (center + origin).tolist(), "fovy": fovy, "width": width, "height": height,
            "identical_camera_in_both_worlds": True, "clip_id": clip["id"]}
