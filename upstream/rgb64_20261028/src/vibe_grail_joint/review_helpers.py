"""Evaluation-only camera and visual controls; never imported by training."""

import numpy as np
import torch

from dataclasses import dataclass
from .terrain import JointAtlasCfg


@dataclass(kw_only=True)
class PerturbedAtlasCfg(JointAtlasCfg):
    """Evaluation stress test: change physical stair heights, keep references fixed."""

    height_scale: float = 1.0

    def function(self, difficulty, spec, rng):
        result = super().function(difficulty, spec, rng)
        for index in range(len(self.rows)):
            field = spec.hfield(f"stair_{index}")
            size = field.size.copy()
            size[2] *= self.height_scale
            field.size = size
        return result


def perturb_stair_heights(cfg, scale):
    if not 0.8 <= scale <= 1.2:
        raise ValueError("Review height scaling is limited to a small controlled perturbation")
    old = cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"]
    cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"] = PerturbedAtlasCfg(
        root=old.root, rows=old.rows, height_scale=scale,
    )


def alter_visual(obs, mode):
    if mode == "none":
        return obs
    changed = obs.clone()
    if mode == "zero_tokens":
        changed["kv_tokens", "img_tokens"].zero_()
        changed["q_cls"].zero_()
    elif mode == "shuffle_tokens":
        changed["kv_tokens", "img_tokens"] = torch.roll(obs["kv_tokens", "img_tokens"], 1, 0)
        changed["q_cls"] = torch.roll(obs["q_cls"], 1, 0)
    else:
        raise ValueError(mode)
    return changed


def attach_observer(cfg):
    from mjlab.sensor import CameraSensorCfg
    from scipy.spatial.transform import Rotation

    position = np.array([-2.8, -2.8, 1.6])
    target = np.array([0.7, 0.0, 0.2])
    z = position - target
    z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()[[3, 0, 1, 2]]
    cfg.scene.sensors = (*cfg.scene.sensors, CameraSensorCfg(
        name="review_camera", parent_body="robot/pelvis", pos=tuple(position),
        quat=tuple(quat), fovy=55, width=640, height=360, data_types=("rgb",),
        enabled_geom_groups=(0, 2), use_shadows=False, use_textures=True,
    ))


def disable_adapters(actor):
    with torch.no_grad():
        for parameter in actor.decoder.adapters.parameters():
            parameter.zero_()


def write_video(path, frames):
    """Pad display-only frames to even dimensions for browser-compatible H.264."""
    import imageio.v2 as iio

    padded = [np.pad(frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0)),
                     mode="edge") for frame in frames]
    iio.mimwrite(path, padded, fps=25, macro_block_size=1, ffmpeg_params=["-threads", "2"])
