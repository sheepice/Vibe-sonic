"""One continuous flat / 20 cm stairs / flat evaluation route."""

from dataclasses import dataclass
import itertools

import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from ..terrain import JointAtlasCfg, STAIR_RGBA
from .eval_command import EvaluationCommand, EvaluationCommandCfg
from .terrain import VelocityAtlasCfg, height_numpy


def probe_tiles():
    return (dict(level=0, kind="up_down", n=6, height=.2, tread=.35,
                 start=8., plateau=.85, goal=20.9,
                 origin=[-30., -30., 0.]),)


def probe_cases(settings):
    matches = [i for i, speed in enumerate(settings["planner_speeds"]) if abs(speed-.4) < 1e-9]
    if len(matches) != 1:
        raise ValueError("Long route requires one existing 0.4 m/s stream")
    return [dict(id=0, tile=0, stream=matches[0], speed=.4, phase=0,
                 spawn_xy=[-.15, -.15])]


@dataclass(kw_only=True)
class LongWalkAtlasCfg(VelocityAtlasCfg):
    def function(self, difficulty, spec, rng):
        result = JointAtlasCfg.function(self, difficulty, spec, rng)
        body = spec.body("terrain")
        for index, tile in enumerate(self.random_tiles):
            start, n, tread = tile["start"], tile["n"], tile["tread"]
            down = start+n*tread+tile["plateau"]
            breaks = sorted({-2., tile["goal"]+2.,
                             *(start+j*tread for j in range(n)),
                             *(down+j*tread for j in range(n))})
            for segment, (left, right) in enumerate(itertools.pairwise(breaks)):
                height = float(height_numpy(np.array([(left+right)/2]), tile)[0])
                if height <= 1e-7:
                    continue
                ox, oy, _ = tile["origin"]
                geom = body.add_geom(
                    name=f"long_walk_stair_{index}_{segment}", type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=((right-left)/2, 2., height/2),
                    pos=(ox+(left+right)/2, oy, height/2),
                    rgba=STAIR_RGBA, friction=(.8, .005, .0001))
                result.geometries.append(TerrainGeometry(geom=geom, color=STAIR_RGBA))
        return TerrainOutput(origin=result.origin, geometries=result.geometries)


class LongWalkCommand(EvaluationCommand):
    def ground_height(self, xy, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        tile = self.tile[ids]
        expand = (slice(None),)+(None,)*(xy.ndim-2)
        x = xy[..., 0]-self.tile_origins[tile, 0][expand]
        y = xy[..., 1]-self.tile_origins[tile, 1][expand]
        p = {key: values[tile][expand] for key, values in self.tile_data.items()}
        up = torch.minimum(((x-p["start"])/p["tread"]).floor().add(1).clamp(min=0), p["n"])
        down = torch.minimum(
            ((x-p["start"]-p["n"]*p["tread"]-p["plateau"])/p["tread"]).floor().add(1).clamp(min=0),
            p["n"])
        return torch.where(y.abs() > 2., 0., (up-down)*p["height"])


@dataclass(kw_only=True)
class LongWalkCommandCfg(EvaluationCommandCfg):
    def build(self, env):
        return LongWalkCommand(self, env)


def apply_long_walk_probe(cfg):
    old = cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"]
    tiles = probe_tiles()
    cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"] = LongWalkAtlasCfg(
        root=old.root, rows=(), random_tiles=tiles)
    cfg.commands["motion"].random_tiles = tiles
    return dict(schema="long_flat_stairs_flat_v1", evaluation_only=True, riser_height_m=.2,
                steps_per_flight=6, tread_depth_m=.35, width_m=4., summit_height_m=1.2,
                first_edge_x_m=8., summit_start_x_m=9.75, descent_start_x_m=10.95,
                final_edge_x_m=12.7, finish_root_x_m=20.9,
                initial_flat_m=8.15, final_flat_m=8.2, tiles=tiles,
                command_m_s=.4, no_resets_or_policy_switches=True)


def phase_name(x):
    if x < 8.:
        return "FLAT BEFORE"
    if x < 9.75:
        return "STAIR UP"
    if x < 10.95:
        return "SUMMIT"
    if x < 12.7:
        return "STAIR DOWN"
    return "FLAT AFTER"


def fixed_views(origin):
    """Five fixed world cameras with overlapping full-body framing."""
    from scipy.spatial.transform import Rotation
    z = np.array([-.12, -1., .28]); z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z], axis=1)
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    views = []
    for left, right, top in ((-.55, 4.5, 1.45), (3.1, 8.5, 1.9),
                             (6.6, 14.2, 2.65), (12.8, 18., 1.5), (16.7, 21.8, 1.45)):
        lo, hi = np.array([left, -.65, -.08]), np.array([right, .65, top])
        center = (lo+hi)/2
        corners = np.array(list(itertools.product(*zip(lo, hi))))
        projected = (corners-center)@rotation
        tan_y = np.tan(np.deg2rad(43/2))
        distance = max(np.max(abs(projected[:, 0])/(tan_y*960/540)+projected[:, 2]),
                       np.max(abs(projected[:, 1])/tan_y+projected[:, 2]))*1.03
        views.append(dict(position=(np.asarray(origin)+center+distance*z).tolist(),
                          quaternion=quat.tolist(), distance_m=float(distance),
                          bounds=[lo.tolist(), hi.tolist()]))
    return views, rotation


def view_index(x):
    return int(np.searchsorted([3.5, 7.2, 13.5, 17.3], x, side="right"))
