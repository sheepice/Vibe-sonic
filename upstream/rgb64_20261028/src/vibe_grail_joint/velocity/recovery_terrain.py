"""Irregular stair curriculum with physical/reward geometry from the same profile.

Original terrain code and the held-out GRAIL staircase are unchanged. These
profiles are generated from a separate seed; no evaluation mesh is imported.
"""

from dataclasses import dataclass, fields
import itertools

import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from ..terrain import JointAtlasCfg, STAIR_RGBA
from .command import VelocityMotionCommand, VelocityMotionCommandCfg


def profile_tiles(seed, levels=11, variants=3):
    rng = np.random.default_rng(seed)
    tiles = []
    # Preserve contiguous level/kind blocks so sampling is explicit and auditable.
    for level, variant, kind in itertools.product(range(levels), range(variants), ('up', 'down', 'up_down', 'flat')):
        n = int(rng.integers(4, 9))
        maximum = .04 + .02 * level
        if variant == 0:
            risers = np.full(n, rng.uniform(.95, 1.05) * maximum)
            treads = np.full(n, rng.uniform(.28, .40))
        else:
            risers = rng.uniform(.55, 1.05, n) * maximum
            treads = rng.uniform(max(.15, .28 - level * .013), .40, n)
        risers = np.clip(risers, .025, .25)
        start, plateau = float(rng.uniform(.8, 1.8)), float(rng.uniform(.7, 1.4))
        up_edges = start + np.concatenate([[0.], np.cumsum(treads[:-1])])
        end_up = start + treads.sum()
        down_edges = end_up + plateau + np.concatenate([[0.], np.cumsum(treads[::-1][:-1])])
        if kind == 'up':
            edges, jumps, base, end = up_edges, risers, 0., end_up
        elif kind == 'down':
            edges, jumps, base, end = up_edges, -risers[::-1], float(risers.sum()), end_up
        elif kind == 'up_down':
            edges = np.concatenate([up_edges, down_edges])
            jumps = np.concatenate([risers, -risers[::-1]])
            base, end = 0., float(end_up + plateau + treads.sum())
        else:
            edges, jumps, base, end = np.array([]), np.array([]), 0., 4.
        i = len(tiles)
        tiles.append(dict(level=level, variant=variant, kind=kind, n=n,
                          height=float(risers.max()), tread=float(treads.mean()),
                          start=start, plateau=plateau, goal=float(end + .6),
                          origin=[-30. - (i % 12) * 16., -30. - (i // 12) * 10., 0.],
                          edges=edges.tolist(), jumps=jumps.tolist(), base=base,
                          risers=risers.tolist(), treads=treads.tolist(),
                          total_height=float(risers.sum())))
    return tiles


def profile_height_numpy(x, tile):
    x = np.asarray(x)
    value = np.full_like(x, tile['base'], dtype=np.float64)
    for edge, jump in zip(tile['edges'], tile['jumps']):
        value += (x >= edge) * jump
    return np.where((x >= -2.) & (x <= 12.), np.maximum(value, 0.), 0.)


@dataclass(kw_only=True)
class ProfileAtlasCfg(JointAtlasCfg):
    random_tiles: tuple = ()

    def function(self, difficulty, spec, rng):
        result = super().function(difficulty, spec, rng)
        body = spec.body('terrain')
        for i, tile in enumerate(self.random_tiles):
            ox, oy, _ = tile['origin']
            breaks = sorted(set([-2., *tile['edges'], 12.]))
            for j, (left, right) in enumerate(itertools.pairwise(breaks)):
                height = float(profile_height_numpy((left + right) / 2, tile))
                if height <= 1e-7:
                    continue
                geom = body.add_geom(name=f'profile_stair_{i}_{j}', type=mujoco.mjtGeom.mjGEOM_BOX,
                                     size=((right-left)/2, 2., height/2),
                                     pos=(ox+(right+left)/2, oy, height/2),
                                     rgba=STAIR_RGBA, friction=(.8, .005, .0001))
                result.geometries.append(TerrainGeometry(geom=geom, color=STAIR_RGBA))
        return TerrainOutput(origin=result.origin, geometries=result.geometries)


class ProfileMotionCommand(VelocityMotionCommand):
    def _init_task(self):
        super()._init_task()
        self._profile_base = torch.tensor([t['base'] for t in self.cfg.random_tiles], device=self.device)
        self._profile_edges = torch.full((len(self.cfg.random_tiles), 16), float('inf'), device=self.device)
        self._profile_jumps = torch.zeros_like(self._profile_edges)
        for i, tile in enumerate(self.cfg.random_tiles):
            size = len(tile['edges'])
            self._profile_edges[i, :size] = torch.tensor(tile['edges'], device=self.device)
            self._profile_jumps[i, :size] = torch.tensor(tile['jumps'], device=self.device)

    def ground_height(self, xy, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        tile = self.tile[ids]
        expand = (slice(None),) + (None,) * (xy.ndim - 2)
        x = xy[..., 0] - self.tile_origins[tile, 0][expand]
        y = xy[..., 1] - self.tile_origins[tile, 1][expand]
        height = self._profile_base[tile][expand] + (
            (x[..., None] >= self._profile_edges[tile][expand]) * self._profile_jumps[tile][expand]
        ).sum(-1)
        return torch.where((x < -2.) | (x > 12.) | (y.abs() > 2.), 0., height.clamp(min=0.))

    def _sample_init_frame(self, env_ids):
        # Original replay allocation and source mixture remain exactly 60/20/20.
        chosen, frames = super()._sample_init_frame(env_ids)
        mask = self.online[env_ids]
        ids = env_ids[mask]
        if len(ids):
            update = int(getattr(self._env, 'policy_update_count', 0))
            max_level = min(self.cfg.profile_levels - 1, 1 + update // self.cfg.curriculum_interval)
            level = torch.randint(0, max_level + 1, (len(ids),), device=self.device)
            variant = torch.randint(0, self.cfg.profile_variants, (len(ids),), device=self.device)
            kind = torch.multinomial(torch.tensor([.35, .25, .25, .15], device=self.device), len(ids), replacement=True)
            self.tile[ids] = (level * self.cfg.profile_variants + variant) * 4 + kind
            spawn = self.tile_origins[self.tile[ids]].clone()
            spawn[:, 0] += torch.empty(len(ids), device=self.device).uniform_(-.3, .3)
            spawn[:, 1] += torch.empty(len(ids), device=self.device).uniform_(-.35, .35)
            spawn[:, 2] = self.ground_height(spawn[:, :2], ids)
            spawn[:, :2] -= self.motion.body_pos_w[frames[mask], 0, :2]
            self._env.scene.env_origins[ids] = spawn
        return chosen, frames


@dataclass(kw_only=True)
class ProfileMotionCommandCfg(VelocityMotionCommandCfg):
    profile_levels: int = 11
    profile_variants: int = 3

    def build(self, env):
        return ProfileMotionCommand(self, env)


def configure_profile_task(cfg, settings, rank):
    tiles = tuple(profile_tiles(settings['seed'] + rank * 7919, settings.get('profile_levels', 11),
                                settings.get('profile_variants', 3)))
    old = cfg.commands['motion']
    params = {f.name: getattr(old, f.name) for f in fields(ProfileMotionCommandCfg) if hasattr(old, f.name)}
    params.update(random_tiles=tiles, profile_levels=settings.get('profile_levels', 11),
                  profile_variants=settings.get('profile_variants', 3))
    cfg.commands['motion'] = ProfileMotionCommandCfg(**params)
    atlas = cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs'] = ProfileAtlasCfg(
        root=atlas.root, rows=atlas.rows, random_tiles=tiles)
    # Eight irregular ascending + descending steps can take over 30 s at 0.25 m/s.
    cfg.episode_length_s = settings.get('episode_length_s', 45.)
    cfg.terminations['online_success'].func = profile_success
    return cfg


def profile_success(env):
    """Full crossing, both feet on the far landing, upright for half a second."""
    c = env.command_manager.get_term('motion')
    step = int(env.common_step_counter)
    if step != c.success_step:
        root, feet = c.robot.data.root_link_pos_w, c.robot_body_pos_w[:, [3, 6]]
        origin = c.tile_origins[c.tile]
        goal = c.tile_data['goal'][c.tile]
        passed = (root[:, 0] - origin[:, 0] >= goal) & (c.speed > .1)
        passed &= (feet[:, :, 0] - origin[:, None, 0]).min(-1).values >= goal - .45
        passed &= (feet[:, :, 2] - c.ground_height(feet[:, :, :2])).abs().max(-1).values < .18
        passed &= c.robot.data.projected_gravity_b[:, 2] < -.85
        passed &= root[:, 2] - c.ground_height(root[:, :2]) > .45
        c.success_hold[:] = torch.where(passed & c.online, c.success_hold + 1, 0)
        c.success_step = step
    result = (c.success_hold >= 25) & c.online
    c.terminal_success = result.clone()
    return result
