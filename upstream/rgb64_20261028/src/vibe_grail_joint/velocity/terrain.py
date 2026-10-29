"""Seeded random stair tiles, alongside the unchanged GRAIL atlas."""

import itertools
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import TerrainGeometry, TerrainOutput

from ..terrain import STAIR_RGBA, JointAtlasCfg


def tile_specs(seed, levels=8):
    rng = np.random.default_rng(seed)
    tiles = []
    for level in range(levels):
        for kind in ("up", "down", "up_down", "flat"):
            n = int(rng.integers(4, 7))
            tread = float(rng.uniform(0.28, 0.40))
            start = float(rng.uniform(1.1, 1.7))
            plateau = float(rng.uniform(0.9, 1.3))
            h = float(0.04 + level * 0.02 + rng.uniform(-0.004, 0.004))
            end = start + n * tread + (plateau + n * tread if kind == "up_down" else 0)
            i = len(tiles)
            tiles.append(
                {
                    "level": level,
                    "kind": kind,
                    "n": n,
                    "tread": tread,
                    "start": start,
                    "plateau": plateau,
                    "height": h,
                    "goal": end + 0.6,
                    "origin": [-30.0 - (i % 8) * 16.0, -30.0 - (i // 8) * 10.0, 0.0],
                }
            )
    return tiles


def height_numpy(x, tile):
    n, h, start, tread = (tile[k] for k in ("n", "height", "start", "tread"))
    up = np.clip(np.floor((x - start) / tread) + 1, 0, n)
    if tile["kind"] == "flat":
        return np.zeros_like(x)
    if tile["kind"] == "down":
        return (n - up) * h
    if tile["kind"] == "up_down":
        down = np.clip(np.floor((x - (start + n * tread + tile["plateau"])) / tread) + 1, 0, n)
        return (up - down) * h
    return up * h


@dataclass(kw_only=True)
class VelocityAtlasCfg(JointAtlasCfg):
    random_tiles: tuple = ()

    def function(self, difficulty, spec, rng):
        result = super().function(difficulty, spec, rng)
        body = spec.body("terrain")
        for i, tile in enumerate(self.random_tiles):
            if tile["kind"] == "flat":
                continue
            # Individual watertight boxes avoid height-field edge interpolation
            # and make the analytic reward heights match actual contact surfaces.
            breaks = [-2.0, tile["start"]]
            breaks += [tile["start"] + j * tile["tread"] for j in range(1, tile["n"])]
            if tile["kind"] == "up_down":
                ds = tile["start"] + tile["n"] * tile["tread"] + tile["plateau"]
                breaks += [ds + j * tile["tread"] for j in range(tile["n"])]
            breaks += [12.0]
            breaks = sorted(set(breaks))
            ox, oy, _ = tile["origin"]
            for j, (left, right) in enumerate(itertools.pairwise(breaks)):
                top = float(height_numpy(np.array([(left + right) / 2]), tile)[0])
                if top <= 0:
                    continue
                geom = body.add_geom(
                    name=f"velocity_stair_{i}_{j}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=((right - left) / 2, 2.0, top / 2),
                    pos=(ox + (right + left) / 2, oy, top / 2),
                    rgba=STAIR_RGBA,
                    friction=(0.8, 0.005, 0.0001),
                )
                result.geometries.append(TerrainGeometry(geom=geom, color=STAIR_RGBA))
        return TerrainOutput(origin=result.origin, geometries=result.geometries)
