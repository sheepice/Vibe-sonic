"""Evaluation-only exact 20 cm stairs, shared by collision and height queries."""

from __future__ import annotations

import math


def probe_tiles():
    """Six risers; the continuous summit in up_down is exactly 1.2 m long.

    The legacy plateau field excludes the final ascending tread, so it is
    1.2 - .35, rather than 1.2. Single-direction landings extend to tile bounds.
    """
    tiles = []
    for index, kind in enumerate(("up", "down", "up_down", "flat")):
        n, height, tread, start, plateau = 6, .20, .35, 1.4, 1.2 - .35
        end = start + n * tread + (plateau + n * tread if kind == "up_down" else 0.)
        tiles.append(dict(level=0, kind=kind, n=n, height=height, tread=tread,
                          start=start, plateau=plateau, goal=end + .6,
                          origin=[-30. - index * 16., -30., 0.]))
    return tuple(tiles)


def probe_cases(settings):
    speeds = settings["planner_speeds"]
    matches = [i for i, speed in enumerate(speeds) if math.isclose(speed, .4, abs_tol=1e-12)]
    if len(matches) != 1:
        raise ValueError("The 20 cm probe requires one existing 0.4 m/s planner stream")
    return [dict(id=i, tile=i, stream=matches[0], speed=.4, phase=0,
                 spawn_xy=[-.15, -.15]) for i in range(3)]


def apply_uniform_probe20(cfg):
    from .terrain import VelocityAtlasCfg
    atlas = cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"]
    if type(atlas) is not VelocityAtlasCfg:
        raise ValueError("Exact uniform probe requires the uniform box terrain configuration")
    tiles = probe_tiles()
    atlas.random_tiles = cfg.commands["motion"].random_tiles = tiles
    return dict(schema="uniform_probe_20cm_v1", evaluation_only=True,
                riser_height_m=.20, risers=6, total_height_m=1.2,
                tread_depth_m=.35, physical_width_m=4., first_edge_x_m=1.4,
                up_down_continuous_summit_length_m=1.2,
                legacy_plateau_field_m=1.2 - .35,
                single_direction_landing_extent_x_m=[-2., 12.],
                command_m_s=.4, tiles=tiles,
                height_novelty="20 cm lies within the irregular training height range; this exact uniform layout is a new evaluation scene",
                criterion_unchanged=True)
