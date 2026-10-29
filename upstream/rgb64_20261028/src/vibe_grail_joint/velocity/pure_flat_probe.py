"""Evaluation-only plane; no stair geometry or terrain motion demonstration."""

import math


def probe_tiles():
    return (dict(level=0, kind="flat", n=0, height=0., tread=.35,
                 start=1.4, plateau=.85, goal=4.2,
                 origin=[-30., -30., 0.]),)


def probe_cases(settings):
    matches = [i for i, speed in enumerate(settings["planner_speeds"])
               if math.isclose(speed, .4, abs_tol=1e-12)]
    if len(matches) != 1:
        raise ValueError("Flat gait review needs one existing 0.4 m/s stream")
    return [dict(id=0, tile=0, stream=matches[0], speed=.4, phase=0,
                 spawn_xy=[-.15, -.15])]


def apply_pure_flat_probe(cfg):
    from .terrain import VelocityAtlasCfg

    atlas = cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"]
    if type(atlas) is not VelocityAtlasCfg:
        raise ValueError("Flat evaluation requires the uniform atlas configuration")
    atlas.rows = ()
    atlas.random_tiles = cfg.commands["motion"].random_tiles = probe_tiles()
    return dict(schema="pure_flat_gait_review_v1", evaluation_only=True,
                risers=0, terrain_height_m=0., tiles=probe_tiles(),
                initialization="Online generated walking state, not standstill",
                command_m_s=.4)
