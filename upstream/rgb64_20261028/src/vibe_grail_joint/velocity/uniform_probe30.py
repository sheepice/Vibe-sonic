"""Evaluation-only 30 cm risers, retaining the exact 25 cm probe layout."""

from .uniform_probe import probe_cases, probe_tiles as probe_tiles25


def probe_tiles():
    return tuple(dict(tile, height=.30) for tile in probe_tiles25())


def apply_uniform_probe30(cfg):
    from .terrain import VelocityAtlasCfg

    atlas = cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    if type(atlas) is not VelocityAtlasCfg:
        raise ValueError('The 30 cm probe requires uniform box terrain')
    tiles = probe_tiles()
    atlas.random_tiles = cfg.commands['motion'].random_tiles = tiles
    return dict(schema='uniform_probe_30cm_v1', evaluation_only=True,
                riser_height_m=.30, risers=6, total_height_m=1.8,
                tread_depth_m=.35, physical_width_m=4., first_edge_x_m=1.4,
                up_down_continuous_summit_length_m=1.2,
                legacy_plateau_field_m=.85,
                single_direction_landing_extent_x_m=[-2., 12.],
                command_m_s=.4, tiles=tiles,
                height_novelty='30 cm exceeds the 25 cm cap of the procedural velocity-training terrain; no claim is made about all GRAIL replay or base-model pretraining terrain',
                criterion_unchanged=True)
