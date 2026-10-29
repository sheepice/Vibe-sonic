"""One rank's paired stairs and a separate clear flat region in one terrain body."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

FLOOR_RGBA = (0.35, 0.36, 0.38, 1.0)
STAIR_RGBA = (0.50, 0.36, 0.22, 1.0)


def atlas_layout(rows: list[dict]):
    extent = max((max(row["xy_extent"]) for row in rows), default=4.0)
    spacing = 2 * extent + 4.0
    columns = max(1, int(np.ceil(np.sqrt(len(rows)))))
    origins = np.zeros((len(rows) + 1, 3), np.float32)  # final entry is flat
    for i in range(len(rows)):
        origins[i, :2] = (50 + (i % columns) * spacing, 50 + (i // columns) * spacing)
    size = max(100.0, 100 + columns * spacing)
    return origins, (size, size)


@dataclass(kw_only=True)
class JointAtlasCfg(SubTerrainCfg):
    root: str = ""
    rows: tuple[dict, ...] = ()

    def function(self, difficulty, spec: mujoco.MjSpec, rng) -> TerrainOutput:
        del difficulty, rng
        body = spec.body("terrain")
        origins, _ = atlas_layout(list(self.rows))
        geometry = [
            TerrainGeometry(
                geom=body.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.1), rgba=FLOOR_RGBA),
                color=FLOOR_RGBA,
            )
        ]
        for i, row in enumerate(self.rows):
            with np.load(Path(self.root) / row["terrain"], allow_pickle=False) as field:
                heights, xy_min = field["heights"], field["xy_min"]
                resolution = float(field["resolution"])
            if heights.ndim != 2 or not np.isfinite(heights).all() or np.any(heights < 0):
                raise ValueError(f"Invalid terrain {row['id']}")
            nrow, ncol = heights.shape
            half = (np.array([ncol, nrow]) - 1) * resolution / 2
            top = max(float(heights.max()), 1e-3)
            name = f"stair_{i}"
            spec.add_hfield(
                name=name,
                nrow=nrow,
                ncol=ncol,
                size=(*half, top, 0.1),
                userdata=(heights / top).ravel().tolist(),
            )
            center = xy_min + half + origins[i, :2]
            geometry.append(
                TerrainGeometry(
                    geom=body.add_geom(
                        name=name,
                        type=mujoco.mjtGeom.mjGEOM_HFIELD,
                        hfieldname=name,
                        pos=(*center, 0),
                        rgba=STAIR_RGBA,
                        friction=(0.8, 0.005, 0.0001),
                    ),
                    color=STAIR_RGBA,
                )
            )
        # Command origins add this tile's generated origin to the same local atlas offsets.
        return TerrainOutput(origin=np.zeros(3), geometries=geometry)
