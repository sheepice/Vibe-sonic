"""Evaluation-only color changes applied before rendering the actual policy RGB."""
from dataclasses import dataclass, fields
import hashlib
import numpy as np
from ..velocity.recovery_terrain import ProfileAtlasCfg


@dataclass(kw_only=True)
class ColorAtlasCfg(ProfileAtlasCfg):
    stair_rgba: tuple = (.75,.78,.82,1.)

    def function(self,difficulty,spec,rng):
        result=super().function(difficulty,spec,rng)
        count=0
        for entry in result.geometries:
            if entry.geom is not None and entry.geom.name.startswith('profile_stair_'):
                entry.geom.rgba=self.stair_rgba
                entry.color=self.stair_rgba
                count+=1
        if not count:raise ValueError('No stair geometry colored')
        return result


def configure_color(cfg,rgba):
    old=cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    params={f.name:getattr(old,f.name) for f in fields(ColorAtlasCfg) if hasattr(old,f.name)}
    cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']=ColorAtlasCfg(**params,stair_rgba=tuple(rgba))


def audit_color(raw,scene):
    import mujoco
    model=raw.sim._mj_model
    terrain=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'terrain')
    ids=[i for i in range(model.ngeom) if model.geom_bodyid[i]==terrain and model.geom_type[i]==mujoco.mjtGeom.mjGEOM_BOX]
    colors=model.geom_rgba[ids]
    assert ids and np.allclose(colors,scene['stair_rgba'],atol=1e-6), str(dict(ids=ids,colors=colors.tolist(),names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(model.ngeom)]))
    digest=hashlib.sha256()
    for name in ('geom_pos','geom_quat','geom_size','geom_friction','geom_type','geom_contype','geom_conaffinity','body_mass','body_inertia','dof_damping'):
        digest.update(name.encode());digest.update(np.asarray(getattr(model,name)).tobytes())
    return dict(stair_count=len(ids),requested_rgba=scene['stair_rgba'],compiled_rgba=colors.tolist(),
                physics_sha256=digest.hexdigest(),color_applied='MuJoCo stair geom RGBA and renderer geometry color, before both cameras render',
                floor_unchanged=True,depth_input=False)
