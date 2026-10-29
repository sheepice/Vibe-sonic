"""Real GRAIL heightfields, rigidly aligned, with a documented exit landing."""
from dataclasses import dataclass,fields
import numpy as np
import torch
import mujoco
from ..terrain import JointAtlasCfg,STAIR_RGBA
from ..velocity.recovery_terrain import ProfileAtlasCfg
from ..velocity.eval_profile_command import ProfileEvaluationCommand,ProfileEvaluationCommandCfg
from mjlab.terrains.terrain_generator import TerrainGeometry,TerrainOutput


def field_arrays(scene):
    f=np.load(scene['heightfield']['path'])
    return np.rot90(f['heights'],scene['heightfield']['rotation_quarters']).copy(),float(f['resolution'])


@dataclass(kw_only=True)
class GrailAtlasCfg(ProfileAtlasCfg):
    evaluation_scene: dict=None
    def function(self,difficulty,spec,rng):
        result=JointAtlasCfg.function(self,difficulty,spec,rng)
        scene=self.evaluation_scene;field=scene['heightfield'];h,res=field_arrays(scene)
        ox,oy,_=scene['tile']['origin'];half=(np.array(h.shape[::-1])-1)*res/2;top=float(h.max())
        spec.add_hfield(name='eval_grail',nrow=h.shape[0],ncol=h.shape[1],size=(*half,top,.1),userdata=(h/top).ravel().tolist())
        body=spec.body('terrain')
        geom=body.add_geom(name='eval_grail_surface',type=mujoco.mjtGeom.mjGEOM_HFIELD,hfieldname='eval_grail',pos=(ox+field['xy_min'][0]+half[0],oy+field['xy_min'][1]+half[1],0.),rgba=STAIR_RGBA,friction=(.8,.005,.0001))
        result.geometries.append(TerrainGeometry(geom=geom,color=STAIR_RGBA))
        landing=scene['landing'];a,b=landing['x_range'];z=landing['height']
        geom=body.add_geom(name='eval_exit_landing',type=mujoco.mjtGeom.mjGEOM_BOX,size=((b-a)/2,landing['half_width'],z/2),pos=(ox+(a+b)/2,oy,z/2),rgba=STAIR_RGBA,friction=(.8,.005,.0001))
        result.geometries.append(TerrainGeometry(geom=geom,color=STAIR_RGBA))
        return TerrainOutput(origin=result.origin,geometries=result.geometries)


class GrailCommand(ProfileEvaluationCommand):
    def _init_task(self):
        super()._init_task()
        h,res=field_arrays(self.cfg.evaluation_scene)
        self.field=torch.as_tensor(h,device=self.device);self.field_resolution=res
    def ground_height(self,xy,env_ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if env_ids is None else env_ids
        expand=(slice(None),)+(None,)*(xy.ndim-2)
        local=xy-self.tile_origins[self.tile[ids],:2][expand]
        scene=self.cfg.evaluation_scene;field=scene['heightfield'];h=self.field
        gx=(local[...,0]-field['xy_min'][0])/self.field_resolution
        gy=(local[...,1]-field['xy_min'][1])/self.field_resolution
        ix=gx.floor().long().clamp(0,h.shape[1]-2);iy=gy.floor().long().clamp(0,h.shape[0]-2)
        u=(gx-ix).clamp(0,1);v=(gy-iy).clamp(0,1)
        # MuJoCo triangulates grid cells along the bottom-left/top-right diagonal.
        h00=h[iy,ix];h10=h[iy,ix+1];h01=h[iy+1,ix];h11=h[iy+1,ix+1]
        height=torch.where(u>=v,h00+u*(h10-h00)+v*(h11-h10),h00+v*(h01-h00)+u*(h11-h01))
        valid=(gx>=0)&(gy>=0)&(gx<=h.shape[1]-1)&(gy<=h.shape[0]-1)
        height=torch.where(valid,height,0.)
        landing=scene['landing'];a,b=landing['x_range']
        inside=(local[...,0]>=a)&(local[...,0]<=b)&(local[...,1].abs()<=landing['half_width'])
        return torch.maximum(height,torch.where(inside,landing['height'],0.))


@dataclass(kw_only=True)
class GrailCommandCfg(ProfileEvaluationCommandCfg):
    evaluation_scene:dict=None
    def build(self,env):return GrailCommand(self,env)


def configure_field(cfg,scene):
    old=cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    params={f.name:getattr(old,f.name) for f in fields(GrailAtlasCfg) if hasattr(old,f.name)}
    params.update(evaluation_scene=scene,rows=())
    cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']=GrailAtlasCfg(**params)
    old=cfg.commands['motion']
    params={f.name:getattr(old,f.name) for f in fields(GrailCommandCfg) if hasattr(old,f.name)}
    cfg.commands['motion']=GrailCommandCfg(**params,evaluation_scene=scene)
    return GrailCommandCfg


def audit_field(raw):
    c=raw.command_manager.get_term('motion');scene=c.cfg.evaluation_scene;h,res=field_arrays(scene)
    rng=np.random.default_rng(29117);lo=np.array(scene['heightfield']['xy_min']);hi=lo+res*(np.array(h.shape[::-1])-1)
    local=rng.uniform(lo,hi,(100,2));world=local+c.tile_origins[0,:2].cpu().numpy()
    ids=torch.zeros(100,dtype=torch.long,device=raw.device)
    predicted=c.ground_height(torch.tensor(world,dtype=torch.float32,device=raw.device),ids).cpu().numpy()
    model=raw.sim._mj_model;data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    group=np.ones(6,dtype=np.uint8);group[0]=0
    values=[]
    # Terrain geoms are group 0 in some mjlab versions: select by body instead.
    terrain=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'terrain')
    for x,y in world:
        point=np.array([x,y,10.]);direction=np.array([0.,0.,-1.]);distances=[]
        for geom in range(model.ngeom):
            if int(model.geom_bodyid[geom])!=terrain:continue
            kind=int(model.geom_type[geom])
            if kind==int(mujoco.mjtGeom.mjGEOM_HFIELD):
                d=mujoco.mj_rayHfield(model,data,geom,point,direction)
            else:
                d=mujoco.mju_rayGeom(data.geom_xpos[geom],data.geom_xmat[geom],model.geom_size[geom],point,direction,kind)
            if d>=0:distances.append(d)
        values.append(10-min(distances))
    error=np.abs(predicted-np.array(values))
    result=dict(samples=100,max_height_error_m=float(error.max()),mean_height_error_m=float(error.mean()),source='MuJoCo CPU terrain ray queries vs evaluator ground height',depth_sensor=False)
    if error.max()>.002:raise RuntimeError(f'Heightfield physics/query mismatch: {result}')
    return result
