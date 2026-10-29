"""Shared physical and analytic primitives for raised boxes/cylinders and gaps."""
from dataclasses import dataclass
import hashlib,json
import numpy as np
import torch
from ..velocity.normal_stairs_geometry import normal_tiles

FAMILIES=('stairs','box_stones','cylinder_stones','stair_gap')

def box(x0,x1,y0,y1,top):
    assert x1>x0 and y1>y0 and top>0
    return dict(shape='box',x=(x0+x1)/2,y=(y0+y1)/2,hx=(x1-x0)/2,hy=(y1-y0)/2,top=float(top))

def cylinder(x,y,r,top):
    return dict(shape='cylinder',x=float(x),y=float(y),hx=float(r),hy=float(r),top=float(top))

def key(tile):
    physical={k:tile[k] for k in ('solids','edges','jumps','base') if k in tile}
    return hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest()

def make_tiles(seed,levels=6,variants=3):
    # Keep old 11 x variants x 4 stair indices for inherited replay/planner reset logic.
    tiles=normal_tiles(seed,11,variants)
    for t in tiles:t.update(family='stairs',course_level=t['level'])
    rng=np.random.default_rng(seed+41093)
    for level in range(levels):
        difficulty=level/max(1,levels-1)
        for variant in range(variants):
            for family in FAMILIES[1:]:
                gap=.04+.16*difficulty
                variation=0. if level<2 else (.02+.03*difficulty)
                top=.30
                solids=[];gaps=[]
                start=1.3+float(rng.uniform(-.1,.1));n=5
                if family!='stair_gap':
                    width=.68-.12*difficulty
                    length=width if family=='cylinder_stones' else .62-.06*difficulty
                    solids.append(box(-2,start,-1.2,1.2,top))
                    end=start;previous=top
                    for j in range(n):
                        g=gap if level<2 else gap*float(rng.uniform(.8,1.15))
                        gap_start=end;end+=g;gaps.append([gap_start,end])
                        height=top if level<2 else float(np.clip(previous+rng.uniform(-variation,variation),.22,.43))
                        y=0. if level<3 else float(rng.uniform(-.06,.06))
                        if family=='cylinder_stones':solids.append(cylinder(end+length/2,y,width/2,height))
                        else:solids.append(box(end,end+length,y-width/2,y+width/2,height))
                        end+=length;previous=height
                    gaps.append([end,end+gap]);end+=gap
                    solids.append(box(end,12,-1.2,1.2,previous))
                    goal=end+.75
                    edges=[];jumps=[];base=top
                else:
                    n=4;h=.08+.10*difficulty;tread=.36
                    # Upstairs, full-foot departure ledge, actual void, landing
                    # platform, downstairs, final floor. No hidden bridging box.
                    edges=[start+j*tread for j in range(n)]
                    for j in range(n-1):solids.append(box(edges[j],edges[j+1],-1.3,1.3,(j+1)*h))
                    ledge=edges[-1]+.80;landing=ledge+gap
                    solids.append(box(edges[-1],ledge,-1.3,1.3,n*h))
                    down=landing+1.25
                    solids.append(box(landing,down,-1.3,1.3,n*h))
                    for j in range(n-1):solids.append(box(down+j*tread,down+(j+1)*tread,-1.3,1.3,(n-1-j)*h))
                    end=down+(n-1)*tread;goal=end+.8;gaps=[[ledge,landing]]
                    # Profile used only by the legacy diagnostics. Actual heights
                    # always come from the primitives, including the void.
                    edges+= [ledge,landing]+[down+j*tread for j in range(n)]
                    jumps=[h]*n+[-n*h,n*h]+[-h]*n;base=0.
                i=len(tiles)
                tiles.append(dict(family=family,course_level=level,level=level,variant=variant,
                                  kind='up_down',n=n,height=max(s['top'] for s in solids),tread=.36,
                                  start=start,plateau=1.25,goal=goal,base=base,edges=edges,jumps=jumps,
                                  origin=[-30.-(i%12)*16.,-30.-(i//12)*10.,0.],
                                  solids=solids,gaps=gaps,gap_m=gap,height_delta_max_m=variation,
                                  approach_height_m=top if family!='stair_gap' else 0.))
    assert all(t['goal']<11 for t in tiles)
    return tiles

class PrimitiveHeights:
    def __init__(self,tiles,device):
        self.device=device
        count=max(1,max(len(t.get('solids',[])) for t in tiles))
        self.values=torch.zeros(len(tiles),count,6,device=device)
        self.valid=torch.zeros(len(tiles),count,device=device,dtype=torch.bool)
        self.polygonal=any(s.get('collision_sides',0)>2 for t in tiles for s in t.get('solids',[]))
        for i,t in enumerate(tiles):
            for j,s in enumerate(t.get('solids',[])):
                self.values[i,j]=torch.tensor([s['x'],s['y'],s['hx'],s['hy'],s['top'],s.get('collision_sides',1) if s['shape']=='cylinder' else 0],device=device)
                self.valid[i,j]=True
    def query(self,xy,tile,origins):
        # Broadcast primitive metadata over arbitrary sole/grid query dimensions.
        expand=(slice(None),)+(None,)*(xy.ndim-2)
        q=xy-origins[tile,:2][expand]
        p=self.values[tile][expand]
        dx=(q[...,0,None]-p[...,0]);dy=(q[...,1,None]-p[...,1])
        inside_box=(dx.abs()<=p[...,2])&(dy.abs()<=p[...,3])
        radius=p[...,2]
        if self.polygonal:
            sides=p[...,5].clamp_min(3)
            phi=torch.pi/sides
            delta=torch.remainder(torch.atan2(dy,dx),2*phi)-phi
            polygon_radius=radius*torch.cos(phi)/torch.cos(delta)
            radius=torch.where(p[...,5]>2,polygon_radius,radius)
        inside_round=dx.square()+dy.square()<=radius.square()
        inside=torch.where(p[...,5]>0,inside_round,inside_box)&self.valid[tile][expand]
        heights=torch.where(inside,p[...,4],0.)
        height,index=heights.max(-1)
        surface=torch.where(height>0,index,torch.full_like(index,-1))
        return height,surface

@dataclass(kw_only=True)
class _Unused:pass

# Import simulator only when task construction needs it; CPU tests use primitives.
def atlas_class():
    from ..velocity.recovery_terrain import ProfileAtlasCfg
    from mjlab.terrains.terrain_generator import TerrainGeometry
    from ..terrain import STAIR_RGBA
    import mujoco
    @dataclass(kw_only=True)
    class StoneAtlasCfg(ProfileAtlasCfg):
        def function(self,difficulty,spec,rng):
            # Retain paired GRAIL atlas and ordinary stairs, skip custom profiles.
            original=self.random_tiles
            self.random_tiles=tuple(t for t in original if 'solids' not in t)
            try:result=super().function(difficulty,spec,rng)
            finally:self.random_tiles=original
            body=spec.body('terrain')
            for i,t in enumerate(original):
                ox,oy,oz=t['origin']
                for j,s in enumerate(t.get('solids',[])):
                    round=s['shape']=='cylinder'
                    if round and s.get('collision_sides',0)>2:
                        count=s['collision_sides'];angles=np.arange(count)*2*np.pi/count
                        ring=np.stack((s['hx']*np.cos(angles),s['hx']*np.sin(angles)),axis=-1)
                        vertices=np.concatenate((np.c_[ring,np.full(count,-s['top']/2)],np.c_[ring,np.full(count,s['top']/2)]))
                        faces=[]
                        for k in range(count):
                            m=(k+1)%count;faces.extend(((k,m,count+m),(k,count+m,count+k)))
                        for k in range(1,count-1):faces.extend(((0,k+1,k),(count,count+k,count+k+1)))
                        mesh=spec.add_mesh(name=f'round_stone_{i}_{j}',uservert=vertices.flatten().tolist(),userface=np.asarray(faces).flatten().tolist())
                        geom=body.add_geom(name=f'stone_{i}_{j}',type=mujoco.mjtGeom.mjGEOM_MESH,
                            meshname=mesh.name,pos=(ox+s['x'],oy+s['y'],oz+s['top']/2),rgba=STAIR_RGBA,friction=(.8,.005,.0001))
                        result.geometries.append(TerrainGeometry(geom=geom,color=STAIR_RGBA))
                        continue
                    geom=body.add_geom(name=f'stone_{i}_{j}',
                        type=mujoco.mjtGeom.mjGEOM_CYLINDER if round else mujoco.mjtGeom.mjGEOM_BOX,
                        size=(s['hx'],s['top']/2,0) if round else (s['hx'],s['hy'],s['top']/2),
                        pos=(ox+s['x'],oy+s['y'],oz+s['top']/2),rgba=STAIR_RGBA,friction=(.8,.005,.0001))
                    result.geometries.append(TerrainGeometry(geom=geom,color=STAIR_RGBA))
            return result
    return StoneAtlasCfg
