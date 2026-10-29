"""Independent MuJoCo raycasts verify training labels and real unbridged gaps."""
from pathlib import Path
import json,numpy as np,torch,mujoco
from .terrain import make_tiles,PrimitiveHeights,atlas_class,FAMILIES

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--convex64',action='store_true');args=parser.parse_args()
    tiles=make_tiles(93157)
    if args.convex64:
        for tile in tiles:
            for solid in tile.get('solids',[]):
                if solid['shape']=='cylinder':solid['collision_sides']=64
    query=PrimitiveHeights(tiles,'cpu')
    spec=mujoco.MjSpec();spec.worldbody.add_body(name='terrain')
    atlas_class()(rows=(),root='',random_tiles=tuple(tiles)).function(0.,spec,np.random.default_rng(1))
    model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    origins=torch.tensor([t['origin'] for t in tiles]);rng=np.random.default_rng(7123)
    errors=[];gaps=0;round_corners=0;mismatches=[]
    for i,t in enumerate(tiles):
        if t.get('family','stairs')=='stairs':continue
        local=rng.uniform([-.3,-.75],[t['goal']+.3,.75],(80,2))
        extra=[]
        for s in t['solids']:
            extra.append([s['x']+(.000013 if args.convex64 else 0),s['y']+(.000021 if args.convex64 else 0)])
            if s['shape']=='cylinder':
                extra.append([s['x']+.85*s['hx'],s['y']+.85*s['hy']]);round_corners+=1
        for a,b in t['gaps']:
            extra.append([(a+b)/2,0]);gaps+=1
        local=np.concatenate((local,extra));world=local+np.array(t['origin'][:2])
        analytic=query.query(torch.tensor(world[None],dtype=torch.float64),torch.tensor([i]),origins)[0][0].numpy()
        for point,height in zip(world,analytic):
            gid=np.array([-1],dtype=np.int32)
            distance=mujoco.mj_ray(model,data,np.r_[point,5.],np.array([0.,0.,-1.]),None,1,-1,gid)
            assert distance>=0
            errors.append(abs((5.-distance)-height))
            if errors[-1]>2e-5:
                mismatches.append(dict(tile=i,point=point.tolist(),analytic=float(height),physics=5.-distance,geom=int(gid[0])))
        for a,b in t['gaps']:
            point=np.array([[(a+b)/2+t['origin'][0],t['origin'][1]]])
            height=query.query(torch.tensor(point[None]),torch.tensor([i]),origins)[0]
            assert float(height)==0.,(i,a,b,height)
    if mismatches:
        Path('records/stones_20261025/geometry_mismatches.json').write_text(json.dumps(mismatches,indent=2))
        print(mismatches[:5],flush=True)
    assert max(errors)<2e-5,max(errors)
    report=dict(cases=len(errors),maximum_height_error_m=float(max(errors)),real_gap_midpoints=gaps,
                circular_corner_checks=round_corners,levels=6,physical_geom_types=['BOX','MESH64' if args.convex64 else 'CYLINDER'],circular_boundary_max_error_m=float(.34*(1-np.cos(np.pi/64))) if args.convex64 else 0.,
                seed=93157,training_seed=137,no_hidden_gap_bridge=True)
    path=Path('records/stones_20261025/geometry_validation_convex64.json' if args.convex64 else 'records/stones_20261025/geometry_validation.json');path.write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
