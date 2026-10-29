"""Check measured visual/collision geometry alignment before importing a scene.

Input NPZ: visual_xyz[N,3], collision_xyz[N,3], optionally measured_pixels[N,2].
These must be independently matched landmarks, not two copies of one mesh.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def fit_similarity(source,target):
    source,target=np.asarray(source,float),np.asarray(target,float)
    if source.shape!=target.shape or source.ndim!=2 or source.shape[1]!=3 or len(source)<4:
        raise ValueError('At least four paired 3D landmarks required')
    if not np.isfinite(source).all() or not np.isfinite(target).all(): raise ValueError('Nonfinite landmarks')
    a,b=source-source.mean(0),target-target.mean(0)
    if np.linalg.matrix_rank(a)<2 or np.linalg.matrix_rank(b)<2: raise ValueError('Landmarks are collinear')
    u,s,vt=np.linalg.svd(b.T@a/len(a)); sign=np.eye(3); sign[-1,-1]=np.linalg.det(u@vt)
    rotation=u@sign@vt; scale=float(np.trace(np.diag(s)@sign)/(a*a).sum()*len(a))
    translation=target.mean(0)-scale*rotation@source.mean(0)
    aligned=scale*source@rotation.T+translation
    return scale,rotation,translation,aligned


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--points',required=True); p.add_argument('--output',required=True)
    p.add_argument('--camera',help='JSON: world_to_optical 4x4; fx,fy,cx,cy. Optical coordinates x right,y down,z forward.')
    p.add_argument('--rgb'); p.add_argument('--max-rmse-m',type=float,default=.02)
    args=p.parse_args()
    from .contracts import write_json
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    points=np.load(args.points); scale,rotation,translation,aligned=fit_similarity(points['visual_xyz'],points['collision_xyz'])
    error=np.linalg.norm(aligned-points['collision_xyz'],axis=1)
    report=dict(scale=scale,rotation=rotation.tolist(),translation_m=translation.tolist(),
        rmse_m=float(np.sqrt((error*error).mean())),maximum_m=float(error.max()),landmarks=len(error),
        in_sample_fit=True,independent_holdout_required=True,
        note='A similarity fit checks scale/pose only; it does not certify dense collision mesh accuracy or camera latency.')
    if 'holdout_visual_xyz' in points:
        h=scale*points['holdout_visual_xyz']@rotation.T+translation
        e=np.linalg.norm(h-points['holdout_collision_xyz'],axis=1)
        report['heldout_rmse_m']=float(np.sqrt((e*e).mean())); report['heldout_maximum_m']=float(e.max())
    if args.camera:
        camera=json.loads(Path(args.camera).read_text()); matrix=np.asarray(camera['world_to_optical'])
        if matrix.shape!=(4,4): raise ValueError('Expected 4x4 world_to_optical')
        xyz=aligned@matrix[:3,:3].T+matrix[:3,3]; visible=xyz[:,2]>0
        uv=np.stack((camera['fx']*xyz[:,0]/xyz[:,2]+camera['cx'],camera['fy']*xyz[:,1]/xyz[:,2]+camera['cy']),-1)
        if 'measured_pixels' in points:
            report['reprojection_rmse_px']=float(np.sqrt(((uv[visible]-points['measured_pixels'][visible])**2).sum(-1).mean()))
        if args.rgb:
            from PIL import Image,ImageDraw
            im=Image.open(args.rgb).convert('RGB'); draw=ImageDraw.Draw(im)
            for i,(x,y) in enumerate(uv):
                if visible[i] and 0<=x<im.width and 0<=y<im.height:
                    draw.ellipse((x-3,y-3,x+3,y+3),fill=(0,255,0)); draw.text((x+4,y),str(i),fill='yellow')
            im.save(out/'projection.png')
    report['within_requested_fit_tolerance']=report['rmse_m']<=args.max_rmse_m
    write_json(out/'alignment.json',report); print(json.dumps(report,indent=2))
    if not report['within_requested_fit_tolerance']: raise SystemExit('Alignment exceeds tolerance; inspect measurements/mesh before training')


if __name__=='__main__': main()
