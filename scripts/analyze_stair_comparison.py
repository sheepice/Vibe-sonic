"""Offline analysis of retained real preview/control log and local MuJoCo captures."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim2sim import scene
import mujoco

OUT = ROOT/'records/g1_sim_compare_20261029'
REAL = ROOT/'records/g1_recordings/trial_20261028_235905'

def quant(x):
    return dict(zip(['min','median','p99','max'], np.quantile(x,[0,.5,.99,1]).tolist()))

def fk(model, qs, quats):
    data = mujoco.MjData(model)
    bodies=[model.body(s+'_ankle_roll_link').id for s in ['left','right']]
    ankles=[];toes=[]
    for q,quat in zip(qs,quats):
        data.qpos[:3]=0
        data.qpos[3:7]=quat
        data.qpos[7:36]=q
        mujoco.mj_kinematics(model,data)
        ankles.append(np.array([data.xpos[b].copy() for b in bodies]))
        # Same representative toe point for both feet, NOT measured ground clearance.
        toes.append(np.array([data.xpos[b]+data.xmat[b].reshape(3,3)@np.array([.132,0,-.025]) for b in bodies]))
    return np.array(ankles), np.array(toes)

def attn_stats(frames):
    a=np.array([f['attention'] for f in frames if f.get('attention') is not None])
    entropy=-np.sum(a*np.log(np.maximum(a,1e-30)),axis=-1)/np.log(28)
    return dict(samples=len(a), entropy_median=np.median(entropy,axis=0).tolist(),
        peak_median=np.median(a.max(-1),axis=0).tolist(),
        top_slot_counts=[np.bincount(x,minlength=28).tolist() for x in a.argmax(-1).T])

def run():
    rows=[json.loads(l) for l in (REAL/'control.jsonl').open()]
    began=next(r['monotonic'] for r in rows if r['event']=='policy_started')
    ticks=[r for r in rows if r['event']=='control_tick']
    preview=json.loads((REAL/'preview_checked_20261029/preview.json').read_text())
    manifest=preview['metadata']['manifest']
    model=scene.make_model(manifest['robot'],manifest['camera'],scene.terrain('up',.15,3,tread=.30))
    t=np.array([r['monotonic']-began for r in ticks])
    q=np.array([r['q'] for r in ticks]);target=np.array([r['q_target'] for r in ticks]);quat=np.array([r['quat'] for r in ticks])
    rpy=Rotation.from_quat(quat[:,[1,2,3,0]]).as_euler('xyz')
    rpy[:,2]=np.unwrap(rpy[:,2]);rpy[:,2]-=rpy[0,2]
    feet,toes=fk(model,q,quat)
    command=np.array([r['speed_command'] for r in ticks])
    image_age=np.array([r['image_age_s'] for r in ticks])
    np.savez_compressed(OUT/'real_telemetry.npz',time=t,q=q,target=target,quat=quat,rpy=rpy,
        feet=feet,toes=toes,command=command,image_age=image_age)
    summary=dict(policy_started_monotonic=began,control_ticks=len(ticks),duration_s=t[-1],
        positive_command_start_s=float(t[command>1e-6][0]),positive_command_end_s=float(t[command>1e-6][-1]),
        positive_command_duration_s=float(np.sum(command>1e-6)*.02),
        full_speed_start_s=float(t[command>.399][0]),full_speed_end_s=float(t[command>.399][-1]),
        integrated_command_m=float(np.trapezoid(command,t)),
        yaw_command_abs_max=float(max(abs(r['yaw_rate_command']) for r in ticks)),
        image_age_s=quant(image_age),worker_s=quant([r['worker_s'] for r in ticks]),
        rpy_degrees_min=np.rad2deg(rpy).min(0).tolist(),rpy_degrees_max=np.rad2deg(rpy).max(0).tolist(),
        real_attention=attn_stats(preview['frames']),
        events=[r for r in rows if r['event'] in ['stopped','health_fault','damping_hold_exit']],
        windows={})
    for name,lo,hi in [('idle',5,18),('moving',19,25),('stopping',25,29),('late_zero',30,44.9)]:
        m=(t>=lo)&(t<hi)
        summary['windows'][name]=dict(time_range=[lo,hi],
            ankle_target_error_median_rad=np.median(abs(target[m]-q[m]),axis=0)[[4,10]].tolist(),
            knee_median_rad=np.median(q[m],axis=0)[[3,9]].tolist(),
            right_minus_left_ankle_height_m=quant(feet[m,1,2]-feet[m,0,2]),
            right_minus_left_toe_height_m=quant(toes[m,1,2]-toes[m,0,2]),
            yaw_relative_range_deg=np.rad2deg(rpy[m,2])[[np.argmin(rpy[m,2]),np.argmax(rpy[m,2])]].tolist())
    simulations={}
    for d in sorted(OUT.glob('sim_*/summary.json')):
        folder=d.parent;v=np.load(folder/'telemetry.npz');simq=v['q'];simquat=v['quat']
        srpy=Rotation.from_quat(simquat[:,[1,2,3,0]]).as_euler('xyz')
        sf,st=fk(model,simq,simquat)
        traj=json.loads((folder/'trajectory.json').read_text())
        scmd=np.array([r['speed_command'] for r in traj])
        np.savez_compressed(folder/'analysis.npz',time=v['time'],rpy=srpy,feet=sf,toes=st,command=scmd)
        m=v['time']>v['time'][-1]-2
        p=json.loads((folder/'preview/preview.json').read_text())
        simulations[folder.name]=dict(summary=json.loads(d.read_text()),
            late_knee_median_rad=np.median(simq[m],axis=0)[[3,9]].tolist(),
            late_right_minus_left_ankle_height_m=quant(sf[m,1,2]-sf[m,0,2]),
            attention=attn_stats(p['frames']))
    summary['simulations']=simulations
    (OUT/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['simulations','real_attention']},indent=2))
    for k,v in simulations.items():print(k,json.dumps({n:vv for n,vv in v.items() if n not in ['summary','attention']}))

if __name__=='__main__':run()
