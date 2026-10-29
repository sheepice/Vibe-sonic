"""Local-only stair trial with exact policy images/attention in a compact ROS 2 bag.

Records existing Simulation.step, unchanged actions, and optional 10 Hz JPEG
external views. Never imports the Unitree SDK or opens a physical camera.
"""
import argparse
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'hardware/src'))
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from sim2sim import engine,scene
from sim2sim.validation import evaluate
from vibe_grail_joint.deploy.recording import BagSink
from vibe_grail_joint.deploy.observations import RobotState
from vibe_grail_joint.deploy.visualize_bag import export, viewer_html


class CapturePolicy:
    def __init__(self,session):
        self.session=session
        self.last=None
    def __getattr__(self,name):
        return getattr(self.session,name)
    def run(self,names,feed):
        action,attention=self.session.run(['actions','attn'],feed)
        self.last=(feed,action[0].copy(),attention[0].copy())
        return [{'actions':action,'attn':attention}[n] for n in names]


def run(args):
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(out)
    if shutil.disk_usage(out.parent).free<3*1024**3:raise RuntimeError('Less than 3 GiB free; refusing capture')
    out.mkdir(parents=True)
    sim=engine.Simulation(ROOT/'models/vggt_hierarchical_rgb64')
    sim.reset('up',args.height,args.count,args.speed,args.seed,tread=args.tread,start=args.start)
    capture=CapturePolicy(sim.runtime.policy);sim.runtime.policy=capture
    epoch=time.time_ns()
    metadata=dict(schema='vibe.diagnostic_bag.v1',manifest=sim.runtime.manifest,
        wall_minus_monotonic_ns=epoch,raw_hz=0.,attention_queries=['q_task_cmd','q_proprio','q_cls'],
        source='MuJoCo simulation; no physical camera or motors',actual_input_images=True,
        raw_shape=None,depth_input=False,motor_transport=False,
        attention_meaning='Exported cross attention over spatially mixed patch slots; not causal saliency.')
    sink=BagSink(out/'bag',metadata)
    sink.string('/vibe/control/events',0.,dict(event='simulation_start',monotonic=0.,parameters=vars(args)))
    overview_dir=out/'overview';overview_dir.mkdir()
    movie=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x540',
        '-r','10','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','24','-pix_fmt','yuv420p',
        str(out/'external_view.mp4')],stdin=subprocess.PIPE)
    records=[];arrays={k:[] for k in ['time','q','dq','quat','target','action','attention','feet','root','ctrl_force']}
    reached_at=None;failure=None;first_contact=None;start=time.monotonic()
    try:
        for tick in range(round(args.seconds*50)):
            t=sim.step_count*.02
            q=sim.data.qpos[sim.qids].astype(np.float32).copy()
            dq=sim.data.qvel[sim.vids].astype(np.float32).copy()
            quat=sim.data.qpos[3:7].astype(np.float32).copy()
            gyro=sim.data.sensor('imu_ang_vel').data.astype(np.float32).copy()
            pre_root=sim.data.qpos[:3].copy()
            pre_feet=np.array([sim.data.xpos[sim.model.body(s+'_ankle_roll_link').id].copy() for s in ['left','right']])
            if tick%5==0:
                overview=sim.render();movie.stdin.write(overview.tobytes())
                im=Image.fromarray(overview);im.thumbnail((640,360));stream=io.BytesIO();im.save(stream,format='JPEG',quality=83)
                encoded=stream.getvalue();(overview_dir/f'{tick}.jpg').write_bytes(encoded)
                msg=sink.types['sensor_msgs/msg/CompressedImage'](sink.header(t,'world'),'jpeg',np.frombuffer(encoded,np.uint8))
                sink.write('/vibe/sim/overview/image/compressed',t,msg)
            requested=args.speed if tick>=100 and reached_at is None else 0.
            if args.command_seconds is not None and t>=2.+args.command_seconds:requested=0.
            m=sim.step(requested)
            feed,action,attention=capture.last
            target=sim.runtime.default+sim.runtime.scale*action
            image=sim.rgb[:,:,::-1].copy()
            sink.consume('camera',dict(captured=t,policy=image,frame_id=tick,device_time=t,raw_slot=None),None)
            sink.consume('features',dict(captured=t,frame_id=tick,features=(feed['kv_tokens__img_tokens'][0],feed['q_cls'][0])),None)
            state=RobotState(q,dq,quat,gyro,t)
            sink.consume('step',dict(at=t,state=state,step_id=tick,image_at=t,frame_id=tick,
                observations=feed,action=action,target=target,attention=attention,
                speed=sim.speed,heading=sim.heading,yaw_rate=sim.yaw_rate,reset=tick==0),None)
            compact={k:m[k] for k in ['t','position','actual_vx','actual_forward_speed','feet','upright','clearance','fall','reached']}
            compact.update(speed_command=sim.speed,stair_contacts=sorted(set(tuple(c) for c in m['stair_contacts'])),
                target_error_max_rad=float(np.max(abs(target-q))),
                image_at=t,frame_id=tick,policy_elapsed_s=t,
                rpy=Rotation.from_quat(quat[[1,2,3,0]]).as_euler('xyz').tolist())
            if compact['stair_contacts'] and first_contact is None:first_contact=t
            sink.string('/vibe/control/events',t,dict(event='control_tick',monotonic=t,lowcmd_enabled=False,**compact))
            records.append(compact)
            for name,value in dict(time=t,q=q,dq=dq,quat=quat,target=target,action=action,attention=attention,
                                   feet=pre_feet,root=pre_root,ctrl_force=sim.data.actuator_force.copy()).items():
                arrays[name].append(value)
            if tick%250==0:
                sink.checkpoint()
                print(json.dumps(dict(t=m['t'],x=m['position'][0],fall=m['fall'],reached=m['reached'])),flush=True)
                if shutil.disk_usage(out).free<2*1024**3:raise RuntimeError('Disk reserve reached')
            if m['reached'] and reached_at is None:reached_at=m['t']
            if m['fall'] or (reached_at is not None and m['t']>=reached_at+4.):break
    except BaseException as exc:
        failure=repr(exc)
        raise
    finally:
        movie.stdin.close();movie.wait()
        sink.string('/vibe/recording/status',sim.step_count*.02,dict(error=failure,limit_reached=False,counters={}))
        sink.close()
        np.savez_compressed(out/'telemetry.npz',**{k:np.asarray(v) for k,v in arrays.items()})
        validation=evaluate(sim.trace,sim.tile,sim.model)
        summary=dict(parameters=vars(args),terrain=sim.tile,first_stair_contact_s=first_contact,
            reached_at_s=reached_at,fall=sim.fall,validation=validation,frames=len(records),
            elapsed_s=sim.step_count*.02,wall_s=time.monotonic()-start,
            model_files=sim.runtime.manifest['files'],failure=failure,
            camera_hz=50,control_hz=50,physical_camera_opened=False,real_motor_commands=False)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (out/'trajectory.json').write_text(json.dumps(records,separators=(',',':'))+'\n')
        sim.close_renderers()
    result=export(out/'bag',out/'preview',fps=10.)
    preview=json.loads((out/'preview/preview.json').read_text())
    preview['metadata']['overview_hz']=10.
    for f in preview['frames']:
        view=overview_dir/f"{f['step']['step_id']}.jpg"
        if view.exists():f['raw']='../overview/'+view.name
    (out/'preview/preview.json').write_text(json.dumps(preview,ensure_ascii=False)+'\n')
    (out/'preview/index.html').write_text(viewer_html(preview))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--height',type=float,default=.15)
    p.add_argument('--tread',type=float,default=.30)
    p.add_argument('--count',type=int,default=3)
    p.add_argument('--speed',type=float,default=.4)
    p.add_argument('--start',type=float,default=1.5)
    p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--command-seconds',type=float)
    p.add_argument('--output',required=True)
    run(p.parse_args())
