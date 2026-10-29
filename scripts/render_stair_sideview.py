"""Replay saved simulation states from a closer side camera, without re-running policy."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sim2sim.scene import make_model
OUT=ROOT/'records/g1_sim_compare_20261029'
bundle=json.loads((ROOT/'models/vggt_hierarchical_rgb64/bundle.json').read_text())
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
for summary_path in sorted(OUT.glob('sim_*/summary.json')):
    folder=summary_path.parent;summary=json.loads(summary_path.read_text())
    output=folder/'external_side.mp4'
    if output.exists():continue
    v=np.load(folder/'telemetry.npz');traj=json.loads((folder/'trajectory.json').read_text())
    model=make_model(bundle['robot'],bundle['camera'],summary['terrain']);data=mujoco.MjData(model)
    opt=mujoco.MjvOption();opt.geomgroup[:]=[1,0,1,0,0,0]
    cam=mujoco.MjvCamera();cam.azimuth=90;cam.elevation=-15;cam.distance=3.2
    movie=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','960x540','-r','10',
        '-i','-','-an','-c:v','libx264','-crf','22','-preset','veryfast','-pix_fmt','yuv420p',str(output)],stdin=subprocess.PIPE)
    with mujoco.Renderer(model,height=540,width=960) as renderer:
        for i in range(0,len(v['time']),5):
            data.qpos[:3]=v['root'][i];data.qpos[3:7]=v['quat'][i];data.qpos[7:36]=v['q'][i]
            mujoco.mj_forward(model,data);cam.lookat[:]=data.qpos[:3]+[.1,0,-.2]
            renderer.update_scene(data,camera=cam,scene_option=opt);im=Image.fromarray(renderer.render())
            draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,58),fill=(20,27,34))
            draw.text((10,5),f"SIM state replay | t={v['time'][i]:.2f}s | command={traj[i]['speed_command']:.2f}m/s",fill='white',font=font)
            draw.text((10,30),'Riser 15cm / tread 30cm | closer side view | same saved trajectory',fill='#b8d0df',font=font)
            movie.stdin.write(im.tobytes())
            if i==len(v['time'])//5*5-5:im.save(folder/'side_final.jpg',quality=90)
    movie.stdin.close();assert movie.wait()==0
    print(output)
