"""Run a reproducible stationary-start physical traversal, retaining all evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from .engine import Simulation
from .validation import evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle', help='Model bundle directory; defaults to the imported RGB64 hierarchical model')
    p.add_argument("--terrain", choices=["flat","up","down","up_down"], default="up")
    p.add_argument("--height",type=float,default=.15)
    p.add_argument("--count",type=int,default=3)
    p.add_argument("--tread",type=float,default=.30)
    p.add_argument("--speed",type=float,default=.4)
    p.add_argument("--seconds",type=float,default=40)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",required=True)
    p.add_argument("--video",action="store_true")
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():
        p.error("Choose a new report path to preserve previous results")
    output.parent.mkdir(parents=True,exist_ok=True)
    sim=Simulation(args.bundle)
    sim.reset(args.terrain,args.height,args.count,args.speed,args.seed,tread=args.tread)
    video=None
    if args.video:
        video=subprocess.Popen(["ffmpeg","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","960x540",
              "-r","10","-i","-","-an","-c:v","libx264","-preset","veryfast","-crf","24",
              "-pix_fmt","yuv420p",str(output.with_suffix(".mp4"))],stdin=subprocess.PIPE)
    start=time.monotonic()
    reached_at=None
    try:
        for tick in range(round(args.seconds*50)):
            speed=args.speed if tick>=100 and reached_at is None else 0.
            m=sim.step(speed)
            if video and tick%5==0:
                video.stdin.write(sim.render().tobytes())
            if tick%100==0:
                print(json.dumps({k:m[k] for k in ("t","position","upright","clearance","fall","reached")}),flush=True)
            if m["reached"] and reached_at is None:
                reached_at=m["t"]
            if m["fall"] or (reached_at and m["t"]>=reached_at+4):
                break
    finally:
        if video:
            video.stdin.close()
            video.wait()
        validation=evaluate(sim.trace,sim.tile,sim.model)
        report=dict(parameters=vars(args),terrain=sim.tile,bundle_files=sim.runtime.manifest["files"],
                    checkpoint=sim.runtime.manifest['checkpoint'],checkpoint_sha256=sim.runtime.manifest['checkpoint_sha256'],
                    initial_state="stationary_default_pose",engine="MuJoCo native implicitfast, dt=0.005; policy=50Hz",
                    observation_inputs="RGB camera + IMU + joint state + online SONIC reference + velocity command",
                    execution="Training PD targets with native torque limits; no added hardware deployment guards",
                    fall=sim.fall,reached=sim.reached,reached_at=reached_at,final=sim.metrics,
                    passed=validation["passed"],validation=validation,
                    wall_seconds=time.monotonic()-start,trace=sim.trace)
        output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        np.savez_compressed(output.with_suffix(".npz"),qpos=np.asarray(sim.qpos_trace),dt=.02)
        sim.close_renderers()
        print(json.dumps({k:v for k,v in report.items() if k!="trace"},ensure_ascii=False,indent=2),flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__=="__main__":
    main()
