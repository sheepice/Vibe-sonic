"""Local web controls and live MuJoCo frames. The simulation thread owns OpenGL."""
import argparse
import base64
from contextlib import asynccontextmanager
import io
import json
import queue
import threading
import time
import traceback
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from PIL import Image
import uvicorn

from .engine import Simulation
from .scene import ROOT
from .validation import evaluate
from .sensitivity import AsyncSensitivity


class Controls(BaseModel):
    action: str
    kind: str = "up"
    height: float = Field(default=.15, ge=.02, le=.20)
    count: int = Field(default=6,ge=1,le=8)
    tread: float = Field(default=.30,ge=.10,le=.60)
    speed: float = Field(default=.4,ge=0,le=.7)
    forward: Literal[-1, 0, 1] = 1
    turn: Literal[-1, 0, 1] = 0
    azimuth: float | None = None
    elevation: float | None = None
    distance: float | None = None


class Worker:
    def __init__(self, bundle=None):
        self.bundle = bundle
        self.commands=queue.Queue()
        self.status={"ready":False,"message":"加载 ONNX 模型…"}
        self.frame=self.rgb=b""
        self.vision={"image":None}
        self.sensitivity={"status":"loading","image":None}
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,name="mujoco",daemon=True)

    @staticmethod
    def jpeg(array,quality=82):
        out=io.BytesIO()
        Image.fromarray(array).save(out,format="JPEG",quality=quality)
        return out.getvalue()

    def run(self):
        sim=probe=None
        try:
            sim=Simulation(self.bundle)
            probe=AsyncSensitivity(sim.runtime.root)
            running=auto=walking=False
            reached_at=None
            validation=None
            report_file=None
            speed=.4
            forward=turn=0
            heartbeat=0.
            last_render=0.
            next_tick=time.monotonic()
            while not self.stop.is_set():
                now=time.monotonic()
                while not self.commands.empty():
                    c=self.commands.get_nowait()
                    if c.action=="reset":
                        sim.reset(c.kind,c.height,c.count,c.speed,tread=c.tread)
                        probe.reset()
                        speed=c.speed
                        running=walking=auto=False
                        forward=turn=0
                        reached_at=validation=report_file=None
                    elif c.action in {"walk", "drive"}:
                        forward,turn=(1,0) if c.action=="walk" else (c.forward,c.turn)
                        walking=bool(forward or turn)
                        running=True
                        auto=False
                        speed=c.speed
                        heartbeat=now
                        validation=report_file=None
                    elif c.action=="heartbeat":
                        heartbeat=now
                    elif c.action=="stop":
                        walking=auto=False
                        forward=turn=0
                    elif c.action=="pause":
                        running=False
                        walking=auto=False
                        forward=turn=0
                    elif c.action=="resume":
                        running=True
                    elif c.action=="auto":
                        auto=running=True
                        walking=False
                        forward=turn=0
                        speed=c.speed
                    elif c.action=="camera":
                        if c.azimuth is not None: sim.camera.azimuth=c.azimuth
                        if c.elevation is not None: sim.camera.elevation=max(-85,min(-3,c.elevation))
                        if c.distance is not None: sim.camera.distance=max(2,min(14,c.distance))
                    if c.action in {"reset", "resume", "auto"}:
                        next_tick=now
                if walking and now-heartbeat>.6:
                    walking=False
                    forward=turn=0
                if running and not sim.fall and now>=next_tick:
                    request=(min(speed,.25) if forward<0 else speed)*forward if walking else 0.
                    if auto and sim.step_count>=100 and not sim.reached:
                        request=speed
                    sim.step(request, .45*turn if walking else 0.)
                    if auto and sim.reached and reached_at is None:
                        reached_at=sim.metrics["t"]
                    if auto and (sim.fall or (reached_at is not None and sim.metrics["t"]>=reached_at+4)):
                        validation=evaluate(sim.trace,sim.tile,sim.model)
                        report_file=f"browser_auto_{time.time_ns()}.json"
                        (ROOT/"records").mkdir(exist_ok=True)
                        (ROOT/"records"/report_file).write_text(json.dumps(dict(
                            source="live_browser",initial_state="stationary_default_pose" if sim.trace[0]["t"]==.02 else "continued",
                            checkpoint=sim.runtime.manifest['checkpoint'],
                            checkpoint_sha256=sim.runtime.manifest['checkpoint_sha256'],
                            terrain=sim.tile,bundle_files=sim.runtime.manifest["files"],validation=validation,
                            passed=validation["passed"],fall=sim.fall,reached=sim.reached,final=sim.metrics,trace=sim.trace
                        ),ensure_ascii=False,indent=2)+"\n")
                        running=auto=False
                    next_tick+=.02
                    if next_tick<now-.1:
                        next_tick=now
                probe.update(sim)
                self.sensitivity=probe.packet
                if now-last_render>=.08:
                    self.frame=self.jpeg(sim.render())
                    self.rgb=self.jpeg(sim.rgb,95)
                    # Publish image + features from the same completed policy
                    # call as one immutable packet, avoiding cross-request races.
                    self.vision=dict(sim.vision,image='data:image/jpeg;base64,'+base64.b64encode(self.rgb).decode('ascii'))
                    last_render=now
                    self.status=dict(sim.metrics,ready=True,running=running,walking=walking,auto=auto,
                                     model=dict(bundle=str(sim.runtime.root),
                                         checkpoint=sim.runtime.manifest['checkpoint'].split('/')[-1],
                                         family=sim.runtime.manifest.get('model_family','velocity'),
                                         iteration=sim.runtime.manifest.get('training_iteration'),
                                         image_shape=list(sim.runtime.image_shape),
                                         patch_shape=list(sim.runtime.patch_shape),
                                         files=sim.runtime.manifest['files']),
                                     forward=forward,turn=turn,
                                     validation=validation,report_file=report_file,
                                     camera=dict(azimuth=sim.camera.azimuth,elevation=sim.camera.elevation,distance=sim.camera.distance))
                self.stop.wait(.001 if running else .02)
        except Exception as exc:
            traceback.print_exc()
            self.status={"ready":False,"error":str(exc)}
        finally:
            if probe: probe.close()
            if sim: sim.close_renderers()


worker=Worker()


@asynccontextmanager
async def lifespan(app):
    worker.thread.start()
    yield
    worker.stop.set()
    worker.thread.join(timeout=5)


app=FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(ROOT/"web/index.html")


@app.get("/api/state")
def state():
    return worker.status


@app.get("/frame.jpg")
def frame():
    return Response(worker.frame,media_type="image/jpeg",headers={"Cache-Control":"no-store"})


@app.get("/rgb.jpg")
def rgb():
    return Response(worker.rgb,media_type="image/jpeg",headers={"Cache-Control":"no-store"})


@app.get("/api/vision")
def vision():
    return worker.vision


@app.get("/api/sensitivity")
def sensitivity():
    return worker.sensitivity


@app.get("/heatmap.js")
def heatmap_renderer():
    return FileResponse(ROOT/"hardware/src/vibe_grail_joint/deploy/heatmap.js",
                        media_type="text/javascript",headers={"Cache-Control":"no-store"})


@app.post("/api/control")
def control(c:Controls):
    if c.action not in {"reset","walk","drive","heartbeat","stop","pause","resume","auto","camera"}:
        raise HTTPException(400,"Unknown control")
    if c.kind not in {"flat","up","down","up_down"}:
        raise HTTPException(400,"Unknown terrain")
    if not worker.status.get("ready"):
        raise HTTPException(503,"Simulation is loading or has failed")
    worker.commands.put(c)
    return {"queued":True}


@app.get("/api/reports")
def reports():
    result=[]
    for path in sorted((ROOT/"records").glob("acceptance_*.json")):
        report=json.loads(path.read_text())
        if report.get('bundle_files') != worker.status.get('model',{}).get('files'):
            continue
        result.append({k:report[k] for k in ("terrain","passed","fall","reached","final")}|{"file":path.name})
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--host",default="127.0.0.1")
    p.add_argument("--port",type=int,default=8080)
    p.add_argument('--bundle', help='Override the default RGB64 hierarchical bundle')
    args=p.parse_args()
    worker.bundle = args.bundle
    uvicorn.run(app,host=args.host,port=args.port,log_level="info")


if __name__=="__main__":
    main()
