"""Actual MuJoCo dynamics driven by the unmodified exported ONNX models."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

import time
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation

from .scene import ROOT, make_model, ground, terrain
from .vendor.runtime import Runtime
from .vendor.reference import OnlineReference
from .vendor.observations import RobotState, ObservationBuilder


class Simulation:
    def __init__(self, bundle=None):
        self.runtime = Runtime(bundle or ROOT / "models/vggt_hierarchical_rgb64")
        self.reference = OnlineReference(self.runtime.root / "planner.onnx", self.runtime.contract)
        self.runtime.warmup()
        self.rgb_renderer = self.viewer = None
        self.reset()

    def reset(self, kind="up", height=.15, count=6, speed=.4, seed=42, *, tread=.30, start=1.5):
        self.close_renderers()
        self.tile = terrain(kind, height, count, tread=tread, start=start)
        self.model = make_model(self.runtime.contract, self.runtime.manifest["camera"], self.tile)
        self.data = mujoco.MjData(self.model)
        self.qids = np.array([self.model.joint(n).qposadr[0] for n in self.runtime.contract["joint_names"]])
        self.vids = np.array([self.model.joint(n).dofadr[0] for n in self.runtime.contract["joint_names"]])
        assert np.array_equal(self.qids, np.arange(7,36))
        self.data.qpos[:3] = [0,0,float(ground(0,self.tile))+self.runtime.contract["root_height"]]
        self.data.qpos[3:7] = [1,0,0,0]
        self.data.qpos[self.qids] = self.runtime.default
        self.data.ctrl[:] = self.runtime.default
        mujoco.mj_forward(self.model, self.data)
        self.reference = OnlineReference(self.runtime.root / "planner.onnx", self.runtime.contract,
                                         seed=seed, session_override=self.reference.session)
        self.builder = ObservationBuilder(self.runtime.contract, self.runtime.patch_shape)
        height, width, _ = self.runtime.image_shape
        self.rgb_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.viewer = mujoco.Renderer(self.model, height=540, width=960)
        self.render_option = mujoco.MjvOption()
        self.render_option.geomgroup[:] = [1,0,1,0,0,0]
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth = 125
        self.camera.elevation = -23
        self.camera.distance = 7.0
        self.step_count = 0
        self.speed = 0.
        self.yaw_rate = 0.
        self.heading = 0.
        self.requested_speed = speed
        self.fall = False
        self.reached = False
        self.trace = []
        self.qpos_trace = []
        self.rgb = self.render_head()
        self.vision = dict(source_step=None, time=0., attention=None, feature_norm=None,
                           width=width, height=height)
        self.visual_inputs = None
        self.metrics = self.measure()
        return self.metrics

    def close_renderers(self):
        for renderer in (self.rgb_renderer, self.viewer):
            if renderer is not None:
                renderer.close()

    def render_head(self):
        self.rgb_renderer.update_scene(self.data, camera="head_rgb", scene_option=self.render_option)
        self.rgb_renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        return self.rgb_renderer.render().copy()

    def measure(self):
        p = self.data.qpos[:3]
        upright = float(Rotation.from_quat(self.data.qpos[[4,5,6,3]]).as_matrix()[2,2])
        clearance = float(p[2]-(ground(p[0],self.tile) if abs(p[1])<=2 else 0.))
        feet = [self.data.xpos[self.model.body(s+"_ankle_roll_link").id].copy() for s in ("left","right")]
        contacts = []
        for c in self.data.contact:
            g1,g2 = [self.model.geom(int(g)).name or "" for g in (c.geom1,c.geom2)]
            if g1.startswith("stair_") or g2.startswith("stair_"):
                contacts.append([g1,g2])
        finite = bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
        self.fall = self.fall or not finite or upright < .45 or clearance < .32
        self.reached = bool(self.reached or (min(f[0] for f in feet) > self.tile["goal"] and not self.fall))
        return dict(t=round(self.step_count*.02,3), position=p.tolist(), speed_command=float(self.speed),
                    heading_command=float(self.heading), yaw_rate_command=float(self.yaw_rate),
                    yaw=float(np.arctan2(self.data.xmat[1,3],self.data.xmat[1,0])),
                    actual_vx=float(self.data.qvel[0]), upright=upright, clearance=clearance,
                    actual_forward_speed=float(self.data.qvel[:3] @ self.data.xmat[1].reshape(3,3)[:,0]),
                    feet=[f.tolist() for f in feet], stair_contacts=contacts,
                    fall=self.fall, reached=self.reached, goal=self.tile["goal"], terrain=self.tile)

    def step(self, requested_speed, requested_yaw_rate=0.):
        if not np.isfinite([requested_speed, requested_yaw_rate]).all():
            raise ValueError("Commands must be finite")
        if not -.3 <= requested_speed <= .7 or abs(requested_yaw_rate) > .5:
            raise ValueError("Movement command out of range")
        if self.fall:
            return self.metrics
        before = time.perf_counter()
        elapsed = self.step_count * .02
        self.speed += float(np.clip(requested_speed-self.speed, -.01, .01))
        self.yaw_rate += float(np.clip(requested_yaw_rate-self.yaw_rate, -.02, .02))
        self.heading += self.yaw_rate * .02
        if self.step_count % 10 == 0:
            self.reference.replan(elapsed, self.speed, self.heading)
        ref = self.reference.sample(elapsed)
        self.rgb = self.render_head()
        patches, cls = self.runtime.encode(self.rgb[...,::-1].copy())
        state = RobotState(self.data.qpos[self.qids].astype(np.float32),
                           self.data.qvel[self.vids].astype(np.float32),
                           self.data.qpos[3:7].astype(np.float32),
                           self.data.sensor("imu_ang_vel").data.astype(np.float32), elapsed)
        obs = self.builder.build(state, ref, patches, cls, self.speed,
                                 heading=self.heading, yaw_rate=self.yaw_rate)
        action, target, _, attention = self.runtime.act(obs, with_attention=True)
        # Snapshot queries from this exact image/observation before histories advance.
        self.visual_inputs = {k: obs[k].copy() for k in
                              ('kv_tokens__img_tokens', 'q_task_cmd', 'q_proprio', 'q_cls')}
        self.vision = dict(source_step=self.step_count, time=elapsed,
                           attention=attention.tolist(), feature_norm=np.linalg.norm(patches,axis=-1).tolist(),
                           width=self.rgb.shape[1], height=self.rgb.shape[0])
        self.builder.accept_action(action)
        self.data.ctrl[:] = target
        # Native position servos include PD derivatives in implicit integration,
        # and enforce the exported actuator torque limits at each physics tick.
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.step_count += 1
        self.metrics = self.measure()
        self.metrics["compute_ms"] = 1000*(time.perf_counter()-before)
        self.trace.append(self.metrics.copy())
        self.qpos_trace.append(self.data.qpos.copy())
        return self.metrics

    def render(self, follow=True):
        if follow:
            self.camera.lookat[:] = [self.data.qpos[0]+1.1, self.data.qpos[1], max(.5,self.data.qpos[2]-.15)]
        self.viewer.update_scene(self.data, camera=self.camera, scene_option=self.render_option)
        return self.viewer.render().copy()
