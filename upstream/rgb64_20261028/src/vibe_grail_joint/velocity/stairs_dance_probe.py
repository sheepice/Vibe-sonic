"""Evaluation-only reference transitions; never writes the physical robot state."""

from dataclasses import dataclass
from pathlib import Path
import itertools

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from ..kinematics import angular_velocity
from .long_walk_probe import LongWalkAtlasCfg, LongWalkCommand, LongWalkCommandCfg, probe_cases


def probe_tiles():
    return (dict(level=0, kind="up_down", n=6, height=.2, tread=.35,
                 start=1.4, plateau=4.35, goal=12.4, origin=[-30., -30., 0.]),)


def apply_stairs_dance_probe(cfg):
    old = cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"]
    tiles = probe_tiles()
    cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"] = LongWalkAtlasCfg(
        root=old.root, rows=(), random_tiles=tiles)
    cfg.commands["motion"].random_tiles = tiles
    return dict(schema="stairs_dance_stairs_v1", evaluation_only=True, riser_height_m=.2,
                steps_per_flight=6, tread_depth_m=.35, width_m=4., summit_height_m=1.2,
                first_edge_x_m=1.4, summit_start_x_m=3.15, descent_start_x_m=7.85,
                final_edge_x_m=9.6, finish_root_x_m=12.4, dance_trigger_x_m=4.5,
                dance_seconds=12., entry_blend_s=1., exit_blend_s=.6,
                command_m_s=.4, same_policy_all_stages=True, tiles=tiles)


def phase_name(x):
    if x < 1.4:
        return "FLAT APPROACH"
    if x < 3.15:
        return "STAIR UP"
    if x < 7.85:
        return "SUMMIT"
    if x < 9.6:
        return "STAIR DOWN"
    return "FLAT EXIT"


def fixed_views(origin):
    z = np.array([-.12, -1., .28]); z /= np.linalg.norm(z)
    x = np.cross([0., 0., 1.], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.stack([x, y, z], axis=1)
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    views = []
    for left, right, top in ((-.6, 4.6, 2.8), (2.3, 6.7, 2.9),
                             (5.3, 9.2, 2.9), (7., 11.3, 2.9), (9.9, 13.5, 1.6)):
        lo, hi = np.array([left, -.8, -.08]), np.array([right, .8, top])
        center = (lo+hi)/2
        projected = (np.array(list(itertools.product(*zip(lo, hi))))-center)@rotation
        tan_y = np.tan(np.deg2rad(43/2))
        distance = max(np.max(abs(projected[:, 0])/(tan_y*960/540)+projected[:, 2]),
                       np.max(abs(projected[:, 1])/tan_y+projected[:, 2]))*1.03
        views.append(dict(position=(np.asarray(origin)+center+distance*z).tolist(),
                          quaternion=quat.tolist(), distance_m=float(distance),
                          bounds=[lo.tolist(), hi.tolist()]))
    return views, rotation


def view_index(x):
    return int(np.searchsorted([2.7, 5.6, 7.8, 10.6], x, side="right"))


def blend_poses(left, right, weight):
    """Interpolate a free-root G1 reference; quaternion interpolation is on SO(3)."""
    left = np.broadcast_to(left, right.shape)
    output = left*(1-weight[:, None])+right*weight[:, None]
    for i, w in enumerate(weight):
        pair = np.stack([left[i, [4, 5, 6, 3]], right[i, [4, 5, 6, 3]]])
        output[i, 3:7] = Slerp([0, 1], Rotation.from_quat(pair))([w]).as_quat()[0, [3, 0, 1, 2]]
    return output.astype(np.float32)


class StairsDanceCommand(LongWalkCommand):
    def _init_task(self):
        super()._init_task()
        assert self.num_envs == 2
        self.stage = np.zeros(self.num_envs, dtype=int)
        self.stage_start = np.zeros(self.num_envs, dtype=int)
        self.transitions = []
        self.exit_endpoint = {}
        self.buffer_clips = [
            next(i for i, c in enumerate(self.motion.metadata["clips"])
                 if c["id"] == f"evaluation/dance_buffer_{world}")
            for world in range(self.num_envs)]
        path = Path(self.cfg.bank_path).parents[1]/"dance_source.npz"
        with np.load(path, allow_pickle=False) as data:
            self.dance_poses = np.concatenate(
                [data["body_pos_w"][:, 0], data["body_quat_w"][:, 0], data["joint_pos"]], axis=-1)
        assert self.dance_poses.shape == (601, 36)
        # Rotate the whole source once to start facing the commanded +X direction.
        w, x, y, z = self.dance_poses[0, 3:7]
        yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        rot = Rotation.from_euler("z", -yaw)
        self.dance_poses[:, :3] = rot.apply(self.dance_poses[:, :3])
        self.dance_poses[:, 3:7] = (
            rot*Rotation.from_quat(self.dance_poses[:, [4, 5, 6, 3]])
        ).as_quat()[:, [3, 0, 1, 2]]

    def physical_pose(self, world):
        return np.concatenate([self.robot.data.root_link_pos_w[world].cpu().numpy(),
                               self.robot.data.root_link_quat_w[world].cpu().numpy(),
                               self.robot.data.joint_pos[world].cpu().numpy()])

    def upload_reference(self, world, poses):
        """CPU FK for a reference only. pool.data is NOT the simulation state."""
        count = len(poses)
        pos, quat = np.empty((count, 14, 3), np.float32), np.empty((count, 14, 4), np.float32)
        for i, pose in enumerate(poses):
            self.pool.data.qpos[:] = pose
            mujoco.mj_kinematics(self.pool.model, self.pool.data)
            pos[i] = self.pool.data.xpos[self.pool.body_ids]
            quat[i] = self.pool.data.xquat[self.pool.body_ids]
        values = dict(joint_pos=poses[:, 7:], joint_vel=np.gradient(poses[:, 7:], .02, axis=0),
                      body_pos_w=pos, body_quat_w=quat,
                      body_lin_vel_w=np.gradient(pos, .02, axis=0),
                      body_ang_vel_w=angular_velocity(quat, .02))
        clip = self.buffer_clips[world]
        offset, length = int(self.motion.clip_offsets[clip]), int(self.motion.clip_lengths[clip])
        assert count <= length
        for name, value in values.items():
            target = getattr(self.motion, name)
            target[offset:offset+count] = torch.as_tensor(value, device=self.device)
            target[offset+count:offset+length] = target[offset+count-1]
            if "vel" in name:
                target[offset+count:offset+length] = 0.
        self._clip_ids[world] = clip
        self.time_steps[world] = offset
        self._steps_past_end[world] = 0
        self._env.scene.env_origins[world] = 0.

    def event(self, world, kind, step):
        self.transitions.append(dict(world=world, event=kind, time_s=step*.02,
                                     root_world=self.physical_pose(world)[:3].tolist()))
        print("REFERENCE_TRANSITION", self.transitions[-1], flush=True)

    def start_dance(self, world, step):
        actual = self.physical_pose(world)
        dance = self.dance_poses.copy()
        dance[:, :2] += actual[:2]-dance[0, :2]
        dance[:, 2] += 1.2
        t = np.linspace(0, 1, 51)
        entrance = blend_poses(actual, np.repeat(dance[:1], 51, axis=0), t*t*(3-2*t))
        program = np.concatenate([entrance[:-1], dance, np.repeat(dance[-1:], 50, axis=0)])
        self.upload_reference(world, program)
        self.online[world], self.flat[world] = False, True
        self.online_ids = self.online.nonzero().flatten()
        self.stage[world], self.stage_start[world] = 1, step
        self.event(world, "walk_to_dance_blend_start", step)

    def start_exit(self, world, step):
        actual = self.physical_pose(world)
        stream, cursor = int(self.stream[world]), self.pool.cursor
        # Plan sufficient future context for a 0.6 s bridge AND the 0.9 s tokenizer.
        for ahead in (0, 10, 20):
            self.pool._replan(stream, cursor+ahead)
        self._upload_pool()
        target = self.pool.poses[stream, cursor:cursor+90].copy()
        assert self.pool.valid_end[stream] >= cursor+90
        target[:, :2] += actual[:2]-target[0, :2]
        target[:, 2] += 1.2
        t = np.clip(np.arange(90)/30., 0, 1)
        bridge = blend_poses(actual, target, t*t*(3-2*t))
        self.exit_endpoint[world] = bridge[30].copy()
        self.upload_reference(world, bridge)
        self.stage[world], self.stage_start[world] = 2, step
        self.event(world, "dance_12s_complete_walk_bridge_start", step)

    def resume_walk(self, world, step):
        stream, cursor = int(self.stream[world]), self.pool.cursor
        self._env.scene.env_origins[world, :2] = torch.as_tensor(
            self.exit_endpoint[world][:2]-self.pool.poses[stream, cursor, :2], device=self.device)
        self._env.scene.env_origins[world, 2] = 1.2
        self.online[world], self.flat[world] = True, False
        self.online_ids = self.online.nonzero().flatten()
        self._clip_ids[world] = self.replay_clips+stream
        self.time_steps[world] = self.replay_frames+stream*self.pool.capacity+cursor
        self.stage[world] = 3
        self.event(world, "online_velocity_walk_resumed", step)

    def _update_command(self, env_ids=None):
        super()._update_command(env_ids)
        if env_ids is None:
            step = int(self._env.common_step_counter)
            for world in range(self.num_envs):
                root_x = float(self.robot.data.root_link_pos_w[world, 0]-
                               self.tile_origins[self.tile[world], 0])
                if (self.stage[world] == 0 and root_x >= 4.5
                        and float(self.robot.data.projected_gravity_b[world, 2]) < -.85):
                    self.start_dance(world, step)
                elif self.stage[world] == 1 and step-self.stage_start[world] >= 650:
                    self.start_exit(world, step)
                elif self.stage[world] == 2 and step-self.stage_start[world] >= 30:
                    self.resume_walk(world, step)
        self.speed[:] = torch.where(self.online, self.fixed_speeds, 0.)
        self.update_relative_body_poses()

    def phase_label(self, world, x):
        if self.stage[world] == 1:
            elapsed = (int(self._env.common_step_counter)-self.stage_start[world])*.02
            return "BLEND TO DANCE" if elapsed < 1 else f"DANCE {elapsed-1:.1f}/12.0s"
        if self.stage[world] == 2:
            return "BLEND TO WALK"
        return phase_name(x)


@dataclass(kw_only=True)
class StairsDanceCommandCfg(LongWalkCommandCfg):
    def build(self, env):
        return StairsDanceCommand(self, env)
