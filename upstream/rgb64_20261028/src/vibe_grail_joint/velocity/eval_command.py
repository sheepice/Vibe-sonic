"""Deterministic, online-only held-out trials; no recorded stair motion is selected."""

from dataclasses import dataclass

import torch

from .command import VelocityMotionCommand, VelocityMotionCommandCfg


class EvaluationCommand(VelocityMotionCommand):
    def _init_task(self):
        super()._init_task()
        self.online[:] = True
        self.paired[:] = False
        self.flat[:] = False
        self.online_ids = torch.arange(self.num_envs, device=self.device)
        self.cases = self.cfg.cases
        self.fixed_speeds = torch.tensor([c['speed'] for c in self.cases], device=self.device)

    def _sample_init_frame(self, env_ids):
        ids = env_ids.tolist()
        self.tile[env_ids] = torch.tensor([self.cases[i]['tile'] for i in ids], device=self.device)
        self.stream[env_ids] = torch.tensor([self.cases[i]['stream'] for i in ids], device=self.device)
        self.phase[env_ids] = torch.tensor([self.cases[i]['phase'] for i in ids], device=self.device)
        self.speed[env_ids] = self.fixed_speeds[env_ids]
        chosen = self.replay_clips + self.stream[env_ids]
        frames = self.motion.clip_offsets[chosen] + self.pool.cursor - self.phase[env_ids]
        spawn = self.tile_origins[self.tile[env_ids]].clone()
        spawn[:, :2] += torch.tensor([self.cases[i]['spawn_xy'] for i in ids], device=self.device)
        spawn[:, 2] = self.ground_height(spawn[:, :2], env_ids)
        spawn[:, :2] -= self.motion.body_pos_w[frames, 0, :2]
        self._env.scene.env_origins[env_ids] = spawn
        return chosen, frames

    def _reset_task(self, env_ids, clip_ids, time_steps, origins):
        self.success_hold[env_ids] = 0
        self._write_reference_state_to_sim(
            env_ids,
            self.motion.body_pos_w[time_steps, 0] + origins,
            self.motion.body_quat_w[time_steps, 0],
            self.motion.body_lin_vel_w[time_steps, 0],
            self.motion.body_ang_vel_w[time_steps, 0],
            self.motion.joint_pos[time_steps],
            self.motion.joint_vel[time_steps],
        )

    def _update_command(self, env_ids=None):
        super()._update_command(env_ids)
        self.speed[:] = self.fixed_speeds


@dataclass(kw_only=True)
class EvaluationCommandCfg(VelocityMotionCommandCfg):
    cases: tuple = ()

    def build(self, env):
        return EvaluationCommand(self, env)
