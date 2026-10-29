"""Route ONLY PPO worlds into rollout storage; collect flat teacher labels."""

import torch
from mjlab.rl import RslRlVecEnvWrapper

from .dagger_buffer import DaggerBuffer
from .dagger_policy import build_teacher


class DaggerVecEnv(RslRlVecEnvWrapper):
    def __init__(self, wrapped, settings, rank):
        # The inner wrapper already reset the full simulator. Never advertise
        # DAgger worlds to RSL-RL: even advantage normalization must exclude them.
        self.env = wrapped
        self.settings, self.rank = settings, rank
        self.device, self.num_actions = wrapped.device, wrapped.num_actions
        self.max_episode_length, self.clip_actions = wrapped.max_episode_length, wrapped.clip_actions
        c = self.unwrapped.command_manager.get_term("motion")
        self.ppo_ids = c.ppo_mask.nonzero().flatten()
        self.dagger_ids = c.dagger_mask.nonzero().flatten()
        self.num_envs = len(self.ppo_ids)
        self.buffer = DaggerBuffer(settings["dagger"]["buffer_capacity"],
                                   len(c.flat_sources) + len(c.cfg.planner_speeds), settings["seed"] + rank)
        self.actor = self.teacher = None
        self._full_obs = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def posture_steps(self):
        return self.env.posture_steps

    @posture_steps.setter
    def posture_steps(self, value):
        self.env.posture_steps = value

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf[self.ppo_ids]

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf[self.ppo_ids] = value

    def attach(self, algorithm, actor_cfg, obs_groups):
        self.actor = algorithm.get_policy()
        self.teacher, digest = build_teacher(self._full_obs[:1], actor_cfg, obs_groups,
                                             self.settings["dagger"]["teacher_checkpoint"], self.device)
        algorithm.dagger_buffer = self.buffer
        algorithm.teacher_sha256 = digest

    def get_observations(self):
        self._full_obs = self.env.get_observations()
        return self._full_obs[self.ppo_ids]

    def reset(self):
        self._full_obs, extras = self.env.reset()
        return self._full_obs[self.ppo_ids], extras

    @torch.no_grad()
    def step(self, actions):
        obs = self._full_obs[self.dagger_ids]
        # Student history and previous actions are already in obs. Both models
        # see the SAME pre-step reference and proprioception; only student acts.
        teacher = self.teacher(obs, stochastic_output=False)
        student = self.actor(obs, stochastic_output=False)
        if not torch.isfinite(student).all() or not torch.isfinite(teacher).all():
            raise RuntimeError("Non-finite flat student action or SONIC teacher label")
        cached = self.actor.cache_obs(obs)
        # Critic-only terrain is irrelevant to supervised learning.
        cached = cached.exclude("critic")
        c = self.unwrapped.command_manager.get_term("motion")
        self.buffer.add(cached, teacher, c.dagger_strata())
        full_action = actions.new_empty((self.unwrapped.num_envs, self.num_actions))
        full_action[self.ppo_ids], full_action[self.dagger_ids] = actions, student
        self.actor.update_normalization(obs)
        self._full_obs, rewards, dones, extras = self.env.step(full_action)
        extras = dict(extras)
        if "time_outs" in extras:
            extras["time_outs"] = extras["time_outs"][self.ppo_ids]
        extras.setdefault("log", {})["DAgger/collected_action_mse"] = (student-teacher).square().mean()
        return self._full_obs[self.ppo_ids], rewards[self.ppo_ids], dones[self.ppo_ids], extras
