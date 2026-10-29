"""Separate flat/stair reward curves so the larger flat group cannot hide regressions."""

import torch
from mjlab.rl import RslRlVecEnvWrapper


class JointLoggingWrapper(RslRlVecEnvWrapper):
    def __init__(self, env, flat_to_terrain_ratio):
        super().__init__(env)
        self._terrain_mask = (
            torch.arange(self.num_envs, device=self.device) % (flat_to_terrain_ratio + 1) == 0
        )
        self._episode_returns = torch.zeros(self.num_envs, device=self.device)
        self._episode_steps = torch.zeros(self.num_envs, device=self.device)

    def step(self, actions):
        obs, reward, dones, extras = super().step(actions)
        self._episode_returns += reward
        self._episode_steps += 1
        log = extras.setdefault("log", {})
        for name, mask in (("stairs", self._terrain_mask), ("flat", ~self._terrain_mask)):
            log[f"Joint/{name}_step_reward"] = reward[mask].mean()
            ended = mask & dones.bool()
            if bool(ended.any()):
                log[f"Joint/{name}_episode_return"] = self._episode_returns[ended].mean()
                log[f"Joint/{name}_episode_steps"] = self._episode_steps[ended].mean()
        self._episode_returns[dones.bool()] = 0
        self._episode_steps[dones.bool()] = 0
        return obs, reward, dones, extras
