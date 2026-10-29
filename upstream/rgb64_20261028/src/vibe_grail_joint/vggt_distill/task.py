"""Original stairs mixture or the archived ascent-only experiment, selected explicitly."""
import copy
from dataclasses import dataclass, fields, replace

import torch
from ..velocity.dagger_task import DaggerMotionCommand, DaggerMotionCommandCfg
from ..velocity.quality_task import QualityStairsEnv, REMOVED_REWARDS
from ..velocity.quality_task import env_config as base_env_config, options as base_options
from ..velocity.config import agent_config as base_agent_config


class AscentMotionCommand(DaggerMotionCommand):
    def _init_task(self):
        super()._init_task()
        slot = torch.arange(self.num_envs, device=self.device) % 20
        self.ppo_online = slot < 14
        self.paired = torch.zeros_like(self.ppo_online)
        self.dagger_planner = (slot >= 14) & (slot < 17)
        self.flat = slot >= 17
        self.dagger_mask = self.dagger_planner | self.flat
        self.ppo_mask = self.ppo_online
        self.online = self.ppo_online | self.dagger_planner
        self.online_ids = self.online.nonzero().flatten()
        self.metrics['terrain_fraction'] = self.ppo_online.float()

    def _sample_init_frame(self, env_ids):
        chosen, frames = super()._sample_init_frame(env_ids)
        mask = self.ppo_online[env_ids]
        ids = env_ids[mask]
        if len(ids):
            self.tile[ids] = self.tile[ids] // 4 * 4
            self.stream[ids] = torch.randint(1, len(self.cfg.planner_speeds), (len(ids),), device=self.device)
            chosen[mask] = self.replay_clips + self.stream[ids]
            frames[mask] = self.motion.clip_offsets[chosen[mask]] + self.pool.cursor - self.phase[ids]
            nominal = torch.tensor(self.cfg.planner_speeds, device=self.device)[self.stream[ids]]
            self.speed[ids] = (nominal + torch.empty_like(nominal).uniform_(-.04, .04)).clamp_min(.15)
            spawn = self.tile_origins[self.tile[ids]].clone()
            spawn[:, 0] += torch.empty(len(ids), device=self.device).uniform_(-.35, .35)
            spawn[:, 1] += torch.empty(len(ids), device=self.device).uniform_(-.25, .25)
            spawn[:, 2] = self.ground_height(spawn[:, :2], ids)
            spawn[:, :2] -= self.motion.body_pos_w[frames[mask], 0, :2]
            self._env.scene.env_origins[ids] = spawn
            if (self.tile_data['kind'][self.tile[ids]] != 0).any():
                raise RuntimeError('Nonascending terrain entered the PPO training split')
        return chosen, frames


@dataclass(kw_only=True)
class AscentMotionCommandCfg(DaggerMotionCommandCfg):
    def build(self, env):
        return AscentMotionCommand(self, env)


def options(settings):
    result = base_options(settings)
    if settings.get('gaussgym') or settings.get('hiking_perception_experiment'):
        raise ValueError('This experiment must have no independent geometry branch')
    result.update(ascent_only=not original_mixture(settings), distillation=copy.deepcopy(settings['distillation']),
                  camera=copy.deepcopy(settings['camera']))
    return result


def original_mixture(settings):
    mode = settings.get('task_mode', 'ascent_only_v1')
    if mode not in ('ascent_only_v1', 'stairs_original_mixture_v1'):
        raise ValueError(f'Unknown distillation task mode: {mode}')
    return mode == 'stairs_original_mixture_v1'


def task_fractions(settings):
    return dict(online_ppo=.5 if original_mixture(settings) else .7,
                paired_grail_ppo=.2 if original_mixture(settings) else 0.,
                flat_dagger_planner=.15, flat_dagger_replay=.15)


def env_config(bank_path, settings, num_envs):
    options(settings)
    cfg = base_env_config(bank_path, settings, num_envs)
    if original_mixture(settings):
        # Preserve DaggerMotionCommand, paired GRAIL terrain, all four online
        # terrain kinds, and the original speed sampling distribution.
        camera = settings['camera']
        cfg.scene.sensors = tuple(replace(s, width=camera['width'], height=camera['height'], data_types=('rgb',))
                                  if s.name == 'head_cam' else s for s in cfg.scene.sensors)
        return cfg
    old = cfg.commands['motion']
    # Keep four-slot indexing expected by the existing flat-preservation sampler,
    # but physically replace both downward slots with upward copies as well.
    tiles = []
    for i, tile in enumerate(old.random_tiles):
        row = copy.deepcopy(old.random_tiles[i // 4 * 4] if i % 4 != 3 else tile)
        row['origin'] = list(tile['origin'])
        tiles.append(row)
    params = {f.name: getattr(old, f.name) for f in fields(AscentMotionCommandCfg)}
    params['random_tiles'] = tuple(tiles)
    cfg.commands['motion'] = AscentMotionCommandCfg(**params)
    atlas = cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs'] = replace(atlas, random_tiles=tuple(tiles), rows=())
    camera = settings['camera']
    cfg.scene.sensors = tuple(replace(s, width=camera['width'], height=camera['height'], data_types=('rgb',))
                              if s.name == 'head_cam' else s for s in cfg.scene.sensors)
    return cfg


def agent_config(bank_path, settings, num_envs):
    from .model import configure_agent
    return configure_agent(base_agent_config(bank_path, settings, num_envs), settings)
