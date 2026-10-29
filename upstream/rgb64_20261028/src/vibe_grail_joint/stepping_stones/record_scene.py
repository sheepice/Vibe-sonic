"""Two independent matched worlds, without training-only DAgger padding."""
import copy
from dataclasses import dataclass, fields, replace

import torch

from ..velocity.eval_profile_command import ProfileEvaluationCommand, ProfileEvaluationCommandCfg
from .terrain import PrimitiveHeights, FAMILIES


class StoneEvaluationCommand(ProfileEvaluationCommand):
    def _init_task(self):
        super()._init_task()
        self.primitives = PrimitiveHeights(self.cfg.random_tiles, self.device)
        self.family = torch.tensor([FAMILIES.index(t['family']) for t in self.cfg.random_tiles], device=self.device)
        self.ppo_online = self.online.clone()
        self.ppo_mask = self.online.clone()
        self.dagger_mask = torch.zeros_like(self.online)
        self.dagger_planner = torch.zeros_like(self.online)
        self.gap_bounds = torch.tensor([t['gaps'][0] if t.get('family')=='stair_gap'
                                       else [-1e5,-1e5] for t in self.cfg.random_tiles], device=self.device)

    def ground_height(self, xy, env_ids=None):
        base = super().ground_height(xy, env_ids)
        if not hasattr(self, 'primitives'):
            return base
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        tile = self.tile[ids]
        mask = self.family[tile]>0
        if mask.any():
            base[mask] = self.primitives.query(xy[mask], tile[mask], self.tile_origins)[0]
        return base

    def surface_ids(self, xy, env_ids):
        return self.primitives.query(xy, self.tile[env_ids], self.tile_origins)[1]


@dataclass(kw_only=True)
class StoneEvaluationCommandCfg(ProfileEvaluationCommandCfg):
    def build(self, env):
        return StoneEvaluationCommand(self, env)


def build(settings, case, seed, duration, *, with_review_camera=True):
    from pathlib import Path
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from orcs.core.robustness import strip_domain
    from ..hiking_gaussgym.simulation import add_review_camera
    from ..velocity.step_quality_observer import StepQualityObserver
    from .task import env_config
    from .labels import rgb_only
    settings = copy.deepcopy(settings)
    settings['seed'] = seed
    settings['bank_root'] = str(Path(__file__).resolve().parents[3]/'data/smoke')
    cfg = env_config(settings['bank_root']+'/rank_0', settings, 20)
    cfg.scene.num_envs = 2
    tile = copy.deepcopy(cfg.commands['motion'].random_tiles[case['tile']])
    if tile['family'] != case['family'] or tile['course_level'] != case['level']:
        raise ValueError('Recorded case does not match generated terrain')
    cfg.commands['motion'].random_tiles = (tile,)
    atlas = cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    atlas.random_tiles = (tile,)
    atlas.rows = ()
    cases = tuple(dict(case, id=i, source_case_id=case['id'], source_tile=case['tile'], tile=0) for i in range(2))
    old = cfg.commands['motion']
    params = {f.name:getattr(old,f.name) for f in fields(StoneEvaluationCommandCfg) if hasattr(old,f.name)}
    params.update(cases=cases, pose_range={}, velocity_range={}, joint_position_range=(0.,0.))
    cfg.commands['motion'] = StoneEvaluationCommandCfg(**params)
    camera_tile = copy.deepcopy(tile)
    if tile.get('solids'):
        camera_tile.update(base=max(s['top'] for s in tile['solids']), edges=[], jumps=[])
    if with_review_camera:
        add_review_camera(cfg, camera_tile)
        cfg.scene.sensors = tuple(replace(s,width=960,height=540) if s.name=='review_camera' else s for s in cfg.scene.sensors)
    strip_domain(cfg)
    cfg.events.pop('policy_update_counter',None)
    for group in cfg.observations.values():
        group.enable_corruption=False
        for term in group.terms.values():term.noise=None
    cfg.terminations={}
    cfg.rewards={}
    cfg.episode_length_s=duration+5
    rgb_only(cfg)

    class Observed(ManagerBasedRlEnv):
        def __init__(self,cfg,**kwargs):
            super().__init__(cfg,**kwargs)
            self.quality=StepQualityObserver(self)
            previous=self.metrics_manager.compute_substep
            def substep():previous();self.quality.observe_substep()
            self.metrics_manager.compute_substep=substep
        def step(self,actions):
            self.quality.begin_step()
            return super().step(actions)
        def _reset_idx(self,ids=None):
            if hasattr(self,'quality'):self.quality.reset(ids)
            return super()._reset_idx(ids)

    env = Observed(cfg,device='cuda:0')
    from .targets import install as install_targets
    install_targets(env,settings,strict=True)
    return RslRlVecEnvWrapper(env), settings, cases, tile
