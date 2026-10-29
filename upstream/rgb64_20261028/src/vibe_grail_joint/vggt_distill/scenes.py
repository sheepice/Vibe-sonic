"""Fixed up/down/combined stair cases, independent of training terrain seeds."""
import copy
from dataclasses import fields
import itertools

import numpy as np


def cases_and_tiles(settings, seed=260027, heights=(.12, .16, .20, .25, .30),
                    speeds=(.25, .4, .7), approaches=(0., 2.), irregular=False, route='up'):
    if route not in ('up', 'down', 'up_down'):
        raise ValueError('Unknown review route')
    rng = np.random.default_rng(seed)
    tiles, cases = [], []
    for height, approach in itertools.product(heights, approaches):
        n = 6
        risers = rng.uniform(.75, 1., n) * height if irregular else np.full(n, height)
        treads = rng.uniform(.28, .40, n) if irregular else np.full(n, .35)
        start = 1.4 + approach
        edges = start + np.r_[0., np.cumsum(treads[:-1])]
        jumps, base = risers.copy(), 0.
        end = start + treads.sum()
        if route == 'down':
            jumps, base = -risers[::-1], float(risers.sum())
        elif route == 'up_down':
            # A complete route with an explicit approach, top landing and exit.
            start = 2.4 + approach
            edges = start + np.r_[0., np.cumsum(treads[:-1])]
            down_start = start + treads.sum() + 1.2
            edges = np.r_[edges, down_start + np.r_[0., np.cumsum(treads[::-1][:-1])]]
            jumps = np.r_[risers, -risers[::-1]]
            end = down_start + treads.sum() + 1.4
        i = len(tiles)
        tiles.append(dict(level=0, variant=0, kind=route, n=n, height=float(risers.max()),
                          nominal_height=height, tread=float(treads.mean()), start=start, plateau=1.2 if route == 'up_down' else 1.,
                          goal=float(end+.6), origin=[-30.-(i%12)*16., -30.-(i//12)*10., 0.],
                          edges=edges.tolist(), jumps=jumps.tolist(), base=base, risers=risers.tolist(),
                          treads=treads.tolist(), total_height=float(risers.sum())))
        for speed, phase in itertools.product(speeds, (0, 5, 10, 15)):
            cases.append(dict(id=len(cases), tile=i, speed=speed,
                              stream=settings['planner_speeds'].index(speed), phase=phase,
                              spawn_xy=[float(rng.uniform(-.2, .2)), float(rng.uniform(-.15, .15))],
                              nominal_height_m=height, approach_extra_m=approach,
                              out_of_training_height=height > .25, route=route,
                              out_of_training_direction=(route != 'up' and settings.get('task_mode') != 'stairs_original_mixture_v1')))
    return tiles, cases


def evaluation_env(settings, *, seed, duration, heights, speeds, approaches, irregular=False,
                   record_case=None, limit_cases=None, route='up'):
    import mjlab
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from orcs.core.robustness import strip_domain
    from .task import env_config
    from ..velocity.eval_profile_command import ProfileEvaluationCommandCfg
    from ..velocity.step_quality_observer import StepQualityObserver
    from ..hiking_gaussgym.simulation import add_review_camera
    settings = copy.deepcopy(settings); settings['seed'] = seed
    tiles, cases = cases_and_tiles(settings, seed, heights, speeds, approaches, irregular, route)
    if record_case is not None:
        if not 0 <= record_case < len(cases): raise ValueError('Recording case out of range')
        cases = [cases[record_case]]
    elif limit_cases is not None:
        if limit_cases < 1: raise ValueError('Case limit must be positive')
        cases = cases[:limit_cases]
    cfg = env_config(settings['bank_root'] + '/rank_0', settings, 20)
    cfg.scene.num_envs = len(cases)
    old = cfg.commands['motion']
    params = {f.name: getattr(old, f.name) for f in fields(ProfileEvaluationCommandCfg) if hasattr(old, f.name)}
    params.update(cases=tuple(cases), random_tiles=tuple(tiles), pose_range={}, velocity_range={}, joint_position_range=(0., 0.))
    cfg.commands['motion'] = ProfileEvaluationCommandCfg(**params)
    atlas = cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    atlas.random_tiles = tuple(tiles); atlas.rows = ()
    if record_case is not None: add_review_camera(cfg, tiles[cases[0]['tile']])
    strip_domain(cfg)
    cfg.events.pop('policy_update_counter', None)
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values(): term.noise = None
    cfg.terminations.clear(); cfg.rewards = {}; cfg.episode_length_s = duration + 5

    class ObservedEnv(ManagerBasedRlEnv):
        def __init__(self, cfg, **kwargs):
            super().__init__(cfg, **kwargs)
            self.quality = StepQualityObserver(self)
            previous = self.metrics_manager.compute_substep
            def substep():
                previous(); self.quality.observe_substep()
            self.metrics_manager.compute_substep = substep

        def step(self, actions):
            self.quality.begin_step()
            return super().step(actions)

        def _reset_idx(self, ids=None):
            if hasattr(self, 'quality'): self.quality.reset(ids)
            return super()._reset_idx(ids)

    env = RslRlVecEnvWrapper(ObservedEnv(cfg, device='cuda:0'))
    return env, settings, cases
