"""Evaluation-only two-world paired RGB ablations on a single specified scene."""
import copy
from dataclasses import fields
import torch


def evaluation_env(settings, scene, seed=170017, duration=40.):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from orcs.core.robustness import strip_domain
    from .task import env_config
    from .simulation import add_review_camera
    from .labels import rgb_only
    from ..velocity.eval_profile_command import ProfileEvaluationCommandCfg
    from ..velocity.step_quality_observer import StepQualityObserver
    settings=copy.deepcopy(settings);settings['seed']=seed
    cfg=env_config(settings['bank_root']+'/rank_0',settings,20)
    cfg.scene.num_envs=2
    tile=scene['tile']
    cfg.commands['motion'].random_tiles=(tile,)
    atlas=cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    atlas.random_tiles=(tile,);atlas.rows=()
    command_class=ProfileEvaluationCommandCfg
    if scene.get('heightfield'):
        from .suite_heightfield import configure_field
        command_class=configure_field(cfg,scene)
    cases=[dict(id=i,tile=0,speed=speed,stream=settings['planner_speeds'].index(speed),
                phase=0,spawn_xy=[-.15,-.15],intervention=mode)
           for i,(speed,mode) in enumerate(((.4,'normal'),(.4,'black_rgb')))]
    old=cfg.commands['motion']
    params={f.name:getattr(old,f.name) for f in fields(command_class) if hasattr(old,f.name)}
    params.update(cases=tuple(cases),pose_range={},velocity_range={},joint_position_range=(0.,0.))
    cfg.commands['motion']=command_class(**params)
    add_review_camera(cfg,tile)
    from dataclasses import replace
    cfg.scene.sensors=tuple(replace(sensor,width=1280,height=720) if sensor.name=="review_camera" else sensor for sensor in cfg.scene.sensors)
    strip_domain(cfg);cfg.events.pop('policy_update_counter',None)
    for group in cfg.observations.values():
        group.enable_corruption=False
        for term in group.terms.values():term.noise=None
    cfg.terminations.clear();cfg.rewards={};cfg.episode_length_s=duration+5
    rgb_only(cfg)
    class ObservedEnv(ManagerBasedRlEnv):
        def __init__(self,cfg,**kwargs):
            super().__init__(cfg,**kwargs)
            self.quality=StepQualityObserver(self)
            previous=self.metrics_manager.compute_substep
            def substep():
                previous();self.quality.observe_substep()
            self.metrics_manager.compute_substep=substep
        def step(self,actions):
            self.quality.begin_step()
            return super().step(actions)
        def _reset_idx(self,ids=None):
            if hasattr(self,'quality'):self.quality.reset(ids)
            return super()._reset_idx(ids)
    return RslRlVecEnvWrapper(ObservedEnv(cfg,device='cuda:0')),settings,cases
