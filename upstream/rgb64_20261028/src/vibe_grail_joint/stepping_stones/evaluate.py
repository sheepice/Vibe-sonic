"""Held-out online-command evaluation; floor shortcuts explicitly fail."""
import argparse,json,copy,math
from dataclasses import dataclass,fields
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    for k in ('config','checkpoint','output'):p.add_argument('--'+k,required=True)
    p.add_argument('--seed',type=int,default=271831)
    p.add_argument('--duration',type=float,default=40.)
    p.add_argument('--mask-rgb',action='store_true')
    p.add_argument('--rgb-delay-steps',type=int,default=0)
    p.add_argument('--terrain-style',choices=('default','same_gray'),default='default')
    a=p.parse_args()
    if not 0<=a.rgb_delay_steps<=20 or (a.mask_rgb and a.rgb_delay_steps):p.error('Use black RGB or a delay of 0..20 policy steps')
    import os
    if os.environ.get('STEPPING_STONE_SHARED_REVIEW')=='1':
        from .review import verify_review_worker
        verify_review_worker()
    else:
        from ..gpu_guard import verify_guarded_worker
        verify_guarded_worker()
    import mjlab,torch,numpy as np
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from orcs.core.robustness import strip_domain
    from ..velocity.eval_command import EvaluationCommand,EvaluationCommandCfg
    from ..velocity.step_quality_observer import StepQualityObserver
    from ..velocity.recovery_terrain import profile_success
    from .command import StoneCommand,StoneCommandCfg
    from .task import env_config,floor_contact
    from .labels import rgb_only
    from .terrain import FAMILIES
    from .runtime import load_torch_control
    from .route_audit import RouteAudit
    from .targets import install as install_targets, guard_failed, guard_diagnostic, contract_summary
    from ..hiking_gaussgym.black_rgb import BlackRGB
    torch.set_num_threads(2);torch.manual_seed(a.seed)
    settings=json.loads(Path(a.config).read_text());settings['seed']=a.seed;settings['bank_root']=str(Path(__file__).resolve().parents[3]/'data/smoke')
    cfg=env_config(settings['bank_root']+'/rank_0',settings,60)
    gap_enabled = bool(settings['stones'].get('gap_crossing'))
    if gap_enabled:
        from .gap_crossing_task import configure, install
        configure(cfg, settings, rewards=False)
    tiles=cfg.commands['motion'].random_tiles
    cases=[]
    for family in FAMILIES:
        levels=[8] if family=='stairs' else [0,2,5]
        kinds=['up','down','up_down'] if family=='stairs' else ['up_down']
        for level in levels:
            for kind in kinds:
                selected=next(i for i,t in enumerate(tiles) if t['family']==family and t['course_level']==level and t['kind']==kind and t['variant']==1)
                for speed in (.25,.4):
                    for phase in (0,10):
                        cases.append(dict(id=len(cases),tile=selected,stream=settings['planner_speeds'].index(speed),speed=speed,
                            phase=phase,spawn_xy=[-.1,0. if phase==0 else .05],family=family,level=level,kind=kind))
    count=len(cases);n=math.ceil(count/20)*20
    cases += [dict(cases[i%count],id=count+i,padding=True) for i in range(n-count)]
    class EvalCommand(EvaluationCommand,StoneCommand):
        def _init_task(self):
            super()._init_task();self.ppo_online[:]=True;self.dagger_mask[:]=False;self.ppo_mask[:]=True;self.dagger_planner[:]=False
            self.gap_bounds=torch.tensor([t['gaps'][0] if t.get('family')=='stair_gap' else [-1e5,-1e5] for t in self.cfg.random_tiles],device=self.device)
    @dataclass(kw_only=True)
    class EvalCfg(EvaluationCommandCfg,StoneCommandCfg):
        def build(self,env):return EvalCommand(self,env)
    old=cfg.commands['motion'];params={f.name:getattr(old,f.name) for f in fields(EvalCfg) if hasattr(old,f.name)}
    params.update(cases=tuple(cases),pose_range={},velocity_range={},joint_position_range=(0.,0.))
    cfg.commands['motion']=EvalCfg(**params);cfg.scene.num_envs=n
    cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs'].rows=()
    strip_domain(cfg);cfg.events.pop('policy_update_counter',None)
    if a.terrain_style=='same_gray':
        from mjlab.managers import EventTermCfg
        from vibe.core.mdp.events import set_terrain_color
        cfg.events['evaluation_same_gray']=EventTermCfg(func=set_terrain_color,mode='startup',
            params={'rgba':(.5,.5,.5,1.),'ground_rgba':None})
    for group in cfg.observations.values():
        group.enable_corruption=False
        for term in group.terms.values():term.noise=None
    cfg.terminations.clear();cfg.rewards={};cfg.episode_length_s=a.duration+5;rgb_only(cfg)
    class Observed(ManagerBasedRlEnv):
        def __init__(self,cfg,**kwargs):
            super().__init__(cfg,**kwargs);self.quality=StepQualityObserver(self)
            old=self.metrics_manager.compute_substep
            def substep():old();self.quality.observe_substep()
            self.metrics_manager.compute_substep=substep
        def step(self,actions):self.quality.begin_step();return super().step(actions)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    env=RslRlVecEnvWrapper(Observed(cfg,device='cuda:0'))
    if gap_enabled: install(env.unwrapped, settings)
    install_targets(env.unwrapped,settings,strict=True)
    try:
        obs,_=env.reset();raw=env.unwrapped;c=raw.command_manager.get_term('motion');robot=raw.scene['robot']
        actor,memory=load_torch_control(a.checkpoint,obs,settings,raw.device)
        black=BlackRGB(raw) if a.mask_rgb else None
        from collections import deque
        history=deque(maxlen=a.rgb_delay_steps+1)
        route=RouteAudit(raw)
        reset=torch.ones(n,dtype=torch.bool,device=raw.device)
        active=torch.ones(n,dtype=torch.bool,device=raw.device);rows=[None]*n
        impacts=torch.zeros(n,2,device=raw.device);error=torch.zeros(n,device=raw.device);steps=torch.zeros_like(error)
        for step in range(round(a.duration/raw.step_dt)):
            with torch.inference_mode():
                acted=obs.clone()
                if black:
                    acted['kv_tokens','img_tokens'].copy_(black.tokens);acted['q_cls'].copy_(black.cls)
                elif a.rgb_delay_steps:
                    history.append((obs['kv_tokens','img_tokens'].clone(),obs['q_cls'].clone()))
                    acted['kv_tokens','img_tokens'].copy_(history[0][0]);acted['q_cls'].copy_(history[0][1])
                acted['hiking_latent']=memory.step(acted,reset)
                actions=actor(acted,stochastic_output=False)
                if not torch.isfinite(actions).all():raise RuntimeError('Nonfinite action')
                obs,_,done,_=env.step(actions);reset=done.bool()
                if done.any():raise RuntimeError('Unexpected auto-reset')
                root=robot.data.root_link_pos_w;local=root-c.tile_origins[c.tile]
                clearance=root[:,2]-c.ground_height(root[:,:2])
                fell=(robot.data.projected_gravity_b[:,2]>-.4)|(clearance<.25)|floor_contact(raw)
                route.update(active)
                guard=guard_failed(raw)
                success=profile_success(raw)&~fell&route.qualified&~guard
                if gap_enabled: success &= raw.gap_crossing.qualified
                error+=((robot.data.root_link_lin_vel_w[:,0]-c.speed).abs())*active;steps+=active
                q=raw.quality.step_data
                impacts[:,0]+=q['foot_riser_impulse_ns'].sum(-1)*active
                impacts[:,1]+=q['shin_impulse_ns'].sum(-1)*active
                terminal=active&(success|fell|guard|(step==round(a.duration/raw.step_dt)-1))
                for i in terminal.nonzero().flatten().tolist():
                    rows[i]=dict(case=cases[i],success=bool(success[i]),fall=bool(fell[i]),actuator_guard_failed=bool(guard[i]),time_s=(step+1)*raw.step_dt,
                        guard_detected_at_s=step*raw.step_dt if guard[i] else None,
                        guard_diagnostic=guard_diagnostic(raw,i) if guard[i] else None,
                        progress_m=float(local[i,0]),speed_error_m_s=float(error[i]/steps[i]),
                        foot_impulse_ns=float(impacts[i,0]),shin_impulse_ns=float(impacts[i,1]),**route.row(i))
                    if gap_enabled: rows[i].update(raw.gap_crossing.row(i))
                active[terminal]=False
            if step%250==0:print(json.dumps(dict(step=step,active=int(active.sum()))),flush=True)
            if not active.any():break
        report=dict(checkpoint=str(Path(a.checkpoint).resolve()),seed=a.seed,rgb_masked=a.mask_rgb,rgb_delay_steps=a.rgb_delay_steps,
                    terrain_style=a.terrain_style,cases=rows[:count],
                    passed=sum(r['success'] for r in rows[:count]),total=count,includes_floor_shortcut_failure=True,
                    verifies_stair_gap_platform_order=True,deployment_targets=contract_summary(settings),
                    initial_state='training_walking_RSI',rgb_update_hz=1/env.unwrapped.step_dt,
                    target_projection_rounding='inward_float32_v1')
        if gap_enabled: report.update(verifies_airborne_gap_crossing=True, gap_crossing_settings=settings['stones']['gap_crossing'])
        (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    finally:env.close()
if __name__=='__main__':main()
