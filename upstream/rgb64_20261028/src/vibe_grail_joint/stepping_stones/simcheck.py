"""Physics audit of the portable RGB runtime and reconstructed robot observations.

Parity mode isolates observation construction from image-encoder precision.
Portable mode feeds raw RGB and the deployable online planner; simulator truth
is used only to initialize the trial and report progress/falls, never as input.
"""
import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ('bundle', 'config', 'output'):
        p.add_argument('--'+k, required=True)
    p.add_argument('--mode', choices=('parity', 'portable'), default='parity')
    p.add_argument('--family', choices=('stairs', 'box_stones', 'cylinder_stones', 'stair_gap'), default='stairs')
    p.add_argument('--level', type=int, default=0)
    p.add_argument('--seed', type=int, default=271831)
    p.add_argument('--terrain-variant', type=int, choices=(0,1,2), default=1)
    p.add_argument('--capture-video', action='store_true')
    p.add_argument('--duration', type=float, default=5.)
    p.add_argument('--speed', type=float, choices=(.25, .4), default=.25)
    p.add_argument('--actuator-guards', action='store_true',
                   help='Apply the target conditioning and fault checks used by real control')
    p.add_argument('--stop-hold-seconds', type=float, default=3.,
                   help='Time from releasing forward command to the end, including deceleration')
    p.add_argument('--camera-fps', type=float, default=30.,
                   help='Portable camera cadence; hold the last RGB features between frames (default D435i 30 Hz)')
    args = p.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        p.error('Use a positive finite duration')
    if args.mode == 'portable' and not 1.8 <= args.stop_hold_seconds < args.duration-1.:
        p.error('Stop interval must be at least 1.8 seconds and leave time to walk')
    if args.actuator_guards and args.mode != 'portable':
        p.error('Actuator conditioning is a portable closed-loop test, not a parity test')
    if args.capture_video and (args.mode!='portable' or not args.actuator_guards or args.camera_fps!=30.):
        p.error('Portable recording uses guarded 30 Hz RGB trials')
    if not math.isfinite(args.camera_fps) or not 0 < args.camera_fps <= 50:
        p.error('Camera FPS must be in (0, 50]')
    from .review import verify_review_worker
    verify_review_worker()
    import mjlab, torch, numpy as np
    from scipy.spatial.transform import Rotation
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from orcs.core.robustness import strip_domain
    from ..velocity.eval_command import EvaluationCommand, EvaluationCommandCfg
    from ..deploy.observations import ObservationBuilder, RobotState, Reference
    from ..deploy.reference import OnlineReference
    from ..deploy.safety import Limits, target_fault, health_fault, acceptance_signature
    from .safety import condition_target, validate_sim_acceptance, TARGET_PROJECTION_ROUNDING
    from .command import StoneCommand, StoneCommandCfg
    from .task import env_config, floor_contact
    from .labels import rgb_only
    from .runtime import StoneRuntime, load_torch_control
    from .route_audit import RouteAudit
    from .camera_clock import CameraClock
    from ..velocity.step_quality_observer import StepQualityObserver
    from ..velocity.recovery_terrain import profile_success
    torch.set_num_threads(2)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    settings = json.loads(Path(args.config).read_text())
    if (settings['stones'].get('deployment_targets',{}).get('enabled',False)
            and args.mode == 'portable' and not args.actuator_guards):
        p.error('This training variant requires --actuator-guards for portable deployment trials')
    settings['seed'] = args.seed
    settings['bank_root'] = str(Path(__file__).resolve().parents[3]/'data/smoke')
    cfg = env_config(settings['bank_root']+'/rank_0', settings, 20)
    gap_enabled = bool(settings['stones'].get('gap_crossing'))
    if gap_enabled:
        from .gap_crossing_task import configure, install
        configure(cfg, settings, rewards=False)
    tile = next(i for i,t in enumerate(cfg.commands['motion'].random_tiles)
                if t['family'] == args.family and t['course_level'] == args.level
                and t['variant'] == args.terrain_variant and t['kind'] == 'up_down')
    case = dict(tile=tile, stream=settings['planner_speeds'].index(args.speed), speed=args.speed,
                phase=0, spawn_xy=[-.1, 0.])

    class EvalCommand(EvaluationCommand, StoneCommand):
        def _init_task(self):
            super()._init_task()
            self.ppo_online[:] = True
            self.ppo_mask[:] = True
            self.dagger_mask[:] = False
            self.dagger_planner[:] = False
            self.gap_bounds = torch.tensor([t['gaps'][0] if t.get('family') == 'stair_gap'
                                           else [-1e5, -1e5] for t in self.cfg.random_tiles], device=self.device)

    @dataclass(kw_only=True)
    class EvalCfg(EvaluationCommandCfg, StoneCommandCfg):
        def build(self, env): return EvalCommand(self, env)

    old = cfg.commands['motion']
    params = {f.name:getattr(old, f.name) for f in fields(EvalCfg) if hasattr(old, f.name)}
    params.update(cases=tuple(dict(case, id=i) for i in range(20)), pose_range={}, velocity_range={}, joint_position_range=(0.,0.))
    cfg.commands['motion'] = EvalCfg(**params)
    if args.capture_video:
        from ..hiking_gaussgym.simulation import add_review_camera
        from dataclasses import replace
        camera_tile=copy.deepcopy(cfg.commands['motion'].random_tiles[tile])
        if camera_tile.get('solids'):
            camera_tile.update(base=max(s['top'] for s in camera_tile['solids']),edges=[],jumps=[])
        add_review_camera(cfg,camera_tile)
        cfg.scene.sensors=tuple(replace(s,width=960,height=540) if s.name=='review_camera' else s for s in cfg.scene.sensors)
    cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs'].rows = ()
    strip_domain(cfg)
    cfg.events.pop('policy_update_counter', None)
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values(): term.noise = None
    cfg.terminations = {}
    cfg.rewards = {}
    cfg.episode_length_s = args.duration+10
    rgb_only(cfg)
    runtime = StoneRuntime(args.bundle, 'cpu')
    builder = ObservationBuilder(runtime.contract)
    builder.heading = Rotation.identity()
    class Observed(ManagerBasedRlEnv):
        def __init__(self,cfg,**kwargs):
            super().__init__(cfg,**kwargs)
            self.quality=StepQualityObserver(self)
            previous=self.metrics_manager.compute_substep
            def substep():
                previous();self.quality.observe_substep()
                if hasattr(self, 'actuation_audit'):
                    self.actuation_audit.observe_substep()
            self.metrics_manager.compute_substep=substep
        def step(self,actions):
            self.quality.begin_step()
            return super().step(actions)
        def _reset_idx(self,ids=None):
            if hasattr(self,'quality'):self.quality.reset(ids)
            return super()._reset_idx(ids)

    env = RslRlVecEnvWrapper(Observed(cfg, device='cuda:0'))
    if gap_enabled: install(env.unwrapped, settings)
    report = dict(mode=args.mode, bundle=str(Path(args.bundle).resolve()), family=args.family, level=args.level,
                  seed=args.seed,terrain_variant=args.terrain_variant,command_speed_m_s=args.speed,
                  truth_used_as_actor_input=False, hardware_test=False, padded_physics_worlds=20,
                  scored_worlds=1, camera_encoder_used=args.mode=='portable',
                  requested_seconds=args.duration, stop_hold_seconds=args.stop_hold_seconds,
                  no_physics_guards_applied=not args.actuator_guards,
                  bundle_files=runtime.manifest['files'],
                  acceptance_signature=acceptance_signature(runtime.manifest),
                  limits=asdict(Limits()), target_guard_counts={})
    report['target_projection_rounding'] = TARGET_PROJECTION_ROUNDING
    errors = {}
    max_action_error = 0.
    fall = False
    video = None
    try:
        obs, _ = env.reset()
        raw = env.unwrapped
        command = raw.command_manager.get_term('motion')
        robot = raw.scene['robot']
        route=RouteAudit(raw)
        complete_time=None
        actor, memory = load_torch_control(runtime.manifest['checkpoint'], obs, settings, raw.device)
        reset = torch.ones(20, dtype=torch.bool, device=raw.device)
        planner = OnlineReference(Path(args.bundle)/'planner.onnx', runtime.contract) if args.mode=='portable' else None
        camera_clock = CameraClock(args.camera_fps, raw.step_dt) if planner else None
        encoder_times, policy_times = [], []
        if planner:
            root = robot.data.default_root_state.clone()
            root[:, :2] = command.tile_origins[command.tile, :2]
            root[:, 2] = command.ground_height(root[:, :2])+runtime.contract['root_height']
            root[:, 3:7] = torch.tensor([1.,0.,0.,0.], device=raw.device)
            root[:, 7:] = 0
            robot.write_root_state_to_sim(root)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_pos))
            raw.sim.forward()
            raw.scene.update(raw.step_dt)
            raw.sim.sense()
            raw.observation_manager.reset()
            from tensordict import TensorDict
            obs = TensorDict(raw.observation_manager.compute(update_history=True), batch_size=[raw.num_envs], device=raw.device)
        from .actuation_audit import ActuationAudit
        raw.actuation_audit = ActuationAudit(raw, runtime.contract)
        if args.capture_video:
            from .portable_video import PortableVideo
            video=PortableVideo(raw,out,args.bundle,args.family,args.level)
        trace = []
        speed = 0.
        applied_steps = 0
        previous = robot.data.joint_pos[0].cpu().numpy().copy()
        max_projection = max_slew = 0.
        for step in range(round(args.duration/raw.step_dt)):
            elapsed = step*raw.step_dt
            state = RobotState(robot.data.joint_pos[0].cpu().numpy(), robot.data.joint_vel[0].cpu().numpy(),
                               robot.data.root_link_quat_w[0].cpu().numpy(), robot.data.root_link_ang_vel_b[0].cpu().numpy(), elapsed)
            if planner:
                wanted = args.speed if 1. <= elapsed < args.duration-args.stop_hold_seconds else 0.
                speed += float(np.clip(wanted-speed, -.01, .01))
                if step%10 == 0: planner.replan(elapsed, speed)
                reference = planner.sample(elapsed)
                fresh, image_at = camera_clock.tick(elapsed)
                if fresh:
                    rgb = raw.scene['head_cam'].data.rgb[0].cpu().numpy()
                    before_encode = time.perf_counter()
                    patches, cls = runtime.encode(rgb[..., ::-1].copy())
                    encoder_times.append(time.perf_counter()-before_encode)
            else:
                idx = command.future_frames(10,5)[0]
                reference = Reference(command.motion.joint_pos[idx].cpu().numpy(), command.motion.joint_vel[idx].cpu().numpy(),
                                      command.motion.body_quat_w[idx,0].cpu().numpy(), command.anchor_lin_vel_w[0].cpu().numpy(),
                                      command.anchor_ang_vel_w[0].cpu().numpy())
                speed = float(command.speed[0])
                patches = obs['kv_tokens','img_tokens'][0].cpu().numpy()
                cls = obs['q_cls'][0].cpu().numpy()
                image_at = elapsed
            rebuilt = builder.build(state, reference, patches, cls, speed, velocity=True)
            with torch.inference_mode():
                augmented = obs.clone(False)
                augmented['hiking_latent'] = memory.step(obs, reset)
                actions = actor(augmented, stochastic_output=False)
            if not planner:
                for name, value in rebuilt.items():
                    expected = obs['kv_tokens','img_tokens'] if name=='kv_tokens__img_tokens' else obs[name]
                    errors[name] = max(errors.get(name,0.), float(np.abs(value-expected[:1].cpu().numpy()).max()))
            action, target, inference_s = runtime.act(rebuilt)
            policy_times.append(inference_s)
            if not planner:
                max_action_error = max(max_action_error, float(np.abs(action-actions[0].cpu().numpy()).max()))
            projection = slew = 0.
            reason = None
            if args.actuator_guards:
                # Only measured-equivalent joints/IMU enter these checks. Simulated
                # camera/state freshness cannot establish real hardware timing.
                reason = health_fault(state, elapsed, image_at, elapsed, Limits())
                try:
                    target, projection, slew = condition_target(target, previous, runtime.contract, Limits())
                except ValueError as exc:
                    reason = reason or str(exc)
                reason = reason or target_fault(target, previous, state, runtime.contract, Limits())
                if reason:
                    report['target_guard_counts'][reason] = report['target_guard_counts'].get(reason, 0)+1
                    report.update(guard_stopped_at_s=elapsed, guard_stop_reason=reason,
                                  guard_joint_diagnostic=dict(
                                      tracking_joint=runtime.contract['joint_names'][int(np.argmax(abs(target-state.q)))],
                                      max_target_error_rad=float(abs(target-state.q).max()),
                                      max_target_step_rad=float(abs(target-previous).max()),
                                      target=target.tolist(), previous=previous.tolist(), measured=state.q.tolist()))
                    break
                action = (target-runtime.default)/runtime.scale
            max_projection = max(max_projection, projection)
            max_slew = max(max_slew, slew)
            builder.accept_action(action)
            previous = target.copy()
            with torch.inference_mode():
                actions[0] = torch.as_tensor(action, device=raw.device)
                obs, _, done, _ = env.step(actions)
            applied_steps += 1
            reset = done.bool()
            if done.any(): raise RuntimeError('Unexpected auto-reset in deployment check')
            root = robot.data.root_link_pos_w
            local = root-command.tile_origins[command.tile]
            fall = bool((robot.data.projected_gravity_b[0,2]>-.45) | ((root[:,2]-command.ground_height(root[:,:2]))[0]<.25) | floor_contact(raw)[0])
            route.update(torch.ones(20,dtype=torch.bool,device=raw.device))
            gap_qualified = not gap_enabled or bool(raw.gap_crossing.qualified[0])
            if complete_time is None and bool((profile_success(raw)&route.qualified)[0]) and not fall and gap_qualified:
                complete_time=(step+1)*raw.step_dt
            if planner or step%10==0 or fall:
                trace.append(dict(t=elapsed+raw.step_dt, command=speed, speed_command=speed,
                                  position=local[0].cpu().tolist(), fall=fall,
                                  measured_velocity_world_m_s=robot.data.root_link_lin_vel_w[0].cpu().tolist(),
                                  image_age_s=elapsed-image_at,
                                  limit_projection_rad=projection, slew_projection_rad=slew))
            if video and step%2==1:video.append(rgb,(step+1)*raw.step_dt,speed)
            if fall: break
        passed = args.mode!='parity' or (max(errors.values(),default=0.)<2e-5 and max_action_error<5e-4)
        report.update(steps=applied_steps, simulated_seconds=applied_steps*raw.step_dt, fall=fall,
                      observation_max_abs_errors=errors, action_max_abs_error=max_action_error,
                      observation_contract_passed=passed if args.mode=='parity' else None, trace=trace,
                      initial_state='stationary_default_pose' if planner else 'training_walking_RSI',
                      deploy_onnx_drove_scored_world=True,
                      deploy_onnx_drove_physics=True, raw_rgb_encoder_used=bool(planner),
                      max_limit_projection_rad=max_projection, max_slew_projection_rad=max_slew,
                      course_success=complete_time is not None and not fall,course_completion_time_s=complete_time,
                      route_audit=route.row(0),
                      actuation_audit=raw.actuation_audit.summary(),
                      note='Portable mode is a simulation trial, not hardware approval; parity uses matching RGB features to isolate observation construction.')
        if gap_enabled:
            report.update(verifies_airborne_gap_crossing=True,
                          gap_crossing_settings=settings['stones']['gap_crossing'],
                          gap_crossing_audit=raw.gap_crossing.row(0))
            report['course_success'] &= bool(raw.gap_crossing.qualified[0])
        if planner:
            report['camera_cadence'] = camera_clock.summary()
            def timing(values):
                return dict(samples=len(values),p50_ms=float(np.percentile(values,50))*1000,
                            p99_ms=float(np.percentile(values,99))*1000,max_ms=max(values)*1000)
            report['onnx_component_wall_times'] = dict(encoder=timing(encoder_times),
                memory_and_policy=timing(policy_times),
                includes_simulator_io=False,real_time_scheduler_validated=False)
            # Compatible artifact for the existing real-control admission check.
            # "keyboard" denotes the scripted W press/release; it is not live hardware.
            acceptance = dict(report, mode='keyboard', scripted_keyboard=True)
            acceptance_path = out/'actuator_acceptance.json'
            acceptance_path.write_text(json.dumps(acceptance, indent=2)+'\n')
            try:
                validate_sim_acceptance(acceptance_path, runtime.manifest, args.speed)
                report['standing_start_stop_sim_passed'] = True
            except ValueError as exc:
                report.update(standing_start_stop_sim_passed=False, start_stop_rejection=str(exc))
            if trace:
                final = trace[-1]
                stopped = [r for r in trace if final['t']-r['t'] <= 1.00001]
                report['last_second'] = dict(
                    duration_s=final['t']-stopped[0]['t'],
                    maximum_speed_command_m_s=max(abs(r['speed_command']) for r in stopped),
                    xy_displacement_m=float(np.linalg.norm(np.array(final['position'][:2])-stopped[0]['position'][:2])),
                    final_velocity_world_m_s=final['measured_velocity_world_m_s'])
            from .outcome import portable_outcome
            report.update(portable_outcome(report))
            # Preserve the original start/stop admission semantics, while making
            # task completion explicit in both artifacts. A stalled robot that
            # moves 0.5 m and then stops has NOT traversed the route.
            acceptance_path.write_text(json.dumps(dict(report, mode='keyboard',
                scripted_keyboard=True), indent=2)+'\n')
        if video:report['video']=video.finish(report,rgb,speed)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='trace'},indent=2),flush=True)
        if not passed: raise RuntimeError('Deployment observation/action contract failed')
    finally:
        if video:video.close()
        env.close()


if __name__ == '__main__': main()
