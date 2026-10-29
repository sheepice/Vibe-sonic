"""Held-out velocity trials and fixed-camera SONIC/ViBe comparisons with actual RGB."""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields
import itertools
import json
from pathlib import Path

from ..gpu_guard import verify_guarded_worker


def main(guard_check=verify_guarded_worker):
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=170017)
    p.add_argument('--duration', type=float, default=30.)
    p.add_argument('--record-case', type=int)
    p.add_argument('--record-speed', type=float, help='Recording only: use an existing planner stream at this constant command speed')
    p.add_argument('--label', default='ViBe velocity | update 1 (interrupted run)')
    p.add_argument('--visual-ablation', choices=('none', 'zero_tokens'), default='none')
    p.add_argument('--initial-vibe', action='store_true', help='Load previous ViBe with the new command branch kept exactly zero')
    p.add_argument('--irregular', action='store_true', help='Evaluate held-out irregular profiles with their exact height function')
    p.add_argument('--uniform-probe-25cm', action='store_true', help='Evaluation only: three new regular six-riser routes, exactly 25 cm per riser')
    p.add_argument('--uniform-probe-30cm', action='store_true', help='Evaluation only: same layout as the 25 cm probe with exactly 30 cm risers')
    p.add_argument('--posture-diagnostics', action='store_true', help='Measure separate torso, waist, arm and speed tracking before each terminal')
    args = p.parse_args()
    if args.record_speed is not None and args.record_case is None:
        p.error('--record-speed requires --record-case')
    if sum((args.uniform_probe_25cm, args.uniform_probe_30cm, args.irregular)) > 1:
        p.error('--uniform-probe-25cm, --uniform-probe-30cm and --irregular are mutually exclusive')
    uniform_probe = args.uniform_probe_25cm or args.uniform_probe_30cm
    guard_check()
    import mjlab  # noqa: F401
    import numpy as np
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from orcs.core.robustness import strip_domain
    from rsl_rl.utils import resolve_callable
    from ..catalog import write_json
    from .config import agent_config, env_config
    from .eval_command import EvaluationCommandCfg

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    settings = json.loads(Path(args.config).read_text())
    settings['seed'] = args.seed  # New stair geometry AND planner seed, absent from training.
    bank = str(Path(settings['bank_root']) / 'rank_0')
    cases = []
    for level, kind, stream, repeat in itertools.product([0, 1, 3, 5, 7], range(3), [1, 2, 3], range(4)):
        cases.append(dict(id=len(cases), tile=level * 4 + kind, stream=stream,
                          speed=settings['planner_speeds'][stream], phase=[0, 5, 10, 15][repeat],
                          spawn_xy=[[-.15, -.15], [0., .1], [.15, -.1], [0., .2]][repeat]))
    for stream, repeat in itertools.product([1, 2, 3], range(4)):
        cases.append(dict(id=len(cases), tile=3, stream=stream,
                          speed=settings['planner_speeds'][stream], phase=repeat * 5,
                          spawn_xy=[0., (repeat - 1.5) * .1]))
    command_type = EvaluationCommandCfg
    if args.irregular:
        if settings.get('terrain_profile') != 'irregular_v1':
            raise ValueError('--irregular requires the irregular_v1 training configuration')
        from .eval_profile_command import ProfileEvaluationCommandCfg, profile_cases
        command_type = ProfileEvaluationCommandCfg
        cases = profile_cases(settings)
    elif settings.get('terrain_profile'):
        raise ValueError('Use --irregular for profile terrain; uniform height queries would be invalid')
    if uniform_probe:
        from .uniform_probe import probe_cases
        cases = probe_cases(settings)
    if args.record_case is not None:
        cases = [copy.deepcopy(cases[args.record_case]) for _ in range(2)]
        if args.record_speed is not None:
            matches = [i for i, speed in enumerate(settings['planner_speeds'])
                       if abs(speed - args.record_speed) < 1e-9]
            if len(matches) != 1:
                raise ValueError('--record-speed must match one configured planner stream')
            for case in cases:
                case.update(speed=args.record_speed, stream=matches[0])
        cases[0]['policy'] = 'frozen_sonic'
        cases[1]['policy'] = 'velocity_vibe'
    else:
        for c in cases:
            c['policy'] = 'velocity_vibe'
    n = len(cases)
    cfg = env_config(bank, settings, n)
    probe_metadata = None
    if args.uniform_probe_25cm:
        from .uniform_probe import apply_uniform_probe25
        probe_metadata = apply_uniform_probe25(cfg)
    elif args.uniform_probe_30cm:
        from .uniform_probe30 import apply_uniform_probe30
        probe_metadata = apply_uniform_probe30(cfg)
    if probe_metadata is not None and args.record_speed is not None:
        probe_metadata['command_m_s'] = args.record_speed
    strip_domain(cfg)
    cfg.events.pop('policy_update_counter', None)
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            term.noise = None
    # All worlds use generated references. Distant replay stairs need not be rendered.
    atlas = cfg.scene.terrain.terrain_generator.sub_terrains['paired_stairs']
    atlas.rows = ()
    old = cfg.commands['motion']
    common = {f.name: getattr(old, f.name) for f in fields(command_type) if hasattr(old, f.name)}
    common.update(cases=tuple(cases), pose_range={}, velocity_range={}, joint_position_range=(0., 0.))
    cfg.commands['motion'] = command_type(**common)
    cfg.terminations.clear()
    cfg.rewards = {}
    cfg.episode_length_s = args.duration + 5
    camera_spec = None
    if args.record_case is not None:
        from mjlab.sensor import CameraSensorCfg
        from scipy.spatial.transform import Rotation
        tile = atlas.random_tiles[cases[0]['tile']]
        height = tile.get('total_height', tile['n'] * tile['height'])
        camera_half_width = 2.1 if uniform_probe else 1.
        lower, upper = np.array([-.6, -camera_half_width, -.1]), np.array([tile['goal'] + .9, camera_half_width, height + 1.25])
        center = (lower + upper) / 2
        z = np.array([-.55, -1., .6]); z /= np.linalg.norm(z)
        x = np.cross([0., 0., 1.], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rotation = np.stack([x, y, z], axis=1)
        corners = np.array(list(itertools.product(*zip(lower, upper))))
        projected = (corners - center) @ rotation
        tan_y = np.tan(np.deg2rad(43 / 2))
        distance = max(np.max(np.abs(projected[:, 0]) / (tan_y * 640 / 360) + projected[:, 2]),
                       np.max(np.abs(projected[:, 1]) / tan_y + projected[:, 2])) * 1.04
        size = cfg.scene.terrain.terrain_generator.size
        origin = np.array(tile['origin']) + [-size[0] / 2, -size[1] / 2, 0.]
        position = center + origin + distance * z
        quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        cfg.scene.sensors = (*cfg.scene.sensors, CameraSensorCfg(
            name='review_camera', parent_body=None, pos=tuple(position), quat=tuple(quat),
            width=640, height=360, fovy=43., data_types=('rgb',), enabled_geom_groups=(0, 2),
            use_shadows=False, use_textures=True))
        camera_spec = dict(position=position.tolist(), quaternion=quat.tolist(), parent='world')

    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device='cuda:0'))
    try:
        agent = agent_config(bank, settings, n)
        actor_cfg = copy.deepcopy(agent['actor'])
        cls = resolve_callable(actor_cfg.pop('class_name'))
        actor = cls(obs=env.get_observations(), obs_groups=agent['obs_groups'], obs_set='actor',
                    output_dim=29, **actor_cfg).to(env.device).eval()
        checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
        missing, unexpected = actor.load_state_dict(checkpoint['actor_state_dict'], strict=not args.initial_vibe)
        if args.initial_vibe:
            expected = {'command_encoder.0.weight', 'command_encoder.0.bias',
                        'command_encoder.2.weight', 'command_encoder.2.bias'}
            assert set(missing) == expected and not unexpected
            assert not actor.command_encoder[-1].weight.any() and not actor.command_encoder[-1].bias.any()
        checkpoint_iteration = checkpoint.get('iter')
        del checkpoint
        obs, _ = env.reset()
        raw = env.unwrapped
        c = raw.command_manager.get_term('motion')
        robot = raw.scene['robot']
        posture_recorder = posture_summary = None
        if args.posture_diagnostics:
            from .posture_diagnostics import PostureDiagnostics
            from .posture_summary import PostureSummary
            posture_recorder = PostureDiagnostics(c)
            posture_summary = PostureSummary(n, env.device, keep_trace=args.record_case is not None)
        initial = robot.data.root_link_pos_w.cpu().numpy().copy()
        tile_origin = c.tile_origins[c.tile].cpu().numpy().copy()
        tile_data = [atlas.random_tiles[case['tile']] for case in cases]
        goals = np.array([t['goal'] for t in tile_data])
        exit_x = goals - .6
        held = np.zeros(n, int)
        max_gain = np.zeros(n)
        max_progress = np.zeros(n)
        sum_error = np.zeros(n)
        samples = np.zeros(n, int)
        terminals = {}
        trajectory, frames, rgb_frames = [], [], []
        camera_drift = 0.
        if camera_spec:
            camera = raw.scene['review_camera']
            ci = camera.camera_idx
            assert raw.sim.mj_model.cam_bodyid[ci] == 0
            camera_position = raw.sim.data.cam_xpos[:, ci].clone()
            camera_rotation = raw.sim.data.cam_xmat[:, ci].clone()
            assert float((robot.data.joint_pos[0] - robot.data.joint_pos[1]).abs().max()) < 1e-7
            assert float((robot.data.root_link_pos_w[0] - robot.data.root_link_pos_w[1]).abs().max()) < 1e-7
        max_steps = round(args.duration * 50)
        with torch.inference_mode():
            for step in range(max_steps + 1):
                if posture_recorder is not None:
                    active = torch.tensor([i not in terminals for i in range(n)], device=env.device)
                    posture_summary.update(posture_recorder.sample(elapsed_dt_s=.02), step / 50, active)
                root = robot.data.root_link_pos_w.cpu().numpy().copy()
                feet_t = c.robot_body_pos_w[:, [3, 6]]
                feet = feet_t.cpu().numpy().copy()
                ground = c.ground_height(feet_t[:, :, :2]).cpu().numpy()
                root_ground = c.ground_height(robot.data.root_link_pos_w[:, :2]).cpu().numpy()
                gravity = robot.data.projected_gravity_b[:, 2].cpu().numpy()
                velocity = robot.data.root_link_lin_vel_w.cpu().numpy()
                local = root - tile_origin
                feet_x = feet[:, :, 0] - tile_origin[:, None, 0]
                gain = root[:, 2] - initial[:, 2]
                passed = (local[:, 0] >= goals) & (feet_x.min(1) >= exit_x + .15)
                stable = ((np.abs(feet[:, :, 2] - ground).max(1) < .18) & (gravity < -.85)
                          & (np.abs(local[:, 1]) < 1.5) & ((root[:, 2] - root_ground) > .45))
                held = np.where(passed & stable, held + 1, 0)
                for i, case in enumerate(cases):
                    if i in terminals:
                        continue
                    max_gain[i] = max(max_gain[i], gain[i])
                    max_progress[i] = max(max_progress[i], local[i, 0])
                    sum_error[i] += np.linalg.norm(velocity[i, :2] - [case['speed'], 0])
                    samples[i] += 1
                    reason = None
                    if not np.isfinite(root[i]).all():
                        reason = 'nonfinite_state'
                    elif gravity[i] > -.45 or root[i, 2] - root_ground[i] < .32:
                        reason = 'physical_fall'
                    elif abs(local[i, 1]) > 1.5:
                        reason = 'left_stair_lane'
                    elif held[i] >= 25:
                        reason = 'complete_crossing_stable_0.5s'
                    elif step == max_steps:
                        reason = 'time_limit'
                    if reason:
                        terminals[i] = dict(case=case, terrain=tile_data[i], success=bool(held[i] >= 25),
                                            reason=reason, step=step, time_s=step / 50,
                                            root_local=local[i].tolist(), height_change_m=float(gain[i]),
                                            max_height_gain_m=float(max_gain[i]),
                                            max_forward_m=float(max_progress[i]),
                                            mean_velocity_error_m_s=float(sum_error[i] / samples[i]))
                if camera_spec and step % 2 == 0:
                    frames.append(camera.data.rgb.cpu().numpy().copy())
                    rgb_frames.append(raw.scene['head_cam'].data.rgb.cpu().numpy().copy())
                    trajectory.append(dict(time_s=step / 50, root_local=local.tolist(),
                                           height_change_m=gain.tolist(), velocity=velocity.tolist()))
                    camera_drift = max(camera_drift,
                                       float((raw.sim.data.cam_xpos[:, ci] - camera_position).abs().max()),
                                       float((raw.sim.data.cam_xmat[:, ci] - camera_rotation).abs().max()))
                if step % 100 == 0:
                    print(f'STEP {step}: ended={len(terminals)}/{n} successes={sum(t["success"] for t in terminals.values())}', flush=True)
                if len(terminals) == n:
                    break
                if args.visual_ablation == 'zero_tokens':
                    obs = obs.clone()
                    obs['kv_tokens']['img_tokens'].zero_()
                    obs['q_cls'].zero_()
                actions = actor(obs)
                if camera_spec:
                    actions[0] = actor.sonic_action(obs)[0]
                obs, _, done, _ = env.step(actions)
                assert not done.any(), 'Evaluation must never auto-reset'
        rows = [terminals[i] for i in range(n)]
        grouped = {}
        for row in rows:
            key = f'{row["case"]["policy"]}/{row["terrain"]["kind"]}/level_{row["terrain"]["level"]}'
            entry = grouped.setdefault(key, dict(total=0, success=0, falls=0))
            entry['total'] += 1
            entry['success'] += int(row['success'])
            entry['falls'] += int(row['reason'] == 'physical_fall')
        report = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_iteration=checkpoint_iteration,
                      terrain_profile=probe_metadata['schema'] if probe_metadata else settings.get('terrain_profile', 'uniform_v1'),
                      label=args.label, seed=args.seed, duration_s=args.duration, cases=rows, groups=grouped,
                      visual_ablation=args.visual_ablation,
                      initial_vibe_without_velocity_training=args.initial_vibe,
                      actor_receives_rgb=True, actor_receives_depth=False, terrain_actor_input=False,
                      prerecorded_stair_reference=False, no_auto_reset=True,
                      planner='5Hz runtime SONIC; previous generated context; no terrain/visual input',
                      command='constant forward speed and world +X heading',
                      initialization='generated flat walking phase on starting landing; no stair demonstration',
                      criterion='root beyond end+0.6m, both feet beyond end+0.15m, upright and feet near ground for 0.5s',
                      camera=camera_spec, observer_max_pose_drift=camera_drift)
        if args.record_speed is not None:
            report['record_speed_override_m_s'] = args.record_speed
        if probe_metadata is not None:
            import hashlib
            report['uniform_probe'] = probe_metadata
            with Path(args.checkpoint).open('rb') as checkpoint_file:
                report['checkpoint_sha256'] = hashlib.file_digest(checkpoint_file, 'sha256').hexdigest()
        write_json(destination / 'report.json', report)
        if posture_summary is not None:
            write_json(destination / 'posture_summary.json', posture_summary.finalize())
            if args.record_case is not None:
                write_json(destination / 'posture_trajectory.json', posture_summary.trace)
        if camera_spec:
            assert camera_drift < 1e-7
            import imageio.v2 as iio
            from PIL import Image, ImageDraw, ImageFont
            font, small = ImageFont.load_default(size=18), ImageFont.load_default(size=15)

            def panel(frame, world):
                source = min(frame, len(frames) - 1, terminals[world]['step'] // 2)
                image = Image.new('RGB', (640, 540), (18, 23, 32))
                image.paste(Image.fromarray(frames[source][world]), (0, 40))
                image.paste(Image.fromarray(rgb_frames[source][world]).resize((224, 126)), (8, 405))
                d = ImageDraw.Draw(image)
                title = 'Frozen SONIC | same online walking plan' if world == 0 else args.label
                d.text((10, 10), title, font=font, fill='white')
                ended = frame * 2 >= terminals[world]['step']
                status = ('SUCCESS' if terminals[world]['success'] else terminals[world]['reason'].upper()) if ended else 'RUNNING'
                t = trajectory[source]
                tile = tile_data[world]
                color = (110, 230, 145) if ended and terminals[world]['success'] else (255, 205, 150)
                d.text((242, 407), f't={t["time_s"]:.2f}s | {status}', font=small, fill=color)
                if 'risers' in tile:
                    risers = tile['risers']
                    terrain_label = f'{tile["kind"]} | {min(risers)*100:.1f}-{max(risers)*100:.1f}cm | {tile["n"]} steps/flight'
                    d.text((242, 435), terrain_label, font=small, fill='white')
                else:
                    d.text((242, 435), f'{tile["kind"]} | riser {tile["height"] * 100:.1f} cm', font=font, fill='white')
                d.text((242, 463), f'W: {cases[world]["speed"]:.2f} m/s | x={t["root_local"][world][0]:.2f}m', font=font, fill='white')
                d.text((242, 491), 'Actual head RGB | fixed world camera', font=small, fill='white')
                d.text((242, 516), 'Frame held after terminal' if ended else 'No recorded stair reference', font=small, fill='white')
                return np.asarray(image)

            paths = [destination / 'sonic_vs_velocity_vibe_fixed_rgb.mp4', destination / 'velocity_vibe_fixed_rgb.mp4']
            writers = [iio.get_writer(str(path), fps=25, codec='libx264', quality=8, macro_block_size=1,
                                     ffmpeg_params=['-threads', '1', '-pix_fmt', 'yuv420p']) for path in paths]
            try:
                for f in range(len(frames) + 25):
                    a, b = panel(f, 0), panel(f, 1)
                    writers[0].append_data(np.concatenate([a, b], axis=1))
                    if f <= terminals[1]['step'] // 2 + 25:
                        writers[1].append_data(b)
            finally:
                for w in writers:
                    w.close()
            for f in sorted({0, min(50, len(frames) - 1), len(frames) - 1}):
                Image.fromarray(np.concatenate([panel(f, 0), panel(f, 1)], axis=1)).save(destination / f'preview_{f:04d}.jpg')
            write_json(destination / 'trajectory.json', trajectory)
        print('GROUPS', json.dumps(grouped), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
