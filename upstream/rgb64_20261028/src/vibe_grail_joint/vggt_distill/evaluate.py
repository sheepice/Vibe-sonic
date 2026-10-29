"""Held-out stair trials and fixed-camera videos; no VGGT is loaded."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path


def latest_checkpoint(directory):
    files = [p for p in Path(directory).glob('model_*.pt') if p.stem[6:].isdigit()]
    if not files: raise FileNotFoundError(f'No complete model checkpoint in {directory}')
    return max(files, key=lambda p: int(p.stem[6:]))


def load_actor(settings, obs, checkpoint, raw):
    import torch
    from .task import agent_config
    from .model import DistilledActor
    state = torch.load(checkpoint, map_location=raw.device, weights_only=False)
    metadata = state.get('infos', {}).get('vggt_distillation')
    if not metadata or metadata['settings'] != settings['distillation'] or metadata['camera'] != settings['camera']:
        raise ValueError('Checkpoint and evaluation architecture/configuration do not match')
    cfg = agent_config(settings['bank_root'] + '/rank_0', settings, 20)
    acfg = dict(cfg['actor']); acfg.pop('class_name')
    actor = DistilledActor(obs=obs, obs_groups=cfg['obs_groups'], obs_set='actor', output_dim=29, **acfg).to(raw.device).eval()
    actor.load_state_dict(state['actor_state_dict'], strict=True)
    # Verify and physically remove the training-only modules before evaluation.
    with torch.inference_mode(): before = actor(obs, stochastic_output=False)
    del actor.alignment_heads, actor.relation_head, actor.progress_head
    with torch.inference_mode(): after = actor(obs, stochastic_output=False)
    if not torch.equal(before, after): raise RuntimeError('Removing auxiliary heads changed deployed actions')
    return actor, int(state['iter'])


class VisualIntervention:
    def __init__(self, mode, delay, rgb_shape=None):
        self.mode = mode; self.history = deque(maxlen=delay + 1)
        self.rgb_shape = rgb_shape
        self.black_pair = None
        self.applied_steps = 0

    def _black_features(self, obs):
        import torch
        from ..vision import FrozenTheiaFeature
        from vibe.core.mdp.observations import image_feature
        if self.rgb_shape is None:
            raise ValueError('Black RGB intervention needs the actual camera shape')
        device = obs['kv_tokens']['img_tokens'].device
        if self.black_pair is None:
            model = FrozenTheiaFeature._load('theia', 'theia-tiny-patch16-224-cddsv', device, torch.float16)
            mean, std = image_feature._NORM['imagenet']
            mean = torch.tensor(mean, device=device, dtype=torch.float16)[None, :, None, None]
            std = torch.tensor(std, device=device, dtype=torch.float16)[None, :, None, None]
            black = torch.zeros(1, 3, *self.rgb_shape, device=device, dtype=torch.float16)
            with torch.inference_mode():
                features = model.backbone.model(pixel_values=(black-mean)/std,
                    interpolate_pos_encoding=True).last_hidden_state.float()
            self.black_pair = features[:, 1:].clone(), features[:, 0].clone()
        n = obs['kv_tokens']['img_tokens'].shape[0]
        tokens, cls = self.black_pair
        assert tokens.shape[1:] == obs['kv_tokens']['img_tokens'].shape[1:]
        return tokens.expand(n, -1, -1), cls.expand(n, -1)

    def __call__(self, obs):
        obs = obs.clone()
        pair = (obs['kv_tokens']['img_tokens'].clone(), obs['q_cls'].clone())
        if self.mode != 'freeze' or not self.history: self.history.append(pair)
        if self.mode in ('delay', 'freeze'):
            obs['kv_tokens']['img_tokens'], obs['q_cls'] = self.history[0]
        elif self.mode == 'zero_tokens':
            obs['kv_tokens']['img_tokens'].zero_(); obs['q_cls'].zero_()
        elif self.mode == 'black_rgb':
            # A frozen encoder is deterministic: cache the output of the same
            # all-black image, replacing BOTH visual routes on every step.
            obs['kv_tokens']['img_tokens'], obs['q_cls'] = self._black_features(obs)
        elif self.mode == 'swap_worlds':
            if len(pair[0]) < 2: raise ValueError('World swap needs at least two cases')
            obs['kv_tokens']['img_tokens'], obs['q_cls'] = pair[0].roll(1, 0), pair[1].roll(1, 0)
        self.applied_steps += 1
        return obs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint'); group.add_argument('--run-dir')
    p.add_argument('--output', required=True); p.add_argument('--duration', type=float, default=45.)
    p.add_argument('--seed', type=int, default=260027)
    p.add_argument('--heights', default='.12,.16,.20,.25,.30')
    p.add_argument('--speeds', default='.25,.4,.7'); p.add_argument('--approaches', default='0,2')
    p.add_argument('--irregular', action='store_true'); p.add_argument('--record-case', type=int)
    p.add_argument('--route', choices=('up', 'down', 'up_down'), default='up')
    p.add_argument('--completion-mode', choices=('strict', 'walking'),
                   help='Default: walking for videos, strict for the original benchmark')
    p.add_argument('--limit-cases', type=int)
    p.add_argument('--intervention', choices=('normal', 'black_rgb', 'zero_tokens', 'delay', 'freeze', 'swap_worlds'), default='normal')
    p.add_argument('--delay-steps', type=int, default=5)
    args = p.parse_args()
    args.completion_mode = args.completion_mode or ('walking' if args.record_case is not None else 'strict')
    if args.duration <= 0 or args.delay_steps < 1: p.error('Positive duration and delay required')
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.run_dir)
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import numpy as np
    import torch
    from .scenes import evaluation_env
    from ..catalog import write_json
    from ..velocity.posture_diagnostics import PostureDiagnostics
    from ..velocity.posture_summary import PostureSummary
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    settings = json.loads(Path(args.config).read_text())
    trained_ascent_only = settings.get('task_mode') != 'stairs_original_mixture_v1'
    parse = lambda value: tuple(float(x) for x in value.split(','))
    heights, speeds, approaches = map(parse, (args.heights, args.speeds, args.approaches))
    if min(heights) <= 0 or min(speeds) <= 0 or min(approaches) < 0:
        p.error('Step heights and forward speeds must be positive; approaches must be nonnegative')
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'request.json', dict(args=vars(args), checkpoint=str(checkpoint.resolve()), settings=settings))
    env, settings, cases = evaluation_env(settings, seed=args.seed, duration=args.duration,
        heights=heights, speeds=speeds, approaches=approaches, irregular=args.irregular,
        record_case=args.record_case, limit_cases=args.limit_cases, route=args.route)
    video = None; frame_count = 0
    try:
        obs, _ = env.reset(); raw = env.unwrapped; n = raw.num_envs
        robot = raw.scene['robot']; c = raw.command_manager.get_term('motion')
        actor, iteration = load_actor(settings, obs, checkpoint, raw)
        intervention = VisualIntervention(args.intervention, args.delay_steps,
            (settings['camera']['height'], settings['camera']['width']))
        tiles = [c.cfg.random_tiles[case['tile']] for case in cases]
        if args.route == 'up' and any(t['kind'] != 'up' or any(j <= 0 for j in t['jumps']) for t in tiles):
            raise RuntimeError('Nonascending evaluation case')
        initial = dict(root=robot.data.root_link_pos_w.cpu().tolist(), joints=robot.data.joint_pos.cpu().tolist())
        active = np.ones(n, bool); held = np.zeros(n, int); rows = [None] * n; trace = []
        error_sum = np.zeros(n); counts = np.zeros(n, int)
        goals = np.array([t['goal'] for t in tiles])
        posture = PostureDiagnostics(c); summary = PostureSummary(n, env.device, keep_trace=False)
        max_camera_drift = 0.
        if args.record_case is not None:
            import imageio.v2 as iio
            video = iio.get_writer(out / 'ascent_fixed_rgb.mp4', fps=25, codec='libx264', quality=8, macro_block_size=1)
            ci = raw.scene['review_camera'].camera_idx
            camera_pose = (raw.sim.data.cam_xpos[:, ci].clone(), raw.sim.data.cam_xmat[:, ci].clone())
        with torch.inference_mode():
            for step in range(round(args.duration / raw.step_dt)):
                if video is not None and step % 2 == 0:
                    frame = raw.scene['review_camera'].data.rgb[0].cpu().numpy().copy()
                    rgb = raw.scene['head_cam'].data.rgb[0].cpu().numpy().copy()
                    if args.intervention == 'black_rgb':
                        rgb.fill(0)
                action = actor(intervention(obs), stochastic_output=False)
                if not torch.isfinite(action).all(): raise RuntimeError('Nonfinite evaluation action')
                summary.update(posture.sample(elapsed_dt_s=raw.step_dt), step * raw.step_dt,
                               torch.as_tensor(active, device=env.device))
                obs, _, done, _ = env.step(action)
                if done.any(): raise RuntimeError('Unexpected reset in fixed-case evaluation')
                q = raw.quality
                record = {k: v.cpu().numpy().copy() for k, v in {**q.step_data, **q.geometry()}.items()}
                record.update(action=action.cpu().numpy(), active=active.copy())
                trace.append(record)
                root = robot.data.root_link_pos_w
                local = (root - c.tile_origins[c.tile]).cpu().numpy()
                feet = c.robot_body_pos_w[:, q.feet]; ground = c.ground_height(feet[:, :, :2])
                gravity = robot.data.projected_gravity_b[:, 2].cpu().numpy()
                root_height = (root[:, 2] - c.ground_height(root[:, :2])).cpu().numpy()
                record.update(projected_gravity_z=gravity.copy(), root_ground_clearance_m=root_height.copy())
                passed = (local[:, 0] >= goals) & ((feet[:, :, 0] - c.tile_origins[c.tile, None, 0]).amin(1).cpu().numpy() >= goals - .45)
                support_ok = ((feet[:, :, 2] - ground).abs().amax(1).cpu().numpy() < .18
                              if args.completion_mode == 'strict' else q.sensor_support.any(1).cpu().numpy())
                stable = support_ok & (gravity < -.85) & (abs(local[:, 1]) < 1.5) & (root_height > .45)
                held = np.where(passed & stable, held + 1, 0)
                fallen = (gravity > -.4) | (root_height < .25)
                velocity = robot.data.root_link_lin_vel_w[:, :2].cpu().numpy()
                error_sum += np.linalg.norm(velocity - np.array([[x['speed'], 0] for x in cases]), axis=1) * active
                counts += active
                terminal = active & ((held >= 25) | fallen | (step == round(args.duration / raw.step_dt) - 1))
                for i in np.flatnonzero(terminal):
                    rows[i] = dict(case=cases[i], success=bool(held[i] >= 25), fall=bool(fallen[i]),
                        steps=step + 1, time_s=(step + 1) * raw.step_dt, progress_m=float(local[i, 0]),
                        speed_error_m_s=float(error_sum[i] / counts[i]))
                active[terminal] = False
                if video is not None and step % 2 == 0:
                    from PIL import Image, ImageDraw
                    image = Image.new('RGB', (640, 544), (18, 23, 32))
                    image.paste(Image.fromarray(frame), (0, 30))
                    image.paste(Image.fromarray(rgb).resize((224, 128)), (8, 408))
                    draw = ImageDraw.Draw(image)
                    draw.text((8, 8), f'ViBe+VGGT | {settings["distillation"]["variant"]} | t={step*.02:.2f}s', fill='white')
                    draw.text((8, 392), 'Robot RGB', fill='white')
                    route_label = {'up': 'Upstairs', 'down': 'Downstairs', 'up_down': 'Flat-Up-Down-Flat'}[args.route]
                    draw.text((246, 410), f'{route_label} {cases[0]["nominal_height_m"]*100:.0f} cm | {cases[0]["speed"]} m/s', fill='white')
                    draw.text((246, 434), f'Checkpoint update {iteration} | {args.intervention}', fill='white')
                    draw.text((246, 458), f'Root progress: {local[0,0]:.2f} / {goals[0]:.2f} m', fill='white')
                    label = 'PASS' if rows[0] and rows[0]['success'] else 'FALL' if rows[0] and rows[0]['fall'] else 'TIMEOUT' if rows[0] else 'RUNNING'
                    training_label = 'ascent-only training' if trained_ascent_only else 'mixed-stairs training'
                    draw.text((246, 482), f'{label} | {training_label}', fill='white')
                    video.append_data(np.asarray(image)); frame_count += 1
                    max_camera_drift = max(max_camera_drift,
                        float((raw.sim.data.cam_xpos[:, ci] - camera_pose[0]).abs().max()),
                        float((raw.sim.data.cam_xmat[:, ci] - camera_pose[1]).abs().max()))
                if step % 250 == 0: print(json.dumps(dict(step=step, active=int(active.sum()))), flush=True)
                if not active.any(): break
        arrays = {k: np.stack([r[k] for r in trace]) for k in trace[0]}
        np.savez_compressed(out / 'trace.npz', **arrays)
        for i, row in enumerate(rows):
            a = {k: v[:row['steps'], i] for k, v in arrays.items()}
            margins = a['sole_edge_margin_m'][a['stable_support']]
            foot, shin = a['foot_riser_force_n'] > 15, a['shin_force_n'] > 15
            first = a['root_local'][:, 0] < tiles[i]['edges'][0] + .45
            starts = lambda contacts: contacts & ~np.concatenate((np.zeros((1, 2), bool), contacts[:-1]))
            row.update(foot_impulse_ns=float(a['foot_riser_impulse_ns'].sum()),
                shin_impulse_ns=float(a['shin_impulse_ns'].sum()), foot_strikes=int(starts(foot).sum()), shin_strikes=int(starts(shin).sum()),
                first_step_collision=bool((foot[first] | shin[first]).any()), collision_free=not bool((foot | shin).any()),
                support_slip_speed_m_s=float(a['slip_distance_m'].sum() / max(a['support_seconds'].sum(), 1e-8)),
                edge_margin_p05_m=float(np.quantile(margins, .05)) if len(margins) else None,
                support_overhang_fraction=float((margins < 0).mean()) if len(margins) else None)
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        write_json(out / 'report.json', dict(args=vars(args), checkpoint=str(checkpoint.resolve()), checkpoint_iteration=iteration,
            checkpoint_sha256=digest, variant=settings['distillation']['variant'], cases=rows, initial_state=initial, tiles=tiles,
            completed=sum(r['success'] for r in rows), total=n, ascent_only=args.route == 'up', route=args.route,
            trained_on_ascent_only=trained_ascent_only, teacher_loaded=False, auxiliary_heads_removed=True,
            completion_mode=args.completion_mode,
            visual_intervention=dict(mode=args.intervention, applied_steps=intervention.applied_steps,
                image_and_cls_replaced=args.intervention == 'black_rgb',
                source_rgb_constant=0 if args.intervention == 'black_rgb' else None),
            collision_free_completed=sum(r['success'] and r['collision_free'] for r in rows),
            camera_max_pose_drift=max_camera_drift, video_frames=frame_count,
            note='All requested cases retained including failures. First-step collision window uses root x; it is not exact edge attribution. Heights above 25cm are outside online training support.'))
        write_json(out / 'posture.json', summary.finalize())
        print(json.dumps(dict(output=str(out), completed=sum(r['success'] for r in rows), total=n)), flush=True)
    finally:
        if video is not None: video.close()
        env.close()


if __name__ == '__main__': main()
