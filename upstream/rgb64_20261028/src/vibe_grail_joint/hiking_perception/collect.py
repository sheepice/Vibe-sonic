"""Collect synchronized RGB features, terrain labels and action transitions."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True); p.add_argument('--num-envs', type=int, default=132)
    p.add_argument('--steps', type=int, default=1500); p.add_argument('--seed', type=int, default=180017)
    p.add_argument('--chunk-steps', type=int, default=128); p.add_argument('--preview-every', type=int, default=25)
    p.add_argument('--domain-randomization', action='store_true')
    p.add_argument('--uniform', action='store_true'); p.add_argument('--approach-extra',type=float,default=0.)
    p.add_argument('--grid-nx',type=int,default=16); p.add_argument('--grid-ny',type=int,default=12)
    p.add_argument('--history-frames',type=int,default=0,help='Retain past camera/depth supervision views at 10 Hz; labels only')
    args = p.parse_args()
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import torch
    from PIL import Image, ImageDraw
    from .simulation import evaluation_env, GeometryLabels
    from .evaluate import load_control, augment
    from .contracts import contract, GeometrySpec
    from .dataset import ShardWriter
    from .control import file_hash
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    settings = json.loads(Path(args.config).read_text())
    env, settings, cases = evaluation_env(settings, args.num_envs, args.seed, duration=args.steps*.02,
        uniform=args.uniform,approach_extra=args.approach_extra,domain_randomization=args.domain_randomization,collection_cases=True)
    try:
        obs, _ = env.reset()
        actor, memory = load_control(settings, obs, args.checkpoint, env.unwrapped)
        geometry = GeometrySpec(nx=args.grid_nx, ny=args.grid_ny)
        labels = GeometryLabels(env.unwrapped, geometry, history_frames=args.history_frames)
        writer = ShardWriter(args.output, dict(contract=contract(geometry), checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=file_hash(args.checkpoint),
            settings=settings, cases=cases, seed=args.seed, domain_randomization=args.domain_randomization,
            history_supervision_frames=args.history_frames,history_stride_control_steps=5,
            scene_key_version=2,tiles=list(env.unwrapped.command_manager.get_term('motion').cfg.random_tiles),
            scene_keys={str(int(k[:15], 16)): k for k in labels.keys},
            split_rule='sha256(scene geometry) prefix modulo 10: 0 test, 1 val, else train',
            depth_role='supervision visibility only; never an actor input',
            synchronization='obs[t], labels[t], action[t], contact during t→t+1, state[t+1]'), args.chunk_steps)
        ids = torch.arange(args.num_envs, device=env.device)
        alive = torch.ones(args.num_envs, dtype=torch.bool, device=env.device)
        reset = alive.clone(); stats = {'visible_cells': 0, 'valid_samples': 0, 'collision_transitions': 0}
        for step in range(args.steps):
            with torch.no_grad():
                before = labels.sample(ids)
                actor_obs = augment(obs, memory, env.unwrapped, reset)
                action = actor(actor_obs, stochastic_output=False)
                if not torch.isfinite(action).all():
                    raise RuntimeError('Nonfinite collection policy')
                # Capture BEFORE physics updates mutable observation/render buffers.
                row = {'tokens': obs['kv_tokens']['img_tokens'].clone().half(), 'proprio': obs['q_proprio'].clone(),
                    'age': torch.full((args.num_envs,), env.unwrapped.step_dt, device=env.device),
                    'height': before['height'], 'edge': before['edge'], 'visible': before['visible'],
                    'historical_visible':before['historical_visible'],'unknown':before['unknown'],
                    'historical_seen_at_s':before['historical_seen_at_s'],
                    'scene_id': before['scene_id'], 'action': action.clone(),
                    'command': obs['velocity_command'].clone(), 'reference_twist': obs['augmentation'].clone(),
                    'reset': reset.clone(), 'valid': alive.clone(),
                    'time_s': torch.full((args.num_envs,), step*env.unwrapped.step_dt, device=env.device)}
                if step % args.preview_every == 0:
                    sensor = env.unwrapped.scene['head_cam']
                    image = Image.fromarray(sensor.data.rgb[0].cpu().numpy()).resize((672, 378))
                    draw = ImageDraw.Draw(image)
                    for uv, vis in zip(before['uv'][0].cpu().tolist(), before['visible'][0].cpu().tolist()):
                        if vis:
                            x, y = uv[0]*6, uv[1]*6
                            draw.ellipse((x-2, y-2, x+2, y+2), fill=(0,255,0))
                    image.save(Path(args.output)/f'projection_{step:06d}.png')
                obs, _, done, _ = env.step(action)
                labels.reset(done.bool().nonzero().flatten())
                after = labels.sample(ids)
                robot = env.unwrapped.scene['robot']
                fall = robot.data.projected_gravity_b[:, 2] > -.4
                row.update(next_state=after['state'], contact=labels.contacts(ids), done=done.bool() | fall)
                row['valid'] &= ~done.bool()  # auto-reset successor is not a physical next state
                stats['visible_cells'] += int((row['visible'] & row['valid'][:, None]).sum())
                stats['valid_samples'] += int(row['valid'].sum())
                stats['collision_transitions'] += int((row['contact'].any(-1) & row['valid']).sum())
                writer.add(row)
                alive &= ~row['done']; reset = done.bool()
            if step % 100 == 0:
                print(json.dumps(dict(step=step, alive=int(alive.sum()), **stats)), flush=True)
        writer.close(**stats)
        if not stats['visible_cells']:
            raise RuntimeError('No visible geometry labels; inspect depth projection before fitting')
        print(json.dumps(dict(status='complete', output=args.output, **stats)), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
