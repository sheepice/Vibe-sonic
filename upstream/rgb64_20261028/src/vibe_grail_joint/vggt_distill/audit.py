"""Verify synchronized parameters and actual task/data membership."""
import hashlib

import torch


def shared_initial_digest(algorithm):
    digest = hashlib.sha256()
    for model in (algorithm.get_policy(), algorithm._raw_critic):
        for name, parameter in model.named_parameters():
            if not name.startswith('alignment_heads.'):
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def distributed_parameter_audit(algorithm):
    digest = hashlib.sha256()
    for model in (algorithm.get_policy(), algorithm._raw_critic):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    hashes = [digest.hexdigest()]
    if torch.distributed.is_initialized():
        hashes = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(hashes, digest.hexdigest())
    if len(set(hashes)) != 1:
        raise RuntimeError(f'Trainable parameters diverged across ranks: {hashes}')
    return dict(trainable_sha256_by_rank=hashes, identical_across_ranks=True)


def task_audit(command, task_mode=None):
    if task_mode == 'stairs_original_mixture_v1':
        return original_task_audit(command)
    ppo_kind = command.tile_data['kind'][command.tile[command.ppo_online]]
    if (ppo_kind != 0).any() or command.paired.any():
        raise RuntimeError('PPO encountered a nonascending task')
    if (command.speed[command.ppo_online] <= 0).any():
        raise RuntimeError('Stair PPO sampled a stationary command')
    return dict(ppo_ascending_worlds=int(command.ppo_online.sum()),
                paired_worlds=int(command.paired.sum()), flat_dagger_worlds=int(command.dagger_mask.sum()))


def original_task_audit(c):
    n = c.num_envs
    counts = dict(online_ppo=int(c.ppo_online.sum()), paired_grail_ppo=int(c.paired.sum()),
                  flat_dagger_planner=int(c.dagger_planner.sum()), flat_dagger_replay=int(c.flat.sum()))
    expected = dict(online_ppo=n//2, paired_grail_ppo=n//5,
                    flat_dagger_planner=3*n//20, flat_dagger_replay=3*n//20)
    if counts != expected:
        raise RuntimeError(f'Original mixture changed: {counts} != {expected}')
    if not torch.equal(c.ppo_mask, c.ppo_online | c.paired):
        raise RuntimeError('Paired GRAIL or online worlds are missing from PPO')
    kinds = c.tile_data['kind'][c.tile[c.ppo_online]]
    terrain_counts = torch.bincount(kinds, minlength=4).tolist()
    if not all(terrain_counts):
        raise RuntimeError(f'Missing online terrain kind: {terrain_counts}')
    stream_counts = torch.bincount(c.stream[c.ppo_online], minlength=len(c.cfg.planner_speeds)).tolist()
    if not all(stream_counts):
        raise RuntimeError(f'Missing online speed stream: {stream_counts}')
    speed_count = len(c.cfg.planner_speeds)
    terrain_speeds = torch.bincount(kinds * speed_count + c.stream[c.ppo_online],
                                  minlength=4 * speed_count).reshape(4, speed_count).tolist()
    if not all(count > 0 for row in terrain_speeds[:3] for count in row[1:]):
        raise RuntimeError(f'Missing walking speed on a stair direction: {terrain_speeds}')
    variants = torch.tensor([tile['variant'] for tile in c.cfg.random_tiles], device=c.device)
    stair_variants = variants[c.tile[c.ppo_online]][kinds != 3]
    variant_counts = torch.bincount(stair_variants, minlength=3).tolist()
    if not all(variant_counts):
        raise RuntimeError(f'Missing regular or irregular online stairs: {variant_counts}')
    paired_clips = c._clip_ids[c.paired]
    if not torch.isin(paired_clips, c.source_groups['grail']).all():
        raise RuntimeError('Paired PPO selected a non-GRAIL clip')
    if not (c.motion.terrain_ids[paired_clips] >= 0).all():
        raise RuntimeError('Paired GRAIL motion lacks its terrain')
    if not (c.tile_data['kind'][c.tile[c.dagger_planner]] == 3).all():
        raise RuntimeError('Planner DAgger left flat terrain')
    weights = {s: float(c.motion.metadata['config']['flat_sources'][s]['weight']) for s in c.flat_sources}
    if weights != dict(lafan=.3, style100=.2, sonic=.4, real_pico=.1):
        raise RuntimeError(f'Original flat source weights changed: {weights}')
    source_counts = {s: int(torch.isin(c._clip_ids[c.flat], c.source_groups[s]).sum()) for s in c.flat_sources}
    return dict(**counts, ppo_total=int(c.ppo_mask.sum()),
                online_terrain_counts=dict(zip(('up','down','up_down','flat'), terrain_counts)),
                online_speed_stream_counts=dict(zip(map(str,c.cfg.planner_speeds), stream_counts)),
                online_terrain_speed_counts={name: dict(zip(map(str,c.cfg.planner_speeds), row))
                    for name, row in zip(('up','down','up_down','flat'), terrain_speeds)},
                online_stair_variant_counts=dict(zip(('regular_0','irregular_1','irregular_2'), variant_counts)),
                flat_source_weights=weights, actual_flat_replay_source_counts=source_counts,
                paired_grail_clip_source_verified=True, paired_terrain_verified=True,
                paired_terrain_rows=len(c.motion.metadata['terrain_rows']))
