"""Checkpoint provenance and fail-fast cleanup for long distributed velocity runs."""


def configure_distributed(runner, timeout_seconds=180):
    """Same RSL-RL manual gradient reduction, with a bounded communication timeout."""
    import os
    from datetime import timedelta
    import torch

    runner.gpu_world_size = int(os.getenv('WORLD_SIZE', '1'))
    runner.is_distributed = runner.gpu_world_size > 1
    if not runner.is_distributed:
        runner.gpu_local_rank = runner.gpu_global_rank = 0
        runner.cfg['multi_gpu'] = None
        return
    runner.gpu_local_rank = int(os.environ['LOCAL_RANK'])
    runner.gpu_global_rank = int(os.environ['RANK'])
    if runner.device != f'cuda:{runner.gpu_local_rank}':
        raise ValueError('Distributed device must match LOCAL_RANK')
    runner.cfg['multi_gpu'] = dict(global_rank=runner.gpu_global_rank,
                                   local_rank=runner.gpu_local_rank,
                                   world_size=runner.gpu_world_size)
    torch.cuda.set_device(runner.gpu_local_rank)
    torch.distributed.init_process_group(backend='nccl', rank=runner.gpu_global_rank,
                                         world_size=runner.gpu_world_size,
                                         timeout=timedelta(seconds=timeout_seconds))


def checkpoint_metadata(runner, settings, infos):
    from ..train_stairs import frozen_digest

    out = dict(infos or {})
    out['velocity_training'] = dict(
        version=2,
        settings=settings,
        completed_policy_updates=int(runner.current_learning_iteration) + 1,
        curriculum_update=int(getattr(runner.env.unwrapped, 'policy_update_count', 0)),
        frozen_sonic_sha256=frozen_digest(runner.alg.get_policy()),
        replay_fraction=.4,
        online_fraction=.6,
        resume_semantics='optimizer and model restored; curriculum absolute update restored; physics and planner start new episodes',
    )
    if settings.get('dagger') is not None:
        out['velocity_training'].update(
            replay_fraction=.2, online_fraction=.5, dagger_fraction=.3,
            resume_semantics='model, optimizer, curriculum and rank-local DAgger reservoirs restored; physics and planner start new episodes',
        )
    return out
