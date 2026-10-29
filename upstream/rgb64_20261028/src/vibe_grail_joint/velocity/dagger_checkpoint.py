"""Rank-local replay sidecars; model checkpoints only follow completed writes."""

import os
from pathlib import Path
import torch


def save_replay(env, out, iteration, rank, world, num_envs, teacher_sha256):
    path = Path(out) / f"dagger_{iteration}_rank_{rank}.pt"
    tmp = path.with_suffix(".pt.tmp")
    state = dict(iteration=iteration, rank=rank, world_size=world,
                 num_envs_per_rank=num_envs, teacher_sha256=teacher_sha256,
                 replay=env.buffer.state_dict(), torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(env.device))
    torch.save(state, tmp)
    os.replace(tmp, path)
    if world > 1:
        torch.distributed.barrier()


def restore_replay(env, checkpoint, iteration, rank, world, num_envs, teacher_sha256, infos,
                   *, allow_env_count_change=False):
    meta = infos.get("dagger_replay", {})
    saved_num_envs = meta.get("num_envs_per_rank")
    if meta.get("world_size") != world:
        raise ValueError("DAgger resume requires the same rank count")
    if not isinstance(saved_num_envs, int) or saved_num_envs < 20 or saved_num_envs % 20:
        raise ValueError("Invalid saved environment allocation")
    if num_envs < 20 or num_envs % 20:
        raise ValueError("New environment allocation must preserve the 50/20/15/15 fractions")
    if saved_num_envs != num_envs and not allow_env_count_change:
        raise ValueError("Changing environment count on resume requires explicit opt-in")
    path = Path(checkpoint).parent / meta["pattern"].format(rank=rank)
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in dict(iteration=iteration, rank=rank, world_size=world,
                              num_envs_per_rank=saved_num_envs, teacher_sha256=teacher_sha256).items():
        if state[key] != expected:
            raise ValueError(f"DAgger sidecar mismatch: {key}")
    env.buffer.load_state_dict(state["replay"])
    # Reservoirs contain independent labelled observations, not live environment
    # IDs. PPO starts a fresh rollout and physics starts new episodes on resume,
    # so an explicitly requested world-count change can retain all labels.
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"], env.device)
    return dict(saved_num_envs_per_rank=saved_num_envs, num_envs_per_rank=num_envs,
                environment_count_changed=saved_num_envs != num_envs,
                history_samples_restored=len(env.buffer))
