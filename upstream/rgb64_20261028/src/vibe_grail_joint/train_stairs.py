"""Stair-only ViBe training; the original mixed-data entrypoint is unchanged."""

from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import asdict
from pathlib import Path

from .catalog import write_json
from .gpu_guard import verify_guarded_worker


def frozen_digest(actor):
    digest = hashlib.sha256()
    for name, value in actor.named_parameters():
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true", help="At most 10 PPO updates; accepts a small bank")
    args = parser.parse_args()
    # Run BEFORE torch/mjlab can create a CUDA context.
    verify_guarded_worker()
    import mjlab  # noqa: F401
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner

    from .env_cfg import build_agent_cfg
    from .stairs import StairsLoggingWrapper, build_stairs_env_cfg, validate_stairs_bank

    rank, local_rank, world = (
        int(os.environ.get(key, default))
        for key, default in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"))
    )
    path = Path(args.bank_root) / f"rank_{rank}"
    bank = validate_stairs_bank(path)
    if bank["world_size"] != world or bank["rank"] != rank or bank["split"] != "train":
        raise ValueError("Bank sharding/split does not match this training launch")
    if bank["smoke_only"] and not args.smoke:
        raise ValueError("A smoke-test subset cannot be used for full training")
    settings = bank["config"]
    iterations = args.iterations or (2 if args.smoke else settings["max_iterations"])
    if iterations < 1 or (args.smoke and iterations > 10):
        raise ValueError("Invalid iteration count (smoke permits 1..10)")
    num_envs = args.num_envs or (8 if args.smoke else settings["num_envs_per_rank"])
    if num_envs < 4 or num_envs % (settings["flat_to_terrain_ratio"] + 1):
        raise ValueError("num_envs must preserve the configured flat/stair ratio")
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    cfg = build_stairs_env_cfg(str(path), num_envs=num_envs)
    cfg.seed = settings["seed"] + rank
    agent = build_agent_cfg(settings)
    agent.seed = cfg.seed
    if args.smoke:
        agent.save_interval = 1
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "initializing",
        "training_mode": "stairs_only",
        "initialization": args.resume or "released_sonic_zero_initialized_adapters",
        "rank": rank,
        "world_size": world,
        "smoke_only": args.smoke,
        "bank_identity": bank["identity"],
        "config": settings,
        "num_envs": num_envs,
        "iterations_requested": iterations,
        "source_counts": bank["source_motion_counts"],
    }
    record_path = log_dir / f"rank_{rank}_status.json"
    write_json(record_path, record)
    env = None
    try:
        env = StairsLoggingWrapper(ManagerBasedRlEnv(cfg, device=device))
        runner = MjlabOnPolicyRunner(env, asdict(agent), str(log_dir), device)
        if args.resume:
            runner.load(args.resume, map_location=device)
        actor = runner.alg.get_policy()
        before = frozen_digest(actor)
        record.update(
            status="running",
            frozen_sha256=before,
            trainable_parameters={
                name: value.numel() for name, value in actor.named_parameters() if value.requires_grad
            },
        )
        write_json(record_path, record)
        if rank == 0:
            write_json(log_dir / "agent_cfg.json", asdict(agent))
            write_json(
                log_dir / "observation_contract.json",
                {
                    "robot": "Unitree G1 29 DOF",
                    "camera": settings["camera"],
                    "control_hz": 50,
                    "joint_names": bank["joint_names"],
                    "body_names": bank["body_names"],
                    "actor_groups": [
                        "policy",
                        "tokenizer",
                        "augmentation",
                        "kv_tokens",
                        "q_task_cmd",
                        "q_proprio",
                        "q_cls",
                    ],
                    "depth_input": False,
                    "requires_reference_motion": True,
                    "reference_frames": 10,
                    "reference_stride_seconds": 0.1,
                    "deployment_validated": False,
                },
            )
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
        after = frozen_digest(actor)
        if before != after:
            raise RuntimeError("Frozen SONIC parameters changed during post-training")
        record.update(
            status="completed",
            last_iteration=runner.current_learning_iteration,
            frozen_unchanged=True,
            frozen_sha256_after=after,
        )
        write_json(record_path, record)
    except BaseException as error:
        record.update(status="failed", error=repr(error))
        write_json(record_path, record)
        raise
    finally:
        if env is not None:
            env.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
