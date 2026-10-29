"""Held-out evaluation for the stair-only task, preserving the original success checks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .catalog import write_json
from .gpu_guard import verify_guarded_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--visual-ablation", choices=("none", "zero_tokens"), default="none")
    parser.add_argument("--video-dir", help="Record successful episodes from the policy RGB camera")
    args = parser.parse_args()
    verify_guarded_worker()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Evaluation uses one GPU; launch with --gpus <one index>")
    import mjlab  # noqa: F401
    import torch
    from mjlab.rl import RslRlVecEnvWrapper

    from .stairs import build_stairs_env_cfg, validate_stairs_bank
    from .evaluation_env import EvaluationEnv
    from .policy import make_actor

    bank = validate_stairs_bank(Path(args.bank))
    if bank["split"] != "validation" or bank["smoke_only"]:
        raise ValueError("Use the full held-out validation bank for reported stair evaluation")
    cfg = build_stairs_env_cfg(args.bank, num_envs=args.num_envs, play=True)
    cfg.seed = args.seed
    env = RslRlVecEnvWrapper(EvaluationEnv(cfg, device="cuda:0"))
    results = []
    try:
        actor = make_actor(bank["config"], env.get_observations()).to("cuda:0").eval()
        actor.load_state_dict(
            torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)["actor_state_dict"]
        )
        command = env.unwrapped.command_manager.get_term("motion")
        selected = [i for i, clip in enumerate(bank["clips"]) if clip["source"] == "grail"]
        if args.limit:
            selected = selected[: args.limit]
        for start in range(0, len(selected), args.num_envs):
            batch = selected[start : start + args.num_envs]
            padded = batch + [batch[-1]] * (args.num_envs - len(batch))
            command.forced_clip_ids = torch.tensor(padded, device=env.device)
            env.unwrapped.record_terminals = False
            obs, _ = env.reset()
            env.unwrapped.record_terminals = True
            finished = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
            sums = torch.zeros(args.num_envs, device=env.device)
            counts = torch.zeros(args.num_envs, device=env.device)
            films = []
            max_steps = (
                max(bank["clips"][i]["length"] for i in batch)
                + round(bank["config"]["motion_end_hold_seconds"] * 50)
                + 10
            )
            for step in range(max_steps):
                if args.video_dir and step % 2 == 0:
                    films.append(env.unwrapped.scene["head_cam"].data.rgb[: len(batch)].cpu().numpy().copy())
                if args.visual_ablation == "zero_tokens":
                    obs = obs.clone()
                    obs["kv_tokens", "img_tokens"].zero_()
                    obs["q_cls"].zero_()
                with torch.inference_mode():
                    env.unwrapped.terminal_records = {}
                    obs, _, dones, _ = env.step(actor(obs))
                robot = env.unwrapped.scene["robot"]
                root = robot.data.root_link_pos_w
                current_ref = (
                    command.motion.body_pos_w[command.time_steps, 0] + env.unwrapped.scene.env_origins
                )
                error = (root - current_ref).norm(dim=-1)
                for index, terminal in env.unwrapped.terminal_records.items():
                    error[index] = terminal["terminal_tracking_error_m"]
                sums += error * ~finished
                counts += ~finished
                new = (dones.bool() & ~finished).nonzero().flatten().tolist()
                for index in new:
                    if index >= len(batch):
                        continue
                    row = {
                        "clip_id": bank["clips"][batch[index]]["id"],
                        "steps": step + 1,
                        "mean_root_error_m": float(sums[index] / counts[index]),
                        **env.unwrapped.terminal_records[index],
                    }
                    if args.video_dir and row["traversal_completed"] and films:
                        import imageio.v2 as iio

                        directory = Path(args.video_dir)
                        directory.mkdir(parents=True, exist_ok=True)
                        movie = directory / f"{row['clip_id']}.mp4"
                        iio.mimwrite(movie, [frame[index] for frame in films], fps=25, macro_block_size=1)
                        row["rgb_video"] = str(movie.resolve())
                    results.append(row)
                finished |= dones.bool()
                if bool(finished[: len(batch)].all()):
                    break
            for index in range(len(batch)):
                if not bool(finished[index]):
                    results.append(
                        {
                            "clip_id": bank["clips"][batch[index]]["id"],
                            "traversal_completed": False,
                            "ascent_success": False,
                            "error": "evaluation_horizon_exhausted",
                        }
                    )
            report = {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "bank_identity": bank["identity"],
                "split": bank["split"],
                "smoke_only": bank["smoke_only"],
                "seed": args.seed,
                "visual_ablation": args.visual_ablation,
                "episodes": results,
                "completed": sum(r["traversal_completed"] for r in results),
                "ascent_successes": sum(r["ascent_success"] for r in results),
                "ascent_attempts": sum(r.get("reference_is_ascent", False) for r in results),
                "deployment_validated": False,
            }
            write_json(Path(args.output), report)
            print(
                f"Evaluated {len(results)}/{len(selected)}; completed={report['completed']}, "
                f"ascents={report['ascent_successes']}/{report['ascent_attempts']}",
                flush=True,
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
