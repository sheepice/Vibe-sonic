"""Record matched comparisons with a stationary world camera.

This evaluation-only entrypoint does not modify any training code or checkpoint.
Terminal worlds are frozen in the EDITED VIDEO, not shown after automatic reset.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .gpu_guard import verify_guarded_worker


def panel(observer, rgb, title, clip_id, frame, terminal, *, final=False):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (640, 544), (18, 23, 32))
    image.paste(Image.fromarray(observer), (0, 40))
    image.paste(Image.fromarray(rgb).resize((224, 126), Image.Resampling.NEAREST), (8, 410))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=19)
    small = ImageFont.load_default(size=15)
    draw.text((12, 9), title, font=font, fill=(235, 240, 255))
    done = final or frame * 2 >= terminal.get("steps", 100000)
    success = terminal.get("traversal_completed", False) and (
        not terminal.get("reference_is_ascent", False) or terminal.get("ascent_success", False))
    state = ("SUCCESS" if success else "FAILED - frame held") if done else "RUNNING"
    if done and terminal.get("traversal_completed", False) and not success:
        state = "COMPLETED - height short"
    color = ((92, 228, 140) if success else (255, 120, 100)) if done else (235, 240, 255)
    draw.text((246, 411), state, font=font, fill=color)
    elapsed = min(frame * .04, terminal.get("steps", 100000) * .02)
    draw.text((246, 441), f"Simulation time: {elapsed:.2f} s", font=small, fill="white")
    draw.text((246, 465), "RGB: actual head camera (112 x 63)", font=small, fill="white")
    draw.text((246, 490), f"Clip: {clip_id[:20]}", font=small, fill=(170, 185, 200))
    if done:
        reason = ", ".join(terminal.get("termination_reasons", ["horizon"]))
        draw.text((246, 515), reason[:44], font=small, fill=color)
    else:
        draw.text((246, 515), "Observer: fixed world camera", font=small, fill=(170, 185, 200))
    return np.asarray(image)


def encode(path, observer, head, worlds, terminals, clip_id, *, paired):
    import imageio.v2 as iio
    import numpy as np

    frames = max((terminals[w]["steps"] + 1) // 2 for w in worlds)
    last_frames = {w: min(len(observer) - 1, (terminals[w]["steps"] - 1) // 2) for w in worlds}
    with iio.get_writer(path, fps=25, codec="libx264", quality=8,
                        macro_block_size=1, ffmpeg_params=["-threads", "1", "-pix_fmt", "yuv420p"]) as writer:
        # The final second is explicitly a result freeze; simulation is never slowed.
        for frame in range(frames + 25):
            panes = []
            for local, w in enumerate(worlds):
                source = min(frame, last_frames[w])
                title = "SONIC | original frozen base" if paired and local == 0 else "ViBe | RGB + LoRA | model_14999"
                panes.append(panel(observer[source][w], head[source][w], title, clip_id,
                                   frame, terminals[w], final=frame >= frames))
            writer.append_data(np.concatenate(panes, axis=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-report")
    parser.add_argument("--clip-ids")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--exact-start", action="store_true")
    parser.add_argument("--save-failures", action="store_true")
    args = parser.parse_args()
    verify_guarded_worker()
    import mjlab  # noqa: F401
    import torch
    from mjlab.rl import RslRlVecEnvWrapper
    from .catalog import write_json
    from .evaluation_env import EvaluationEnv
    from .policy import make_actor
    from .review_helpers import disable_adapters
    from .fixed_camera import attach_fixed_observer
    from .stairs import build_stairs_env_cfg, validate_stairs_bank

    torch.set_num_threads(2)
    bank = validate_stairs_bank(Path(args.bank))
    if bank["split"] != "validation" or bank["smoke_only"]:
        raise ValueError("Record from the full held-out validation bank")
    selected = list(range(len(bank["clips"])))
    wanted = None
    if args.case_report:
        source = json.loads(Path(args.case_report).read_text())
        wanted = {r["clip_id"] for r in source["episodes"] if r["traversal_completed"]}
    if args.clip_ids:
        wanted = set(args.clip_ids.split(","))
    if wanted is not None:
        known = {c["id"] for c in bank["clips"]}
        if wanted - known:
            raise ValueError(f"Unknown clips: {wanted - known}")
        selected = [i for i in selected if bank["clips"][i]["id"] in wanted]
    selected = [index for index in selected for _ in range(args.repeats)]
    selected = selected[args.shard_index::args.shard_count]
    if args.paired and args.num_envs % 2:
        raise ValueError("Paired recordings need an even world count")
    cfg = build_stairs_env_cfg(args.bank, num_envs=args.num_envs, play=True)
    cfg.seed = args.seed
    if args.paired or args.exact_start:
        # Domain stripping leaves RSI velocity/joint noise. Explicitly remove it
        # so both policies receive exactly the same reference initial state.
        motion = cfg.commands["motion"]
        motion.pose_range, motion.velocity_range = {}, {}
        motion.joint_position_range = (0., 0.)
    if len(set(selected)) != 1 or not args.paired:
        raise ValueError("Fixed camera recording expects one clip and --paired")
    camera_spec = attach_fixed_observer(cfg, bank, args.bank, selected[0])
    env = RslRlVecEnvWrapper(EvaluationEnv(cfg, device="cuda:0"))
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        actor = make_actor(bank["config"], env.get_observations()).to(env.device).eval()
        actor.load_state_dict(torch.load(args.checkpoint, map_location=env.device,
                                        weights_only=False)["actor_state_dict"])
        base = copy.deepcopy(actor) if args.paired else None
        if base is not None:
            disable_adapters(base)
        command = env.unwrapped.command_manager.get_term("motion")
        batch_size = args.num_envs // (2 if args.paired else 1)
        for start in range(0, len(selected), batch_size):
            batch = selected[start:start + batch_size]
            padded = batch + [batch[-1]] * (batch_size - len(batch))
            world_clips = [c for c in padded for _ in range(2)] if args.paired else padded
            command.forced_clip_ids = torch.tensor(world_clips, device=env.device)
            env.unwrapped.record_terminals = False
            obs, _ = env.reset()
            env.unwrapped.record_terminals = True
            initial_error = 0.
            if args.paired:
                robot = env.unwrapped.scene["robot"]
                for name in ("root_link_pos_w", "root_link_quat_w", "root_link_lin_vel_w",
                             "root_link_ang_vel_w", "joint_pos", "joint_vel"):
                    values = getattr(robot.data, name)
                    initial_error = max(initial_error, float((values[0::2] - values[1::2]).abs().max()))
                if initial_error > 1e-7:
                    raise RuntimeError(f"Unmatched comparison initial state: {initial_error}")
            camera = env.unwrapped.scene["review_camera"]
            camera_index = camera.camera_idx
            assert int(env.unwrapped.sim.mj_model.cam_bodyid[camera_index]) == 0
            fixed_pos = env.unwrapped.sim.data.cam_xpos[:, camera_index].clone()
            fixed_mat = env.unwrapped.sim.data.cam_xmat[:, camera_index].clone()
            camera_max_drift = 0.0
            terminals, observer, head = {}, [], []
            world_count = len(batch) * (2 if args.paired else 1)
            horizon = max(bank["clips"][i]["length"] for i in batch) + 60
            with torch.no_grad():
                for step in range(horizon):
                    if step % 2 == 0:
                        camera_max_drift = max(camera_max_drift,
                            float((env.unwrapped.sim.data.cam_xpos[:, camera_index] - fixed_pos).abs().max()),
                            float((env.unwrapped.sim.data.cam_xmat[:, camera_index] - fixed_mat).abs().max()))
                        if camera_max_drift > 1e-7:
                            raise RuntimeError(f"Observer camera moved: {camera_max_drift}")
                        observer.append(env.unwrapped.scene["review_camera"].data.rgb[:world_count].cpu().numpy().copy())
                        head.append(env.unwrapped.scene["head_cam"].data.rgb[:world_count].cpu().numpy().copy())
                    actions = actor(obs)
                    if args.paired:
                        actions[0::2] = base(obs[0::2])
                    env.unwrapped.terminal_records = {}
                    obs, _, dones, _ = env.step(actions)
                    for world in range(world_count):
                        if bool(dones[world]) and world not in terminals:
                            terminals[world] = {"steps": step + 1, **env.unwrapped.terminal_records[world]}
                    if len(terminals) == world_count:
                        break
            for world in range(world_count):
                terminals.setdefault(world, {"steps": horizon, "traversal_completed": False,
                                             "termination_reasons": ["horizon_exhausted"]})
            for local, index in enumerate(batch):
                clip = bank["clips"][index]
                w = 2 * local + 1 if args.paired else local
                row = {"clip_id": clip["id"], "vibe": terminals[w],
                       "source_stem": bank["terrain_rows"][clip["terrain_id"]]["source_stem"]}
                if args.paired:
                    row.update(sonic=terminals[w - 1], initial_state_max_abs_difference=initial_error)
                if terminals[w]["traversal_completed"] or args.save_failures:
                    suffix = f"_trial{start + local:03d}" if args.repeats > 1 else ""
                    target = destination / (clip["id"] + suffix + "_vibe_rgb.mp4")
                    encode(target, observer, head, [w], terminals, clip["id"], paired=False)
                    row["video"] = str(target.resolve())
                    if args.paired:
                        target = destination / (clip["id"] + suffix + "_sonic_vs_vibe.mp4")
                        encode(target, observer, head, [w - 1, w], terminals, clip["id"], paired=True)
                        row["comparison_video"] = str(target.resolve())
                results.append(row)
            write_json(destination / "recordings.json", {
                "checkpoint": str(Path(args.checkpoint).resolve()), "bank_identity": bank["identity"],
                "seed": args.seed, "paired": args.paired, "shard_index": args.shard_index,
                "shard_count": args.shard_count, "initial_state": "exact reference" if args.paired or args.exact_start else "evaluation RSI noise",
                "observer_camera": camera_spec, "observer_max_pose_drift": camera_max_drift,
                "observer_camera_parent_is_world": True,
                "rgb_is_actual_policy_camera": True, "observer_is_not_policy_input": True,
                "frozen_frame_after_terminal": True, "playback_speed": 1., "fps": 25,
                "success_criterion": "full reference + 1s hold; no termination failure; final root error < .3m; upright",
                "episodes": results, "successes": sum(r["vibe"]["traversal_completed"] for r in results),
            })
            print(f"Recorded {len(results)}/{len(selected)}, success={sum(r['vibe']['traversal_completed'] for r in results)}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
