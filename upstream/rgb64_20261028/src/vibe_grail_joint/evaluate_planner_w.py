"""Record a matched reference-vs-W comparison without changing training code."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from .gpu_guard import verify_guarded_worker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--planner", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip-id", default="f0dda732bca2f56b09412be7")
    parser.add_argument("--duration", type=float, default=15.)
    parser.add_argument("--speed", type=float, default=.5)
    parser.add_argument("--heading-deg", type=float, default=-90.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flat", action="store_true")
    args = parser.parse_args()
    verify_guarded_worker()
    import mjlab  # noqa: F401
    import imageio.v2 as iio
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from .catalog import write_json
    from .fixed_camera import attach_fixed_observer
    from .online_planner_probe import PlannerProbeCommandCfg
    from .policy import make_actor
    from .stairs import build_stairs_env_cfg, validate_stairs_bank

    torch.set_num_threads(2)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    bank = validate_stairs_bank(Path(args.bank))
    index = next(i for i, c in enumerate(bank["clips"]) if c["id"] == args.clip_id)
    clip = bank["clips"][index]
    terrain = bank["terrain_rows"][clip["terrain_id"]]
    cfg = build_stairs_env_cfg(args.bank, num_envs=2, play=True)
    cfg.seed = args.seed
    old = cfg.commands["motion"]
    common = {f.name: getattr(old, f.name) for f in fields(PlannerProbeCommandCfg) if hasattr(old, f.name)}
    common.update(bank_path=args.bank, planner_path=args.planner, clip_id=args.clip_id,
                  duration_s=args.duration, speed=args.speed, heading_deg=args.heading_deg,
                  seed=args.seed, pose_range={}, velocity_range={}, joint_position_range=(0., 0.))
    cfg.commands["motion"] = PlannerProbeCommandCfg(**common)
    camera_spec = attach_fixed_observer(cfg, bank, args.bank, index)
    if args.flat:
        cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"].rows = ()
    # A flat-height planner should not be declared fallen merely because a
    # successful climb departs vertically from its reference. Evaluate actual
    # posture, root-to-ground clearance, progress and foot height below.
    # Terminal capture is manual below, avoiding any auto-reset or replay.
    cfg.terminations.clear()
    cfg.episode_length_s = args.duration + 5
    cfg.rewards = {}
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg, device="cuda:0"))
    try:
        actor = make_actor(bank["config"], env.get_observations()).to(env.device).eval()
        actor.load_state_dict(torch.load(args.checkpoint, map_location=env.device,
                                        weights_only=False)["actor_state_dict"])
        obs, _ = env.reset()
        robot = env.unwrapped.scene["robot"]
        command = env.unwrapped.command_manager.get_term("motion")
        initial_error = max(float((getattr(robot.data, n)[0] - getattr(robot.data, n)[1]).abs().max())
                            for n in ("root_link_pos_w", "root_link_quat_w", "root_link_lin_vel_w",
                                      "root_link_ang_vel_w", "joint_pos", "joint_vel"))
        assert initial_error < 1e-7, initial_error
        initial_root = robot.data.root_link_pos_w.cpu().numpy().copy()
        origins = env.unwrapped.scene.env_origins.cpu().numpy().copy()
        foot_ids = robot.find_bodies(("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True)[0]
        direction = command.planner.direction[0]
        source_root = command.source["body_pos_w"][:, 0]
        goal_progress = float((source_root[-1] - source_root[0]) @ direction)
        goal_height = float(source_root[-1, 2] - source_root[0, 2])
        with np.load(Path(bank["config"]["terrain_root"]) / terrain["terrain"]) as data:
            height_field = data["heights"].copy()
            xy_min = data["xy_min"].copy()
            resolution = float(data["resolution"])

        def ground_at(xy):
            ij = np.rint((xy - xy_min) / resolution).astype(int)
            valid = (ij[..., 0] >= 0) & (ij[..., 0] < height_field.shape[1]) & (ij[..., 1] >= 0) & (ij[..., 1] < height_field.shape[0])
            value = np.zeros(xy.shape[:-1], np.float32)
            value[valid] = height_field[ij[..., 1][valid], ij[..., 0][valid]]
            return value * (0 if args.flat else 1)

        camera = env.unwrapped.scene["review_camera"]
        camera_index = camera.camera_idx
        assert int(env.unwrapped.sim.mj_model.cam_bodyid[camera_index]) == 0
        camera_pos = env.unwrapped.sim.data.cam_xpos[:, camera_index].clone()
        camera_mat = env.unwrapped.sim.data.cam_xmat[:, camera_index].clone()
        observer, head, records, terminals = [], [], [], {}
        max_drift = 0.
        held_success = [0, 0]
        max_steps = round(args.duration * 50)
        with torch.inference_mode():
            for step in range(max_steps + 1):
                root = robot.data.root_link_pos_w.cpu().numpy().copy()
                feet = robot.data.body_link_pos_w[:, foot_ids].cpu().numpy().copy()
                gravity = robot.data.projected_gravity_b[:, 2].cpu().numpy().copy()
                ground = ground_at((feet - origins[:, None])[:, :, :2])
                root_ground = ground_at((root - origins)[:, :2])
                progress = (root - initial_root) @ direction
                gain = root[:, 2] - initial_root[:, 2]
                reference_height = command.anchor_pos_w[:, 2].cpu().numpy()
                row = {"step": step, "time_s": step / 50,
                       "root_local": (root - origins).tolist(), "root_height_gain_m": gain.tolist(),
                       "forward_progress_m": progress.tolist(), "gravity_z": gravity.tolist(),
                       "foot_local": (feet - origins[:, None]).tolist(), "foot_ground_height_m": ground.tolist(),
                       "reference_root_height_m": (reference_height - origins[:, 2]).tolist()}
                records.append(row)
                for world in range(2):
                    if world in terminals:
                        continue
                    upright = gravity[world] < -.7
                    top = (not args.flat and progress[world] >= goal_progress - .2 and
                           gain[world] >= goal_height - .15 and upright and
                           np.min(feet[world, :, 2] - origins[world, 2]) >= terrain["max_height"] - .15)
                    held_success[world] = held_success[world] + 1 if top else 0
                    reason = None
                    if held_success[world] >= 25:
                        reason = "top_reached_upright_for_0.5s"
                    elif gravity[world] > -.3 or root[world, 2] - origins[world, 2] - root_ground[world] < .3:
                        reason = "physical_fall"
                    elif step == max_steps:
                        reason = "flat_walk_completed" if args.flat and upright and progress[world] > 1. else "time_limit"
                    if reason:
                        terminals[world] = {"step": step, "time_s": step / 50, "reason": reason,
                                            "success": held_success[world] >= 25 or reason == "flat_walk_completed",
                                            "height_gain_m": float(gain[world]),
                                            "forward_progress_m": float(progress[world])}
                        print("TERMINAL", world, json.dumps(terminals[world]), flush=True)
                if step % 2 == 0:
                    observer.append(camera.data.rgb.cpu().numpy().copy())
                    head.append(env.unwrapped.scene["head_cam"].data.rgb.cpu().numpy().copy())
                    max_drift = max(max_drift, float((env.unwrapped.sim.data.cam_xpos[:, camera_index] - camera_pos).abs().max()),
                                    float((env.unwrapped.sim.data.cam_xmat[:, camera_index] - camera_mat).abs().max()))
                if step % 50 == 0:
                    print(f"STEP {step} progress={progress.tolist()} height={gain.tolist()}", flush=True)
                if len(terminals) == 2:
                    break
                obs, _, _, _ = env.step(actor(obs))
        assert max_drift < 1e-7
        font = ImageFont.load_default(size=18)
        small = ImageFont.load_default(size=15)

        def panel(frame, world):
            stop = min(len(observer) - 1, terminals[world]["step"] // 2)
            source = min(frame, stop)
            image = Image.new("RGB", (640, 540), (18, 23, 32))
            image.paste(Image.fromarray(observer[source][world]), (0, 40))
            image.paste(Image.fromarray(head[source][world]).resize((224, 126)), (8, 405))
            draw = ImageDraw.Draw(image)
            title = "ViBe | recorded stair reference" if world == 0 else "ViBe | W -> SONIC online planner"
            draw.text((10, 10), title, font=font, fill="white")
            measured = records[min(source * 2, len(records) - 1)]
            ended = frame * 2 >= terminals[world]["step"]
            status = terminals[world]["reason"] if ended else "RUNNING"
            color = (110, 230, 145) if ended and terminals[world]["success"] else (240, 220, 180)
            draw.text((245, 410), f"t={measured['time_s']:.2f}s | {status}", font=small, fill=color)
            draw.text((245, 438), f"Forward: {measured['forward_progress_m'][world]:.2f} m", font=font, fill="white")
            draw.text((245, 466), f"Root height gain: {measured['root_height_gain_m'][world]:.2f} m", font=font, fill="white")
            draw.text((245, 495), "Actual policy RGB | fixed observer", font=small, fill="white")
            speed_label = "mode default speed" if args.speed <= 0 else f"speed {args.speed} m/s"
            detail = "recorded stair motion" if world == 0 else speed_label
            draw.text((245, 518), "Result frame held after terminal" if ended else f"model_14999 | {detail}", font=small, fill="white")
            return np.asarray(image)

        video = destination / "vibe_reference_vs_w_fixed_rgb.mp4"
        single = destination / "vibe_w_fixed_rgb.mp4"
        total = max(t["step"] for t in terminals.values()) // 2 + 1
        writers = [iio.get_writer(p, fps=25, codec="libx264", quality=8, macro_block_size=1,
                                 ffmpeg_params=["-threads", "1", "-pix_fmt", "yuv420p"])
                   for p in (video, single)]
        try:
            for frame in range(total + 25):
                panes = [panel(frame, w) for w in range(2)]
                writers[0].append_data(np.concatenate(panes, axis=1))
                if frame <= terminals[1]["step"] // 2 + 25:
                    writers[1].append_data(panes[1])
        finally:
            for writer in writers:
                writer.close()
        for frame in sorted(set([0, min(total - 1, 50), min(total - 1, terminals[1]["step"] // 2)])):
            Image.fromarray(np.concatenate([panel(frame, w) for w in range(2)], axis=1)).save(destination / f"preview_{frame:04d}.jpg")
        report = {"checkpoint": str(Path(args.checkpoint).resolve()), "planner": str(Path(args.planner).resolve()),
                  "clip_id": args.clip_id, "terrain": terrain, "flat_control": args.flat,
                  "command": {"key": "W held", "walk_mode": 2, "speed_m_s": args.speed,
                              "heading_deg": args.heading_deg, "planner_hz": 5, "prediction_frames_30hz": 44},
                  "initial_state": "same recorded first state, no future stair poses for online branch",
                  "initial_state_max_abs_difference": initial_error, "initial_fk_max_abs_error": command.fk_initial_error,
                  "actor_receives_rgb": True, "planner_receives_terrain_or_images": False,
                  "planner_context": "previous generated plan; four identical initial poses at startup",
                  "stair_future_reference_used_by_world_1": False,
                  "tracking_error_terminations_disabled": True, "no_auto_reset": True,
                  "success_criterion": ("full duration; upright; forward progress > 1m" if args.flat else
                                        "forward progress and height match top; both feet high; upright for 0.5s"),
                  "goal_forward_progress_m": goal_progress, "goal_height_gain_m": goal_height,
                  "camera": camera_spec, "observer_max_pose_drift": max_drift,
                  "reference_control": terminals[0], "online_w": terminals[1],
                  "planner_calls": command.planner.calls, "video": str(video.resolve()),
                  "online_video": str(single.resolve())}
        for world, key in [(0, "reference_control"), (1, "online_w")]:
            selected = records[:terminals[world]["step"] + 1]
            report[key]["max_height_gain_m"] = max(r["root_height_gain_m"][world] for r in selected)
            report[key]["max_forward_progress_m"] = max(r["forward_progress_m"][world] for r in selected)
        write_json(destination / "report.json", report)
        write_json(destination / "trajectory.json", records)
        np.save(destination / "generated_reference_qpos.npy", command.planner.poses)
        print("REPORT", json.dumps({k: report[k] for k in ("reference_control", "online_w", "video")}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
