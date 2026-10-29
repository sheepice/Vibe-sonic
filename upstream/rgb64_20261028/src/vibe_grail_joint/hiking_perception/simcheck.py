"""Guarded physics checks of deployment observations/ONNX and standing start-stop.

`parity` uses the training reference solely to compare observation construction.
`keyboard` uses only the portable planner and a scripted W press/release. The
actor receives IMU/joints/RGB/reference/command; world position is metrics only.
"""
import argparse
from dataclasses import fields
import json
import math
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=("parity", "keyboard"), default="parity")
    p.add_argument("--terrain", choices=("flat", "up", "down", "up_down"), default="flat")
    p.add_argument("--seconds", type=float, default=12.)
    p.add_argument("--speed", type=float, default=.25)
    p.add_argument("--walk-seconds", type=float, default=6.)
    p.add_argument("--actuator-guards", action="store_true", help="Apply the same target projection and fault checks as real control")
    args = p.parse_args()
    if (not all(math.isfinite(x) for x in (args.seconds, args.speed, args.walk_seconds))
            or args.seconds <= 0 or args.walk_seconds <= 0 or not 0 < args.speed <= .7):
        p.error("Use positive finite durations and a forward speed in (0, 0.7]")
    output = Path(args.output)
    if output.exists() or output.with_suffix(".rgb.mp4").exists():
        p.error("Use a new output path; previous evidence is never overwritten")
    from ..gpu_guard import verify_guarded_worker
    verify_guarded_worker()
    import mjlab
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from orcs.core.robustness import strip_domain
    from ..velocity.config import env_config
    from ..velocity.eval_command import EvaluationCommandCfg
    from ..deploy.observations import ObservationBuilder, RobotState, Reference
    from ..deploy.reference import OnlineReference
    from .runtime import HikingRuntime as Runtime
    from ..deploy.safety import Limits, target_fault, condition_target, acceptance_signature
    from .evaluate import load_control, augment

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    settings = json.loads(Path(args.config).read_text())
    settings["seed"] = 180018
    bank = str(Path(settings["bank_root"]) / "rank_0")
    from .simulation import evaluation_env
    case_index = ("up", "down", "up_down", "flat").index(args.terrain)*12
    env, settings, cases = evaluation_env(settings, 1, 180018, uniform=True,
                                         duration=args.seconds+5, record_case=case_index)
    atlas = env.unwrapped.cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"]
    tile = cases[0]["tile"]
    rt = Runtime(args.bundle, "cpu")
    report = dict(mode=args.mode, terrain=atlas.random_tiles[tile], requested_seconds=args.seconds,
                  bundle=str(Path(args.bundle).resolve()), bundle_files=rt.manifest["files"],
                  acceptance_signature=acceptance_signature(rt.manifest), hardware_test=False)
    try:
        obs, _ = env.reset()
        raw = env.unwrapped
        robot = raw.scene["robot"]
        c = raw.command_manager.get_term("motion")
        builder = ObservationBuilder(rt.contract)
        builder.heading = Rotation.identity()
        planner = None
        actor = None
        vision_reference = None
        if args.mode == "keyboard":
            # Replace RSI walking initialization with stationary, nominal pose.
            root_state = robot.data.default_root_state.clone()
            root_state[:, :2] = c.tile_origins[c.tile, :2]
            root_state[:, 2] = c.ground_height(root_state[:, :2]) + rt.contract["root_height"]
            root_state[:, 3:7] = torch.tensor([1., 0, 0, 0], device=env.device)
            root_state[:, 7:] = 0
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(robot.data.default_joint_pos, torch.zeros_like(robot.data.default_joint_pos))
            raw.sim.forward()
            raw.scene.update(raw.step_dt)
            raw.observation_manager.reset()
            obs = raw.observation_manager.compute(update_history=True)
            planner = OnlineReference(Path(args.bundle) / "planner.onnx", rt.contract)
        else:
            bank_data = json.loads((Path(bank) / "bank.json").read_text())
            actor, training_memory = load_control(settings, obs, rt.manifest["checkpoint"], raw)
            actor.cpu()
            from ..vision import FrozenTheiaFeature
            vision_reference = FrozenTheiaFeature._load("theia", bank_data["config"]["vision_encoder"], "cpu", torch.float32)
        error = {name: 0. for name in rt.inputs}
        action_error = 0.
        precision_action_error = 0.
        memory_same_input_error = 0.
        vision_cpu_error = 0.
        target_faults = {}
        previous = robot.data.joint_pos[0].cpu().numpy().copy()
        trace, images = [], []
        started = time.monotonic()
        for step in range(round(args.seconds * 50)):
            elapsed = step * .02
            state = RobotState(robot.data.joint_pos[0].cpu().numpy(), robot.data.joint_vel[0].cpu().numpy(),
                               robot.data.root_link_quat_w[0].cpu().numpy(), robot.data.root_link_ang_vel_b[0].cpu().numpy(), elapsed)
            rgb = raw.scene["head_cam"].data.rgb[0].cpu().numpy().copy()
            if step % 5 == 0:
                images.append(rgb)
            patches, cls = rt.encode(rgb[..., ::-1].copy())
            if planner:
                # Smooth velocity ramp, as in the keyboard runtime; default W interval t=2..8.
                requested = args.speed if 2 <= elapsed < min(2.+args.walk_seconds, args.seconds-2.) else 0.
                # Use the same +/-0.01-per-tick ramp as run.py.
                last_speed = trace[-1]["speed_command"] if trace else 0.
                speed = last_speed + float(np.clip(requested-last_speed, -.01, .01))
                if step % 10 == 0:
                    planner.replan(elapsed, speed)
                ref = planner.sample(elapsed)
            else:
                idx = c.future_frames(10, 5)[0]
                ref = Reference(c.motion.joint_pos[idx].cpu().numpy(), c.motion.joint_vel[idx].cpu().numpy(),
                                c.motion.body_quat_w[idx, 0].cpu().numpy(), c.anchor_lin_vel_w[0].cpu().numpy(),
                                c.anchor_ang_vel_w[0].cpu().numpy())
                speed = float(c.speed[0])
            deploy_obs = builder.build(state, ref, patches, cls, speed, velocity=rt.velocity)
            action, target, _ = rt.act(deploy_obs)
            if actor:
                memory_feed=dict(tokens=obs['kv_tokens','img_tokens'].cpu().numpy(),proprio=obs['q_proprio'].cpu().numpy(),
                    elapsed=np.array([.02],np.float32),hidden=training_memory.h.cpu().numpy().copy(),cell=training_memory.c.cpu().numpy().copy(),reset=np.array([step==0],bool))
                obs = augment(obs, training_memory, raw, torch.tensor([step == 0], device=env.device))
                checked_h,checked_c=rt.memory.run(None,memory_feed)
                memory_same_input_error=max(memory_same_input_error,float(abs(checked_h-training_memory.h.cpu().numpy()).max()),float(abs(checked_c-training_memory.c.cpu().numpy()).max()))
                deploy_obs["hiking_latent"] = rt.hidden
                # Training uses fp16 vision; the portable ONNX encoder uses fp32.
                # Measure that precision gap separately from export correctness.
                with torch.no_grad():
                    x = torch.from_numpy(rgb).permute(2,0,1)[None].float()/255.
                    x = (x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
                    hidden = vision_reference.backbone.model(pixel_values=x, interpolate_pos_encoding=True).last_hidden_state.numpy()
                vision_cpu_error = max(vision_cpu_error, float(abs(hidden[0,1:]-patches).max()), float(abs(hidden[0,0]-cls).max()))
                for name in rt.inputs:
                    expected = obs["kv_tokens", "img_tokens"] if name == "kv_tokens__img_tokens" else obs[name]
                    error[name] = max(error[name], float(abs(expected.cpu().numpy()-deploy_obs[name]).max()))
                # Same-input parity isolates export math from GPU encoder roundoff.
                with torch.no_grad():
                    expected_action = actor(obs.cpu()).numpy()[0]
                feed = {n: (obs["kv_tokens", "img_tokens"] if n == "kv_tokens__img_tokens" else obs[n]).cpu().numpy() for n in rt.inputs}
                got = rt.policy.run(["actions"], feed)[0][0]
                action_error = max(action_error, float(abs(expected_action-got).max()))
                precision_action_error = max(precision_action_error, float(abs(expected_action-action).max()))
            projection = 0.
            slew = 0.
            if args.actuator_guards:
                try:
                    target, projection, slew = condition_target(target, previous, rt.contract, Limits())
                    action = (target-rt.default)/rt.scale
                except ValueError as exc:
                    report["guard_stopped_at_s"] = elapsed
                    report["guard_stop_reason"] = str(exc)
                    break
            reason = target_fault(target, previous, state, rt.contract, Limits())
            if reason:
                target_faults[reason] = target_faults.get(reason, 0) + 1
                if args.actuator_guards:
                    report["guard_stopped_at_s"] = elapsed
                    report["guard_stop_reason"] = reason
                    report["guard_joint_diagnostic"] = dict(
                        step_joint=rt.contract["joint_names"][int(np.argmax(abs(target-previous)))],
                        tracking_joint=rt.contract["joint_names"][int(np.argmax(abs(target-state.q)))],
                        max_target_step_rad=float(abs(target-previous).max()),
                        max_target_error_rad=float(abs(target-state.q).max()),
                        target=target.tolist(), previous=previous.tolist())
                    break
            builder.accept_action(action)
            previous = target.copy()
            position = robot.data.root_link_pos_w[0].cpu().numpy()
            ground = float(c.ground_height(robot.data.root_link_pos_w[:, :2])[0])
            upright = float(robot.data.projected_gravity_b[0, 2])
            trace.append(dict(t=elapsed, speed_command=speed, position=position.tolist(),
                              clearance=float(position[2]-ground), gravity_z=upright, target_guard=reason,
                              limit_projection_rad=projection, slew_projection_rad=slew))
            if upright > -.45 or position[2]-ground < .32:
                report["fall_at_s"] = elapsed
                break
            obs, _, _, _ = env.step(torch.from_numpy(action[None]).to(env.device))
        report.update(steps=len(trace), simulated_seconds=len(trace)*.02, fall="fall_at_s" in report,
                      elapsed_wall_s=time.monotonic()-started, observation_max_abs_errors=error if actor else None,
                      actor_onnx_same_input_max_abs=action_error if actor else None, target_guard_counts=target_faults,
                      memory_onnx_same_input_max_abs=memory_same_input_error if actor else None,
                      full_pipeline_action_max_abs_gap=precision_action_error if actor else None,
                      initial_state="stationary_default_pose" if planner else "training_walking_RSI",
                      deploy_onnx_drove_physics=True, raw_rgb_encoder_used=True,
                      no_physics_guards_applied=not args.actuator_guards,
                      encoder_cpu_onnx_max_abs=vision_cpu_error if actor else None,
                      training_vision_dtype="float16", deployment_vision_dtype="float32",
                      note="FP16 training/FP32 deployment feature differences are reported separately. Sim survival does not establish hardware acceptance.",
                      trace=trace)
        if actor:
            report["observation_contract_passed"] = all(v < 2e-5 for k,v in error.items() if k not in ("kv_tokens__img_tokens", "q_cls", "hiking_latent")) and vision_cpu_error < 1e-4 and memory_same_input_error < 1e-5
            report['precision_gap_requires_behavioral_validation'] = True
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        import imageio.v2 as imageio
        if images:
            imageio.mimsave(output.with_suffix(".rgb.mp4"), [np.pad(im, ((0, 1), (0, 0), (0, 0)), mode="edge") for im in images], fps=10, macro_block_size=1)
        print(json.dumps({k:v for k,v in report.items() if k != "trace"}, indent=2), flush=True)
        if actor and not report["observation_contract_passed"]:
            raise RuntimeError("Deployment observation contract comparison failed; see report")
    finally:
        env.close()


if __name__ == "__main__":
    main()
