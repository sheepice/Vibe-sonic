"""Distributed RGB stairs and stepping-stones trainer; original trainers preserved."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import traceback
from pathlib import Path

from ..gpu_guard import verify_guarded_worker


def validate_mesh_resume(metadata, edge_hashes, world):
    """Each rank uses its own seeded terrain; compare the full rank-ordered set."""
    if metadata.get("world_size") != world:
        raise ValueError("Foothold resume world size changed")
    saved = metadata.get("edge_mesh_sha256_by_rank")
    if not isinstance(saved, list) or len(saved) != world:
        raise ValueError("Checkpoint lacks complete per-rank terrain mesh hashes")
    if saved != edge_hashes:
        raise ValueError("Hiking terrain mesh changed on resume")


def main(guard_check=verify_guarded_worker):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--num-envs", type=int)
    p.add_argument("--iterations", type=int)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--diagnostic", action="store_true", help="Bounded two-rank fault reproduction, not a full experiment")
    p.add_argument("--resume")
    p.add_argument("--revise-curriculum", action="store_true",
                   help="Explicitly enable recent-window sampling while restoring all model/optimizer state")
    p.add_argument("--continue-policy", help="New run after perception refit: actor-only continuation, fresh optimizer/replay")
    p.add_argument("--mode", choices=("geometry",), required=True)
    p.add_argument("--perception-checkpoint")
    args = p.parse_args()
    if args.resume and args.continue_policy:
        p.error("Choose exact resume OR actor continuation after perception refit")
    if args.revise_curriculum and not args.resume:
        p.error('--revise-curriculum requires --resume')
    guard_check()
    import mjlab  # noqa: F401 -- initialize upstream task discovery
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from ..catalog import write_json
    from ..train_stairs import frozen_digest
    from .task import agent_config, env_config, QualityStairsEnv, options, REMOVED_REWARDS

    settings = json.loads(Path(args.config).read_text())
    from ..hiking_perception.contracts import require_hiking
    from .control import PerceptionWrapper
    from .policy import configure_agent
    require_hiking(settings)
    experiment_options = options(settings)
    from .curriculum import validate_resume_options
    dagger_enabled = settings.get("dagger") is not None
    if not dagger_enabled:
        raise ValueError("Foothold comparison requires the existing DAgger split")
    if settings.get("speed_refinement") is not None and not args.resume:
        raise ValueError("Speed refinement requires --resume; actor-only warm start would reset the critic, optimizer and curriculum")
    rank, local, world = [
        int(os.environ.get(k, d)) for k, d in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1"))
    ]
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    device = f"cuda:{local}"
    torch.manual_seed(settings["seed"] + rank)
    n = args.num_envs or settings["num_envs_per_rank"]
    iterations = args.iterations or settings["iterations"]
    if n < 10 or n % 10:
        raise ValueError("num-envs must be a multiple of 10, preserving 60/20/20 task fractions")
    if dagger_enabled and (n < 20 or n % 20):
        raise ValueError("DAgger num-envs must be divisible by 20 for 50/20/15/15 fractions")
    if not args.smoke and not args.diagnostic and world != settings["expected_world_size"]:
        raise ValueError("World size does not match this isolated experiment configuration")
    if args.diagnostic and iterations > 200:
        raise ValueError("Fault reproduction is limited to 200 updates")
    if args.smoke and iterations > 10:
        raise ValueError("Smoke is limited to 10 updates")
    if world < 2:
        raise ValueError("Every stepping-stones experiment requires at least two training ranks")
    bank = str(Path(settings["bank_root"]) / f"rank_{0 if args.smoke else rank}")
    out = Path(args.log_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / f"rank_{rank}_status.json"
    if status_path.exists() and not args.resume:
        raise ValueError(f"Refusing to overwrite an existing run: {out}")
    record = {
        "status": "initializing",
        "rank": rank,
        "world_size": world,
        "num_envs": n,
        "settings": settings,
        "requested_iterations": iterations,
        "smoke": args.smoke,
        "diagnostic": args.diagnostic,
        "initialization": args.resume or args.continue_policy or settings["initial_checkpoint"],
    }
    write_json(status_path, record)
    started = time.monotonic()

    class LoggingWrapper(RslRlVecEnvWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.returns = torch.zeros(n, device=device)
            self.lengths = torch.zeros(n, device=device)
            self.counters = {"online_episodes": 0, "online_success": 0, "online_falls": 0}
            for name in ("up", "down", "up_down", "online_flat"):
                for suffix in ("episodes", "success", "falls"):
                    self.counters[f"{name}_{suffix}"] = 0
            for level in range(settings.get("profile_levels", 8)):
                for suffix in ("episodes", "success", "falls"):
                    self.counters[f"level_{level}_{suffix}"] = 0
            self.recent = {}
            self.posture_sums = {}
            self.posture_steps = 0
            self.quality_sums = {}
            self.quality_steps = 0

        def step(self, actions):
            c = self.unwrapped.command_manager.get_term("motion")
            online_mask = getattr(c, "ppo_online", c.online)
            posture_metrics = {}
            if settings.get("posture_revision") is not None:
                from ..velocity.posture_task import posture_scalar_metrics

                posture_metrics = {f"Posture/{k}": v for k, v in posture_scalar_metrics(self.unwrapped).items()}
                for key, value in posture_metrics.items():
                    self.posture_sums[key] = self.posture_sums.get(key, 0.) + value.detach()
                self.posture_steps += 1
            # Before step: these are genuine states, never auto-reset positions.
            vel = c.robot.data.root_link_lin_vel_w
            error = ((vel[:, 0] - c.speed).square() + vel[:, 1].square()).sqrt()
            progress = c.robot.data.root_link_pos_w[:, 0] - c.tile_origins[c.tile, 0]
            terrain_kind = c.tile_data["kind"][c.tile].clone()
            terrain_level = c.tile_data["level"][c.tile].clone()
            moving = c.speed > 0.1
            obs, reward, done, extras = super().step(actions)
            self.returns += reward
            self.lengths += 1
            logs = extras.setdefault("log", {})
            logs.update(posture_metrics)
            logs.update(self.unwrapped.guidance.metrics)
            for key, value in self.unwrapped.guidance.metrics.items():
                if key.startswith(("StepQuality/", "Hiking/", "SSR/", "FootPose/")):
                    self.quality_sums[key] = self.quality_sums.get(key, 0.) + (value.detach() if torch.is_tensor(value) else value)
            self.quality_steps += 1
            for name in ("online", "paired", "flat"):
                mask = online_mask if name == "online" else getattr(c, name)
                logs[f"Velocity/{name}_reward"] = reward[mask].mean()
                ended = mask & done.bool()
                if ended.any():
                    logs[f"Velocity/{name}_episode_return"] = self.returns[ended].mean()
                    logs[f"Velocity/{name}_episode_seconds"] = self.lengths[ended].mean() * 0.02
            ended = online_mask & done.bool()
            success = c.terminal_success & ended
            fallen = c.terminal_fall & ended
            self.counters["online_episodes"] += int(ended.sum())
            self.counters["online_success"] += int(success.sum())
            self.counters["online_falls"] += int(fallen.sum())
            for kind, name in enumerate(("up", "down", "up_down", "online_flat")):
                terminal = ended & moving & (terrain_kind == kind)
                self.counters[f"{name}_episodes"] += int(terminal.sum())
                self.counters[f"{name}_success"] += int((terminal & success).sum())
                self.counters[f"{name}_falls"] += int((terminal & fallen).sum())
                if terminal.any():
                    logs[f"Velocity/{name}_success_rate"] = (
                        terminal & success
                    ).sum().float() / terminal.sum()
            for level in range(settings.get("profile_levels", 8)):
                terminal = ended & moving & (terrain_kind != 3) & (terrain_level == level)
                self.counters[f"level_{level}_episodes"] += int(terminal.sum())
                self.counters[f"level_{level}_success"] += int((terminal & success).sum())
                self.counters[f"level_{level}_falls"] += int((terminal & fallen).sum())
            logs["Velocity/online_speed_error_m_s"] = error[online_mask].mean()
            logs["Velocity/online_progress_m"] = progress[online_mask].mean()
            logs["Velocity/terrain_level"] = c.tile_data["level"][c.tile[online_mask]].float().mean()
            logs["Velocity/planner_calls"] = float(c.pool.calls)
            logs["Velocity/gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            if ended.any():
                logs["Velocity/online_success_rate"] = success.sum().float() / ended.sum()
                logs["Velocity/online_fall_rate"] = fallen.sum().float() / ended.sum()
            self.recent = {k: float(v) for k, v in logs.items()
                           if k.startswith(("Velocity/", "SSR/", "Hiking/", "FootPose/", "StepQuality/", "Stones/", "Course/", "DeploymentTarget/"))}
            self.returns[done.bool()], self.lengths[done.bool()] = 0, 0
            return obs, reward, done, extras

    class Runner(MjlabOnPolicyRunner):
        def _configure_multi_gpu(self):
            from ..velocity.recovery_support import configure_distributed

            configure_distributed(self, settings.get("collective_timeout_seconds", 180))

        def save(self, path, infos=None):
            # Atomic checkpoint writes keep the latest complete file reviewable.
            if frozen_digest(self.alg.get_policy()) != before:
                raise RuntimeError("Frozen SONIC parameters changed before checkpoint save")
            tmp = str(path) + ".tmp"
            from ..velocity.recovery_support import checkpoint_metadata

            metadata = checkpoint_metadata(self, settings, infos)
            if dagger_enabled:
                metadata["dagger_replay"] = dict(world_size=world, num_envs_per_rank=n,
                                                 pattern=f"dagger_{self.current_learning_iteration}_rank_{{rank}}.pt")
            metadata["hiking_perception"] = perception_env.metadata
            metadata["gaussgym"] = perception_env.snapshot()
            metadata["stepping_stones"]=dict(version=1,settings=settings['stones'],
                curriculum=env.unwrapped.command_manager.get_term('motion').course_state(),
                deploy_inputs=['RGB','robot proprioception','velocity command','online SONIC reference'],
                no_privileged_actor_inputs=True)
            metadata["foothold_experiment"] = dict(version=2, options=experiment_options, world_size=world,
                edge_mesh_sha256=env.unwrapped.guidance.edge_hash,
                edge_mesh_sha256_by_rank=edge_hashes)
            super().save(tmp, metadata)
            os.replace(tmp, path)

    env = None
    try:
        cfg = env_config(bank, settings, n)
        # Establish the absolute update before the first resumed episode reset.
        resume_updates = 0
        if args.resume:
            header = torch.load(args.resume, map_location="cpu", weights_only=False)
            record['curriculum_resume'] = validate_resume_options(
                header.get('infos', {}).get('foothold_experiment', {}).get('options'),
                experiment_options, allow_revision=args.revise_curriculum)
            resume_updates = int(header["iter"]) + 1
            previous_objective = header.get('infos',{}).get('gaussgym',{}).get('settings',{}).get('occupancy_objective')
            current_objective = settings['gaussgym'].get('occupancy_objective')
            record['geometry_supervision_resume'] = dict(previous=previous_objective, current=current_objective,
                objective_changed=previous_objective != current_objective,
                actor_inputs_changed=False, restores_optimizer_and_curriculum=True)
            del header
        agent = agent_config(bank, settings, n)
        if args.smoke:
            agent["save_interval"] = 1
        from .labels import rgb_only
        rgb_only(cfg)
        logging_env = LoggingWrapper(QualityStairsEnv(cfg, device=device, settings=settings))
        from .targets import install as install_targets, configure_algorithm, contract_summary
        install_targets(logging_env.unwrapped, settings)
        record['deployment_targets'] = contract_summary(settings)
        logging_env.unwrapped.diagnostic_dir=str(out)
        if settings['stones'].get('physics_audit'):
            from .physics_audit import install
            install(logging_env.unwrapped, out)
            record['physics_substep_audit_active'] = True
        checkpoint = args.perception_checkpoint
        if args.resume and args.mode == "geometry":
            checkpoint = args.resume
        perception_env = PerceptionWrapper(logging_env, mode=args.mode, checkpoint=checkpoint,
            settings=settings["gaussgym"])
        agent = configure_agent(agent, perception_env.dimension,settings["stones"]["variant"])
        env = perception_env
        if dagger_enabled:
            from .control import OnlineDaggerVecEnv as DaggerVecEnv

            env = DaggerVecEnv(env, settings, rank)
        if settings.get("posture_revision") is not None:
            action = env.unwrapped.action_manager.get_term("joint_pos")
            names = list(action.target_names)
            if names != list(env.unwrapped.scene["robot"].joint_names):
                raise RuntimeError("SONIC action order differs from robot joint order")
            scale = torch.as_tensor(action.scale, device=device)
            if scale.ndim == 0:
                scale = scale.expand(29)
            elif scale.ndim == 2:
                if not torch.equal(scale, scale[:1].expand_as(scale)):
                    raise RuntimeError("Upper action prior requires the same target scales in every world")
                scale = scale[0]
            agent["algorithm"].update(action_scales=scale.cpu().tolist(), action_joint_names=names)
        configure_algorithm(agent, env, settings)
        agent_snapshot = copy.deepcopy(agent)
        runner = Runner(env, agent, str(out), device)
        perception_env.synchronize()
        edge_hashes = [env.unwrapped.guidance.edge_hash]
        if world > 1:
            edge_hashes = [None] * world
            torch.distributed.all_gather_object(edge_hashes, env.unwrapped.guidance.edge_hash)
        env.unwrapped.guidance.synchronize_initial_model()
        actor = runner.alg.get_policy()
        if rank == 0 and args.mode == "geometry" and not args.resume:
            import shutil
            shutil.copy2(args.perception_checkpoint, out / "perception.pt")
        if dagger_enabled:
            env.attach(runner.alg, agent_snapshot["actor"], agent_snapshot["obs_groups"])
        if args.resume:
            restored_infos = runner.load(args.resume, map_location=device)
            if restored_infos.get("hiking_perception") != perception_env.metadata:
                raise ValueError("Resume perception weights/contract mismatch")
            if dagger_enabled:
                from ..velocity.dagger_checkpoint import restore_replay

                record["dagger_replay_restore"] = restore_replay(
                    env, args.resume, runner.current_learning_iteration, rank, world, n,
                    runner.alg.teacher_sha256, restored_infos,
                    allow_env_count_change=settings["dagger"].get("allow_env_count_change_on_resume", False),
                )
            validate_resume_options(restored_infos.get('foothold_experiment', {}).get('options'),
                                    experiment_options, allow_revision=args.revise_curriculum)
            validate_mesh_resume(restored_infos["foothold_experiment"], edge_hashes, world)
            env.unwrapped.guidance.load(args.resume, runner.current_learning_iteration, rank, world)
            for key,value in restored_infos['stepping_stones']['curriculum'].items():
                getattr(env.unwrapped.command_manager.get_term('motion'),key).copy_(value)
            runner.current_learning_iteration += 1
            # Mjlab restores common_step_counter; also synchronize the cached
            # curriculum value before resetting the newly created physics worlds.
            env.unwrapped.common_step_counter = resume_updates * agent_snapshot["num_steps_per_env"]
            env.unwrapped.policy_update_count = resume_updates
            env.unwrapped.command_manager.get_term("motion")._init_phase_max = max(
                0.0, 1.0 - resume_updates / max(1, settings["replay_phase_anneal_iterations"])
            )
            env.reset()
        else:
            if args.continue_policy:
                state = torch.load(args.continue_policy, map_location=device, weights_only=False)
                if state.get("infos", {}).get("hiking_perception", {}).get("mode") != args.mode:
                    raise ValueError("Continuation requires the same RGB geometry actor architecture")
                missing,unexpected=actor.load_state_dict(state["actor_state_dict"],strict=False)
                permitted={k for k in actor.state_dict() if k.startswith(('experts.','router.'))}
                if unexpected or not set(missing).issubset(permitted):raise RuntimeError((missing,unexpected))
                record["actor_continuation"] = dict(checkpoint=args.continue_policy,
                    fresh_critic_optimizer=True, fresh_dagger_replay=True, curriculum_restarted=True)
                del state
            else:
                state = torch.load(settings["initial_checkpoint"], map_location=device, weights_only=False)
                if any(k.startswith("command_encoder.") for k in state["actor_state_dict"]):
                    raise ValueError("Fresh stage 2 requires the stage-1 actor WITHOUT a trained velocity command branch")
                missing, unexpected = actor.load_state_dict(state["actor_state_dict"], strict=False)
                expected = {"command_encoder.0.weight", "command_encoder.0.bias",
                            "command_encoder.2.weight", "command_encoder.2.bias"}
                expected |= {k for k in actor.state_dict() if k.startswith("geometry_encoder.")}
                if unexpected or set(missing) != expected:
                    raise RuntimeError(f"Unexpected stage-1 checkpoint mismatch: {missing}, {unexpected}")
                if actor.geometry_encoder[-1].weight.any() or actor.geometry_encoder[-1].bias.any():
                    raise RuntimeError("Geometry condition must initialize to zero")
                if actor.command_encoder[-1].weight.any() or actor.command_encoder[-1].bias.any():
                    raise RuntimeError("New second-stage command output must start at zero")
                record["fresh_stage2"] = dict(stage1_checkpoint=str(Path(settings["initial_checkpoint"]).resolve()),
                    stage1_iteration=state.get("iter"), fresh_critic_optimizer=True,
                    command_output_zero=True, dagger_history_empty=len(env.buffer)==0,
                    auxiliary_predictor_updates=env.unwrapped.guidance.update_count)
                del state
        with torch.no_grad():
            obs_check=env.get_observations()[:4].clone()
            sample=actor(obs_check,stochastic_output=False)
            changed=obs_check.clone()
            if 'critic' in changed:changed['critic'].fill_(918.25)
            changed['fake_ground_truth']=torch.full((4,100),-999.,device=device)
            difference=(actor(changed,stochastic_output=False)-sample).abs().max()
            if difference>1e-6:raise RuntimeError('Privileged actor leakage')
            record['privileged_input_invariance_error']=float(difference)
        before = frozen_digest(actor)
        if dagger_enabled:
            with torch.no_grad():
                sample = env._full_obs[:min(n, 8)]
                error = (env.teacher(sample) - actor.sonic_action(sample)).abs().max()
                if float(error) > 1e-5:
                    raise RuntimeError(f"Student frozen base differs from original SONIC teacher: {float(error)}")
            record["teacher_base_max_error"] = float(error)
        record.update(
            status="running",
            frozen_sha256=before,
            total_actor_parameters=sum(p.numel() for p in actor.parameters()),
            trainable_actor_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad),
            actor_groups=list(env.get_observations().keys()),
            hiking_perception=perception_env.metadata,
            resumed_after_updates=resume_updates,
            started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        write_json(status_path, record)
        if rank == 0:
            (out / "agent_cfg.json").write_text(json.dumps(agent_snapshot, indent=2, default=str))
            write_json(
                out / "task_contract.json",
                {
                    "robot": "G1 29DOF",
                    "actor_camera": "D435i RGB 112x63",
                    "depth_input": False,
                    "hiking_perception": perception_env.metadata,
                    "base": "frozen SONIC encoder, FSQ and decoder base weights; train ViBe LoRA, extractor, command and geometry projections; online GaussGym supervised perception updates between PPO rollouts",
                    "online_reference": "runtime CPU ONNX, 5Hz, five shared rolling streams per rank; independent physics, phase and terrain",
                    "command": "independent target planar speed and +X heading, velocity_command[8]; stop uses planner idle mode",
                    "fractions": ({"online_ppo": .5, "paired_grail_ppo": .2,
                                   "flat_dagger_planner": .15, "flat_dagger_replay": .15}
                                  if dagger_enabled else
                                  {"online": .6, "paired_grail": .2, "flat_mixed_including_sonic": .2}),
                    "dagger": settings.get("dagger"),
                    "foothold_online_ppo_only": experiment_options,
                    "removed_rewards": list(REMOVED_REWARDS),
                    "quality_revision": settings["step_quality"],
                    "slip_measurement": "force-gated actual contact-point tangential velocity at every physics substep",
                    "contact_point_slip_weight": cfg.rewards["online_slip"].weight,
                    "auxiliary_deployment_required": args.mode == "geometry",
                    "ssr_paper": "https://arxiv.org/html/2605.30770v1#S3.SS2",
                    "method": settings["foothold_experiment"]["method"],
                    "hiking_paper": "https://arxiv.org/html/2601.07718v1#S3.SS3",
                    "edge_mesh_sha256": env.unwrapped.guidance.edge_hash,
                    "edge_mesh_sha256_by_rank": edge_hashes,
                    "edge_counts": env.unwrapped.guidance.edge_counts,
                    "teacher_sha256": getattr(runner.alg, "teacher_sha256", None),
                    "privileged": "height/velocity targets absent from actor inputs; no depth camera stream; recurrent module updated by separate supervision",
                    "curriculum": ("ordinary stairs levels 0-10 retained; restored custom frontiers preserved; promotion uses recent-window evidence configured below; easier-level sampling retained"
                                   if settings['stones'].get('recent_curriculum') else
                                   "ordinary stairs levels 0-10 retained; separate box/cylinder/gap frontiers start at 0; promote >=80 episodes and >=72% frontier success, minimum 150 updates per promotion"),
                    "recent_curriculum": settings['stones'].get('recent_curriculum'),
                    "curriculum_resume": record.get('curriculum_resume'),
                    "stones": settings['stones'],
                    "geometry_supervision": "2D box/cylinder top query; saturation occupancy trained; no depth input",
                    "geometry_occupancy_objective": settings['gaussgym'].get('occupancy_objective', {'kind':'original_bce'}),
                    "actor_normalization": "frozen at staircase checkpoint for consistent multi-rank inference",
                    "scope": "forward/stop locomotion; arbitrary turns and large within-episode speed transitions need later extension",
                    "frozen_sha256": before,
                    "posture_revision": settings.get("posture_revision"),
                    "posture_metrics": "pre-step online measurements averaged across the complete rollout" if settings.get("posture_revision") is not None else None,
                    "initialization_mode": "restore model, critic and optimizer" if args.resume else "actor continuation after perception refit; new critic/optimizer/replay" if args.continue_policy else "stage-1 actor warm start; new critic and optimizer",
                },
            )
        # Record all ranks after every update, not just rank-zero console output.
        original_update = runner.alg.update

        def monitored_update():
            losses, info = original_update()
            update_index = resume_updates + record.get("completed_updates", 0)
            losses.update(perception_env.update_perception(update_index, env))
            losses.update(env.unwrapped.guidance.update())
            course_metrics=env.unwrapped.command_manager.get_term('motion').update_course(update_index+1)
            if runner.logger.writer is not None:
                for name, value in course_metrics.items():
                    runner.logger.writer.add_scalar(name, value, update_index)
            import math
            if any(not math.isfinite(float(value)) for value in losses.values()):
                raise RuntimeError(f"Nonfinite training losses: {losses}")
            if dagger_enabled:
                # Base runner writes model files only on rank zero. Every rank
                # first writes its own replay atomically, then synchronizes.
                from ..velocity.dagger_checkpoint import save_replay

                it = resume_updates + record.get("completed_updates", 0)
                if it % agent_snapshot["save_interval"] == 0 or it == resume_updates + iterations - 1:
                    save_replay(env, out, it, rank, world, n, runner.alg.teacher_sha256)
                    env.unwrapped.guidance.save(out, it, rank, world)
            rollout_metrics = dict(logging_env.recent)
            rollout_metrics.update(course_metrics)
            rollout_metrics.update({f'DeploymentTarget/{k}':float(v) for k,v in info.items()
                                    if k.startswith('deployment_')})
            if logging_env.quality_steps:
                rollout_metrics.update({k: float(v / logging_env.quality_steps) for k, v in logging_env.quality_sums.items()})
                logging_env.quality_sums.clear()
                logging_env.quality_steps = 0
            rollout_metrics.update(env.unwrapped.guidance.update_metrics())
            if logging_env.posture_steps:
                rollout_metrics.update({k: float(v / logging_env.posture_steps) for k, v in logging_env.posture_sums.items()})
                logging_env.posture_sums.clear()
                logging_env.posture_steps = 0
                rollout_metrics.update({f"PostureAction/{k}": float(v) for k, v in info.items()
                                        if k.startswith("online_")})
            motion = env.unwrapped.command_manager.get_term("motion")
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            used_gib = (total_bytes - free_bytes) / 2**30
            record.update(
                status="running",
                completed_updates=record.get("completed_updates", 0) + 1,
                absolute_completed_updates=resume_updates + record.get("completed_updates", 0) + 1,
                policy_update_count=int(getattr(env.unwrapped, "policy_update_count", 0)),
                replay_init_phase_max=float(motion._init_phase_max),
                learning_rate=float(runner.alg.learning_rate),
                updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                elapsed_s=time.monotonic() - started,
                loss=losses,
                metrics=rollout_metrics,
                counters=logging_env.counters,
                gpu_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                gpu_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                device_used_gib=used_gib,
                max_observed_device_used_gib=max(record.get("max_observed_device_used_gib", 0.0), used_gib),
            )
            write_json(status_path, record)
            with (out / f"rank_{rank}_metrics.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            k: record[k]
                            for k in (
                                "completed_updates",
                                "elapsed_s",
                                "loss",
                                "metrics",
                                "counters",
                                "gpu_peak_reserved_gib",
                            )
                        }
                    )
                    + "\n"
                )
            return losses, info

        runner.alg.update = monitored_update
        runner.learn(iterations, init_at_random_ep_len=False)
        after = frozen_digest(actor)
        if before != after:
            raise RuntimeError("Frozen SONIC weights changed")
        if dagger_enabled:
            if env.teacher.training or any(p.requires_grad or p.grad is not None for p in env.teacher.parameters()):
                raise RuntimeError("DAgger teacher must remain frozen and in eval mode")
            record.update(teacher_sha256=runner.alg.teacher_sha256, teacher_frozen=True,
                          ppo_envs=env.num_envs, dagger_envs=len(env.dagger_ids),
                          dagger_history_samples=len(env.buffer))
        record.update(status="completed", frozen_unchanged=True, frozen_sha256_after=after)
        write_json(status_path, record)
    except BaseException as e:
        # Write the ORIGINAL stack before cleanup can block on another NCCL rank.
        stack = traceback.format_exc()
        print(stack, flush=True)
        (out / f"rank_{rank}_traceback.txt").write_text(stack)
        record.update(status="failed", error=repr(e), traceback=stack)
        write_json(status_path, record)
        if world > 1:
            # A failing rank must not wait in CUDA/NCCL teardown while its peers
            # are blocked in a collective. torchrun will terminate our sibling ranks.
            import sys

            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        raise
    finally:
        if env is not None:
            env.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
