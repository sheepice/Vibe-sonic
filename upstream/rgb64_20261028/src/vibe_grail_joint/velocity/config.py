"""New task assembly; original configurations and vendor code stay untouched."""

import math
from dataclasses import asdict, fields
from pathlib import Path

from mjlab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg

from ..env_cfg import build_agent_cfg, build_env_cfg
from ..prepare import validate_bank
from . import mdp
from .command import VelocityMotionCommandCfg
from .terrain import VelocityAtlasCfg, tile_specs


def env_config(bank_path, settings, num_envs):
    cfg = build_env_cfg(bank_path, num_envs=num_envs)
    bank = validate_bank(Path(bank_path))
    tiles = tuple(tile_specs(settings["seed"] + bank["rank"] * 7919))
    old = cfg.commands["motion"]
    params = {f.name: getattr(old, f.name) for f in fields(VelocityMotionCommandCfg) if hasattr(old, f.name)}
    params.update(
        planner_path=settings["planner"],
        planner_speeds=tuple(settings["planner_speeds"]),
        random_tiles=tiles,
        seed=settings["seed"] + bank["rank"],
        init_phase_anneal_iterations=settings["replay_phase_anneal_iterations"],
        curriculum_interval=settings["curriculum_interval"],
    )
    cfg.commands["motion"] = VelocityMotionCommandCfg(**params)
    old_atlas = cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"]
    cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"] = VelocityAtlasCfg(
        root=old_atlas.root, rows=old_atlas.rows, random_tiles=tiles
    )
    cfg.observations["velocity_command"] = ObservationGroupCfg(
        terms={"command": ObservationTermCfg(func=mdp.explicit_command)},
        concatenate_terms=True,
        enable_corruption=False,
    )
    cfg.observations["critic"].terms["explicit_command"] = ObservationTermCfg(func=mdp.explicit_command)
    # Keep physical regularization for all tasks, tracking rewards only on replay.
    for name, term in cfg.rewards.items():
        if name not in ("action_rate_l2", "joint_pos_limits"):
            term.params = {"base_func": term.func, "base_params": term.params}
            term.func = mdp.replay_reward
    for name, func, weight in (
        ("command_velocity", mdp.velocity_tracking, 2.0),
        ("command_heading", mdp.heading_tracking, 0.5),
        ("online_upright", mdp.upright_reward, 0.5),
        ("online_upper_style", mdp.upper_body_style, 0.3),
        ("online_foot_target", mdp.terrain_foot_target, 0.4),
        ("online_progress", mdp.online_progress, 0.3),
        ("online_slip", mdp.foot_slip, -0.2),
        ("online_fall", mdp.falling_cost, -10.0),
    ):
        cfg.rewards[name] = RewardTermCfg(func=func, weight=weight)
    for name, term in cfg.terminations.items():
        if name != "time_out":
            term.params = {"base_func": term.func, "base_params": term.params}
            term.func = mdp.replay_termination
    cfg.terminations["online_fall"] = TerminationTermCfg(func=mdp.online_fall)
    cfg.terminations["online_success"] = TerminationTermCfg(func=mdp.online_success, time_out=True)
    cfg.episode_length_s = 30.0
    cfg.seed = settings["seed"] + bank["rank"]
    if settings.get("terrain_profile") == "irregular_v1":
        from .recovery_terrain import configure_profile_task

        cfg = configure_profile_task(cfg, settings, bank["rank"])
    if settings.get("posture_revision") is not None:
        from .posture_task import configure_posture_task

        cfg = configure_posture_task(cfg, settings)
    if settings.get("speed_refinement") is not None:
        from .speed_refinement import configure_speed_refinement

        cfg = configure_speed_refinement(cfg, settings)
    if settings.get("dagger") is not None:
        from .dagger_task import configure_dagger_task

        cfg = configure_dagger_task(cfg, settings)
    return cfg


def agent_config(bank_path, settings, num_envs):
    base = validate_bank(Path(bank_path))["config"]
    cfg = asdict(build_agent_cfg(base))
    cfg.update(
        experiment_name="vibe_online_velocity_stairs",
        max_iterations=settings["iterations"],
        save_interval=settings["save_interval"],
        seed=settings["seed"],
        amp_dtype=None,
    )
    cfg["actor"]["class_name"] = "vibe_grail_joint.velocity.policy.CommandSonicActor"
    cfg["algorithm"].update(
        class_name="vibe_grail_joint.velocity.policy.PreservationPPO",
        learning_rate=settings.get("learning_rate", 3e-4),
        num_learning_epochs=3,
        num_mini_batches=max(
            4, math.ceil(num_envs * cfg["num_steps_per_env"] / settings["minibatch_env_steps"])
        ),
        preservation_coef=settings["preservation_coef"],
    )
    if "learning_rate_schedule" in settings:
        cfg["algorithm"]["schedule"] = settings["learning_rate_schedule"]
    if settings.get("posture_revision") is not None:
        revision = settings["posture_revision"]
        cfg["algorithm"].update(
            class_name="vibe_grail_joint.velocity.posture_policy.UpperBodyPreservationPPO",
            upper_preservation_coef=revision.get("upper_preservation_coef", .02),
            upper_preservation_std_rad=revision.get("upper_preservation_std_rad", .2),
            upper_waist_weight=revision.get("upper_waist_weight", 1.0),
            upper_arms_weight=revision.get("upper_arms_weight", 1.0),
            upper_preservation_samples=revision.get("upper_preservation_samples", 128),
        )
    if settings.get("dagger") is not None:
        if settings.get("posture_revision") is None:
            raise ValueError("DAgger stage uses the existing posture task")
        cfg["algorithm"].update(
            class_name="vibe_grail_joint.velocity.dagger_policy.DaggerPPO",
            preservation_coef=0.0,
            dagger_coef=settings["dagger"]["coefficient"],
            dagger_batch_size=settings["dagger"]["batch_size"],
            num_mini_batches=max(1, math.ceil((num_envs * 7 // 10) * cfg["num_steps_per_env"]
                                              / settings["minibatch_env_steps"])),
        )
    return cfg
