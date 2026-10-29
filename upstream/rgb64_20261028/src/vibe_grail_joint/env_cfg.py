"""Full GRAIL stairs + existing flat library, with a strictly RGB visual actor."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import mjlab  # noqa: F401 -- initialize upstream task discovery first
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from orcs.core.robustness import strip_domain
from orcs.tasks.perloco.env_cfg import _core
from orcs.tasks.perloco.roster import Roster
from orcs.tasks.perloco.sensors import PERLOCO_KILL_BODIES, TERRAIN_SCAN_SENSOR_NAME
from vibe.core.env_cfgs import apply_query_noise, apply_render_domain
from vibe.core.sensors import attach_head_cam
from vibe.tasks.perloco.config.g1.observation_cfgs import ObsCtx, attach_ext_obs

from .commands import JointMotionCommandCfg
from .prepare import validate_bank
from .terminations import completed_reference
from .terrain import FLOOR_RGBA, JointAtlasCfg, atlas_layout
from .vision import THEIA_REVISION, FrozenTheiaFeature


def build_env_cfg(bank_path: str, *, num_envs: int, play: bool = False):
    metadata = validate_bank(Path(bank_path))
    settings = metadata["config"]
    max_length = max(c["length"] for c in metadata["clips"])
    cfg = _core(
        "vibe_grail_joint",
        Roster(("joint",), (0.0,), {}),
        "",
        max_length,
        agent="sonic",
        play=False,
        scan_frame="pelvis",
        tile_size=(12.0, 12.0),
        num_steps_per_env=24,
        robot_cfg=get_g1_robot_cfg,
        kill_bodies=PERLOCO_KILL_BODIES,
        kill_exclude=(),
        anchor_pos_thresh=settings["anchor_position_threshold"],
        anchor_ori_thresh=settings["anchor_orientation_threshold"],
    )
    if play:
        # Upstream's visualization mode removes tracking failures. Evaluation must retain them.
        strip_domain(cfg)
        cfg.events.pop("policy_update_counter", None)
    rows = metadata["terrain_rows"]
    _, size = atlas_layout(rows)
    cfg.scene.terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=size,
            num_rows=1,
            num_cols=1,
            border_width=0.0,
            curriculum=False,
            sub_terrains={"paired_stairs": JointAtlasCfg(root=settings["terrain_root"], rows=tuple(rows))},
        ),
    )
    old = cfg.commands["motion"]
    common = {
        field.name: getattr(old, field.name)
        for field in fields(JointMotionCommandCfg)
        if hasattr(old, field.name)
    }
    common.update(
        bank_path=str(Path(bank_path).resolve()),
        terrain_only=metadata["split"] == "validation",
        future_steps=10,
        debug_vis=False,
        start_from_zero=play,
        init_phase_anneal_iterations=settings["phase_anneal_iterations"],
    )
    cfg.commands["motion"] = JointMotionCommandCfg(**common)
    cfg.scene.num_envs = num_envs
    # A large atlas must not inflate MuJoCo's near clipping distance around the robot.
    cfg.scene.extent = 5.0
    cfg.sim.nconmax, cfg.sim.njmax = 512, 1500
    hold = settings["motion_end_hold_seconds"]
    cfg.episode_length_s = max_length / settings["control_hz"] + hold + 0.1
    cfg.terminations["exceeded_motion"].func = completed_reference
    cfg.terminations["exceeded_motion"].params = {
        "command_name": "motion",
        "terrain_hold_steps": round(hold * settings["control_hz"]),
    }
    camera = settings["camera"]
    attach_head_cam(cfg, **{key: camera[key] for key in ("width", "height", "fovy", "pos", "quat")})
    attach_ext_obs(cfg, ctx=ObsCtx(p={"command_name": "motion"}, model=settings["vision_encoder"]))
    if settings.get("vision_revision", THEIA_REVISION) != THEIA_REVISION:
        raise ValueError("Encoder revision differs from this experiment's pinned implementation")
    cfg.observations["kv_tokens"].terms["img_tokens"].func = FrozenTheiaFeature
    cfg.observations["q_cls"].terms["img_cls"].func = FrozenTheiaFeature
    apply_render_domain(cfg, play=play, terrain_ground_rgba=FLOOR_RGBA)
    apply_query_noise(cfg, play=play)
    for sensor in cfg.scene.sensors:
        if getattr(sensor, "name", None) == TERRAIN_SCAN_SENSOR_NAME:
            sensor.debug_vis = False
    if cfg.sim.mujoco.timestep * cfg.decimation != 1 / settings["control_hz"]:
        raise ValueError("Reference and simulation control frequencies differ")
    return cfg


def build_agent_cfg(settings):
    from vibe.tasks.perloco.config.g1.agent_cfgs import adapt_sonic_ext_agent_cfg

    cfg = adapt_sonic_ext_agent_cfg("vibe_grail_joint", rank=settings["lora_rank"])
    cfg.max_iterations = settings["max_iterations"]
    cfg.save_interval = settings["save_interval"]
    cfg.amp_dtype = settings["amp_dtype"]
    cfg.logger = "tensorboard"
    return cfg
