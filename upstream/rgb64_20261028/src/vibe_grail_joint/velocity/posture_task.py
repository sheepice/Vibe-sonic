"""Upper-body preservation for the existing velocity/visual stair task.

This module adds no actor observations and leaves replay rewards and leg
reference tracking unchanged. All new rewards are masked to online worlds.
Reference body orientations are aligned by the actual/reference pelvis yaw;
neither root translation nor world-space root height is a tracking target.

Enable with ``settings['posture_revision'] = {...}``, then call
``configure_posture_task`` after terrain configuration. ``posture_scalar_metrics``
can be sampled before stepping to avoid measuring auto-reset states. It has no
history/integrator and returns scalar device tensors without CPU synchronization.

The optional support-height reward is disabled by default. Its ground height is
queried under feet, with soft near-contact weighting and a tolerance, but a
discontinuous stair surface can still cause abrupt targets during contact
transitions. Enable only after checking the resulting gait; torso tracking is
the primary posture constraint, not this height heuristic.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .posture_diagnostics import orientation_error_deg, posture_angles_deg, quaternion_multiply


DEFAULTS = {
    "waist_position_weight": 0.5,
    "waist_position_std_rad": 0.18,
    "waist_position_tolerance_rad": 0.03,
    "arms_position_weight": 1.0,
    "arms_position_std_rad": 0.18,
    "arms_position_tolerance_rad": 0.03,
    "waist_velocity_weight": 0.05,
    "waist_velocity_std_rad_s": 1.5,
    "waist_velocity_tolerance_rad_s": 0.15,
    "arms_velocity_weight": 0.10,
    "arms_velocity_std_rad_s": 1.5,
    "arms_velocity_tolerance_rad_s": 0.15,
    "torso_orientation_weight": 0.8,
    "torso_orientation_std_deg": 15.0,
    "torso_orientation_tolerance_deg": 5.0,
    "pelvis_orientation_weight": 0.15,
    "pelvis_orientation_std_deg": 20.0,
    "pelvis_orientation_tolerance_deg": 8.0,
    "support_height_weight": 0.0,
    "support_height_std_m": 0.12,
    "support_height_tolerance_m": 0.08,
    "support_ankle_height_m": 0.04,
    "support_contact_std_m": 0.06,
    "corridor_weight": 0.15,
    "corridor_std_m": 0.60,
    "corridor_tolerance_m": 0.20,
    # Both current procedural generators build boxes with this physical half-width.
    "corridor_half_width_m": 2.0,
    "corridor_margin_m": 0.20,
    "minimum_root_clearance_m": 0.32,
    "fall_gravity_z": -0.45,
}


def _command(env):
    return env.command_manager.get_term("motion")


def _indices(c) -> dict[str, Any]:
    """Cache name-derived geometry indices, never episode state or measurements."""
    cached = getattr(c, "_posture_revision_indices", None)
    if cached is not None:
        return cached
    names = list(c.robot.joint_names)
    bodies = list(c.cfg.body_names)
    waist = [i for i, n in enumerate(names) if n.startswith("waist_")]
    arms = [i for i, n in enumerate(names)
            if n.startswith(("left_", "right_"))
            and any(part in n for part in ("shoulder_", "elbow_", "wrist_"))]
    if len(names) != 29 or len(waist) != 3 or len(arms) != 14:
        raise ValueError("Posture revision requires G1 29-DOF names: 3 waist and 14 arm joints")
    device = c.robot_joint_pos.device
    cached = {
        "waist": torch.tensor(waist, dtype=torch.long, device=device),
        "arms": torch.tensor(arms, dtype=torch.long, device=device),
        "pelvis": bodies.index("pelvis"),
        "torso": bodies.index("torso_link"),
        "feet": torch.tensor([bodies.index("left_ankle_roll_link"),
                              bodies.index("right_ankle_roll_link")], device=device),
    }
    c._posture_revision_indices = cached
    return cached


def soft_tracking(error: torch.Tensor, std: float, tolerance: float) -> torch.Tensor:
    """Unit reward inside a soft deadband, Gaussian decay outside it."""
    return torch.exp(-((error.abs() - tolerance).clamp_min(0) / std).square())


def upper_joint_tracking(env, group: str, std: float, tolerance: float,
                         velocity: bool = False):
    c = _command(env)
    indices = _indices(c)[group]
    actual, reference = ((c.robot_joint_vel, c.joint_vel) if velocity
                         else (c.robot_joint_pos, c.joint_pos))
    delta = actual[:, indices] - reference[:, indices]
    # Apply the tolerance per joint, then average squared residuals. Waist and
    # arms are separate rewards so a wrist cannot dilute a large waist error.
    error_sq = (delta.abs() - tolerance).clamp_min(0).square().mean(-1)
    return torch.exp(-error_sq / std**2) * c.online


def _heading_aligned_orientations(c):
    indices = _indices(c)
    selected = [indices["pelvis"], indices["torso"]]
    actual = c.robot_body_quat_w[:, selected]
    reference = c.body_quat_w[:, selected]
    a, r = actual[:, 0], reference[:, 0]
    yaw_a = torch.atan2(2 * (a[:, 0]*a[:, 3] + a[:, 1]*a[:, 2]),
                        1 - 2 * (a[:, 2].square() + a[:, 3].square()))
    yaw_r = torch.atan2(2 * (r[:, 0]*r[:, 3] + r[:, 1]*r[:, 2]),
                        1 - 2 * (r[:, 2].square() + r[:, 3].square()))
    half = (yaw_a - yaw_r) * 0.5
    zeros = torch.zeros_like(half)
    align = torch.stack((half.cos(), zeros, zeros, half.sin()), -1)
    return actual, quaternion_multiply(align[:, None], reference)


def body_orientation_tracking(env, body: str, std_deg: float, tolerance_deg: float):
    c = _command(env)
    actual, reference = _heading_aligned_orientations(c)
    index = {"pelvis": 0, "torso": 1}[body]
    error = orientation_error_deg(actual[:, index], reference[:, index])
    return soft_tracking(error, std_deg, tolerance_deg) * c.online


def support_height_tracking(env, std: float, tolerance: float, ankle_height: float,
                            contact_std: float):
    c = _command(env)
    indices = _indices(c)
    feet = c.robot_body_pos_w[:, indices["feet"]]
    floor = c.ground_height(feet[:, :, :2])
    # Soft contact proxy, not a force sensor. The nonzero ankle-body clearance
    # matters because body origin and sole surface are different points.
    weights = torch.exp(-((feet[:, :, 2] - floor - ankle_height) / contact_std).square())
    total = weights.sum(-1)
    normalized = weights / total.clamp_min(1e-8)[:, None]
    support_ground = (normalized * floor).sum(-1)
    ref = c.motion.body_pos_w[c.time_steps]
    ref_clearance = ref[:, indices["pelvis"], 2] - (
        normalized * ref[:, indices["feet"], 2]).sum(-1) + ankle_height
    actual_clearance = c.robot.data.root_link_pos_w[:, 2] - support_ground
    score = soft_tracking(actual_clearance - ref_clearance, std, tolerance)
    # No meaningful support estimate in flight: become neutral, not an invented
    # global-height target. This term must not serve as an anti-jumping reward.
    confidence = total.clamp(0, 1)
    return (1 - confidence * (1 - score)) * c.online


def corridor_tracking(env, std: float, tolerance: float):
    c = _command(env)
    offset = c.robot.data.root_link_pos_w[:, 1] - c.tile_origins[c.tile, 1]
    return soft_tracking(offset, std, tolerance) * c.online


def posture_online_fall(env, half_width: float, margin: float,
                        minimum_clearance: float, gravity_z: float):
    c = _command(env)
    root = c.robot.data.root_link_pos_w
    clearance = root[:, 2] - c.ground_height(root[:, :2])
    lateral = (root[:, 1] - c.tile_origins[c.tile, 1]).abs()
    result = c.online & ((clearance < minimum_clearance)
                         | (c.robot.data.projected_gravity_b[:, 2] > gravity_z)
                         | (lateral > half_width - margin)
                         | ~torch.isfinite(root).all(-1))
    c.terminal_fall = result.clone()
    return result


def posture_falling_cost(env, **kwargs):
    return posture_online_fall(env, **kwargs).float()


@torch.no_grad()
def posture_scalar_metrics(env) -> dict[str, torch.Tensor]:
    """Online-only current-state means; call before a step/automatic reset.

    Returned tensors have no gradients and stay on the simulation device. Names
    carry units. A zero online count returns zero-valued means plus count zero,
    so callers can distinguish no data from perfect tracking.
    """
    c = _command(env)
    indices = _indices(c)
    online_mask = getattr(c, "ppo_online", c.online)
    mask = online_mask.to(c.robot_joint_pos.dtype)
    count = mask.sum()
    def mean(value):
        # where avoids propagating non-online NaNs through multiplication by 0.
        return torch.where(online_mask, value, 0.0).sum() / count.clamp_min(1)
    actual, reference = _heading_aligned_orientations(c)
    result = {"online_count": count}
    for i, name in enumerate(("pelvis", "torso")):
        angles = posture_angles_deg(actual[:, i])
        for label in ("tilt_deg", "pitch_deg", "roll_deg"):
            result[f"{name}_{label}"] = mean(angles[label])
        result[f"{name}_reference_error_deg"] = mean(
            orientation_error_deg(actual[:, i], reference[:, i]))
    q_error = c.robot_joint_pos - c.joint_pos
    qd_error = c.robot_joint_vel - c.joint_vel
    for name in ("waist", "arms"):
        selection = indices[name]
        result[f"{name}_joint_rmse_deg"] = mean(torch.rad2deg(
            q_error[:, selection].square().mean(-1).sqrt()))
        result[f"{name}_joint_velocity_rmse_rad_s"] = mean(
            qd_error[:, selection].square().mean(-1).sqrt())
    return result


def _physical_half_width(atlas) -> float:
    # These are the current concrete box geometries, not inferred from a desired
    # penalty width. A new geometry must expose its implemented physical width.
    from .terrain import VelocityAtlasCfg
    from .recovery_terrain import ProfileAtlasCfg
    if type(atlas) in (VelocityAtlasCfg, ProfileAtlasCfg):
        return 2.0
    width = getattr(atlas, "physical_half_width_m", None)
    if width is None or not math.isfinite(float(width)) or float(width) <= 0:
        raise ValueError("Unknown stair geometry must expose actual physical_half_width_m")
    return float(width)


def configure_posture_task(cfg, settings):
    """Configure new online terms; no effect when posture_revision is absent.

    ``posture_revision`` may contain actor-specific options used elsewhere;
    this function reads only keys in DEFAULTS. Coefficients and tolerances are
    fully configurable, including disabling individual rewards with weight 0.
    """
    revision = settings.get("posture_revision")
    if revision is None:
        return cfg
    if not isinstance(revision, dict):
        raise ValueError("posture_revision must be a JSON object")
    from mjlab.managers import RewardTermCfg

    p = {**DEFAULTS, **{k: v for k, v in revision.items() if k in DEFAULTS}}
    for key, value in p.items():
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
            raise ValueError(f"posture_revision.{key} must be finite")
        if key != "fall_gravity_z" and value < 0:
            raise ValueError(f"posture_revision.{key} must be nonnegative")
        if ("_std_" in key or key == "corridor_half_width_m") and value <= 0:
            raise ValueError(f"posture_revision.{key} must be positive")
    if not -1.0 < p["fall_gravity_z"] < 0.0:
        raise ValueError("fall_gravity_z must be between -1 and 0")
    if p["corridor_margin_m"] >= p["corridor_half_width_m"]:
        raise ValueError("corridor margin must be less than physical half-width")
    atlas = cfg.scene.terrain.terrain_generator.sub_terrains["paired_stairs"]
    physical = _physical_half_width(atlas)
    if not math.isclose(p["corridor_half_width_m"], physical, abs_tol=1e-6):
        raise ValueError(f"Corridor half-width {p['corridor_half_width_m']} does not match "
                         f"physical stairs {physical}; change geometry and height queries together")
    if p["corridor_tolerance_m"] >= physical - p["corridor_margin_m"]:
        raise ValueError("corridor tolerance must fit within the allowed corridor")

    cfg.rewards.pop("online_upper_style", None)
    for group in ("waist", "arms"):
        for channel, velocity, unit in (("position", False, "rad"), ("velocity", True, "rad_s")):
            key = f"{group}_{channel}"
            cfg.rewards[f"online_{key}"] = RewardTermCfg(
                func=upper_joint_tracking, weight=p[f"{key}_weight"],
                params=dict(group=group, std=p[f"{key}_std_{unit}"],
                            tolerance=p[f"{key}_tolerance_{unit}"], velocity=velocity))
    for body in ("torso", "pelvis"):
        key = f"{body}_orientation"
        cfg.rewards[f"online_{key}"] = RewardTermCfg(
            func=body_orientation_tracking, weight=p[f"{key}_weight"],
            params=dict(body=body, std_deg=p[f"{key}_std_deg"],
                        tolerance_deg=p[f"{key}_tolerance_deg"]))
    cfg.rewards["online_support_height"] = RewardTermCfg(
        func=support_height_tracking, weight=p["support_height_weight"],
        params=dict(std=p["support_height_std_m"], tolerance=p["support_height_tolerance_m"],
                    ankle_height=p["support_ankle_height_m"], contact_std=p["support_contact_std_m"]))
    cfg.rewards["online_corridor"] = RewardTermCfg(
        func=corridor_tracking, weight=p["corridor_weight"],
        params=dict(std=p["corridor_std_m"], tolerance=p["corridor_tolerance_m"]))
    fall_params = dict(half_width=physical, margin=p["corridor_margin_m"],
                       minimum_clearance=p["minimum_root_clearance_m"], gravity_z=p["fall_gravity_z"])
    # Update both termination and fall-cost function: the original cost calls
    # the old 1.5 m corridor check, which would otherwise silently remain active.
    cfg.terminations["online_fall"].func = posture_online_fall
    cfg.terminations["online_fall"].params = fall_params.copy()
    cfg.rewards["online_fall"].func = posture_falling_cost
    cfg.rewards["online_fall"].params = fall_params.copy()
    return cfg
