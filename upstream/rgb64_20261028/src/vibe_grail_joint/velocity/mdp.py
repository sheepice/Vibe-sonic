"""Task-separated rewards: reference replay versus perception-adaptive locomotion."""

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
from orcs.core.mdp.observations import robot_root_ang_vel_cmd, robot_root_lin_vel_cmd


def command(env):
    return env.command_manager.get_term("motion")


def explicit_command(env):
    c = command(env)
    q = yaw_quat(c.robot.data.root_link_quat_w)
    velocity = torch.zeros(env.num_envs, 3, device=env.device)
    velocity[:, 0] = c.speed
    velocity = quat_apply_inverse(q, velocity)
    ref_v = robot_root_lin_vel_cmd(env, "motion")
    ref_w = robot_root_ang_vel_cmd(env, "motion")
    out = torch.zeros(env.num_envs, 8, device=env.device)
    out[:, :2] = torch.where(c.online[:, None], velocity[:, :2], ref_v[:, :2])
    out[:, 2] = torch.where(c.online, 0.0, ref_w[:, 2])
    # Heading is relative to the commanded world +X direction; no position sensor.
    out[:, 3] = (2 * q[:, 3].square()) * c.online
    out[:, 4] = (-2 * q[:, 0] * q[:, 3]) * c.online
    out[:, 5], out[:, 6], out[:, 7] = c.online.float(), c.paired.float(), c.flat.float()
    return out


def replay_reward(env, base_func, base_params):
    return base_func(env, **base_params) * (~command(env).online)


def replay_termination(env, base_func, base_params):
    return base_func(env, **base_params) & ~command(env).online


def velocity_tracking(env):
    c = command(env)
    v = c.robot.data.root_link_lin_vel_w
    error = (v[:, 0] - c.speed).square() + v[:, 1].square()
    upright = (-c.robot.data.projected_gravity_b[:, 2]).clamp(0, 1)
    return torch.exp(-error / 0.25**2) * upright * c.online


def heading_tracking(env):
    c = command(env)
    q = yaw_quat(c.robot.data.root_link_quat_w)
    return torch.exp(-2 * q[:, 3].square() / 0.25) * c.online


def upright_reward(env):
    c = command(env)
    return (-c.robot.data.projected_gravity_b[:, 2]).clamp(0, 1).square() * c.online


def upper_body_style(env):
    c = command(env)
    # Waist and arms preserve full-body style; legs are free to adapt step height.
    error = (c.robot_joint_pos[:, 12:] - c.joint_pos[:, 12:]).square().mean(-1)
    return torch.exp(-error / 0.3**2) * c.online


def terrain_foot_target(env):
    c = command(env)
    foot_indices = [3, 6]
    ref = c.motion.body_pos_w[c.time_steps]
    root = c.robot.data.root_link_pos_w
    # BRACE-inspired reward target only: lift nominal flat-ground foot clearance
    # onto the surface at the nominal foothold. Never sent to the SONIC encoder.
    target_xy = root[:, None, :2] + ref[:, foot_indices, :2] - ref[:, :1, :2]
    ref_ground = ref[:, foot_indices, 2].min(dim=1).values
    clearance = (ref[:, foot_indices, 2] - ref_ground[:, None]).clamp(0, 0.35)
    desired_z = c.ground_height(target_xy) + clearance
    feet = c.robot_body_pos_w[:, foot_indices, 2]
    return torch.exp(-(feet - desired_z).square().mean(-1) / 0.10**2) * c.online


def foot_slip(env):
    c = command(env)
    pos = c.robot_body_pos_w[:, [3, 6]]
    near_ground = pos[:, :, 2] - c.ground_height(pos[:, :, :2]) < 0.07
    velocity = c.robot_body_lin_vel_w[:, [3, 6], :2].square().sum(-1)
    return (velocity * near_ground).sum(-1) * c.online


def online_fall(env):
    c = command(env)
    root = c.robot.data.root_link_pos_w
    clearance = root[:, 2] - c.ground_height(root[:, :2])
    lateral = (root[:, 1] - c.tile_origins[c.tile, 1]).abs()
    result = c.online & (
        (clearance < 0.32)
        | (c.robot.data.projected_gravity_b[:, 2] > -0.45)
        | (lateral > 1.5)
        | ~torch.isfinite(root).all(-1)
    )
    c.terminal_fall = result.clone()
    return result


def online_success(env):
    c = command(env)
    step = int(env.common_step_counter)
    if step != c.success_step:
        root = c.robot.data.root_link_pos_w
        x = root[:, 0] - c.tile_origins[c.tile, 0]
        feet = c.robot_body_pos_w[:, [3, 6]]
        supported = (feet[:, :, 2] - c.ground_height(feet[:, :, :2])).abs().max(-1).values < 0.18
        passed = (x >= c.tile_data["goal"][c.tile]) & (c.speed > 0.1) & supported
        passed &= c.robot.data.projected_gravity_b[:, 2] < -0.8
        c.success_hold[:] = torch.where(passed & c.online, c.success_hold + 1, 0)
        c.success_step = step
    result = (c.success_hold >= 10) & c.online
    c.terminal_success = result.clone()
    return result


def online_progress(env):
    c = command(env)
    # Signed bounded forward velocity avoids rewarding vertical jumping.
    return (
        (c.robot.data.root_link_lin_vel_w[:, 0] / c.speed.clamp(min=0.2)).clamp(-1, 1.5)
        * (c.speed > 0.1)
        * c.online
    )


def falling_cost(env):
    return online_fall(env).float()
