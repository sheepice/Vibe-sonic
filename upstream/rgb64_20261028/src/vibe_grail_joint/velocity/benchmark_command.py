"""Express world heading/velocity in the robot frame without using any terrain."""

import math
import torch
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat


def benchmark_velocity_observation(env, speed, heading_deg):
    c = env.command_manager.get_term('motion')
    q = yaw_quat(c.robot.data.root_link_quat_w)
    angle = math.radians(heading_deg)
    desired = torch.zeros(env.num_envs, 3, device=env.device)
    desired[:, 0], desired[:, 1] = speed * math.cos(angle), speed * math.sin(angle)
    local = quat_apply_inverse(q, desired)
    sin_yaw, cos_yaw = 2 * q[:, 0] * q[:, 3], 1 - 2 * q[:, 3].square()
    out = torch.zeros(env.num_envs, 8, device=env.device)
    out[:, :2] = local[:, :2]
    out[:, 3] = 1 - (math.cos(angle) * cos_yaw + math.sin(angle) * sin_yaw)
    out[:, 4] = math.sin(angle) * cos_yaw - math.cos(angle) * sin_yaw
    out[:, 5] = 1
    return out
