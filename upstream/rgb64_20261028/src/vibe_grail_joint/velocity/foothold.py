"""Reward-only G1 sole geometry. No terrain information enters the actor.

The seven 1 cm-radius foot capsules match mjlab's G1 collision asset. Sample
their centerlines and lower hemisphere, transformed by the actual foot pose.
Costs are bounded and averaged over both feet, then gated to online PPO only.
"""

import math
import torch
from mjlab.utils.lab_api.math import quat_apply


def sole_points(position, quaternion):
    # Foot 1 differs by 1 mm across sides; the conservative shared envelope is
    # below simulation contact tolerance. All other sampled centers match XML.
    ends = position.new_tensor([
        [[.10, -.026, -.025], [.05, -.027, -.025]],
        [[-.044, -.018, -.025], [.123, -.018, -.025]],
        [[-.052, -.010, -.025], [.130, -.010, -.025]],
        [[-.054, 0., -.025], [.132, 0., -.025]],
        [[-.052, .010, -.025], [.130, .010, -.025]],
        [[-.044, .018, -.025], [.123, .018, -.025]],
        [[.10, .026, -.025], [.05, .026, -.025]],
    ])
    t = position.new_tensor([0., .5, 1.])
    centers = (ends[:, :1] * (1-t)[None, :, None] + ends[:, 1:] * t[None, :, None]).reshape(-1, 3)
    centers = centers.expand(*position.shape[:-1], -1, 3)
    q = quaternion.unsqueeze(-2).expand(*centers.shape[:-1], 4)
    centers = quat_apply(q.reshape(-1, 4), centers.reshape(-1, 3)).reshape_as(centers)
    centers = centers + position.unsqueeze(-2)
    r = .01 / math.sqrt(2)
    offsets = position.new_tensor([[0, 0, -.01], [r, 0, -r], [-r, 0, -r], [0, r, -r], [0, -r, -r]])
    return (centers.unsqueeze(-2) + offsets).flatten(-3, -2)


def support_deficit(points, heights, minimum_fraction=.7, tolerance=.025):
    """Soft fraction of sole samples near terrain; tolerates toe/heel rolling."""
    gap = (points[..., 2] - heights).abs()
    supported = (1 - (gap - tolerance).clamp(min=0) / tolerance).clamp(0, 1).mean(-1)
    return ((minimum_fraction - supported) / minimum_fraction).clamp(0, 1)


def _measure(env, settings):
    c = env.command_manager.get_term("motion")
    step = int(env.common_step_counter)
    if getattr(c, "_foothold_step", -1) == step:
        return c._foothold_measurements
    positions = c.robot_body_pos_w[:, [3, 6]]
    points = sole_points(positions, c.robot_body_quat_w[:, [3, 6]])
    heights = c.ground_height(points[..., :2])
    ref = c.motion.body_pos_w[c.time_steps][:, [3, 6], 2]
    ref_clearance = ref - ref.min(-1, keepdim=True).values
    swing = ((ref_clearance - .015) / .055).clamp(0, 1)
    ref_vz = c.motion.body_lin_vel_w[c.time_steps][:, [3, 6], 2]
    stance = (1-swing) * torch.exp(-(ref_vz / .30).square())
    contact, side_current, side_history = [], [], []
    for side in ("left", "right"):
        sensor = env.scene[f"foothold_{side}"].data
        contact.append(((sensor.found > 0) & (sensor.normal[..., 2].abs() > .7)
                        & (sensor.force[..., 2].abs() > 5.)).any(-1))
        side_current.append(torch.where(
            (sensor.found > 0) & (sensor.normal[..., 2].abs() < .5),
            sensor.force.norm(dim=-1), 0.).amax(-1))
        # History covers all simulation substeps. A horizontal-force excess
        # only counts near a physical riser, avoiding ordinary flat friction.
        force = sensor.force_history
        side_history.append((force[..., :2].norm(dim=-1) - .8*force[..., 2].abs() - 8.)
                            .clamp(min=0).flatten(1).amax(-1))
    contact = torch.stack(contact, -1)
    shift = points.new_tensor([settings.get("lookahead_m", .03), 0.])
    ahead = c.ground_height(points[..., :2] + shift)
    margin = settings.get("clearance_m", .025)
    obstacle = torch.maximum(heights, ahead)
    clearance = ((obstacle + margin - points[..., 2]).clamp(min=0) / margin).clamp(0, 1)
    clearance = clearance.amax(-1) * swing * (~contact)
    # Support is evaluated at low reference swing and real support contact.
    # Front margin checks the next riser; rear/side overhang uses all samples.
    support = support_deficit(points, heights, settings.get("minimum_support_fraction", .7))
    toe_heights = c.ground_height(points[..., :2] + points.new_tensor([.015, 0.]))
    toe_intrusion = ((toe_heights - points[..., 2] - .01) / .03).clamp(0, 1).amax(-1)
    support = torch.maximum(support, toe_intrusion) * stance * contact
    # Exact profile edges and heights bound the history-based collision test.
    x = points[..., 0] - c.tile_origins[c.tile, 0][:, None, None]
    edges = c._profile_edges[c.tile][:, None, None, :]
    near_edge = ((x[..., None] - edges).abs() < .04) & (c._profile_jumps[c.tile][:, None, None, :].abs() > .005)
    near_edge = near_edge.any(-1).any(-1)
    low_near_edge = (points[..., 2] < obstacle + .04).any(-1)
    force = torch.maximum(torch.stack(side_current, -1),
                          torch.stack(side_history, -1) * near_edge * low_near_edge)
    collision = (force / settings.get("collision_force_n", 80.)).clamp(0, 1)
    warmup = min(1., (int(getattr(env, "policy_update_count", 0)) + 1)
                 / max(1, settings.get("warmup_updates", 200)))
    gate = c.ppo_online.float() * warmup
    result = {"clearance": clearance.mean(-1)*gate,
              "collision": collision.mean(-1)*gate, "support": support.mean(-1)*gate}
    c._foothold_step, c._foothold_measurements = step, result
    return result


def swing_clearance_cost(env, settings):
    return _measure(env, settings)["clearance"]


def riser_collision_cost(env, settings):
    return _measure(env, settings)["collision"]


def landing_support_cost(env, settings):
    return _measure(env, settings)["support"]
