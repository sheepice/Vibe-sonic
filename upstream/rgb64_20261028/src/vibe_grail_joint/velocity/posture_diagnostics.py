"""Read-only, per-environment posture and speed diagnostics for G1 tracking.

Quaternions are scalar-first (w, x, y, z). Body orientations come from the
command's *tracked body* order, not the robot's complete body order. Reference
orientations are ``command.body_quat_w`` in world coordinates; the moving planner
buffer's translation/recentering never enters the displacement calculation.

``PostureDiagnostics(command).sample()`` returns detached tensors on the same
device, without reducing across environments or copying to the CPU. Sample at
reset and then after every control step, before any auto-reset, for evaluation.
For training, pass ``reset_mask`` in the selected-environment order immediately
after auto-reset; terminal posture cannot be recovered from an already reset
state. Filter samples after each environment's terminal event externally.

The default command heading is world +X, matching VelocityMotionCommand. A
benchmark using another heading must supply ``heading_rad`` and, if it has no
``command.speed``, an explicit ``command_speed`` on each sample. Commanded
distance integrates the *previous sample's* speed over the elapsed interval.
Reference motion distance and nominal planner speed are deliberately not used
as the commanded distance: these are different objectives in the online task.

Terrain stage is a geometric root-X segment label, not a contact/stance label:
-1 unknown/replay, 0 flat or landing, 1 ascending flight, 2 descending flight,
3 outside the 4 m wide generated tile. It is never an actor observation.
"""

from __future__ import annotations

import math
from typing import Any

import torch


TERRAIN_STAGE_NAMES = {-1: "unknown_or_replay", 0: "flat_or_landing", 1: "up", 2: "down", 3: "outside_tile"}


def _unit_quaternion(q: torch.Tensor) -> torch.Tensor:
    # An invalid all-zero quaternion stays invalid, rather than appearing upright.
    norm = q.norm(dim=-1, keepdim=True)
    return q / torch.where(norm > 1e-12, norm, torch.full_like(norm, float("nan")))


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product, broadcasting leading dimensions, wxyz convention."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                        aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def orientation_error_deg(actual: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Shortest SO(3) angle, invariant to quaternion sign, range [0, 180]."""
    actual, reference = _unit_quaternion(actual), _unit_quaternion(reference)
    inverse = reference.clone()
    inverse[..., 1:] *= -1
    delta = quaternion_multiply(inverse, actual)
    # atan2 is more accurate near zero than acos(dot(q1,q2)).
    return torch.rad2deg(2 * torch.atan2(delta[..., 1:].norm(dim=-1), delta[..., 0].abs()))


def posture_angles_deg(q: torch.Tensor) -> dict[str, torch.Tensor]:
    """World tilt plus ZYX pitch/roll/yaw; positive pitch means forward lean.

    Tilt is the angle of the body's local +Z axis to world +Z. Pitch and roll
    remove world yaw, so a common world-yaw rotation does not change them.
    Euler pitch/roll become ambiguous near a sideways/inverted fall; retain
    total tilt and SO(3) error to interpret such samples.
    """
    w, x, y, z = _unit_quaternion(q).unbind(-1)
    r00, r10 = 1 - 2 * (y*y + z*z), 2 * (x*y + w*z)
    r20, r21, r22 = 2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)
    return {
        "tilt_deg": torch.rad2deg(torch.acos(r22.clamp(-1, 1))),
        "pitch_deg": torch.rad2deg(torch.atan2(-r20, torch.sqrt(r00*r00 + r10*r10))),
        "roll_deg": torch.rad2deg(torch.atan2(r21, r22)),
        "yaw_deg": torch.rad2deg(torch.atan2(r10, r00)),
    }


def _yaw_quaternion(angle: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(angle)
    return torch.stack((torch.cos(angle / 2), zero, zero, torch.sin(angle / 2)), -1)


class PostureDiagnostics:
    """A small stateful recorder; no policy, environment, or reward mutation.

    ``env_ids`` fixes the selection and its ordering for this recorder's life.
    ``heading_rad`` can be a scalar or one heading for each selected environment.
    It is fixed for the recorded episode; construct a new recorder if it changes.
    All angle outputs use degrees; joint-velocity errors use radians/second.
    """

    def __init__(self, command: Any, env_ids=None, *, heading_rad=0.0):
        self.command = command
        device = command.robot.data.joint_pos.device
        total = command.robot.data.joint_pos.shape[0]
        self.env_ids = (torch.arange(total, device=device) if env_ids is None
                        else torch.as_tensor(env_ids, dtype=torch.long, device=device))
        if self.env_ids.ndim != 1:
            raise ValueError("env_ids must be a one-dimensional list of environment indices")
        names = list(command.robot.joint_names)
        bodies = list(command.cfg.body_names)
        self.body_ids = {name: bodies.index(name) for name in ("pelvis", "torso_link")}
        self.joint_groups = {
            "waist": [i for i, name in enumerate(names) if name.startswith("waist_")],
            "left_arm": [i for i, name in enumerate(names) if name.startswith("left_")
                         and any(part in name for part in ("shoulder_", "elbow_", "wrist_"))],
            "right_arm": [i for i, name in enumerate(names) if name.startswith("right_")
                          and any(part in name for part in ("shoulder_", "elbow_", "wrist_"))],
        }
        if [len(self.joint_groups[k]) for k in ("waist", "left_arm", "right_arm")] != [3, 7, 7]:
            raise ValueError("Diagnostics expect G1 29-DOF joint names: 3 waist and 7 per arm")
        self.joint_groups["arms"] = self.joint_groups["left_arm"] + self.joint_groups["right_arm"]
        self.joint_names = {k: [names[i] for i in ids] for k, ids in self.joint_groups.items()}
        self.waist_names = [names[i] for i in self.joint_groups["waist"]]
        self._groups = {k: torch.tensor(v, device=device) for k, v in self.joint_groups.items()}
        heading = torch.as_tensor(heading_rad, dtype=torch.float32, device=device)
        self.heading = torch.broadcast_to(heading, (len(self.env_ids),)).clone()
        self.forward = torch.stack((self.heading.cos(), self.heading.sin()), -1)
        self.lateral = torch.stack((-self.heading.sin(), self.heading.cos()), -1)
        self._segments = self._terrain_segments(device)
        self._previous_root = None
        self._origin = None
        self._previous_speed = None
        self._elapsed = torch.zeros_like(self.heading)
        self._command_distance = torch.zeros_like(self.heading)
        self._path_length = torch.zeros_like(self.heading)

    def _terrain_segments(self, device):
        tiles = getattr(self.command.cfg, "random_tiles", ())
        segments = []
        for tile in tiles:
            up_start = up_end = down_start = down_end = float("inf")
            if "edges" in tile:
                positive = [i for i, jump in enumerate(tile["jumps"]) if jump > 0]
                negative = [i for i, jump in enumerate(tile["jumps"]) if jump < 0]
                if positive:
                    up_start = tile["edges"][positive[0]]
                    up_end = tile["edges"][positive[-1]] + tile["treads"][-1]
                if negative:
                    down_start = tile["edges"][negative[0]]
                    # Only an up_down tile reverses its tread order for descent;
                    # a down-only tile uses the original up_edges/treads array.
                    last_tread = tile["treads"][0 if tile["kind"] == "up_down" else -1]
                    down_end = tile["edges"][negative[-1]] + last_tread
            else:
                length = tile["n"] * tile["tread"]
                if tile["kind"] in ("up", "up_down"):
                    up_start, up_end = tile["start"], tile["start"] + length
                if tile["kind"] == "down":
                    down_start, down_end = tile["start"], tile["start"] + length
                elif tile["kind"] == "up_down":
                    down_start = tile["start"] + length + tile["plateau"]
                    down_end = down_start + length
            segments.append([up_start, up_end, down_start, down_end])
        return torch.tensor(segments, device=device) if segments else None

    @torch.no_grad()
    def sample(self, *, elapsed_dt_s=None, reset_mask=None, command_speed=None,
               include_joint_velocity=True) -> dict[str, torch.Tensor]:
        """Observe current state, integrating from the preceding sample.

        The first sample initializes origins and ignores elapsed_dt_s. Later
        calls default to command._env.step_dt. For decimated sampling explicitly
        pass the full elapsed interval. ``reset_mask`` uses *selected* indices.
        ``command_speed`` is a scalar or one value per selected environment.
        For mixed training roles, inspect ``online`` before interpreting command
        speed: replay/flat ``c.speed`` is not their reference-velocity objective.
        """
        c, ids = self.command, self.env_ids
        root = c.robot.data.root_link_pos_w[ids].detach().clone()
        v = c.robot.data.root_link_lin_vel_w[ids]
        speed = (c.speed[ids] if command_speed is None else
                 torch.broadcast_to(torch.as_tensor(command_speed, device=root.device, dtype=root.dtype), (len(ids),)))
        reset = (torch.zeros(len(ids), dtype=torch.bool, device=root.device) if reset_mask is None
                 else torch.as_tensor(reset_mask, dtype=torch.bool, device=root.device))
        if reset.shape != (len(ids),):
            raise ValueError("reset_mask must match selected environment count")
        if self._previous_root is None:
            self._origin = root.clone()
        else:
            dt = float(c._env.step_dt if elapsed_dt_s is None else elapsed_dt_s)
            if not math.isfinite(dt) or dt < 0:
                raise ValueError("elapsed_dt_s must be finite and nonnegative")
            self._elapsed += dt
            self._command_distance += self._previous_speed * dt
            self._path_length += (root[:, :2] - self._previous_root[:, :2]).norm(dim=-1)
            self._elapsed[reset] = self._command_distance[reset] = self._path_length[reset] = 0
            self._origin[reset] = root[reset]
        self._previous_root, self._previous_speed = root.clone(), speed.detach().clone()
        displacement = root[:, :2] - self._origin[:, :2]
        actual_speed = (v[:, :2] * self.forward).sum(-1)
        lateral_speed = (v[:, :2] * self.lateral).sum(-1)
        result = {
            "elapsed_s": self._elapsed.clone(),
            "command_speed_m_s": speed.detach().clone(),
            "forward_speed_m_s": actual_speed,
            "lateral_speed_m_s": lateral_speed,
            "signed_speed_error_m_s": actual_speed - speed,
            "velocity_error_m_s": torch.sqrt((actual_speed-speed).square() + lateral_speed.square()),
            "forward_displacement_m": (displacement * self.forward).sum(-1),
            "lateral_displacement_m": (displacement * self.lateral).sum(-1),
            "commanded_distance_m": self._command_distance.clone(),
            "horizontal_path_length_m": self._path_length.clone(),
        }
        result["distance_error_m"] = result["forward_displacement_m"] - result["commanded_distance_m"]
        if hasattr(c, "online"):
            result["online"] = c.online[ids].clone()
        actual, reference = c.robot_body_quat_w[ids], c.body_quat_w[ids]
        pelvis = self.body_ids["pelvis"]
        yaw_actual = posture_angles_deg(actual[:, pelvis])["yaw_deg"]
        yaw_reference = posture_angles_deg(reference[:, pelvis])["yaw_deg"]
        align = _yaw_quaternion(torch.deg2rad(yaw_actual - yaw_reference))
        for label, body_name in (("pelvis", "pelvis"), ("torso", "torso_link")):
            index = self.body_ids[body_name]
            for suffix, value in posture_angles_deg(actual[:, index]).items():
                result[f"{label}_{suffix}"] = value
            for suffix, value in posture_angles_deg(reference[:, index]).items():
                result[f"reference_{label}_{suffix}"] = value
            result[f"{label}_reference_orientation_error_deg"] = orientation_error_deg(actual[:, index], reference[:, index])
            aligned_reference = quaternion_multiply(align, reference[:, index])
            result[f"{label}_heading_aligned_orientation_error_deg"] = orientation_error_deg(actual[:, index], aligned_reference)
        delta = c.robot_joint_pos[ids] - c.joint_pos[ids]
        velocity_delta = c.robot_joint_vel[ids] - c.joint_vel[ids] if include_joint_velocity else None
        for label, group in self._groups.items():
            error = delta[:, group]
            result[f"{label}_joint_rmse_deg"] = torch.rad2deg(error.square().mean(-1).sqrt())
            result[f"{label}_joint_max_error_deg"] = torch.rad2deg(error.abs().max(-1).values)
            if velocity_delta is not None:
                result[f"{label}_joint_velocity_rmse_rad_s"] = velocity_delta[:, group].square().mean(-1).sqrt()
        for index, name in zip(self.joint_groups["waist"], self.waist_names):
            result[f"{name}_signed_error_deg"] = torch.rad2deg(delta[:, index])
        stage = torch.full((len(ids),), -1, dtype=torch.long, device=root.device)
        if self._segments is not None and hasattr(c, "tile"):
            tile = c.tile[ids]
            local = root - c.tile_origins[tile]
            x = local[:, 0]
            a, b, d, e = self._segments[tile].unbind(-1)
            stage[:] = 0
            stage[(x >= a) & (x < b)] = 1
            stage[(x >= d) & (x < e)] = 2
            stage[(x < -2) | (x > 12) | (local[:, 1].abs() > 2)] = 3
            if hasattr(c, "online"):
                stage[~c.online[ids]] = -1
            result["tile"] = tile.clone()
        result["terrain_stage"] = stage
        return {key: value.detach() for key, value in result.items()}
