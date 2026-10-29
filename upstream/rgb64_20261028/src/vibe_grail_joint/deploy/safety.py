"""Pure, testable admission/health checks. These are not a hardware E-stop."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .observations import rotation


@dataclass
class Limits:
    state_age_s: float = .05
    image_age_s: float = .15
    command_age_s: float = .15
    target_age_s: float = .06
    max_tilt_rad: float = .8
    max_temperature_c: float = 75.
    max_target_step_rad: float = .35
    max_target_error_rad: float = 1.2
    max_limit_projection_rad: float = .25
    target_slew_rad_per_step: float = .20  # 10 rad/s at the required 50 Hz
    max_slew_projection_rad: float = .5


def health_fault(state, now, image_at, command_at, limits):
    if not np.isfinite([now, state.received_at, image_at, command_at]).all():
        return "invalid_timestamp"
    if now - state.received_at > limits.state_age_s or state.received_at > now + .005:
        return "stale_robot_state"
    if now - image_at > limits.image_age_s or image_at > now + .005:
        return "stale_camera"
    if now - command_at > limits.command_age_s or command_at > now + .005:
        return "stale_keyboard"
    if state.remote_stop:
        return "remote_emergency_stop"
    if any(state.motor_errors):
        return "motor_error"
    if not np.isfinite(state.temperature) or state.temperature > limits.max_temperature_c:
        return "motor_temperature"
    if any(np.shape(value) != shape or not np.isfinite(value).all() for value, shape in
           ((state.q, (29,)), (state.dq, (29,)), (state.gyro, (3,)), (state.quat, (4,)))):
        return "invalid_robot_state"
    try:
        up = rotation(state.quat).apply([0, 0, 1])[2]
    except ValueError:
        return "invalid_imu"
    if up < np.cos(limits.max_tilt_rad):
        return "excessive_tilt"
    return None


def target_fault(q, previous, state, contract, limits):
    q = np.asarray(q)
    if q.shape != (29,) or not np.isfinite(q).all():
        return "invalid_target"
    bounds = np.asarray(contract["joint_limits"])
    if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
        return "joint_limit"
    if np.max(abs(q - previous)) > limits.max_target_step_rad:
        return "target_discontinuity"
    if np.max(abs(q - state.q)) > limits.max_target_error_rad:
        return "target_tracking_error"
    return None


def project_target(q, contract, limits):
    """Bound small PD-target overshoots; reject large ones and report every change.

    Training does not clamp PD targets to joint limits. This explicit deploy
    projection changes that behavior and must be exercised in closed-loop tests.
    """
    q = np.asarray(q, dtype=np.float32)
    if q.shape != (29,) or not np.isfinite(q).all():
        raise ValueError("invalid_target")
    bounds = np.asarray(contract["joint_limits"], dtype=np.float32)
    projected = np.clip(q, bounds[:, 0], bounds[:, 1])
    difference = float(np.max(abs(projected-q)))
    if difference > limits.max_limit_projection_rad:
        raise ValueError(f"excessive_joint_limit_projection: {difference:.4f} rad")
    return projected, difference


def condition_target(q, previous, contract, limits):
    bounded, projection = project_target(q, contract, limits)
    previous = np.asarray(previous, dtype=np.float32)
    if previous.shape != (29,) or not np.isfinite(previous).all():
        raise ValueError("invalid_previous_target")
    limited = np.clip(bounded, previous-limits.target_slew_rad_per_step,
                       previous+limits.target_slew_rad_per_step)
    slew = float(abs(limited-bounded).max())
    if slew > limits.max_slew_projection_rad:
        raise ValueError(f"excessive_target_slew: {slew:.4f} rad")
    return limited, projection, slew


class KeyboardIntent:
    """Explicit arm, dead-man forward, release-to-zero, latched emergency stop."""
    def __init__(self, speed=.25):
        if not 0 < speed <= .7:
            raise ValueError("Speed must lie in (0, .7]")
        self.speed = speed
        self.armed = False
        self.fault = None

    def update(self, *, w=False, arm=False, space=False, escape=False, focus=True):
        if escape:
            self.fault = "keyboard_emergency_stop"
        if not focus:
            self.fault = "keyboard_focus_lost"
        if self.fault:
            self.armed = False
            return 0.
        if arm and not w:
            self.armed = True
        return self.speed if self.armed and w and not space else 0.


def acceptance_signature(manifest):
    """Bind an admission result to the control contract as well as model bytes."""
    data = dict(version=1, files=manifest["files"], robot=manifest["robot"],
                camera=manifest["camera"], kind=manifest["kind"], limits=asdict(Limits()))
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_sim_acceptance(path, manifest, speed=None):
    """Require evidence for the SAME model/guards before the control entrypoint.

    This is an engineering admission check, not proof of hardware safety.
    """
    report = json.loads(Path(path).read_text())
    if report.get("bundle_files") != manifest["files"]:
        raise ValueError("Acceptance report belongs to different model/encoder/planner bytes")
    if report.get("acceptance_signature") != acceptance_signature(manifest):
        raise ValueError("Acceptance report is missing or has a different robot/camera/guard contract")
    if (report.get("mode") != "keyboard" or report.get("initial_state") != "stationary_default_pose"
            or report.get("no_physics_guards_applied", True) or report.get("fall", True)
            or not report.get("deploy_onnx_drove_physics") or not report.get("raw_rgb_encoder_used")
            or "guard_stop_reason" in report or report.get("target_guard_counts")
            or report.get("simulated_seconds", 0) < 12
            or report["simulated_seconds"] + .021 < report.get("requested_seconds", 0)):
        raise ValueError("Standing start/stop with deployment actuator guards has not passed")
    trace = report.get("trace", [])
    if len(trace) < 101:
        raise ValueError("Acceptance report does not include a completed stop interval")
    positions = np.asarray([x["position"] for x in trace], dtype=float)
    commands = np.asarray([x["speed_command"] for x in trace], dtype=float)
    times = np.asarray([x["t"] for x in trace], dtype=float)
    if (positions.shape != (len(trace), 3) or not np.isfinite(positions).all()
            or not np.isfinite(commands).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or times[-1] < 11.9):
        raise ValueError("Acceptance trace is invalid or incomplete")
    if np.any(abs(commands[-51:]) > 1e-6) or times[-1]-times[-51] < .99:
        raise ValueError("Acceptance report does not include a completed stop interval")
    if speed is not None and (not np.isfinite(speed) or speed <= 0 or speed > commands.max() + 1e-6):
        raise ValueError("Requested speed exceeds the speed checked in the acceptance report")
    moved = positions[-1, 0]-positions[0, 0]
    end_delta = positions[-1, :2]-positions[-51, :2]
    if moved < .5 or np.linalg.norm(end_delta) > .12:
        raise ValueError("Acceptance report does not establish forward motion followed by a stop")
    return report
