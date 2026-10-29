"""G1 keyboard deployment: inspect -> shadow -> explicitly enabled control.

W held: forward; release/Space: request stand; Enter: engage; Esc/focus loss:
latched damping. A physical support and robot-side stop remain necessary.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ..deploy.observations import ObservationBuilder
from ..deploy.safety import KeyboardIntent, Limits, health_fault, target_fault
from .safety import condition_target, validate_sim_acceptance


class KeyboardWindow:
    def __init__(self, speed):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((720, 260))
        pygame.display.set_caption("G1 ViBe | Enter engage | HOLD W | Space stand | Esc damping")
        self.font = pygame.font.Font(None, 28)
        self.intent = KeyboardIntent(speed)
        self.quit = False

    def poll(self, status):
        pg = self.pg
        arm = False
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit = True
            if event.type == pg.KEYDOWN and event.key == pg.K_RETURN:
                arm = True
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self.quit = True
        keys = pg.key.get_pressed()
        speed = self.intent.update(w=bool(keys[pg.K_w]), space=bool(keys[pg.K_SPACE]), arm=arm,
                                   escape=self.quit, focus=bool(pg.key.get_focused()))
        self.screen.fill((20, 24, 28))
        for i, text in enumerate([status, f"Forward command: {speed:.2f} m/s",
                                  "Enter: engage; HOLD W: forward; release/Space: stand",
                                  "Esc or lost focus: latched damping (requires support)",
                                  str(self.intent.fault or "")]):
            self.screen.blit(self.font.render(text, True, (230, 230, 230)), (15, 20 + i * 42))
        pg.display.flip()
        return speed

    def close(self):
        self.pg.quit()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--interface", required=True, help="Wired interface facing this G1")
    p.add_argument("--mode", choices=("inspect", "shadow", "control"), default="inspect")
    p.add_argument("--execute", action="store_true", help="Allow creation of an rt/lowcmd publisher in control mode")
    p.add_argument("--release-sport-mode", action="store_true", help="Explicitly release an active Unitree high-level controller on engage")
    p.add_argument("--calibration", help="Measured camera mount JSON; required in control mode")
    p.add_argument("--acceptance-report", help="Completed standing start/stop simcheck report for this exact bundle and actuator guards")
    p.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--serial")
    p.add_argument("--speed", type=float, default=.25)
    p.add_argument("--seconds", type=float, default=60.)
    p.add_argument("--log", required=True)
    args = p.parse_args()
    if args.mode == "control" and (not args.execute or not args.calibration or not args.acceptance_report):
        p.error("Control requires --execute, --calibration and --acceptance-report; inspect/shadow never publish commands")
    if args.mode != "control" and (args.execute or args.release_sport_mode):
        p.error("Control flags are forbidden in inspect/shadow")
    if not np.isfinite(args.seconds) or args.seconds <= 0 or not 0 < args.speed <= .7:
        p.error("Invalid duration or forward speed")
    from .runtime import StoneRuntime as Runtime
    from ..deploy.camera import RealSenseCamera, validate_mount
    from ..deploy.reference import PlannerWorker
    from ..deploy.unitree import UnitreeBackend

    root = Path(args.bundle)
    manifest = json.loads((root / "bundle.json").read_text())
    contract = manifest["robot"]
    if args.mode == "control":
        validate_sim_acceptance(args.acceptance_report, manifest, args.speed)
    if abs(contract["control_dt"] - .02) > 1e-8:
        raise ValueError("Only the trained 50 Hz controller is supported")
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError("Use a new log path for each trial")
    backend = camera = planner = window = None
    limits = Limits()
    with log_path.open("x", buffering=1) as log:
        def record(data):
            log.write(json.dumps({"monotonic": time.monotonic(), **data}) + "\n")
        try:
            backend = UnitreeBackend(args.interface, contract, execute=args.execute,
                                     release_sport_mode=args.release_sport_mode)
            record(dict(event="start", mode=args.mode, bundle=str(root.resolve()),
                        checkpoint_sha256=manifest["checkpoint_sha256"], interface=args.interface))
            deadline = time.monotonic() + 10
            while backend.state is None:
                if time.monotonic() > deadline:
                    raise RuntimeError("No CRC-valid G1 state after 10 seconds")
                time.sleep(.02)
            if args.mode == "inspect":
                deadline = time.monotonic() + args.seconds
                while time.monotonic() < deadline:
                    state = backend.latest()
                    row = dict(q=state.q.tolist(), dq=state.dq.tolist(), quat=state.quat.tolist(), gyro=state.gyro.tolist(),
                               tick=state.tick, state_age_s=time.monotonic()-state.received_at,
                               mode_machine=state.mode_machine, errors=state.motor_errors,
                               temperature=state.temperature, remote_select=state.remote_stop,
                               crc_failures=backend.crc_failures, lowcmd_published=False)
                    record(row)
                    print(json.dumps(row), flush=True)
                    time.sleep(.1)
                return
            runtime = Runtime(root, args.provider)
            record(dict(event="inference_warmup", elapsed_s=runtime.warmup()))
            camera = RealSenseCamera(args.serial, manifest["camera"]["fovy"])
            if args.mode == "control":
                validate_mount(args.calibration, camera, manifest["camera"])
            planner = PlannerWorker(root / "planner.onnx", contract)
            deadline = time.monotonic() + 10
            while camera.frame is None or planner.reference.updated_at is None:
                if camera.error or planner.error or time.monotonic() > deadline:
                    raise RuntimeError(f"Sensors/planner not ready: {camera.error}, {planner.error}")
                time.sleep(.02)
            window = KeyboardWindow(args.speed)
            builder = ObservationBuilder(contract)
            default = np.asarray(contract["default_joint_pos"], np.float32)
            previous = backend.latest().q.copy()
            prepared = args.mode == "shadow"
            preparation_start = None
            ramp_start = None
            start, next_tick = time.monotonic(), time.monotonic()
            last_camera = None
            features = None
            speed = 0.
            durations = []
            while time.monotonic() - start < args.seconds:
                begin = time.monotonic()
                status = "SHADOW: no motor commands" if args.mode == "shadow" else (
                    "CONTROL: ready" if prepared else "CONTROL: Enter engages a 3-second nominal-pose transition")
                requested = window.poll(status)
                command_at = time.monotonic()
                if window.intent.fault:
                    raise RuntimeError(window.intent.fault)
                state = backend.latest()
                bgr, image_at, frame_id = camera.latest()
                fault = health_fault(state, time.monotonic(), image_at, command_at, limits)
                if fault:
                    raise RuntimeError(fault)
                if backend.fault:
                    raise RuntimeError(backend.fault)
                if args.mode == "control" and window.intent.armed and not backend.armed:
                    if np.max(abs(state.q-default)) > .5 or np.max(abs(state.dq)) > .5:
                        raise RuntimeError("Start from a supported, stationary nominal standing pose; seated auto-stand is unsupported")
                    backend.arm()
                    preparation_start = time.monotonic()
                    ramp_start = state.q.copy()
                    previous = ramp_start.copy()
                speed += np.clip((requested if prepared else 0.) - speed, -.01, .01)
                planner.speed = speed
                if backend.armed and not prepared:
                    fraction = min(1., (time.monotonic() - preparation_start) / 3.)
                    target = ramp_start + (default-ramp_start) * fraction
                    backend.submit(target, image_at, command_at)
                    previous = target.copy()
                    if fraction == 1.:
                        builder.reset()
                        runtime.reset()
                        builder.align_heading(state.quat)
                        prepared = True
                        record(dict(event="nominal_pose_ready"))
                else:
                    if frame_id != last_camera:
                        features = runtime.encode(bgr)
                        last_camera = frame_id
                    ref = planner.sample(time.monotonic())
                    observations = builder.build(state, ref, *features, speed, velocity=runtime.velocity)
                    action, target, inference_s = runtime.act(observations)
                    target, projection, slew = condition_target(target, previous, contract, limits)
                    # Idle/shadow targets are logged, never published. Reset on physical engage.
                    fault = target_fault(target, previous, state, contract, limits)
                    if backend.armed and fault:
                        raise RuntimeError(fault)
                    if backend.armed:
                        backend.submit(target, image_at, command_at)
                    applied_action = (target-runtime.default)/runtime.scale
                    builder.accept_action(applied_action)
                    previous = target.copy()
                    record(dict(event="control_tick", speed_command=speed, action=action.tolist(),
                                q_target=target.tolist(), q=state.q.tolist(), quat=state.quat.tolist(),
                                policy_s=inference_s, image_age_s=time.monotonic()-image_at,
                                target_check=fault, limit_projection_rad=projection, slew_projection_rad=slew,
                                lowcmd_enabled=backend.armed, frame_id=frame_id,
                                planner_s=planner.reference.seconds))
                elapsed = time.monotonic() - begin
                durations.append(elapsed)
                if backend.armed and elapsed > .04:
                    raise RuntimeError("Policy loop exceeded 40 ms")
                next_tick += .02
                if next_tick < time.monotonic() - .02:
                    next_tick = time.monotonic()
                time.sleep(max(0., next_tick-time.monotonic()))
            record(dict(event="completed", loop_p50_s=float(np.percentile(durations, 50)),
                        loop_p99_s=float(np.percentile(durations, 99)), loop_max_s=max(durations)))
        except BaseException as exc:
            if backend:
                backend.trip(str(exc))
            record(dict(event="stopped", reason=str(exc), hardware_validated=False))
            raise
        finally:
            if backend:
                backend.close()
            if planner:
                planner.close()
            if camera:
                camera.close()
            if window:
                window.close()


if __name__ == "__main__":
    main()
