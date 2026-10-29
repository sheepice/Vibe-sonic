"""G1 remote/keyboard deployment: inspect -> shadow -> explicitly enabled control.

W held: forward; release/Space: request stand; Enter: initialize; P: start policy; X/SELECT/focus loss:
latched damping; Esc: exit. PD target guards are opt-in. Software damping
requires a functioning process/connection and physical support.
Remote: R1+START hold to initialize, A hold to start policy, arrows move/turn, B stand, SELECT damping,
Y hold after a latched stop exits. Remote mode needs no graphical window.
"""
import argparse
import json
from pathlib import Path
import time
import sys

import numpy as np

from .safety import KeyboardIntent, Limits, health_fault, prepare_target, validate_sim_acceptance
from .initialization import NominalPoseTransition, pose_report


class KeyboardWindow:
    def __init__(self, speed, on_stop):
        import pygame
        self.pg = pygame
        pygame.init()
        self.screen = pygame.display.set_mode((900, 330))
        pygame.display.set_caption("G1 ViBe | X: LATCH DAMPING | Esc: exit")
        self.font = pygame.font.Font(None, 28)
        self.intent = KeyboardIntent(speed)
        self.on_stop = on_stop
        self.stop_button = pygame.Rect(15, 260, 870, 55)
        self.quit = False

    def poll(self, status):
        pg = self.pg
        arm = False
        start_policy = False
        stop = False
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit = True
            if event.type == pg.KEYDOWN and event.key == pg.K_RETURN:
                arm = True
            if event.type == pg.KEYDOWN and event.key == pg.K_p:
                start_policy = True
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self.quit = True
            if event.type == pg.KEYDOWN and event.key == pg.K_x:
                stop = True
            if (event.type == pg.MOUSEBUTTONDOWN and event.button == 1
                    and self.stop_button.collidepoint(event.pos)):
                stop = True
        keys = pg.key.get_pressed()
        speed = self.intent.update(w=bool(keys[pg.K_w]), space=bool(keys[pg.K_SPACE]), arm=arm,
                                   start_policy=start_policy,
                                   stop=stop or bool(keys[pg.K_x]),
                                   escape=self.quit, focus=bool(pg.key.get_focused()))
        # Latch the writer before drawing or running another inference.
        if self.intent.fault:
            self.on_stop(self.intent.fault)
        self.screen.fill((20, 24, 28))
        controls = ("STOP LATCHED: W and Enter disabled; support robot, Esc to exit"
                    if self.intent.fault else "Enter: initialize; P: start policy; W: forward; Space: stand")
        for i, text in enumerate([status, f"Forward command: {speed:.2f} m/s",
                                  controls,
                                  "X / red button / SELECT: damping; Esc: exit (requires support)",
                                  str(self.intent.fault or "")]):
            self.screen.blit(self.font.render(text, True, (230, 230, 230)), (15, 20 + i * 42))
        pg.draw.rect(self.screen, (190, 35, 35), self.stop_button, border_radius=6)
        self.screen.blit(self.font.render("X  STOP / LATCH DAMPING - no automatic resume", True,
                                         (255, 255, 255)), (35, 277))
        pg.display.flip()
        return speed

    def close(self):
        self.pg.quit()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--interface", required=True, help="Wired interface facing this G1")
    p.add_argument("--mode", choices=("inspect", "shadow", "control"), default="inspect")
    p.add_argument("--execute", action="store_true", help="Allow creation of an rt/lowcmd publisher in control mode")
    p.add_argument("--release-sport-mode", action="store_true", help="Explicitly release an active Unitree high-level controller on engage")
    camera_mount = p.add_mutually_exclusive_group()
    camera_mount.add_argument("--calibration", help="Measured camera mount JSON for control mode")
    camera_mount.add_argument("--allow-unmeasured-camera", action="store_true",
                             help="Explicitly run control without mount calibration; recorded as UNMEASURED")
    p.add_argument("--actuator-guards", action="store_true",
                   help="Opt in to PD target limits/slew/tracking stops; default is unchanged policy targets")
    p.add_argument("--acceptance-report", help="Required for control with --actuator-guards; not applicable when those guards are off")
    p.add_argument("--provider", choices=("cpu", "cuda", "tensorrt"), default="cpu")
    p.add_argument("--engine-dir", help="Validated TensorRT cache directory; required with --provider tensorrt")
    p.add_argument("--serial")
    p.add_argument("--input",choices=("keyboard","remote"),default="keyboard")
    p.add_argument("--inference-cpus",help="CPU list for the inference child; use the tested allocation in G1_RUN_COMMANDS_ZH.md")
    p.add_argument("--backward-speed",type=float,default=.1)
    p.add_argument("--turn-rate",type=float,default=.45,help="Remote turn rate in rad/s")
    p.add_argument("--speed", type=float, default=.4)
    p.add_argument("--seconds", type=float, default=60.)
    p.add_argument("--log", required=True)
    p.add_argument('--record-bag', help='New ROS 2 bag directory; optional file-only background recording')
    p.add_argument('--record-raw-hz', type=float, default=10., help='Raw RGB recording rate, up to camera 30 Hz')
    p.add_argument('--record-max-mib', type=float, default=4096., help='Stop recording at this size; does not stop control')
    p.add_argument('--record-cpus', default='0,1', help='Low-priority recorder CPUs on this G1')
    args = p.parse_args(argv)
    if (args.provider == "tensorrt") != (args.engine_dir is not None):
        p.error("Use --engine-dir exactly when --provider tensorrt is selected")
    if args.mode == "control" and (not args.execute or not (args.calibration or args.allow_unmeasured_camera)):
        p.error("Control requires --execute and either --calibration or explicit --allow-unmeasured-camera")
    if args.mode == "control" and args.actuator_guards and not args.acceptance_report:
        p.error("Control with --actuator-guards requires --acceptance-report")
    if args.acceptance_report and (args.mode != "control" or not args.actuator_guards):
        p.error("--acceptance-report applies only to control with --actuator-guards")
    if args.mode != "control" and (args.execute or args.release_sport_mode or args.allow_unmeasured_camera):
        p.error("Control flags are forbidden in inspect/shadow")
    if not np.isfinite(args.seconds) or args.seconds <= 0 or not 0 < args.speed <= .7:
        p.error("Invalid duration or forward speed")
    if not 0<args.backward_speed<=.3 or not 0<args.turn_rate<=.5:
        p.error("Invalid backward speed or turn rate")
    if not 0<args.record_raw_hz<=30 or not np.isfinite(args.record_max_mib) or args.record_max_mib<=0:
        p.error('Invalid recording rate or size limit')
    if args.record_bag and args.mode=='inspect':
        p.error('--record-bag requires shadow or control, which have a camera/inference pipeline')
    return args


def hold_damping(backend, window, record):
    """Keep the writer alive after a latched stop; never call policy/arm here."""
    if not backend.armed or window is None or window.quit:
        return
    window.intent.fault = window.intent.fault or backend.fault
    window.intent.armed = False
    window.intent.policy_enabled = False
    record(dict(event="damping_hold", reason=backend.fault, hardware_validated=False))
    print(f"DAMPING LATCHED: {backend.fault}. Support robot before exit.",flush=True)
    try:
        while not window.quit:
            window.poll(f"DAMPING LATCHED: {backend.fault} | support robot, Esc to exit")
            time.sleep(.02)
    except KeyboardInterrupt:
        pass
    record(dict(event="damping_hold_exit", reason=backend.fault))


def main():
    args = parse_args()
    # The default 5 ms GIL quantum exceeds two 500 Hz writer periods.
    # Shorter handoffs improve DDS callback/writer fairness without changing dt.
    previous_switch_interval = sys.getswitchinterval()
    from .inference_process import InferenceProcess
    from .unitree import UnitreeBackend

    root = Path(args.bundle)
    manifest = json.loads((root / "bundle.json").read_text())
    contract = manifest["robot"]
    if args.mode == "control" and args.actuator_guards:
        validate_sim_acceptance(args.acceptance_report, manifest, args.speed)
    if abs(contract["control_dt"] - .02) > 1e-8:
        raise ValueError("Only the trained 50 Hz controller is supported")
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        raise FileExistsError("Use a new log path for each trial")
    backend = worker = window = recorder = None
    limits = Limits()
    with log_path.open("x", buffering=1) as log:
        def record(data):
            row = {"monotonic": time.monotonic(), **data}
            log.write(json.dumps(row) + "\n")
            if recorder:
                recorder.event(row)
        def check_pose(state, stage):
            report = pose_report(state, contract, initializing=stage != 'policy_start')
            record(dict(event='pose_check',stage=stage,lowcmd_enabled=backend.armed,**report))
        try:
            sys.setswitchinterval(.001)
            if args.record_bag:
                from .recording import BagRecorder
                recorder = BagRecorder(args.record_bag,manifest,raw_hz=args.record_raw_hz,
                    max_mib=args.record_max_mib,cpus=args.record_cpus)
                print(f'BAG RECORDING: {recorder.path}; raw={args.record_raw_hz:g} Hz; '
                      'policy images/features/attention retain source timestamps.',flush=True)
            backend = UnitreeBackend(args.interface, contract, execute=args.execute,
                                     release_sport_mode=args.release_sport_mode)
            record(dict(event="start", mode=args.mode, bundle=str(root.resolve()),
                        checkpoint_sha256=manifest["checkpoint_sha256"], interface=args.interface,
                        actuator_guards=args.actuator_guards, hardware_validated=False,
                        acceptance_report=args.acceptance_report,
                        input_source=args.input,
                        recording_path=str(recorder.path) if recorder else None,
                        forward_speed=args.speed, backward_speed=args.backward_speed, turn_rate=args.turn_rate,
                        pose_and_tilt_stops=False, scheduling_deadlines_diagnostic_only=True,
                        freshness_limits_s=dict(state=limits.state_age_s, image=limits.image_age_s,
                                                command=limits.command_age_s, target=limits.target_age_s),
                        camera_mount_unmeasured_override=args.allow_unmeasured_camera,
                        camera_calibration_file=args.calibration,
                        manual_stop="X / red button / remote SELECT"))
            if args.allow_unmeasured_camera:
                print("UNMEASURED CAMERA: mount validation explicitly skipped; "
                      "physical camera pose relative to training is unknown. Supported flat-ground trial only.",flush=True)
            deadline = time.monotonic() + 10
            while backend.state is None:
                if time.monotonic() > deadline:
                    raise RuntimeError("No CRC-valid G1 state after 10 seconds")
                time.sleep(.02)
            if args.mode == "inspect":
                from .remote import button_names
                deadline = time.monotonic() + args.seconds
                while time.monotonic() < deadline:
                    state = backend.latest()
                    row = dict(q=state.q.tolist(), dq=state.dq.tolist(), quat=state.quat.tolist(), gyro=state.gyro.tolist(),
                               tick=state.tick, state_age_s=time.monotonic()-state.received_at,
                               mode_machine=state.mode_machine, errors=state.motor_errors,
                               temperature=state.temperature, remote_select=state.remote_stop,
                               remote_buttons=state.remote_buttons,
                               remote_buttons_hex=f"0x{state.remote_buttons:04x}",
                               remote_button_names=button_names(state.remote_buttons),
                               stop_latched=backend.fault is not None, stop_reason=backend.fault,
                               crc_failures=backend.crc_failures, lowcmd_published=False)
                    record(row)
                    print(json.dumps(row), flush=True)
                    time.sleep(.1)
                return
            worker = InferenceProcess(dict(bundle=str(root.resolve()),provider=args.provider,
                engine_dir=str(Path(args.engine_dir).resolve()) if args.engine_dir else None,
                serial=args.serial,calibration=args.calibration if args.mode == "control" else None,
                planner_threads=3,cpus=args.inference_cpus,
                recording=recorder.transport if recorder else None))
            record(dict(event="inference_ready",**worker.metadata))
            if args.input == "remote":
                from .remote import RemoteController
                window = RemoteController(backend,args.speed,args.backward_speed,args.turn_rate)
                print(f"REMOTE READY ({args.mode}): release all buttons, hold R1+START 1 s to initialize; "
                      "wait for pose ready, release all buttons, hold A 1 s to start policy, release; "
                      "arrows move/turn; B stand; SELECT latch damping; hold Y 2 s after stop to exit.",flush=True)
            else:
                window = KeyboardWindow(args.speed, on_stop=backend.trip)
            default = np.asarray(contract["default_joint_pos"], np.float32)
            scale = np.asarray(contract["scale"], np.float32)
            previous = backend.latest().q.copy()
            previous_action = reset_heading = None
            prepared = policy_started = False
            preparation_start = None
            transition = None
            start, next_tick = time.monotonic(), time.monotonic()
            speed = 0.
            command_heading = yaw_rate = 0.
            durations = []
            recording_active = True
            next_recording_status = 0.
            while time.monotonic() - start < args.seconds:
                begin = time.monotonic()
                if recorder and begin >= next_recording_status:
                    next_recording_status = begin+1.
                    status = recorder.status()
                    record(dict(event='recording_status',**status))
                    if recording_active and not status['active']:
                        print('BAG RECORDING ENDED: inspect recording_status.json; control state is unchanged.',flush=True)
                    recording_active = status['active']
                stage = ("policy active" if policy_started else
                         "POSE HOLD: support required; A hold / P starts policy" if prepared else
                         "R1+START hold / Enter initializes nominal pose")
                status = ("SHADOW (no motor commands): " if args.mode == "shadow" else "CONTROL: ") + stage
                status += " | target guards " + ("ON" if args.actuator_guards else "OFF")
                requested = window.poll(status)
                command_at = time.monotonic()
                if window.intent.fault:
                    raise RuntimeError(window.intent.fault)
                if backend.fault:
                    raise RuntimeError(backend.fault)
                state = backend.latest()
                if args.mode == "shadow" and window.intent.armed and not prepared:
                    prepared = True
                    window.intent.set_pose_ready()
                    record(dict(event="shadow_pose_ready",lowcmd_enabled=False))
                    print("SHADOW initialized (no motor commands): release all buttons, "
                          "hold A 1 s / press P to start shadow policy.",flush=True)
                if prepared and window.intent.policy_enabled and not policy_started:
                    if backend.armed:
                        check_pose(state,'policy_start')
                    policy_started = True
                    reset_heading = state.quat.copy()
                    previous_action = None
                    command_heading = yaw_rate = speed = 0.
                    record(dict(event="policy_started",lowcmd_enabled=backend.armed))
                    print("Policy started at zero velocity; release start button before movement.",flush=True)
                speed += np.clip((requested if policy_started else 0.) - speed, -.01, .01)
                requested_yaw = getattr(window.intent,'yaw_rate',0.) if policy_started else 0.
                yaw_rate += float(np.clip(requested_yaw-yaw_rate,-.02,.02))
                command_heading += yaw_rate*.02
                inference_start = time.monotonic()
                compute = policy_started
                movement = dict(command_heading=command_heading,yaw_rate=yaw_rate) if args.input=="remote" else {}
                result = worker.step(state,speed,previous_action=previous_action,
                    reset=reset_heading is not None,heading=reset_heading,
                    compute=compute,**movement)
                inference_roundtrip_s = time.monotonic()-inference_start
                if compute:
                    reset_heading = None
                image_at,frame_id = result['image_at'],result['frame_id']
                fault = health_fault(state, time.monotonic(), image_at, command_at, limits)
                if fault or backend.fault:
                    record(dict(event='health_fault', reason=fault or backend.fault,
                                frame_id=frame_id, image_age_s=time.monotonic()-image_at,
                                state_age_s=time.monotonic()-state.received_at,
                                policy_enabled=policy_started, lowcmd_enabled=backend.armed))
                    raise RuntimeError(fault or backend.fault)
                if args.mode == "control" and window.intent.armed and not backend.armed:
                    check_pose(state,'initialization')
                    backend.arm()
                    # DDS publisher initialization is a blocking handoff. Start
                    # the 50 Hz clock after it, and refresh all source timestamps
                    # before the first transition target can be submitted.
                    begin = next_tick = preparation_start = time.monotonic()
                    window.poll(status)
                    command_at = time.monotonic()
                    if window.intent.fault or backend.fault:
                        raise RuntimeError(window.intent.fault or backend.fault)
                    state = backend.latest()
                    result = worker.step(state,0.,compute=False)
                    image_at,frame_id = result['image_at'],result['frame_id']
                    fault = health_fault(state,time.monotonic(),image_at,command_at,limits)
                    if fault or backend.fault:
                        record(dict(event='health_fault', reason=fault or backend.fault,
                                    frame_id=frame_id, image_age_s=time.monotonic()-image_at,
                                    state_age_s=time.monotonic()-state.received_at,
                                    policy_enabled=policy_started, lowcmd_enabled=backend.armed))
                        raise RuntimeError(fault or backend.fault)
                    check_pose(state,'handoff')
                    transition = NominalPoseTransition(state.q,default)
                    preparation_start = time.monotonic()
                    previous = state.q.copy()
                    previous_action = None
                    record(dict(event='nominal_pose_transition',duration_s=transition.duration,
                                max_target_speed_rad_s=transition.max_target_speed_rad_s,
                                q_start=state.q.tolist(),q_target=default.tolist(),lowcmd_enabled=True))
                    print(f"Initializing nominal pose over {transition.duration:.1f} s; "
                          "keep physical support, policy is disabled.",flush=True)
                if backend.armed and not prepared:
                    target, done = transition.sample(time.monotonic() - preparation_start)
                    backend.submit(target, image_at, command_at)
                    previous = target.copy()
                    if done:
                        previous_action = None
                        prepared = True
                        window.intent.set_pose_ready()
                        command_heading = yaw_rate = speed = 0.
                        record(dict(event="nominal_pose_ready",policy_enabled=False,lowcmd_enabled=True))
                        print("Nominal pose ready: PD HOLD ONLY, no active balance. Keep physical support. "
                              "With feet on ground: release all buttons, hold A 1 s / press P to start policy.",flush=True)
                elif backend.armed and not policy_started:
                    # Remain in the initialized pose. Never substitute a policy
                    # target before the separate, post-transition start command.
                    backend.submit(default,image_at,command_at)
                    previous = default.copy()
                    record(dict(event="nominal_pose_hold",policy_enabled=False,lowcmd_enabled=True,
                                q_target=default.tolist(),frame_id=frame_id))
                elif policy_started:
                    action,target,inference_s = result['action'],result['target'],result['policy_s']
                    target, projection, slew, diagnostic = prepare_target(
                        target, previous, state, contract, limits, actuator_guards=args.actuator_guards)
                    # Shadow targets are logged, never published.
                    if args.actuator_guards and backend.armed and diagnostic:
                        raise RuntimeError(diagnostic)
                    if backend.armed:
                        backend.submit(target, image_at, command_at)
                    previous_action = (target-default)/scale
                    step_error = float(np.max(abs(target-previous)))
                    previous = target.copy()
                    record(dict(event="control_tick", speed_command=speed,
                                heading_command=command_heading,yaw_rate_command=yaw_rate,
                                input_source=args.input,remote_buttons=state.remote_buttons,action=action.tolist(),
                                q_target=target.tolist(), q=state.q.tolist(), quat=state.quat.tolist(),
                                policy_s=inference_s, image_age_s=time.monotonic()-image_at,
                                target_check=diagnostic, actuator_guards=args.actuator_guards,
                                target_error_max_rad=float(np.max(abs(target-state.q))),
                                target_step_max_rad=step_error,
                                limit_projection_rad=projection, slew_projection_rad=slew,
                                lowcmd_enabled=backend.armed, frame_id=frame_id,
                                planner_s=result['planner_s'],worker_s=result['worker_s'],
                                inference_roundtrip_s=inference_roundtrip_s))
                elapsed = time.monotonic() - begin
                durations.append(elapsed)
                if elapsed > .04:
                    record(dict(event='loop_deadline_warning', elapsed_s=elapsed,
                                lowcmd_enabled=backend.armed, automatic_stop=False))
                next_tick += .02
                if next_tick < time.monotonic() - .02:
                    next_tick = time.monotonic()
                time.sleep(max(0., next_tick-time.monotonic()))
            record(dict(event="completed", loop_p50_s=float(np.percentile(durations, 50)),
                        loop_p99_s=float(np.percentile(durations, 99)), loop_max_s=max(durations),
                        writer=getattr(backend, 'writer_diagnostics', lambda: None)()))
        except BaseException as exc:
            reason = str(exc) or type(exc).__name__
            if backend:
                backend.trip(reason)
            record(dict(event="stopped", reason=reason, hardware_validated=False,
                        writer=getattr(backend, 'writer_diagnostics', lambda: None)()))
            if backend and backend.armed and isinstance(exc, Exception):
                if worker:
                    worker.close()
                    worker = None
                hold_damping(backend, window, record)
            raise
        finally:
            if backend:
                backend.close()
            if worker:
                worker.close()
            if window:
                window.close()
            if recorder:
                recorder.close()
            sys.setswitchinterval(previous_switch_interval)


if __name__ == "__main__":
    main()
