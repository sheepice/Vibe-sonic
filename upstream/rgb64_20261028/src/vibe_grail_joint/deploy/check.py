"""Offline deployment benchmark and stationary/start/walk/stop planner check."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .observations import ObservationBuilder, RobotState
from .reference import OnlineReference, PlannerWorker
from .runtime import Runtime


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--seconds", type=float, default=10.)
    p.add_argument("--wall-clock", action="store_true", help="Exercise the actual asynchronous planner on a 50 Hz wall-clock schedule")
    p.add_argument("--warmup-steps", type=int, default=5, help="Discard first-use kernels before timing, as the control entrypoint does; 0 measures cold start")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        p.error("Duration must be positive and finite")
    if args.warmup_steps < 0:
        p.error("Warmup steps cannot be negative")
    if Path(args.output).exists():
        p.error("Use a new output path")
    rt = Runtime(args.bundle, args.provider)
    warmup_s = rt.warmup(args.warmup_steps)
    worker = PlannerWorker(Path(args.bundle) / "planner.onnx", rt.contract) if args.wall_clock else None
    planner = worker.reference if worker else OnlineReference(Path(args.bundle) / "planner.onnx", rt.contract)
    builder = ObservationBuilder(rt.contract)
    rng = np.random.default_rng(3)
    image = rng.integers(0, 256, (63, 112, 3), np.uint8)
    state = RobotState(rt.default.copy(), np.zeros(29), np.array([1., 0, 0, 0]), np.zeros(3), 0.)
    durations, plan_times, samples = [], [], []
    maximum_action = 0.
    late_ticks = 0
    failure = None
    elapsed = 0.
    try:
        if worker:
            deadline = time.monotonic() + 10
            while planner.updated_at is None:
                if worker.error or time.monotonic() > deadline:
                    raise RuntimeError(f"Planner startup failed: {worker.error}")
                time.sleep(.005)
        origin = time.monotonic()
        last_plan = None
        for step in range(round(args.seconds * 50)):
            if worker:
                time.sleep(max(0., origin+step*.02-time.monotonic()))
                late_ticks += int(time.monotonic()-(origin+step*.02) > .02)
            elapsed = step / 50
            speed = .25 if 2 <= elapsed < 6 else 0.
            if worker:
                worker.speed = speed
                if planner.updated_at != last_plan:
                    last_plan = planner.updated_at
                    plan_times.append(planner.seconds)
            elif step % 10 == 0:
                planner.replan(elapsed, speed)
                plan_times.append(planner.seconds)
            before = time.monotonic()
            features = rt.encode(image)
            ref = worker.sample(time.monotonic()) if worker else planner.sample(elapsed)
            obs = builder.build(state, ref, *features, speed, velocity=rt.velocity)
            action, target, _ = rt.act(obs)
            durations.append(time.monotonic()-before)
            maximum_action = max(maximum_action, float(abs(action).max()))
            builder.accept_action(action)
            if step % 25 == 0:
                samples.append(dict(t=elapsed, speed=speed, reference_vx=float(ref.root_lin_vel[0]),
                                    reference_dq_max=float(abs(ref.joint_vel).max())))
    except Exception as exc:
        failure = dict(reason=str(exc), step_time_s=elapsed,
                       planner_last_inference_s=planner.seconds,
                       planner_worker_error=str(worker.error) if worker and worker.error else None)
    finally:
        if worker:
            worker.close()
    report = dict(bundle=str(Path(args.bundle).resolve()), provider=args.provider,
                  observation_frames=len(durations), finite_actions=failure is None,
                  completed=failure is None, failure=failure,
                  inference_plus_observation_p50_ms=float(np.percentile(durations, 50)*1000) if durations else None,
                  inference_plus_observation_p99_ms=float(np.percentile(durations, 99)*1000) if durations else None,
                  inference_plus_observation_max_ms=float(max(durations)*1000) if durations else None,
                  planner_p99_ms=float(np.percentile(plan_times, 99)*1000) if plan_times else None,
                  planner_max_ms=float(max(plan_times)*1000) if plan_times else None,
                  wall_clock=args.wall_clock, late_ticks_over_20ms=late_ticks if worker else None,
                  warmup_steps=args.warmup_steps, warmup_s=warmup_s,
                  max_abs_action=maximum_action, command_transition_samples=samples,
                  physics_simulation=False, hardware_test=False,
                  note="Fixed synthetic robot state/RGB: tests interfaces and timing, NOT balance or stair success")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != 'command_transition_samples'}, indent=2))
    if failure:
        raise RuntimeError("Deployment timing check failed; see the saved report")


if __name__ == "__main__":
    main()
