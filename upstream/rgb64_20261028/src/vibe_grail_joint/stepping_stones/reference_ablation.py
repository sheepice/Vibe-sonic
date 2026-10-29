"""Simulation-only planner-command comparison; actor command and guards stay fixed.

Run through stepping_stones.review. The process-local substitution affects only
OnlineReference.replan; no deployment source or installed policy is changed.
Reports are tagged diagnostic and cannot serve as real-control admission.
"""
import argparse
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

MODES = ('ramped', 'wanted_step')


def planner_speed(mode, elapsed, incoming, desired, duration, stop_hold):
    if mode not in MODES:
        raise ValueError('Unknown reference command mode')
    if not all(math.isfinite(x) for x in (elapsed, incoming, desired, duration, stop_hold)):
        raise ValueError('Non-finite command schedule')
    if mode == 'ramped':
        return incoming
    return desired if 1. <= elapsed < duration-stop_hold else 0.


def make_reference_class(base, mode, desired, duration, stop_hold, calls):
    class DiagnosticReference(base):
        def replan(self, elapsed, speed):
            selected = planner_speed(mode, elapsed, speed, desired, duration, stop_hold)
            calls.append(dict(t=float(elapsed), actor_speed_m_s=float(speed),
                              planner_speed_m_s=float(selected)))
            return super().replan(elapsed, selected)
    return DiagnosticReference


def verify_actor_command_trace(report, desired, duration, stop_hold):
    """Check the unchanged 50 Hz, 0.01 m/s-per-step policy command schedule."""
    speed = 0.
    worst = 0.
    trace = report.get('trace', [])
    for step, row in enumerate(trace):
        elapsed = step*.02
        wanted = desired if 1. <= elapsed < duration-stop_hold else 0.
        speed += max(-.01, min(.01, wanted-speed))
        worst = max(worst, abs(float(row['speed_command'])-speed))
        if abs(float(row['t'])-(step+1)*.02) > 1e-6:
            raise ValueError('Diagnostic trace changed the control cadence')
    if not trace or worst > 1e-6:
        raise ValueError('Diagnostic changed the actor speed command')
    return dict(steps_checked=len(trace), maximum_error_m_s=worst,
                unchanged=True, control_hz=50., speed_increment_per_step_m_s=.01)


def write_json(path, value):
    temporary = path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def tag_diagnostic_output(out, metadata, completed):
    """Preserve measured outcomes, remove ordinary deployment-admission files."""
    report_path = out/'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        report.update(mode='reference_ablation', base_simulation_mode='portable',
                      default_deployment_protocol=False, diagnostic_run_completed=completed,
                      reference_command_ablation=metadata)
        write_json(report_path, report)
    acceptance = out/'actuator_acceptance.json'
    if acceptance.exists():
        report = json.loads(acceptance.read_text())
        report.update(mode='reference_ablation', default_deployment_protocol=False,
                      reference_command_ablation=metadata, diagnostic_run_completed=completed)
        write_json(out/'diagnostic_actuator_evidence.json', report)
        acceptance.unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--reference-command-mode', choices=MODES, required=True)
    selected, simulation_args = p.parse_known_args()
    check = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    for key in ('bundle', 'config', 'output', 'mode'):
        check.add_argument('--'+key, required=True)
    check.add_argument('--speed', type=float, required=True)
    check.add_argument('--duration', type=float, required=True)
    check.add_argument('--stop-hold-seconds', type=float, required=True)
    check.add_argument('--camera-fps', type=float, required=True)
    check.add_argument('--actuator-guards', action='store_true')
    check.add_argument('--capture-video', action='store_true')
    args, _ = check.parse_known_args(simulation_args)
    if (args.mode != 'portable' or not args.actuator_guards or args.capture_video
            or args.camera_fps != 30. or args.speed not in (.25, .4)):
        p.error('Use a headless, guarded, 30 Hz RGB portable diagnostic')
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from ..deploy import reference
    from . import simcheck
    from .deployment_suite import digest, utc
    source = Path(__file__).resolve()
    sources = [source, source.with_name('review.py'), source.with_name('simcheck.py'),
               source.with_name('safety.py'), source.parent.parent/'deploy/reference.py',
               source.parent.parent/'deploy/safety.py']
    calls = []
    metadata = dict(version=1, mode=selected.reference_command_mode, started_utc=utc(),
                    desired_speed_m_s=args.speed, duration_s=args.duration,
                    stop_hold_seconds=args.stop_hold_seconds, planner_seed=42,
                    variable_changed='planner_speed_input_only',
                    source_sha256={str(s):digest(s) for s in sources},
                    config_sha256=digest(args.config),
                    bundle_manifest_sha256=digest(Path(args.bundle)/'bundle.json'))
    write_json(Path(str(out)+'.reference_ablation_plan.json'), metadata)
    substituted = make_reference_class(reference.OnlineReference, selected.reference_command_mode,
                                       args.speed, args.duration, args.stop_hold_seconds, calls)
    complete = False
    try:
        with patch.object(reference, 'OnlineReference', substituted), \
                patch.object(sys, 'argv', [sys.argv[0], *simulation_args]):
            simcheck.main()
        report = json.loads((out/'report.json').read_text())
        metadata['actor_command_check'] = verify_actor_command_trace(
            report, args.speed, args.duration, args.stop_hold_seconds)
        if not calls:
            raise ValueError('No planner call was observed')
        complete = True
    finally:
        metadata['planner_calls'] = calls
        metadata['finished_utc'] = utc()
        tag_diagnostic_output(out, metadata, complete)
    print(json.dumps(dict(diagnostic_completed=complete, reference_command_mode=selected.reference_command_mode,
                          output=str(out), default_deployment_protocol=False)), flush=True)


if __name__ == '__main__':
    main()
