"""CPU-only measurement of the existing training/deployment reference paths.

This does not run a robot, change a policy, or establish a cause of failures.
The alternate planner command is a diagnostic, never a deployment default.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command_schedule(duration, speed):
    ticks = np.arange(round(duration * 50))
    elapsed = ticks * .02
    wanted = np.where((elapsed >= 1.) & (elapsed < duration - 3.), speed, 0.)
    ramped = np.empty_like(wanted)
    value = 0.
    for i, target in enumerate(wanted):
        value += float(np.clip(target - value, -.01, .01))
        ramped[i] = value
    return elapsed, wanted, ramped


def pack(rows, elapsed, wanted, command, planner_command):
    names = ('joint_pos', 'joint_vel', 'root_quat', 'root_lin_vel', 'root_ang_vel')
    out = {name: np.stack([getattr(row, name) for row in rows]) for name in names}
    out.update(time_s=elapsed, wanted=wanted, command=command, planner_command=planner_command)
    if out['joint_pos'].shape != (len(elapsed), 10, 29):
        raise ValueError('Unexpected SONIC reference window')
    if not all(np.isfinite(value).all() for value in out.values()):
        raise ValueError('Non-finite reference trace')
    return out


def summarize(trace, contract, duration):
    q, dq, times = trace['joint_pos'][:, 0], trace['joint_vel'][:, 0], trace['time_s']
    bounds = np.asarray(contract['joint_limits'])
    deviation = np.maximum(bounds[:, 0] - q, q - bounds[:, 1]).clip(min=0)
    delta = np.abs(np.diff(q, axis=0))
    result = dict(max_current_joint_step_rad=float(delta.max()),
                  p99_current_joint_step_rad=float(np.percentile(delta, 99)),
                  max_current_reference_joint_speed_rad_s=float(np.abs(dq).max()),
                  max_reference_joint_rom_excess_rad=float(deviation.max()),
                  rom_note='Reference poses are not commanded joint targets; this is not an actuator acceptance test.')
    result['windows'] = {}
    for label, begin, end in (('standing', 0., 1.), ('start', 1., 2.5),
                              ('steady', 2.5, duration - 3.), ('stop', duration - 3., duration)):
        mask = (times >= begin) & (times < end)
        vx = trace['root_lin_vel'][mask, 0]
        command = trace['command'][mask]
        result['windows'][label] = dict(samples=int(mask.sum()),
            reference_vx_mean_m_s=float(vx.mean()),
            reference_vx_command_rmse_m_s=float(np.sqrt(np.mean((vx-command)**2))),
            joint_speed_rms_rad_s=float(np.sqrt(np.mean(dq[mask]**2))))
    # Measure revisions of the 0.9-second lookahead, not physical tracking error.
    revised = trace['joint_pos'][:-45, 9] - q[45:]
    result['lookahead_0p9s_revision_rmse_rad'] = float(np.sqrt(np.mean(revised**2)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--speed', type=float, choices=(.25, .4), default=.4)
    p.add_argument('--duration', type=float, default=8.)
    a = p.parse_args()
    if not 7. <= a.duration <= 12.:
        p.error('Use a bounded 7..12-second CPU audit')
    root = Path(a.bundle).resolve()
    manifest = json.loads((root / 'bundle.json').read_text())
    planner = root / 'planner.onnx'
    if sha256(planner) != manifest['files']['planner.onnx']:
        raise ValueError('Planner checksum mismatch')
    contract = manifest['robot']
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    from ..deploy.reference import OnlineReference
    from ..deploy.runtime import session
    from ..deploy.observations import Reference
    from ..velocity.planner import PlannerPool
    elapsed, wanted, ramped = command_schedule(a.duration, a.speed)
    speeds = [0., .25, .4, .5, .7]
    stream = speeds.index(a.speed)
    seeds = (42, 137 + stream)
    inference = session(planner, 'cpu')
    traces, results = {}, {}
    started = time.monotonic()
    for seed in seeds:
        for mode in ('ramped', 'wanted_step'):
            name = f'deploy_seed{seed}_{mode}'
            reference = OnlineReference(planner, contract, seed=seed, session_override=inference)
            planner_command = ramped if mode == 'ramped' else wanted
            rows, calls = [], []
            for tick, now in enumerate(elapsed):
                if tick % 10 == 0:
                    reference.replan(float(now), float(planner_command[tick]))
                    calls.append(dict(time_s=float(now), speed=float(planner_command[tick])))
                rows.append(reference.sample(float(now)))
            trace = pack(rows, elapsed, wanted, ramped, planner_command)
            traces[name] = trace
            np.savez_compressed(out / (name + '.npz'), **trace)
            results[name] = dict(summarize(trace, contract, a.duration), planner_calls=calls,
                                reference_class='deploy.reference.OnlineReference', seed=seed)
            print(name, 'completed', flush=True)
    # Use the actual training pool, including its constructor warmup, five streams,
    # interpolation, reference FK, and phase endpoints. No physics world is built.
    seed_pose = np.r_[0., 0., contract['root_height'], 1., 0., 0., 0., contract['default_joint_pos']].astype(np.float32)
    pool = PlannerPool(planner, seed_pose, speeds, seed=137)
    initial_cursor = pool.cursor
    channels = pool.channels()
    phase_rows = {0: [], 19: []}
    for tick in range(len(elapsed)):
        if tick:
            changed, _ = pool.advance()
            if changed:
                channels = pool.channels()
        for phase, rows in phase_rows.items():
            local = pool.cursor - phase
            if local < 0 or local + 45 >= pool.valid_end[stream]:
                raise RuntimeError('Invalid training pool horizon')
            indices = stream * pool.capacity + local + np.arange(10) * 5
            current = int(indices[0])
            rows.append(Reference(channels['joint_pos'][indices].copy(), channels['joint_vel'][indices].copy(),
                                  channels['body_quat_w'][indices, 0].copy(),
                                  channels['body_lin_vel_w'][current, 0].copy(),
                                  channels['body_ang_vel_w'][current, 0].copy()))
    for phase, rows in phase_rows.items():
        name = f'training_pool_phase{phase}'
        constant = np.full_like(elapsed, a.speed)
        trace = pack(rows, elapsed, constant, constant, constant)
        traces[name] = trace
        np.savez_compressed(out / (name + '.npz'), **trace)
        results[name] = dict(summarize(trace, contract, a.duration), initial_pool_cursor=initial_cursor,
                            phase=phase, planner_seed=137+stream, fixed_speed=True,
                            reference_class='velocity.planner.PlannerPool')
    differences = {}
    for seed in seeds:
        delta = traces[f'deploy_seed{seed}_ramped']['joint_pos'] - traces[f'deploy_seed{seed}_wanted_step']['joint_pos']
        differences[str(seed)] = dict(current_joint_rmse_rad=float(np.sqrt(np.mean(delta[:, 0]**2))),
                                     full_future_window_rmse_rad=float(np.sqrt(np.mean(delta**2))))
    source_root = Path(__file__).resolve().parents[1]
    report = dict(bundle=str(root), planner_sha256=manifest['files']['planner.onnx'], speed_m_s=a.speed,
        duration_s=a.duration, elapsed_wall_s=time.monotonic()-started,
        cpu_only=True, physical_robot_or_world_used=False, policy_actions_measured=False,
        causal_failure_explanation_established=False,
        note='Exact reference generators measured; differences alone do not establish locomotion quality or a causal explanation of guard failures. Training pool windows remain fixed-speed, including the time labelled stop.',
        protocol_sources={str(path): sha256(path) for path in (
            Path(__file__), source_root/'deploy/reference.py', source_root/'velocity/planner.py',
            source_root/'stepping_stones/simcheck.py')}, traces=results,
        same_seed_ramp_vs_step= differences)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(output=str(out), wall_seconds=report['elapsed_wall_s'],
                          same_seed_ramp_vs_step=differences), indent=2), flush=True)


if __name__ == '__main__':
    main()
