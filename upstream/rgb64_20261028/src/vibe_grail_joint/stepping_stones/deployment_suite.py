"""Fixed, repeatable raw-RGB deployment trials on bounded spare GPU memory.

Uses the existing review launcher and preserves every failure. Never restarts
training, retries a failed trial, changes actuator guards, or sends hardware I/O.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from .outcome import portable_outcome


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def make_trials(plan):
    if plan.get('version') != 1 or plan.get('repeats') != 2:
        raise ValueError('Version 1 requires two preselected repeats')
    candidates = plan['candidates']
    names = [c['name'] for c in candidates]
    if not names or len(set(names)) != len(names):
        raise ValueError('Candidate names must be nonempty and unique')
    for name in names:
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in name):
            raise ValueError('Use plain lowercase candidate names')
    # One moderate held-out scene per family, both trained command speeds.
    scenes = [('stairs', 8), ('box_stones', 2), ('cylinder_stones', 2), ('stair_gap', 2)]
    trials = []
    for repeat in range(2):
        for family, level in scenes:
            for speed in (.4, .25):
                for candidate in candidates:
                    key = f"{candidate['name']}_{family}_l{level}_v{str(speed).replace('.', 'p')}_r{repeat+1}"
                    trials.append(dict(id=key, candidate=candidate['name'], family=family,
                                       level=level, speed=speed, seed=93157, variant=2,
                                       repeat=repeat+1, duration=28 if speed == .4 else 40))
    return trials


def validate_report(report, trial, candidate):
    expected = dict(mode='portable', family=trial['family'], level=trial['level'],
                    seed=trial['seed'], terrain_variant=trial['variant'],
                    command_speed_m_s=trial['speed'], requested_seconds=trial['duration'],
                    stop_hold_seconds=3., initial_state='stationary_default_pose',
                    raw_rgb_encoder_used=True, no_physics_guards_applied=False,
                    truth_used_as_actor_input=False, hardware_test=False,
                    deploy_onnx_drove_scored_world=True)
    for key, value in expected.items():
        if report.get(key) != value:
            raise ValueError(f'Mismatched deployment protocol: {key}')
    if report.get('bundle') != candidate['bundle'] or report.get('bundle_files') != candidate['bundle_files']:
        raise ValueError('Report does not belong to the fixed candidate bundle')
    if report.get('camera_cadence', {}).get('requested_fps') != 30.:
        raise ValueError('Expected 30 Hz causally sampled RGB')
    if 'video' in report:
        raise ValueError('This suite uses independent headless repeats')
    from ..deploy.safety import Limits
    from dataclasses import asdict
    if report.get('limits') != asdict(Limits()):
        raise ValueError('Actuator guards changed')
    result = portable_outcome(report)
    return dict(passed=result['course_and_stop_passed'],
                reasons=result['task_failure_reasons'],
                course_success=report['course_success'],
                completion_time_s=report['course_completion_time_s'],
                fall=report['fall'], guard_stop_reason=report.get('guard_stop_reason'),
                simulated_seconds=report['simulated_seconds'])


def process_live(pid, command, created=None):
    import psutil
    try:
        p = psutil.Process(pid)
        return (p.is_running() and p.status() != psutil.STATUS_ZOMBIE
                and p.cmdline() == command
                and (created is None or abs(p.create_time()-created) < .01))
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False


def summarize(state):
    result = {}
    for candidate in state['candidates']:
        rows = [r for r in state['trials'] if r['candidate'] == candidate['name']]
        done = [r for r in rows if r['status'] == 'completed']
        pairs = {}
        for row in done:
            pairs.setdefault((row['family'], row['speed']), []).append(row['result']['passed'])
        result[candidate['name']] = dict(
            planned=len(rows), completed=len(done), passed=sum(r['result']['passed'] for r in done),
            process_failures=sum(r['status'] == 'process_failed' for r in rows),
            scene_speed_pairs=8,
            both_repeats_passed=sum(len(v) == 2 and all(v) for v in pairs.values()))
    return result


def run(args):
    import psutil
    from ..gpu_guard import query
    root = Path(__file__).resolve().parents[3]
    out = Path(args.output).resolve()
    plan = json.loads(Path(args.plan).read_text())
    trial_specs = make_trials(plan)
    gpus = [int(x) for x in args.gpus.split(',')]
    if not gpus or len(gpus) != len(set(gpus)) or not set(gpus) <= {0, 1, 2, 3, 6, 7}:
        raise ValueError('Only distinct GPUs 0-3,6,7 are authorized')
    if args.resume:
        if not (out/'state.json').is_file():
            raise ValueError('Resume requires a saved suite state')
    else:
        out.mkdir(parents=True, exist_ok=False)
    lock = (out/'suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    candidates = []
    for item in plan['candidates']:
        c = dict(item)
        for key in ('config', 'checkpoint', 'bundle'):
            c[key] = str((root/c[key]).resolve())
        manifest = json.loads((Path(c['bundle'])/'bundle.json').read_text())
        if str(Path(manifest['checkpoint']).resolve()) != c['checkpoint']:
            raise ValueError('Export/checkpoint mismatch')
        c['bundle_files'] = manifest['files']
        for name, expected in c['bundle_files'].items():
            if digest(Path(c['bundle'])/name) != expected:
                raise ValueError('Bundle file changed: '+name)
        c['config_sha256'] = digest(c['config'])
        c['checkpoint_sha256'] = digest(c['checkpoint'])
        c['manifest_sha256'] = digest(Path(c['bundle'])/'bundle.json')
        candidates.append(c)
    if args.resume:
        state = json.loads((out/'state.json').read_text())
        if state['plan'] != plan or state['candidates'] != candidates or state['gpus'] != gpus:
            raise ValueError('Cannot change a started evaluation suite')
        if len(state['trials']) != len(trial_specs) or any(
                any(row.get(k) != v for k, v in spec.items())
                for row, spec in zip(state['trials'], trial_specs)):
            raise ValueError('Saved trial matrix changed')
    else:
        state = dict(started_utc=utc(), plan=plan, candidates=candidates, gpus=gpus,
                     hardware_validated=False, retries=0, trials=[dict(t, status='pending') for t in trial_specs])
    by_name = {c['name']: c for c in candidates}
    owned = {}
    while True:
        occupied = set()
        for trial in state['trials']:
            if trial['status'] != 'running':
                continue
            if trial['id'] in owned:
                owned[trial['id']].poll()  # reap our children; resumed workers are checked by identity
            alive = process_live(trial['pid'], trial['command'], trial['process_created'])
            resource_path = Path(trial['output']+'.resource.json')
            try:
                resource = json.loads(resource_path.read_text()) if resource_path.exists() else trial.get('last_resource', {})
            except json.JSONDecodeError:
                # review writes this small diagnostic file in place. A partial
                # observation is not evidence that the worker has terminated.
                resource = trial.get('last_resource', {})
            if resource:
                trial['last_resource'] = resource
            # If a controller or review parent was interrupted, wait for its
            # existing worker instead of launching a second copy of the trial.
            child_alive = bool(resource.get('worker_pid') and process_live(resource['worker_pid'], resource['command']))
            if not alive and not child_alive:
                # A parent may have died before the first resource write. Find
                # its uniquely named worker before classifying the trial ended.
                for proc in psutil.process_iter(['pid', 'cmdline', 'status']):
                    cmd = proc.info['cmdline'] or []
                    if ('vibe_grail_joint.stepping_stones.simcheck' in cmd and '--output' in cmd
                            and cmd[cmd.index('--output')+1] == trial['output']
                            and proc.info['status'] != psutil.STATUS_ZOMBIE):
                        child_alive = True
                        trial['recovered_worker_pid'] = proc.pid
                        break
            if alive or child_alive:
                occupied.add(trial['gpu'])
                continue
            report_path = Path(trial['output'])/'report.json'
            try:
                if resource.get('exit_code') not in (None, 0) or resource.get('status') == 'stopped_own_review_budget':
                    raise ValueError('Review process failed or exceeded its resource allowance')
                if not report_path.is_file():
                    raise ValueError('Process exited without a final report')
                report = json.loads(report_path.read_text())
                trial['result'] = validate_report(report, trial, by_name[trial['candidate']])
                if trial['result']['passed']:
                    from .safety import validate_sim_acceptance
                    manifest = json.loads((Path(by_name[trial['candidate']]['bundle'])/'bundle.json').read_text())
                    validate_sim_acceptance(report_path.parent/'actuator_acceptance.json', manifest, trial['speed'])
                trial.update(status='completed', finished_utc=utc(), resource_exit_code=resource.get('exit_code'))
            except (ValueError, KeyError, OSError) as error:
                trial.update(status='process_failed', finished_utc=utc(), error=str(error))
        available = {}
        for row in query('index,memory.total,memory.used', 'gpu'):
            available[int(row[0])] = int(row[1])-int(row[2])
        for gpu in gpus:
            if gpu in occupied or available.get(gpu, 0) < 8192:
                continue
            pending = next((t for t in state['trials'] if t['status'] == 'pending'), None)
            if pending is None:
                break
            review_lock = (root/'records'/f'gpu_{gpu}_review.lock').open('a')
            try:
                fcntl.flock(review_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                review_lock.close()
                continue
            review_lock.close()  # the existing review launcher owns the actual job lock
            c = by_name[pending['candidate']]
            dest = out/pending['id']
            if dest.exists() or (out/(pending['id']+'.log')).exists():
                raise RuntimeError('Unrecorded trial artifacts exist; reconcile its process handle before resuming')
            command = [sys.executable, '-m', 'vibe_grail_joint.stepping_stones.review',
                       '--gpu', str(gpu), '--config', c['config'], '--checkpoint', c['checkpoint'],
                       '--bundle', c['bundle'], '--deploy-mode', 'portable', '--family', pending['family'],
                       '--level', str(pending['level']), '--speed', str(pending['speed']),
                       '--seed', str(pending['seed']), '--terrain-variant', str(pending['variant']),
                       '--duration', str(pending['duration']), '--camera-fps', '30', '--actuator-guards',
                       '--stop-hold-seconds', '3', '--output', str(dest)]
            with (out/(pending['id']+'.log')).open('xb') as log:
                proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
            pending.update(status='running', gpu=gpu, pid=proc.pid, command=command,
                           process_created=psutil.Process(proc.pid).create_time(), output=str(dest), started_utc=utc())
            owned[pending['id']] = proc
            write(out/'state.json', state)
        state['updated_utc'] = utc()
        state['summary'] = summarize(state)
        state['status'] = 'running' if any(t['status'] in ('pending', 'running') for t in state['trials']) else 'completed'
        write(out/'state.json', state)
        if state['status'] == 'completed':
            print(json.dumps(state['summary'], indent=2), flush=True)
            return
        time.sleep(15)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpus', default='1,3,7')
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
