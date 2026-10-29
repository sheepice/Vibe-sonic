"""Run preselected paired planner-command diagnostics without altering training.

Each pair uses one physical GPU sequentially. All failures are kept; no retries,
hardware admission, or automatic training changes are performed.
"""
import argparse
import json
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .deployment_suite import digest, utc, validate_report, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpus', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    plan = json.loads(Path(args.plan).read_text())
    gpus = [int(x) for x in args.gpus.split(',')]
    if not gpus or len(set(gpus)) != len(gpus) or not set(gpus) <= {0,1,2,3,6,7}:
        raise ValueError('Use distinct authorized GPUs 0-3,6,7')
    if plan.get('version') != 1 or plan.get('repeats') != 2:
        raise ValueError('Two preselected independent repeats required')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    candidates = []
    for item in plan['candidates']:
        c = dict(item)
        for key in ('config', 'checkpoint', 'bundle'):
            c[key] = str((root/c[key]).resolve())
        manifest = json.loads((Path(c['bundle'])/'bundle.json').read_text())
        if str(Path(manifest['checkpoint']).resolve()) != c['checkpoint']:
            raise ValueError('Candidate export/checkpoint mismatch')
        c['bundle_files'] = manifest['files']
        for name, expected in c['bundle_files'].items():
            if digest(Path(c['bundle'])/name) != expected:
                raise ValueError('Changed exported model: '+name)
        c.update(config_sha256=digest(c['config']), checkpoint_sha256=digest(c['checkpoint']))
        candidates.append(c)
    if len({c['name'] for c in candidates}) != len(candidates):
        raise ValueError('Candidate names must be unique')
    trials = []
    pairs = queue.Queue()
    for repeat in (1, 2):
        for family in ('box_stones', 'stair_gap'):
            for candidate in candidates:
                pair = []
                modes = ('ramped', 'wanted_step') if repeat == 1 else ('wanted_step', 'ramped')
                for mode in modes:
                    key = f"{candidate['name']}_{family}_l2_{mode}_r{repeat}"
                    trial = dict(id=key, candidate=candidate['name'], family=family, level=2,
                                 speed=.4, duration=28., seed=93157, variant=2, repeat=repeat,
                                 reference_command_mode=mode, status='pending')
                    trials.append(trial)
                    pair.append(trial)
                pairs.put((candidate, pair))
    state = dict(started_utc=utc(), plan=plan, gpus=gpus, candidates=candidates,
                 diagnostic_only=True, default_deployment_protocol=False,
                 retries=0, trials=trials, status='running')
    lock = threading.Lock()

    def save():
        state['updated_utc'] = utc()
        summary = {}
        for candidate in candidates:
            by_mode = {}
            for mode in ('ramped', 'wanted_step'):
                rows = [r for r in trials if r['candidate']==candidate['name']
                        and r['reference_command_mode']==mode]
                completed = [r for r in rows if r['status']=='completed']
                by_mode[mode] = dict(planned=len(rows), completed=len(completed),
                                    diagnostic_course_and_stop_passed=sum(r['result']['passed'] for r in completed),
                                    process_failures=sum(r['status']=='process_failed' for r in rows))
            summary[candidate['name']] = by_mode
        state['summary'] = summary
        write(out/'state.json', state)

    save()

    def worker(gpu):
        while True:
            try:
                candidate, pair = pairs.get_nowait()
            except queue.Empty:
                return
            for trial in pair:
                destination = out/trial['id']
                command = [sys.executable, '-m', 'vibe_grail_joint.stepping_stones.review',
                    '--gpu', str(gpu), '--config', candidate['config'], '--checkpoint', candidate['checkpoint'],
                    '--bundle', candidate['bundle'], '--output', str(destination), '--duration', str(trial['duration']),
                    '--deploy-mode', 'portable', '--actuator-guards', '--speed', str(trial['speed']),
                    '--family', trial['family'], '--level', str(trial['level']), '--seed', str(trial['seed']),
                    '--terrain-variant', str(trial['variant']), '--camera-fps', '30', '--stop-hold-seconds', '3',
                    '--reference-command-mode', trial['reference_command_mode']]
                try:
                    with (out/(trial['id']+'.log')).open('w') as logfile:
                        process = subprocess.Popen(command, cwd=root, stdout=logfile, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        with lock:
                            trial.update(status='running', gpu=gpu, pid=process.pid, command=command,
                                         output=str(destination), started_utc=utc())
                            save()
                        code = process.wait()
                    if code != 0:
                        raise RuntimeError(f'Review exited with code {code}')
                    report = json.loads((destination/'report.json').read_text())
                    metadata = report.get('reference_command_ablation', {})
                    if (report.get('mode') != 'reference_ablation' or report.get('default_deployment_protocol') is not False
                            or report.get('diagnostic_run_completed') is not True
                            or metadata.get('mode') != trial['reference_command_mode']
                            or metadata.get('actor_command_check',{}).get('unchanged') is not True
                            or (destination/'actuator_acceptance.json').exists()):
                        raise ValueError('Missing isolated planner diagnostic evidence')
                    # Reuse invariant checks only; the stored report remains diagnostic.
                    result = validate_report(dict(report, mode='portable'), trial, candidate)
                    result['diagnostic_only'] = True
                    with lock:
                        trial.update(status='completed', result=result, report_sha256=digest(destination/'report.json'),
                                     finished_utc=utc())
                        save()
                except Exception as error:
                    with lock:
                        trial.update(status='process_failed', error=str(error), finished_utc=utc())
                        save()
            pairs.task_done()

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        list(pool.map(worker, gpus))
    state['status'] = 'completed' if all(t['status']=='completed' for t in trials) else 'completed_with_process_failures'
    state['finished_utc'] = utc()
    save()
    print(json.dumps(state['summary'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
