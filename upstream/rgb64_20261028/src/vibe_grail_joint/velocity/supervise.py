"""Persistent bounded training stages with visible status, checkpoints and fixed evaluation.

Only signals process groups created here. Ownership guards protect unrelated jobs.
Automatic environment reduction is restricted to an actual out-of-memory error.
Other errors stop with their full traceback for diagnosis.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ..catalog import write_json


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stop_owned_process(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)


def run_monitored(command, directory, environment, training):
    directory.mkdir(parents=True, exist_ok=True)
    last_progress = time.monotonic()
    progress = -1
    failure = None
    with (directory / 'console.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment, start_new_session=True)
        launched = time.monotonic()
        write_json(directory / 'process.json', dict(pid=process.pid, command=command, started=now()))
        while process.poll() is None:
            if not training and time.monotonic() - launched > 900:
                failure = 'Evaluation exceeded 900s wall-clock limit'
                stop_owned_process(process)
                break
            if training:
                records = []
                for p in directory.glob('rank_*_status.json'):
                    try:
                        records.append(json.loads(p.read_text()))
                    except (OSError, json.JSONDecodeError):
                        continue
                failed = [r for r in records if r.get('status') == 'failed']
                if failed:
                    failure = '; '.join(f'rank {r["rank"]}: {r.get("error")}' for r in failed)
                    stop_owned_process(process)
                    break
                current = min((r.get('absolute_completed_updates', r.get('completed_updates', 0)) for r in records), default=0)
                if current != progress:
                    progress, last_progress = current, time.monotonic()
                # Startup includes bank loading and compilation; first update gets more time.
                deadline = 600 if current == 0 else 240
                if time.monotonic() - last_progress > deadline:
                    failure = f'No all-rank progress for {deadline}s at absolute update {current}'
                    stop_owned_process(process)
                    break
            time.sleep(5)
        code = process.wait()
    write_json(directory / 'exit.json', dict(exit_code=code, failure=failure, finished=now()))
    return code, failure


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--stages', default='200,1000,2000,4000,6000')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    out = Path(args.run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / 'supervisor_status.json'
    if status_path.exists():
        raise ValueError(f'Refusing to overwrite existing supervisor: {out}')
    settings = json.loads(Path(args.config).read_text())
    stages = [int(s) for s in args.stages.split(',')]
    if stages != sorted(set(stages)) or stages[-1] != settings['iterations']:
        raise ValueError('Stages must increase and end at the configured total updates')
    config_path = out / 'config.json'
    write_json(config_path, settings)
    status = dict(status='starting', pid=os.getpid(), started=now(), stages=stages,
                  completed_updates=0, num_envs_per_rank=settings['num_envs_per_rank'],
                  latest_checkpoint=None, evaluations=[], failure=None)
    write_json(status_path, status)
    environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONFAULTHANDLER='1',
                       TORCH_SHOW_CPP_STACKTRACES='1', TORCH_NCCL_TRACE_BUFFER_SIZE='4096',
                       TORCH_NCCL_DUMP_ON_TIMEOUT='1')
    try:
        for target in stages:
            # A stage can resume from its own last good checkpoint after actual OOM.
            attempt = 0
            while status['completed_updates'] < target:
                start = status['completed_updates']
                directory = out / f'train_{start:05d}_{target:05d}_attempt{attempt}'
                cmd = [sys.executable, '-m', 'vibe_grail_joint.velocity.launch', '--gpus', '0,1,2,3,4,5,6,7',
                       '--config', str(config_path), '--log-dir', str(directory),
                       '--num-envs', str(status['num_envs_per_rank']), '--iterations', str(target-start)]
                if status['latest_checkpoint']:
                    cmd += ['--resume', status['latest_checkpoint']]
                status.update(status='training', active_directory=str(directory), target_updates=target, updated=now())
                write_json(status_path, status)
                code, failure = run_monitored(cmd, directory, environment, training=True)
                checkpoints = sorted(directory.glob('model_*.pt'), key=lambda q:int(q.stem.split('_')[-1]))
                if checkpoints:
                    last = checkpoints[-1]
                    status.update(latest_checkpoint=str(last), completed_updates=int(last.stem.split('_')[-1])+1)
                if code == 0:
                    ranks = [json.loads((directory / f'rank_{r}_status.json').read_text()) for r in range(8)]
                    if not all(r.get('status') == 'completed' and r.get('frozen_unchanged') for r in ranks):
                        raise RuntimeError('Training exited without all eight completion/frozen-weight checks')
                    if status['completed_updates'] != target:
                        raise RuntimeError('Final checkpoint does not match completed stage')
                    break
                log = (directory / 'console.log').read_text(errors='replace')
                oom = 'out of memory' in log.lower()
                if not oom or attempt >= 2 or status['num_envs_per_rank'] <= 510:
                    raise RuntimeError(f'Training exited {code}: {failure}; see {directory}')
                status['num_envs_per_rank'] = max(510, (status['num_envs_per_rank']//20)*10)
                attempt += 1
            evaluation = out / f'evaluate_original_stair_{target:05d}'
            status.update(status='evaluating', active_directory=str(evaluation), updated=now())
            write_json(status_path, status)
            cmd = [sys.executable, '-m', 'vibe_grail_joint.velocity.launch_original_stair', '--gpu', '0',
                   '--bank', str(root / 'data/stairs_validation/rank_0'),
                   '--checkpoint', status['latest_checkpoint'], '--base-checkpoint', settings['initial_checkpoint'],
                   '--planner', settings['planner'], '--output', str(evaluation),
                   '--duration', '20', '--speed', '.5', '--heading-deg', '-90', '--seed', '42']
            code, failure = run_monitored(cmd, evaluation, environment, training=False)
            if code:
                # Record the eval failure distinctly; never call it a policy failure/success.
                status['evaluations'].append(dict(updates=target, exit_code=code, error=failure,
                                                 directory=str(evaluation)))
            else:
                report = json.loads((evaluation / 'report.json').read_text())
                status['evaluations'].append(dict(updates=target, exit_code=0, online_w=report['online_w'],
                                                 directory=str(evaluation)))
            write_json(status_path, status)
            if target >= 1000:
                ablation = out / f'evaluate_original_stair_{target:05d}_zero_tokens'
                status.update(status='evaluating_visual_ablation', active_directory=str(ablation), updated=now())
                write_json(status_path, status)
                blind_cmd = list(cmd)
                blind_cmd[blind_cmd.index('--output') + 1] = str(ablation)
                blind_cmd += ['--visual-ablation', 'zero_tokens']
                code, failure = run_monitored(blind_cmd, ablation, environment, training=False)
                entry = dict(updates=target, visual_ablation='zero_tokens', exit_code=code,
                             directory=str(ablation))
                if code:
                    entry['error'] = failure
                else:
                    entry['online_w'] = json.loads((ablation / 'report.json').read_text())['online_w']
                status['evaluations'].append(entry)
                write_json(status_path, status)
        status.update(status='completed', finished=now())
        write_json(status_path, status)
    except BaseException as error:
        status.update(status='failed', failure=repr(error), finished=now())
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    main()
