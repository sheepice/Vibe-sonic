"""Bounded evaluation alongside this experiment's verified training process tree.

The normal idle-only launchers and their ownership guards are unchanged. This
explicit entrypoint only shares with an identified, healthy velocity supervisor;
unrelated GPU users are rejected. Only the evaluation child may be terminated.
"""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from ..catalog import write_json
from ..gpu_guard import assert_gpus_idle, query


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def training_state(lease):
    root = Path(lease['training_run']).resolve()
    state = json.loads((root / 'supervisor_status.json').read_text())
    if state['status'] != 'training' or state['pid'] != lease['training_pid']:
        raise RuntimeError('The identified training stage is no longer active')
    owner = psutil.Process(lease['training_pid'])
    if abs(owner.create_time() - lease['training_create_time']) > .01:
        raise RuntimeError('Training PID identity changed')
    command = owner.cmdline()
    if 'vibe_grail_joint.velocity.supervise' not in command:
        raise RuntimeError('The owner is not the velocity supervisor')
    run_arg = command[command.index('--run-dir') + 1]
    if (Path(owner.cwd()) / run_arg).resolve() != root:
        raise RuntimeError('Supervisor command belongs to another experiment')
    active = Path(state['active_directory']).resolve()
    if not active.is_relative_to(root):
        raise RuntimeError('Training stage is outside the declared run')
    ranks = [json.loads((active / f'rank_{r}_status.json').read_text()) for r in range(8)]
    if not all(r['status'] == 'running' for r in ranks):
        raise RuntimeError('Sharing requires all eight training ranks to be running')
    children = owner.children(recursive=True)
    workers = []
    for child in children:
        child_command = child.cmdline()
        if '-m' in child_command and child_command[child_command.index('-m') + 1] == 'vibe_grail_joint.velocity.train':
            workers.append(child)
    if len(workers) != 8:
        raise RuntimeError(f'Expected eight training workers, found {len(workers)}')
    return {owner.pid, *(p.pid for p in children)}, min(r['absolute_completed_updates'] for r in ranks)


def gpu_memory(gpu):
    rows = query('index,memory.free,memory.used', 'gpu')
    row = next(r for r in rows if int(r[0]) == gpu)
    return {'free_mib': int(row[1]), 'used_mib': int(row[2])}


def verify_shared_worker():
    lease = json.loads(Path(os.environ['VIBE_SHARED_EVAL_LEASE']).read_text())
    parent_pid = lease['evaluation_launcher_pid']
    if parent_pid not in [p.pid for p in psutil.Process().parents()]:
        raise RuntimeError('Shared evaluation must be a child of its verified launcher')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(lease['gpu']):
        raise RuntimeError('Evaluation GPU differs from its lease')
    training_pids, _ = training_state(lease)
    parent = psutil.Process(parent_pid)
    evaluation_pids = {parent.pid, *(p.pid for p in parent.children(recursive=True))}
    assert_gpus_idle([lease['gpu']], training_pids | evaluation_pids)
    if gpu_memory(lease['gpu'])['free_mib'] < lease['startup_free_mib']:
        raise RuntimeError('Insufficient memory headroom for bounded evaluation')
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20, 0)


def terminate_child(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=('original', 'uniform'))
    parser.add_argument('--kind', choices=('original', 'uniform'))
    parser.add_argument('--training-run')
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--output')
    args, rest = parser.parse_known_args()
    if args.worker:
        # The evaluation itself runs the selected guard before initializing CUDA.
        sys.argv = [sys.argv[0], '--output', args.output, *rest]
        if args.worker == 'original':
            from .evaluate_original_stair import main as evaluate
        else:
            from .evaluate import main as evaluate
        evaluate(guard_check=verify_shared_worker)
        return
    if not args.kind or not args.training_run or not args.output:
        parser.error('--kind, --training-run and --output are required')
    root = Path(__file__).resolve().parents[3]
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f'Refusing to overwrite an evaluation: {output}')
    output.mkdir(parents=True)
    run = Path(args.training_run).resolve()
    state = json.loads((run / 'supervisor_status.json').read_text())
    owner = psutil.Process(state['pid'])
    lease = dict(training_run=str(run), training_pid=owner.pid,
                 training_create_time=owner.create_time(), evaluation_launcher_pid=os.getpid(),
                 gpu=args.gpu, kind=args.kind, startup_free_mib=12 * 1024,
                 minimum_free_mib=8 * 1024, created_utc=utc_now())
    lease_path = output / 'shared_lease.json'
    write_json(lease_path, lease)
    with (root / 'records' / f'shared_evaluation_gpu_{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        allowed, progress = training_state(lease)
        assert_gpus_idle([args.gpu], allowed)
        if gpu_memory(args.gpu)['free_mib'] < lease['startup_free_mib']:
            raise RuntimeError('Insufficient startup memory headroom')
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                           CUDA_DEVICE_ORDER='PCI_BUS_ID', OMP_NUM_THREADS='2',
                           HF_HOME=str(root / 'models/hf'), HF_HUB_OFFLINE='1',
                           PYTHONUNBUFFERED='1', VIBE_SHARED_EVAL_LEASE=str(lease_path))
        command = [sys.executable, '-m', 'vibe_grail_joint.velocity.shared_evaluation',
                   '--worker', args.kind, '--output', str(output), *rest]
        started = last_progress = time.monotonic()
        error = None
        monitor = []
        with (output / 'console.log').open('w') as log:
            process = subprocess.Popen(command, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True, cwd=root)
            write_json(output / 'process.json', dict(pid=process.pid, command=command, started_utc=utc_now()))
            try:
                while process.poll() is None:
                    allowed, current = training_state(lease)
                    eval_pids = {os.getpid(), process.pid, *(p.pid for p in psutil.Process(process.pid).children(recursive=True))}
                    assert_gpus_idle([args.gpu], allowed | eval_pids)
                    memory = gpu_memory(args.gpu)
                    monitor.append(dict(time_utc=utc_now(), training_update=current, **memory))
                    if memory['free_mib'] < lease['minimum_free_mib']:
                        raise RuntimeError('Evaluation stopped to preserve 8 GiB training headroom')
                    if current != progress:
                        progress, last_progress = current, time.monotonic()
                    if time.monotonic() - last_progress > 60:
                        raise RuntimeError('Evaluation stopped because training has not progressed for 60s')
                    if time.monotonic() - started > 900:
                        raise RuntimeError('Evaluation exceeded 900s wall-clock budget')
                    time.sleep(2)
            except BaseException as exc:
                error = repr(exc)
                terminate_child(process)
            finally:
                code = process.wait()
                write_json(output / 'resource_monitor.json', monitor)
                write_json(output / 'exit.json', dict(exit_code=code, error=error,
                                                    finished_utc=utc_now(), elapsed_s=time.monotonic()-started))
        if error:
            raise RuntimeError(error)
        raise SystemExit(code)


if __name__ == '__main__':
    main()
