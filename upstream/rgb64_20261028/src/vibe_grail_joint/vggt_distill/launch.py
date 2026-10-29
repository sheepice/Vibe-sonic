"""GPU-owning launcher: >=2 GPUs per training arm, any of the eight authorized GPUs."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys

from ..gpu_guard import assert_gpus_idle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', required=True)
    p.add_argument('command', choices=('train', 'evaluate', 'teacher_probe', 'feature_probe'))
    args, rest = p.parse_known_args()
    if rest[:1] == ['--']: rest = rest[1:]
    indices = [int(x) for x in args.gpus.split(',')]
    if len(set(indices)) != len(indices) or not set(indices) <= set(range(8)):
        p.error('Select distinct physical GPU indices 0..7')
    if args.command == 'train' and len(indices) not in (2, 4, 8):
        p.error('Training requires at least two GPUs, normally four per arm')
    if args.command != 'train' and len(indices) != 1:
        p.error('Evaluation and teacher probing use one GPU')
    root = Path(__file__).resolve().parents[3]
    locks = []
    child = None
    try:
        for index in sorted(indices):
            lock = (root / 'records' / f'gpu_{index}.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(lock)
        assert_gpus_idle(indices)
        cache = root / 'records' / 'vggt_distill_cache'
        cache.mkdir(exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   VIBE_JOINT_GUARD_PID=str(os.getpid()), OMP_NUM_THREADS='2',
                   HF_HOME=str(root / 'models' / 'hf'), HF_HUB_OFFLINE='1',
                   PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1',
                   TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), WARP_CACHE_PATH=str(cache / 'warp'),
                   MPLCONFIGDIR=str(cache / 'matplotlib'), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        command = [sys.executable]
        if args.command == 'train':
            command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={len(indices)}']
        command += ['-m', f'vibe_grail_joint.vggt_distill.{args.command}', *rest]
        child = subprocess.Popen(command, env=env, cwd=root)
        def forward_signal(signum, _frame):
            if child.poll() is None: child.send_signal(signum)
        signal.signal(signal.SIGTERM, forward_signal)
        signal.signal(signal.SIGINT, forward_signal)
        raise SystemExit(child.wait())
    finally:
        for lock in locks: lock.close()


if __name__ == '__main__': main()
