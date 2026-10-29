"""Exclusive GPU 4–7 launcher. Never terminates an existing task."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

ALLOWED = {'collect', 'fit', 'train', 'evaluate', 'validate', 'simcheck'}


def check_indices(indices):
    if not indices or len(set(indices)) != len(indices) or not set(indices) <= {4, 5, 6, 7}:
        raise ValueError('Only distinct physical GPUs 4,5,6,7 are permitted')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus', required=True)
    p.add_argument('command', choices=sorted(ALLOWED))
    args, rest = p.parse_known_args()
    if rest[:1] == ['--']:
        rest = rest[1:]
    indices = [int(v) for v in args.gpus.split(',')]; check_indices(indices)
    if args.command != 'train' and len(indices) != 1:
        p.error('Collection, supervised fitting and evaluation use one GPU per job')
    from ..gpu_guard import assert_gpus_idle
    root = Path(__file__).resolve().parents[3]
    locks = []
    try:
        for i in sorted(indices):
            lock = (root/'records'/f'gpu_{i}.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(lock)
        assert_gpus_idle(indices)
        cache = root/'records'/'hiking_perception_cache'; cache.mkdir(exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   VIBE_JOINT_GUARD_PID=str(os.getpid()), OMP_NUM_THREADS='2',
                   HF_HOME=str(root/'models'/'hf'), HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1',
                   PYTHONDONTWRITEBYTECODE='1', TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
                   WARP_CACHE_PATH=str(cache/'warp'), MPLCONFIGDIR=str(cache/'matplotlib'),
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        command = [sys.executable]
        if args.command == 'train':
            command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={len(indices)}']
        command += ['-m', f'vibe_grail_joint.hiking_perception.{args.command}', *rest]
        raise SystemExit(subprocess.call(command, env=env, cwd=root))
    finally:
        for lock in locks:
            lock.close()


if __name__ == '__main__':
    main()
