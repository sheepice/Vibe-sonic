"""Single-GPU ownership guard for the unchanged original stair benchmark."""

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from ..gpu_guard import assert_gpus_idle


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    args, rest = p.parse_known_args()
    root = Path(__file__).resolve().parents[3]
    with (root / 'records' / f'gpu_{args.gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_gpus_idle([args.gpu])
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                           VIBE_JOINT_GUARD_PID=str(os.getpid()), OMP_NUM_THREADS='2',
                           HF_HOME=str(root / 'models' / 'hf'), HF_HUB_OFFLINE='1', PYTHONUNBUFFERED='1')
        raise SystemExit(subprocess.call([sys.executable, '-m',
                                         'vibe_grail_joint.velocity.evaluate_original_stair', *rest], env=environment))


if __name__ == '__main__':
    main()
