"""GPU-ownership guarded launcher for the independent velocity experiment."""

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path

from ..gpu_guard import assert_gpus_idle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args, rest = p.parse_known_args()
    root = Path(__file__).resolve().parents[3]
    indices = [int(i) for i in args.gpus.split(",")]
    locks = []
    try:
        for i in sorted(indices):
            lock = (root / "records" / f"gpu_{i}.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        assert_gpus_idle(indices)
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=args.gpus,
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            VIBE_JOINT_GUARD_PID=str(os.getpid()),
            OMP_NUM_THREADS="2",
            HF_HOME=str(root / "models" / "hf"),
            HF_HUB_OFFLINE="1",
            PYTHONUNBUFFERED="1",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(indices)}",
            "-m",
            "vibe_grail_joint.velocity.train",
            *rest,
        ]
        raise SystemExit(subprocess.call(command, env=environment))
    finally:
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    main()
