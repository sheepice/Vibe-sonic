"""Guarded launcher for the independent stair-only training and evaluation tasks."""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
from pathlib import Path

from .gpu_guard import assert_gpus_idle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--entrypoint", choices=("train_stairs", "evaluate_stairs"), default="train_stairs")
    args, train_args = parser.parse_known_args()
    indices = [int(value) for value in args.gpus.split(",")]
    root = Path(__file__).resolve().parents[2]
    (root / "records").mkdir(exist_ok=True)
    locks = []
    try:
        for index in sorted(indices):
            lock = (root / "records" / f"gpu_{index}.lock").open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        assert_gpus_idle(indices)
        print(f"Selected GPUs are idle: {indices}", flush=True)
        if args.check_only:
            return
        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=args.gpus,
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            VIBE_JOINT_GUARD_PID=str(os.getpid()),
            OMP_NUM_THREADS="2",
            HF_HOME=str(root / "models" / "hf"),
            HF_HUB_OFFLINE="1",
        )
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(indices)}",
            "-m",
            f"vibe_grail_joint.{args.entrypoint}",
            *train_args,
        ]
        raise SystemExit(subprocess.call(command, env=environment))
    except (RuntimeError, subprocess.SubprocessError, OSError) as error:
        parser.exit(2, f"Launch refused: {error}\n")
    finally:
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    main()
