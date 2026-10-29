"""Refuse occupied GPUs without ever signalling or changing another process."""

from __future__ import annotations

import csv
import io
import os
import subprocess


def query(fields: str, kind: str):
    result = subprocess.run(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        timeout=20,
        check=True,
    )
    return [[cell.strip() for cell in row] for row in csv.reader(io.StringIO(result.stdout)) if row]


def assert_gpus_idle(indices: list[int], allowed_pids=()):
    devices = {int(row[0]): row[1] for row in query("index,uuid", "gpu")}
    if len(indices) != len(set(indices)) or any(index not in devices for index in indices):
        raise RuntimeError(f"Invalid GPU selection: {indices}")
    selected = {devices[index] for index in indices}
    # Some driver versions report invalid aggregate memory.used. Process ownership is decisive.
    occupied = [
        row
        for row in query("gpu_uuid,pid,process_name,used_gpu_memory", "compute-apps")
        if row[0] in selected and int(row[1]) not in allowed_pids
    ]
    if occupied:
        raise RuntimeError(
            "Selected GPUs are occupied; existing jobs are left untouched:\n"
            + "\n".join(", ".join(row) for row in occupied)
        )
    return devices


def verify_guarded_worker():
    import psutil

    guard_pid = int(os.environ.get("VIBE_JOINT_GUARD_PID", "0"))
    if guard_pid not in [parent.pid for parent in psutil.Process().parents()]:
        raise RuntimeError("Start through python -m vibe_grail_joint.launch so GPU ownership is checked")
    parent = psutil.Process(guard_pid)
    allowed = {guard_pid, *(process.pid for process in parent.children(recursive=True))}
    indices = [int(value) for value in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    assert_gpus_idle(indices, allowed)
