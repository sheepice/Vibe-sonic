"""Bounded evaluation beside an identified direct eight-rank training launcher.

Uses the established shared-evaluation memory limits and lock. Training and
idle-only launchers remain unchanged. Only our own evaluation session is ever
signalled. All training worker identities are pinned before starting CUDA.
"""

import argparse
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
from ..gpu_guard import assert_gpus_idle
from .posture_shared_evaluation import argument, live_process, python_module, resolved_argument
from .shared_evaluation import gpu_memory, utc_now

MODULE = "vibe_grail_joint.velocity.closeup_shared_evaluation"
OWNER = "vibe_grail_joint.velocity.launch"
WORKER = "vibe_grail_joint.velocity.train"
LEASE_ENV = "VIBE_DIRECT_SHARED_EVAL_LEASE"


def training_state(lease, pin=False):
    run = Path(lease["training_run"]).resolve()
    owner = live_process(lease["training_pid"], lease["training_create_time"])
    if python_module(owner.cmdline()) != OWNER or resolved_argument(owner, "--log-dir") != run:
        raise RuntimeError("Training owner or run differs from the pinned direct launcher")
    workers = [p for p in owner.children(recursive=True) if python_module(p.cmdline()) == WORKER]
    if len(workers) != 8:
        raise RuntimeError("Expected exactly eight live training workers")
    identities, allowed, progress = {}, {owner.pid}, []
    config = resolved_argument(owner, "--config")
    for worker in workers:
        live_process(worker.pid)
        if resolved_argument(worker, "--log-dir") != run or resolved_argument(worker, "--config") != config:
            raise RuntimeError("Training worker run/config mismatch")
        env = worker.environ()
        rank = int(env["RANK"])
        if rank not in range(8) or rank in identities or int(env["WORLD_SIZE"]) != 8:
            raise RuntimeError("Invalid or duplicate training rank")
        state = json.loads((run / f"rank_{rank}_status.json").read_text())
        if state["status"] != "running" or state["rank"] != rank or state["world_size"] != 8:
            raise RuntimeError("Training rank is not healthy")
        if state.get("smoke") or state.get("diagnostic"):
            raise RuntimeError("Expected the full training job")
        identities[rank] = {"pid": worker.pid, "create_time": worker.create_time()}
        progress.append(state["absolute_completed_updates"])
        allowed.add(worker.pid)
        # Allow only ancestors connecting known workers to this launcher.
        for parent in worker.parents():
            allowed.add(parent.pid)
            if parent.pid == owner.pid:
                break
        else:
            raise RuntimeError("Worker is outside the training process tree")
    identities = {str(k): v for k, v in identities.items()}
    if not pin and identities != lease["training_workers"]:
        raise RuntimeError("A pinned training worker was replaced")
    return allowed, min(progress), identities


def verify_shared_worker():
    lease = json.loads(Path(os.environ[LEASE_ENV]).read_text())
    launcher = live_process(lease["evaluation_launcher_pid"], lease["evaluation_launcher_create_time"])
    me = psutil.Process()
    if me.ppid() != launcher.pid or python_module(launcher.cmdline()) != MODULE:
        raise RuntimeError("Evaluation is not the direct child of its pinned launcher")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(lease["gpu"]):
        raise RuntimeError("Evaluation GPU differs from the lease")
    allowed, _, _ = training_state(lease)
    assert_gpus_idle([lease["gpu"]], allowed | {launcher.pid, me.pid})
    if gpu_memory(lease["gpu"])["free_mib"] < 12 * 1024:
        raise RuntimeError("Evaluation needs 12 GiB of startup headroom")
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.20, 0)


def terminate_owned_child(process, created, protected):
    if process.poll() is not None:
        return
    try:
        worker = live_process(process.pid, created)
    except RuntimeError:
        # Reap our own child if it finished between poll() and the liveness check.
        if process.poll() is not None:
            return
        raise
    if (worker.ppid() != os.getpid() or python_module(worker.cmdline()) != MODULE
            or "--worker" not in worker.cmdline()
            or os.getpgid(worker.pid) != worker.pid or os.getsid(worker.pid) != worker.pid):
        raise RuntimeError("Refusing to signal an unowned evaluation session")
    members = {worker.pid, *(p.pid for p in worker.children(recursive=True))}
    if members & protected:
        raise RuntimeError("Refusing to signal training")
    os.killpg(worker.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        live_process(worker.pid, created)
        os.killpg(worker.pid, signal.SIGKILL)
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--training-pid", type=int)
    parser.add_argument("--training-run")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--output", required=True)
    args, rest = parser.parse_known_args()
    if args.worker:
        sys.argv = [sys.argv[0], "--output", args.output, *rest]
        from .evaluate_uniform20_closeup import main as evaluate
        return evaluate(guard_check=verify_shared_worker)
    if not args.training_pid or not args.training_run:
        parser.error("--training-pid and --training-run are required")
    root = Path(__file__).resolve().parents[3]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    owner = live_process(args.training_pid)
    me = psutil.Process()
    lease = dict(training_run=str(Path(args.training_run).resolve()),
                 training_pid=owner.pid, training_create_time=owner.create_time(),
                 evaluation_launcher_pid=me.pid, evaluation_launcher_create_time=me.create_time(),
                 gpu=args.gpu, created_utc=utc_now())
    protected, progress, identities = training_state(lease, pin=True)
    lease["training_workers"] = identities
    with (root / "records" / f"shared_evaluation_gpu_{args.gpu}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_gpus_idle([args.gpu], protected)
        if gpu_memory(args.gpu)["free_mib"] < 12 * 1024:
            raise RuntimeError("Insufficient startup headroom")
        lease_path = out / "shared_lease.json"
        write_json(lease_path, lease)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                           CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="2",
                           HF_HOME=str(root / "models/hf"), HF_HUB_OFFLINE="1",
                           PYTHONUNBUFFERED="1", **{LEASE_ENV: str(lease_path)})
        command = [sys.executable, "-m", MODULE, "--worker", "--output", str(out), *rest]
        monitor, error = [], None
        started = last_progress = time.monotonic()
        with (out / "console.log").open("w") as log:
            process = subprocess.Popen(command, env=environment, cwd=root, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            created = psutil.Process(process.pid).create_time()
            write_json(out / "process.json", dict(pid=process.pid, create_time=created,
                                                 launcher_pid=os.getpid(), command=command))
            try:
                while process.poll() is None:
                    protected, current, _ = training_state(lease)
                    try:
                        worker = live_process(process.pid, created)
                    except RuntimeError:
                        # Normal completion can race this monitor's poll(). The
                        # actual exit code is still collected below.
                        if process.poll() is not None:
                            break
                        raise
                    eval_pids = {os.getpid(), worker.pid, *(p.pid for p in worker.children(recursive=True))}
                    assert_gpus_idle([args.gpu], protected | eval_pids)
                    memory = gpu_memory(args.gpu)
                    monitor.append(dict(time_utc=utc_now(), training_update=current, **memory))
                    if memory["free_mib"] < 8 * 1024:
                        raise RuntimeError("Stopping evaluation to retain 8 GiB training headroom")
                    now = time.monotonic()
                    if current != progress:
                        progress, last_progress = current, now
                    if now - last_progress > 60:
                        raise RuntimeError("Stopping evaluation: training has not progressed for 60s")
                    if now - started > 900:
                        raise RuntimeError("Evaluation exceeded 900s")
                    time.sleep(2)
            except BaseException as exc:
                error = repr(exc)
                terminate_owned_child(process, created, protected)
            code = process.wait()
        write_json(out / "resource_monitor.json", monitor)
        write_json(out / "exit.json", dict(exit_code=code, error=error,
                                          elapsed_s=time.monotonic()-started, finished_utc=utc_now()))
        if error:
            raise RuntimeError(error)
        raise SystemExit(code)


if __name__ == "__main__":
    main()
