"""Bounded shared evaluation owned by a verified posture training supervisor.

This explicit launcher preserves the original idle-only GPU guards and the old
shared launcher. It accepts exactly ``velocity.posture_supervise`` as owner,
pins its current stage and eight worker identities, and owns a separate new
session for its evaluation child. A stage transition is an aborted evaluation,
even if the child happened to return zero at the same time.

The shared GPU lock is the SAME lock used by ``shared_evaluation``. Start with
12 GiB free, retain 8 GiB free, cap PyTorch's allocator to 20%, and stop only the
evaluation if all-rank progress stalls for 60 s or evaluation exceeds 900 s.
Unrelated GPU users are never permitted and are never signalled.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import psutil

from ..catalog import write_json
from ..gpu_guard import assert_gpus_idle
from .shared_evaluation import gpu_memory, utc_now


OWNER_MODULE = "vibe_grail_joint.velocity.posture_supervise"
WORKER_MODULE = "vibe_grail_joint.velocity.train"
THIS_MODULE = "vibe_grail_joint.velocity.posture_shared_evaluation"
KINDS = ("uniform", "original", "replay")
LEASE_ENV = "VIBE_POSTURE_SHARED_EVAL_LEASE"
LIMITS = dict(startup_free_mib=12 * 1024, minimum_free_mib=8 * 1024,
              memory_fraction=.20, stall_seconds=60, wall_seconds=900)


def python_module(command):
    """Return the actual first Python execution mode's module, never a substring.

    Only known interpreter switches before -m are accepted. In particular a
    script/-c whose arguments mention a trusted module is not a trusted owner.
    """
    flags = {"-u", "-B", "-E", "-I", "-s", "-S", "-O", "-OO", "-q"}
    if not command or not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(command[0]).name):
        return None
    for index, token in enumerate(command[1:], 1):
        if token == "-m":
            return command[index + 1] if index + 1 < len(command) else None
        if token not in flags:
            return None
    return None


def argument(command, name):
    if command.count(name) != 1:
        raise RuntimeError(f"Expected one unambiguous {name} argument")
    index = command.index(name)
    if index + 1 >= len(command):
        raise RuntimeError(f"Missing value for {name}")
    return command[index + 1]


def live_process(pid, expected_create_time=None):
    try:
        process = psutil.Process(int(pid))
        if not process.is_running() or process.status() in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            raise RuntimeError(f"Process {pid} is not live")
        if expected_create_time is not None:
            if not math.isfinite(expected_create_time) or abs(process.create_time() - expected_create_time) > .01:
                raise RuntimeError(f"Process {pid} creation time changed (PID reuse)")
        return process
    except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
        raise RuntimeError(f"Cannot verify live process {pid}") from error


def resolved_argument(process, name):
    return (Path(process.cwd()) / argument(process.cmdline(), name)).resolve()


def training_state(lease, *, pin_workers=False):
    """Verify owner, pinned stage, statuses, exact modules and eight rank PIDs.

    pin_workers=True is used once by the launcher to capture identities; every
    subsequent launcher/worker check requires these same creation-time pairs.
    Only the ancestors/descendants of verified training workers are allowed.
    An unrelated sibling child of the supervisor is not implicitly allowlisted.
    """
    run = Path(lease["training_run"]).resolve()
    state = json.loads((run / "supervisor_status.json").read_text())
    if state.get("status") != "training" or state.get("pid") != lease["training_pid"]:
        raise RuntimeError("The identified training stage is no longer active")
    if state.get("world_size") != 8 or state.get("run_kind") != "full_8gpu":
        raise RuntimeError("Sharing requires the full eight-GPU posture run")
    if abs(state.get("process_create_time", -1) - lease["training_create_time"]) > .01:
        raise RuntimeError("Supervisor recorded creation time changed")
    owner = live_process(lease["training_pid"], lease["training_create_time"])
    if python_module(owner.cmdline()) != OWNER_MODULE:
        raise RuntimeError("Owner must execute exactly the posture supervisor module")
    if resolved_argument(owner, "--run-dir") != run:
        raise RuntimeError("Supervisor belongs to another run directory")
    active = Path(state["active_directory"]).resolve()
    if not active.is_relative_to(run) or active == run:
        raise RuntimeError("Active training stage is outside the declared run")
    if active != Path(lease["training_stage"]).resolve() or state.get("target_updates") != lease["target_updates"]:
        raise RuntimeError("Pinned training stage changed; shared evaluation must stop")
    ranks = [json.loads((active / f"rank_{rank}_status.json").read_text()) for rank in range(8)]
    for index, row in enumerate(ranks):
        if (row.get("rank") != index or row.get("world_size") != 8 or row.get("status") != "running"
                or row.get("smoke") is not False or row.get("diagnostic") is not False):
            raise RuntimeError(f"Rank {index} is not a live full-training rank")
        if row.get("num_envs") != state.get("num_envs_per_rank"):
            raise RuntimeError(f"Rank {index} environment count differs from supervisor")
        progress = row.get("absolute_completed_updates")
        if not isinstance(progress, int) or not 0 <= progress < lease["target_updates"]:
            raise RuntimeError(f"Rank {index} is at a stage boundary or has invalid progress")
    workers = []
    for child in owner.children(recursive=True):
        child = live_process(child.pid)
        if python_module(child.cmdline()) == WORKER_MODULE:
            workers.append(child)
    if len(workers) != 8:
        raise RuntimeError(f"Expected exactly eight live training workers, found {len(workers)}")
    identities, allowed = {}, {owner.pid}
    for worker in workers:
        if resolved_argument(worker, "--log-dir") != active:
            raise RuntimeError("Training worker belongs to another stage")
        if resolved_argument(worker, "--config") != run / "config.json":
            raise RuntimeError("Training worker belongs to another configuration")
        env = worker.environ()
        try:
            rank, local, size = (int(env[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
        except (KeyError, ValueError) as error:
            raise RuntimeError("Training worker rank environment is invalid") from error
        if size != 8 or rank != local or rank not in range(8) or str(rank) in identities:
            raise RuntimeError("Training workers do not map one-to-one to ranks 0..7")
        identity = {"pid": worker.pid, "create_time": worker.create_time()}
        if not pin_workers:
            expected = lease["training_workers"].get(str(rank))
            if (expected is None or expected["pid"] != worker.pid
                    or abs(expected["create_time"] - worker.create_time()) > .01):
                raise RuntimeError(f"Training rank {rank} process identity changed")
        identities[str(rank)] = identity
        ancestors = worker.parents()
        if owner.pid not in [p.pid for p in ancestors]:
            raise RuntimeError("Training worker left the supervisor process tree")
        allowed.add(worker.pid)
        for parent in ancestors:
            parent = live_process(parent.pid)
            allowed.add(parent.pid)
            if parent.pid == owner.pid:
                break
        allowed.update(live_process(child.pid).pid for child in worker.children(recursive=True))
    if not pin_workers and set(lease["training_workers"]) != {str(i) for i in range(8)}:
        raise RuntimeError("Lease must bind exactly eight training worker identities")
    return allowed, min(row["absolute_completed_updates"] for row in ranks), identities


def validate_lease(lease):
    if lease.get("schema_version") != 1 or lease.get("module") != THIS_MODULE or lease.get("kind") not in KINDS:
        raise RuntimeError("Invalid posture shared-evaluation lease")
    if any(lease.get(key) != value for key, value in LIMITS.items()):
        raise RuntimeError("Shared evaluation resource limits cannot be relaxed")


def evaluation_identity(process, launcher_pid, output):
    return {"pid": process.pid, "create_time": process.create_time(),
            "launcher_pid": launcher_pid, "output": str(Path(output).resolve())}


def owned_evaluation_process(identity):
    process = live_process(identity["pid"], identity["create_time"])
    if process.ppid() != identity["launcher_pid"]:
        raise RuntimeError("Evaluation is not a direct child of its verified launcher")
    command = process.cmdline()
    if python_module(command) != THIS_MODULE or argument(command, "--worker") not in KINDS:
        raise RuntimeError("Evaluation process is not this launcher's worker module")
    if resolved_argument(process, "--output") != Path(identity["output"]).resolve():
        raise RuntimeError("Evaluation worker output differs from its lease")
    if os.getpgid(process.pid) != process.pid or os.getsid(process.pid) != process.pid:
        raise RuntimeError("Evaluation worker must own a separate session and process group")
    return process


def verify_worker_identity(lease):
    """Pure process/file ownership check, called before any CUDA initialization."""
    validate_lease(lease)
    launcher = live_process(lease["evaluation_launcher_pid"], lease["evaluation_launcher_create_time"])
    command = launcher.cmdline()
    if python_module(command) != THIS_MODULE or "--worker" in command:
        raise RuntimeError("Shared worker parent is not this verified launcher")
    if argument(command, "--kind") != lease["kind"]:
        raise RuntimeError("Launcher evaluation kind differs from lease")
    if int(argument(command, "--gpu")) != lease["gpu"]:
        raise RuntimeError("Launcher GPU differs from lease")
    if resolved_argument(launcher, "--training-run") != Path(lease["training_run"]).resolve():
        raise RuntimeError("Launcher training run differs from lease")
    if resolved_argument(launcher, "--output") != Path(lease["output"]).resolve():
        raise RuntimeError("Launcher output differs from lease")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(lease["gpu"]):
        raise RuntimeError("Evaluation GPU differs from lease")
    # The launcher writes this immediately after Popen. A brief startup handshake
    # avoids racing that write; an expired handshake never authorizes the worker.
    record_path = Path(lease["output"]) / "process.json"
    deadline = time.monotonic() + 5
    while not record_path.exists():
        live_process(launcher.pid, lease["evaluation_launcher_create_time"])
        if time.monotonic() >= deadline:
            raise RuntimeError("Launcher did not record the owned evaluation child")
        time.sleep(.05)
    identity = json.loads(record_path.read_text())["identity"]
    if identity["pid"] != os.getpid() or identity["launcher_pid"] != launcher.pid:
        raise RuntimeError("Current process is not the recorded evaluation worker")
    worker = owned_evaluation_process(identity)
    if argument(worker.cmdline(), "--worker") != lease["kind"]:
        raise RuntimeError("Worker evaluation kind differs from lease")
    training_pids, _, _ = training_state(lease)
    evaluation_pids = {worker.pid, *(live_process(p.pid).pid for p in worker.children(recursive=True))}
    return training_pids | evaluation_pids


def verify_shared_worker():
    lease = json.loads(Path(os.environ[LEASE_ENV]).read_text())
    allowed = verify_worker_identity(lease)
    assert_gpus_idle([lease["gpu"]], allowed)
    if gpu_memory(lease["gpu"])["free_mib"] < LIMITS["startup_free_mib"]:
        raise RuntimeError("Insufficient 12 GiB startup headroom")
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(LIMITS["memory_fraction"], 0)


def terminate_owned_child(process, identity, protected_pids):
    """Signal only the direct child/session created by this live launcher."""
    if process.poll() is not None:
        return
    if identity["launcher_pid"] != os.getpid() or process.pid != identity["pid"]:
        raise RuntimeError("Refusing to terminate a process not created by this launcher")
    owned = owned_evaluation_process(identity)
    members = {owned.pid, *(p.pid for p in owned.children(recursive=True))}
    if members & set(protected_pids):
        raise RuntimeError("Refusing to signal any training-owned process")
    os.killpg(owned.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # Re-check creation time, parent, module and session before escalation.
        owned_evaluation_process(identity)
        os.killpg(owned.pid, signal.SIGKILL)
        process.wait(timeout=10)


def check_limits(lease, memory, current, progress, last_progress, started, now):
    if memory["free_mib"] < LIMITS["minimum_free_mib"]:
        raise RuntimeError("Evaluation stopped to preserve 8 GiB training headroom")
    if current < progress:
        raise RuntimeError("Training progress moved backwards")
    if current > progress:
        progress, last_progress = current, now
    if now - last_progress >= LIMITS["stall_seconds"]:
        raise RuntimeError("Evaluation stopped: all-rank training progress stalled for 60s")
    if now - started >= LIMITS["wall_seconds"]:
        raise RuntimeError("Evaluation exceeded the 900s wall-clock budget")
    return progress, last_progress


def monitor_evaluation(process, identity, lease, initial_progress):
    started = last_progress = time.monotonic()
    progress, monitor, error = initial_progress, [], None
    protected = {lease["training_pid"], *(row["pid"] for row in lease["training_workers"].values())}
    try:
        while process.poll() is None:
            training_pids, current, _ = training_state(lease)
            protected.update(training_pids)
            child = owned_evaluation_process(identity)
            evaluation_pids = {child.pid, *(live_process(p.pid).pid for p in child.children(recursive=True))}
            assert_gpus_idle([lease["gpu"]], training_pids | evaluation_pids)
            memory = gpu_memory(lease["gpu"])
            monitor.append(dict(time_utc=utc_now(), training_update=current, **memory))
            progress, last_progress = check_limits(lease, memory, current, progress, last_progress,
                                                   started, time.monotonic())
            time.sleep(2)
        # An exit-zero racing a stage transition is not accepted as completed.
        training_state(lease)
    except BaseException as exc:
        error = repr(exc)
        try:
            terminate_owned_child(process, identity, protected)
        except BaseException as cleanup_error:
            error += f"; safe cleanup refused/failed: {cleanup_error!r}"
    code = process.poll()
    if code is None:
        # Identity failure must never turn into an unbounded wait or broaden the
        # set of signalled processes. Record a non-successful, unverified child.
        error = (error or "") + "; evaluation process has not been verified stopped"
    success = code == 0 and error is None
    return monitor, dict(exit_code=code, error=error, evaluation_completed=success,
                         status="completed" if success else "aborted_or_failed",
                         finished_utc=utc_now(), elapsed_s=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=KINDS)
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--training-run")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--output", required=True)
    args, rest = parser.parse_known_args()
    if args.worker:
        if args.kind or args.training_run:
            parser.error("Worker mode cannot also act as a launcher")
        sys.argv = [sys.argv[0], "--output", args.output, *rest]
        if args.worker == "original":
            from .evaluate_original_stair import main as evaluate
        elif args.worker == "uniform":
            from .evaluate import main as evaluate
        else:
            from .evaluate_replay import main as evaluate
        evaluate(guard_check=verify_shared_worker)
        return 0
    if not args.kind or not args.training_run:
        parser.error("--kind and --training-run are required")
    # Require an explicit GPU token so the worker can bind it to the parent's
    # real command line instead of trusting a mutable lease default.
    argument(sys.argv, "--gpu")
    root = Path(__file__).resolve().parents[3]
    output, run = Path(args.output).resolve(), Path(args.training_run).resolve()
    output.mkdir(parents=True, exist_ok=False)
    monitor, result = [], None
    try:
        state = json.loads((run / "supervisor_status.json").read_text())
        owner, launcher = live_process(state["pid"]), live_process(os.getpid())
        lease = dict(schema_version=1, module=THIS_MODULE, kind=args.kind, gpu=args.gpu,
                     training_run=str(run), training_stage=str(Path(state["active_directory"]).resolve()),
                     target_updates=state["target_updates"], training_pid=owner.pid,
                     training_create_time=owner.create_time(), evaluation_launcher_pid=launcher.pid,
                     evaluation_launcher_create_time=launcher.create_time(), output=str(output),
                     created_utc=utc_now(), **LIMITS)
        lease_path = output / "shared_lease.json"
        # Deliberately shared with the previous implementation, not a new lock.
        with (root / "records" / f"shared_evaluation_gpu_{args.gpu}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            allowed, progress, identities = training_state(lease, pin_workers=True)
            lease["training_workers"] = identities
            validate_lease(lease)
            write_json(lease_path, lease)
            assert_gpus_idle([args.gpu], allowed)
            if gpu_memory(args.gpu)["free_mib"] < LIMITS["startup_free_mib"]:
                raise RuntimeError("Insufficient 12 GiB startup headroom")
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
                               OMP_NUM_THREADS="2", HF_HOME=str(root / "models/hf"), HF_HUB_OFFLINE="1",
                               PYTHONUNBUFFERED="1", **{LEASE_ENV: str(lease_path)})
            command = [sys.executable, "-m", THIS_MODULE, "--worker", args.kind, "--output", str(output), *rest]
            with (output / "console.log").open("w") as log:
                process, identity = None, None
                try:
                    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True, cwd=root)
                    identity = evaluation_identity(psutil.Process(process.pid), os.getpid(), output)
                    write_json(output / "process.json", dict(identity=identity, pid=process.pid, command=command,
                                                             started_utc=utc_now()))
                    monitor, result = monitor_evaluation(process, identity, lease, progress)
                finally:
                    # Also cover interruption/error after Popen but before the
                    # monitoring loop. No Popen handle is ever used as authority
                    # to signal a reused PID without its checked identity.
                    if process is not None and process.poll() is None and identity is not None:
                        protected = {lease["training_pid"], *(row["pid"] for row in identities.values())}
                        terminate_owned_child(process, identity, protected)
    except BaseException as error:
        result = dict(exit_code=None, error=repr(error), evaluation_completed=False,
                      status="startup_rejected", finished_utc=utc_now())
    finally:
        write_json(output / "resource_monitor.json", monitor)
        write_json(output / "exit.json", result)
    if not result["evaluation_completed"]:
        print(result["error"] or f"Evaluation worker exited {result['exit_code']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
