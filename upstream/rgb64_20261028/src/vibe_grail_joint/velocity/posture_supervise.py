"""Eight-GPU posture refinement with explicit stage feedback and owned processes.

Waiting never signals the previous run. GPU ownership is acquired exclusively
by the existing guarded launchers. Only a child process group created here may
be stopped; automatic retries are restricted to explicit CUDA OOM failures.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

from ..catalog import write_json
from .supervise import now, run_monitored, stop_owned_process

_OWNER_MODULES = {
    "vibe_grail_joint.velocity.supervise",
    "vibe_grail_joint.velocity.posture_supervise",
}
_OOM = re.compile(
    r"(?:cuda out of memory|(?:torch\.(?:cuda\.)?)?OutOfMemoryError|CUDA_ERROR_OUT_OF_MEMORY)", re.IGNORECASE
)


def read_json(path):
    return json.loads(Path(path).read_text())


def nonfinite_paths(value, prefix=""):
    """Find actual NaN/Inf numbers, without interpreting text or optional nulls."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.extend(nonfinite_paths(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(nonfinite_paths(item, f"{prefix}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        found.append(prefix)
    return found


def prior_identity(run, state):
    """Bind a live supervisor by PID, creation time, module and resolved run path."""
    try:
        owner = psutil.Process(int(state["pid"]))
        if not owner.is_running() or owner.status() == psutil.STATUS_ZOMBIE:
            return None
        command = owner.cmdline()
        module = command[command.index("-m") + 1] if "-m" in command else None
        if module not in _OWNER_MODULES or "--run-dir" not in command:
            raise RuntimeError("Recorded prior PID belongs to another command; refusing PID reuse")
        directory = (Path(owner.cwd()) / command[command.index("--run-dir") + 1]).resolve()
        if directory != run:
            raise RuntimeError("Prior supervisor command belongs to another run directory")
        created = owner.create_time()
        recorded_create = state.get("process_create_time")
        if recorded_create is not None and abs(created - recorded_create) > 0.01:
            raise RuntimeError("Prior supervisor creation time changed")
        if recorded_create is None:
            # The old supervisor predates explicit creation-time recording.
            started = datetime.datetime.fromisoformat(state["started"]).timestamp()
            if abs(created - started) > 120:
                raise RuntimeError("Prior process creation time does not match its recorded start")
        return {"pid": owner.pid, "process_create_time": created, "command": command, "run": str(run)}
    except psutil.NoSuchProcess:
        return None


def wait_for_prior(run, status, status_path, poll_seconds=5):
    run = Path(run).resolve()
    state = read_json(run / "supervisor_status.json")
    identity = prior_identity(run, state)
    while identity is not None:
        current = prior_identity(run, state)
        if current is None:
            break
        if current != identity:
            raise RuntimeError("Prior supervisor identity changed while waiting")
        status.update(
            status="waiting_prior",
            prior_run=str(run),
            prior_process=identity,
            prior_status=state["status"],
            prior_verified_alive_utc=now(),
            updated=now(),
        )
        write_json(status_path, status)
        time.sleep(poll_seconds)
        state = read_json(run / "supervisor_status.json")
    # A stale status file, missing PID or observation timeout is never treated as
    # successful training. Read the terminal state again after verified exit.
    state = read_json(run / "supervisor_status.json")
    if state.get("status") != "completed":
        raise RuntimeError(f"Prior supervisor exited without completed status: {state.get('status')}")
    status.update(prior_run=str(run), prior_terminal_status=state["status"], prior_wait_finished=now())
    write_json(status_path, status)


def run_training(command, directory, environment, start, on_progress):
    """Full eight-rank monitor; resumed startup gets the full startup deadline."""
    directory.mkdir(parents=True, exist_ok=False)
    failure = None
    progress, first_update = start, False
    started = last_progress = time.monotonic()
    with (directory / "console.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, env=environment, start_new_session=True
        )
        write_json(
            directory / "process.json",
            {
                "pid": process.pid,
                "command": command,
                "started": now(),
                "run_kind": "full_8gpu",
                "expected_ranks": 8,
            },
        )
        try:
            while process.poll() is None:
                ranks = []
                for rank in range(8):
                    path = directory / f"rank_{rank}_status.json"
                    if path.exists():
                        ranks.append(read_json(path))
                if any(r.get("smoke") or r.get("diagnostic") or r.get("world_size") != 8 for r in ranks):
                    raise RuntimeError("Full stage received smoke/diagnostic or non-eight-rank worker status")
                failed = [r for r in ranks if r.get("status") == "failed"]
                if failed:
                    raise RuntimeError("; ".join(f"rank {r['rank']}: {r.get('error')}" for r in failed))
                for rank in ranks:
                    invalid = nonfinite_paths({k: rank.get(k) for k in ("loss", "metrics", "learning_rate")})
                    if invalid:
                        raise RuntimeError(f"Numerically invalid training rank {rank['rank']}: {invalid}")
                if len(ranks) == 8:
                    current = min(r.get("absolute_completed_updates", start) for r in ranks)
                    if current > progress:
                        progress, last_progress = current, time.monotonic()
                        first_update = True
                    on_progress(progress, len(ranks))
                elapsed = time.monotonic() - (last_progress if first_update else started)
                deadline = 240 if first_update else 600
                if elapsed > deadline:
                    raise RuntimeError(f"No all-eight-rank progress for {deadline}s at update {progress}")
                time.sleep(5)
        except BaseException as error:
            failure = repr(error)
            (directory / "monitor_traceback.txt").write_text(traceback.format_exc())
            stop_owned_process(process)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            code = process.wait()
            write_json(
                directory / "exit.json",
                {
                    "exit_code": code,
                    "failure": failure,
                    "finished": now(),
                    "minimum_observed_updates": progress,
                },
            )
    return code, failure


def summarize_evaluation(directory, kind, ablation="none"):
    """Keep failures and pooled valid-sample posture values visible together."""
    report = read_json(directory / "report.json")
    result = {
        "kind": kind,
        "visual_ablation": ablation,
        "directory": str(directory),
        "numerical_errors": nonfinite_paths(report),
        "failures": [],
    }
    if kind == "original":
        online = report["online_w"]
        result.update(online_w=online, reference_control=report["reference_control"])
        if not online["success"]:
            result["failures"].append({"kind": "original_stair", "reason": online["reason"]})
        if "nonfinite" in online.get("reason", ""):
            result["numerical_errors"].append("online_w.nonfinite_state")
        return result
    cases = report["cases"]
    if not cases:
        raise ValueError("Evaluation has no cases")
    stairs = [row for row in cases if row["terrain"]["kind"] != "flat"]
    if not stairs:
        raise ValueError("Evaluation has no stair cases")
    result.update(
        total=len(cases),
        success=sum(bool(row["success"]) for row in cases),
        stairs_total=len(stairs),
        stairs_success=sum(bool(row["success"]) for row in stairs),
        groups=report["groups"],
    )
    result["stairs_success_rate"] = result["stairs_success"] / result["stairs_total"]
    for row in cases:
        if not row["success"]:
            result["failures"].append(
                {
                    "case_id": row["case"]["id"],
                    "kind": row["terrain"]["kind"],
                    "level": row["terrain"]["level"],
                    "reason": row["reason"],
                }
            )
        if "nonfinite" in row["reason"]:
            result["numerical_errors"].append(f"case_{row['case']['id']}.nonfinite_state")
    try:
        posture = read_json(directory / "posture_summary.json")
        pooled = posture["by_stage"]["all"]
        if pooled["sample_count"] <= 0 or len(posture["per_env"]) != len(cases):
            raise ValueError("Posture summary does not cover the complete evaluation batch")
    except (OSError, ValueError, KeyError) as error:
        # Crossing collapse and nonfinite-state evidence remain actionable even
        # when a diagnostics artifact is missing or malformed.
        result["infrastructure_error"] = f"Posture diagnostics unavailable: {error!r}"
        return result
    result["posture_all_valid_samples"] = pooled
    result["posture_by_stage"] = posture["by_stage"]
    result["posture_aggregation"] = posture["aggregation"]
    result["numerical_errors"] += nonfinite_paths(posture)
    for metric, values in pooled["metrics"].items():
        if values["invalid_count"] or values["count"] <= 0:
            result["numerical_errors"].append(f"posture.{metric}: invalid_count={values['invalid_count']}")
    for row in posture["per_env"]:
        if row["sample_count"] <= 0 or any(value is None for value in row["endpoint"].values()):
            result["numerical_errors"].append(f"posture.env_{row['env_index']}.invalid_endpoint")
    return result


def evaluation_commands(root, config, settings, checkpoint, target, out, zero_tokens):
    specs = []
    for kind, cfg, seed, duration in (
        ("uniform", root / "configs/online_velocity.json", 170017, 30),
        ("irregular", config, 170019, 45),
    ):
        directory = out / f"evaluate_{kind}_{target:05d}"
        command = [
            sys.executable,
            "-m",
            "vibe_grail_joint.velocity.launch_eval",
            "--gpu",
            "0",
            "--config",
            str(cfg),
            "--checkpoint",
            checkpoint,
            "--output",
            str(directory),
            "--seed",
            str(seed),
            "--duration",
            str(duration),
            "--posture-diagnostics",
            "--label",
            f"Posture ViBe | {target} refinement updates",
        ]
        if kind == "irregular":
            command.append("--irregular")
        specs.append((kind, "none", directory, command))
        if kind == "irregular" and zero_tokens and target >= 500:
            blind_directory = out / f"evaluate_irregular_{target:05d}_zero_tokens"
            blind_command = list(command)
            blind_command[blind_command.index("--output") + 1] = str(blind_directory)
            blind_command += ["--visual-ablation", "zero_tokens"]
            specs.append((kind, "zero_tokens", blind_directory, blind_command))
    directory = out / f"evaluate_original_{target:05d}"
    command = [
        sys.executable,
        "-m",
        "vibe_grail_joint.velocity.launch_original_stair",
        "--gpu",
        "0",
        "--bank",
        str(root / "data/stairs_validation/rank_0"),
        "--checkpoint",
        checkpoint,
        "--base-checkpoint",
        settings["reference_checkpoint"],
        "--planner",
        settings["planner"],
        "--output",
        str(directory),
        "--duration",
        "20",
        "--speed",
        ".5",
        "--heading-deg",
        "-90",
        "--seed",
        "42",
    ]
    specs.append(("original", "none", directory, command))
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stages", default="100,500,1500,3000")
    parser.add_argument("--wait-for-run")
    parser.add_argument("--resume-checkpoint", help="Start a new, validated refinement branch from a complete checkpoint")
    parser.add_argument("--skip-zero-tokens", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    os.chdir(root)
    settings = read_json(args.config)
    stages = [int(value) for value in args.stages.split(",")]
    if not stages or stages != sorted(set(stages)) or stages[0] < 1 or stages[-1] != settings["iterations"]:
        raise ValueError("Stages must be positive, strictly increasing and end at configured iterations")
    if settings["num_envs_per_rank"] != 2040 or settings.get("terrain_profile") != "irregular_v1":
        raise ValueError(
            "Posture refinement starts with 2040 environments on each of eight irregular-terrain ranks"
        )
    for key in ("initial_checkpoint", "reference_checkpoint", "planner"):
        settings[key] = str(Path(settings[key]).resolve())
        if not Path(settings[key]).is_file():
            raise FileNotFoundError(f"Missing {key}: {settings[key]}")
    initial_resume = None
    if args.resume_checkpoint or settings.get("speed_refinement") is not None:
        from .speed_refinement import initial_resume_state

        initial_resume = initial_resume_state(settings, args.resume_checkpoint, stages)
    out = Path(args.run_dir).resolve()
    # Parent launch scripts may already save source/config provenance in out.
    # Only empty/provenance-only directories are accepted; all run outputs below
    # are subsequently created without reusing names.
    out.mkdir(parents=True, exist_ok=True)
    status_path, config_path = out / "supervisor_status.json", out / "config.json"
    if (
        status_path.exists()
        or config_path.exists()
        or any(out.glob("train_*"))
        or any(out.glob("evaluate_*"))
    ):
        raise ValueError(f"Refusing to overwrite existing run artifacts: {out}")
    write_json(config_path, settings)
    status = {
        "status": "starting",
        "run_kind": "full_8gpu",
        "pid": os.getpid(),
        "process_create_time": psutil.Process().create_time(),
        "started": now(),
        "stages": stages,
        "completed_updates": 0,
        "current_minimum_updates": 0,
        "completed_stages": [],
        "world_size": 8,
        "num_envs_per_rank": 2040,
        "latest_checkpoint": None,
        "evaluations": [],
        "failures": [],
        "failure": None,
        "quality_verified": False,
        "training_completed": False,
    }
    if initial_resume is not None:
        status.update(initial_resume)
        status["current_minimum_updates"] = initial_resume["completed_updates"]
    write_json(status_path, status)
    environment = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        PYTHONFAULTHANDLER="1",
        TORCH_SHOW_CPP_STACKTRACES="1",
        TORCH_NCCL_TRACE_BUFFER_SIZE="4096",
        TORCH_NCCL_DUMP_ON_TIMEOUT="1",
    )
    try:
        if args.wait_for_run:
            wait_for_prior(args.wait_for_run, status, status_path)
        for target in stages:
            feedback_path = out / f"stage_{target:05d}_feedback.json"
            feedback = {
                "target_updates": target,
                "status": "training",
                "attempts": [],
                "evaluations": [],
                "failures": [],
            }
            write_json(feedback_path, feedback)
            attempt = 0
            while status["completed_updates"] < target:
                start = status["completed_updates"]
                directory = out / f"train_{start:05d}_{target:05d}_attempt{attempt}"
                command = [
                    sys.executable,
                    "-m",
                    "vibe_grail_joint.velocity.launch",
                    "--gpus",
                    "0,1,2,3,4,5,6,7",
                    "--config",
                    str(config_path),
                    "--log-dir",
                    str(directory),
                    "--num-envs",
                    str(status["num_envs_per_rank"]),
                    "--iterations",
                    str(target - start),
                ]
                if status["latest_checkpoint"]:
                    command += ["--resume", status["latest_checkpoint"]]
                status.update(
                    status="training", active_directory=str(directory), target_updates=target, updated=now()
                )
                write_json(status_path, status)

                def progress(current, rank_count):
                    status.update(current_minimum_updates=current, observed_ranks=rank_count, updated=now())
                    write_json(status_path, status)

                code, failure = run_training(command, directory, environment, start, progress)
                checkpoints = sorted(
                    directory.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[-1])
                )
                if checkpoints:
                    latest = checkpoints[-1]
                    completed = int(latest.stem.split("_")[-1]) + 1
                    if not start < completed <= target:
                        raise RuntimeError("Checkpoint update is outside the requested stage")
                    status.update(latest_checkpoint=str(latest), completed_updates=completed)
                feedback["attempts"].append(
                    {
                        "directory": str(directory),
                        "exit_code": code,
                        "error": failure,
                        "num_envs_per_rank": status["num_envs_per_rank"],
                    }
                )
                write_json(feedback_path, feedback)
                write_json(status_path, status)
                if code == 0 and failure is None:
                    ranks = [read_json(directory / f"rank_{rank}_status.json") for rank in range(8)]
                    if any(
                        nonfinite_paths({k: rank.get(k) for k in ("loss", "metrics", "learning_rate")})
                        for rank in ranks
                    ):
                        raise RuntimeError("Numerically invalid final training status; checkpoint retained")
                    checks = [
                        {
                            "rank": r["rank"],
                            "status": r["status"],
                            "frozen_unchanged": r.get("frozen_unchanged"),
                            "absolute_completed_updates": r.get("absolute_completed_updates"),
                            "before": r.get("frozen_sha256"),
                            "after": r.get("frozen_sha256_after"),
                        }
                        for r in ranks
                    ]
                    feedback["rank_checks"] = checks
                    write_json(feedback_path, feedback)
                    if not all(
                        r["status"] == "completed"
                        and r["frozen_unchanged"]
                        and r["before"]
                        and r["before"] == r["after"]
                        and r["absolute_completed_updates"] == target
                        for r in checks
                    ):
                        raise RuntimeError("Stage lacks all eight completion/frozen-weight checks")
                    if status["completed_updates"] != target:
                        raise RuntimeError("Final checkpoint does not match completed stage")
                    break
                log = (directory / "console.log").read_text(errors="replace")
                if (
                    "Numerically invalid" in (failure or "")
                    or not _OOM.search(log + (failure or ""))
                    or attempt >= 2
                    or status["num_envs_per_rank"] <= 510
                ):
                    raise RuntimeError(f"Training exited {code}: {failure}; see {directory}")
                status["num_envs_per_rank"] = max(510, status["num_envs_per_rank"] // 20 * 10)
                attempt += 1
                # If an OOM happened after the final checkpoint, its frozen
                # audit may be missing. Do not silently skip that audit.
                if status["completed_updates"] == target:
                    raise RuntimeError(
                        "OOM after final checkpoint but before verified rank completion; checkpoint retained"
                    )
            gate_reasons = []
            feedback.update(status="evaluating", checkpoint=status["latest_checkpoint"])
            write_json(feedback_path, feedback)
            for kind, ablation, directory, command in evaluation_commands(
                root,
                config_path,
                settings,
                status["latest_checkpoint"],
                target,
                out,
                not args.skip_zero_tokens,
            ):
                if directory.exists():
                    raise RuntimeError(f"Refusing to overwrite evaluation: {directory}")
                status.update(status="evaluating", active_directory=str(directory), updated=now())
                write_json(status_path, status)
                code, failure = run_monitored(command, directory, environment, training=False)
                entry = {
                    "updates": target,
                    "kind": kind,
                    "visual_ablation": ablation,
                    "directory": str(directory),
                    "exit_code": code,
                }
                try:
                    if code != 0 or failure is not None:
                        raise RuntimeError(f"Evaluation process exited {code}: {failure}")
                    entry.update(summarize_evaluation(directory, kind, ablation))
                    if entry["numerical_errors"]:
                        gate_reasons.append(
                            {
                                "kind": kind,
                                "ablation": ablation,
                                "numerical_errors": entry["numerical_errors"],
                            }
                        )
                    if kind != "original" and ablation == "none" and entry["stairs_success_rate"] < 0.85:
                        gate_reasons.append(
                            {
                                "kind": kind,
                                "stairs_success_rate": entry["stairs_success_rate"],
                                "minimum": 0.85,
                            }
                        )
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    entry["infrastructure_error"] = repr(error)
                    (directory / "summary_traceback.txt").write_text(traceback.format_exc())
                if (
                    entry.get("failures")
                    or entry.get("infrastructure_error")
                    or entry.get("numerical_errors")
                ):
                    failure_entry = {
                        "updates": target,
                        "kind": kind,
                        "ablation": ablation,
                        "directory": str(directory),
                        "cases": entry.get("failures", []),
                        "error": entry.get("infrastructure_error"),
                        "numerical_errors": entry.get("numerical_errors", []),
                    }
                    status["failures"].append(failure_entry)
                    feedback["failures"].append(failure_entry)
                status["evaluations"].append(entry)
                feedback["evaluations"].append(entry)
                write_json(status_path, status)
                write_json(feedback_path, feedback)
            feedback.update(
                status="quality_gate_stopped" if gate_reasons else "completed",
                gate_reasons=gate_reasons,
                finished=now(),
            )
            write_json(feedback_path, feedback)
            if gate_reasons:
                status.update(
                    status="quality_gate_stopped",
                    gate_reasons=gate_reasons,
                    finished=now(),
                    failure="Numerical invalidity or major stair crossing collapse; checkpoint retained for review",
                )
                write_json(status_path, status)
                return 2
            status["completed_stages"].append(target)
            write_json(status_path, status)
        status.update(
            status="completed_with_evaluation_failures" if status["failures"] else "completed",
            training_completed=True,
            finished=now(),
            quality_verified=False,
            review_note="All requested training stages completed. Posture/velocity improvement and RGB dependence require comparative review.",
        )
        write_json(status_path, status)
        return 0
    except BaseException as error:
        stack = traceback.format_exc()
        (out / "supervisor_traceback.txt").write_text(stack)
        if "feedback_path" in locals():
            feedback.update(status="failed", failure=repr(error), finished=now())
            write_json(feedback_path, feedback)
        status.update(status="failed", failure=repr(error), finished=now())
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
