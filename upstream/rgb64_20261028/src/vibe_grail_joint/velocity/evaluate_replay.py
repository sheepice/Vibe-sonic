"""Deterministic retention evaluation on a fixed small flat/GRAIL replay bank.

This evaluates the replay roles of a command-aware SONIC + ViBe actor. It is not
an online-W test with arbitrary upper-body commands. There is no planner: both
the motion tokenizer and 8-D explicit command come from the selected reference.
Terminal posture is captured before automatic reset, and padding/already-ended
worlds never contribute to statistics. All selected cases, including failures
and evaluator-horizon exhaustion, remain in the report.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

import mjlab  # noqa: F401 -- initialize upstream task discovery
import torch

from ..evaluation_env import EvaluationEnv
from ..gpu_guard import verify_guarded_worker
from .posture_diagnostics import PostureDiagnostics
from .posture_summary import PostureSummary


_DISCARD_FORWARD_METRICS = {
    "command_speed_m_s", "forward_speed_m_s", "lateral_speed_m_s", "signed_speed_error_m_s",
    "velocity_error_m_s", "forward_displacement_m", "lateral_displacement_m",
    "commanded_distance_m", "horizontal_path_length_m", "distance_error_m",
}


def replay_velocity_command(env):
    """Exactly the existing velocity task's replay branch, with no online state."""
    from orcs.core.mdp.observations import robot_root_ang_vel_cmd, robot_root_lin_vel_cmd

    c = env.command_manager.get_term("motion")
    velocity = robot_root_lin_vel_cmd(env, "motion")
    angular = robot_root_ang_vel_cmd(env, "motion")
    paired = c.motion.terrain_ids[c._clip_ids] >= 0
    result = torch.zeros(velocity.shape[0], 8, dtype=velocity.dtype, device=velocity.device)
    result[:, :2] = velocity[:, :2]
    result[:, 2] = angular[:, 2]
    result[:, 6], result[:, 7] = paired.to(velocity.dtype), (~paired).to(velocity.dtype)
    return result


@torch.no_grad()
def sample_replay_metrics(command, recorder, time_s):
    """Instantaneous posture plus actual-minus-reference world root errors.

    PostureDiagnostics' +X command/integral channels do not apply to replay.
    Passing command_speed=0 avoids inventing a command.speed on JointMotionCommand;
    dropping those channels avoids mislabelling a moving reference as zero-speed.
    Its integrator is deliberately not advanced because no integrated output is
    used. The externally supplied episode clock is retained as the endpoint.
    """
    raw = recorder.sample(command_speed=0.0, elapsed_dt_s=0.0)
    result = {k: v.clone() for k, v in raw.items() if k not in _DISCARD_FORWARD_METRICS}
    robot = command.robot.data
    actual_position, reference_position = robot.root_link_pos_w, command.anchor_pos_w
    actual_velocity, reference_velocity = robot.root_link_lin_vel_w, command.anchor_lin_vel_w
    position_error = actual_position - reference_position
    velocity_error = actual_velocity - reference_velocity
    result["root_position_error_m"] = position_error.norm(dim=-1)
    result["root_velocity_error_m_s"] = velocity_error.norm(dim=-1)
    result["root_horizontal_velocity_error_m_s"] = velocity_error[:, :2].norm(dim=-1)
    result["actual_root_speed_m_s"] = actual_velocity.norm(dim=-1)
    result["reference_root_speed_m_s"] = reference_velocity.norm(dim=-1)
    for index, axis in enumerate("xyz"):
        result[f"root_position_error_{axis}_m"] = position_error[:, index].clone()
        result[f"root_velocity_error_{axis}_m_s"] = velocity_error[:, index].clone()
    paired = command.motion.terrain_ids[command._clip_ids] >= 0
    result["terrain_stage"] = torch.where(paired, -torch.ones_like(command._clip_ids),
                                           torch.zeros_like(command._clip_ids))
    result["elapsed_s"] = torch.full_like(actual_position[:, 0], float(time_s))
    result["online"] = torch.zeros_like(paired)
    return result


def replace_terminal_rows(post_step_metrics, snapshots):
    """Replace reset-state rows with detached pre-reset metrics; leave others intact."""
    result = {key: value.clone() for key, value in post_step_metrics.items()}
    for indices, values in snapshots:
        if set(values) != set(result):
            raise ValueError("Pre-reset and post-step diagnostic schemas differ")
        for key in result:
            result[key][indices] = values[key]
    return result


class ReplayEvaluationEnv(EvaluationEnv):
    """EvaluationEnv's terminal outcome plus pre-reset upper-body measurements."""

    replay_recorder = None
    replay_active_mask = None
    replay_time_s = 0.0

    def _reset_idx(self, env_ids=None):
        if (self.record_terminals and env_ids is not None and self.replay_recorder is not None
                and self.replay_active_mask is not None):
            selected = env_ids[self.replay_active_mask[env_ids]]
            if len(selected):
                c = self.command_manager.get_term("motion")
                metrics = sample_replay_metrics(c, self.replay_recorder, self.replay_time_s)
                self.replay_snapshots.append((selected.clone(), {k: v[selected].clone() for k, v in metrics.items()}))
                for index in selected.tolist():
                    self.replay_terminal_flags[index] = dict(
                        terminated=bool(self.termination_manager.terminated[index]),
                        timed_out=bool(self.termination_manager.time_outs[index]))
        # Preserve the original terminal reason/root-goal calculation. That
        # helper may record other reset worlds; the evaluator only consumes
        # active real cases, never padding or a case's second episode.
        super()._reset_idx(env_ids)


def validate_manifest(bank, manifest):
    """Require a bijection to bank clips so no clip silently disappears or repeats."""
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Replay manifest must contain nonempty cases")
    if float(manifest.get("control_hz", 0)) != 50.0 or float(bank["config"]["control_hz"]) != 50.0:
        raise ValueError("Replay evaluator requires the bank's 50 Hz reference/control rate")
    if sorted(c.get("case_id", -1) for c in cases) != list(range(len(cases))):
        raise ValueError("Manifest case_id must be exactly 0..N-1")
    if sorted(c.get("bank_clip_index", -1) for c in cases) != list(range(len(bank["clips"]))):
        raise ValueError("Manifest must cover every bank clip once")
    for case in cases:
        clip = bank["clips"][case["bank_clip_index"]]
        if case.get("id") != clip["id"] or case.get("source") != clip["source"] or case.get("length") != clip["length"]:
            raise ValueError(f"Manifest does not match bank clip for case {case['case_id']}")
        expected = "grail_heldout" if clip["terrain_id"] >= 0 else "flat_preservation"
        if case.get("subset") != expected:
            raise ValueError(f"Case {case['case_id']} has incorrect replay subset/terrain role")
    return sorted(cases, key=lambda case: case["case_id"])


def build_replay_config(bank_path, num_envs, seed):
    """CPU-constructible original replay task plus the command-aware actor input."""
    from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
    from ..env_cfg import build_env_cfg

    cfg = build_env_cfg(str(bank_path), num_envs=num_envs, play=True)
    cfg.seed = seed
    cfg.commands["motion"].pose_range = {}
    cfg.commands["motion"].velocity_range = {}
    cfg.commands["motion"].joint_position_range = (0.0, 0.0)
    cfg.commands["motion"].start_from_zero = True
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            term.noise = None
    cfg.observations["velocity_command"] = ObservationGroupCfg(
        terms={"command": ObservationTermCfg(func=replay_velocity_command)},
        concatenate_terms=True, enable_corruption=False)
    return cfg


def _pool_scopes(scopes, metric_names):
    total_samples = sum(scope["sample_count"] for scope in scopes)
    metrics = {}
    for name in metric_names:
        records = [scope["metrics"][name] for scope in scopes]
        count = sum(record["count"] for record in records)
        valid = [record for record in records if record["count"]]
        metrics[name] = dict(count=count, invalid_count=total_samples-count)
        for statistic in ("mean", "abs_mean"):
            metrics[name][statistic] = (math.fsum(r[statistic]*r["count"] for r in valid)/count
                                       if count else None)
        metrics[name]["rms"] = (math.sqrt(math.fsum(r["rms"]**2*r["count"] for r in valid)/count)
                                 if count else None)
        metrics[name]["max"] = max((r["max"] for r in valid), default=None)
        metrics[name]["abs_max"] = max((r["abs_max"] for r in valid), default=None)
    return dict(sample_count=total_samples, metrics=metrics)


def merge_replay_summaries(batch_summaries, cases):
    """Join already-masked batch statistics without averaging their RMS values."""
    per_env, metric_names = [], None
    for summary, batch_case_ids in batch_summaries:
        if metric_names is None:
            metric_names = summary["metric_names"]
        elif metric_names != summary["metric_names"]:
            raise ValueError("Replay batches have different diagnostic schemas")
        for local_index, case_id in enumerate(batch_case_ids):
            row = copy.deepcopy(summary["per_env"][local_index])
            if row["env_index"] != local_index:
                raise ValueError("Unexpected batch diagnostic order")
            row["env_index"], row["batch_env_index"] = case_id, local_index
            per_env.append(row)
    per_env.sort(key=lambda row: row["env_index"])
    if [row["env_index"] for row in per_env] != [case["case_id"] for case in cases]:
        raise ValueError("Missing or duplicated replay diagnostic case")
    metric_names = metric_names or []
    subsets = sorted({case["subset"] for case in cases})
    return dict(
        schema_version=1, evaluation_role="reference_replay_preservation", metric_names=metric_names,
        aggregation="Per-control-step sample-weighted including reset and first terminal sample; failures retained.",
        rms_definition="For a joint_rmse metric, mean is mean instantaneous joint RMSE; rms is the joint-and-time "
                       "RMSE over the fixed joint group and finite samples. Root norm-error rms is vector/time RMSE. "
                       "Raw torso pitch rms describes pose magnitude, not reference tracking error.",
        stage_definition="Flat source samples are flat_or_landing; GRAIL stage is unknown. No inferred up/down segmentation.",
        per_env=per_env,
        by_stage={stage: _pool_scopes([row["stages"][stage] for row in per_env], metric_names)
                  for stage in ("all", "flat_or_landing", "up", "down")},
        by_subset={subset: _pool_scopes([row["stages"]["all"] for row, case in zip(per_env, cases)
                                        if case["subset"] == subset], metric_names) for subset in subsets},
        common_window=None,
        limitations=["Different policy failures produce different episode durations; these are not common-time paired statistics.",
                     "Nonfinite metric samples remain visible via invalid_count; a finite conditional mean does not certify complete data."])


def _horizon_terminal(c, index):
    clip = c._clip_ids[index]
    first, last = c.motion.clip_offsets[clip], c.motion.clip_ends[clip]-1
    origin = c._env.scene.env_origins[index]
    root = c.robot.data.root_link_pos_w[index]
    return dict(
        termination_reasons=["evaluation_horizon_exhausted"], terminated=False, timed_out=True,
        traversal_completed=False, ascent_success=False,
        reference_height_gain_m=float(c.motion.body_pos_w[last, 0, 2]-c.motion.body_pos_w[first, 0, 2]),
        robot_height_gain_m=float(root[2]-c.motion.body_pos_w[first, 0, 2]-origin[2]),
        final_root_error_m=float((root-c.motion.body_pos_w[last, 0]-origin).norm()),
        terminal_tracking_error_m=float((root-c.anchor_pos_w[index]).norm()),
        reference_is_ascent=bool(c.motion.body_pos_w[last, 0, 2]-c.motion.body_pos_w[first, 0, 2] > 0.2))


def main(guard_check=verify_guarded_worker):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="New evaluation directory")
    parser.add_argument("--seed", type=int, default=170021)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--duration", type=float, default=30.0, help="Maximum seconds per case, not clip truncation during preparation")
    parser.add_argument("--max-steps", type=int, help="Optional tighter safety cap per batch")
    parser.add_argument("--policy", choices=("velocity_vibe", "sonic"), default="velocity_vibe")
    args = parser.parse_args()
    if args.num_envs < 1 or not math.isfinite(args.duration) or args.duration < 0.02:
        raise ValueError("num-envs must be positive and duration at least one 50 Hz step")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Replay evaluation requires a single worker")
    destination = Path(args.output).resolve()
    if any((destination/name).exists() for name in ("report.json", "posture_summary.json")):
        raise FileExistsError(f"Refusing to overwrite replay evaluation: {destination}")
    guard_check()
    import numpy as np
    from mjlab.rl import RslRlVecEnvWrapper
    from ..catalog import write_json
    from ..env_cfg import build_agent_cfg
    from ..prepare import validate_bank
    from .policy import CommandSonicActor

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    bank_path = Path(args.bank).resolve()
    bank = validate_bank(bank_path)
    manifest_bytes = (bank_path/"manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    cases = validate_manifest(bank, manifest)
    cfg = build_replay_config(bank_path, args.num_envs, args.seed)
    destination.mkdir(parents=True, exist_ok=True)
    env = RslRlVecEnvWrapper(ReplayEvaluationEnv(cfg, device="cuda:0"))
    results, batch_summaries = [], []
    try:
        agent = build_agent_cfg(bank["config"])
        actor_cfg = copy.deepcopy(agent.actor)
        actor_cfg.pop("class_name")
        actor = CommandSonicActor(obs=env.get_observations(), obs_groups=agent.obs_groups,
                                  obs_set="actor", output_dim=29, **actor_cfg).to(env.device).eval()
        checkpoint = torch.load(args.checkpoint, map_location=env.device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        checkpoint_iteration = checkpoint.get("iter")
        del checkpoint
        raw, command = env.unwrapped, env.unwrapped.command_manager.get_term("motion")
        if not math.isclose(raw.step_dt, 0.02, abs_tol=1e-10):
            raise ValueError("Replay environment must run at 50 Hz")
        horizon = max(1, int(args.duration / raw.step_dt))
        if args.max_steps is not None:
            horizon = min(horizon, args.max_steps)
        with torch.inference_mode():
            for start in range(0, len(cases), args.num_envs):
                batch = cases[start:start+args.num_envs]
                clip_ids = [case["bank_clip_index"] for case in batch]
                padded = clip_ids + [clip_ids[-1]]*(args.num_envs-len(batch))
                command.forced_clip_ids = torch.tensor(padded, device=env.device)
                raw.record_terminals = False
                raw.replay_recorder = None
                obs, _ = env.reset()
                valid = torch.arange(args.num_envs, device=env.device) < len(batch)
                finished = ~valid
                initial_frames = (command.time_steps-command.motion.clip_offsets[command._clip_ids]).cpu().tolist()
                raw.replay_recorder = PostureDiagnostics(command)
                raw.replay_active_mask = ~finished
                raw.record_terminals = True
                summary = PostureSummary(args.num_envs, env.device, keep_trace=False)
                summary.update(sample_replay_metrics(command, raw.replay_recorder, 0.0), 0.0, ~finished)
                for step in range(1, horizon+1):
                    raw.replay_time_s = step*raw.step_dt
                    raw.replay_active_mask = ~finished
                    raw.replay_snapshots, raw.replay_terminal_flags, raw.terminal_records = [], {}, {}
                    actions = actor(obs) if args.policy == "velocity_vibe" else actor.sonic_action(obs)
                    obs, _, dones, _ = env.step(actions)
                    metrics = sample_replay_metrics(command, raw.replay_recorder, raw.replay_time_s)
                    metrics = replace_terminal_rows(metrics, raw.replay_snapshots)
                    summary.update(metrics, raw.replay_time_s, ~finished)
                    newly_done = dones.bool() & ~finished
                    for index in newly_done.nonzero().flatten().tolist():
                        if index not in raw.terminal_records or index not in raw.replay_terminal_flags:
                            raise RuntimeError("A terminal replay case was not captured before reset")
                        terminal = {**raw.terminal_records[index], **raw.replay_terminal_flags[index]}
                        results.append(dict(case=batch[index], step=step, time_s=raw.replay_time_s,
                                            initial_reference_frame_offset=initial_frames[index],
                                            success=terminal["traversal_completed"], **terminal))
                    finished |= dones.bool()
                    if bool(finished.all()):
                        break
                for index in range(len(batch)):
                    if not bool(finished[index]):
                        terminal = _horizon_terminal(command, index)
                        results.append(dict(case=batch[index], step=horizon, time_s=horizon*raw.step_dt,
                                            initial_reference_frame_offset=initial_frames[index], success=False, **terminal))
                batch_summaries.append((summary.finalize(), [case["case_id"] for case in batch]))
                print(f"REPLAY {len(results)}/{len(cases)} complete, successes={sum(r['success'] for r in results)}", flush=True)
        results.sort(key=lambda row: row["case"]["case_id"])
        if [row["case"]["case_id"] for row in results] != [case["case_id"] for case in cases]:
            raise RuntimeError("Replay evaluator omitted or duplicated a case")
        combined = merge_replay_summaries(batch_summaries, cases)
        for row, diagnostic in zip(results, combined["per_env"]):
            if diagnostic["sample_count"] != row["step"]+1:
                raise RuntimeError("Replay diagnostics must include reset through first terminal/horizon once")
        groups = {}
        for row in results:
            group = groups.setdefault(row["case"]["subset"], dict(total=0, success=0, failures=[]))
            group["total"] += 1
            group["success"] += int(row["success"])
            if not row["success"]:
                group["failures"].append(dict(case_id=row["case"]["case_id"], reasons=row["termination_reasons"], time_s=row["time_s"]))
        report = dict(
            schema_version=1, evaluation_role="reference_replay_preservation", policy=args.policy,
            checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_iteration=checkpoint_iteration,
            bank=str(bank_path), bank_identity=bank["identity"], manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            seed=args.seed, num_envs=args.num_envs, duration_cap_s=args.duration, effective_max_steps=horizon,
            command="Reference-frame vx, vy, yaw-rate; heading-error=0; role [online=0, paired, flat].",
            reference="Complete selected bank clips from first state; no planner, no generated walking plan.",
            actor_receives_rgb=args.policy == "velocity_vibe", actor_receives_depth=False,
            deterministic_action_mean=True, observation_noise=False, reset_randomization=False,
            terminal_posture_before_reset=True, already_finished_and_padding_excluded=True,
            success_criterion="Existing EvaluationEnv: exceeded_motion without failure, final root goal error <0.3m and upright; original tracking/physical terminations retained.",
            terminal_state_timing="Pre-reset terminal snapshots use the same derived state as termination checks; upstream can lag one physics substep. Nonterminal samples use post-step observations.",
            groups=groups, total=len(results), success=sum(row["success"] for row in results), cases=results,
            limitations=["Small selected replay bank measures replay-role retention only, not arbitrary upper-body motions combined with online-W stair adaptation.",
                         "Flat clips are preservation samples; GRAIL held-out provenance is defined by manifest, not the compatibility bank.split field.",
                         "Different policies may terminate at different times; full-episode means are not exact common-window comparisons.",
                         "Terminal failures and horizon exhaustion are retained; inspect invalid_count before interpreting conditional metric means."],
        )
        write_json(destination/"posture_summary.json", combined)
        write_json(destination/"report.json", report)
        print(json.dumps(dict(report=str(destination/"report.json"), total=len(results), success=report["success"])), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
