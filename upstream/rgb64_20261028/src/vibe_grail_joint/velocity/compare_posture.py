"""Compare matching 192-case single-actor posture evaluation batches on CPU.

Example::

    python -m vibe_grail_joint.velocity.compare_posture \
      --baseline-dir records/20261017_upper_body_revision/baseline_uniform_full \
      --candidate-dir runs/new_run/evaluate_00100_uniform \
      --output records/new_comparison.json

Case-averaged statistics give each case equal weight. Original pooled statistics
give each 50 Hz sample equal weight, so longer episodes receive more weight.
Neither comparison establishes an exact common-time window between checkpoints:
each evaluation stops each case at its own terminal event. No winner or training
completion is inferred, and failed or diagnostically incomplete cases are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


METRICS = (
    "torso_heading_aligned_orientation_error_deg",
    "waist_joint_rmse_deg",
    "arms_joint_rmse_deg",
    "velocity_error_m_s",
)
PROTOCOL_FIELDS = (
    "seed", "terrain_profile", "duration_s", "visual_ablation",
    "actor_receives_rgb", "actor_receives_depth", "terrain_actor_input",
    "prerecorded_stair_reference", "no_auto_reset", "planner", "command",
    "initialization", "criterion", "initial_vibe_without_velocity_training",
)
EXPECTED_CASE_COUNT = 192


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _mean(values):
    return math.fsum(values) / len(values) if values else None


def _delta(before, after):
    return after - before if before is not None and after is not None else None


def _report_cases(report, label):
    rows = report.get("cases")
    if not isinstance(rows, list) or len(rows) != EXPECTED_CASE_COUNT:
        raise ValueError(f"{label}: expected exactly {EXPECTED_CASE_COUNT} cases")
    if report.get("camera", "missing") is not None:
        raise ValueError(f"{label}: expected a single-actor batch, not recorded SONIC/ViBe worlds")
    result = {}
    for env_index, row in enumerate(rows):
        case = row.get("case", {})
        case_id = case.get("id")
        if not _integer(case_id) or case_id in result:
            raise ValueError(f"{label}: missing, noninteger or duplicated case ID {case_id!r}")
        if case.get("policy") != "velocity_vibe":
            raise ValueError(f"{label}: case {case_id} is not the single velocity_vibe actor role")
        for key in ("tile", "stream", "speed", "phase", "spawn_xy"):
            if key not in case:
                raise ValueError(f"{label}: case {case_id} missing command/init field {key}")
        terrain = row.get("terrain")
        if not isinstance(terrain, dict) or "kind" not in terrain or "level" not in terrain:
            raise ValueError(f"{label}: case {case_id} missing terrain definition")
        if not isinstance(row.get("success"), bool) or not isinstance(row.get("reason"), str):
            raise ValueError(f"{label}: case {case_id} missing explicit terminal outcome")
        if not _integer(row.get("step")) or row["step"] < 0 or not _finite(row.get("time_s")):
            raise ValueError(f"{label}: case {case_id} missing terminal step/time")
        if not math.isclose(row["time_s"], row["step"] / 50.0, abs_tol=1e-7):
            raise ValueError(f"{label}: case {case_id} does not follow the 50 Hz evaluation protocol")
        result[case_id] = (env_index, row)
    return result


def _summary_envs(summary, label):
    if summary is None:
        return {}
    if summary.get("schema_version") != 1:
        raise ValueError(f"{label}: unsupported posture summary schema")
    if summary.get("common_window") is not None:
        raise ValueError(f"{label}: two-world common-window summary is not a single-actor batch")
    rows = summary.get("per_env", [])
    if not isinstance(rows, list):
        raise ValueError(f"{label}: posture per_env must be a list")
    result = {}
    for row in rows:
        index = row.get("env_index")
        if not _integer(index) or not 0 <= index < EXPECTED_CASE_COUNT or index in result:
            raise ValueError(f"{label}: invalid or duplicate diagnostic env_index {index!r}")
        result[index] = row
    return result


def _metric_record(raw, expected_samples, context_issues=()):
    issues = list(context_issues)
    if not isinstance(raw, dict):
        return dict(mean=None, valid_samples=None, invalid_samples=None,
                    expected_samples=expected_samples, complete=False,
                    issues=[*issues, "missing_metric"])
    mean, count, invalid = raw.get("mean"), raw.get("count"), raw.get("invalid_count")
    if not _integer(count) or not 0 <= count <= expected_samples:
        issues.append("invalid_valid_sample_count")
    if not _integer(invalid) or invalid < 0 or not _integer(count) or count + invalid != expected_samples:
        issues.append("invalid_missing_sample_count")
    if not _finite(mean):
        issues.append("missing_or_nonfinite_mean")
    if _integer(invalid) and invalid > 0:
        issues.append("nonfinite_samples_excluded_by_source_summary")
    if count != expected_samples:
        issues.append("incomplete_sample_coverage")
    return dict(mean=float(mean) if _finite(mean) else None,
                valid_samples=count if _integer(count) else None,
                invalid_samples=invalid if _integer(invalid) else None,
                expected_samples=expected_samples, complete=not issues, issues=issues)


def _case_diagnostics(row, diagnostic):
    expected = row["step"] + 1  # Evaluation includes reset sample and terminal sample.
    issues = []
    stage = diagnostic.get("stages", {}).get("all", {}) if diagnostic else {}
    if not diagnostic:
        issues.append("missing_case_diagnostics")
    else:
        if diagnostic.get("sample_count") != expected or stage.get("sample_count") != expected:
            issues.append("sample_count_does_not_match_terminal_step")
        if stage.get("first_time_s") != 0.0:
            issues.append("diagnostics_do_not_start_at_reset")
        last = stage.get("last_time_s")
        if not _finite(last) or not math.isclose(last, row["time_s"], abs_tol=1e-7):
            issues.append("diagnostics_do_not_end_at_terminal")
    metrics = stage.get("metrics", {})
    result = {name: _metric_record(metrics.get(name), expected, issues) for name in METRICS}
    # This independently computed report scalar is a cheap guard against pairing
    # a report with the wrong summary file from another evaluation.
    report_speed = row.get("mean_velocity_error_m_s")
    speed = result["velocity_error_m_s"]
    if _finite(report_speed) and speed["mean"] is not None and not math.isclose(
            report_speed, speed["mean"], abs_tol=2e-6, rel_tol=1e-5):
        speed["issues"].append("velocity_mean_disagrees_with_report")
        speed["complete"] = False
    return result


def _success_counts(pairs):
    total = len(pairs)
    before = sum(p["baseline"]["success"] for p in pairs)
    after = sum(p["candidate"]["success"] for p in pairs)
    return dict(total=total, baseline_success=before, candidate_success=after,
                success_count_delta=after-before,
                baseline_success_rate=before/total if total else None,
                candidate_success_rate=after/total if total else None)


def _equal_case_statistics(pairs):
    result = {}
    for metric in METRICS:
        complete = [p for p in pairs if p["metrics"][metric]["complete_pair"]]
        before = _mean([p["metrics"][metric]["baseline"]["mean"] for p in complete])
        after = _mean([p["metrics"][metric]["candidate"]["mean"] for p in complete])
        all_complete = bool(pairs) and len(complete) == len(pairs)
        result[metric] = dict(
            case_count=len(pairs), complete_paired_case_count=len(complete),
            incomplete_case_ids=[p["case"]["id"] for p in pairs if not p["metrics"][metric]["complete_pair"]],
            baseline_mean=before if all_complete else None,
            candidate_mean=after if all_complete else None,
            candidate_minus_baseline=after-before if all_complete else None,
            complete_pairs_subset=dict(
                case_ids=[p["case"]["id"] for p in complete], baseline_mean=before,
                candidate_mean=after, candidate_minus_baseline=_delta(before, after)),
        )
    return result


def _pooled_statistics(summary, pairs, side):
    expected = sum(p[side]["step"] + 1 for p in pairs)
    scope = summary.get("by_stage", {}).get("all", {}) if summary else {}
    issues = [] if scope.get("sample_count") == expected else ["pooled_sample_count_mismatch_or_missing"]
    result = {}
    for name in METRICS:
        raw = scope.get("metrics", {}).get(name)
        record = _metric_record(raw, expected, issues)
        # Validate source pooling independently of its named aggregation string.
        samples = [p["metrics"][name][side] for p in pairs]
        if not all(s["complete"] for s in samples):
            record["issues"].append("pooled_scope_has_missing_or_invalid_case_diagnostics")
            record["complete"] = False
        if all(s["complete"] for s in samples) and record["mean"] is not None:
            recomputed = math.fsum(s["mean"] * s["valid_samples"] for s in samples) / expected
            record["recomputed_from_case_means"] = recomputed
            if not math.isclose(record["mean"], recomputed, rel_tol=1e-8, abs_tol=1e-8):
                record["issues"].append("pooled_mean_disagrees_with_case_samples")
                record["complete"] = False
        else:
            record["recomputed_from_case_means"] = None
        result[name] = record
    return result


def compare_payloads(baseline_report, baseline_summary, candidate_report, candidate_summary):
    """Validate and compare serialized evaluator output; no file/GPU side effects."""
    for field in PROTOCOL_FIELDS:
        if field not in baseline_report or field not in candidate_report:
            raise ValueError(f"Missing evaluation protocol field: {field}")
        if baseline_report[field] != candidate_report[field]:
            raise ValueError(f"Evaluation protocol mismatch: {field}")
    baseline = _report_cases(baseline_report, "baseline")
    candidate = _report_cases(candidate_report, "candidate")
    if baseline.keys() != candidate.keys():
        raise ValueError("Case identifier sets differ")
    diagnostics = {
        "baseline": _summary_envs(baseline_summary, "baseline"),
        "candidate": _summary_envs(candidate_summary, "candidate"),
    }
    pairs, issues = [], []
    for case_id in sorted(baseline):
        bi, b = baseline[case_id]
        ci, c = candidate[case_id]
        if b["case"] != c["case"]:
            raise ValueError(f"Case {case_id}: command/initialization definition mismatch")
        if b["terrain"] != c["terrain"]:
            raise ValueError(f"Case {case_id}: physical terrain definition mismatch")
        b_metrics = _case_diagnostics(b, diagnostics["baseline"].get(bi))
        c_metrics = _case_diagnostics(c, diagnostics["candidate"].get(ci))
        pair = dict(case=b["case"], terrain=b["terrain"], metrics={})
        for side, row, index in (("baseline", b, bi), ("candidate", c, ci)):
            pair[side] = {key: row[key] for key in ("success", "reason", "step", "time_s")}
            pair[side]["env_index"] = index
        for name in METRICS:
            before, after = b_metrics[name], c_metrics[name]
            complete = before["complete"] and after["complete"]
            pair["metrics"][name] = dict(
                baseline=before, candidate=after, complete_pair=complete,
                candidate_minus_baseline=_delta(before["mean"], after["mean"]),
                delta_covers_all_terminal_samples=complete,
            )
            for side, entry in (("baseline", before), ("candidate", after)):
                if entry["issues"]:
                    issues.append(dict(case_id=case_id, side=side, metric=name, issues=entry["issues"]))
        pair["regression_reasons"] = []
        if b["success"] and not c["success"]:
            pair["regression_reasons"].append("lost_baseline_success")
        if not c["success"] and c["time_s"] < b["time_s"] - 1e-7:
            pair["regression_reasons"].append("candidate_failure_before_baseline_terminal")
        pairs.append(pair)

    both_success = [p for p in pairs if p["baseline"]["success"] and p["candidate"]["success"]]
    by_direction, by_level, by_direction_level = {}, {}, {}
    for pair in pairs:
        kind, level = pair["terrain"]["kind"], pair["terrain"]["level"]
        by_direction.setdefault(kind, []).append(pair)
        by_level.setdefault(str(level), []).append(pair)
        by_direction_level.setdefault(f"{kind}/level_{level}", []).append(pair)
    pooled = {side: _pooled_statistics(summary, pairs, side) for side, summary in (
        ("baseline", baseline_summary), ("candidate", candidate_summary))}
    for side in pooled:
        for name, entry in pooled[side].items():
            if entry["issues"]:
                issues.append(dict(scope="pooled_full_batch", side=side, metric=name, issues=entry["issues"]))
    transition = lambda p: dict(case_id=p["case"]["id"], direction=p["terrain"]["kind"],
                                level=p["terrain"]["level"], baseline=p["baseline"],
                                candidate=p["candidate"], regression_reasons=p["regression_reasons"])
    result = dict(
        schema_version=1,
        protocol_validation=dict(case_count=len(pairs), actor_role="velocity_vibe",
                                 metadata={field: baseline_report[field] for field in PROTOCOL_FIELDS},
                                 case_definitions_identical=True, physical_terrain_identical=True),
        checkpoints={side: {key: report.get(key) for key in ("checkpoint", "checkpoint_iteration", "label")}
                     for side, report in (("baseline", baseline_report), ("candidate", candidate_report))},
        comparison_complete=not issues,
        diagnostic_issues=issues,
        success=dict(
            all_cases=_success_counts(pairs),
            stairs_only=_success_counts([p for p in pairs if p["terrain"]["kind"] != "flat"]),
            by_direction={k: _success_counts(v) for k, v in sorted(by_direction.items())},
            by_level={k: _success_counts(v) for k, v in sorted(by_level.items())},
            by_direction_level={k: _success_counts(v) for k, v in sorted(by_direction_level.items())}),
        regressions=[transition(p) for p in pairs if p["regression_reasons"]],
        improved_successes=[transition(p) for p in pairs
                            if not p["baseline"]["success"] and p["candidate"]["success"]],
        equal_case_weighted=dict(
            definition="Mean of each case's all-terminal-samples mean; equal weight per case. Negative delta means less error.",
            all_cases=_equal_case_statistics(pairs),
            matched_both_successes=dict(case_ids=[p["case"]["id"] for p in both_success],
                                        metrics=_equal_case_statistics(both_success)),
            missing_data_policy="A main scope mean is null unless every included case has all finite diagnostic samples. "
                                "The explicitly named complete_pairs_subset is a conditional subset, not a replacement."),
        existing_pooled_sample_weighted_full_batch=dict(
            definition="Original by_stage.all metrics, one weight per 50 Hz sample, including reset/terminal samples. "
                       "Approximately duration-weighted; longer episodes contribute more than short failures. "
                       "This is different from equal-case weighting and is not a matched common-time statistic.",
            baseline=pooled["baseline"], candidate=pooled["candidate"],
            candidate_minus_baseline={name: _delta(pooled["baseline"][name]["mean"], pooled["candidate"][name]["mean"])
                                      if pooled["baseline"][name]["complete"] and pooled["candidate"][name]["complete"]
                                      else None for name in METRICS}),
        limitations=[
            "Cases end at their own success/failure/time-limit. Different durations and visited states prevent an exact common-time comparison.",
            "All-case posture means include failures through their terminal sample; a shorter failure can artificially reduce measured posture error.",
            "Matched-both-successes is a conditional subset and must be read together with all-case success regressions.",
            "Averages of per-step joint RMSE are not a single global joint/time RMSE.",
            "Posture summaries do not embed checkpoint hashes; report/summary timing and velocity consistency are checked, but provenance still relies on their evaluation directory.",
            "This report measures differences only; it does not automatically select a winner or declare the training/user goal complete.",
        ],
        cases=pairs,
    )
    return result


def _load_directory(directory):
    root = Path(directory).resolve()
    report_path, summary_path = root / "report.json", root / "posture_summary.json"
    report_bytes = report_path.read_bytes()
    summary_bytes = summary_path.read_bytes() if summary_path.is_file() else None
    report = json.loads(report_bytes)
    summary = json.loads(summary_bytes) if summary_bytes is not None else None
    source = dict(directory=str(root), report_path=str(report_path),
                  report_sha256=hashlib.sha256(report_bytes).hexdigest(),
                  posture_summary_path=str(summary_path), posture_summary_present=summary_bytes is not None,
                  posture_summary_sha256=hashlib.sha256(summary_bytes).hexdigest() if summary_bytes else None)
    return report, summary, source


def compare_directories(baseline_dir, candidate_dir):
    b, bs, bsource = _load_directory(baseline_dir)
    c, cs, csource = _load_directory(candidate_dir)
    result = compare_payloads(b, bs, c, cs)
    result["sources"] = dict(baseline=bsource, candidate=csource)
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True, help="New JSON path; existing files are never replaced")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output}")
    result = compare_directories(args.baseline_dir, args.candidate_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(output=str(output), comparison_complete=result["comparison_complete"],
                          success=result["success"]["all_cases"], regressions=len(result["regressions"]),
                          diagnostic_issues=len(result["diagnostic_issues"])), ensure_ascii=False))


if __name__ == "__main__":
    main()
