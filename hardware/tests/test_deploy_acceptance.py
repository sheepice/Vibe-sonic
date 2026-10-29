"""A physics pass must belong to these exact artifacts, settings and speed."""
import copy
import json

import numpy as np
import pytest

from vibe_grail_joint.deploy.safety import acceptance_signature, validate_sim_acceptance


@pytest.fixture
def evidence(tmp_path):
    manifest = dict(files={"policy.onnx": "test-policy", "encoder.onnx": "test-encoder", "planner.onnx": "test-planner"},
                    robot={"stiffness": [40.]*29}, camera={"fovy": 42.5}, kind="tracking")
    report = dict(bundle_files=manifest["files"], acceptance_signature=acceptance_signature(manifest),
                  mode="keyboard", initial_state="stationary_default_pose", no_physics_guards_applied=False,
                  fall=False, target_guard_counts={}, simulated_seconds=12., requested_seconds=12.,
                  deploy_onnx_drove_physics=True, raw_rgb_encoder_used=True,
                  trace=[dict(t=i*.02, speed_command=.25 if 100 <= i < 400 else 0.,
                              position=[float(np.clip((i-100)*.005, 0, 1.5)), 0., .76]) for i in range(600)])
    return tmp_path / "evidence.json", manifest, report


def test_matching_completed_start_stop_passes(evidence):
    path, manifest, report = evidence
    path.write_text(json.dumps(report))
    validate_sim_acceptance(path, manifest, .25)
    with pytest.raises(ValueError, match="speed"):
        validate_sim_acceptance(path, manifest, .4)
    changed = copy.deepcopy(manifest)
    changed["robot"]["stiffness"][0] = 100
    with pytest.raises(ValueError, match="contract"):
        validate_sim_acceptance(path, changed, .25)


@pytest.mark.parametrize("problem", ["model", "signature", "fall", "guard", "unguarded", "early_stop",
                                    "no_stop", "no_move", "drift", "nan", "synthetic", "wrong_clock"])
def test_failed_or_inapplicable_evidence_is_refused(evidence, problem):
    path, manifest, original = evidence
    report = copy.deepcopy(original)
    if problem == "model": report["bundle_files"]["policy.onnx"] = "different"
    if problem == "signature": report.pop("acceptance_signature")
    if problem == "fall": report["fall"] = True
    if problem == "guard": report["guard_stop_reason"] = "joint_limit"
    if problem == "unguarded": report["no_physics_guards_applied"] = True
    if problem == "early_stop": report["requested_seconds"] = 20
    if problem == "no_stop": report["trace"][-1]["speed_command"] = .25
    if problem == "no_move":
        for row in report["trace"]: row["position"][0] = 0.
    if problem == "drift": report["trace"][-1]["position"][1] = .5
    if problem == "nan": report["trace"][-1]["position"][0] = float("nan")
    if problem == "synthetic": report["deploy_onnx_drove_physics"] = False
    if problem == "wrong_clock": report["trace"][-1]["t"] = 0.
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError): validate_sim_acceptance(path, manifest)
