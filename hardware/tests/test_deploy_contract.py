"""Observation math, deployment failures, and camera transport regression tests."""
import io
import struct

import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from vibe_grail_joint.deploy.observations import JOINT_NAMES, ObservationBuilder, Reference, RobotState
from vibe_grail_joint.deploy.safety import KeyboardIntent, Limits, health_fault, target_fault
from vibe_grail_joint.deploy.safety import condition_target
from vibe_grail_joint.deploy.camera import bilinear, pinhole_rays
from vibe_grail_joint.deploy.camera_server import encode_packet


def fixture_values():
    contract = dict(joint_names=JOINT_NAMES, default_joint_pos=[.1]*29, joint_limits=[[-3., 3.]]*29)
    state = RobotState(np.zeros(29), np.zeros(29), np.array([1., 0, 0, 0]), np.zeros(3), 1.)
    ref = Reference(np.arange(290).reshape(10, 29)/100, np.arange(290, 580).reshape(10, 29)/100,
                    np.tile([1., 0, 0, 0], (10, 1)), np.array([.4, 0, 0]), np.zeros(3))
    return contract, state, ref


def test_tokenizer_matches_training_nonflat_chop_and_rotation_columns():
    c, s, r = fixture_values()
    b = ObservationBuilder(c)
    b.heading = Rotation.identity()
    s.quat = Rotation.from_euler("xyz", [.15, -.1, .4]).as_quat()[[3, 0, 1, 2]]
    o = b.build(s, r, np.zeros((21, 192)), np.zeros(192), .4)
    expected = np.concatenate([r.joint_pos.ravel(), r.joint_vel.ravel()]).reshape(10, 58)
    np.testing.assert_allclose(o["tokenizer"].reshape(10, 64)[:, :58], expected, atol=1e-6)
    expected_rot = Rotation.from_quat(s.quat[[1, 2, 3, 0]]).inv().as_matrix()[:, :2].ravel()
    np.testing.assert_allclose(o["tokenizer"].reshape(10, 64)[:, 58:], np.tile(expected_rot, (10, 1)), atol=1e-6)
    np.testing.assert_allclose(o["velocity_command"][0], [.4*np.cos(.4), -.4*np.sin(.4), 0, 1-np.cos(.4), -np.sin(.4), 1, 0, 0], atol=1e-6)


def test_history_term_major_oldest_first_and_actual_last_action():
    c, s, r = fixture_values()
    b = ObservationBuilder(c)
    for i in range(13):
        s.gyro[:] = i
        b.accept_action(np.full(29, i-1))
        o = b.build(s, r, np.zeros((21, 192)), np.zeros(192), .4)
    h = o["policy"][0]
    np.testing.assert_array_equal(h[:30].reshape(10, 3)[:, 0], np.arange(3, 13))
    np.testing.assert_array_equal(h[610:900].reshape(10, 29)[:, 0], np.arange(2, 12))
    b.reset()
    o = b.build(s, r, np.zeros((21, 192)), np.zeros(192), 0)
    assert np.all(o["policy"][0, 610:900] == 0)
    assert np.all(o["policy"][0, :30] == 12)


def test_reference_twist_uses_reference_frame_not_live_frame():
    c, s, r = fixture_values()
    s.quat = Rotation.from_euler("z", .7).as_quat()[[3, 0, 1, 2]]
    r.root_quat[:] = Rotation.from_euler("z", np.pi/2).as_quat()[[3, 0, 1, 2]]
    o = ObservationBuilder(c).build(s, r, np.zeros((21, 192)), np.zeros(192), .4)
    np.testing.assert_allclose(o["augmentation"][0], [0, -.4, 0, 0, 0, 0], atol=1e-6)
    np.testing.assert_array_equal(o["augmentation"], o["q_task_cmd"])


def test_wrong_motor_order_refused():
    c, _, _ = fixture_values()
    c["joint_names"] = list(reversed(JOINT_NAMES))
    with pytest.raises(ValueError):
        ObservationBuilder(c)


@pytest.mark.parametrize("kind", ["state", "camera", "keyboard", "imu", "motor", "remote", "temperature"])
def test_health_failures_are_detected(kind):
    _, s, _ = fixture_values()
    image = command = 1.
    if kind == "state": s.received_at = 0
    if kind == "camera": image = 0
    if kind == "keyboard": command = 0
    if kind == "imu": s.quat[:] = 0
    if kind == "motor": s.motor_errors = (0, 1)
    if kind == "remote": s.remote_stop = True
    if kind == "temperature": s.temperature = 90
    assert health_fault(s, 1.01, image, command, Limits())


def test_keyboard_deadman_and_fault_latch():
    k = KeyboardIntent(.4)
    assert k.update(w=True) == 0
    assert k.update(arm=True) == 0
    assert k.update(start_policy=True) == 0
    assert not k.policy_enabled
    k.set_pose_ready()
    assert k.update(w=True) == 0
    assert k.update(start_policy=True,w=True) == 0
    assert not k.policy_enabled
    assert k.update(start_policy=True) == 0
    assert k.update(w=True) == .4
    assert k.update() == 0
    assert k.update(w=True, space=True) == 0
    assert k.update(w=True, focus=False) == 0
    assert k.update(arm=True, w=True) == 0


def test_tilt_does_not_stop_but_invalid_imu_and_remote_stop_do():
    _, s, _ = fixture_values()
    s.quat = np.array([0., 1., 0., 0.])
    assert health_fault(s, 1.01, 1., 1., Limits()) is None
    s.remote_stop = True
    assert health_fault(s, 1.01, 1., 1., Limits()) == 'remote_emergency_stop'
    s.remote_stop = False
    s.quat[:] = np.nan
    assert health_fault(s, 1.01, 1., 1., Limits()) == 'invalid_robot_state'


def test_target_bounds_nan_and_discontinuity():
    c, s, _ = fixture_values()
    assert target_fault(np.full(29, np.nan), s.q, s, c, Limits()) == "invalid_target"
    assert target_fault(np.ones(29)*4, s.q, s, c, Limits()) == "joint_limit"
    assert target_fault(np.ones(29)*.5, s.q, s, c, Limits()) == "target_discontinuity"
    assert target_fault(s.q, s.q, s, c, Limits()) is None


def test_conditioner_preserves_normal_targets_and_limits_small_slew():
    c, s, _ = fixture_values()
    safe, projection, slew = condition_target(np.ones(29)*.1, s.q, c, Limits())
    np.testing.assert_allclose(safe, .1)
    assert projection == slew == 0
    safe, _, slew = condition_target(np.ones(29)*.4, s.q, c, Limits())
    np.testing.assert_allclose(safe, .2)
    assert slew == pytest.approx(.2)
    assert target_fault(safe, s.q, s, c, Limits()) is None
    with pytest.raises(ValueError, match="excessive_target_slew"):
        condition_target(np.ones(29)*1.5, s.q, c, Limits())


def test_camera_pinhole_and_lossless_original_vibe_wire():
    rays = pinhole_rays()
    np.testing.assert_allclose(rays[31, 55:57, 0].sum(), 0, atol=1e-10)
    np.testing.assert_allclose(rays[31, :, 1], 0)
    rng = np.random.default_rng(2)
    bgr = rng.integers(0, 256, (63, 112, 3), np.uint8)
    packet = encode_packet(bgr)
    assert struct.unpack("<I", packet[:4])[0] == len(packet)-4
    rgb = np.asarray(Image.open(io.BytesIO(packet[4:])))
    np.testing.assert_array_equal(rgb[..., ::-1], bgr)
    image = np.zeros((4, 4, 3), np.uint8)
    image[:, 2:] = 100
    result = bilinear(image, np.array([[[1.5, 1.5]]]))
    np.testing.assert_array_equal(result, [[[50, 50, 50]]])
    with pytest.raises(ValueError):
        bilinear(image, np.array([[[-1., 0.]]]))
