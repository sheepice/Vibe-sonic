"""Require parity with the pre-optimization reference math, including boundaries."""
from dataclasses import fields
import threading

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from vibe_grail_joint.deploy.observations import Reference
from vibe_grail_joint.deploy.reference import OnlineReference, ReferenceWindow, interpolate, blend_prefix
from vibe_grail_joint.kinematics import angular_velocity, resample_qpos


def old_interpolate(poses, locations):
    indices = np.arange(len(poses))
    out = np.stack([np.interp(locations, indices, poses[:, j]) for j in range(36)], axis=-1)
    out[:, 3:7] = Slerp(indices, Rotation.from_quat(poses[:, [4, 5, 6, 3]]))(locations).as_quat()[:, [3, 0, 1, 2]]
    return out.astype(np.float32)


def old_sample(poses, at):
    locations = at + np.arange(10) * 5
    qpos = old_interpolate(poses, locations)
    qvel = np.gradient(poses[:, 7:], .02, axis=0)
    joint_vel = np.stack([np.interp(locations, np.arange(len(poses)), qvel[:, j]) for j in range(29)], axis=-1)
    lin = np.gradient(poses[:, :3], .02, axis=0)
    ang = angular_velocity(poses[:, 3:7], .02)
    twist = [np.array([np.interp(at, np.arange(len(poses)), data[:, j]) for j in range(3)]) for data in (lin, ang)]
    return Reference(qpos[:, 7:], joint_vel.astype(np.float32), qpos[:, 3:7], *twist)


@pytest.mark.parametrize('seed', [1, 8, 21])
def test_cached_reference_matches_original_across_quaternion_signs_and_endpoints(seed):
    rng = np.random.default_rng(seed)
    poses = rng.normal(size=(96, 36)).astype(np.float32)
    poses[:, 3:7] = Rotation.random(96, random_state=rng).as_quat()[:, [3, 0, 1, 2]]
    poses[::3, 3:7] *= -1
    window = ReferenceWindow(poses)
    for at in (0., .00001, .1, 1., 8.333, 49.99999, 50.):
        expected, actual = old_sample(poses, at), window.sample(at)
        for f in fields(Reference):
            np.testing.assert_allclose(getattr(actual, f.name), getattr(expected, f.name), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(interpolate(poses, at + np.arange(10)*5),
                                   old_interpolate(poses, at + np.arange(10)*5), rtol=1e-6, atol=1e-6)
    frozen = window.sample(0).joint_pos.copy()
    poses[:, 7:] = 0
    np.testing.assert_array_equal(window.sample(0).joint_pos, frozen)
    for at in (-.0001, 50.0001, float('nan')):
        with pytest.raises(RuntimeError, match='horizon'):
            window.sample(at)


def test_reference_staleness_limit_is_unchanged():
    ref = object.__new__(OnlineReference)
    ref.lock = threading.Lock()
    ref.window, ref.start, ref.updated_at, ref.seconds = None, 0, 1., .2
    with pytest.raises(RuntimeError, match='stale'):
        ref.sample(1.45001)


@pytest.mark.parametrize('source_dt,target_dt', [(1/30,.02), (.02,.02), (.025,.04)])
def test_motion_resampling_matches_original(source_dt, target_dt):
    rng = np.random.default_rng(8)
    poses = rng.normal(size=(44, 36)).astype(np.float32)
    poses[:, 3:7] = Rotation.random(44, random_state=rng).as_quat()[:, [3,0,1,2]]
    qpos = poses.astype(np.float64)
    times = np.arange(len(poses))*source_dt
    samples = np.arange(int(np.floor(times[-1]/target_dt + 1e-7))+1)*target_dt
    samples = np.minimum(samples, times[-1])
    expected = np.stack([np.interp(samples, times, qpos[:,i]) for i in range(36)],axis=1)
    quat = qpos[:,[4,5,6,3]] / np.linalg.norm(qpos[:,3:7],axis=1)[:,None]
    expected[:,3:7] = Slerp(times,Rotation.from_quat(quat))(samples).as_quat()[:,[3,0,1,2]]
    np.testing.assert_allclose(resample_qpos(poses, source_dt, target_dt), expected.astype(np.float32), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('n', [1,2,8])
def test_batch_blend_matches_pairwise_slerp(n):
    rng = np.random.default_rng(n)
    old, prediction = rng.normal(size=(n,36)).astype(np.float32), rng.normal(size=(20,36)).astype(np.float32)
    old[:,3:7] = Rotation.random(n,random_state=rng).as_quat()[:,[3,0,1,2]]
    prediction[:,3:7] = Rotation.random(20,random_state=rng).as_quat()[:,[3,0,1,2]]
    expected = prediction.copy()
    for i,weight in enumerate(np.linspace(0,1,n)):
        quats = [old[i,[4,5,6,3]],expected[i,[4,5,6,3]].copy()]
        expected[i] = old[i]*(1-weight)+expected[i]*weight
        expected[i,3:7] = Slerp([0,1],Rotation.from_quat(quats))([weight]).as_quat()[0,[3,0,1,2]]
    blend_prefix(old,prediction)
    np.testing.assert_allclose(prediction,expected,rtol=1e-6,atol=1e-6)
