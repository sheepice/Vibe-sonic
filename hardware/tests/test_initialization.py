"""Supported initialization regression; SDK integration is entirely fake."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_deploy_transport import sdk
from vibe_grail_joint.deploy.initialization import NominalPoseTransition, pose_report

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((ROOT/'models/velocity3999/bundle.json').read_text())['robot']
DEFAULT = np.asarray(CONTRACT['default_joint_pos'], np.float32)


def inspected_pose():
    q = DEFAULT.copy()
    q[18], q[25] = 1.3281817, 1.4769958
    q[0], q[3], q[4] = .10537596, .26128805, -.7357664
    return q


def test_observed_elbow_pose_is_diagnostic_for_both_stages():
    state = SimpleNamespace(q=inspected_pose(), dq=np.full(29, .0515))
    assert pose_report(state, CONTRACT, initializing=True)['admitted']
    report = pose_report(state, CONTRACT, initializing=False)
    assert report['admitted'] and report['thresholds_diagnostic_only']
    assert {v['joint'] for v in report['violations']} == {'left_elbow_joint', 'right_elbow_joint'}


@pytest.mark.parametrize('index', [0, 3, 12, 15, 18, 25])
def test_pose_deviation_is_logged_without_requiring_manual_alignment(index):
    q = DEFAULT.copy()
    q[index] += 1.01 if index in (18, 25) else .51
    report = pose_report(SimpleNamespace(q=q, dq=np.zeros(29)), CONTRACT, initializing=True)
    assert report['admitted']
    assert report['violations'][0]['joint'] == CONTRACT['joint_names'][index]
    assert report['violations'][0]['kind'] == 'pose_error'


def test_joint_speed_is_diagnostic():
    dq = np.zeros(29)
    dq[3] = .51
    report = pose_report(SimpleNamespace(q=DEFAULT, dq=dq), CONTRACT, initializing=True)
    assert report['admitted']
    assert report['violations'] == [dict(joint='left_knee_joint', kind='joint_speed', value=.51, limit=.5)]


@pytest.mark.parametrize('bad', [np.full(29, np.nan), np.zeros(28), np.full(29, np.inf)])
def test_invalid_measurements_still_block_initialization(bad):
    with pytest.raises(ValueError, match='admission data'):
        pose_report(SimpleNamespace(q=bad, dq=np.zeros(29)), CONTRACT, initializing=True)


def test_full_joint_range_start_uses_actual_pose_and_speed_bound():
    bounds = np.asarray(CONTRACT['joint_limits'], np.float32)
    for start in (bounds[:, 0], bounds[:, 1]):
        assert pose_report(SimpleNamespace(q=start, dq=np.zeros(29)), CONTRACT, initializing=True)['admitted']
        ramp = NominalPoseTransition(start, DEFAULT)
        np.testing.assert_array_equal(ramp.sample(0)[0], start)
        times = np.linspace(0, ramp.duration, 1001)
        targets = np.array([ramp.sample(t)[0] for t in times])
        assert abs(np.diff(targets, axis=0)/np.diff(times)[:, None]).max() < .251


def test_interpolation_endpoints_speed_bounds_and_smooth_start_stop():
    start = inspected_pose()
    ramp = NominalPoseTransition(start, DEFAULT)
    assert 6.5 < ramp.duration < 6.7
    times = np.linspace(0., ramp.duration, 1001)
    targets = np.array([ramp.sample(t)[0] for t in times])
    np.testing.assert_array_equal(ramp.sample(-1.)[0], start)
    np.testing.assert_array_equal(ramp.sample(ramp.duration+1.)[0], DEFAULT)
    assert not ramp.sample(ramp.duration-.01)[1] and ramp.sample(ramp.duration)[1]
    assert np.all(targets >= np.minimum(start, DEFAULT)-1e-7)
    assert np.all(targets <= np.maximum(start, DEFAULT)+1e-7)
    rates = np.diff(targets, axis=0)/np.diff(times)[:, None]
    assert np.max(abs(rates)) <= .2501
    assert np.max(abs(rates[[0, -1]])) < .001
    with pytest.raises(ValueError): ramp.sample(float('nan'))


@pytest.mark.parametrize('stop_during_ramp', [False, True])
def test_interpolation_and_select_with_fake_sdk(sdk, monkeypatch, tmp_path, stop_during_ramp):
    from vibe_grail_joint.deploy import run, unitree, inference_process, remote
    calls, clock, _, state_message = sdk
    submitted, computed = [], []
    stop_at = 2. if stop_during_ramp else 11.4

    def buttons():
        t = clock[0]-100.
        if t >= stop_at+.1: return remote.Y
        if t >= stop_at: return remote.SELECT
        if .2 <= t < 1.4: return remote.R1|remote.START
        if 10. <= t < 11.2: return remote.A
        return 0

    class Backend(unitree.UnitreeBackend):
        def __init__(self, *a, **kw):
            self.position = inspected_pose()
            super().__init__(*a, **kw)
            self.refresh()
        def refresh(self):
            msg = state_message(1 if self.last_tick is None else self.last_tick+1)
            for m, q in zip(msg.motor_state, self.position): m.q = float(q)
            msg.wireless_remote = bytes(2)+buttons().to_bytes(2, 'little')+bytes(36)
            self._state(msg)
        def latest(self):
            self.refresh()
            if self.armed: self._write_once()
            return super().latest()
        def submit(self, q, image_at, command_at):
            super().submit(q, image_at, command_at)
            if self.armed:
                submitted.append((clock[0], q.copy()))
                self.position = q.copy()
                self.refresh()
                self._write_once()

    class Inference:
        def __init__(self, config): self.metadata = dict(motor_transport=False)
        def step(self, state, speed, **kwargs):
            result = dict(image_at=clock[0], frame_id=1, planner_s=0., worker_s=0.)
            if kwargs.get('compute', True):
                computed.append((clock[0], state.q.copy()))
                result.update(action=np.zeros(29, np.float32), target=DEFAULT.copy(), policy_s=0.)
            return result
        def close(self): pass

    def sleep(dt): clock[0] += max(dt, .02)
    fake_time = SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep)
    monkeypatch.setattr(run, 'time', fake_time)
    monkeypatch.setattr(remote, 'time', fake_time)
    monkeypatch.setattr(unitree, 'UnitreeBackend', Backend)
    monkeypatch.setattr(inference_process, 'InferenceProcess', Inference)
    log = tmp_path/'interpolation.jsonl'
    args = run.parse_args(['--bundle', str(ROOT/'models/velocity3999'), '--interface', 'fake0',
        '--mode', 'control', '--execute', '--input', 'remote', '--allow-unmeasured-camera',
        '--seconds', '15', '--log', str(log)])
    monkeypatch.setattr(run, 'parse_args', lambda: args)
    with pytest.raises(RuntimeError, match='remote_emergency_stop'): run.main()
    rows = [json.loads(s) for s in log.read_text().splitlines()]
    checks = [r for r in rows if r['event']=='pose_check']
    assert checks[0]['admitted'] and checks[0]['max_pose_error_rad'] > .87
    np.testing.assert_array_equal(submitted[0][1], inspected_pose())
    times, targets = zip(*submitted)
    rates = np.diff(np.array(targets), axis=0)/np.diff(times)[:, None]
    assert np.max(abs(rates)) <= .2501
    if stop_during_ramp:
        assert not computed
        assert not any(r['event']=='nominal_pose_ready' for r in rows)
    else:
        assert computed and computed[0][0] >= 111.
        np.testing.assert_array_equal(computed[0][1], DEFAULT)
        assert any(r['event']=='nominal_pose_hold' for r in rows)
        assert checks[-1]['stage']=='policy_start' and checks[-1]['admitted']
    assert len(calls.publishers)==1 and calls.releases==0
    assert all(m.kp==0 for m in calls.publishers[0].messages[-1].motor_cmd[:29])
