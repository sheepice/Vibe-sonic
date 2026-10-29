"""Manual-stop integration with fake SDK/sensors: never opens robot channels."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_deploy_transport import sdk
from vibe_grail_joint.deploy import run, unitree, inference_process
from vibe_grail_joint.deploy.observations import RobotState
from vibe_grail_joint.deploy.safety import KeyboardIntent, Limits, prepare_target

ROOT = Path(__file__).resolve().parents[2]


def test_original_tracking_failure_is_diagnostic_in_raw_mode():
    manifest = json.loads((ROOT / 'models/velocity3999/bundle.json').read_text())
    report = json.loads((ROOT / 'records/hardware_preflight_20261019/source_velocity_flat_acceptance_v2.json').read_text())
    diag = report['guard_joint_diagnostic']
    target = np.asarray(diag['target'], np.float32)
    previous = np.asarray(diag['previous'], np.float32)
    # Reconstruct only the documented failing joint error, not a full replay.
    q = target.copy()
    q[4] -= diag['max_target_error_rad']
    state = RobotState(q, np.zeros(29), np.array([1., 0, 0, 0]), np.zeros(3), 0.)
    out, projection, slew, warning = prepare_target(target, previous, state, manifest['robot'], Limits())
    np.testing.assert_array_equal(out, target)
    assert projection == slew == 0.
    assert warning == 'target_tracking_error'


def test_raw_mode_preserves_out_of_range_and_fast_targets_but_rejects_nan():
    state = RobotState(np.zeros(29), np.zeros(29), np.array([1., 0, 0, 0]), np.zeros(3), 0.)
    contract = {'joint_limits': [[-1., 1.]] * 29}
    raw = np.full(29, 2., np.float32)
    out, _, _, warning = prepare_target(raw, state.q, state, contract, Limits())
    np.testing.assert_array_equal(out, raw)
    assert warning == 'joint_limit'
    with pytest.raises(ValueError, match='excessive_joint_limit_projection'):
        prepare_target(raw, state.q, state, contract, Limits(), actuator_guards=True)
    for invalid in (np.full(29, np.nan), np.full(29, np.inf), np.zeros(28)):
        with pytest.raises(ValueError, match='invalid_target'):
            prepare_target(invalid, state.q, state, contract, Limits())


def test_x_stop_cannot_rearm_or_be_overwritten_by_focus_or_escape():
    intent = KeyboardIntent()
    intent.update(arm=True)
    intent.set_pose_ready()
    intent.update(start_policy=True)
    assert intent.update(w=True) == .25
    assert intent.update(w=True, stop=True) == 0.
    assert intent.fault == 'keyboard_manual_stop'
    assert not intent.armed
    for command in ({'arm': True}, {'w': True}, {'focus': False}, {'escape': True}):
        assert intent.update(**command) == 0.
        assert not intent.armed and intent.fault == 'keyboard_manual_stop'


def test_cli_raw_control_requires_execute_and_calibration_but_not_old_report():
    base = ['--bundle', 'fake', '--interface', 'fake0', '--log', 'fake.jsonl']
    assert run.parse_args(base).mode == 'inspect'
    assert not run.parse_args(base).actuator_guards
    control = base + ['--mode', 'control', '--execute', '--calibration', 'measured.json']
    assert run.parse_args(control).acceptance_report is None
    for extra in (['--mode', 'control'], ['--mode', 'control', '--execute'], ['--execute']):
        with pytest.raises(SystemExit): run.parse_args(base + extra)
    with pytest.raises(SystemExit): run.parse_args(control + ['--actuator-guards'])
    with pytest.raises(SystemExit): run.parse_args(control + ['--acceptance-report', 'old.json'])
    assert run.parse_args(control + ['--actuator-guards', '--acceptance-report', 'old.json']).actuator_guards


def test_unmeasured_camera_is_explicit_control_only_and_never_claims_calibration():
    base=['--bundle','fake','--interface','fake0','--log','fake.jsonl']
    override=['--mode','control','--execute','--allow-unmeasured-camera']
    args=run.parse_args(base+override)
    assert args.allow_unmeasured_camera is True and args.calibration is None
    for extra in (['--allow-unmeasured-camera'],
                  ['--mode','shadow','--allow-unmeasured-camera'],
                  ['--mode','control','--allow-unmeasured-camera'],
                  override+['--calibration','measured.json']):
        with pytest.raises(SystemExit): run.parse_args(base+extra)


def test_short_remote_press_latches_before_any_writer_tick(sdk):
    calls, clock, contract, state = sdk
    backend = unitree.UnitreeBackend('fake0', contract, execute=True)
    backend._state(state())
    backend.arm()
    pressed = state(2)
    pressed.wireless_remote = bytes([0, 0, 8, 0] + [0] * 36)
    backend._state(pressed)
    assert backend.fault == 'remote_emergency_stop'
    backend._state(state(3))  # release before writer ever runs
    backend.submit(np.zeros(29), clock[0], clock[0])
    backend._write_once()
    assert all(m.kp == 0 for m in calls.publishers[0].messages[-1].motor_cmd[:29])
    with pytest.raises(RuntimeError, match='arm refused'): backend.arm()
    backend.close()


def test_keyboard_stop_packet_is_complete_and_latched(sdk):
    calls, clock, contract, state = sdk
    backend = unitree.UnitreeBackend('fake0', contract, execute=True)
    backend._state(state())
    backend.arm()
    backend.trip('keyboard_manual_stop')
    for tick in range(2, 5):
        backend._state(state(tick))
        backend.submit(np.ones(29), clock[0], clock[0])
        backend._write_once()
        sent = calls.publishers[0].messages[-1]
        assert all(m.kp == 0 and m.q == 0 and 0 < m.kd <= 2 for m in sent.motor_cmd[:29])
        assert sent.crc == backend.crc.Crc(sent)
    assert backend.fault == 'keyboard_manual_stop'
    backend.close()


def test_control_loop_sends_raw_target_then_x_holds_damping_without_resuming(sdk, monkeypatch, tmp_path):
    calls, clock, _, state_message = sdk
    manifest = json.loads((ROOT / 'models/velocity3999/bundle.json').read_text())
    default = np.asarray(manifest['robot']['default_joint_pos'], np.float32)
    target = default.copy()
    target[3] += 1.3  # exceeds both tracking and per-step diagnostic thresholds
    execution = SimpleNamespace(actions=0, backend=None, stop_packet_index=None, hold_polls=0)

    class Backend(unitree.UnitreeBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            execution.backend = self
            self.refresh()
        def refresh(self):
            msg = state_message(1 if self.last_tick is None else self.last_tick + 1)
            for motor, q in zip(msg.motor_state, default): motor.q = float(q)
            self._state(msg)
        def submit(self, *args):
            super().submit(*args)
            if self.armed: self._write_once()

    class Inference:
        def __init__(self, config): self.metadata = dict(motor_transport=False)
        def step(self, state, speed, **kwargs):
            result = dict(image_at=clock[0],frame_id=int(clock[0]*1000),
                          planner_s=0.,worker_s=.001,source_state_at=state.received_at)
            if kwargs.get('compute',True):
                execution.actions += 1
                scale = np.asarray(manifest['robot']['scale'],np.float32)
                result.update(action=(target-default)/scale,target=target.copy(),policy_s=.001)
            return result
        def close(self): pass

    class Window:
        def __init__(self, speed, on_stop):
            self.intent, self.on_stop, self.quit = KeyboardIntent(speed), on_stop, False
        def poll(self, status):
            execution.backend.refresh()
            if execution.actions:
                self.intent.update(stop=True)
                self.on_stop(self.intent.fault)
                if execution.stop_packet_index is None:
                    execution.stop_packet_index = len(calls.publishers[0].messages)
                execution.backend._write_once()
                if status.startswith('DAMPING LATCHED'):
                    execution.hold_polls += 1
                    assert self.intent.update(arm=True, w=True) == 0.
                    self.quit = execution.hold_polls == 3
                return 0.
            return self.intent.update(arm=not self.intent.armed,start_policy=self.intent.pose_ready)
        def close(self): pass

    def sleep(dt): clock[0] += max(dt, .02)
    monkeypatch.setattr(run, 'time', SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(unitree, 'UnitreeBackend', Backend)
    monkeypatch.setattr(inference_process,'InferenceProcess',Inference)
    from unitree_sdk2py.core.channel import ChannelPublisher
    def delayed_init(publisher):
        clock[0] += .25  # Real SDK publisher initialization blocks for discovery.
        execution.backend.refresh()
    monkeypatch.setattr(ChannelPublisher,'Init',delayed_init)
    monkeypatch.setattr(run, 'KeyboardWindow', Window)
    log = tmp_path / 'control.jsonl'
    args = run.parse_args(['--bundle', str(ROOT / 'models/velocity3999'), '--interface', 'fake0',
                           '--mode', 'control', '--execute', '--calibration', 'fake.json', '--seconds', '10',
                           '--log', str(log)])
    monkeypatch.setattr(run, 'parse_args', lambda: args)
    with pytest.raises(RuntimeError, match='keyboard_manual_stop'): run.main()
    assert execution.actions == 1 and execution.hold_polls == 3
    sent = calls.publishers[0].messages
    assert any(np.allclose([m.q for m in packet.motor_cmd[:29]], target) for packet in sent)
    assert all(all(m.kp == 0 for m in packet.motor_cmd[:29]) for packet in sent[execution.stop_packet_index:])
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    tick = next(row for row in rows if row['event'] == 'control_tick')
    assert tick['actuator_guards'] is False and tick['target_check'] == 'target_discontinuity'
    assert tick['target_step_max_rad'] > 1.2 and tick['target_error_max_rad'] > 1.2
    assert {'damping_hold', 'damping_hold_exit'} <= {row['event'] for row in rows}


@pytest.mark.parametrize('trigger', ['x', 'button', 'esc', 'focus'])
def test_real_sdl_events_latch_stop_before_display_update(trigger, monkeypatch, tmp_path):
    monkeypatch.setenv('SDL_VIDEODRIVER', 'dummy')
    monkeypatch.setenv('SDL_AUDIODRIVER', 'dummy')
    monkeypatch.setenv('PYGAME_HIDE_SUPPORT_PROMPT', '1')
    import pygame
    stopped = []
    window = run.KeyboardWindow(.25, on_stop=stopped.append)
    try:
        pygame.event.clear()
        monkeypatch.setattr(pygame.key, 'get_focused', lambda: trigger != 'focus')
        real_flip = pygame.display.flip
        def flip():
            assert stopped, 'Writer must be latched before display work'
            real_flip()
        monkeypatch.setattr(pygame.display, 'flip', flip)
        if trigger in ('x', 'esc'):
            key = pygame.K_x if trigger == 'x' else pygame.K_ESCAPE
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key))
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=key))
        if trigger == 'button':
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                               pos=window.stop_button.center))
        assert window.poll('CONTROL: target guards OFF') == 0.
        expected = {'x': 'keyboard_manual_stop', 'button': 'keyboard_manual_stop',
                    'esc': 'keyboard_emergency_stop', 'focus': 'keyboard_focus_lost'}[trigger]
        assert stopped[0] == expected
        assert window.quit == (trigger == 'esc')
        assert window.intent.update(arm=True, w=True) == 0.
        pygame.image.save(window.screen, str(tmp_path / f'{trigger}.png'))
    finally:
        window.close()
