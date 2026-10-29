"""SDK boundary tests with fake messages: no hardware or DDS connection is opened."""
import copy
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from vibe_grail_joint.deploy import unitree
from vibe_grail_joint.deploy.observations import JOINT_NAMES


@pytest.fixture
def sdk(monkeypatch):
    clock = [100.]
    calls = SimpleNamespace(publishers=[], subscribers=[], releases=0, active_mode="", write_ok=True)

    class Thread:
        def __init__(self, **kwargs): pass
        def start(self): pass
        def join(self, **kwargs): pass

    class Subscriber:
        def __init__(self, topic, kind):
            assert topic == "rt/lowstate"
            calls.subscribers.append(self)
        def Init(self, callback, queue): self.callback = callback
        def Close(self): pass

    class Publisher:
        def __init__(self, topic, kind):
            assert topic == "rt/lowcmd"
            self.messages = []
            calls.publishers.append(self)
        def Init(self): pass
        def Write(self, message):
            self.messages.append(copy.deepcopy(message))
            return calls.write_ok
        def Close(self): pass

    class CRC:
        def Crc(self, message):
            if hasattr(message, "motor_state"):
                return 123
            return message.mode_machine + sum(m.mode + round(m.kp) for m in message.motor_cmd)

    class Switcher:
        def SetTimeout(self, timeout): pass
        def Init(self): pass
        def CheckMode(self): return 0, {"name": calls.active_mode}
        def ReleaseMode(self):
            calls.releases += 1
            calls.active_mode = ""
            return 0, {}

    def lowcmd():
        return SimpleNamespace(mode_machine=0, mode_pr=0, crc=0, motor_cmd=[
            SimpleNamespace(mode=0, q=0., dq=0., tau=0., kp=0., kd=0.) for _ in range(35)])

    def install(name, **attributes):
        parts = name.split(".")
        for length in range(1, len(parts)+1):
            key = ".".join(parts[:length])
            if key not in sys.modules:
                monkeypatch.setitem(sys.modules, key, ModuleType(key))
        for key, value in attributes.items():
            monkeypatch.setattr(sys.modules[name], key, value, raising=False)

    install("unitree_sdk2py.core.channel", ChannelFactoryInitialize=lambda *a: None,
            ChannelSubscriber=Subscriber, ChannelPublisher=Publisher)
    install("unitree_sdk2py.idl.unitree_hg.msg.dds_", LowState_=object, LowCmd_=object)
    install("unitree_sdk2py.utils.crc", CRC=CRC)
    install("unitree_sdk2py.idl.default", unitree_hg_msg_dds__LowCmd_=lowcmd)
    install("unitree_sdk2py.comm.motion_switcher.motion_switcher_client", MotionSwitcherClient=Switcher)
    monkeypatch.setattr(unitree, "threading", SimpleNamespace(
        Lock=threading.Lock, Event=threading.Event, Thread=Thread))
    monkeypatch.setattr(unitree, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda dt: None))
    contract = dict(joint_names=JOINT_NAMES, stiffness=[40.]*29, damping=[1.5]*29)

    def state(tick=1):
        return SimpleNamespace(crc=123, tick=tick, mode_machine=5, wireless_remote=bytes(40),
            imu_state=SimpleNamespace(quaternion=[1., 0, 0, 0], gyroscope=[0., 0, 0]),
            motor_state=[SimpleNamespace(q=i/100, dq=0., motorstate=0, temperature=[30, 32])
                         for i in range(35)])
    return calls, clock, contract, state


def test_readonly_never_creates_motor_publisher_and_bad_crc_does_not_refresh(sdk):
    calls, clock, contract, state = sdk
    backend = unitree.UnitreeBackend("fake0", contract)
    assert calls.publishers == []
    assert backend.thread is None
    invalid = state()
    invalid.crc = 0
    backend._state(invalid)
    assert backend.state is None and backend.crc_failures == 1
    backend._state(state())
    received = backend.latest().received_at
    clock[0] += .01
    backend._state(state())
    assert backend.latest().received_at == received
    with pytest.raises(RuntimeError, match="arm refused"):
        backend.arm()
    assert calls.publishers == [] and calls.releases == 0
    backend.close()


def test_packet_mapping_crc_and_remote_stop_remain_latched(sdk):
    calls, clock, contract, state = sdk
    backend = unitree.UnitreeBackend("fake0", contract, execute=True)
    backend._state(state())
    backend.arm()
    q = np.linspace(-.2, .2, 29, dtype=np.float32)
    backend.submit(q, clock[0], clock[0])
    backend._write_once()
    sent = calls.publishers[0].messages[-1]
    assert sent.mode_pr == 0 and sent.mode_machine == 5
    np.testing.assert_array_equal([m.q for m in sent.motor_cmd[:29]], q)
    assert all(m.mode == 1 and m.kp == 40. and m.dq == m.tau == 0 for m in sent.motor_cmd[:29])
    assert all(m.mode == 0 for m in sent.motor_cmd[29:])
    assert sent.crc == backend.crc.Crc(sent)
    stopped = state(2)
    stopped.wireless_remote = bytes([0, 0, 8, 0] + [0]*36)
    backend._state(stopped)
    backend._write_once()
    assert backend.fault == "remote_emergency_stop"
    backend._state(state(3))
    backend.submit(q, clock[0], clock[0])
    backend._write_once()
    assert all(m.kp == 0. and 0 < m.kd <= 2 for m in calls.publishers[0].messages[-1].motor_cmd[:29])
    backend.close()


@pytest.mark.parametrize("failure", ["state", "target", "camera", "mode", "nan"])
def test_writer_damps_on_independent_faults(sdk, failure):
    calls, clock, contract, state = sdk
    backend = unitree.UnitreeBackend("fake0", contract, execute=True)
    backend._state(state())
    backend.arm()
    clock[0] += .07
    if failure != "state": backend._state(state(2))
    if failure not in ("target", "nan"):
        backend.submit(np.zeros(29), clock[0]-(.2 if failure == "camera" else 0), clock[0])
    if failure == "mode":
        changed = state(3)
        changed.mode_machine = 4
        backend._state(changed)
    if failure == "nan":
        with pytest.raises(ValueError): backend.submit(np.full(29, np.nan), clock[0], clock[0])
    backend._write_once()
    assert backend.fault
    sent = calls.publishers[0].messages[-1]
    assert all(m.kp == 0 and np.isfinite(m.q) for m in sent.motor_cmd[:29])
    backend.close()


def test_controller_release_requires_explicit_option_and_checks_tuple_result(sdk):
    calls, _, contract, state = sdk
    calls.active_mode = "normal"
    backend = unitree.UnitreeBackend("fake0", contract, execute=True)
    backend._state(state())
    with pytest.raises(RuntimeError, match="Another robot mode"):
        backend.arm()
    assert calls.releases == 0 and calls.publishers == []
    backend.release_sport_mode = True
    backend.arm()
    assert calls.releases == 1 and backend.armed
    backend.close()


def test_write_failure_is_latched_even_when_robot_state_is_fresh(sdk):
    calls, _, contract, state = sdk
    backend = unitree.UnitreeBackend("fake0", contract, execute=True)
    backend._state(state())
    backend.arm()
    calls.write_ok = False
    backend._write_once()
    assert backend.fault == "lowcmd_write_failed"
    calls.write_ok = True
    backend._write_once()
    assert all(m.kp == 0 for m in calls.publishers[0].messages[-1].motor_cmd[:29])
    backend.close()


def test_empty_stop_reason_still_produces_latched_damping(sdk):
    calls,_,contract,state = sdk
    backend = unitree.UnitreeBackend('fake0',contract,execute=True)
    backend._state(state())
    backend.arm()
    backend.trip('')
    backend._write_once()
    assert backend.fault == 'unspecified_stop'
    assert all(m.kp == 0 for m in calls.publishers[0].messages[-1].motor_cmd[:29])
    with pytest.raises(RuntimeError,match='arm refused'):
        backend.arm()
    backend.close()


@pytest.mark.parametrize('stop_during_discovery',[False,True])
def test_arm_refreshes_state_and_remote_stop_after_publisher_discovery(sdk,monkeypatch,stop_during_discovery):
    calls,clock,contract,state = sdk
    backend = unitree.UnitreeBackend('fake0',contract,execute=True)
    backend._state(state())
    from unitree_sdk2py.core.channel import ChannelPublisher
    def delayed_init(publisher):
        clock[0] += .25
        changed = state(2)
        for motor in changed.motor_state:
            motor.q += .1
        if stop_during_discovery:
            changed.wireless_remote = bytes([0,0,8,0]+[0]*36)
        backend._state(changed)
    monkeypatch.setattr(ChannelPublisher,'Init',delayed_init)
    if stop_during_discovery:
        with pytest.raises(RuntimeError,match='remote_emergency_stop'):
            backend.arm()
        assert not backend.armed and calls.publishers[0].messages == []
    else:
        backend.arm()
        np.testing.assert_array_equal(backend.target[0],backend.latest().q)
        assert backend.target[1] == clock[0]
    backend.close()


@pytest.mark.parametrize('write_delay,wake_delay,expected_warning', [
    (.001, 0., False), (.025, 0., True), (0., .025, True)])
def test_writer_timing_distinguishes_write_and_wakeup_delays(
        sdk, monkeypatch, write_delay, wake_delay, expected_warning):
    calls, clock, contract, state = sdk
    backend = unitree.UnitreeBackend('fake0', contract, execute=True)
    backend._state(state())
    backend.arm()
    publisher = calls.publishers[0]
    original = publisher.Write
    def write(message):
        clock[0] += write_delay
        return original(message)
    monkeypatch.setattr(publisher, 'Write', write)

    class OneCycle:
        done = False
        def is_set(self): return self.done
        def wait(self, delay):
            clock[0] += delay+wake_delay
            self.done = True
        def set(self): self.done = True
    backend.stop = OneCycle()
    backend._writer()
    diagnostic = backend.writer_diagnostics()
    assert diagnostic['cycles'] == 1
    assert diagnostic['last_write']['dds_write_s'] == pytest.approx(write_delay)
    assert diagnostic['max_wakeup_lateness_s'] == pytest.approx(wake_delay)
    assert backend.fault is None and diagnostic['deadline_fault'] is None
    assert diagnostic['deadline_overruns'] == int(expected_warning)
    if expected_warning:
        assert diagnostic['first_deadline_overrun']['lateness_s'] > .02
    backend._write_once()
    assert any(m.kp > 0 for m in publisher.messages[-1].motor_cmd[:29])
    backend.trip('remote_emergency_stop')
    backend._write_once()
    assert all(m.kp == 0 for m in publisher.messages[-1].motor_cmd[:29])
    backend.close()
