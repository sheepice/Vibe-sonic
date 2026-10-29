"""G1 SDK2 backend. Construction is read-only; publishing requires explicit arm.

The writer rechecks state and policy watchdogs at 500 Hz. It is a Python
prototype, not a hard-real-time certified safety controller.
"""
import threading
import time

import numpy as np

from .observations import RobotState
from .safety import Limits, health_fault


class UnitreeBackend:
    def __init__(self, interface, contract, *, domain=0, execute=False, release_sport_mode=False):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        from .crc import optimized_crc
        from .serialization import lowcmd_factory
        self.contract, self.execute = contract, execute
        self.limits = Limits()
        self.crc = optimized_crc()
        self.make_lowcmd = lowcmd_factory()
        self.command_message = self.make_lowcmd()
        self.lock = threading.Lock()
        self.state = None
        self.last_tick = None
        self.crc_failures = 0
        self.target = None
        self.armed = False
        self.armed_machine = None
        self.fault = None
        self.publisher = None
        self.last_write_timing = None
        self.writer_stats = dict(cycles=0, period_s=.002, deadline_limit_s=.02,
                                 deadline_overruns=0,
                                 max_lateness_s=0., max_write_s=0., max_wakeup_lateness_s=0.)
        self.first_deadline_overrun = None
        self.stop = threading.Event()
        self.release_sport_mode = release_sport_mode
        if release_sport_mode and not execute:
            raise ValueError("Shadow/read-only modes cannot release the robot controller")
        ChannelFactoryInitialize(domain, interface)
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._state, 1)
        self.thread = None  # Read-only modes never start a motor writer thread.

    def _state(self, msg):
        if self.crc.Crc(msg) != msg.crc:
            self.crc_failures += 1
            return
        if msg.tick == self.last_tick:
            return  # retransmission must not refresh freshness
        self.last_tick = msg.tick
        motors = msg.motor_state[:29]
        buttons = int.from_bytes(bytes(msg.wireless_remote[2:4]), "little")
        # Unitree wireless SELECT is bit 3; verify the physical button in read-only mode.
        state = RobotState(np.array([m.q for m in motors], np.float32),
                           np.array([m.dq for m in motors], np.float32),
                           np.array(msg.imu_state.quaternion, np.float32),
                           np.array(msg.imu_state.gyroscope, np.float32),
                           time.monotonic(), int(msg.tick), int(msg.mode_machine),
                           tuple(int(m.motorstate) for m in motors),
                           max(max(m.temperature) for m in motors), bool(buttons & (1 << 3)),buttons)
        # Latch even a short button press in the DDS callback, independently
        # of policy/UI progress. Releasing SELECT must not clear the stop.
        if state.remote_stop:
            self.trip("remote_emergency_stop")
        with self.lock:
            self.state = state

    def latest(self):
        with self.lock:
            if self.state is None:
                raise RuntimeError("Waiting for valid rt/lowstate")
            return self.state

    def arm(self):
        if not self.execute or self.fault or self.armed:
            raise RuntimeError("Control arm refused")
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
        switcher = MotionSwitcherClient()
        switcher.SetTimeout(2.)
        switcher.Init()
        code, mode = switcher.CheckMode()
        if code != 0:
            raise RuntimeError(f"Cannot check active controller: {code}")
        if mode.get("name"):
            if not self.release_sport_mode:
                raise RuntimeError("Another robot mode is active; release it explicitly before control")
            if switcher.ReleaseMode()[0] != 0:
                raise RuntimeError("MotionSwitcher ReleaseMode failed")
            deadline = time.monotonic() + 2.
            while time.monotonic() < deadline:
                code, mode = switcher.CheckMode()
                if code == 0 and not mode.get("name"):
                    break
                time.sleep(.02)
            else:
                raise RuntimeError("Robot high-level controller did not release")
        state = self.latest()
        now = time.monotonic()
        fault = health_fault(state, now, now, now, self.limits)
        if fault or self.fault:
            raise RuntimeError(fault or self.fault)
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        # Publisher discovery can block. Refresh state after initialization so
        # the first held target does not use a pre-handoff sensor snapshot.
        state = self.latest()
        now = time.monotonic()
        fault = health_fault(state,now,now,now,self.limits)
        if fault or self.fault:
            raise RuntimeError(fault or self.fault)
        self.submit(state.q, now, now)
        self.armed_machine = state.mode_machine
        self.armed = True
        self.thread = threading.Thread(target=self._writer,name="g1-lowcmd-500hz",daemon=True)
        self.thread.start()

    def submit(self, q, image_at, command_at):
        q = np.asarray(q, np.float32)
        if q.shape != (29,) or not np.isfinite(q).all():
            self.trip("invalid_submitted_target")
            raise ValueError("Invalid 29-joint target")
        with self.lock:
            self.target = (q.copy(), time.monotonic(), image_at, command_at)

    def trip(self, reason):
        with self.lock:
            if self.fault is None:
                self.fault = str(reason) or "unspecified_stop"

    def _writer(self):
        deadline = time.monotonic()
        while not self.stop.is_set():
            deadline += .002
            begin = time.monotonic()
            if self.armed:
                try:
                    self._write_once()
                except Exception as exc:
                    self.trip(f"writer_exception: {exc}")
            finished = time.monotonic()
            self.stop.wait(max(0., deadline - time.monotonic()))
            woke = time.monotonic()
            late = max(0., woke-deadline)
            self.writer_stats['cycles'] += 1
            self.writer_stats['max_lateness_s'] = max(self.writer_stats['max_lateness_s'], late)
            self.writer_stats['max_write_s'] = max(self.writer_stats['max_write_s'], finished-begin)
            self.writer_stats['max_wakeup_lateness_s'] = max(
                self.writer_stats['max_wakeup_lateness_s'], woke-max(deadline, finished))
            if late > .02:
                if self.armed:
                    self.writer_stats['deadline_overruns'] += 1
                    if self.first_deadline_overrun is None:
                        self.first_deadline_overrun = dict(
                            lateness_s=late, write_s=finished-begin,
                            start_lateness_s=max(0., begin-(deadline-.002)),
                            wakeup_lateness_s=max(0., woke-max(deadline, finished)),
                            phases=self.last_write_timing,
                            state_age_s=woke-self.state.received_at if self.state else None)
                    # A late scheduler wakeup is diagnostic. _write_once still
                    # enforces real state/target loss and SELECT on every write.
                deadline = time.monotonic()

    def writer_diagnostics(self):
        return dict(**self.writer_stats, last_write=self.last_write_timing,
                    deadline_fault=None, first_deadline_overrun=self.first_deadline_overrun)

    def _write_once(self):
        started = time.monotonic()
        with self.lock:
            state, target = self.state, self.target
        state_copied = time.monotonic()
        if state is None or target is None:
            self.trip("missing_control_state")
            return
        q, produced, image_at, command_at = target
        now = time.monotonic()
        fault = health_fault(state, now, image_at, command_at, self.limits)
        if fault:
            self.trip(fault)
        if state.mode_machine != self.armed_machine:
            self.trip("robot_mode_machine_changed")
        if now - produced > self.limits.target_age_s:
            self.trip("policy_watchdog")
        checked = time.monotonic()
        # Only the writer owns this object; SDK Write serializes synchronously.
        msg = self.command_message
        msg.mode_pr = 0
        msg.mode_machine = state.mode_machine
        # Serialize a complete packet with stop latching: after trip() returns,
        # no normal-PD packet can be published and a packet cannot mix modes.
        with self.lock:
            acquired = time.monotonic()
            for i in range(29):
                m = msg.motor_cmd[i]
                m.mode, m.dq, m.tau = 1, 0., 0.
                m.q = float(q[i]) if not self.fault else 0.
                m.kp = float(self.contract["stiffness"][i]) if not self.fault else 0.
                m.kd = float(self.contract["damping"][i]) if not self.fault else min(2., float(self.contract["damping"][i]))
            # Unused motor slots stay disabled; CRC computed after all fields.
            msg.crc = self.crc.Crc(msg)
            packed = time.monotonic()
            written = self.publisher.Write(msg)
            sent = time.monotonic()
        self.last_write_timing = dict(state_lock_s=state_copied-started,
            health_s=checked-state_copied, packet_lock_s=acquired-checked,
            pack_crc_s=packed-acquired, dds_write_s=sent-packed, total_s=sent-started)
        if not written:
            self.trip("lowcmd_write_failed")

    def close(self):
        if self.armed:
            self.trip("shutdown")
            time.sleep(.25)  # bounded damping handoff; physical support remains necessary
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1.)
        self.subscriber.Close()
        if self.publisher:
            self.publisher.Close()
