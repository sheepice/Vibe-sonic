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
        from unitree_sdk2py.utils.crc import CRC
        self.contract, self.execute = contract, execute
        self.limits = Limits()
        self.crc = CRC()
        self.lock = threading.Lock()
        self.state = None
        self.last_tick = None
        self.crc_failures = 0
        self.target = None
        self.armed = False
        self.armed_machine = None
        self.fault = None
        self.publisher = None
        self.stop = threading.Event()
        self.release_sport_mode = release_sport_mode
        if release_sport_mode and not execute:
            raise ValueError("Shadow/read-only modes cannot release the robot controller")
        ChannelFactoryInitialize(domain, interface)
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self._state, 1)
        self.thread = threading.Thread(target=self._writer, name="g1-lowcmd-500hz", daemon=True)
        self.thread.start()

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
                           max(max(m.temperature) for m in motors), bool(buttons & (1 << 3)))
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
        if fault:
            raise RuntimeError(fault)
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.submit(state.q, now, now)
        self.armed_machine = state.mode_machine
        self.armed = True

    def submit(self, q, image_at, command_at):
        q = np.asarray(q, np.float32)
        if q.shape != (29,) or not np.isfinite(q).all():
            self.trip("invalid_submitted_target")
            raise ValueError("Invalid 29-joint target")
        with self.lock:
            self.target = (q.copy(), time.monotonic(), image_at, command_at)

    def trip(self, reason):
        if self.fault is None:
            self.fault = str(reason)

    def _writer(self):
        deadline = time.monotonic()
        while not self.stop.is_set():
            deadline += .002
            if self.armed:
                try:
                    self._write_once()
                except Exception as exc:
                    self.trip(f"writer_exception: {exc}")
            self.stop.wait(max(0., deadline - time.monotonic()))
            if time.monotonic() - deadline > .02:
                if self.armed:
                    self.trip("writer_deadline")
                deadline = time.monotonic()

    def _write_once(self):
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        with self.lock:
            state, target = self.state, self.target
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
        msg = unitree_hg_msg_dds__LowCmd_()
        msg.mode_pr = 0
        msg.mode_machine = state.mode_machine
        for i in range(29):
            m = msg.motor_cmd[i]
            m.mode, m.dq, m.tau = 1, 0., 0.
            m.q = float(q[i]) if not self.fault else 0.
            m.kp = float(self.contract["stiffness"][i]) if not self.fault else 0.
            m.kd = float(self.contract["damping"][i]) if not self.fault else min(2., float(self.contract["damping"][i]))
        # Unused motor slots stay disabled; CRC computed after all fields.
        msg.crc = self.crc.Crc(msg)
        if not self.publisher.Write(msg):
            self.trip("lowcmd_write_failed")

    def close(self):
        if self.armed:
            self.trip("shutdown")
            time.sleep(.25)  # bounded damping handoff; physical support remains necessary
        self.stop.set()
        self.thread.join(timeout=1.)
        self.subscriber.Close()
        if self.publisher:
            self.publisher.Close()
