"""Isolate 5 Hz planning from the high-frequency Python DDS receiver.

Only reference snapshots cross the pipe. This process cannot access the robot
backend and never publishes motor commands. Source timestamps and staleness
limits are identical to the threaded planner.
"""
import multiprocessing as mp
import threading
import time
import traceback

import numpy as np

from .reference import OnlineReference


def _planner_main(path, contract, speed, stop, output, reference_factory):
    try:
        reference = reference_factory(path, contract)
        origin = time.monotonic()
        while not stop.value:
            begin = time.monotonic()
            reference.replan(begin-origin, speed.value)
            output.send(('reference', origin, reference.start, reference.updated_at,
                         reference.seconds, reference.window))
            time.sleep(max(0., .2-(time.monotonic()-begin)))
    except BaseException:
        try:
            output.send(('error', traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        output.close()


class ProcessPlannerWorker:
    def __init__(self, planner, contract, *, reference_factory=OnlineReference):
        context = mp.get_context('spawn')
        # Single-writer, aligned scalar mailboxes on our 64-bit hosts. Avoid
        # process-shared locks: killing a child while it owns one can otherwise
        # deadlock the controller's shutdown or speed setter.
        self._speed = context.RawValue('d', 0.)
        self._stop = context.RawValue('b', 0)
        self.stop = threading.Event()
        self._input, output = context.Pipe(duplex=False)
        # Parent uses only OnlineReference.sample; it has no planner session.
        self.reference = OnlineReference('', contract, session_override=object())
        self.origin = None
        self.error = None
        self.process = context.Process(target=_planner_main,
                                       args=(str(planner), contract, self._speed, self._stop,
                                             output, reference_factory),
                                       name='sonic-planner', daemon=True)
        self.process.start()
        output.close()
        self.thread = threading.Thread(target=self._receive, name='planner-reference', daemon=True)
        self.thread.start()

    @property
    def speed(self):
        return self._speed.value

    @speed.setter
    def speed(self, value):
        if not np.isfinite(value) or not 0 <= value <= .7:
            raise ValueError('Unsupported forward speed')
        self._speed.value = float(value)

    def _receive(self):
        try:
            while not self.stop.is_set():
                if not self._input.poll(.1):
                    if not self.process.is_alive():
                        raise RuntimeError(f'Planner process exited: {self.process.exitcode}')
                    continue
                message = self._input.recv()
                if message[0] == 'error':
                    raise RuntimeError(message[1])
                _, origin, start, updated, seconds, window = message
                for array in (window.poses, window.joint_vel, window.lin_vel, window.ang_vel):
                    array.flags.writeable = False
                with self.reference.lock:
                    self.origin = origin
                    self.reference.start = start
                    self.reference.seconds = seconds
                    self.reference.window = window
                    self.reference.updated_at = updated
        except (Exception, EOFError) as exc:
            if not self.stop.is_set():
                self.error = exc

    def sample(self, now):
        if self.error:
            raise RuntimeError(f'Planner worker failed: {self.error}') from self.error
        if self.origin is None:
            raise RuntimeError('Planner is not initialized')
        return self.reference.sample(now-self.origin)

    def close(self):
        self.stop.set()
        self._stop.value = 1
        self.process.join(timeout=2.)
        if self.process.is_alive():
            # This child only computes references; it owns no motor connection.
            self.process.terminate()
            self.process.join(timeout=1.)
        self.thread.join(timeout=1.)
        self._input.close()
