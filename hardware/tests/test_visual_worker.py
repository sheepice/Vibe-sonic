import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vibe_grail_joint.deploy.vision import VisualWorker


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Worker did not make progress')
        time.sleep(.001)


def test_delayed_encoder_preserves_capture_time_and_uses_latest_frame():
    started, finish = threading.Event(), threading.Event()
    second_started, second_finish = threading.Event(), threading.Event()
    values = (np.zeros((21, 192), np.float32), np.zeros(192, np.float32))
    camera = SimpleNamespace(frame=(np.zeros((63,112,3), np.uint8), 10., 1), error=None)
    camera.latest = lambda: camera.frame
    calls = []
    def encode(image):
        calls.append(int(image[0,0,0]))
        if len(calls) == 1:
            started.set()
            assert finish.wait(2)
        elif len(calls) == 2:
            second_started.set()
            assert second_finish.wait(2)
        return tuple(v.copy() for v in values)
    worker = VisualWorker(SimpleNamespace(encode=encode), camera)
    try:
        assert started.wait(2)
        with pytest.raises(RuntimeError, match='first encoded'):
            worker.latest()
        # No timestamp refresh while encoding. A later image cannot be attached
        # to features computed from the older input.
        camera.frame = (np.full((63,112,3), 9, np.uint8), 20., 9)
        finish.set()
        assert second_started.wait(2)
        features, stamp, number = worker.latest()
        assert stamp == 10. and number == 1
        assert all(not v.flags.writeable for v in features)
        second_finish.set()
        wait_for(lambda: worker.frame[2] == 9)
        assert worker.latest()[1:] == (20., 9)
        time.sleep(.01)
        assert calls == [0, 9]
        camera.error = RuntimeError('camera disconnected')
        wait_for(lambda: worker.error is not None)
        with pytest.raises(RuntimeError, match='camera disconnected'):
            worker.latest()
    finally:
        finish.set()
        second_finish.set()
        worker.close()
    assert not worker.thread.is_alive()


def test_encoding_failure_does_not_silently_reuse_previous_features():
    camera = SimpleNamespace(frame=(np.zeros((63,112,3), np.uint8), 10., 1), error=None)
    camera.latest = lambda: camera.frame
    def fail(_):
        raise RuntimeError('encoder failed')
    worker = VisualWorker(SimpleNamespace(encode=fail), camera)
    try:
        wait_for(lambda: worker.error is not None)
        with pytest.raises(RuntimeError, match='encoder failed'):
            worker.latest()
    finally:
        worker.close()
