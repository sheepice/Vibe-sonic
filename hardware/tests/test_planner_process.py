import time

import numpy as np
import pytest

from vibe_grail_joint.deploy.planner_process import ProcessPlannerWorker
from vibe_grail_joint.deploy.reference import ReferenceWindow


class FakeReference:
    def __init__(self, path, contract):
        if path == 'broken':
            raise RuntimeError('model load failed')
    def replan(self, elapsed, speed):
        poses = np.zeros((100,36),np.float32)
        poses[:,3] = 1.
        poses[:,7] = speed
        self.window = ReferenceWindow(poses)
        self.start = int(elapsed*50)
        self.updated_at, self.seconds = elapsed, .001


def wait_for(predicate):
    deadline = time.monotonic()+5
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Planner did not make progress')
        time.sleep(.005)


CONTRACT = dict(root_height=.76,default_joint_pos=[0.]*29)


def test_process_transfers_reference_speed_and_original_timebase():
    worker = ProcessPlannerWorker('fake', CONTRACT, reference_factory=FakeReference)
    try:
        wait_for(lambda: worker.reference.updated_at is not None or worker.error is not None)
        assert worker.error is None
        np.testing.assert_array_equal(worker.sample(time.monotonic()).joint_pos[:,0],0.)
        worker.speed = .25
        wait_for(lambda: worker.sample(time.monotonic()).joint_pos[0,0] == .25)
        with pytest.raises(RuntimeError,match='stale'):
            worker.sample(worker.origin+worker.reference.updated_at+.451)
        with pytest.raises(ValueError):
            worker.speed = float('nan')
    finally:
        worker.close()
    assert not worker.process.is_alive() and not worker.thread.is_alive()


def test_process_failure_is_reported_without_fake_reference():
    worker = ProcessPlannerWorker('broken', CONTRACT, reference_factory=FakeReference)
    try:
        wait_for(lambda: worker.error is not None)
        assert worker.reference.updated_at is None
        with pytest.raises(RuntimeError,match='model load failed'):
            worker.sample(time.monotonic())
    finally:
        worker.close()


def test_unexpected_child_exit_is_reported():
    worker = ProcessPlannerWorker('fake', CONTRACT, reference_factory=FakeReference)
    try:
        wait_for(lambda: worker.reference.updated_at is not None)
        worker.process.terminate()
        worker.process.join(timeout=2)
        wait_for(lambda: worker.error is not None)
        with pytest.raises(RuntimeError,match='Planner worker failed'):
            worker.sample(time.monotonic())
    finally:
        worker.close()
