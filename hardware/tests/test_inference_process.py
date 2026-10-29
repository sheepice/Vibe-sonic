"""Exercise process boundaries and failure/timeout handling without robot SDKs."""
import multiprocessing as mp
import time
from types import SimpleNamespace

import numpy as np
import pytest

from vibe_grail_joint.deploy.inference_process import InferenceProcess, startup_failure_message


def test_startup_timeout_identifies_camera_and_planner_independently():
    camera=SimpleNamespace(frame=None,error=None)
    vision=SimpleNamespace(frame=None,error=None)
    planner=SimpleNamespace(reference=SimpleNamespace(updated_at=1.),error=None)
    text=startup_failure_message(camera,vision,planner)
    assert 'camera: no usable RGB frame' in text
    assert 'encoder: no encoded camera frame' in text
    assert 'planner:' not in text
    camera.frame=vision.frame=object()
    planner.reference.updated_at=None
    text=startup_failure_message(camera,vision,planner)
    assert 'planner: no reference trajectory' in text and 'camera:' not in text
    camera.error=RuntimeError('USB disconnected')
    assert 'USB disconnected' in startup_failure_message(camera,vision,planner)


class FakePipeline:
    def __init__(self,config):
        self.case = config.get('case')
        if self.case == 'startup_failure':
            raise RuntimeError('model could not load')
        self.metadata = dict(motor_transport=False,provider='fake')

    def step(self,request):
        if self.case == 'slow':
            time.sleep(.15)
        if self.case == 'one_late_tick':
            time.sleep(.043)
        if self.case == 'failure':
            raise RuntimeError('encoder stopped')
        return dict(target=np.full(29,request['speed'],np.float32),image_at=12.5,
                    source_state_at=request['state'].received_at,
                    previous_action=request['previous_action'],reset=request['reset'],
                    compute=request['compute'])

    def close(self):
        pass


def test_state_action_and_source_timestamps_survive_process_boundary():
    worker = InferenceProcess({},pipeline_factory=FakePipeline,startup_timeout=5.)
    try:
        assert worker.metadata['motor_transport'] is False
        previous = np.linspace(-1.,1.,29,dtype=np.float32)
        result = worker.step(SimpleNamespace(received_at=12.6),.25,previous_action=previous)
        np.testing.assert_array_equal(result['target'],np.full(29,.25,np.float32))
        np.testing.assert_array_equal(result['previous_action'],previous)
        assert result['image_at'] == 12.5 and result['source_state_at'] == 12.6
        status = worker.step(SimpleNamespace(received_at=12.7),0.,reset=True,compute=False)
        assert status['reset'] and not status['compute']
    finally:
        worker.close()
    assert not worker.process.is_alive()
    worker.close()  # idempotent shutdown


def test_late_result_cannot_be_reused_or_retried():
    worker = InferenceProcess({'case':'slow'},pipeline_factory=FakePipeline,startup_timeout=5.)
    try:
        with pytest.raises(RuntimeError,match='deadline'):
            worker.step(SimpleNamespace(received_at=0.),0.,timeout=.01)
        with pytest.raises(RuntimeError,match='unavailable'):
            worker.step(SimpleNamespace(received_at=0.),0.)
    finally:
        worker.close()
    assert not worker.process.is_alive()


def test_worker_failure_propagates_and_latches_client_unavailable():
    worker = InferenceProcess({'case':'failure'},pipeline_factory=FakePipeline,startup_timeout=5.)
    try:
        with pytest.raises(RuntimeError,match='encoder stopped'):
            worker.step(SimpleNamespace(received_at=0.),0.)
        assert worker.failed is not None
    finally:
        worker.close()


def test_startup_failure_reaps_child():
    with pytest.raises(RuntimeError,match='model could not load'):
        InferenceProcess({'case':'startup_failure'},pipeline_factory=FakePipeline,startup_timeout=5.)
    assert not any(p.name == 'g1-inference' and p.is_alive() for p in mp.active_children())


def test_inference_timeout_cannot_exceed_control_watchdog():
    worker = InferenceProcess({},pipeline_factory=FakePipeline,startup_timeout=5.)
    try:
        for invalid in (0.,-.1,.061,float('nan')):
            with pytest.raises(ValueError,match='60 ms'):
                worker.step(SimpleNamespace(received_at=0.),0.,timeout=invalid)
    finally:
        worker.close()


def test_occasional_40ms_overrun_does_not_latch_inference():
    worker = InferenceProcess({'case':'one_late_tick'},pipeline_factory=FakePipeline,startup_timeout=5.)
    try:
        result = worker.step(SimpleNamespace(received_at=12.6),.4)
        assert result['source_state_at'] == 12.6 and worker.failed is None
    finally:
        worker.close()
