import json
from pathlib import Path
import queue
import threading
import time

import numpy as np
import pytest

pytest.importorskip('rosbags')
from rosbags.highlevel import AnyReader
from vibe_grail_joint.deploy.observations import RobotState
from vibe_grail_joint.deploy.recording import BagRecorder, RecordingClient, RAW_SHAPE, COUNTERS
from vibe_grail_joint.deploy.runtime import Runtime
from vibe_grail_joint.deploy.visualize_bag import export, rgb, array

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT/'models/vggt_hierarchical_rgb64/bundle.json').read_text())


def wait_for(predicate):
    deadline = time.monotonic()+5
    while not predicate():
        if time.monotonic()>deadline:
            raise AssertionError('Recorder did not make progress')
        time.sleep(.01)


def test_queue_pressure_drops_record_without_waiting_or_control_side_effects():
    enabled = threading.Event(); enabled.set()
    transport = dict(queue=queue.Queue(1),enabled=enabled,counts=[0]*len(COUNTERS))
    client = RecordingClient(transport)
    assert client._put('step',{'id':1})
    assert not client._put('step',{'id':2})
    assert transport['counts'][4:6] == [1,1]
    assert enabled.is_set()
    assert transport['queue'].get_nowait()[1]['id']==1


def test_real_rosbag_roundtrip_and_exact_image_attention_join(tmp_path):
    path = tmp_path/'bag'
    recorder = BagRecorder(path,MANIFEST,raw_hz=10.)
    raw = np.full(RAW_SHAPE, (12,34,56), np.uint8)
    image = np.full((64,112,3), (6,8,10), np.uint8)
    patches = np.arange(28*192,dtype=np.float32).reshape(28,192)/100
    attention = np.zeros((3,28),np.float32); attention[:,9]=1.
    state = RobotState(np.zeros(29,np.float32),np.zeros(29,np.float32),
                       np.array([1.,0,0,0],np.float32),np.zeros(3,np.float32),time.monotonic())
    at = state.received_at
    try:
        recorder.event(dict(monotonic=at,event='camera_profile',intrinsics=dict(
            width=1280,height=720,fx=908.,fy=907.,ppx=650.,ppy=367.,coeffs=[0.]*5)))
        recorder.client.camera(raw,image,at,123,4.)
        recorder.client.features((patches,np.ones(192,np.float32)),at,123)
        common=dict(state=state,observations={'q_cls':np.ones((1,192),np.float32)},
            action=np.arange(29,dtype=np.float32),target=np.zeros(29,np.float32),
            attention=attention,speed=.4,heading=0.,yaw_rate=.45,reset=True)
        recorder.client.step(at=at+.01,step_id=0,image_at=at,frame_id=123,**common)
        # Deliberately absent source frame: the viewer must not use frame 123.
        recorder.client.step(at=at+.3,step_id=1,image_at=at+.29,frame_id=124,**common)
    finally:
        recorder.close()
    report=json.loads((path/'recording_status.json').read_text())
    assert report['complete'] and report['error'] is None
    assert not report['motor_transport']
    assert all(report['counters'][k]==0 for k in COUNTERS if k.endswith('_dropped'))
    decoded={}
    with AnyReader([path]) as reader:
        for conn,stamp,data in reader.messages():
            decoded.setdefault(conn.topic,[]).append((stamp,reader.deserialize(data,conn.msgtype)))
    np.testing.assert_array_equal(rgb(decoded['/vibe/camera/color/image_raw'][0][1]),raw[:,:,::-1])
    np.testing.assert_array_equal(rgb(decoded['/vibe/camera/policy/image'][0][1]),image[:,:,::-1])
    np.testing.assert_array_equal(array(decoded['/vibe/policy/attention'][0][1]),attention)
    np.testing.assert_array_equal(array(decoded['/vibe/vision/patches'][0][1]),patches)
    assert decoded['/vibe/camera/color/camera_info'][0][1].k[0]==908.
    assert decoded['/vibe/camera/policy/camera_info'][0][1].width==112
    assert decoded['/vibe/camera/policy/camera_info'][0][1].k[2]==55.5
    step=json.loads(decoded['/vibe/policy/step'][0][1].data)
    assert step['image_stamp_ns']==decoded['/vibe/camera/policy/image'][0][0]
    summary=export(path,tmp_path/'preview',fps=50)
    assert summary['frames']==2 and summary['missing_policy_images']==1
    preview=json.loads((tmp_path/'preview/preview.json').read_text())
    assert preview['frames'][0]['attention'][0][9]==1.
    assert preview['frames'][1]['policy'] is None
    assert '4×7' in (tmp_path/'preview/index.html').read_text()
    with pytest.raises(FileExistsError):
        BagRecorder(path,MANIFEST)


def test_size_limit_stops_only_recorder_and_closes_valid_bag(tmp_path):
    path=tmp_path/'limited'
    recorder=BagRecorder(path,MANIFEST,max_mib=.01)
    try:
        at=time.monotonic()
        recorder.client.camera(np.zeros(RAW_SHAPE,np.uint8),np.zeros((64,112,3),np.uint8),at,1,0.)
        wait_for(lambda:not recorder.status()['active'])
    finally:
        recorder.close()
    report=json.loads((path/'recording_status.json').read_text())
    assert report['complete'] and report['limit_reached'] and report['error'] is None
    with AnyReader([path]) as reader:
        assert reader.message_count>0


@pytest.mark.parametrize('bundle',['velocity3999','vggt_hierarchical_rgb64'])
def test_requesting_exported_attention_does_not_change_actions(bundle):
    runtime=Runtime(ROOT/'models'/bundle)
    try:
        feed={k:np.zeros(shape,np.float32) for k,shape in runtime.inputs.items()}
        patches,cls=runtime.encode(np.full(runtime.image_shape,127,np.uint8))
        feed['kv_tokens__img_tokens'],feed['q_cls']=patches[None],cls[None]
        original=runtime.act(feed)
        recorded=runtime.act(feed,with_attention=True)
        np.testing.assert_array_equal(original[0],recorded[0])
        np.testing.assert_array_equal(original[1],recorded[1])
        assert recorded[3].shape==(3,runtime.patch_shape[0])
        np.testing.assert_allclose(recorded[3].sum(-1),1.,atol=1e-5)
    finally:
        runtime.close()
