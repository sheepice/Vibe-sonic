"""Optional, bounded ROS 2 recording in a separate, file-only process.

The control/inference processes never serialize ROS messages or write images
to disk. Large raw images cross shared memory; the small-message queue is
bounded and nonblocking. Recording loss is reported, never a motor command.
"""
import collections
import json
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import os
from pathlib import Path
import queue
import shutil
import time
import traceback

import numpy as np

RAW_SHAPE = (720, 1280, 3)
RAW_BYTES = int(np.prod(RAW_SHAPE))
COUNTERS = ('camera_queued', 'camera_dropped', 'features_queued', 'features_dropped',
            'step_queued', 'step_dropped', 'event_queued', 'event_dropped',
            'raw_queued', 'raw_dropped')


class RecordingClient:
    def __init__(self, transport):
        self.transport = transport
        self.queue = transport['queue']
        self.enabled = transport['enabled']
        self.counts = transport['counts']
        self.shm = None
        self.raw = None
        self.next_raw = 0.

    def _put(self, kind, payload):
        if not self.enabled.is_set():
            return False
        index = {'camera': 0, 'features': 2, 'step': 4, 'event': 6}[kind]
        try:
            self.queue.put_nowait((kind, payload))
        except (queue.Full, ValueError, OSError):
            self.counts[index+1] += 1
            return False
        self.counts[index] += 1
        return True

    def camera(self, raw, policy, captured, frame_id, device_time):
        if not self.enabled.is_set():
            return
        slot = None
        if captured >= self.next_raw:
            self.next_raw = captured + 1./self.transport['raw_hz'] - .002
            try:
                slot = self.transport['free'].get_nowait()
            except queue.Empty:
                self.counts[9] += 1
            if slot is not None:
                if self.shm is None:
                    self.shm = SharedMemory(name=self.transport['shm'])
                    self.raw = np.ndarray((self.transport['slots'], *RAW_SHAPE), np.uint8,
                                          buffer=self.shm.buf)
                if raw.shape != RAW_SHAPE or raw.dtype != np.uint8:
                    self.transport['free'].put_nowait(slot)
                    self.counts[9] += 1
                    slot = None
                else:
                    np.copyto(self.raw[slot], raw)
        sent = self._put('camera', dict(policy=policy, captured=captured, frame_id=frame_id,
                                       device_time=device_time, raw_slot=slot))
        if slot is not None:
            if sent:
                self.counts[8] += 1
            else:
                self.transport['free'].put_nowait(slot)
                self.counts[9] += 1

    def features(self, features, captured, frame_id):
        self._put('features', dict(features=features, captured=captured, frame_id=frame_id))

    def step(self, **data):
        self._put('step', data)

    def event(self, data):
        self._put('event', data)

    def close(self):
        self.raw = None
        if self.shm is not None:
            self.shm.close()
            self.shm = None


class BagSink:
    """ROS serialization is confined to the recorder process."""
    def __init__(self, path, metadata):
        from rosbags.rosbag2 import Writer
        from rosbags.typesys import Stores, get_typestore
        self.store = get_typestore(Stores.ROS2_HUMBLE)
        self.types = self.store.types
        self.writer = Writer(path, version=9)
        self.writer.open()
        self.connections = {}
        self.offset = metadata['wall_minus_monotonic_ns']
        self.names = metadata['manifest']['robot']['joint_names']
        self.camera = metadata['manifest']['camera']
        self.bytes_written = 0
        self.messages = collections.Counter()
        self.string('/vibe/recording/metadata', time.monotonic(), metadata)

    def stamp(self, monotonic):
        return self.offset + round(monotonic*1e9)

    def header(self, monotonic, frame):
        stamp = self.stamp(monotonic)
        t = self.types['builtin_interfaces/msg/Time'](stamp//10**9, stamp%10**9)
        return self.types['std_msgs/msg/Header'](t, frame)

    def write(self, topic, at, message):
        name = message.__msgtype__
        if topic not in self.connections:
            self.connections[topic] = self.writer.add_connection(topic, name, typestore=self.store)
        encoded = self.store.serialize_cdr(message, name)
        self.writer.write(self.connections[topic], self.stamp(at), encoded)
        self.bytes_written += len(encoded)
        self.messages[topic] += 1

    def string(self, topic, at, value):
        self.write(topic, at, self.types['std_msgs/msg/String'](
            json.dumps(value, allow_nan=False, separators=(',', ':'))))

    def image(self, topic, at, bgr, frame):
        h, w = bgr.shape[:2]
        self.write(topic, at, self.types['sensor_msgs/msg/Image'](
            self.header(at, frame), h, w, 'bgr8', 0, w*3, bgr.reshape(-1)))

    def array(self, topic, at, values, labels=None):
        values = np.asarray(values, np.float32)
        dim = self.types['std_msgs/msg/MultiArrayDimension']
        dims = [dim((labels or [f'axis_{j}' for j in range(values.ndim)])[i], size,
                    int(np.prod(values.shape[i:]))) for i, size in enumerate(values.shape)]
        layout = self.types['std_msgs/msg/MultiArrayLayout'](dims, 0)
        self.write(topic, at, self.types['std_msgs/msg/Float32MultiArray'](layout, values.ravel()))

    def camera_info(self, at, intrinsics):
        camera = self.camera
        focal = camera['height']/(2*np.tan(np.deg2rad(camera['fovy']/2)))
        policy = dict(width=camera['width'],height=camera['height'],fx=focal,fy=focal,
                      ppx=(camera['width']-1)/2,ppy=(camera['height']-1)/2,coeffs=[0.]*5)
        for name, data, frame in [('color',intrinsics,'g1_d435_color_optical'),
                                  ('policy',policy,'g1_policy_camera_optical')]:
            # This D435i reports zero color distortion. Do not relabel a
            # nonzero RealSense inverse-distortion model as ROS plumb_bob.
            if any(data['coeffs']):
                continue
            k=np.array([[data['fx'],0,data['ppx']],[0,data['fy'],data['ppy']],[0,0,1]],np.float64)
            roi=self.types['sensor_msgs/msg/RegionOfInterest'](0,0,0,0,False)
            self.write('/vibe/camera/'+name+'/camera_info',at,self.types['sensor_msgs/msg/CameraInfo'](
                self.header(at,frame),data['height'],data['width'],'plumb_bob',
                np.asarray(data['coeffs'],np.float64),k.ravel(),np.eye(3).ravel(),
                np.c_[k,np.zeros(3)].ravel(),0,0,roi))

    def consume(self, kind, p, raw):
        if kind == 'camera':
            at = p['captured']
            self.image('/vibe/camera/policy/image', at, p['policy'], 'g1_policy_camera_optical')
            if p['raw_slot'] is not None:
                self.image('/vibe/camera/color/image_raw', at, raw[p['raw_slot']], 'g1_d435_color_optical')
            self.string('/vibe/camera/frame', at, dict(frame_id=p['frame_id'],
                image_at=at, device_time_s=p['device_time'], raw_recorded=p['raw_slot'] is not None))
        elif kind == 'features':
            at = p['captured']
            self.array('/vibe/vision/patches', at, p['features'][0], ['patch', 'channel'])
            self.array('/vibe/vision/cls', at, p['features'][1])
            self.string('/vibe/vision/frame', at, dict(frame_id=p['frame_id'], image_at=at))
        elif kind == 'step':
            at, state = p['at'], p['state']
            header = self.header(state.received_at, 'pelvis')
            self.write('/vibe/robot/joint_states', at, self.types['sensor_msgs/msg/JointState'](
                header, self.names, state.q.astype(np.float64), state.dq.astype(np.float64),
                np.empty(0, np.float64)))
            q, g = state.quat, state.gyro
            quat = self.types['geometry_msgs/msg/Quaternion'](*[float(x) for x in q[[1,2,3,0]]])
            vector = self.types['geometry_msgs/msg/Vector3']
            covariance = np.zeros(9, np.float64)
            unavailable = covariance.copy(); unavailable[0] = -1.
            self.write('/vibe/robot/imu', at, self.types['sensor_msgs/msg/Imu'](header, quat,
                covariance, vector(*[float(x) for x in g]), covariance,
                vector(0.,0.,0.), unavailable))
            self.string('/vibe/policy/step', at, dict(step_id=p['step_id'], image_frame_id=p['frame_id'],
                image_at=p['image_at'], image_stamp_ns=self.stamp(p['image_at']),
                state_at=state.received_at, inference_at=at, compute=p['observations'] is not None,
                speed=p['speed'], heading=p['heading'], yaw_rate=p['yaw_rate'],
                reset=p['reset'], remote_buttons=state.remote_buttons,
                motor_errors=list(state.motor_errors), temperature=float(state.temperature)))
            if p['observations'] is not None:
                for name, value in p['observations'].items():
                    self.array('/vibe/policy/inputs/'+name, at, value)
                self.array('/vibe/policy/actions', at, p['action'])
                self.array('/vibe/policy/raw_target', at, p['target'])
                if p['attention'] is not None:
                    self.array('/vibe/policy/attention', at, p['attention'], ['query', 'patch'])
                else:
                    self.string('/vibe/recording/warnings',at,dict(step_id=p['step_id'],reason='invalid_attention'))
        elif kind == 'event':
            self.string('/vibe/control/events', p['monotonic'], p)
            if p.get('event')=='camera_profile':
                self.camera_info(p['monotonic'],p['intrinsics'])

    def close(self):
        self.writer.close()

    def checkpoint(self):
        # Pinned rosbags 0.11.9 uses one SQLite transaction until close by
        # default. Commit periodically so an interrupted trial retains data.
        self.writer.storage.conn.commit()


def _record_main(transport, path, metadata, ready):
    sink = shm = None
    error = None
    limit_reached = False
    try:
        from .observations import RobotState  # Warm the packet type before accepting live records.
        if transport['cpus']:
            os.sched_setaffinity(0, {int(x) for x in transport['cpus'].split(',')})
        os.nice(10)
        shm = SharedMemory(name=transport['shm'])
        raw = np.ndarray((transport['slots'], *RAW_SHAPE), np.uint8, buffer=shm.buf)
        sink = BagSink(path, metadata)
        transport['enabled'].set()
        ready.send(None)
        last_status = time.monotonic()
        while True:
            try:
                kind, payload = transport['queue'].get(timeout=.1)
            except queue.Empty:
                if transport['stop'].is_set():
                    break
                continue
            slot = payload.get('raw_slot') if kind == 'camera' else None
            try:
                sink.consume(kind, payload, raw)
            finally:
                if slot is not None:
                    transport['free'].put_nowait(slot)
            if sink.bytes_written >= transport['max_bytes']:
                limit_reached = True
                transport['enabled'].clear()
                break
            if time.monotonic()-last_status > 1.:
                last_status = time.monotonic()
                sink.checkpoint()
                if shutil.disk_usage(Path(path).parent).free < 512*1024**2:
                    raise OSError('Less than 512 MiB disk space remains; recording stopped')
                _status(path, transport, sink, complete=False)
    except BaseException:
        error = traceback.format_exc()
        try:
            ready.send(error)
        except (BrokenPipeError, OSError):
            pass
        print('BAG RECORDING STOPPED: '+error, flush=True)
    finally:
        transport['enabled'].clear()
        if sink:
            try:
                sink.string('/vibe/recording/status', time.monotonic(),
                    dict(error=error, limit_reached=limit_reached, counters=dict(zip(COUNTERS,transport['counts']))))
                sink.close()
            except BaseException:
                error = (error or '') + traceback.format_exc()
        try:
            _status(path, transport, sink, complete=True, error=error, limit_reached=limit_reached)
        except OSError as exc:
            print(f'BAG final status could not be saved: {exc}', flush=True)
        if shm:
            shm.close()
        ready.close()


def _status(path, transport, sink, *, complete, error=None, limit_reached=False):
    topics={'camera':'/vibe/camera/policy/image','features':'/vibe/vision/patches',
            'step':'/vibe/policy/step','event':'/vibe/control/events','raw':'/vibe/camera/color/image_raw'}
    counts=dict(zip(COUNTERS,transport['counts']))
    report = dict(complete=complete, error=error, limit_reached=limit_reached,
        counters=counts,
        queued_not_written={kind:max(0,counts[kind+'_queued']-(sink.messages[topic] if sink else 0))
                            for kind,topic in topics.items()},
        messages=dict(sink.messages) if sink else {},
        serialized_bytes=sink.bytes_written if sink else 0, motor_transport=False)
    target = Path(path)/'recording_status.json'
    if target.parent.exists():
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2)+'\n')
        temp.replace(target)


class BagRecorder:
    def __init__(self, path, manifest, *, raw_hz=10., max_mib=4096, cpus=None, slots=4, queue_size=256):
        import rosbags.rosbag2  # Fail before engaging any controller if the optional dependency is absent.
        self.path = Path(path).resolve()
        if self.path.exists():
            raise FileExistsError(f'Use a new bag directory: {self.path}')
        if not 0 < raw_hz <= 30 or max_mib <= 0:
            raise ValueError('Invalid recording rate or size limit')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.path.parent).free < 1024**3:
            raise OSError('At least 1 GiB free space is required for recording')
        context = mp.get_context('spawn')
        self.shm = SharedMemory(create=True, size=slots*RAW_BYTES)
        self.transport = dict(queue=context.Queue(queue_size), free=context.Queue(slots),
            counts=context.Array('Q', len(COUNTERS), lock=False), shm=self.shm.name, slots=slots,
            enabled=context.Event(), stop=context.Event(), raw_hz=raw_hz,
            max_bytes=round(max_mib*1024**2), cpus=cpus)
        for i in range(slots):
            self.transport['free'].put_nowait(i)
        metadata = dict(schema='vibe.diagnostic_bag.v1', manifest=manifest, raw_hz=raw_hz,
            raw_shape=RAW_SHAPE, wall_minus_monotonic_ns=time.time_ns()-time.monotonic_ns(),
            attention_queries=['q_task_cmd','q_proprio','q_cls'],
            attention_meaning='Cross attention over spatially mixed patch slots; not causal pixel saliency.',
            actual_input_images=True, depth_input=False, motor_transport=False)
        parent, child = context.Pipe(duplex=False)
        self.process = context.Process(target=_record_main,
            args=(self.transport, str(self.path), metadata, child), name='g1-bag-file-writer', daemon=True)
        self.client = RecordingClient(self.transport)
        self.closed = False
        self.process.start()
        child.close()
        try:
            if not parent.poll(15.):
                raise RuntimeError('Bag writer did not initialize')
            error = parent.recv()
            if error:
                raise RuntimeError('Bag initialization failed: '+error)
        except BaseException:
            self.close()
            raise
        finally:
            parent.close()

    def event(self, data):
        self.client.event(data)

    def status(self):
        return dict(path=str(self.path), active=self.process.is_alive() and self.transport['enabled'].is_set(),
                    counters=dict(zip(COUNTERS,self.transport['counts'])))

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.transport['enabled'].clear()
        self.transport['stop'].set()
        self.process.join(10.)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2.)
            print('BAG writer timed out; inspect recording_status.json before using this bag',flush=True)
        self.client.close()
        for key in ('queue','free'):
            self.transport[key].cancel_join_thread()
            self.transport[key].close()
        self.shm.close()
        self.shm.unlink()
