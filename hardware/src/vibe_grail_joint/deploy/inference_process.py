"""Isolate camera/models from the high-frequency SDK receiver and writer.

The child imports no robot SDK, owns no DDS publisher and receives only state
snapshots. The parent retains all stop latches, watchdogs and motor authority.
Original image/state timestamps cross the pipe unchanged.
"""
import multiprocessing as mp
from pathlib import Path
import time
import traceback
import os

import numpy as np


def startup_failure_message(camera, vision, planner):
    """Report missing readiness separately from explicit worker exceptions."""
    missing = []
    if camera.frame is None:
        missing.append('camera: no usable RGB frame')
    if vision.frame is None:
        missing.append('encoder: no encoded camera frame')
    if planner.reference.updated_at is None:
        missing.append('planner: no reference trajectory')
    errors = [f'{name}: {worker.error}' for name, worker in
              [('camera',camera),('encoder',vision),('planner',planner)] if worker.error is not None]
    return 'Inference startup failed; ' + '; '.join(missing + errors)


class LocalInferencePipeline:
    def __init__(self,config):
        from .camera import RealSenseCamera,validate_mount
        from .observations import ObservationBuilder
        from .reference import PlannerWorker
        from .runtime import Runtime
        from .vision import VisualWorker
        self.runtime = self.camera = self.planner = self.vision = None
        self.recording = None
        self.step_id = 0
        try:
            if config.get('recording'):
                from .recording import RecordingClient
                self.recording = RecordingClient(config['recording'])
            bundle = Path(config['bundle'])
            self.runtime = Runtime(bundle,config['provider'],engine_dir=config.get('engine_dir'))
            warmup = self.runtime.warmup()
            self.camera = RealSenseCamera(config.get('serial'),self.runtime.manifest['camera']['fovy'],
                height=self.runtime.image_shape[0], width=self.runtime.image_shape[1],
                on_frame=self.recording.camera if self.recording else None)
            if config.get('calibration'):
                validate_mount(config['calibration'],self.camera,self.runtime.manifest['camera'])
            self.planner = PlannerWorker(bundle/'planner.onnx',self.runtime.contract,
                                         threads=config.get('planner_threads',3))
            self.vision = VisualWorker(self.runtime,self.camera,
                on_features=self.recording.features if self.recording else None)
            self.builder = ObservationBuilder(self.runtime.contract, self.runtime.patch_shape)
            deadline = time.monotonic()+15
            while self.vision.frame is None or self.planner.reference.updated_at is None:
                if self.camera.error or self.vision.error or self.planner.error or time.monotonic()>deadline:
                    raise RuntimeError(startup_failure_message(self.camera,self.vision,self.planner))
                time.sleep(.005)
            self.metadata = dict(warmup_s=warmup,serial=self.camera.serial,
                image_shape=self.runtime.image_shape, patch_shape=self.runtime.patch_shape,
                intrinsics=self.camera.intrinsics,provider=config['provider'],
                timestamp_domain=self.camera.timestamp_domain,
                camera_mount_validation='measured_validated' if config.get('calibration') else 'unmeasured',
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                planner_threads=config.get('planner_threads',3),motor_transport=False)
            if self.recording:
                self.recording.event(dict(monotonic=time.monotonic(),event='camera_profile',
                                          **self.metadata))
        except BaseException:
            self.close()
            raise

    def step(self,request):
        begin = time.monotonic()
        state,speed = request['state'],request['speed']
        heading,yaw_rate = request.get('command_heading',0.),request.get('yaw_rate',0.)
        if not np.isfinite([speed,heading,yaw_rate]).all() or not -.3 <= speed <= .7 or abs(yaw_rate)>.5:
            raise ValueError('Invalid inference speed')
        self.planner.speed = float(speed)
        self.planner.command = (float(speed),float(heading))
        features,image_at,frame_id = self.vision.latest()
        ref = self.planner.sample(time.monotonic())
        result = dict(image_at=image_at,frame_id=frame_id,
                      source_state_at=state.received_at,planner_s=self.planner.reference.seconds)
        observations = action = target = attention = None
        if request.get('compute',True):
            if request.get('reset'):
                self.builder.reset()
                self.builder.align_heading(request.get('heading',state.quat))
            elif request.get('previous_action') is not None:
                self.builder.accept_action(request['previous_action'])
            observations = self.builder.build(state,ref,*features,speed,velocity=self.runtime.velocity,
                                              heading=heading,yaw_rate=yaw_rate)
            if self.recording:
                action,target,seconds,attention = self.runtime.act(observations,with_attention=True)
            else:
                action,target,seconds = self.runtime.act(observations)
            result.update(action=action,target=target,policy_s=seconds)
        if self.recording:
            self.recording.step(at=begin, step_id=self.step_id, state=state,
                image_at=image_at, frame_id=frame_id, observations=observations,
                action=action, target=target, attention=attention, speed=float(speed),
                heading=float(heading), yaw_rate=float(yaw_rate), reset=bool(request.get('reset')))
        self.step_id += 1
        result['worker_s'] = time.monotonic()-begin
        return result

    def close(self):
        if self.vision:
            self.vision.close()
        if self.planner:
            self.planner.close()
        if self.camera:
            self.camera.close()
        if self.runtime and (self.vision is None or not self.vision.thread.is_alive()):
            self.runtime.close()
        if self.recording:
            self.recording.close()


def _inference_main(connection,config,pipeline_factory):
    pipeline = None
    try:
        if config.get('cpus'):
            os.sched_setaffinity(0,{int(x) for x in config['cpus'].split(',')})
        pipeline = pipeline_factory(config)
        connection.send(('ready',0,pipeline.metadata))
        while True:
            kind,sequence,request = connection.recv()
            if kind == 'close':
                break
            if kind != 'step':
                raise ValueError('Unknown inference request')
            connection.send(('result',sequence,pipeline.step(request)))
    except (EOFError,BrokenPipeError):
        pass
    except BaseException:
        try:
            connection.send(('error',0,traceback.format_exc()))
        except (EOFError,OSError):
            pass
    finally:
        if pipeline:
            pipeline.close()
        connection.close()


class InferenceProcess:
    def __init__(self,config,*,pipeline_factory=LocalInferencePipeline,startup_timeout=30.):
        context = mp.get_context('spawn')
        self.connection,child = context.Pipe()
        self.sequence = 0
        self.failed = None
        self.closed = False
        self.process = context.Process(target=_inference_main,
            args=(child,dict(config),pipeline_factory),name='g1-inference',daemon=True)
        self.process.start()
        child.close()
        try:
            self.metadata = self._receive('ready',0,startup_timeout)
        except BaseException:
            self.close()
            raise

    def _receive(self,expected,sequence,timeout):
        deadline = time.monotonic()+timeout
        while not self.connection.poll(min(.005,max(0.,deadline-time.monotonic()))):
            if not self.process.is_alive():
                raise RuntimeError('Inference process exited')
            if time.monotonic()>=deadline:
                raise RuntimeError('Inference process deadline exceeded')
        if time.monotonic()>deadline:
            raise RuntimeError('Inference process deadline exceeded')
        try:
            kind,received,payload = self.connection.recv()
        except (OSError,EOFError) as exc:
            raise RuntimeError('Inference process connection closed') from exc
        if time.monotonic()>deadline:
            raise RuntimeError('Inference process deadline exceeded')
        if kind == 'error':
            raise RuntimeError(f'Inference process failed: {payload}')
        if kind != expected or received != sequence:
            raise RuntimeError('Unexpected or stale inference response')
        return payload

    def step(self,state,speed,*,previous_action=None,reset=False,heading=None,compute=True,timeout=.06,
             command_heading=0.,yaw_rate=0.):
        # A missed 40 ms scheduling target is diagnostic. Never wait beyond
        # the independent writer's 60 ms fresh-target watchdog.
        if not np.isfinite(timeout) or not 0<timeout<=.06:
            raise ValueError('Inference deadline must be positive and at most 60 ms')
        if self.closed or self.failed:
            raise RuntimeError(f'Inference unavailable: {self.failed or "closed"}')
        self.sequence += 1
        request = dict(state=state,speed=float(speed),previous_action=previous_action,
                       reset=reset,compute=compute,command_heading=command_heading,yaw_rate=yaw_rate)
        if heading is not None:
            request['heading'] = heading
        try:
            self.connection.send(('step',self.sequence,request))
            return self._receive('result',self.sequence,timeout)
        except Exception as exc:
            self.failed = str(exc)
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.is_alive():
                self.connection.send(('close',0,None))
        except (OSError,EOFError):
            pass
        self.process.join(timeout=2.)
        if self.process.is_alive():
            # The child only owns models/camera; stopping it cannot publish motors.
            self.process.terminate()
            self.process.join(timeout=1.)
        self.connection.close()
