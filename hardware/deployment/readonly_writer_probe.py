"""Profile the production writer with actual sensors but NO motor transport.

The SDK publisher is forbidden. The optional DDS sink uses a private topic,
domain 197 and loopback only. Never calls arm or MotionSwitcher. Outputs are
discarded; the local writer object is not the sensor backend.
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

from vibe_grail_joint.deploy.unitree import UnitreeBackend
from vibe_grail_joint.deploy.inference_process import InferenceProcess

DOMAIN = 197
TOPIC = 'vibe_diagnostic_discard_only'
CONFIG = '''<CycloneDDS><Domain Id="197"><General><Interfaces>
<NetworkInterface name="lo"/></Interfaces><AllowMulticast>false</AllowMulticast>
</General><Discovery><ParticipantIndex>auto</ParticipantIndex><Peers>
<Peer Address="127.0.0.1"/></Peers></Discovery></Domain></CycloneDDS>'''


def local_dds():
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.topic import Topic
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    domain = Domain(DOMAIN, CONFIG)
    participant = DomainParticipant(DOMAIN)
    return domain, participant, Topic(participant, TOPIC, LowCmd_)


def sink(ready, stop, result):
    try:
        os.sched_setaffinity(0, {6, 7})
        from cyclonedds.sub import DataReader
        domain, participant, topic = local_dds()
        reader = DataReader(participant, topic)
        ready.set()
        count = 0
        while not stop.is_set():
            count += len(reader.take(100))
            stop.wait(.002)
        result.put(dict(received=count))
    except BaseException as exc:
        result.put(dict(error=repr(exc)))
        ready.set()


class DiscardPublisher:
    def __init__(self, dds):
        self.count = 0
        self.writer = None
        if dds:
            from cyclonedds.pub import DataWriter
            self.domain, self.participant, self.topic = local_dds()
            self.writer = DataWriter(self.participant, self.topic)
            time.sleep(1.)

    def Write(self, message):
        if self.writer is None:
            message.serialize(use_version_2=False)
        else:
            self.writer.write(message)
        self.count += 1
        return True


class LocalWriter(UnitreeBackend):
    """Run identical packet/watchdog code against an inert sink, without arm()."""
    def __init__(self, source, publisher):
        if source.execute or source.armed or source.publisher is not None or not isinstance(publisher, DiscardPublisher):
            raise RuntimeError('Probe requires a read-only source and a discard-only publisher')
        self.source = source
        self.contract, self.crc, self.lock = source.contract, source.crc, source.lock
        self.limits = source.limits
        self.command_message = source.make_lowcmd()
        self.publisher = publisher
        self.armed_machine = source.latest().mode_machine
        self.target = None
        self.armed = True  # local scheduler only; source.armed remains False
        self.execute = False
        self.fault = None
        self.stop = threading.Event()
        self.last_write_timing = None
        self.writer_stats = dict(source.writer_stats)
        self.first_deadline_overrun = None
        self.thread = None

    @property
    def state(self):
        return self.source.state

    def arm(self):
        raise RuntimeError('Forbidden in diagnostic writer')

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=float, default=30.)
    p.add_argument('--loopback-dds', action='store_true')
    p.add_argument('--policy', action='store_true')
    p.add_argument('--hold-seconds', type=float, default=0.)
    p.add_argument('--inference-cpus', default='2,3,4,5,6,7')
    p.add_argument('--bundle', type=Path, default=Path('models/velocity3999'))
    p.add_argument('--engine-dir', type=Path, default=Path('engines/velocity3999_trt10_fp32'))
    p.add_argument('--interface', default='enP8p1s0')
    p.add_argument('--serial', default='254522074774')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--record-bag', type=Path)
    p.add_argument('--record-raw-hz', type=float, default=10.)
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or not 0 <= args.hold_seconds < args.seconds:
        p.error('Invalid test duration')
    if args.output.exists():
        raise FileExistsError(args.output)
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            arguments = (process/'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if b'vibe_grail_joint.deploy.run' in arguments:
            raise RuntimeError('Refusing diagnostic while user deployment is running')
    from unitree_sdk2py.core import channel
    attempts = []
    def forbidden(*a, **kw):
        attempts.append(repr(a))
        raise RuntimeError('ALL SDK publishers forbidden by diagnostic')
    channel.ChannelPublisher = forbidden
    sys.setswitchinterval(.001)
    manifest = json.loads((args.bundle/'bundle.json').read_text())
    contract = manifest['robot']
    backend = worker = writer = sink_process = recorder = None
    rows, fault, failure = [], None, None
    ctx = mp.get_context('spawn')
    ready, stop, result_queue = ctx.Event(), ctx.Event(), ctx.Queue()
    sink_result = None
    try:
        if args.record_bag:
            from vibe_grail_joint.deploy.recording import BagRecorder
            recorder = BagRecorder(args.record_bag,manifest,raw_hz=args.record_raw_hz,cpus='0,1')
        backend = UnitreeBackend(args.interface, contract, execute=False)
        until = time.monotonic()+10.
        while backend.state is None:
            if time.monotonic()>until:
                raise RuntimeError('No CRC-valid robot state')
            time.sleep(.02)
        if args.loopback_dds:
            sink_process = ctx.Process(target=sink, args=(ready, stop, result_queue))
            sink_process.start()
            if not ready.wait(10.):
                raise RuntimeError('Local sink did not initialize')
        publisher = DiscardPublisher(args.loopback_dds)
        worker = InferenceProcess(dict(bundle=str(args.bundle.resolve()),
            provider='tensorrt', engine_dir=str(args.engine_dir.resolve()),
            serial=args.serial, planner_threads=3, cpus=args.inference_cpus,
            recording=recorder.transport if recorder else None))
        writer = LocalWriter(backend, publisher)
        origin = time.monotonic()
        previous = None
        policy_active = False
        heading = 0.
        commands = [(0.,0.),(.15,0.),(.25,0.),(.4,0.),(-.1,0.),(0.,.45),(0.,-.45),(0.,0.)]
        for tick in range(round(args.seconds*50)):
            time.sleep(max(0., origin+tick*.02-time.monotonic()))
            begin = time.monotonic()
            state = backend.latest()
            compute = args.policy and tick >= round(args.hold_seconds*50)
            phase = max(0., tick*.02-args.hold_seconds)/(args.seconds-args.hold_seconds)
            speed, yaw = commands[min(len(commands)-1, int(phase*len(commands)))] if compute else (0.,0.)
            heading += yaw*.02
            out = worker.step(state, speed, compute=compute, previous_action=previous,
                reset=compute and not policy_active, command_heading=heading, yaw_rate=yaw)
            policy_active = compute
            target = out['target'] if compute else np.asarray(contract['default_joint_pos'],np.float32)
            if compute:
                previous = out['action']
            writer.submit(target, out['image_at'], begin)
            if recorder:
                recorder.event(dict(monotonic=time.monotonic(), event='control_tick' if compute else 'nominal_pose_hold',
                    speed_command=speed, yaw_rate_command=yaw, heading_command=heading,
                    action=out['action'].tolist() if compute else None,
                    q_target=target.tolist(), q=state.q.tolist(), quat=state.quat.tolist(),
                    frame_id=out['frame_id'], image_age_s=time.monotonic()-out['image_at'],
                    lowcmd_enabled=False, diagnostic_discard_only=True))
            if writer.thread is None:
                writer.thread = threading.Thread(target=writer._writer, name='inert-writer-probe', daemon=True)
                writer.thread.start()
            assert not backend.armed and not backend.execute and backend.publisher is None
            if writer.fault and fault is None:
                fault = dict(tick=tick, elapsed_s=time.monotonic()-origin,
                             reason=writer.fault, writer=writer.writer_diagnostics())
            rows.append(dict(tick=tick, loop_s=time.monotonic()-begin,
                             state_age_s=time.monotonic()-state.received_at,
                             policy_compute=compute, speed=speed, yaw_rate=yaw,
                             writer=writer.last_write_timing))
    except BaseException as exc:
        failure = repr(exc)
    finally:
        if writer:
            writer.close()
        if worker:
            worker.close()
        if backend:
            backend.close()
        if recorder:
            recorder.close()
        stop.set()
        if sink_process:
            sink_process.join(timeout=5.)
            if sink_process.is_alive():
                sink_process.terminate()
                sink_process.join()
            if not result_queue.empty():
                sink_result = result_queue.get()
    def stats(values):
        return dict(p50_ms=float(np.percentile(values,50))*1000,
                    p99_ms=float(np.percentile(values,99))*1000,
                    max_ms=float(max(values))*1000) if values else None
    fields = ('state_lock_s','health_s','packet_lock_s','pack_crc_s','dds_write_s','total_s')
    passed = (failure is None and fault is None and len(rows) == round(args.seconds*50)
              and max((r['loop_s'] for r in rows), default=float('inf')) <= .04
              and not attempts and backend is not None and backend.crc_failures == 0
              and (not args.loopback_dds or (sink_result is not None and sink_result.get('received',0)>0)))
    report = dict(software_health_pass=passed, failure=failure, first_fault=fault, samples=len(rows),
        bundle=str(args.bundle), bundle_files=manifest['files'], camera=manifest['camera'],
        inference_metadata=worker.metadata if worker else None,
        recording=str(args.record_bag) if args.record_bag else None,
        parent_cpus=sorted(os.sched_getaffinity(0)), inference_cpus=args.inference_cpus,
        loopback_dds=args.loopback_dds, policy_compute=args.policy,
        hold_seconds=args.hold_seconds, seconds=args.seconds,
        crc_failures=backend.crc_failures if backend else None,
        loop=stats([r['loop_s'] for r in rows]),
        phases={k:stats([r['writer'][k] for r in rows if r['writer']]) for k in fields},
        writer=writer.writer_diagnostics() if writer else None,
        sink=sink_result, sdk_publisher_attempts=attempts,
        motor_transport=False, real_motor_test=False,
        note='Production packet code to discard sink or private DDS domain 197 on lo only.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    args.output.with_suffix('.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
