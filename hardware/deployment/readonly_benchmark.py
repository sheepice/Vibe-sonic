"""Exercise deployed sensors/inference; motor publishing is forbidden.

Optional packet load creates, checksums and serializes local objects only.
No arm, target submission, controller handoff or DDS writer is used.
"""
import argparse
import json
from pathlib import Path
import threading
import time
import sys
import gc
import os

import numpy as np

from vibe_grail_joint.deploy.inference_process import InferenceProcess
from vibe_grail_joint.deploy.safety import Limits, health_fault, prepare_target
from vibe_grail_joint.deploy.unitree import UnitreeBackend


class PacketLoad:
    def __init__(self,backend,contract):
        self.backend,self.contract = backend,contract
        self.stop = threading.Event()
        self.times,self.lateness = [],[]
        self.error = None
        self.target = None
        self.faults = set()
        self.deadline_faults = 0
        self.thread = threading.Thread(target=self.run,name='local-serialization-only',daemon=True)
        self.thread.start()

    def run(self):
        try:
            deadline = time.monotonic()
            packet = self.backend.make_lowcmd()
            while not self.stop.is_set():
                begin = time.monotonic()
                self.lateness.append(max(0.,begin-deadline))
                state = self.backend.latest()
                with self.backend.lock:
                    target = self.target
                if target is None:
                    deadline = time.monotonic()
                    self.stop.wait(.002)
                    continue
                q,produced,image_at,command_at = target
                now = time.monotonic()
                fault = health_fault(state,now,image_at,command_at,Limits())
                if fault:
                    self.faults.add(fault)
                if now-produced>Limits().target_age_s:
                    self.faults.add('policy_watchdog')
                packet.mode_machine = state.mode_machine
                with self.backend.lock:
                    for i,motor in enumerate(packet.motor_cmd[:29]):
                        motor.mode = 1
                        motor.q = float(q[i])
                        motor.kp = float(self.contract['stiffness'][i])
                        motor.kd = float(self.contract['damping'][i])
                    packet.crc = self.backend.crc.Crc(packet)
                    # Discard bytes locally. This is not a transport send.
                    packet.serialize(use_version_2=None)  # actual DataWriter default
                self.times.append(time.monotonic()-begin)
                deadline += .002
                self.stop.wait(max(0.,deadline-time.monotonic()))
                if time.monotonic()-deadline>.02:
                    self.deadline_faults += 1
                    deadline = time.monotonic()
        except BaseException as exc:
            self.error = repr(exc)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2.)

    def set_target(self,q,image_at,command_at):
        with self.backend.lock:
            self.target=(q.copy(),time.monotonic(),image_at,command_at)


def stats(values):
    if not values:
        return None
    return dict(p50_ms=float(np.percentile(values,50)*1000),
                p99_ms=float(np.percentile(values,99)*1000),max_ms=float(max(values)*1000))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--engine-dir',type=Path,required=True)
    parser.add_argument('--interface',required=True)
    parser.add_argument('--serial',required=True)
    parser.add_argument('--seconds',type=float,default=60.)
    parser.add_argument('--speeds',default='0,0.15,0.25,0.35,0')
    parser.add_argument('--packet-load',action='store_true')
    parser.add_argument('--render',action='store_true',help='Offscreen render cost only; no physical keyboard verification')
    parser.add_argument('--remote-moves',action='store_true',help='Also exercise backward and turning model inputs; discard all outputs')
    parser.add_argument('--inference-cpus',help='Optional Jetson CPU affinity for the model child')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    sys.setswitchinterval(.001)  # same DDS/writer fairness setting as run.main
    speeds = [float(x) for x in args.speeds.split(',')]
    if not np.isfinite(args.seconds) or args.seconds<=0 or not speeds or any(not 0<=s<=.7 for s in speeds):
        parser.error('Invalid duration/speeds')
    if args.output.exists() or args.output.with_suffix('.jsonl').exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Fail closed if future edits accidentally attempt to create a publisher.
    from unitree_sdk2py.core import channel
    publisher_attempts = []
    def forbidden_publisher(*a,**kw):
        publisher_attempts.append(repr(a))
        raise RuntimeError('This benchmark forbids all DDS publishers')
    channel.ChannelPublisher = forbidden_publisher
    manifest = json.loads((args.bundle/'bundle.json').read_text())
    contract = manifest['robot']
    default,scale = (np.asarray(contract[k],np.float32) for k in ('default_joint_pos','scale'))
    backend = worker = load = window = None
    durations,roundtrips,policies,plans,state_ages,image_ages = [],[],[],[],[],[]
    faults = set()
    late = 0
    failure = None
    metadata = None
    gc_start = {}
    gc_pauses = []
    def gc_event(phase,info):
        generation=info['generation']
        if phase=='start':
            gc_start[generation]=time.monotonic()
        elif generation in gc_start:
            gc_pauses.append(time.monotonic()-gc_start.pop(generation))
    gc.callbacks.append(gc_event)
    parts={key:[] for key in ('state','target_check','health_check','logging')}
    count = round(args.seconds*50)
    movement_commands = ([(0.,0.),(.15,0.),(.25,0.),(.35,0.),(-.1,0.),(0.,.2),(0.,-.2),(0.,0.)]
                         if args.remote_moves else [(speed,0.) for speed in speeds])
    try:
        backend = UnitreeBackend(args.interface,contract,execute=False)
        until = time.monotonic()+10
        while backend.state is None:
            if time.monotonic()>until:
                raise RuntimeError('No robot state')
            time.sleep(.01)
        worker = InferenceProcess(dict(bundle=str(args.bundle.resolve()),provider='tensorrt',
            engine_dir=str(args.engine_dir.resolve()),serial=args.serial,planner_threads=3,cpus=args.inference_cpus))
        metadata = worker.metadata
        if args.render:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            os.environ['SDL_AUDIODRIVER'] = 'dummy'
            from vibe_grail_joint.deploy.run import KeyboardWindow
            window = KeyboardWindow(.35,on_stop=lambda reason: None)
        if args.packet_load:
            load = PacketLoad(backend,contract)
        previous = backend.latest().q.copy()
        previous_action = None
        speed = 0.
        heading = yaw_rate = 0.
        origin = time.monotonic()
        with args.output.with_suffix('.jsonl').open('x',buffering=1) as log:
            for step in range(count):
                time.sleep(max(0.,origin+step*.02-time.monotonic()))
                begin = time.monotonic()
                late += int(begin-(origin+step*.02)>.02)
                if window:
                    window.poll('READ ONLY: offscreen drawing cost; no keyboard validation')
                command_at = time.monotonic()
                state = backend.latest()
                parts['state'].append(time.monotonic()-command_at)
                requested,requested_yaw = movement_commands[min(len(movement_commands)-1,step*len(movement_commands)//count)]
                speed += float(np.clip(requested-speed,-.01,.01))
                yaw_rate += float(np.clip(requested_yaw-yaw_rate,-.02,.02))
                heading += yaw_rate*.02
                infer_start = time.monotonic()
                result = worker.step(state,speed,previous_action=previous_action,reset=step==0,
                                     command_heading=heading,yaw_rate=yaw_rate)
                roundtrips.append(time.monotonic()-infer_start)
                action,target = result['action'],result['target']
                if result['source_state_at'] != state.received_at:
                    raise RuntimeError('Source state timestamp changed in transit')
                part_start=time.monotonic()
                target,projection,slew,diagnostic = prepare_target(target,previous,state,contract,Limits())
                previous_action = (target-default)/scale
                parts['target_check'].append(time.monotonic()-part_start)
                part_start=time.monotonic()
                current = backend.latest()
                now = time.monotonic()
                for snapshot in (state,current):
                    state_ages.append(now-snapshot.received_at)
                    fault = health_fault(snapshot,now,result['image_at'],command_at,Limits())
                    if fault:
                        faults.add(fault)
                if backend.fault:
                    faults.add(backend.fault)
                image_ages.append(now-result['image_at'])
                plans.append(result['planner_s'])
                policies.append(result['policy_s'])
                assert not backend.execute and not backend.armed and backend.publisher is None
                if load and load.error:
                    raise RuntimeError(load.error)
                if load:
                    load.set_target(target,result['image_at'],command_at)
                parts['health_check'].append(time.monotonic()-part_start)
                part_start=time.monotonic()
                log.write(json.dumps(dict(event='readonly_tick',speed_command=speed,
                    heading_command=heading,yaw_rate_command=yaw_rate,action=action.tolist(),
                    q_target=target.tolist(),q=state.q.tolist(),quat=state.quat.tolist(),
                    policy_s=result['policy_s'],image_age_s=now-result['image_at'],target_check=diagnostic,
                    target_error_max_rad=float(np.max(abs(target-state.q))),
                    target_step_max_rad=float(np.max(abs(target-previous))),
                    lowcmd_enabled=False,frame_id=result['frame_id'],planner_s=result['planner_s']))+'\n')
                previous = target.copy()
                parts['logging'].append(time.monotonic()-part_start)
                durations.append(time.monotonic()-begin)
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        if load:
            load.close()
        if backend:
            backend.close()
        if worker:
            worker.close()
        if window:
            window.close()
        gc.callbacks.remove(gc_event)
    passed = (failure is None and len(durations)==count and not faults and not publisher_attempts and not late
              and max(durations,default=float('inf'))<=.04
              and (load is None or (not load.error and not load.faults and not load.deadline_faults
                                   and max(load.lateness,default=0)<=.02)))
    report = dict(completed=failure is None,failure=failure,software_health_pass=passed,
        every_cycle_within_20ms=bool(durations) and max(durations)<=.02,
        samples=len(durations),requested_seconds=args.seconds,speeds=speeds,
        movement_commands=[dict(speed=speed,yaw_rate=rate) for speed,rate in movement_commands],
        loop=stats(durations),inference_roundtrip=stats(roundtrips),policy=stats(policies),planner=stats(plans),
        max_state_age_s=max(state_ages,default=None),max_camera_age_s=max(image_ages,default=None),
        late_ticks_over_20ms=late,health_faults=sorted(faults),metadata=metadata,
        crc_failures=backend.crc_failures if backend else None,
        lowcmd_publisher_created=False,publisher_attempts=publisher_attempts,
        packet_load=bool(load),packet_compute=stats(load.times) if load else None,
        packet_lateness=stats(load.lateness) if load else None,
        packet_samples=len(load.times) if load else 0,
        packet_health_faults=sorted(load.faults) if load else [],
        packet_deadline_faults=load.deadline_faults if load else 0,
        python_gc_pauses=stats(gc_pauses),component_timings={k:stats(v) for k,v in parts.items()},
        offscreen_render_test=args.render,physical_keyboard_verified=False,
        remote_movement_inputs=args.remote_moves,
        inference_cpus=args.inference_cpus,
        parent_cpu_affinity=sorted(os.sched_getaffinity(0)),
        hardware_sensor_test=True,hardware_control_test=False,
        note='Production inference path, actual sensors, outputs discarded. Local packet serialization '
             'is not a network writer test. Does not validate balance, mount calibration or physical stopping.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not passed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
