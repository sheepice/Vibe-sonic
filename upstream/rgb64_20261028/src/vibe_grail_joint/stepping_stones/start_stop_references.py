"""Causal SONIC start/stop reference windows for an isolated training extension.

The reference generator is the unchanged deploy OnlineReference. Every control
tick stores the prediction available THEN, including its 0.9 s lookahead. It
must not be replaced by future poses collected after later planner revisions.
Only the fixed robot kinematic model and velocity schedule are used; there is
no terrain, rendered observation, physical robot state, contact, or depth input.
This module generates reference data; it does not launch or alter training.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from ..kinematics import CHANNELS, angular_velocity

DT = .02
FRAME_STRIDE = 50
CURRENT_OFFSET = 2
FUTURE_OFFSETS = CURRENT_OFFSET + np.arange(10)*5


def command_schedule(duration, speed, *, start=1., stop_hold=3.):
    if (not all(np.isfinite(x) for x in (duration, speed, start, stop_hold))
            or speed not in (.25, .4) or duration <= start+stop_hold+1.
            or abs(duration/DT-round(duration/DT)) > 1e-6):
        raise ValueError('Invalid start/stop reference schedule')
    times = np.arange(round(duration/DT))*DT
    wanted = np.where((times >= start) & (times < duration-stop_hold), speed, 0.)
    commands = np.empty(len(times), np.float32)
    value = 0.
    for tick, target in enumerate(wanted):
        value += float(np.clip(target-value, -.01, .01))
        commands[tick] = value
    return times, wanted.astype(np.float32), commands


def causal_window(reference, elapsed, model, data, body_ids):
    """Return 50 named FK frames from one CURRENT planner buffer snapshot."""
    import mujoco
    from ..deploy.reference import interpolate
    with reference.lock:
        poses, begin = reference.poses.copy(), reference.start
    at = elapsed/DT-begin
    locations = at+np.arange(FRAME_STRIDE)-CURRENT_OFFSET
    if locations[-1] > len(poses)-1+1e-6 or at < -1e-6:
        raise ValueError('The current planner prediction does not cover the required horizon')
    # The initial standing snapshot has no negative-time history. Only those
    # two padding frames may repeat the first pose; the future is never padded.
    locations = np.maximum(locations, 0.)
    block = interpolate(poses, locations)
    positions = np.empty((FRAME_STRIDE, len(body_ids), 3), np.float32)
    rotations = np.empty((FRAME_STRIDE, len(body_ids), 4), np.float32)
    for index, pose in enumerate(block):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        positions[index] = data.xpos[body_ids]
        rotations[index] = data.xquat[body_ids]
    linear = np.gradient(positions, DT, axis=0).astype(np.float32)
    angular = angular_velocity(rotations, DT)

    def sample_derivative(values):
        return np.stack([np.interp(locations, np.arange(len(poses)), values[:, j])
                         for j in range(values.shape[-1])], axis=-1).astype(np.float32)

    # Root twist and joint dq use the SAME full-buffer derivatives as deployment,
    # including its past context. Derivatives of a stitched/final trajectory
    # would silently use future replans and create a different actor input.
    linear[:, 0] = sample_derivative(np.gradient(poses[:, :3], DT, axis=0))
    angular[:, 0] = sample_derivative(angular_velocity(poses[:, 3:7], DT))
    return dict(joint_pos=block[:, 7:],
                joint_vel=sample_derivative(np.gradient(poses[:, 7:], DT, axis=0)),
                body_pos_w=positions, body_quat_w=rotations,
                body_lin_vel_w=linear, body_ang_vel_w=angular)


def actor_reference_from_window(window):
    from ..deploy.observations import Reference
    return Reference(window['joint_pos'][FUTURE_OFFSETS], window['joint_vel'][FUTURE_OFFSETS],
                     window['body_quat_w'][FUTURE_OFFSETS, 0],
                     window['body_lin_vel_w'][CURRENT_OFFSET, 0],
                     window['body_ang_vel_w'][CURRENT_OFFSET, 0])


def reference_errors(actual, expected):
    return {name:float(np.max(np.abs(getattr(actual, name)-getattr(expected, name))))
            for name in ('joint_pos', 'joint_vel', 'root_quat', 'root_lin_vel', 'root_ang_vel')}


class CausalReferenceCache:
    """Read-only CPU cache. Training can append its blocks to the existing bank."""
    def __init__(self, path, *, verify_hashes=True, require_validation=True):
        from .deployment_suite import digest
        self.path = Path(path)
        self.manifest = json.loads((self.path/'references.json').read_text())
        if (self.manifest.get('version') != 1 or not self.manifest.get('complete')
                or self.manifest.get('frame_stride') != FRAME_STRIDE
                or self.manifest.get('current_offset') != CURRENT_OFFSET
                or self.manifest.get('control_dt') != DT
                or self.manifest.get('privileged_inputs') != []):
            raise ValueError('Incompatible or incomplete causal reference cache')
        if require_validation:
            validation_path = self.path/'validation.json'
            if not validation_path.is_file():
                raise ValueError('Reference cache has not passed deployment API validation')
            validation = json.loads(validation_path.read_text())
            error = validation.get('roundtrip_max_abs_error', float('inf'))
            if (validation.get('passed') is not True or not np.isfinite(error) or error>2e-5):
                raise ValueError('Reference cache has not passed deployment API validation')
        self.clips = []
        for clip in self.manifest['clips']:
            arrays = {}
            for name, info in clip['files'].items():
                source = self.path/info['path']
                if verify_hashes and digest(source) != info['sha256']:
                    raise ValueError('Changed reference cache: '+str(source))
                array = np.load(source, mmap_mode='r', allow_pickle=False)
                shape = (clip['control_steps']*FRAME_STRIDE, *CHANNELS[name]) if name in CHANNELS else (clip['control_steps'],)
                if array.dtype != np.float32 or array.shape != shape:
                    raise ValueError('Wrong reference cache array shape/type: '+name)
                arrays[name] = array
            if set(arrays) != set(CHANNELS) | {'command', 'wanted'}:
                raise ValueError('Incomplete reference cache channels')
            self.clips.append(arrays)

    def reference(self, clip, tick):
        if not 0 <= clip < len(self.clips):
            raise IndexError('Unknown reference clip')
        count = self.manifest['clips'][clip]['control_steps']
        if not 0 <= tick < count:
            raise IndexError('Reference ended; no automatic wrap or padding')
        offset = tick*FRAME_STRIDE
        return actor_reference_from_window({name:self.clips[clip][name][offset:offset+FRAME_STRIDE]
                                            for name in CHANNELS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true', help='Two eight-second CPU traces for interface validation')
    args = parser.parse_args()
    from ..deploy.reference import OnlineReference
    from ..deploy.runtime import session
    from ..prepare import build_model, model_signature
    from .deployment_suite import digest, utc, write
    import mujoco
    bundle = Path(args.bundle).resolve()
    export = json.loads((bundle/'bundle.json').read_text())
    planner = bundle/'planner.onnx'
    if digest(planner) != export['files']['planner.onnx']:
        raise ValueError('Changed SONIC planner')
    contract = export['robot']
    model, names = build_model()
    joint_names = [model.joint(j).name for j in range(model.njnt)
                   if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if model.nq != 36 or joint_names != contract['joint_names'] or len(names) != 14:
        raise ValueError('Reference FK model does not match G1 contract')
    body_ids = [model.body(name).id for name in names]
    data = mujoco.MjData(model)
    inference = session(planner, 'cpu')
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=1, started_utc=utc(), complete=False, cpu_only=True,
                    frame_stride=FRAME_STRIDE, current_offset=CURRENT_OFFSET, control_dt=DT,
                    future_samples=10, future_skip=5, privileged_inputs=[],
                    generator='unchanged deploy.reference.OnlineReference',
                    semantics='prediction available at each control tick; future replans never substituted',
                    robot=contract, body_names=list(names), model_signature=model_signature(model,names),
                    planner_sha256=export['files']['planner.onnx'], clips=[], smoke=args.smoke,
                    source_sha256={str(p):digest(p) for p in (Path(__file__), Path(__file__).parent.parent/'deploy/reference.py')})
    write(out/'references.incomplete.json', manifest)
    started = time.monotonic()
    specs = [(speed, seed, 8. if args.smoke else (28. if speed==.4 else 40.))
             for speed, training_seed in ((.25,138),(.4,139))
             for seed in ([42] if args.smoke else [42,training_seed])]
    for speed, seed, duration in specs:
        key = f'v{str(speed).replace(".","p")}_seed{seed}'
        destination = out/key
        destination.mkdir()
        times, wanted, command = command_schedule(duration, speed)
        arrays = {name:np.lib.format.open_memmap(destination/(name+'.npy'), mode='w+',
                      dtype=np.float32, shape=(len(times)*FRAME_STRIDE,*tail))
                  for name,tail in CHANNELS.items()}
        np.save(destination/'command.npy', command)
        np.save(destination/'wanted.npy', wanted)
        reference = OnlineReference(planner, contract, seed=seed, session_override=inference)
        errors = {}
        selected = []
        for tick, elapsed in enumerate(times):
            if tick%10==0:
                reference.replan(float(elapsed),float(command[tick]))
            block = causal_window(reference, float(elapsed), model, data, body_ids)
            for name,values in block.items():
                if not np.isfinite(values).all():
                    raise ValueError('Non-finite causal reference data')
                arrays[name][tick*FRAME_STRIDE:(tick+1)*FRAME_STRIDE] = values
            actual, expected = actor_reference_from_window(block), reference.sample(float(elapsed))
            for name, error in reference_errors(actual,expected).items():
                errors[name] = max(errors.get(name,0.),error)
            if max(errors.values()) > 2e-5:
                raise ValueError(f'Deployment-reference mismatch: {errors}')
            # Store a few independently obtained API outputs for round-trip checks.
            if tick in (0,49,50,60,round((duration-3.)/DT),len(times)-1):
                selected.append((tick,expected))
        for array in arrays.values():
            array.flush()
        del arrays
        files = {}
        for name in (*CHANNELS,'command','wanted'):
            source = destination/(name+'.npy')
            files[name] = dict(path=str(source.relative_to(out)),sha256=digest(source),bytes=source.stat().st_size)
        manifest['clips'].append(dict(id=key,speed_m_s=speed,seed=seed,duration_s=duration,
                                      control_steps=len(times),files=files,reference_max_abs_errors=errors))
        np.savez(destination/'selected_deploy_api.npz',ticks=np.array([t for t,_ in selected]),
                 **{name:np.stack([getattr(ref,name) for _,ref in selected]) for name in errors})
        write(out/'references.incomplete.json',manifest)
        print(json.dumps(dict(clip=key,steps=len(times),reference_max_abs_errors=errors)),flush=True)
    manifest.update(complete=True,finished_utc=utc(),elapsed_wall_s=time.monotonic()-started,
                    total_array_bytes=sum(info['bytes'] for clip in manifest['clips'] for info in clip['files'].values()))
    write(out/'references.json',manifest)
    cache = CausalReferenceCache(out, require_validation=False)
    roundtrip = 0.
    revision = []
    for index,clip in enumerate(manifest['clips']):
        from ..deploy.observations import Reference
        with np.load(out/clip['id']/'selected_deploy_api.npz',allow_pickle=False) as rows:
            for k,tick in enumerate(rows['ticks']):
                expected = Reference(*(rows[name][k] for name in ('joint_pos','joint_vel','root_quat','root_lin_vel','root_ang_vel')))
                roundtrip = max(roundtrip,max(reference_errors(cache.reference(index,int(tick)),expected).values()))
        past_future = np.stack([cache.reference(index,tick).joint_pos[-1] for tick in range(clip['control_steps']-45)])
        later_current = np.stack([cache.reference(index,tick+45).joint_pos[0] for tick in range(clip['control_steps']-45)])
        revision.append(dict(clip=clip['id'],lookahead_revision_rmse_rad=float(np.sqrt(np.mean((past_future-later_current)**2)))))
    if roundtrip>2e-5:
        raise ValueError('Saved reference cache failed deployment API roundtrip')
    write(out/'validation.json',dict(passed=True,roundtrip_max_abs_error=roundtrip,
          lookahead_revisions=revision,full_horizon_causal=True,
          training_integration_validated=False,physics_or_hardware_validated=False))
    print(json.dumps(dict(output=str(out),bytes=manifest['total_array_bytes'],
                          roundtrip_max_abs_error=roundtrip,wall_s=manifest['elapsed_wall_s'])),flush=True)


if __name__=='__main__':
    main()
