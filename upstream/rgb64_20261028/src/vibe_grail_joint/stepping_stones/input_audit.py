"""Enforce exported RGB input ports; test isolation from simulator-only fields.

This checks the deployment data boundary, not locomotion or hardware readiness.
The executable uses CPU ONNX and synthetic RGB/joint/IMU samples, with references
generated independently by the bundled SONIC planner. No simulator is imported.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


COMMON_POLICY = {
    'tokenizer': [1, 640], 'policy': [1, 930], 'augmentation': [1, 6],
    'velocity_command': [1, 8], 'hiking_latent': [1, 672],
}
VIBE_POLICY = {
    'kv_tokens__img_tokens': [1, 21, 192], 'q_task_cmd': [1, 6],
    'q_proprio': [1, 64], 'q_cls': [1, 192],
}
MEMORY = {
    'tokens': [1, 21, 192], 'proprio': [1, 64],
    'hidden': [1, 512], 'cell': [1, 512], 'reset': [1],
}


def validate_ports(runtime):
    """Reject extra/missing ports, including a privileged MoE routing input."""
    variant = runtime.manifest.get('variant')
    if variant not in ('dual', 'geometry_only', 'moe'):
        raise ValueError(f'Unsupported stone policy variant: {variant}')
    expected = dict(COMMON_POLICY)
    if variant != 'geometry_only':
        expected.update(VIBE_POLICY)
    contracts = [('policy', runtime.policy, expected),
                 ('memory', runtime.memory, MEMORY),
                 ('encoder', runtime.encoder, {'image_bgr_hwc': [1, 63, 112, 3]})]
    result = {}
    for name, graph, wanted in contracts:
        ports = {p.name: p.shape for p in graph.get_inputs()}
        if ports != wanted:
            raise ValueError(f'{name} violates deployable RGB input contract: {ports}')
        for port in graph.get_inputs():
            dtype = 'tensor(bool)' if port.name == 'reset' else (
                'tensor(uint8)' if name == 'encoder' else 'tensor(float)')
            if port.type != dtype:
                raise ValueError(f'{name}/{port.name}: unexpected type {port.type}')
        result[name] = ports
    for key in ('depth_input', 'privileged_actor_inputs', 'geometry_decoder_in_bundle'):
        if runtime.manifest.get(key) is not False:
            raise ValueError(f'Bundle must explicitly set {key}=false')
    return result


def audit(bundle, steps=32):
    from .runtime import StoneRuntime
    from ..deploy.observations import ObservationBuilder, RobotState
    from ..deploy.reference import OnlineReference
    from ..deploy.export import sha256

    runtime = StoneRuntime(bundle, 'cpu')
    ports = validate_ports(runtime)
    planner = OnlineReference(Path(bundle)/'planner.onnx', runtime.contract)
    builder = ObservationBuilder(runtime.contract)
    rng = np.random.default_rng(91825)
    sequence = []
    for step in range(steps):
        now = step*.02
        speed = 0. if step < 10 else .4
        if step % 10 == 0:
            planner.replan(now, speed)
        # Synthetic measured-equivalent states exercise the input construction;
        # no claims about physically executing these samples are made.
        state = RobotState(runtime.default+rng.normal(0., .005, 29),
                           rng.normal(0., .01, 29), np.array([1., 0., 0., 0.]),
                           rng.normal(0., .005, 3), now)
        bgr = rng.integers(0, 256, (63, 112, 3), dtype=np.uint8)
        sequence.append(builder.build(state, planner.sample(now),
                                      *runtime.encode(bgr), speed))
        builder.accept_action(rng.normal(0., .02, 29))

    def replay(inject):
        runtime.reset()
        values = []
        for step, obs in enumerate(sequence):
            if step == steps//2:
                runtime.reset()
            feed = dict(obs)
            if inject:
                # Poison training-only fields and attempt to override the RGB
                # latent. StoneRuntime must recompute it from its own memory.
                for key in ('critic', 'terrain_height', 'depth', 'contact_forces',
                            'base_lin_vel_b', 'terrain_id', 'oracle_footholds',
                            'fake_ground_truth', 'hiking_latent'):
                    feed[key] = np.full((1, 672), np.nan, np.float32)
            action, target, _ = runtime.act(feed)
            values.append(np.concatenate([action, target,
                                          runtime.hidden.ravel(), runtime.cell.ravel()]))
        return np.stack(values)

    clean, poisoned = replay(False), replay(True)
    error = float(np.abs(clean-poisoned).max())
    if not np.isfinite(clean).all() or not np.isfinite(poisoned).all() or error != 0.:
        raise RuntimeError(f'Training-only field isolation failed: {error}')
    forbidden = [name for name in ('torch', 'mujoco', 'mjlab') if name in sys.modules]
    if forbidden:
        raise RuntimeError(f'Portable audit unexpectedly imported {forbidden}')
    return dict(bundle=str(Path(bundle).resolve()), variant=runtime.manifest['variant'],
                bundle_sha256=sha256(Path(bundle)/'bundle.json'),
                model_files=runtime.manifest['files'], ports=ports,
                steps=steps, reset_steps=[0, steps//2],
                privileged_field_injection_max_abs_error=error,
                external_latent_override_ignored=True,
                synthetic_sensor_inputs=True, planner_source='independent bundled ONNX',
                simulator_imports=forbidden, no_hardware_commands_sent=True,
                interface_audit_passed=True, hardware_validated=False,
                limitation='Port/isolation audit only; cannot prove camera realism, sensor accuracy, locomotion or physical transfer.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.bundle)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: result[key] for key in (
        'bundle', 'variant', 'interface_audit_passed',
        'privileged_field_injection_max_abs_error', 'simulator_imports')}))


if __name__ == '__main__':
    main()
