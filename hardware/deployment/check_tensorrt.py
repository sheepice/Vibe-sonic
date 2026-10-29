"""Validate GPU caches against original ONNX on CPU; never imports a robot SDK."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from vibe_grail_joint.deploy.export import sha256
from vibe_grail_joint.deploy.runtime import Runtime
from vibe_grail_joint.deploy.tensorrt_runtime import TensorRTSession


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--engines',type=Path,required=True)
    p.add_argument('--image',type=Path)
    p.add_argument('--cases',type=int,default=96)
    a = p.parse_args()
    if a.cases < 32:
        p.error('Use at least 32 validation cases')
    report_path = a.engines/'parity.json'
    if report_path.exists():
        raise FileExistsError(report_path)
    cpu = Runtime(a.bundle,'cpu')
    cpu.warmup()
    sessions = {}
    differences = {}
    timings = {name:[] for name in ('cpu_encoder','gpu_encoder','cpu_policy','gpu_policy')}
    failure = None
    tolerance = min(float(cpu.manifest['policy_parity']['tolerance']),1e-5)
    rng = np.random.default_rng(20261019)
    count = 0
    actual = None
    if a.image:
        from PIL import Image
        actual = np.asarray(Image.open(a.image).convert('RGB'))[...,::-1].copy()
        if actual.shape != cpu.image_shape:
            raise ValueError(f'Expected policy camera image {cpu.image_shape}')
    def compare(label,left,right):
        if left.shape != right.shape or not np.isfinite(right).all():
            raise ValueError(f'Invalid GPU output: {label}')
        error = float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64))))
        differences[label] = max(differences.get(label,0.),error)
    def timed(label,session,feed):
        start = time.monotonic()
        output = session.run(None,feed)
        if count >= 5:
            timings[label].append(time.monotonic()-start)
        return output
    try:
        for name in ('encoder','policy'):
            sessions[name] = TensorRTSession(a.bundle/(name+'.onnx'),a.engines/(name+'.plan'),
                                            allow_unvalidated=True)
        for count in range(a.cases):
            image = rng.integers(0,256,cpu.image_shape,dtype=np.uint8)
            if count == 0:
                image.fill(0)
            elif count == 1:
                image.fill(255)
            elif actual is not None and count % 8 == 2:
                image = actual
            feed = {'image_bgr_hwc':image[None]}
            cpu_features = timed('cpu_encoder',cpu.encoder,feed)
            gpu_features = timed('gpu_encoder',sessions['encoder'],feed)
            for node,left,right in zip(cpu.encoder.get_outputs(),cpu_features,gpu_features,strict=True):
                compare('encoder/'+node.name,left,right)
            feed = {k:rng.normal(0,.25,size=shape).astype(np.float32) for k,shape in cpu.inputs.items()}
            feed['kv_tokens__img_tokens'],feed['q_cls'] = cpu_features
            original = timed('cpu_policy',cpu.policy,feed)
            same_inputs = timed('gpu_policy',sessions['policy'],feed)
            for node,left,right in zip(cpu.policy.get_outputs(),original,same_inputs,strict=True):
                compare('policy_same_inputs/'+node.name,left,right)
            feed['kv_tokens__img_tokens'],feed['q_cls'] = gpu_features
            gpu_chain = sessions['policy'].run(None,feed)
            for node,left,right in zip(cpu.policy.get_outputs(),original,gpu_chain,strict=True):
                compare('encoder_policy_chain/'+node.name,left,right)
        count += 1
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    finally:
        for session in sessions.values():
            session.close()
    # The original export gate and Runtime.act consume only `actions`. Attention
    # is an unused diagnostic output; keep its measured differences in the report
    # without turning it into a new control-output acceptance requirement.
    required = ['encoder/'+node.name for node in cpu.encoder.get_outputs()]
    required += ['policy_same_inputs/actions','encoder_policy_chain/actions']
    passed = (failure is None and count == a.cases
              and all(differences.get(name,float('inf')) <= tolerance for name in required))
    report = dict(passed=passed,cases=count,requested_cases=a.cases,failure=failure,
        tolerance_absolute=tolerance,max_absolute_error=differences,
        image_shape=cpu.image_shape, patch_shape=cpu.patch_shape,
        required_outputs=required,diagnostic_only_outputs=[n for n in differences if n not in required],
        sources={n:sha256(a.bundle/(n+'.onnx')) for n in ('policy','encoder')},
        engines={n:sha256(a.engines/(n+'.plan')) for n in ('policy','encoder')},
        timing_ms={k:dict(p50=float(np.percentile(v,50)*1000),p99=float(np.percentile(v,99)*1000))
                   for k,v in timings.items() if v},
        hardware_control_test=False,note='Numerical FP32 parity only; no DDS/camera capture/motor commands.')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    if passed:
        for name in ('policy','encoder'):
            path = a.engines/(name+'.json')
            data = json.loads(path.read_text())
            data.update(numerically_validated=True,parity_sha256=sha256(report_path))
            path.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not passed:
        raise RuntimeError('GPU numerical validation failed; caches remain unusable for deployment')


if __name__ == '__main__':
    main()
