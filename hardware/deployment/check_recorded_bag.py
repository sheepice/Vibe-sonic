"""Replay sampled recorded ONNX inputs on CPU; no camera, SDK or DDS access."""
import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader

from vibe_grail_joint.deploy.runtime import Runtime
from vibe_grail_joint.deploy.visualize_bag import array, rgb


def check(bag, bundle, samples=12, atol=1e-5):
    runtime = Runtime(bundle, 'cpu')
    errors = dict(actions=0., target=0., encoder=0., attention_cpu_difference=0.)
    rows = []
    try:
        with AnyReader([Path(bag)]) as reader:
            steps, metadata = [], None
            for conn, stamp, data in reader.messages(connections=[c for c in reader.connections
                    if c.topic in ('/vibe/policy/step','/vibe/recording/metadata')]):
                item = json.loads(reader.deserialize(data,conn.msgtype).data)
                if conn.topic=='/vibe/recording/metadata':
                    metadata = item
                elif item['compute']:
                    steps.append((stamp,item))
            if metadata is None or metadata['manifest']['files']!=runtime.manifest['files']:
                raise ValueError('Bag model hashes differ from requested bundle')
            if not steps:
                raise ValueError('No computed policy steps in bag')
            def read_at(stamp, topics):
                result = {}
                for conn, _, data in reader.messages(connections=[c for c in reader.connections
                        if c.topic in topics], start=stamp, stop=stamp+1):
                    result[conn.topic] = reader.deserialize(data,conn.msgtype)
                if set(result)!=set(topics):
                    raise ValueError(f'Missing records at {stamp}: {set(topics)-set(result)}')
                return result
            topics = ['/vibe/policy/'+n for n in ('actions','raw_target','attention')]
            topics += ['/vibe/policy/inputs/'+k for k in runtime.inputs]
            image_topics = ['/vibe/camera/policy/image','/vibe/vision/patches','/vibe/vision/cls']
            for i in np.unique(np.linspace(0,len(steps)-1,min(samples,len(steps))).astype(int)):
                stamp, step = steps[i]
                data = read_at(stamp,topics)
                feed = {k:array(data['/vibe/policy/inputs/'+k]) for k in runtime.inputs}
                action,target,_,attention = runtime.act(feed,with_attention=True)
                expected = array(data['/vibe/policy/attention'])
                assert np.isfinite(expected).all() and np.all(expected>=0)
                np.testing.assert_allclose(expected.sum(-1),1.,atol=1e-5)
                source = read_at(step['image_stamp_ns'],image_topics)
                patches,cls = (array(source['/vibe/vision/'+n]) for n in ('patches','cls'))
                np.testing.assert_array_equal(patches,feed['kv_tokens__img_tokens'][0])
                np.testing.assert_array_equal(cls,feed['q_cls'][0])
                actual = runtime.encode(rgb(source['/vibe/camera/policy/image'])[:,:,::-1].copy())
                row = dict(step_id=step['step_id'],source_frame=step['image_frame_id'],
                    actions=float(np.max(abs(action-array(data['/vibe/policy/actions'])))),
                    target=float(np.max(abs(target-array(data['/vibe/policy/raw_target'])))),
                    encoder=max(float(np.max(abs(a-b))) for a,b in zip(actual,(patches,cls))),
                    attention_cpu_difference=float(np.max(abs(attention-expected))))
                for key in errors:
                    errors[key] = max(errors[key],row[key])
                rows.append(row)
    finally:
        runtime.close()
    return dict(passed=all(errors[k]<=atol for k in ('actions','target','encoder')),
        samples=len(rows),max_abs_error=errors,action_encoder_atol=atol,
        exact_input_feature_join=True,attention_normalized=True,rows=rows,
        motor_transport=False, note='Attention CPU/GPU difference reported separately; not used for motor outputs.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag',required=True,type=Path)
    parser.add_argument('--bundle',required=True,type=Path)
    parser.add_argument('--samples',type=int,default=12)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.samples<1:
        parser.error('--samples must be positive')
    report=check(args.bag,args.bundle,args.samples)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
