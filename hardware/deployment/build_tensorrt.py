"""Build FP32/no-TF32 GPU caches; preserve the original ONNX bundle byte-for-byte.

The encoder's uint8 input is exactly promoted to float32 before channel/shape
operations, because TensorRT 10.3 cannot Slice uint8. All weights are unchanged.
Numerical equivalence must still be checked before selecting this backend.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def promote_encoder_input(model):
    import onnx
    if (len(model.graph.input) != 1 or model.graph.input[0].name != 'image_bgr_hwc'
            or model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.UINT8):
        raise ValueError('Expected the original uint8 BGR encoder input')
    promoted = {model.graph.input[0].name}
    casts = 0
    for node in model.graph.node:
        affected = [i for i,name in enumerate(node.input) if name in promoted]
        if not affected:
            continue
        if affected != [0]:
            raise ValueError('Image used as a shape/index input')
        if node.op_type == 'Cast':
            dtype = next(a.i for a in node.attribute if a.name == 'to')
            if dtype != onnx.TensorProto.FLOAT:
                raise ValueError('Encoder input does not cast directly to float32')
            casts += 1
        elif node.op_type in ('Slice','Gather','Transpose','Reshape','Identity'):
            promoted.update(node.output)
        else:
            raise ValueError(f'Unverified pre-cast operation: {node.op_type}')
    if not casts or any(v.name in promoted for v in model.graph.output):
        raise ValueError('Encoder preprocessing does not terminate in float32 casts')
    for value in [*model.graph.input,*model.graph.value_info]:
        if value.name in promoted:
            value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    onnx.checker.check_model(model)


def main():
    import onnx
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trtexec',default='/usr/src/tensorrt/bin/trtexec')
    p.add_argument('--models',nargs='+',choices=('policy','encoder'),default=['policy','encoder'])
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    manifest = json.loads((a.bundle/'bundle.json').read_text())
    for name in a.models:
        source = a.bundle/(name+'.onnx')
        if sha256(source) != manifest['files'][source.name]:
            raise ValueError('Original ONNX hash mismatch')
        engine = a.output/(name+'.plan')
        metadata = engine.with_suffix('.json')
        if engine.exists() or metadata.exists():
            raise FileExistsError(engine)
        model = onnx.load(source)
        inputs = [v.name for v in model.graph.input]
        outputs = [v.name for v in model.graph.output]
        adapters = {}
        output_anchors = []
        compiled_source = source
        if name == 'policy' and manifest.get('spatial_blocks') == 3:
            # TRT 10.3 on this Orin miscompiles the spatial attention path
            # with fused matrix products. Preserve these intermediate
            # tensors as extra engine outputs to prevent that
            # fusion. No weights or arithmetic are changed. They are hidden
            # from the public session outputs, and parity uses original ONNX.
            model = onnx.shape_inference.infer_shapes(model)
            infos = {v.name: v for v in model.graph.value_info}
            from vibe_grail_joint.deploy.tensorrt_runtime import SPATIAL_ANCHOR_SHAPES
            for tensor, shape in SPATIAL_ANCHOR_SHAPES.items():
                value = infos.get(tensor)
                if (value is None or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                        or [d.dim_value for d in value.type.tensor_type.shape.dim] != shape):
                    raise ValueError('Unrecognized spatial attention graph; refuse fusion workaround')
                model.graph.output.append(value)
                output_anchors.append(tensor)
            onnx.checker.check_model(model)
            compiled_source = a.output/'policy_preserved_spatial.onnx'
            onnx.save(model, compiled_source)
        if name == 'encoder':
            promote_encoder_input(model)
            compiled_source = a.output/'encoder_float_input.onnx'
            onnx.save(model,compiled_source)
            adapters['image_bgr_hwc'] = 'uint8_to_float32_exact'
        command = [a.trtexec,'--onnx='+str(compiled_source),'--saveEngine='+str(engine),
                   '--noTF32','--skipInference','--memPoolSize=workspace:1024']
        with (a.output/(name+'_build.log')).open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        if 'TensorRT version: 10.3.0' not in (a.output/(name+'_build.log')).read_text():
            raise ValueError('This cache builder requires the verified TensorRT 10.3.0 installation')
        metadata.write_text(json.dumps(dict(schema='vibe.tensorrt_cache.v1',
            source_sha256=sha256(source),compiled_source_sha256=sha256(compiled_source),
            engine_sha256=sha256(engine),tensorrt_version='10.3.0',
            precision='float32_no_tf32',inputs=inputs,outputs=outputs,input_adapters=adapters,
            output_anchors=output_anchors,
            build_command=command,numerically_validated=False),indent=2)+'\n')
        print(f'Built {engine}; numerical validation still required',flush=True)


if __name__ == '__main__':
    main()
