"""Static-shape TensorRT sessions backed by the robot's existing CUDA runtime.

Weights and input/output names remain tied to the original ONNX hash. Pinned
host buffers and packed device buffers avoid per-tick allocation and eight
separate input transfers. This module has no robot transport capabilities.
"""
import ctypes as ct
import ctypes.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np

from .export import sha256


SPATIAL_ANCHOR_SHAPES = {
    f'/extractors.0/blocks.{i}/{suffix}_output_0': shape
    for i in range(3) for suffix, shape in (
        ('qkv/MatMul', [1,28,576]), ('MatMul', [1,3,28,28]),
        ('attention_out/MatMul', [1,28,192]), ('ff/ff.0/MatMul', [1,28,384]),
        ('ff/ff.2/MatMul', [1,28,192]))
}


class CudaRuntime:
    def __init__(self):
        library = ctypes.util.find_library('cudart') or '/usr/local/cuda/lib64/libcudart.so.12'
        self.lib = ct.CDLL(library)
        signatures = dict(cudaSetDevice=[ct.c_int],
            cudaMalloc=[ct.POINTER(ct.c_void_p),ct.c_size_t],
            cudaFree=[ct.c_void_p],cudaFreeHost=[ct.c_void_p],
            cudaHostAlloc=[ct.POINTER(ct.c_void_p),ct.c_size_t,ct.c_uint],
            cudaStreamCreateWithFlags=[ct.POINTER(ct.c_void_p),ct.c_uint],
            cudaStreamDestroy=[ct.c_void_p],cudaStreamSynchronize=[ct.c_void_p],
            cudaMemcpyAsync=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int,ct.c_void_p])
        for name,args in signatures.items():
            func = getattr(self.lib,name)
            func.argtypes,func.restype = args,ct.c_int
        self.lib.cudaGetErrorString.argtypes = [ct.c_int]
        self.lib.cudaGetErrorString.restype = ct.c_char_p

    def call(self,name,*args):
        error = getattr(self.lib,name)(*args)
        if error:
            message = self.lib.cudaGetErrorString(error).decode('utf-8',errors='replace')
            raise RuntimeError(f'{name}: {message} ({error})')


def validate_cache(source,engine,*,allow_unvalidated=False):
    source,engine = Path(source),Path(engine)
    data = json.loads(engine.with_suffix('.json').read_text())
    if (data.get('schema') != 'vibe.tensorrt_cache.v1'
            or data.get('precision') != 'float32_no_tf32'
            or data.get('source_sha256') != sha256(source)
            or data.get('engine_sha256') != sha256(engine)):
        raise ValueError('TensorRT cache/source checksum or precision mismatch')
    adapters = data.get('input_adapters',{})
    expected = {'image_bgr_hwc':'uint8_to_float32_exact'} if source.stem == 'encoder' else {}
    if adapters != expected:
        raise ValueError('Unverified TensorRT input adapter')
    anchors = data.get('output_anchors', [])
    expected_anchors = list(SPATIAL_ANCHOR_SHAPES)
    if anchors and (source.stem != 'policy' or anchors != expected_anchors):
        raise ValueError('Unverified TensorRT output anchors')
    if not allow_unvalidated:
        report = engine.parent/'parity.json'
        if not data.get('numerically_validated') or data.get('parity_sha256') != sha256(report):
            raise ValueError('TensorRT cache has no matching numerical validation')
        checked = json.loads(report.read_text())
        if (not checked.get('passed') or checked.get('engines',{}).get(source.stem) != sha256(engine)
                or checked.get('sources',{}).get(source.stem) != sha256(source)):
            raise ValueError('TensorRT numerical report does not match this model')
        tolerance = checked.get('tolerance_absolute',float('nan'))
        # Two verified exporters use different names for the same two features.
        declared = set(checked.get('required_outputs', []))
        feature_outputs = next((names for names in (
            {'encoder/patch_tokens', 'encoder/cls_token'}, {'encoder/patches', 'encoder/cls'})
            if names.issubset(declared)), None)
        required = {'policy_same_inputs/actions','encoder_policy_chain/actions'}
        required.update(feature_outputs or {'missing_encoder_checks'})
        errors = checked.get('max_absolute_error',{})
        if (not np.isfinite(tolerance) or not 0<tolerance<=1e-5
                or checked.get('cases',0)<32
                or checked.get('cases') != checked.get('requested_cases')
                or checked.get('failure') is not None
                or not required.issubset(checked.get('required_outputs',[]))
                or any(not np.isfinite(errors.get(name,float('nan')))
                       or not 0<=errors[name]<=tolerance for name in required)):
            raise ValueError('TensorRT numerical report lacks passing control-output checks')
    return data


class TensorRTSession:
    def __init__(self,source,engine,*,allow_unvalidated=False):
        self.metadata = validate_cache(source,engine,allow_unvalidated=allow_unvalidated)
        import tensorrt as trt
        if trt.__version__ != self.metadata['tensorrt_version']:
            raise RuntimeError('TensorRT version differs from the compiled engine')
        self.lock = threading.Lock()
        self.cuda = CudaRuntime()
        self.stream = ct.c_void_p()
        self.allocations = []
        self.views = {}
        self.context = self.engine = self.runtime = None
        self.closed = False
        self.cuda.call('cudaSetDevice',0)
        self.logger = trt.Logger(trt.Logger.ERROR)
        try:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(Path(engine).read_bytes())
            if self.engine is None:
                raise RuntimeError('Cannot deserialize TensorRT engine')
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError('Cannot create TensorRT execution context')
            inputs,outputs = {},{}
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = list(self.engine.get_tensor_shape(name))
                if not shape or any(d <= 0 for d in shape):
                    raise ValueError('Only static TensorRT I/O shapes are supported')
                if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                    raise ValueError('Host-location TensorRT tensors are unsupported')
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
                if dtype != np.dtype(np.float32):
                    raise ValueError('Expected FP32 TensorRT tensors')
                external_type = ('tensor(uint8)' if name in self.metadata['input_adapters']
                                 else 'tensor(float)')
                item = SimpleNamespace(name=name,shape=shape,dtype=dtype,type=external_type)
                target = inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs
                target[name] = item
            anchor_names = self.metadata.get('output_anchors', [])
            if (set(inputs) != set(self.metadata['inputs'])
                    or set(outputs) != set(self.metadata['outputs']+anchor_names)):
                raise ValueError('TensorRT I/O names differ from original ONNX')
            self.inputs = [inputs[n] for n in self.metadata['inputs']]
            self.outputs = [outputs[n] for n in self.metadata['outputs']]
            self.input_block = self._allocate_block(self.inputs)
            # Extra spatial tensors prevent incorrect fusion but never become model
            # features or actions. Allocate them without changing get_outputs.
            anchors = [outputs[n] for n in anchor_names]
            if any(x.shape != SPATIAL_ANCHOR_SHAPES[x.name] for x in anchors):
                raise ValueError('Unexpected spatial attention anchor shape')
            self.output_block = self._allocate_block(self.outputs+anchors)
            self.cuda.call('cudaStreamCreateWithFlags',ct.byref(self.stream),1)
        except BaseException:
            self.close()
            raise

    def _allocate_block(self,items):
        offsets,total = {},0
        for item in items:
            offsets[item.name] = total
            size = int(np.prod(item.shape))*item.dtype.itemsize
            total += (size+255)//256*256
        host,device = ct.c_void_p(),ct.c_void_p()
        self.cuda.call('cudaHostAlloc',ct.byref(host),total,0)
        self.allocations.append(('cudaFreeHost',host))
        self.cuda.call('cudaMalloc',ct.byref(device),total)
        self.allocations.append(('cudaFree',device))
        storage = np.ctypeslib.as_array((ct.c_uint8*total).from_address(host.value))
        storage.fill(0)
        for item in items:
            offset = offsets[item.name]
            self.views[item.name] = np.ndarray(item.shape,dtype=item.dtype,buffer=storage,offset=offset)
            if not self.context.set_tensor_address(item.name,device.value+offset):
                raise RuntimeError(f'Cannot bind TensorRT tensor {item.name}')
        return host,device,total

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_providers(self):
        return ['TensorRTNative']

    def run(self,output_names,feed):
        with self.lock:
            if self.closed:
                raise RuntimeError('TensorRT session is closed')
            if set(feed) != {v.name for v in self.inputs}:
                raise ValueError('TensorRT input names mismatch')
            names = self.metadata['outputs'] if output_names is None else output_names
            if any(n not in self.metadata['outputs'] for n in names):
                raise ValueError('Unknown TensorRT output')
            for item in self.inputs:
                value = feed[item.name]
                expected = (np.dtype(np.uint8) if item.name in self.metadata['input_adapters']
                            else item.dtype)
                if (not isinstance(value,np.ndarray) or list(value.shape) != item.shape
                        or value.dtype != expected):
                    raise ValueError(f'Invalid TensorRT input {item.name}')
                np.copyto(self.views[item.name],value,casting='safe')
            host,device,size = self.input_block
            self.cuda.call('cudaMemcpyAsync',device,host,size,1,self.stream)
            if not self.context.execute_async_v3(self.stream.value):
                raise RuntimeError('TensorRT inference failed')
            host,device,size = self.output_block
            self.cuda.call('cudaMemcpyAsync',host,device,size,2,self.stream)
            self.cuda.call('cudaStreamSynchronize',self.stream)
            return [self.views[name].copy() for name in names]

    def close(self):
        if getattr(self,'closed',True):
            return
        self.closed = True
        if self.stream.value:
            self.cuda.call('cudaStreamSynchronize',self.stream)
        self.context = self.engine = self.runtime = None
        self.views.clear()
        for name,pointer in reversed(self.allocations):
            self.cuda.call(name,pointer)
        self.allocations.clear()
        if self.stream.value:
            self.cuda.call('cudaStreamDestroy',self.stream)
            self.stream = ct.c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
