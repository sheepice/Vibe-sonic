"""Read-only, asynchronous post-LayerNorm occlusion diagnostic for the web UI.

The original ONNX files and simulation policy sessions are never modified.
Only the ancestors of the named visual latent are evaluated here. LoRA in the
downstream action decoder is not part of this particular diagnostic.
"""
import base64
import hashlib
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

from .vendor import onnx_ml_pb2 as pb

LATENT = '/extractors.0/attention/out_norm/LayerNormalization_output_0'
VISUAL_INPUTS = ('kv_tokens__img_tokens', 'q_task_cmd', 'q_proprio', 'q_cls')


def latent_graph(binary):
    original = pb.ModelProto()
    original.ParseFromString(binary)
    producer = {v: i for i, node in enumerate(original.graph.node) for v in node.output}
    if LATENT not in producer:
        raise ValueError('This bundle has no supported post-LayerNorm visual output')
    seen, needed = set(), set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        if name in producer:
            i = producer[name]
            needed.add(i)
            for value in original.graph.node[i].input:
                visit(value)

    visit(LATENT)
    sliced = pb.ModelProto()
    sliced.CopyFrom(original)
    for field in ('node', 'initializer', 'input', 'output', 'value_info'):
        sliced.graph.ClearField(field)
    sliced.graph.node.extend(n for i, n in enumerate(original.graph.node) if i in needed)
    sliced.graph.initializer.extend(v for v in original.graph.initializer if v.name in seen)
    sliced.graph.input.extend(v for v in original.graph.input if v.name in seen)
    output = sliced.graph.output.add()
    output.name = LATENT
    output.type.tensor_type.elem_type = 1
    for size in (1, 128):
        output.type.tensor_type.shape.dim.add().dim_value = size
    return sliced.SerializeToString()


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.add_session_config_entry('session.intra_op.allow_spinning', '0')
    options.log_severity_level = 3
    return ort.InferenceSession(model, sess_options=options, providers=['CPUExecutionProvider'])


class LayerNormProbe:
    def __init__(self, bundle):
        root = Path(bundle)
        manifest = json.loads((root / 'bundle.json').read_text())
        for name in ('policy.onnx', 'encoder.onnx'):
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest['files'][name]:
                raise ValueError(f'Bundle checksum mismatch: {name}')
        self.latent = session(latent_graph((root / 'policy.onnx').read_bytes()))
        self.encoder = session(str(root / 'encoder.onnx'))
        self.names = [x.name for x in self.latent.get_inputs()]
        if set(self.names) != set(VISUAL_INPUTS):
            raise ValueError(f'Unsupported visual latent dependencies: {self.names}')
        self.image_shape = tuple(self.encoder.get_inputs()[0].shape[1:])

    def compute(self, bgr, inputs):
        if bgr.shape != self.image_shape or bgr.dtype != np.uint8:
            raise ValueError('Probe image does not match the policy encoder')
        feed = {k: inputs[k] for k in self.names}
        baseline = self.latent.run(None, feed)[0]
        rows, cols = bgr.shape[0] // 16, bgr.shape[1] // 16
        fill = bgr.mean(axis=(0, 1)).astype(np.uint8)
        scores = []
        for patch in range(rows * cols):
            image = bgr.copy()
            y, x = divmod(patch, cols)
            image[y*16:(y+1)*16, x*16:(x+1)*16] = fill
            tokens, cls = self.encoder.run(None, {'image_bgr_hwc': image[None]})
            z = self.latent.run(None, dict(feed, kv_tokens__img_tokens=tokens, q_cls=cls))[0]
            scores.append(float(np.sqrt(np.mean((z - baseline)**2))))
        if not np.isfinite(scores).all():
            raise ValueError('Non-finite LayerNorm sensitivity')
        return scores


def _process(bundle, requests, results):
    try:
        os.nice(10)
        probe = LayerNormProbe(bundle)
        results.put(dict(status='ready'))
        while True:
            item = requests.get()
            if item is None:
                return
            start = time.monotonic()
            scores = probe.compute(item['bgr'], item['inputs'])
            image = io.BytesIO()
            Image.fromarray(item['bgr'][..., ::-1]).save(image, format='PNG')
            results.put(dict(status='ready', generation=item['generation'],
                             source_step=item['source_step'], time=item['time'],
                             width=item['bgr'].shape[1], height=item['bgr'].shape[0],
                             layernorm_sensitivity=scores, compute_ms=1000*(time.monotonic()-start),
                             image='data:image/png;base64,'+base64.b64encode(image.getvalue()).decode('ascii')))
    except Exception as exc:
        results.put(dict(status='error', error=f'{type(exc).__name__}: {exc}'))


class AsyncSensitivity:
    """At most one immutable frame in flight; no inference on the control thread."""
    def __init__(self, bundle):
        context = mp.get_context('spawn')
        self.requests, self.results = context.Queue(1), context.Queue(2)
        self.process = context.Process(target=_process, args=(str(bundle), self.requests, self.results),
                                       name='layernorm-view', daemon=True)
        self.generation = 0
        self.busy = True
        self.last_source = None
        self.next_submit = 0.
        self.packet = dict(status='loading', generation=0, image=None)
        self.process.start()

    def reset(self):
        self.generation += 1
        self.last_source = None
        self.next_submit = 0.
        self.packet = dict(status='waiting', generation=self.generation, image=None)

    def update(self, sim):
        while True:
            try:
                result = self.results.get_nowait()
            except queue.Empty:
                break
            self.busy = False
            if result['status'] == 'error':
                self.packet = dict(result, generation=self.generation, image=None)
            elif result.get('generation') == self.generation:
                self.packet = result
            elif self.packet['status'] == 'loading':
                self.packet = dict(status='waiting', generation=self.generation, image=None)
        if self.packet['status'] == 'error':
            return
        if not self.process.is_alive():
            self.packet = dict(status='error', generation=self.generation, image=None,
                               error='LayerNorm visualization process stopped')
            return
        now = time.monotonic()
        source = (self.generation, sim.vision['source_step'])
        if self.busy or sim.visual_inputs is None or source == self.last_source or now < self.next_submit:
            return
        item = dict(generation=self.generation, source_step=sim.vision['source_step'],
                    time=sim.vision['time'], bgr=sim.rgb[..., ::-1].copy(),
                    inputs={k: v.copy() for k, v in sim.visual_inputs.items()})
        try:
            self.requests.put_nowait(item)
        except queue.Full:
            return
        self.busy = True
        self.last_source = source
        self.next_submit = now + .5
        if not self.packet.get('image'):
            self.packet = dict(status='computing', generation=self.generation, image=None)

    def close(self):
        try:
            self.requests.put_nowait(None)
        except queue.Full:
            pass
        self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1)
        self.requests.cancel_join_thread()
        self.requests.close()
        self.results.close()
