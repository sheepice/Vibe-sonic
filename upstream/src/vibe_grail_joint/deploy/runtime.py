"""Strict ONNX runtime: no torch imports and no implicit CUDA-to-CPU fallback."""
import json
from pathlib import Path
import time

import numpy as np

from .export import sha256


def session(path, provider="cpu"):
    import onnxruntime as ort
    name = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[provider]
    if name not in ort.get_available_providers():
        raise RuntimeError(f"Requested {name} is unavailable; installed={ort.get_available_providers()}")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    sess = ort.InferenceSession(str(path), sess_options=options, providers=[name])
    if sess.get_providers()[0] != name:
        raise RuntimeError(f"Provider failed to initialize: {name}")
    return sess


class Runtime:
    def __init__(self, bundle, provider="cpu"):
        self.root = Path(bundle)
        self.manifest = json.loads((self.root / "bundle.json").read_text())
        if self.manifest["schema"] != "vibe_grail_joint.sim2real.v1":
            raise ValueError("Unsupported bundle schema")
        for name, checksum in self.manifest["files"].items():
            if sha256(self.root / name) != checksum:
                raise ValueError(f"Bundle checksum mismatch: {name}")
        self.policy = session(self.root / "policy.onnx", provider)
        self.encoder = session(self.root / "encoder.onnx", provider)
        self.inputs = {x.name: x.shape for x in self.policy.get_inputs()}
        if self.inputs != self.manifest["inputs"]:
            raise ValueError("Policy input shapes differ from bundle")
        self.velocity = self.manifest["kind"] == "velocity"
        if ("velocity_command" in self.inputs) != self.velocity:
            raise ValueError("Missing or unexpected velocity branch")
        self.contract = self.manifest["robot"]
        self.default = np.asarray(self.contract["default_joint_pos"], np.float32)
        self.scale = np.asarray(self.contract["scale"], np.float32)

    def warmup(self, steps=5):
        """Exercise first-use kernels before a controller can be engaged.

        These artificial inputs never reach the robot or observation history.
        ONNX graphs are stateless; recurrent state is an explicit input.
        """
        before = time.monotonic()
        feed = {name: np.zeros(shape, np.float32) for name, shape in self.inputs.items()}
        for _ in range(steps):
            patches, cls = self.encode(np.zeros((63, 112, 3), np.uint8))
            feed["kv_tokens__img_tokens"], feed["q_cls"] = patches[None], cls[None]
            self.act(feed)
        return time.monotonic() - before

    def encode(self, bgr):
        if bgr.dtype != np.uint8 or bgr.shape != (63, 112, 3):
            raise ValueError("Vision input must be BGR uint8 HWC 63x112x3")
        patches, cls = self.encoder.run(None, {"image_bgr_hwc": np.ascontiguousarray(bgr[None])})
        if not np.isfinite(patches).all() or not np.isfinite(cls).all():
            raise ValueError("Non-finite visual features")
        return patches[0], cls[0]

    def act(self, observations):
        feed = {k: observations[k] for k in self.inputs}
        for k, shape in self.inputs.items():
            if list(feed[k].shape) != shape or feed[k].dtype != np.float32 or not np.isfinite(feed[k]).all():
                raise ValueError(f"Invalid policy input {k}")
        start = time.monotonic()
        action = self.policy.run(["actions"], feed)[0][0]
        if action.shape != (29,) or not np.isfinite(action).all():
            raise ValueError("Invalid joint action")
        return action, self.default + self.scale * action, time.monotonic() - start
