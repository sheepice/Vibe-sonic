"""Export a checked, command-aware deployment bundle without changing training code."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def make_export_graph(actor, velocity):
    from rsl_rl.models.extractor_model import _OnnxExtractorSonicAdapterModel
    from rsl_rl.models._onnx_export import Port

    class CommandGraph(_OnnxExtractorSonicAdapterModel):
        def _adapter_ports(self, model):
            ports = super()._adapter_ports(model)
            self.command_encoder = copy.deepcopy(model.command_encoder)
            return ports + [Port("velocity_command", (8,), ("velocity_command",))]

        def _adapter_latent(self, inputs):
            return super()._adapter_latent(inputs[:-1]) + self.command_encoder(inputs[-1])

    return (CommandGraph(actor, with_attn=True) if velocity else actor.as_onnx(with_attn=True)).cpu().eval()


def load_actor(checkpoint, bank, velocity):
    import mjlab  # entry point discovery must precede ORCS imports
    import torch
    from rsl_rl.utils import resolve_callable
    from ..policy import observation_template
    from ..env_cfg import build_agent_cfg

    cfg = build_agent_cfg(bank["config"])
    kwargs = copy.deepcopy(cfg.actor)
    name = kwargs.pop("class_name")
    if velocity:
        name = "vibe_grail_joint.velocity.policy.CommandSonicActor"
    obs = observation_template(batch=1)
    if velocity:
        obs["velocity_command"] = torch.zeros(1, 8)
    actor = resolve_callable(name)(obs=obs, obs_groups=cfg.obs_groups, obs_set="actor", output_dim=29, **kwargs)
    data = torch.load(checkpoint, weights_only=False, map_location="cpu")
    actor.load_state_dict(data["actor_state_dict"], strict=True)
    return actor.cpu().eval(), obs


def robot_contract(bank_path):
    from ..env_cfg import build_env_cfg
    from .observations import JOINT_NAMES
    cfg = build_env_cfg(str(bank_path), num_envs=1, play=True)
    robot = cfg.scene.entities["robot"]
    model = robot.spec_fn().compile()
    def resolve(mapping, name, default=None):
        matches = [v for k, v in mapping.items() if re.fullmatch(k, name)]
        if len(matches) > 1 or (not matches and default is None):
            raise ValueError(f"Ambiguous/missing contract for {name}")
        return matches[0] if matches else default
    actuators = {p: a for a in robot.articulation.actuators for p in a.target_names_expr}
    scales = cfg.actions["joint_pos"].scale
    return {
        "joint_names": JOINT_NAMES,
        "default_joint_pos": [resolve(robot.init_state.joint_pos, n, 0.) for n in JOINT_NAMES],
        "scale": [resolve(scales, n) for n in JOINT_NAMES],
        "stiffness": [resolve(actuators, n).stiffness for n in JOINT_NAMES],
        "damping": [resolve(actuators, n).damping for n in JOINT_NAMES],
        "effort_limit": [resolve(actuators, n).effort_limit for n in JOINT_NAMES],
        "joint_limits": [model.joint(n).range.tolist() for n in JOINT_NAMES],
        "root_height": float(robot.init_state.pos[2]),
        "control_dt": float(cfg.sim.mujoco.timestep * cfg.decimation),
        "lowcmd_mode_pr": 0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--kind", choices=("velocity", "tracking"), required=True)
    p.add_argument("--encoder", required=True, help="Pinned, already parity-checked Theia ONNX")
    p.add_argument("--planner", required=True)
    p.add_argument("--output", required=True, help="New directory; existing bundles are never overwritten")
    args = p.parse_args()
    import torch
    import onnx
    import onnxruntime as ort
    from .runtime import session

    torch.set_num_threads(2)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    bank = json.loads((Path(args.bank) / "bank.json").read_text())
    actor, template = load_actor(args.checkpoint, bank, args.kind == "velocity")
    graph = make_export_graph(actor, args.kind == "velocity")
    pending = out / "policy.unchecked.onnx"
    torch.onnx.export(graph, graph.get_dummy_inputs(), str(pending), opset_version=18,
                      dynamo=False, input_names=graph.input_names, output_names=graph.output_names)
    onnx.checker.check_model(str(pending))
    sess = session(pending, "cpu")
    generator = torch.Generator().manual_seed(20261018)
    maximum = 0.
    for i in range(32):
        obs = template.clone()
        for key in obs.keys(True, True):
            obs[key] = torch.randn(obs[key].shape, generator=generator) * .3
        if args.kind == "velocity":
            obs["velocity_command"] = torch.tensor([[i % 5 * .15, 0, 0, 0, 0, 1, 0, 0.]])
        inputs = {n: (obs["kv_tokens", "img_tokens"] if n == "kv_tokens__img_tokens" else obs[n]).numpy()
                  for n in graph.input_names}
        with torch.no_grad():
            expected = actor(obs).numpy()
        got = sess.run(["actions"], inputs)[0]
        maximum = max(maximum, float(abs(expected - got).max()))
    if maximum > 1e-5:
        raise RuntimeError(f"CPU actor parity failed: {maximum}; unchecked artifact retained for diagnosis")
    pending.rename(out / "policy.onnx")
    shutil.copy2(args.encoder, out / "encoder.onnx")
    shutil.copy2(args.planner, out / "planner.onnx")
    enc = session(out / "encoder.onnx", "cpu")
    if [(x.name, x.shape, x.type) for x in enc.get_inputs()] != [("image_bgr_hwc", [1, 63, 112, 3], "tensor(uint8)")]:
        raise ValueError("Encoder does not match the training BGR/112x63 contract")
    encoded = enc.run(None, {"image_bgr_hwc": np.zeros((1, 63, 112, 3), np.uint8)})
    if [x.shape for x in encoded] != [(1, 21, 192), (1, 192)]:
        raise ValueError("Wrong Theia output shapes")
    contract = robot_contract(args.bank)
    if bank["joint_names"] != contract["joint_names"]:
        raise ValueError("Bank joint order differs from SDK PR motor order")
    manifest = {
        "schema": "vibe_grail_joint.sim2real.v1", "kind": args.kind,
        "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha256(args.checkpoint),
        "files": {n: sha256(out / n) for n in ("policy.onnx", "encoder.onnx", "planner.onnx")},
        "robot": contract, "camera": bank["config"]["camera"],
        "vision_revision": bank["config"]["vision_revision"],
        "inputs": {x.name: x.shape for x in sess.get_inputs()},
        "policy_parity": {"cases": 32, "maximum_absolute_error": maximum, "tolerance": 1e-5},
        "reference": {"frames": 10, "stride_s": .1, "anchor": "pelvis", "quaternion": "wxyz"},
        "history": {"length": 10, "order": "oldest_to_newest", "layout": "term_major", "initialization": "repeat_first"},
        "requires_reference_motion": True, "depth_input": False,
        "hardware_validated": False, "standing_start_stop_validated": False,
        "runtime": {"python": "3.11", "onnxruntime": ort.__version__, "onnx": onnx.__version__},
    }
    write_json(out / "bundle.json", manifest)
    print(json.dumps({"bundle": str(out), "actor_parity_max_abs": maximum}, indent=2))


if __name__ == "__main__":
    main()
