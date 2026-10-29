"""Wrap verified upstream PerLoco bytes for the optional Python keyboard runtime.

This does not change the upstream model. The official C++ runtime remains the
recommended reference-motion baseline; attaching SONIC's online planner is a
separate experiment, not a claim of reproducing its published hardware result.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from .export import sha256, write_json
from .observations import JOINT_NAMES
from .runtime import session


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-root", required=True)
    p.add_argument("--template-bundle", required=True, help="Supplies physical limits and nominal camera only")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    source = Path(args.official_root)
    template = Path(args.template_bundle)
    model = source / "models/walk/policy.onnx"
    metadata = json.loads(model.with_suffix(".manifest.json").read_text())
    receipt = json.loads((source / "upstream_receipt.json").read_text())
    for name, record in receipt["files"].items():
        if sha256(source / name) != record["sha256"]:
            raise ValueError(f"Official artifact changed: {name}")
    if metadata["task_id"] not in ("Vibe-PerLoco-Grail-ImgFeat-Ext", "Vibe-PerLoco-Grail-AdaptSonic-ImgFeat-Ext"):
        raise ValueError("Only the released PerLoco GRAIL policy is supported")
    if metadata["action"]["joint_names"] != JOINT_NAMES or metadata["control"]["step_dt"] != .02:
        raise ValueError("Unsupported upstream actuation contract")
    manifest = copy.deepcopy(json.loads((template / "bundle.json").read_text()))
    for key in ("scale", "default_joint_pos", "stiffness", "damping"):
        a, b = np.asarray(manifest["robot"][key]), np.asarray(metadata["action"][key])
        if a.shape != (29,) or b.shape != (29,) or not np.allclose(a, b, atol=1e-6):
            raise ValueError(f"Upstream robot {key} differs; explicit adapter is required")
        manifest["robot"][key] = b.tolist()
    expected = {
        "tokenizer": ["g1_tokenizer"],
        "policy": ["base_ang_vel", "joint_pos", "joint_vel", "actions", "gravity_dir"],
        "augmentation": ["robot_root_lin_vel_cmd", "robot_root_ang_vel_cmd"],
        "kv_tokens__img_tokens": ["img_tokens"],
        "q_proprio": ["projected_gravity", "base_ang_vel", "joint_pos", "joint_vel"],
        "q_cls": ["img_cls"],
    }
    if {p["name"]: [t["name"] for t in p["terms"]] for p in metadata["inputs"]} != expected:
        raise ValueError("Upstream observation terms differ from the supported PerLoco contract")
    for port in metadata["inputs"]:
        for term in port["terms"]:
            history = 10 if port["name"] == "policy" else 0
            if term.get("history_length", 0) != history or not term.get("flatten_history_dim", True):
                raise ValueError("Unsupported upstream history layout")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(model, out / "policy.onnx")
    shutil.copy2(source / "encoders/theia-tiny.onnx", out / "encoder.onnx")
    shutil.copy2(template / "planner.onnx", out / "planner.onnx")
    shutil.copy2(model.with_suffix(".manifest.json"), out / "upstream_policy.manifest.json")
    policy = session(out / "policy.onnx")
    manifest.update(kind="tracking", checkpoint="https://huggingface.co/lkrajan/vibe/tree/v0.1.0/Vibe-PerLoco-Grail-ImgFeat-Ext",
                    checkpoint_sha256="bc024518742424fdb5a186bbdebb608219d4311452fe44724ce62e5cbc191eb7",
                    files={n: sha256(out / n) for n in ("policy.onnx", "encoder.onnx", "planner.onnx")},
                    inputs={x.name: x.shape for x in policy.get_inputs()},
                    policy_parity=None, upstream_receipt=receipt,
                    provenance="Author-released GRAIL ViBe policy, unchanged ONNX bytes",
                    standing_start_stop_validated=False, hardware_validated=False,
                    vision_revision="Author-released encoder, immutable artifact SHA-256 in files")
    write_json(out / "bundle.json", manifest)
    print(out.resolve())


if __name__ == "__main__":
    main()
