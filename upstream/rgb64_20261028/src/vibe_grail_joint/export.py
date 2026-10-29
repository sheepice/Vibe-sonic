"""CPU ONNX export with numerical parity checks; does not control a robot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .catalog import write_json
from .policy import make_actor


def export_checked(actor, destination: Path, tolerance: float = 1e-5):
    import onnxruntime as ort

    module = actor.cpu().eval().as_onnx(with_attn=True)
    module.eval()
    dummy = module.get_dummy_inputs()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".unchecked.onnx")
    torch.onnx.export(
        module,
        dummy,
        str(temporary),
        input_names=module.input_names,
        output_names=module.output_names,
        opset_version=18,
        dynamo=False,
    )
    session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
    maximum = 0.0
    generator = torch.Generator().manual_seed(42)
    with torch.inference_mode():
        for trial in range(12):
            values = (
                dummy if trial == 0 else tuple(torch.randn(x.shape, generator=generator) * 0.2 for x in dummy)
            )
            expected = module(*values)
            actual = session.run(
                None, {name: value.numpy() for name, value in zip(module.input_names, values)}
            )
            for first, second in zip(expected, actual):
                maximum = max(maximum, float(np.abs(first.numpy() - second).max()))
    if maximum > tolerance:
        raise RuntimeError(f"ONNX parity failed: {maximum} > {tolerance}; export remains .unchecked.onnx")
    temporary.replace(destination)
    return {
        "input_names": module.input_names,
        "output_names": module.output_names,
        "input_shapes": [list(value.shape) for value in dummy],
        "synthetic_cpu_parity_max_abs_error": maximum,
        "closed_loop_validated": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", help="Training directory containing observation_contract.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    run_dir = Path(args.run_dir) if args.run_dir else Path(args.checkpoint).resolve().parent
    status = json.loads((run_dir / "rank_0_status.json").read_text())
    actor = make_actor(status["config"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    destination = Path(args.output)
    manifest = export_checked(actor, destination)
    manifest.update(json.loads((run_dir / "observation_contract.json").read_text()))
    manifest.update(
        checkpoint=str(Path(args.checkpoint).resolve()),
        training_iteration=checkpoint["iter"],
        smoke_only=status["smoke_only"],
        encoder_outside_policy_graph=True,
        vision_revision=status["config"]["vision_revision"],
        deployment_validated=False,
    )
    write_json(destination.with_suffix(".manifest.json"), manifest)
    print(manifest)


if __name__ == "__main__":
    main()
