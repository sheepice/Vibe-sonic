"""Pinned, explicitly frozen Theia; upstream RGB normalization and cache semantics."""

from __future__ import annotations

import torch
from vibe.core.mdp.observations import image_feature

THEIA_REVISION = "8ce99426c269617346c30112817f0e694e3cde2d"


class FrozenTheiaFeature(image_feature):
    @classmethod
    def _load(cls, backend, model_name, dev, dtype):
        if backend != "theia" or model_name != "theia-tiny-patch16-224-cddsv":
            raise ValueError("This experiment pins the ViBe Theia-Tiny backbone")
        key = (model_name, THEIA_REVISION, str(dev), str(dtype))
        if key not in cls._MODELS:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                f"theaiinstitute/{model_name}", revision=THEIA_REVISION, trust_remote_code=True
            )
            cls._MODELS[key] = model.eval().requires_grad_(False).to(dev, dtype)
        return cls._MODELS[key]


def main():
    """Download/validate the encoder on CPU; never allocate a CUDA context."""
    from pathlib import Path

    from .catalog import write_json

    torch.set_num_threads(2)
    model = FrozenTheiaFeature._load("theia", "theia-tiny-patch16-224-cddsv", "cpu", torch.float32)
    with torch.inference_mode():
        values = model.backbone.model(
            pixel_values=torch.zeros(1, 3, 63, 112), interpolate_pos_encoding=True
        ).last_hidden_state
    assert values.shape == (1, 22, 192) and torch.isfinite(values).all()
    result = {
        "revision": THEIA_REVISION,
        "shape": list(values.shape),
        "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
    }
    write_json(Path("records/vision_cpu.json"), result)
    print(result)


if __name__ == "__main__":
    main()
