"""CPU-constructible actor and exact ONNX input template for this experiment."""

from __future__ import annotations

import copy

import torch
from tensordict import TensorDict


def observation_template(batch: int = 2):
    # Camera 112x63, stride-16 Theia-Tiny => 21 patches x 192 channels.
    return TensorDict(
        {
            "policy": torch.zeros(batch, 930),
            "tokenizer": torch.zeros(batch, 640),
            "augmentation": torch.zeros(batch, 6),
            "kv_tokens": {"img_tokens": torch.zeros(batch, 21, 192)},
            "q_task_cmd": torch.zeros(batch, 6),
            "q_proprio": torch.zeros(batch, 64),
            "q_cls": torch.zeros(batch, 192),
        },
        batch_size=[batch],
    )


def make_actor(settings, obs=None):
    from rsl_rl.utils import resolve_callable

    from .env_cfg import build_agent_cfg

    cfg = build_agent_cfg(settings)
    actor = copy.deepcopy(cfg.actor)
    cls = resolve_callable(actor.pop("class_name"))
    return cls(
        obs=observation_template() if obs is None else obs,
        obs_groups=cfg.obs_groups,
        obs_set="actor",
        output_dim=29,
        **actor,
    )
