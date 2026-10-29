"""Explicit, single-reward refinement resumed from the completed posture run.

This module does not start a runner or restore only actor weights. Its validated
``latest_checkpoint`` must be passed to the existing launcher's ``--resume`` so
the actor, critic, optimizer, absolute curriculum count and environment counter
are restored together. Physics worlds and planner streams still start new
episodes, as with the existing stage-to-stage resume.

The optional settings object has exactly four required keys: velocity_weight,
resume_completed_updates (3000), initial_checkpoint_sha256, frozen_sonic_sha256.
No settings object means the original task behavior. Once explicitly enabled,
missing/malformed resume evidence is an error, never an actor-only warm start.
"""

from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path


_KEYS = {"velocity_weight", "resume_completed_updates",
         "initial_checkpoint_sha256", "frozen_sonic_sha256"}
_ACTOR = "vibe_grail_joint.velocity.policy.CommandSonicActor"
_ALGORITHM = "vibe_grail_joint.velocity.posture_policy.UpperBodyPreservationPPO"


def _options(settings):
    value = settings.get("speed_refinement")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != _KEYS:
        raise ValueError(f"speed_refinement requires exactly these keys: {sorted(_KEYS)}")
    weight = value["velocity_weight"]
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight <= 0):
        raise ValueError("speed_refinement.velocity_weight must be finite and positive")
    if type(value["resume_completed_updates"]) is not int or value["resume_completed_updates"] != 3000:
        raise ValueError("This refinement must resume from exactly 3000 completed updates")
    for key in ("initial_checkpoint_sha256", "frozen_sonic_sha256"):
        digest = value[key]
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in digest)):
            raise ValueError(f"speed_refinement.{key} must be a pinned SHA256")
    return value


def configure_speed_refinement(cfg, settings):
    """Change only the existing command_velocity term's scalar weight."""
    options = _options(settings)
    if options is None:
        return cfg
    from . import mdp

    term = cfg.rewards.get("command_velocity")
    weight = float(options["velocity_weight"])
    if (term is None or term.func is not mdp.velocity_tracking or term.params
            or term.weight not in (2.0, weight)):
        raise ValueError("Expected unchanged default command_velocity function/parameters/weight")
    # Preserve object identity, role mask, Gaussian width and every other term.
    term.weight = weight
    return cfg


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _frozen_digest(actor):
    # Exactly train_stairs.frozen_digest: registration order, names, tensor bytes.
    digest = hashlib.sha256()
    for name, value in actor.named_parameters():
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _inspect_model(payload, agent):
    """Strict actor/critic and complete Adam restoration on CPU, without steps."""
    import torch
    from tensordict import TensorDict
    from rsl_rl.models.mlp_model import MLPModel
    from ..policy import observation_template
    from .policy import CommandSonicActor

    spec = copy.deepcopy(agent["actor"])
    if spec.pop("class_name", None) != _ACTOR:
        raise ValueError("Resume actor must be CommandSonicActor")
    if (spec.get("freeze_base") is not True or spec.get("adapt_encoder") is not False
            or spec.get("adapt_decoder") is not True):
        raise ValueError("Resume requires the unchanged frozen SONIC encoder/decoder contract")
    spec["base_checkpoint"] = None
    obs = observation_template(1)
    obs["velocity_command"] = torch.tensor([[0., 0., 0., 0., 0., 0., 1., 0.]])
    # Constructor initialization must not perturb the caller's CPU RNG.
    critic_spec = agent.get("critic", {})
    if (critic_spec.get("class_name") != "MLPModel" or critic_spec.get("cnn_cfg") is not None
            or critic_spec.get("rnn_type") is not None):
        raise ValueError("Expected the unchanged feed-forward MLP critic")
    critic_args = {key: critic_spec[key] for key in
                   ("hidden_dims", "activation", "obs_normalization", "distribution_cfg")
                   if key in critic_spec}
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        actor = CommandSonicActor(obs=obs, obs_groups={"actor": ["policy"]},
                                  obs_set="actor", output_dim=29, **spec)
        actor.load_state_dict(payload["actor_state_dict"], strict=True)
        critic = MLPModel(obs=TensorDict({"critic": torch.zeros(1, 999)}, batch_size=[1]),
                          obs_groups={"critic": ["critic"]}, obs_set="critic", output_dim=1,
                          **critic_args)
        critic.load_state_dict(payload["critic_state_dict"], strict=True)
    if any(not bool(torch.isfinite(v).all()) for v in actor.state_dict().values()):
        raise ValueError("Checkpoint actor contains nonfinite values")
    parameters = [*actor.parameters(), *critic.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=agent["algorithm"]["learning_rate"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for parameter in parameters:
        state = optimizer.state.get(parameter, {})
        if parameter.requires_grad and not state:
            raise ValueError("Full resume is missing trained-parameter Adam state")
        if state and any(state[key].shape != parameter.shape for key in ("exp_avg", "exp_avg_sq")):
            raise ValueError("Adam moment shape differs from the restored model parameter")
    return {
        "frozen_sonic_sha256": _frozen_digest(actor),
        "actor_parameters": sum(p.numel() for p in actor.parameters()),
        "trainable_actor_parameters": sum(p.numel() for p in actor.parameters() if p.requires_grad),
        "strict_actor_load": True,
        "strict_critic_load": True,
        "optimizer_state_loaded_on_cpu": True,
    }


def _verify_optimizer(payload, learning_rate):
    import torch

    critic, optimizer = payload.get("critic_state_dict"), payload.get("optimizer_state_dict")
    if not isinstance(critic, dict) or not critic:
        raise ValueError("Full resume requires nonempty critic_state_dict")
    if any(not isinstance(v, torch.Tensor) or not bool(torch.isfinite(v).all()) for v in critic.values()):
        raise ValueError("Critic state must contain finite tensors")
    if (not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict)
            or not optimizer["state"] or not optimizer.get("param_groups")):
        raise ValueError("Full resume requires populated optimizer state and parameter groups")
    groups = optimizer["param_groups"]
    if not isinstance(groups, list) or any(not isinstance(g, dict) for g in groups):
        raise ValueError("Malformed optimizer parameter groups")
    parameter_ids = [p for g in groups for p in g.get("params", [])]
    if not parameter_ids or len(set(parameter_ids)) != len(parameter_ids):
        raise ValueError("Optimizer parameter IDs must be nonempty and unique")
    if any(g.get("lr") != learning_rate for g in groups):
        raise ValueError("Optimizer learning rate differs from unchanged fixed-rate settings")
    for parameter, state in optimizer["state"].items():
        if parameter not in parameter_ids or not isinstance(state, dict):
            raise ValueError("Optimizer state does not belong to its parameter groups")
        step = state.get("step")
        if (not isinstance(step, (int, float, torch.Tensor))
                or (isinstance(step, torch.Tensor) and step.numel() != 1)
                or not math.isfinite(float(step)) or float(step) <= 0):
            raise ValueError("Optimizer state must retain a positive finite Adam step")
        moments = [state.get(key) for key in ("exp_avg", "exp_avg_sq")]
        if (any(not isinstance(x, torch.Tensor) or not bool(torch.isfinite(x).all()) for x in moments)
                or moments[0].shape != moments[1].shape):
            raise ValueError("Optimizer state must retain matching finite Adam moments")
    return len(optimizer["state"])


def initial_resume_state(settings, checkpoint, stages):
    """Validate a full 3000-update CPU checkpoint before creating a new run.

    Returns supervisor initial state plus JSON provenance. This function does
    not mutate settings, checkpoint bytes, optimizer state or a running process.
    The supervisor must pass latest_checkpoint via --resume on its first stage.
    """
    options = _options(settings)
    if options is None:
        if checkpoint is not None:
            raise ValueError("Explicit --resume-checkpoint requires speed_refinement settings")
        return {"completed_updates": 0, "latest_checkpoint": None}
    if checkpoint is None:
        raise ValueError("speed_refinement requires --resume-checkpoint; actor-only warm start is forbidden")
    completed = options["resume_completed_updates"]
    target = settings.get("iterations")
    if type(target) is not int or target <= completed:
        raise ValueError("Refinement iterations must exceed 3000")
    if (not isinstance(stages, (list, tuple)) or not stages
            or any(type(x) is not int for x in stages)
            or list(stages) != sorted(set(stages)) or stages[0] <= completed or stages[-1] != target):
        raise ValueError("Resume stages must increase strictly beyond 3000 and end at iterations")
    path = Path(checkpoint).expanduser().resolve(strict=True)
    initial = settings.get("initial_checkpoint")
    if not isinstance(initial, str) or Path(initial).expanduser().resolve(strict=True) != path:
        raise ValueError("settings.initial_checkpoint must identify the full resume checkpoint")

    import torch

    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != options["initial_checkpoint_sha256"].lower():
        raise ValueError("Initial resume checkpoint SHA256 mismatch")
    payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=False)
    del content
    if not isinstance(payload, dict) or type(payload.get("iter")) is not int or payload["iter"] + 1 != completed:
        raise ValueError("Checkpoint iter+1 must equal 3000")
    infos = payload.get("infos", {})
    metadata = infos.get("velocity_training", {}) if isinstance(infos, dict) else {}
    if (not isinstance(metadata, dict) or metadata.get("version") != 2
            or metadata.get("completed_policy_updates") != completed
            or metadata.get("curriculum_update") != completed):
        raise ValueError("Checkpoint lacks matching completed-update and curriculum metadata")
    expected_frozen = options["frozen_sonic_sha256"].lower()
    if metadata.get("frozen_sonic_sha256") != expected_frozen:
        raise ValueError("Checkpoint frozen SONIC metadata does not match pinned digest")
    if metadata.get("online_fraction") != .6 or metadata.get("replay_fraction") != .4:
        raise ValueError("Checkpoint must retain the existing 60/20/20 role allocation")
    previous = metadata.get("settings")
    if (not isinstance(previous, dict) or previous.get("iterations") != completed
            or "speed_refinement" in previous):
        raise ValueError("Expected an original completed 3000-update posture checkpoint")
    allowed = {"iterations", "initial_checkpoint", "speed_refinement"}
    old_rest = {k: v for k, v in previous.items() if k not in allowed}
    new_rest = {k: v for k, v in settings.items() if k not in allowed}
    if _canonical(old_rest) != _canonical(new_rest):
        changed = sorted(k for k in old_rest.keys() | new_rest.keys()
                         if _canonical(old_rest.get(k)) != _canonical(new_rest.get(k))
                         or (k in old_rest) != (k in new_rest))
        raise ValueError(f"Only velocity reward refinement is permitted; incompatible settings: {changed}")
    if settings.get("posture_revision") is None or settings.get("learning_rate_schedule") != "fixed":
        raise ValueError("Expected original posture revision and fixed learning-rate contract")
    agent_path = path.parent / "agent_cfg.json"
    agent_bytes = agent_path.read_bytes()
    agent = json.loads(agent_bytes)
    if agent.get("algorithm", {}).get("class_name") != _ALGORITHM:
        raise ValueError("The existing UpperBodyPreservationPPO must remain unchanged")
    steps = agent.get("num_steps_per_env")
    if type(steps) is not int or steps <= 0:
        raise ValueError("Missing valid original rollout length")
    env_state = infos.get("env_state")
    if not isinstance(env_state, dict) or env_state.get("common_step_counter") != completed * steps:
        raise ValueError("Full resume requires the existing absolute environment step counter")
    optimizer_count = _verify_optimizer(payload, settings["learning_rate"])
    model = _inspect_model(payload, agent)
    if model["frozen_sonic_sha256"] != expected_frozen:
        raise ValueError("Actual checkpoint frozen SONIC tensors differ from pinned digest")
    return {
        "completed_updates": completed,
        "latest_checkpoint": str(path),
        "resume_provenance": {
            "checkpoint_sha256": digest,
            "checkpoint_iter": payload["iter"],
            "completed_policy_updates": completed,
            "curriculum_update": metadata["curriculum_update"],
            "common_step_counter": infos["env_state"]["common_step_counter"],
            "optimizer_state_entries": optimizer_count,
            "agent_cfg_path": str(agent_path),
            "agent_cfg_sha256": hashlib.sha256(agent_bytes).hexdigest(),
            "original_settings_sha256": hashlib.sha256(_canonical(previous).encode()).hexdigest(),
            "settings_changes_allowed": sorted(allowed),
            "reward_change": {"term": "command_velocity", "old_weight": 2.0,
                              "new_weight": float(options["velocity_weight"])},
            "restore_via": "Existing launcher --resume: actor, critic, optimizer, absolute curriculum and env counter",
            "physics_resume_limit": "Physics worlds and planner streams start new episodes; no exact trajectory continuation.",
            **model,
        },
    }
