"""Optional paired-replay ViBe action prior; not enabled by existing configs.

Integration, only after an explicit experiment decision: select
``PairedViBePreservationPPO`` and supply the actual PD ``action_scales`` and
``action_joint_names`` already required by UpperBodyPreservationPPO, plus
``paired_teacher_checkpoint``, its mandatory ``paired_teacher_sha256``, and an
independent JSON copy of the ORIGINAL actor constructor configuration as
``paired_teacher_actor_config``. Snapshot the actor config before the runner
mutates nested distribution settings. No task rewards or role ratios change.

The teacher is a fresh CommandSonicActor, not a deepcopy of a live model with
non-leaf Gaussian/cached tensors. Its observation template comes from the
student's original export_shapes (including tokenizer before rollout caching).
The full checkpoint is read once into a SHA-verified byte snapshot and loaded
strictly. Its external base_checkpoint is deliberately unused: the full actor
checkpoint supplies every persistent parameter. Frozen encoder/decoder/FSQ and
non-state architecture settings must match the student exactly.

Only role 6 receives 0.5 * mean(((student - teacher) * PD_scale / std_rad)^2),
over all 29 joints. This adds no direct online leg penalty. The inherited flat
and online priors remain active. Teacher weights never enter the student,
optimizer or deployment checkpoint; only provenance metadata is saved. Resume
reconstructs the same SHA-pinned teacher from its external file. Existing paired
checkpoints must also match coefficient, std, sample cap and physical PD scales;
only a genuinely absent paired metadata field denotes legacy initialization.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
from pathlib import Path

import torch
from tensordict import TensorDict

from .policy import CommandSonicActor
from .posture_policy import SONIC_JOINT_ORDER, UpperBodyPreservationPPO


_CLASS_NAME = "vibe_grail_joint.velocity.policy.CommandSonicActor"
_GROUPS = {"full": slice(0, 29), "legs": slice(0, 12),
           "waist": slice(12, 15), "arms": slice(15, 29)}


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _template(shapes):
    """Reconstruct nested vector/token groups without retaining live observations."""
    return TensorDict({key: _template(shape) if isinstance(shape, dict)
                       else torch.zeros((1, *shape), dtype=torch.float32)
                       for key, shape in shapes.items()}, batch_size=[1])


def _actor_contract(actor):
    # These behavior settings are not all represented by state_dict; e.g. an
    # incorrect LoRA alpha can strict-load yet produce the wrong teacher action.
    attributes = ("scale", "low_rank", "eps", "until", "std_type", "std_range",
                  "token_terms", "query_groups", "input_groups", "num_heads",
                  "num_learned_queries", "attn_dim", "latent_dim", "token_dim")
    modules = {
        name: {"class": f"{type(module).__module__}.{type(module).__qualname__}",
               "repr": module.extra_repr(),
               **{key: getattr(module, key) for key in attributes if hasattr(module, key)}}
        for name, module in actor.named_modules()
    }
    return {
        "modules": modules, "export_shapes": actor.export_shapes,
        "obs_groups": actor.obs_groups, "tokenizer_obs_group": actor.tokenizer_obs_group,
        "adapter_obs_groups": actor.adapter_obs_groups,
        "num_tokens": actor.num_tokens, "token_dim": actor.token_dim,
        "cache_tokens": actor.cache_tokens, "adapt_encoder": actor.adapt_encoder,
        "adapt_decoder": actor.adapt_decoder, "freeze_base": actor.freeze_base,
    }


def _base_state(actor):
    if (not isinstance(actor, CommandSonicActor) or not actor.freeze_base
            or actor.adapt_encoder or not actor.adapt_decoder):
        raise ValueError("Paired teacher requires decoder-adapted CommandSonicActor with frozen SONIC encoder")
    state = {}
    for prefix, module in (("encoder", actor.encoder), ("decoder.base", actor.decoder.base)):
        if any(parameter.requires_grad for parameter in module.parameters()):
            raise ValueError(f"SONIC {prefix} must remain frozen")
        state.update({f"{prefix}.{key}": value for key, value in module.state_dict().items()})
    state["fsq_levels"] = actor.fsq_levels
    return state


def verify_frozen_base(student, teacher):
    """Bit-exact frozen basis check; includes nonpersistent FSQ levels."""
    left, right = _base_state(student), _base_state(teacher)
    if left.keys() != right.keys():
        raise ValueError("Teacher and student SONIC frozen state keys differ")
    digest = hashlib.sha256()
    for name in sorted(left):
        a, b = left[name].detach().cpu().contiguous(), right[name].detach().cpu().contiguous()
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise ValueError(f"Teacher and student SONIC frozen basis differs: {name}")
        digest.update(f"{name}|{a.dtype}|{tuple(a.shape)}".encode())
        digest.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_frozen_teacher(student, actor_config, checkpoint_path, expected_sha256, device):
    """Fresh construction is safe even after student stochastic forward/backward."""
    _base_state(student)
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in expected_sha256)):
        raise ValueError("paired_teacher_sha256 must be a pinned 64-digit SHA256")
    # JSON roundtrip accepts constructor configuration, never a live nn.Module.
    config = json.loads(json.dumps(actor_config, allow_nan=False))
    if not isinstance(config, dict):
        raise TypeError("paired_teacher_actor_config must be a JSON object")
    if config.pop("class_name", _CLASS_NAME) != _CLASS_NAME:
        raise ValueError("Teacher constructor must be CommandSonicActor")
    if any(key in config for key in ("obs", "obs_groups", "obs_set", "output_dim", "extractors")):
        raise ValueError("Teacher config must contain constructor settings, not live observations/modules")
    config["base_checkpoint"] = None
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    payload_bytes = path.read_bytes()
    actual_sha = hashlib.sha256(payload_bytes).hexdigest()
    if actual_sha != expected_sha256.lower():
        raise ValueError("Paired teacher checkpoint SHA256 mismatch")
    checkpoint = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=False)
    del payload_bytes
    if not isinstance(checkpoint, dict) or "actor_state_dict" not in checkpoint:
        raise ValueError("Paired teacher checkpoint must contain actor_state_dict")
    # Construction consumes CPU random numbers. Preserve training RNG without
    # creating or saving any CUDA context merely to construct the CPU template.
    with torch.random.fork_rng(devices=[]):
        teacher = CommandSonicActor(
            obs=_template(student.export_shapes), obs_groups={"actor": list(student.obs_groups)},
            obs_set="actor", output_dim=29, **copy.deepcopy(config))
    if _json_digest(_actor_contract(student)) != _json_digest(_actor_contract(teacher)):
        raise ValueError("Teacher actor behavior/observation contract differs from student")
    teacher.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    base_sha = verify_frozen_base(student, teacher)
    teacher.requires_grad_(False).eval().to(device)
    metadata = {
        "schema_version": 1, "checkpoint": str(path), "checkpoint_sha256": actual_sha,
        "checkpoint_iteration": checkpoint.get("iter"), "frozen_sonic_sha256": base_sha,
        "actor_config_sha256": _json_digest(config),
        "actor_contract_sha256": _json_digest(_actor_contract(teacher)),
        "strict_load": True, "teacher_parameters": sum(p.numel() for p in teacher.parameters()),
    }
    return teacher, metadata


class PairedViBePreservationPPO(UpperBodyPreservationPPO):
    """Existing priors plus a separate, immutable ViBe teacher for paired replay."""

    def __init__(self, *args, paired_teacher_checkpoint, paired_teacher_sha256,
                 paired_teacher_actor_config, paired_preservation_coef=0.02,
                 paired_preservation_std_rad=0.2, paired_preservation_samples=128, **kwargs):
        if not math.isfinite(paired_preservation_coef) or paired_preservation_coef < 0:
            raise ValueError("paired_preservation_coef must be finite and nonnegative")
        if not math.isfinite(paired_preservation_std_rad) or paired_preservation_std_rad <= 0:
            raise ValueError("paired_preservation_std_rad must be finite and positive")
        if (isinstance(paired_preservation_samples, bool) or not isinstance(paired_preservation_samples, int)
                or paired_preservation_samples < 1):
            raise ValueError("paired_preservation_samples must be a positive integer")
        super().__init__(*args, **kwargs)
        self.paired_preservation_coef = float(paired_preservation_coef)
        self.paired_preservation_std_rad = float(paired_preservation_std_rad)
        self.paired_preservation_samples = paired_preservation_samples
        self._paired_teacher, self._paired_provenance = build_frozen_teacher(
            self._raw_actor, paired_teacher_actor_config, paired_teacher_checkpoint,
            paired_teacher_sha256, self.device)
        # PPO is not an nn.Module: this independent teacher is not registered in
        # the actor or its optimizer. Assert that invariant instead of assuming it.
        teacher_ids = {id(parameter) for parameter in self._paired_teacher.parameters()}
        if any(id(p) in teacher_ids for group in self.optimizer.param_groups for p in group["params"]):
            raise RuntimeError("Paired teacher accidentally entered the student optimizer")
        self._paired_basis_verified_after_load = False
        self._reset_paired_statistics()

    @property
    def paired_teacher_provenance(self):
        return copy.deepcopy(self._paired_provenance)

    def _reset_paired_statistics(self):
        self._paired_count = self._paired_batch_count = 0
        self._paired_weighted_loss_sum = 0.0
        self._paired_mse_sum = {group: 0.0 for group in _GROUPS}
        self._paired_abs_sum = {group: 0.0 for group in _GROUPS}

    def _extra_backward(self):
        actor = self._raw_actor
        obs = actor._latest_update_obs
        if obs is None or len(obs) == 0:
            raise RuntimeError("Paired prior requires a nonempty current PPO minibatch")
        roles = obs["velocity_command"]
        if roles.ndim != 2 or roles.shape[-1] != 8:
            raise ValueError("Paired prior requires the 8-D role-coded velocity command")
        if not self._paired_basis_verified_after_load:
            # Runner warm-start can directly load actor AFTER algorithm creation.
            # Check once again at first use, after that external load has happened.
            verify_frozen_base(actor, self._paired_teacher)
            self._paired_basis_verified_after_load = True
        # Preserve cache/debug references as well as the PPO Gaussian graph.
        # No deepcopy: live distribution means can be non-leaf tensors.
        from rsl_rl.modules import cross_attention
        latest_attention = cross_attention.LATEST_ATTENTION
        tokens = actor._last_tokens
        distribution = actor.distribution
        gaussian = getattr(distribution, "_distribution", None)
        taps = [(extractor, extractor.last_attn, extractor.last_z)
                for extractor in actor.extractors.values() if hasattr(extractor, "last_attn")]
        try:
            super()._extra_backward()
            # Reject overlap with online/flat even if a malformed command sets
            # more than one role, so this term cannot target an online leg.
            paired = (roles[:, 6] > .5) & (roles[:, 5] <= .5) & (roles[:, 7] <= .5)
            ids = paired.nonzero().flatten()[:self.paired_preservation_samples]
            count = len(ids)
            if not count:
                # Every rank traverses the same student graph. Weight=0 creates
                # zero (not absent) gradients for its trainable mean parameters.
                ids = torch.zeros(1, dtype=torch.long, device=self.device)
            sample = obs[ids]
            with torch.no_grad():
                self._paired_teacher.eval()
                teacher = self._paired_teacher(sample, stochastic_output=False)
            student = actor(sample, stochastic_output=False)
            if student.shape[-1] != 29 or teacher.shape != student.shape:
                raise ValueError("Paired prior requires 29 matching PD action means")
            delta_rad = (student - teacher) * self._upper_action_scales
            mse = delta_rad.square().mean()
            normalized_loss = .5 * mse / self.paired_preservation_std_rad**2
            weight = self.paired_preservation_coef if count else 0.0
            (weight * normalized_loss).backward()
            self._paired_batch_count += 1
            self._paired_weighted_loss_sum += weight * float(normalized_loss.detach())
            if count:
                self._paired_count += count
                for group, section in _GROUPS.items():
                    selected = delta_rad[:, section].detach()
                    self._paired_mse_sum[group] += count * float(selected.square().mean())
                    self._paired_abs_sum[group] += count * float(selected.abs().mean())
        finally:
            actor._latest_update_obs, actor._last_tokens = obs, tokens
            if distribution is not None and hasattr(distribution, "_distribution"):
                distribution._distribution = gaussian
            for extractor, attention, latent in taps:
                extractor.last_attn, extractor.last_z = attention, latent
            cross_attention.LATEST_ATTENTION = latest_attention

    def update(self):
        self._reset_paired_statistics()
        losses, infos = super().update()
        losses["paired_vibe_preservation"] = self._paired_weighted_loss_sum / max(1, self._paired_batch_count)
        for group in _GROUPS:
            mse = self._paired_mse_sum[group] / max(1, self._paired_count)
            infos[f"paired_{group}_vibe_delta_rmse_rad"] = math.sqrt(mse)
            infos[f"paired_{group}_vibe_delta_mae_rad"] = self._paired_abs_sum[group] / max(1, self._paired_count)
        full_mse = self._paired_mse_sum["full"] / max(1, self._paired_count)
        losses["paired_vibe_prior_rad2"] = .5 * full_mse
        losses["paired_vibe_prior"] = .5 * full_mse / self.paired_preservation_std_rad**2
        infos["paired_vibe_prior_samples"] = float(self._paired_count)
        return losses, infos

    def _paired_resume_contract(self):
        # Store the effective float32 scales used in the loss, not a potentially
        # rounded textual environment setting. Resuming an existing paired run
        # must not silently change the objective's strength or physical units.
        return {
            "schema_version": 1, "coefficient": self.paired_preservation_coef,
            "std_rad": self.paired_preservation_std_rad, "samples_per_minibatch": self.paired_preservation_samples,
            "action_scales_rad_per_unit": self._upper_action_scales.detach().cpu().tolist(),
            "action_joint_names": list(SONIC_JOINT_ORDER),
        }

    def save(self):
        payload = super().save()
        payload["paired_preservation"] = {
            "teacher": self.paired_teacher_provenance, **self._paired_resume_contract(),
        }
        return payload

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if "paired_preservation" in loaded_dict:
            saved = loaded_dict["paired_preservation"]
            if not isinstance(saved, dict):
                raise ValueError("Resume paired_preservation metadata must be an object")
            for key, expected in self._paired_resume_contract().items():
                if key not in saved or saved[key] != expected:
                    raise ValueError(f"Resume paired objective contract differs: {key}")
            for key in ("checkpoint", "checkpoint_sha256", "actor_config_sha256", "actor_contract_sha256", "frozen_sonic_sha256"):
                if saved.get("teacher", {}).get(key) != self._paired_provenance[key]:
                    raise ValueError(f"Resume paired teacher provenance differs: {key}")
        # A pre-existing checkpoint without this metadata is a supported explicit
        # initialization of this OPTIONAL experiment, not an implicit old teacher.
        result = super().load(loaded_dict, load_cfg, strict)
        verify_frozen_base(self._raw_actor, self._paired_teacher)
        self._paired_basis_verified_after_load = True
        return result
