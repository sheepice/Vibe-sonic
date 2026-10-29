"""Training-only SONIC action priors for online waist and arm preservation.

The actor architecture and deployed action computation remain unchanged. The
new terms compare effective PD targets in radians, not normalized policy units.
This is an action prior; reference torso/joint tracking still needs task rewards.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from .policy import PreservationPPO

SONIC_JOINT_ORDER = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)


class UpperBodyPreservationPPO(PreservationPPO):
    """Retain flat full-action prior and add an online upper-body-only prior.

    ``action_scales`` must be resolved in ``action_joint_names`` order from the
    actual SONIC environment action configuration. SONIC's profile uses
    ``0.25 * effort_limit / stiffness`` after regrouping robot actuators; do not
    substitute an arbitrary constant or the unmodified G1 asset scale table.
    The common default-position offset cancels in the PD-target difference.

    Each group uses 0.5 * mean((delta_target_radians / std_rad) ** 2), with
    ``std_rad`` explicitly configured rather than borrowed from action noise.
    Its independent mean prevents fourteen arm joints from diluting three waist
    joints. Reported action deviations retain their physical radians units.
    No leg outputs are directly penalized by the online term. Shared network
    parameters can still affect legs indirectly, as with any joint training.
    """

    def __init__(
        self,
        *args,
        action_scales: Sequence[float],
        action_joint_names: Sequence[str],
        upper_preservation_coef: float = 0.02,
        upper_preservation_std_rad: float = 0.2,
        upper_waist_weight: float = 1.0,
        upper_arms_weight: float = 1.0,
        upper_preservation_samples: int = 128,
        **kwargs,
    ):
        if tuple(action_joint_names) != SONIC_JOINT_ORDER:
            raise ValueError("Upper-body prior requires the validated 29-joint SONIC MuJoCo action order")
        scale = torch.as_tensor(action_scales, dtype=torch.float32)
        if scale.shape != (29,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("action_scales must contain 29 finite positive PD radians-per-action values")
        coefficients = (upper_preservation_coef, upper_waist_weight, upper_arms_weight)
        if any(not math.isfinite(v) or v < 0 for v in coefficients):
            raise ValueError("Upper-body prior coefficients must be finite and nonnegative")
        if not math.isfinite(upper_preservation_std_rad) or upper_preservation_std_rad <= 0:
            raise ValueError("upper_preservation_std_rad must be finite and positive")
        if isinstance(upper_preservation_samples, bool) or not isinstance(upper_preservation_samples, int):
            raise TypeError("upper_preservation_samples must be a positive integer")
        if upper_preservation_samples < 1:
            raise ValueError("upper_preservation_samples must be a positive integer")
        super().__init__(*args, **kwargs)
        self._upper_action_scales = scale.to(self.device)
        self.upper_preservation_coef = float(upper_preservation_coef)
        self.upper_preservation_std_rad = float(upper_preservation_std_rad)
        self.upper_waist_weight = float(upper_waist_weight)
        self.upper_arms_weight = float(upper_arms_weight)
        self.upper_preservation_samples = upper_preservation_samples
        self._reset_upper_statistics()

    def _reset_upper_statistics(self):
        self._upper_count = 0
        self._upper_batch_count = 0
        self._upper_weighted_loss_sum = 0.0
        self._upper_mse_sum = {"waist": 0.0, "arms": 0.0}
        self._upper_abs_sum = {"waist": 0.0, "arms": 0.0}

    def _extra_backward(self):
        actor = self._raw_actor
        # The inherited flat prior runs another actor forward and replaces this
        # pointer. Keep the original PPO minibatch for the online selection.
        obs = actor._latest_update_obs
        if obs is None:
            raise RuntimeError("Upper-body preservation requires the current PPO actor minibatch")
        try:
            super()._extra_backward()
            ids = (obs["velocity_command"][:, 5] > 0.5).nonzero().flatten()
            ids = ids[: self.upper_preservation_samples]
            count = len(ids)
            if not count:
                # reduce_parameters packs only non-None gradients. All ranks
                # must therefore run the same trainable graph, even when a
                # minibatch has no online samples (or the coefficient is zero).
                ids = torch.zeros(1, device=self.device, dtype=torch.long)
            sample = obs[ids]
            with torch.no_grad():
                teacher = actor.sonic_action(sample)
            # Deterministic SonicBaseModel.forward does not update its Gaussian
            # distribution: original minibatch mean/std/log-prob stay intact.
            student = actor(sample, stochastic_output=False)
            if student.shape[-1] != 29 or teacher.shape != student.shape:
                raise ValueError("Upper-body prior requires matching 29-dimensional action means")
            delta_rad = (student - teacher) * self._upper_action_scales
            waist_mse = delta_rad[:, 12:15].square().mean()
            arms_mse = delta_rad[:, 15:29].square().mean()
            group_loss = 0.5 * (
                self.upper_waist_weight * waist_mse + self.upper_arms_weight * arms_mse
            ) / self.upper_preservation_std_rad**2
            weight = self.upper_preservation_coef if count else 0.0
            (weight * group_loss).backward()

            # Keep scalar statistics only; no auxiliary graph survives this hook.
            self._upper_batch_count += 1
            self._upper_weighted_loss_sum += weight * float(group_loss.detach())
            if count:
                self._upper_count += count
                for group, section, mse in (
                    ("waist", slice(12, 15), waist_mse),
                    ("arms", slice(15, 29), arms_mse),
                ):
                    self._upper_mse_sum[group] += count * float(mse.detach())
                    self._upper_abs_sum[group] += count * float(delta_rad[:, section].detach().abs().mean())
        finally:
            actor._latest_update_obs = obs

    def update(self):
        self._reset_upper_statistics()
        try:
            losses, infos = super().update()
            losses["online_upper_sonic_preservation"] = self._upper_weighted_loss_sum / max(
                1, self._upper_batch_count
            )
            for group in ("waist", "arms"):
                mse = self._upper_mse_sum[group] / max(1, self._upper_count)
                losses[f"online_{group}_sonic_prior"] = 0.5 * mse / self.upper_preservation_std_rad**2
                losses[f"online_{group}_sonic_prior_rad2"] = 0.5 * mse
                infos[f"online_{group}_sonic_delta_rmse_rad"] = math.sqrt(mse)
                infos[f"online_{group}_sonic_delta_mae_rad"] = self._upper_abs_sum[group] / max(
                    1, self._upper_count
                )
            infos["online_upper_prior_samples"] = float(self._upper_count)
            return losses, infos
        finally:
            self._raw_actor._latest_update_obs = None
