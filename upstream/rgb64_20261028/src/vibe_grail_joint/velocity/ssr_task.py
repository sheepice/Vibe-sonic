"""Independent SSR second-stage task; the previous task builder stays unchanged."""

import copy
import math

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import RewardTermCfg

from .config import env_config as original_env_config, agent_config
from .ssr_runtime import SSRGuidance

REMOVED_REWARDS = (
    "online_swing_clearance", "online_riser_collision", "online_landing_support",
    "online_foot_target", "online_support_height",
)

DEFAULTS = dict(
    version=1, weight=.25, reward_variance=.0625, height_tolerance_m=.025,
    sole_length_m=.206, sole_width_m=.074, sole_center_x_m=.039, sole_center_z_m=-.035,
    patch_spacing_m=.025, sigma_min_m=.015, sigma_max_m=.30,
    learning_rate=5e-4, replay_capacity=65536, pending_slots=24,
    max_contact_horizon_steps=96, sample_interval_steps=4, valid_contact_steps=2,
    contact_force_n=5., geometry_batch_envs=128, predictor_batch_size=1024,
    gradient_steps_per_update=4, predictor_warmup_updates=50, minimum_labels=4096,
    maximum_prediction_rmse_m=.15, minimum_terrain_level=2, reward_warmup_updates=200,
    seed=39171,
)


def options(settings):
    supplied = settings.get("ssr_foothold")
    if not isinstance(supplied, dict):
        raise ValueError("SSR training requires explicit ssr_foothold settings")
    unknown = set(supplied)-set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown SSR options: {sorted(unknown)}")
    value = {**DEFAULTS, **supplied}
    for key, number in value.items():
        if not isinstance(number, (int, float)) or isinstance(number, bool) or not math.isfinite(number):
            raise ValueError(f"Invalid SSR numeric option: {key}")
        if key not in ("sole_center_z_m", "minimum_terrain_level", "seed") and number <= 0:
            raise ValueError(f"SSR option must be positive: {key}")
    for key in ("version", "replay_capacity", "pending_slots", "max_contact_horizon_steps",
                "sample_interval_steps", "valid_contact_steps", "geometry_batch_envs",
                "predictor_batch_size", "gradient_steps_per_update", "predictor_warmup_updates",
                "minimum_labels", "minimum_terrain_level", "reward_warmup_updates", "seed"):
        if type(value[key]) is not int:
            raise ValueError(f"SSR integer option required: {key}")
    if (value["version"] != 1 or value["sigma_min_m"] >= value["sigma_max_m"]
            or value["minimum_terrain_level"] < 0):
        raise ValueError("Invalid SSR version/sigma bounds/terrain gate")
    if value["pending_slots"]*value["sample_interval_steps"] < value["max_contact_horizon_steps"]:
        raise ValueError("Pending contact window is shorter than the label horizon")
    return value


def ssr_reward(env):
    return env.ssr.after_physics_reward()


def env_config(bank_path, settings, num_envs):
    params = options(settings)
    if settings.get("dagger") is None or settings.get("terrain_profile") != "irregular_v1":
        raise ValueError("SSR adapter requires the current DAgger/profile second-stage framework")
    adapted = copy.deepcopy(settings)
    adapted["foothold"] = dict(clearance_weight=0., collision_weight=0., support_weight=0.)
    cfg = original_env_config(bank_path, adapted, num_envs)
    for name in REMOVED_REWARDS:
        cfg.rewards.pop(name, None)
    cfg.rewards["online_ssr_foothold"] = RewardTermCfg(func=ssr_reward, weight=params["weight"])
    return cfg


class SSREnv(ManagerBasedRlEnv):
    def __init__(self, cfg, *, device, ssr_options):
        super().__init__(cfg, device=device)
        self.ssr = SSRGuidance(self, ssr_options)

    def step(self, action):
        self.ssr.before_step(action)
        return super().step(action)

    def _reset_idx(self, env_ids=None):
        if hasattr(self, "ssr"):
            self.ssr.reset(env_ids)
        return super()._reset_idx(env_ids)
