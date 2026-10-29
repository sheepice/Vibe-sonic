"""50% online PPO, 20% paired PPO, 30% flat student-driven DAgger."""

from dataclasses import dataclass, fields
import torch
from mjlab.managers import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from .recovery_terrain import ProfileMotionCommand, ProfileMotionCommandCfg
from . import foothold


def task_masks(num_envs, device="cpu"):
    if num_envs < 20 or num_envs % 20:
        raise ValueError("DAgger task requires num_envs divisible by 20")
    slot = torch.arange(num_envs, device=device) % 20
    return slot < 10, (slot >= 10) & (slot < 14), (slot >= 14) & (slot < 17), slot >= 17


class DaggerMotionCommand(ProfileMotionCommand):
    def _init_task(self):
        super()._init_task()
        self.ppo_online, self.paired, self.dagger_planner, self.flat = task_masks(self.num_envs, self.device)
        self.dagger_mask = self.dagger_planner | self.flat
        self.ppo_mask = ~self.dagger_mask
        # These three flags describe reference mode, never optimizer membership.
        # Planner flat examples have the same actor command flags as runtime.
        self.online = self.ppo_online | self.dagger_planner
        self.online_ids = self.online.nonzero().flatten()
        self.metrics["terrain_fraction"] = (self.ppo_online | self.paired).float()
        self._source_index = torch.zeros(self.motion.n_clips, device=self.device, dtype=torch.long)
        for i, name in enumerate(self.flat_sources):
            self._source_index[self.source_groups[name]] = i

    def dagger_strata(self):
        return torch.where(self.dagger_planner, len(self.flat_sources) + self.stream,
                           self._source_index[self._clip_ids])[self.dagger_mask]

    def _sample_init_frame(self, env_ids):
        chosen, frames = super()._sample_init_frame(env_ids)
        mask = self.dagger_planner[env_ids]
        ids = env_ids[mask]
        if len(ids):
            # Every profile block's fourth tile is physically flat.
            self.tile[ids] = (self.tile[ids] // 4) * 4 + 3
            spawn = self.tile_origins[self.tile[ids]].clone()
            spawn[:, :2] += torch.empty(len(ids), 2, device=self.device).uniform_(-.25, .25)
            spawn[:, 2] = self.ground_height(spawn[:, :2], ids)
            spawn[:, :2] -= self.motion.body_pos_w[frames[mask], 0, :2]
            self._env.scene.env_origins[ids] = spawn
        return chosen, frames


@dataclass(kw_only=True)
class DaggerMotionCommandCfg(ProfileMotionCommandCfg):
    def build(self, env):
        return DaggerMotionCommand(self, env)


def configure_dagger_task(cfg, settings):
    if settings.get("terrain_profile") != "irregular_v1":
        raise ValueError("Foothold geometry requires the physical irregular_v1 profiles")
    task_masks(cfg.scene.num_envs)
    old = cfg.commands["motion"]
    cfg.commands["motion"] = DaggerMotionCommandCfg(**{f.name: getattr(old, f.name)
                                                       for f in fields(DaggerMotionCommandCfg)})
    cfg.rewards["command_velocity"].weight = settings.get("velocity_tracking_weight", 2.0)
    params = settings["foothold"]
    for name, func, key in (("swing_clearance", foothold.swing_clearance_cost, "clearance_weight"),
                            ("riser_collision", foothold.riser_collision_cost, "collision_weight"),
                            ("landing_support", foothold.landing_support_cost, "support_weight")):
        cfg.rewards[f"online_{name}"] = RewardTermCfg(func=func, weight=params[key],
                                                     params={"settings": params})
    for side in ("left", "right"):
        cfg.scene.sensors += (ContactSensorCfg(
            name=f"foothold_{side}",
            primary=ContactMatch(mode="geom", pattern=f"{side}_foot[1-7]_collision", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force", "normal", "tangent"),
            reduce="maxforce", num_slots=2, global_frame=True, history_length=cfg.decimation,
        ),)
    return cfg
