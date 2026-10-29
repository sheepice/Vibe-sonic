"""Exact cylinder/box/floor labels for training; no renderer depth channel."""
import torch
from .terrain import key
from ..hiking_perception.geometry import grid_world,yaw_from_quat
from ..hiking_gaussgym.labels import rgb_only

class PhysicsLabels:
    def __init__(self,env,spec):
        self.c=env.command_manager.get_term('motion');self.spec=spec
        self.keys=[key(t) for t in self.c.cfg.random_tiles]
        self.scene_ids=torch.arange(len(self.keys),device=env.device)
    @torch.no_grad()
    def sample(self,ids):
        c=self.c;root=c.robot.data.root_link_pos_w[ids]
        xy=grid_world(root,yaw_from_quat(c.robot.data.root_link_quat_w[ids]),self.spec)[...,:2]
        height=c.ground_height(xy,ids)-root[:,None,2]
        return dict(height=height,surface_valid=torch.isfinite(height),scene_id=self.scene_ids[c.tile[ids]],
                    base_lin_vel_b=c.robot.data.root_link_lin_vel_b[ids].clone(),
                    base_ang_vel_b=c.robot.data.root_link_ang_vel_b[ids].clone())
