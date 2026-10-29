"""Privileged supervision directly from physics, without any depth camera stream."""
import torch
from ..hiking_perception.geometry import grid_world, height_labels, yaw_from_quat
from ..hiking_perception.simulation import GeometryLabels


class PhysicsLabels(GeometryLabels):
    @torch.no_grad()
    def sample(self,ids):
        c=self.c
        root=c.robot.data.root_link_pos_w[ids]
        world=grid_world(root,yaw_from_quat(c.robot.data.root_link_quat_w[ids]),self.spec)
        tile=c.tile[ids]
        labels=height_labels(world,root,c.tile_origins[tile],self.edges[tile],self.jumps[tile],self.base[tile],self.spec)
        labels['scene_id']=self.scene_ids[tile]
        labels['base_lin_vel_b']=c.robot.data.root_link_lin_vel_b[ids].clone()
        labels['base_ang_vel_b']=c.robot.data.root_link_ang_vel_b[ids].clone()
        return labels


def rgb_only(cfg):
    from dataclasses import replace
    cfg.scene.sensors=tuple(replace(s,data_types=('rgb',)) if s.name=='head_cam' else s for s in cfg.scene.sensors)
    return cfg
