"""Evaluation-only verification of ascent, gap crossing, and descent ordering.

Physical surface IDs/contact observations are scoring data, never actor inputs.
No desired foot assignment is imposed: either foot may establish each landing.
"""
import torch


def platform_indices(tile):
    if tile['family']!='stair_gap':return -1,-1
    left,right=tile['gaps'][0]
    departure=next(i for i,s in enumerate(tile['solids']) if abs(s['x']+s['hx']-left)<1e-6)
    landing=next(i for i,s in enumerate(tile['solids']) if abs(s['x']-s['hx']-right)<1e-6)
    return departure,landing


class RouteAudit:
    def __init__(self,env):
        self.env=env;self.command=env.command_manager.get_term('motion')
        pairs=[platform_indices(t) for t in self.command.cfg.random_tiles]
        self.indices=torch.tensor(pairs,device=env.device)
        self.departure=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
        self.landing=self.departure.clone()
        self.maximum_root_z=torch.full((env.num_envs,),float('-inf'),device=env.device)

    @torch.no_grad()
    def update(self,active):
        c=self.command;ids=torch.arange(self.env.num_envs,device=self.env.device)
        feet=c.robot_body_pos_w[:,[3,6]]
        surface=c.surface_ids(feet[:,:,:2],ids)
        height=c.ground_height(feet[:,:,:2])
        support=self.env.quality.sensor_support & ((feet[:,:,2]-height).abs()<.12)
        dep,land=self.indices[c.tile].unbind(-1)
        self.departure|=active&(dep>=0)&((surface==dep[:,None])&support).any(-1)
        self.landing|=active&self.departure&(land>=0)&((surface==land[:,None])&support).any(-1)
        z=c.robot.data.root_link_pos_w[:,2]-c.tile_origins[c.tile,2]
        self.maximum_root_z=torch.where(active,torch.maximum(self.maximum_root_z,z),self.maximum_root_z)

    @property
    def qualified(self):
        required=self.indices[self.command.tile,0]>=0
        return ~required | (self.departure & self.landing)

    def row(self,index):
        required=bool(self.indices[self.command.tile[index],0]>=0)
        return dict(route_verification_required=required,
                    departure_platform_contact=bool(self.departure[index]) if required else None,
                    landing_platform_contact=bool(self.landing[index]) if required else None,
                    route_order_verified=bool(self.qualified[index]),max_root_local_z=float(self.maximum_root_z[index]))
