"""Contact-resolved diagnostics shared by foothold evaluation and training.

No actor observations, robot dynamics, or rewards are changed by this module.
Substep measurements use actual contact normals and contact-point velocities.
"""
from dataclasses import replace

import torch
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.lab_api.math import quat_apply


def add_quality_sensors(cfg):
    sensors = list(cfg.scene.sensors)
    names = {s.name: i for i, s in enumerate(sensors)}
    for side in ('left', 'right'):
        name = f'foothold_{side}'
        if name in names:
            old = sensors[names[name]]
            sensors[names[name]] = replace(old, fields=('found','force','normal','tangent','pos'),
                                          history_length=cfg.decimation)
        else:
            sensors.append(ContactSensorCfg(name=name,
                primary=ContactMatch(mode='geom', pattern=f'{side}_foot[1-7]_collision', entity='robot'),
                secondary=ContactMatch(mode='body', pattern='terrain'),
                fields=('found','force','normal','tangent','pos'), reduce='maxforce',
                num_slots=2, global_frame=True, history_length=cfg.decimation))
        name = f'quality_shin_{side}'
        if name not in names:
            sensors.append(ContactSensorCfg(name=name,
                primary=ContactMatch(mode='geom', pattern=f'{side}_(shin|linkage_brace)_collision', entity='robot'),
                secondary=ContactMatch(mode='body', pattern='terrain'),
                fields=('found','force','normal','tangent','pos'), reduce='maxforce',
                num_slots=2, global_frame=True, history_length=cfg.decimation))
    cfg.scene.sensors = tuple(sensors)


def contact_point_velocity(linear, angular, body_position, contact_position):
    offset = contact_position-body_position[:,None]
    return linear[:,None]+torch.linalg.cross(angular[:,None].expand_as(offset),offset,dim=-1)


def tile_profile(tile):
    if 'edges' in tile:
        return tile['edges'], tile['jumps'], tile['base']
    n, h, tread, start = tile['n'],tile['height'],tile['tread'],tile['start']
    up = [start+i*tread for i in range(n)]
    if tile['kind']=='flat':return [],[],0.
    if tile['kind']=='up':return up,[h]*n,0.
    if tile['kind']=='down':return up,[-h]*n,n*h
    down = [start+n*tread+tile['plateau']+i*tread for i in range(n)]
    return up+down,[h]*n+[-h]*n,0.


class StepQualityObserver:
    def __init__(self, env, env_ids=None):
        self.env=env;self.c=env.command_manager.get_term('motion')
        self.ids=(torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids)
        self.feet=[list(self.c.cfg.body_names).index(f'{s}_ankle_roll_link') for s in ('left','right')]
        self.last_support=torch.zeros(len(self.ids),2,device=env.device,dtype=torch.bool)
        self.air_vz=torch.zeros(len(self.ids),2,device=env.device)
        self.support_steps=torch.zeros(len(self.ids),2,device=env.device,dtype=torch.long)
        self.sensor_support=self.last_support.clone()
        self.contact_x=torch.zeros(len(self.ids),2,device=env.device)
        self._observed=False
        self.step_data={k:torch.zeros(len(self.ids),2,device=env.device) for k in (
            'foot_riser_impulse_ns','foot_riser_force_n','shin_impulse_ns','shin_force_n',
            'slip_distance_m','support_seconds','touchdowns','touchdown_vz_m_s')}
        tiles=self.c.cfg.random_tiles
        profiles=[tile_profile(t) for t in tiles]
        size=max(1,max(len(p[0]) for p in profiles))
        self.edges=torch.full((len(tiles),size),float('inf'),device=env.device)
        self.jumps=torch.zeros_like(self.edges)
        self.base=torch.tensor([p[2] for p in profiles],device=env.device)
        for i,(edges,jumps,_) in enumerate(profiles):
            self.edges[i,:len(edges)]=torch.tensor(edges,device=env.device)
            self.jumps[i,:len(edges)]=torch.tensor(jumps,device=env.device)
        self.sole_corners=torch.tensor([[-.064,-.037,-.035],[-.064,.037,-.035],
                                       [.142,-.037,-.035],[.142,.037,-.035]],device=env.device)

    def begin_step(self):
        for value in self.step_data.values():value.zero_()

    @torch.no_grad()
    def observe_substep(self):
        c,ids=self.c,self.ids
        position=c.robot_body_pos_w[ids][:,self.feet]
        linear=c.robot_body_lin_vel_w[ids][:,self.feet]
        angular=c.robot_body_ang_vel_w[ids][:,self.feet]
        dt=self.env.physics_dt
        for foot,side in enumerate(('left','right')):
            d=self.env.scene[f'foothold_{side}'].data
            found=d.found[ids]>0;force=d.force[ids];normal=d.normal[ids]
            norm=force.norm(dim=-1)
            wall=found&(normal[...,2].abs()<.5)
            wall_force=torch.where(wall,norm,0.)
            self.step_data['foot_riser_impulse_ns'][:,foot]+=wall_force.sum(-1)*dt
            self.step_data['foot_riser_force_n'][:,foot].copy_(torch.maximum(
                self.step_data['foot_riser_force_n'][:,foot],wall_force.amax(-1)))
            weight=torch.where(found&(normal[...,2].abs()>.7),force[...,2].abs(),0.)
            total=weight.sum(-1);support=total>20.
            velocity=contact_point_velocity(linear[:,foot],angular[:,foot],position[:,foot],d.pos[ids])
            tangent=velocity-(velocity*normal).sum(-1,keepdim=True)*normal
            speed=(tangent.norm(dim=-1)*weight).sum(-1)/total.clamp_min(1.)
            self.step_data['slip_distance_m'][:,foot]+=speed*support*dt
            self.step_data['support_seconds'][:,foot]+=support*dt
            touchdown=support&~self.last_support[:,foot]&self._observed
            self.step_data['touchdowns'][:,foot]+=touchdown
            self.step_data['touchdown_vz_m_s'][:,foot]+=torch.where(touchdown,self.air_vz[:,foot],0.)
            self.air_vz[:,foot]=torch.where(~support,linear[:,foot,2],self.air_vz[:,foot])
            self.contact_x[:,foot]=torch.where(support,(d.pos[ids][...,0]*weight).sum(-1)/total.clamp_min(1.),position[:,foot,0])
            self.last_support[:,foot]=support
            self.sensor_support[:,foot]=support
            shin=self.env.scene[f'quality_shin_{side}'].data
            shin_force=torch.where(shin.found[ids]>0,shin.force[ids].norm(dim=-1),0.)
            self.step_data['shin_impulse_ns'][:,foot]+=shin_force.sum(-1)*dt
            self.step_data['shin_force_n'][:,foot].copy_(torch.maximum(
                self.step_data['shin_force_n'][:,foot],shin_force.amax(-1)))
        self._observed=True

    @torch.no_grad()
    def geometry(self):
        c,ids=self.c,self.ids;tile=c.tile[ids];origin=c.tile_origins[tile]
        pos=c.robot_body_pos_w[ids][:,self.feet];quat=c.robot_body_quat_w[ids][:,self.feet]
        q=quat[:,:,None].expand(-1,-1,4,-1)
        corners=pos[:,:,None]+quat_apply(q.reshape(-1,4),self.sole_corners.expand(len(ids),2,4,3).reshape(-1,3)).reshape(len(ids),2,4,3)
        x=corners[...,0]-origin[:,None,None,0]
        anchor=self.contact_x-origin[:,None,0]
        edge=self.edges[tile][:,None]
        lower=torch.where(edge<=anchor[...,None],edge,-1e4).amax(-1)
        upper=torch.where(edge>anchor[...,None],edge,1e4).amin(-1)
        margin=torch.minimum(x.amin(-1)-lower,upper-x.amax(-1))
        side_margin=2.-(corners[...,1]-origin[:,None,None,1]).abs().amax(-1)
        margin=torch.minimum(margin,side_margin)
        self.support_steps.copy_(torch.where(self.sensor_support,self.support_steps+1,0))
        return dict(sole_edge_margin_m=margin,foot_position_w=pos,foot_quaternion_w=quat,
                    foot_linear_velocity_w=c.robot_body_lin_vel_w[ids][:,self.feet],
                    foot_angular_velocity_w=c.robot_body_ang_vel_w[ids][:,self.feet],
                    supported=self.sensor_support.clone(),stable_support=self.support_steps>=3,
                    root_local=c.robot.data.root_link_pos_w[ids]-origin)

    def reset(self,env_ids=None):
        mask=torch.ones(self.env.num_envs,device=self.env.device,dtype=torch.bool)
        if env_ids is not None:mask.zero_();mask[env_ids]=True
        selected=mask[self.ids]
        self.last_support[selected]=False;self.sensor_support[selected]=False
        self.support_steps[selected]=0;self.air_vz[selected]=0
