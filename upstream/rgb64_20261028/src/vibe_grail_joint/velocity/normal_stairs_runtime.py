"""Two independent training-only foothold methods with shared foot-pose costs."""
import hashlib
import math
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply

from .normal_stairs_geometry import profile_mesh, sharp_mesh_edges, cylinder_penetration, foot_volume_points
from .ssr_foothold import patch_offsets, rotate_xy, yaw_angle, gaussian_quadrature, support_deficiency, foothold_score
from .ssr_runtime import SSRGuidance


def wrap_angle(angle):
    return torch.atan2(angle.sin(), angle.cos())


def deadband_cost(error, tolerance, scale):
    return ((error.abs()-tolerance).clamp_min(0)/scale).square().clamp_max(1.)


def natural_foot_yaw(c, ids, feet):
    # Reference foot yaw relative to reference pelvis; +X is the velocity task's
    # commanded heading. A small natural toe-out is retained, not forced to zero.
    q = c.body_quat_w[ids]
    pelvis = list(c.cfg.body_names).index('pelvis')
    return wrap_angle(yaw_angle(q[:,feet])-yaw_angle(q[:,pelvis])[:,None]).clamp(-math.radians(15),math.radians(15))


class NormalSSRGuidance(SSRGuidance):
    """SSR distribution prediction retained; safe-depth and toe-yaw adaptations."""
    def __init__(self, env, options, adaptation):
        super().__init__(env, options)
        self.adaptation = adaptation
        margin = adaptation['planning_margin_m']
        self.planning_patch = patch_offsets(options['sole_length_m']+2*margin,
                                           options['sole_width_m']+2*margin,
                                           options['patch_spacing_m'],device=env.device)

    @torch.inference_mode()
    def reset(self,env_ids=None):
        super().reset(env_ids)

    @torch.no_grad()
    def before_step(self, actions):
        contact = self.contact_state()
        centers, quat = self.sole_centers()
        features, origins, base_yaw = self.features(actions, contact, centers)
        step = int(self.env.common_step_counter)
        if step % self.options['sample_interval_steps'] == 0:
            self.pending.add(features,origins,base_yaw,~contact,step)
        mu,sigma = self.model(features)
        if not torch.isfinite(mu).all() or not torch.isfinite(sigma).all():
            raise RuntimeError('Nonfinite SSR contact distribution')
        gaussian_xy,weights = gaussian_quadrature(mu,sigma)
        gaussian_xy = rotate_xy(gaussian_xy,base_yaw)+origins[:,None,None]
        # The imagined landing patch follows desired natural foot heading, not
        # current swing yaw; rotating a swing foot alone cannot shrink this patch.
        desired_yaw = natural_foot_yaw(self.c,self.ids,self.feet)
        local = torch.cat((self.patch,torch.zeros_like(self.patch[:,:1])),-1)
        qs = quat[:,:,None].expand(-1,-1,len(local),-1)
        points = centers[:,:,None]+quat_apply(qs.reshape(-1,4),local.expand(*qs.shape[:-1],3).reshape(-1,3)).reshape(*qs.shape[:-1],3)
        stance_parts,imagined_parts = [],[]
        for start in range(0,len(self.ids),self.options['geometry_batch_envs']):
            end = start+self.options['geometry_batch_envs']
            ids = self.ids[start:end]
            stance_h = self.c.ground_height(points[start:end,...,:2],ids)
            # True 3D sole samples: geometry above a sample is not counted as
            # supporting it. This bilateral stance test is an explicit adaptation.
            gap = (points[start:end,...,2]-stance_h).abs()
            stance_parts.append(1-(gap<self.options['height_tolerance_m']).float().mean(-1))
            centers_xy = gaussian_xy[start:end]
            xy = centers_xy.unsqueeze(-2)+rotate_xy(
                self.planning_patch.expand(*centers_xy.shape[:-1],-1,-1),desired_yaw[start:end,:,None])
            heights = self.c.ground_height(xy,ids)
            imagined_parts.append((support_deficiency(heights,tolerance=self.options['height_tolerance_m'])*weights).sum(-1))
        raw_stance,raw_imagined = torch.cat(stance_parts),torch.cat(imagined_parts)
        tolerance = self.adaptation['allowed_support_deficiency']
        stance = ((raw_stance-tolerance)/(1-tolerance)).clamp(0,1)
        imagined = ((raw_imagined-tolerance)/(1-tolerance)).clamp(0,1)
        ready = (self.update_count>=self.options['predictor_warmup_updates'] and
                 self.replay.seen>=self.options['minimum_labels'] and
                 self.prediction_rmse_ema is not None and
                 self.prediction_rmse_ema<=self.options['maximum_prediction_rmse_m'])
        level = self.c.tile_data['level'][self.c.tile[self.ids]]
        enabled = (level>=self.options['minimum_terrain_level'])[:,None] & ready
        score = foothold_score(contact,stance,imagined,enabled,self.options['reward_variance'])
        warmup = min(1.,(self.update_count+1)/self.options['reward_warmup_updates'])
        self.cached_score.zero_(); self.cached_score[self.ids] = score*warmup
        self.metrics = {'SSR/stance_deficiency':raw_stance[contact].mean() if contact.any() else score.new_zeros(()),
                        'SSR/imagined_deficiency':raw_imagined.mean(),'SSR/imagined_enabled_fraction':enabled.float().mean(),
                        'SSR/reward':score.mean(),'SSR/predicted_sigma_m':sigma.mean(),
                        'SSR/warmup':warmup,'SSR/label_count':float(self.replay.seen)}


class FootholdGuidance:
    def __init__(self,env,settings):
        self.env,self.settings = env,settings
        self.c = env.command_manager.get_term('motion')
        self.ids = self.c.ppo_online.nonzero().flatten()
        self.feet = [list(self.c.cfg.body_names).index(f'{side}_ankle_roll_link') for side in ('left','right')]
        self.method = settings['foothold_experiment']['method']
        self.pose = settings['foot_pose']
        self.metrics = {}
        self.last_step = -1
        self.streak = torch.zeros(len(self.ids),2,device=env.device,dtype=torch.long)
        self.rewards = {name:torch.zeros(env.num_envs,device=env.device) for name in ('main','yaw','tilt')}
        self.aux = None
        if self.method == 'ssr_normal':
            from .ssr_task import options
            self.aux = NormalSSRGuidance(env,options(settings),settings['normal_ssr'])
        elif self.method != 'hiking_edges':
            raise ValueError(f'Unknown experiment {self.method}')
        # Common diagnostics use the actual foot pose, including angular velocity.
        self.volume = foot_volume_points(env.device)
        self.edge_hash = None
        self.edge_counts = []
        if self.method == 'hiking_edges':
            arrays=[];digest=hashlib.sha256()
            for tile in self.c.cfg.random_tiles:
                vertices,faces = profile_mesh(tile)
                edges = sharp_mesh_edges(vertices,faces,settings['hiking']['sharpness_deg'])
                arrays.append(edges);self.edge_counts.append(len(edges))
                digest.update(vertices.tobytes());digest.update(faces.tobytes());digest.update(edges.tobytes())
            count = max(1,max(self.edge_counts))
            self.edges = torch.zeros(len(arrays),count,2,3,device=env.device)
            self.valid_edges = torch.zeros(len(arrays),count,device=env.device,dtype=torch.bool)
            for i,edges in enumerate(arrays):
                self.edges[i,:len(edges)] = torch.as_tensor(edges,device=env.device,dtype=torch.float32)
                self.valid_edges[i,:len(edges)] = True
            self.edge_hash=digest.hexdigest()

    @property
    def update_count(self):
        return self.aux.update_count if self.aux else 0

    def synchronize_initial_model(self):
        if self.aux:self.aux.synchronize_initial_model()

    def before_step(self,actions):
        if self.aux:self.aux.before_step(actions)

    def contacts(self):
        result=[]
        for side in ('left','right'):
            s=self.env.scene[f'foothold_{side}'].data
            result.append(((s.found>0)&(s.normal[...,2].abs()>.7)&(s.force[...,2].abs()>5.)).any(-1))
        return torch.stack(result,-1)[self.ids]

    @torch.no_grad()
    def measure(self):
        step=int(self.env.common_step_counter)
        if step==self.last_step:return
        self.last_step=step
        contact=self.contacts()
        self.streak.copy_(torch.where(contact,self.streak+1,0))
        pos=self.c.robot_body_pos_w[self.ids][:,self.feet]
        quat=self.c.robot_body_quat_w[self.ids][:,self.feet]
        linear=self.c.robot_body_lin_vel_w[self.ids][:,self.feet]
        angular=self.c.robot_body_ang_vel_w[self.ids][:,self.feet]
        yaw=yaw_angle(quat)
        desired=natural_foot_yaw(self.c,self.ids,self.feet)
        error=wrap_angle(yaw-desired)
        axis=quat_apply(quat.reshape(-1,4),quat.new_tensor([0.,0.,1.]).expand(len(self.ids)*2,3)).reshape(-1,2,3)
        tilt=axis[...,2].clamp(-1,1).acos()
        steady=contact&(self.streak>=self.pose['minimum_contact_steps'])
        # Mild tilt shaping only during low-speed supported stance; roll-over and
        # toe-off in motion remain free. Yaw is constrained during supported stance.
        tilt_gate=steady*torch.exp(-linear.norm(dim=-1).square()/self.pose['tilt_velocity_scale_m_s']**2)
        yaw_cost=deadband_cost(error,math.radians(self.pose['yaw_tolerance_deg']),math.radians(self.pose['yaw_scale_deg']))
        tilt_cost=deadband_cost(tilt,math.radians(self.pose['tilt_tolerance_deg']),math.radians(self.pose['tilt_scale_deg']))
        warmup=min(1.,(int(getattr(self.env,'policy_update_count',0))+1)/self.pose['warmup_updates'])
        for reward in self.rewards.values():reward.zero_()
        self.rewards['yaw'][self.ids]=(yaw_cost*steady).mean(-1)*warmup
        self.rewards['tilt'][self.ids]=(tilt_cost*tilt_gate).mean(-1)*warmup
        def avg(value,mask):return (value*mask).sum()/mask.sum().clamp_min(1)
        self.metrics={
            'FootPose/stance_yaw_abs_deg':avg(yaw.abs()*180/math.pi,contact),
            'FootPose/stance_yaw_error_deg':avg(error.abs()*180/math.pi,contact),
            'FootPose/stance_tilt_deg':avg(tilt*180/math.pi,contact),
            'FootPose/stance_yaw_over_30_fraction':avg((error.abs()>math.radians(30)).float(),contact),
            'FootPose/stance_samples':contact.sum().float(),'FootPose/warmup':warmup,
            'FootPose/yaw_cost':self.rewards['yaw'][self.ids].mean(),
            'FootPose/tilt_cost':self.rewards['tilt'][self.ids].mean(),
        }
        for k,name in enumerate(('up','down','up_down','flat')):
            mask=contact&(self.c.tile_data['kind'][self.c.tile[self.ids]]==k)[:,None]
            self.metrics[f'FootPose/{name}_yaw_error_deg']=avg(error.abs()*180/math.pi,mask)
            self.metrics[f'FootPose/{name}_stance_samples']=mask.sum().float()
        if self.aux:
            self.rewards['main'].copy_(self.aux.after_physics_reward())
            self.metrics.update(self.aux.metrics)
        else:
            q=quat[:,:,None].expand(-1,-1,len(self.volume),-1)
            offset=quat_apply(q.reshape(-1,4),self.volume.expand(*q.shape[:-1],3).reshape(-1,3)).reshape(*q.shape[:-1],3)
            world_points=pos[:,:,None]+offset
            velocity=linear[:,:,None]+torch.linalg.cross(angular[:,:,None].expand_as(offset),offset,dim=-1)
            tile=self.c.tile[self.ids]
            local=world_points-self.c.tile_origins[tile,None,None]
            depths=[]
            for start in range(0,len(self.ids),self.settings['hiking']['geometry_batch_envs']):
                end=start+self.settings['hiking']['geometry_batch_envs']; chosen=tile[start:end]
                depths.append(cylinder_penetration(local[start:end],self.edges[chosen],self.valid_edges[chosen],self.settings['hiking']['edge_radius_m']))
            depth=torch.cat(depths)
            # Eq. (3), summed over volume points; mean over the two feet is the
            # only normalization. epsilon=1e-3 also penalizes stationary contact.
            raw=(depth*(velocity.norm(dim=-1)+self.settings['hiking']['speed_epsilon'])).sum(-1).mean(-1)
            self.rewards['main'][self.ids]=raw*warmup
            self.metrics.update({'Hiking/volume_penalty':raw.mean(),
                                 'Hiking/maximum_penetration_mm':depth.amax()*1000,
                                 'Hiking/stance_edge_fraction':avg((depth.amax(-1)>0).float(),contact),
                                 'Hiking/volume_point_count':float(len(self.volume)),
                                 'Hiking/edge_count_mean':self.valid_edges[tile].sum(-1).float().mean()})
        if any(not torch.isfinite(value).all() for value in self.rewards.values()):
            raise RuntimeError('Nonfinite foothold reward')
        self.env.extras.setdefault('log',{}).update(self.metrics)

    def reward(self,name):
        self.measure()
        return self.rewards[name]

    def reset(self,env_ids=None):
        mask=torch.ones(self.env.num_envs,device=self.env.device,dtype=torch.bool)
        if env_ids is not None:
            mask.zero_();mask[env_ids]=True
        self.streak[mask[self.ids]]=0
        if self.aux:self.aux.reset(env_ids)

    def update(self):
        return self.aux.update() if self.aux else {}

    def update_metrics(self):
        if self.aux:
            return {'SSR/predictor_updates':self.aux.update_count,'SSR/label_count':self.aux.replay.seen,'SSR/replay_size':self.aux.replay.size}
        return {}

    def save(self,*args):
        if self.aux:self.aux.save(*args)

    def load(self,*args):
        if self.aux:self.aux.load(*args)
