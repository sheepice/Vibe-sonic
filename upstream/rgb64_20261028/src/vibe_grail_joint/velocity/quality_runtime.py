"""Common physical step-quality constraints around separate SSR/Hiking methods."""
import torch
from mjlab.utils.lab_api.math import quat_apply
from .normal_stairs_runtime import FootholdGuidance
from .step_quality_observer import StepQualityObserver
from .quality_costs import (impact_cost, supported_slip_cost, support_margin_cost,
                            swing_riser_cost, positive_cost)


class QualityGuidance(FootholdGuidance):
    def __init__(self,env,settings):
        super().__init__(env,settings)
        self.quality=StepQualityObserver(env,self.ids)
        self.qoptions=settings['step_quality']
        for name in ('riser_impact','shin_impact','swing_path','support_margin','contact_slip','hard_landing'):
            self.rewards[name]=torch.zeros(env.num_envs,device=env.device)
        self.quality_last_step=-1

    def before_step(self,actions):
        self.quality.begin_step()
        super().before_step(actions)

    @torch.no_grad()
    def measure(self):
        step=int(self.env.common_step_counter)
        if self.quality_last_step==step:return
        super().measure()
        self.quality_last_step=step
        q=self.quality;opt=self.qoptions;data=q.step_data;g=q.geometry()
        warmup=min(1.,(int(getattr(self.env,'policy_update_count',0))+1)/opt['warmup_updates'])
        if self.method=='hiking_edges':
            # Same geometric / point-velocity objective; normalize physical units
            # so its scale is comparable to the other rewards, then bound outliers.
            scale=self.settings['hiking']['edge_radius_m']*opt['hiking_speed_scale_m_s']
            self.rewards['main'][self.ids]=(self.rewards['main'][self.ids]/scale).clamp_max(2.)
            self.metrics['Hiking/normalized_penalty']=self.rewards['main'][self.ids].mean()
        tile=self.c.tile[self.ids];origin=self.c.tile_origins[tile]
        quat=g['foot_quaternion_w']
        # Both front corners: the lower one must clear the riser.
        corners=quat.new_tensor([[.142,-.037,-.035],[.142,.037,-.035]])
        rotation=quat[:,:,None].expand(-1,-1,2,-1)
        front=g['foot_position_w'][:,:,None]+quat_apply(rotation.reshape(-1,4),
            corners.expand(len(self.ids),2,2,3).reshape(-1,3)).reshape(len(self.ids),2,2,3)
        front=front-origin[:,None,None]
        moving=self.c.speed[self.ids]>.1
        values={
            'riser_impact':impact_cost(data['foot_riser_impulse_ns'],self.env.step_dt,opt['riser_force_scale_n']),
            'shin_impact':impact_cost(data['shin_impulse_ns'],self.env.step_dt,opt['shin_force_scale_n']),
            'contact_slip':supported_slip_cost(data['slip_distance_m'],data['support_seconds'],g['stable_support'],
                                               opt['slip_deadband_m_s'],opt['slip_scale_m_s']),
            'support_margin':support_margin_cost(g['sole_edge_margin_m'],g['stable_support'],
                                                 opt['desired_margin_m'],opt['margin_scale_m']),
            'swing_path':swing_riser_cost(front[...,0].amax(-1),front[...,2].amin(-1),
                q.edges[tile],q.jumps[tile],q.base[tile],~g['supported']&moving[:,None],
                opt['riser_lookahead_m'],opt['riser_behind_m'],opt['swing_clearance_m'],opt['clearance_scale_m']),
            'hard_landing':(positive_cost(-data['touchdown_vz_m_s']/data['touchdowns'].clamp_min(1),
                opt['landing_vz_deadband_m_s'],opt['landing_vz_scale_m_s'])*(data['touchdowns']>0)).mean(-1),
        }
        for name,value in values.items():
            if not torch.isfinite(value).all():raise RuntimeError(f'Nonfinite quality cost: {name}')
            self.rewards[name][self.ids]=value*warmup
            self.metrics[f'StepQuality/{name}_cost']=value.mean()
        self.metrics['StepQuality/warmup']=warmup
        kinds=self.c.tile_data['kind'][tile]
        for kind,name in enumerate(('up','down','up_down','flat')):
            mask=kinds==kind;den=mask.sum().clamp_min(1)
            for key,value in values.items():self.metrics[f'StepQuality/{name}/{key}']=(value*mask).sum()/den
            stable=g['stable_support']&mask[:,None]
            self.metrics[f'StepQuality/{name}/overhang_fraction']=((g['sole_edge_margin_m']<0)*stable).sum()/stable.sum().clamp_min(1)
            self.metrics[f'StepQuality/{name}/support_samples']=stable.sum().float()
            seconds=(data['support_seconds']*mask[:,None]).sum()
            self.metrics[f'StepQuality/{name}/slip_m_s']=(data['slip_distance_m']*mask[:,None]).sum()/seconds.clamp_min(1e-8)
        self.env.extras.setdefault('log',{}).update(self.metrics)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'quality'):self.quality.reset(env_ids)
